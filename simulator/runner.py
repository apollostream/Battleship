"""simulator.runner — play one seeded game with one named strategy.

Provides two entry points:

* ``GameSession`` — a stateful step machine.  One ``session.step()`` call
  advances the game by exactly one turn: consult the strategy, apply the
  chosen action to the ground-truth state, feed the observation back.
  Returns the turn record.  This is the contract the interactive HTTP bridge
  and ``UserStrategy`` rely on (choose_action can synchronously wait for
  external input because step() drives it one call at a time).
* ``run_game`` — thin wrapper: loop ``session.step()`` until terminated, then
  assemble the full trajectory dict.  Benchmark and CLI use this.

Trajectory schema
-----------------
{
  "strategy": "thompson" | "eig" | "ellr" | "eig_approx" | "ellr_approx" | "user",
  "seed":     int,
  "t_max":    int,
  "N":        int,          # deprecated — was particle count for SMC;
                            # retained in schema for back-compat, ignored now
                            # that strategies use exact inference over |S|
  "eps":      float,        # BSC noise on asks
  "truth":    {"ships": [{"length", "row", "col", "orientation"}, ...]},
  "turns":    [ turn_record, ... ],
  "terminal": {"hits", "turns", "sank", "score_key"}
}

turn_record (shot):
  {"turn": int, "action": {"kind": "shot", "cell": [r, c]},
   "observed": 0|1, "result": "HIT"|"MISS",
   "decision_value": float, "cum_net_reward": float}

turn_record (ask):
  {"turn": int, "action": {"kind": "ask", "question_id": str},
   "observed": 0|1,
   "decision_value": float, "cum_net_reward": float}

Noise model
-----------
Shots are noiseless: the shot's `observed` bit is `truth.X(cell)` exactly.
Asks go over a BSC(ε): the truth answer is flipped with probability ε on the
way to the strategy, using a dedicated ask-noise RNG seeded from `seed`.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np

from engine.board import (
    Configuration,
    GameState,
    Orientation,
    Ship,
    ShotResult,
    score_comparison_key,
)
from engine.metrics import info_gain_ask_bsc, info_gain_shot
from engine.questions import evaluate, question_by_id
from engine.smc import sample_configuration
from strategies.base import AskAction, ShotAction
from strategies._mbayes import ApproxEIGMBayesStrategy, ApproxELLRMBayesStrategy
from strategies.eig import EIGStrategy
from strategies.ellr import ELLRStrategy
from strategies.thompson import ThompsonStrategy
from strategies.user import UserStrategy

_STRATEGY_REGISTRY = {
    "thompson": ThompsonStrategy,
    "eig": EIGStrategy,
    "ellr": ELLRStrategy,
    # Approximate variants — K-sample posterior draws replace the exact
    # |S|×|Q| matmul.  See strategies/_mbayes.py for the BALD/KL derivations.
    "eig_approx": ApproxEIGMBayesStrategy,
    "ellr_approx": ApproxELLRMBayesStrategy,
    # Interactive — action supplied externally via session.strategy.set_next_action().
    "user": UserStrategy,
}


def _build_strategy(name: str, *, eps: float, rng: np.random.Generator):
    try:
        cls = _STRATEGY_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(_STRATEGY_REGISTRY))
        raise ValueError(f"unknown strategy {name!r}; known: {known}") from exc
    return cls(eps=eps, rng=rng)


def _ship_to_dict(s: Ship) -> dict[str, Any]:
    return {
        "length": s.length,
        "row": s.row,
        "col": s.col,
        "orientation": s.orientation.value,
    }


def _flip_bit(bit: int, eps: float, rng: np.random.Generator) -> int:
    if eps <= 0.0:
        return bit
    if rng.random() < eps:
        return 1 - bit
    return bit


def _snapshot_rationale(raw: dict | None) -> dict | None:
    """Freeze the strategy's `last_rationale` into a JSON-safe dict.

    Converts tuple cells to [r, c] lists so the record round-trips through
    ``json.dumps``.  Returns None if the strategy never set a rationale.
    """
    if not raw:
        return None
    shot = raw.get("shot")
    if shot is not None:
        cell = shot.get("cell")
        if isinstance(cell, tuple):
            shot = {**shot, "cell": [cell[0], cell[1]]}
    ask = raw.get("ask")
    if ask is not None:
        ask = dict(ask)
    return {"chosen": raw.get("chosen"), "shot": shot, "ask": ask}


class GameSession:
    """One seeded game, driven one turn at a time.

    ``step()`` asks the strategy for its next action, applies it to the
    ground-truth state, feeds the observation back, and returns the JSON-ready
    turn record.  Call until ``terminated`` is True.  ``trajectory()`` bundles
    the accumulated turn records plus terminal summary into the schema
    documented at the top of this module.

    The session is mutable and not thread-safe — intended for single-game use
    inside one handler (HTTP session, CLI invocation, benchmark worker).
    """

    def __init__(
        self,
        *,
        strategy_name: str,
        truth: Configuration,
        t_max: int = 80,
        N: int = 256,
        eps: float = 0.10,
        seed: int = 0,
    ) -> None:
        self.strategy_name = strategy_name
        self.truth = truth
        self.t_max = t_max
        self.N = N
        self.eps = eps
        self.seed = seed
        self._ask_noise_rng = np.random.default_rng(seed ^ 0xA5A5A5A5)
        self.strategy = _build_strategy(
            strategy_name, eps=eps, rng=np.random.default_rng(seed),
        )
        self.state = GameState(truth=truth, t_max=t_max)
        self.turns: list[dict[str, Any]] = []
        self.cum_net_reward: float = 0.0

    @property
    def terminated(self) -> bool:
        return self.state.terminated

    def step(self) -> dict[str, Any]:
        """Advance the game by one turn.  Returns the turn record just produced."""
        if self.state.terminated:
            raise RuntimeError("game already terminated; call trajectory()")
        action = self.strategy.choose_action(
            shots_fired=self.state.shots_fired, turn=self.state.turn,
        )
        decision_value = float(self.strategy.last_decision_value)
        rationale = _snapshot_rationale(
            getattr(self.strategy, "last_rationale", None)
        )
        if isinstance(action, ShotAction):
            # Capture prior μ(cell) before reweight so info_gain_nats reflects
            # the update we're about to do, not the post-update filter.
            mu_pre = float(self.strategy.filter.cell_marginal(action.cell))
            observed = self.truth.X(action.cell)
            result = self.state.apply_shot(action.cell)
            self.strategy.observe(action, observed=observed)
            info_gain = info_gain_shot(mu_c=mu_pre, observed=int(observed))
            # Canonical (R+, M, C) = (2, 1, 1): hit = +2, miss = -1, shot cost = -1.
            self.cum_net_reward += (2.0 if observed == 1 else -1.0) - 1.0
            record = {
                "turn": self.state.turn - 1,
                "action": {"kind": "shot", "cell": [action.cell[0], action.cell[1]]},
                "observed": int(observed),
                "result": result.value,
                "decision_value": decision_value,
                "info_gain_nats": float(info_gain),
                "rationale": rationale,
                "cum_net_reward": self.cum_net_reward,
            }
        elif isinstance(action, AskAction):
            q = question_by_id(action.question_id)
            # Capture p_hat(q) before reweight using the same filter semantics
            # the strategies use: weights · answers_for(q).
            a = self.strategy.filter.answers_for(q)
            p_hat = float(np.dot(self.strategy.filter.weights, a))
            truth_answer = evaluate(q, self.truth)
            observed = _flip_bit(truth_answer, self.eps, self._ask_noise_rng)
            self.state.apply_ask(action.question_id, observed)
            self.strategy.observe(action, observed=observed)
            info_gain = info_gain_ask_bsc(
                p_hat=p_hat, eps=self.eps, observed=int(observed),
            )
            # Asks have zero immediate reward and zero cost — running total unchanged.
            record = {
                "turn": self.state.turn - 1,
                "action": {"kind": "ask", "question_id": action.question_id},
                "observed": int(observed),
                "decision_value": decision_value,
                "info_gain_nats": float(info_gain),
                "rationale": rationale,
                "cum_net_reward": self.cum_net_reward,
            }
        else:
            raise TypeError(f"unexpected action type: {type(action)!r}")
        self.turns.append(record)
        return record

    def trajectory(self) -> dict[str, Any]:
        """Assemble the JSON-ready trajectory dict from accumulated turn records."""
        key = score_comparison_key(
            hits=self.state.hits, turns=self.state.turn, t_max=self.t_max,
        )
        final_mu = self.strategy.filter.cell_marginal_grid().tolist()
        n_shots = sum(1 for x in self.turns if x["action"]["kind"] == "shot")
        n_asks = self.state.turn - n_shots
        n_misses = n_shots - self.state.hits
        # Canonical scoring for telemetry: (R+, M, C) = (2, 1, 1).  Keeps the
        # benchmark's net_reward apples-to-apples regardless of any per-strategy
        # cost-parameter overrides.
        net_reward = 2.0 * self.state.hits - 1.0 * n_misses - 1.0 * n_shots
        return {
            "strategy": self.strategy_name,
            "seed": self.seed,
            "t_max": self.t_max,
            "N": self.N,
            "eps": self.eps,
            "truth": {"ships": [_ship_to_dict(s) for s in self.truth.ships]},
            "turns": list(self.turns),
            "final_posterior": final_mu,
            "terminal": {
                "hits": self.state.hits,
                "turns": self.state.turn,
                "sank": self.state.hits == sum(s.length for s in self.truth.ships),
                "score_key": list(key),
                "n_shots": n_shots,
                "n_asks": n_asks,
                "n_misses": n_misses,
                "net_reward": net_reward,
            },
        }


def run_game(
    *,
    strategy_name: str,
    truth: Configuration,
    t_max: int = 80,
    N: int = 256,
    eps: float = 0.10,
    seed: int = 0,
) -> dict[str, Any]:
    """Play one game to completion and return the trajectory dict."""
    session = GameSession(
        strategy_name=strategy_name, truth=truth,
        t_max=t_max, N=N, eps=eps, seed=seed,
    )
    while not session.terminated:
        session.step()
    return session.trajectory()


def trajectory_to_json(traj: dict[str, Any], *, indent: int | None = 2) -> str:
    return json.dumps(traj, indent=indent)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m simulator.runner",
        description="Play one seeded Battleship game with one strategy.",
    )
    p.add_argument("--strategy", choices=sorted(_STRATEGY_REGISTRY),
                   default="thompson")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--t-max", type=int, default=80)
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--eps", type=float, default=0.10)
    p.add_argument("--truth-seed", type=int, default=None,
                   help="if given, sample truth with this seed; "
                        "otherwise use --seed")
    p.add_argument("--out", default="-",
                   help="output path for trajectory JSON; '-' for stdout")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    truth_rng = np.random.default_rng(
        args.truth_seed if args.truth_seed is not None else args.seed
    )
    truth = sample_configuration(truth_rng)

    traj = run_game(
        strategy_name=args.strategy, truth=truth,
        t_max=args.t_max, N=args.N, eps=args.eps, seed=args.seed,
    )
    blob = trajectory_to_json(traj)
    if args.out == "-":
        sys.stdout.write(blob + "\n")
    else:
        with open(args.out, "w") as f:
            f.write(blob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
