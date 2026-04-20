"""simulator.runner — play one seeded game with one named strategy.

Produces a trajectory dict (JSON-ready) consumable by the mockup and by
benchmark aggregation layers.

Trajectory schema
-----------------
{
  "strategy": "thompson" | "eig" | "ellr",
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
   "observed": 0|1, "result": "HIT"|"MISS"}

turn_record (ask):
  {"turn": int, "action": {"kind": "ask", "question_id": str},
   "observed": 0|1}

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
from engine.questions import evaluate, question_by_id
from engine.smc import sample_configuration
from strategies.base import AskAction, ShotAction
from strategies._mbayes import ApproxEIGMBayesStrategy, ApproxELLRMBayesStrategy
from strategies.eig import EIGStrategy
from strategies.ellr import ELLRStrategy
from strategies.thompson import ThompsonStrategy

_STRATEGY_REGISTRY = {
    "thompson": ThompsonStrategy,
    "eig": EIGStrategy,
    "ellr": ELLRStrategy,
    # Approximate variants — K-sample posterior draws replace the exact
    # |S|×|Q| matmul.  See strategies/_mbayes.py for the BALD/KL derivations.
    "eig_approx": ApproxEIGMBayesStrategy,
    "ellr_approx": ApproxELLRMBayesStrategy,
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


def run_game(
    *,
    strategy_name: str,
    truth: Configuration,
    t_max: int = 80,
    N: int = 256,
    eps: float = 0.10,
    seed: int = 0,
) -> dict[str, Any]:
    """Play one game and return a JSON-ready trajectory dict."""
    strategy_rng = np.random.default_rng(seed)
    ask_noise_rng = np.random.default_rng(seed ^ 0xA5A5A5A5)

    strategy = _build_strategy(strategy_name, eps=eps, rng=strategy_rng)
    state = GameState(truth=truth, t_max=t_max)

    turns: list[dict[str, Any]] = []

    cum_net_reward = 0.0
    while not state.terminated:
        action = strategy.choose_action(
            shots_fired=state.shots_fired, turn=state.turn,
        )
        decision_value = float(strategy.last_decision_value)
        if isinstance(action, ShotAction):
            observed = truth.X(action.cell)
            result = state.apply_shot(action.cell)
            strategy.observe(action, observed=observed)
            # Canonical (R+, M, C) = (2, 1, 1): hit = +2, miss = -1, shot cost = -1.
            cum_net_reward += (2.0 if observed == 1 else -1.0) - 1.0
            turns.append({
                "turn": state.turn - 1,
                "action": {"kind": "shot", "cell": [action.cell[0], action.cell[1]]},
                "observed": int(observed),
                "result": result.value,
                "decision_value": decision_value,
                "cum_net_reward": cum_net_reward,
            })
        elif isinstance(action, AskAction):
            q = question_by_id(action.question_id)
            truth_answer = evaluate(q, truth)
            observed = _flip_bit(truth_answer, eps, ask_noise_rng)
            state.apply_ask(action.question_id, observed)
            strategy.observe(action, observed=observed)
            # Asks have zero immediate reward and zero cost — running total unchanged.
            turns.append({
                "turn": state.turn - 1,
                "action": {"kind": "ask", "question_id": action.question_id},
                "observed": int(observed),
                "decision_value": decision_value,
                "cum_net_reward": cum_net_reward,
            })
        else:
            raise TypeError(f"unexpected action type: {type(action)!r}")

    key = score_comparison_key(hits=state.hits, turns=state.turn, t_max=t_max)
    final_mu = strategy.filter.cell_marginal_grid().tolist()
    n_shots = sum(1 for x in turns if x["action"]["kind"] == "shot")
    n_asks = state.turn - n_shots
    n_misses = n_shots - state.hits
    # Canonical scoring for telemetry: (R+, M, C) = (2, 1, 1).  Keeps the
    # benchmark's net_reward apples-to-apples regardless of any per-strategy
    # cost-parameter overrides.
    net_reward = 2.0 * state.hits - 1.0 * n_misses - 1.0 * n_shots
    return {
        "strategy": strategy_name,
        "seed": seed,
        "t_max": t_max,
        "N": N,
        "eps": eps,
        "truth": {"ships": [_ship_to_dict(s) for s in truth.ships]},
        "turns": turns,
        "final_posterior": final_mu,
        "terminal": {
            "hits": state.hits,
            "turns": state.turn,
            "sank": state.hits == sum(s.length for s in truth.ships),
            "score_key": list(key),
            "n_shots": n_shots,
            "n_asks": n_asks,
            "n_misses": n_misses,
            "net_reward": net_reward,
        },
    }


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
