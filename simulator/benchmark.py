"""simulator.benchmark — multi-trial, multi-strategy comparison harness.

Each trial uses ONE shared ground-truth board across ALL strategies so the
comparison is apples-to-apples: same truth, different policy.  Per-strategy
RNG streams are derived deterministically from (seed, trial_idx, strategy)
so adding a strategy to the roster does not perturb other strategies'
results on the same trial.

Output is a JSON-ready dict (see tests/test_benchmark.py for the full schema).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from typing import Any, Callable

import numpy as np

from engine.board import score_comparison_key
from engine.smc import sample_configuration
from simulator.runner import _STRATEGY_REGISTRY, _ship_to_dict, run_game


def _derive_seed(master: int, *parts: Any) -> int:
    """Deterministic per-(trial, strategy) seed derived from master + parts.

    Using a hash rather than arithmetic keeps adjacent trial/strategy seeds
    uncorrelated and independent of roster order.
    """
    key = f"{master}|" + "|".join(str(p) for p in parts)
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFF_FFFF_FFFF_FFFF


def _summarise(per_trial: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(per_trial)
    sank = [t for t in per_trial if t["sank"]]
    sink_rate = len(sank) / n
    mean_hits = sum(t["hits"] for t in per_trial) / n
    mean_turns_if_sank = (
        sum(t["turns"] for t in sank) / len(sank) if sank else None
    )
    keys = [tuple(t["score_key"]) for t in per_trial]
    return {
        "sink_rate": sink_rate,
        "mean_hits": mean_hits,
        "mean_turns_if_sank": mean_turns_if_sank,
        "best_score_key": list(min(keys)),
        "worst_score_key": list(max(keys)),
    }


def run_benchmark(
    *,
    strategies: list[str],
    num_trials: int,
    t_max: int = 80,
    N: int = 256,
    eps: float = 0.10,
    seed: int = 0,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run ``num_trials`` games per strategy and return a summary dict.

    If ``progress`` is given, it is called once per completed game with a
    one-line status string (e.g. from the CLI).
    """
    if not strategies:
        raise ValueError("strategies list must be non-empty")
    if num_trials <= 0:
        raise ValueError("num_trials must be >= 1")
    unknown = [s for s in strategies if s not in _STRATEGY_REGISTRY]
    if unknown:
        known = ", ".join(sorted(_STRATEGY_REGISTRY))
        raise ValueError(f"unknown strategy {unknown!r}; known: {known}")

    trials: list[dict[str, Any]] = []
    per_strategy: dict[str, list[dict[str, Any]]] = {s: [] for s in strategies}
    total_games = num_trials * len(strategies)
    game_idx = 0
    t0 = time.perf_counter()

    for trial_idx in range(num_trials):
        truth_seed = _derive_seed(seed, "truth", trial_idx)
        truth_rng = np.random.default_rng(truth_seed)
        truth = sample_configuration(truth_rng)

        results: dict[str, Any] = {}
        for name in strategies:
            strat_seed = _derive_seed(seed, "strategy", trial_idx, name)
            g0 = time.perf_counter()
            traj = run_game(
                strategy_name=name, truth=truth,
                t_max=t_max, N=N, eps=eps, seed=strat_seed,
            )
            dt = time.perf_counter() - g0
            game_idx += 1
            res = {
                "hits": traj["terminal"]["hits"],
                "turns": traj["terminal"]["turns"],
                "sank": traj["terminal"]["sank"],
                "score_key": traj["terminal"]["score_key"],
            }
            results[name] = res
            per_strategy[name].append(res)
            if progress is not None:
                elapsed = time.perf_counter() - t0
                eta = elapsed * (total_games - game_idx) / max(game_idx, 1)
                progress(
                    f"[{game_idx:>3}/{total_games}] "
                    f"trial {trial_idx+1:>2}/{num_trials} · {name:>8} · "
                    f"H={res['hits']:>2}/12 T={res['turns']:>3} · "
                    f"{dt:4.1f}s · ETA {eta:5.0f}s"
                )

        trials.append({
            "trial_idx": trial_idx,
            "truth_seed": truth_seed,
            "truth": {"ships": [_ship_to_dict(s) for s in truth.ships]},
            "results": results,
        })

    summary = {name: _summarise(per_strategy[name]) for name in strategies}
    return {
        "config": {
            "strategies": list(strategies),
            "num_trials": num_trials,
            "t_max": t_max,
            "N": N,
            "eps": eps,
            "seed": seed,
        },
        "trials": trials,
        "summary": summary,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m simulator.benchmark",
        description="Compare strategies across K shared-truth trials.",
    )
    p.add_argument("--strategies", nargs="+",
                   default=["thompson", "eig", "ellr"],
                   help="strategies to compare")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--t-max", type=int, default=80)
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--eps", type=float, default=0.10)
    p.add_argument("--out", default="-",
                   help="output path for benchmark JSON; '-' for stdout")
    p.add_argument("--quiet", action="store_true",
                   help="suppress per-game progress lines on stderr")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    def _progress(line: str) -> None:
        sys.stderr.write(line + "\n")
        sys.stderr.flush()

    result = run_benchmark(
        strategies=args.strategies, num_trials=args.trials,
        t_max=args.t_max, N=args.N, eps=args.eps, seed=args.seed,
        progress=None if args.quiet else _progress,
    )
    blob = json.dumps(result, indent=2)
    if args.out == "-":
        sys.stdout.write(blob + "\n")
    else:
        with open(args.out, "w") as f:
            f.write(blob)
    # brief human-readable tail on stderr for interactive use
    if args.out != "-":
        for name, s in result["summary"].items():
            mt = s["mean_turns_if_sank"]
            mt_s = f"{mt:.1f}" if mt is not None else "—"
            sys.stderr.write(
                f"{name:>10}: sink={s['sink_rate']:.2f}  "
                f"mean_hits={s['mean_hits']:.2f}  mean_T|sank={mt_s}\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
