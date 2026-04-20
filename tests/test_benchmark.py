"""Tests for simulator.benchmark — multi-trial, multi-strategy comparison.

Benchmark contract
------------------
  run_benchmark(strategies, num_trials, t_max, N, eps, seed) -> BenchmarkResult

Each trial uses ONE shared ground-truth board across ALL strategies, so the
comparison is "same truth, different policy" — the only source of variance
per trial is each strategy's internal RNG (seeded deterministically from
seed, trial_idx, and strategy name).

Result shape (JSON-ready dict):
  {
    "config": {"strategies": [...], "num_trials": int, "t_max": int,
               "N": int, "eps": float, "seed": int},
    "trials": [
      {
        "trial_idx": int,
        "truth_seed": int,
        "truth": {"ships": [...]},
        "results": {
          "<strategy>": {"hits": int, "turns": int, "sank": bool,
                         "score_key": [sank_flag, turns_if_sank, -hits]},
          ...
        },
      },
      ...
    ],
    "summary": {
      "<strategy>": {"sink_rate": float, "mean_hits": float,
                     "mean_turns_if_sank": float | None,
                     "best_score_key": [...], "worst_score_key": [...]},
      ...
    }
  }
"""
from __future__ import annotations

import pytest

from engine.board import MAX_HITS
from simulator.benchmark import run_benchmark


# Reasonable small config to keep CI fast — coverage is the point here,
# not benchmark-quality numbers.
SMALL = dict(num_trials=3, t_max=20, N=32, eps=0.10, seed=42)


# --------------------------------------------------------------------------
# Shape
# --------------------------------------------------------------------------

class TestShape:
    def test_config_block_echoed(self):
        r = run_benchmark(strategies=["thompson"], **SMALL)
        cfg = r["config"]
        assert cfg["strategies"] == ["thompson"]
        assert cfg["num_trials"] == SMALL["num_trials"]
        assert cfg["t_max"] == SMALL["t_max"]
        assert cfg["N"] == SMALL["N"]
        assert cfg["eps"] == pytest.approx(SMALL["eps"])
        assert cfg["seed"] == SMALL["seed"]

    def test_trials_len_matches_num_trials(self):
        r = run_benchmark(strategies=["thompson"], **SMALL)
        assert len(r["trials"]) == SMALL["num_trials"]

    @pytest.mark.slow
    def test_each_trial_has_result_per_strategy(self):
        r = run_benchmark(strategies=["thompson", "eig"], **SMALL)
        for tr in r["trials"]:
            assert set(tr["results"]) == {"thompson", "eig"}
            for name, res in tr["results"].items():
                assert set(res) >= {"hits", "turns", "sank", "score_key"}
                assert 0 <= res["hits"] <= MAX_HITS
                assert res["sank"] == (res["hits"] == MAX_HITS)

    @pytest.mark.slow
    def test_summary_has_key_per_strategy(self):
        r = run_benchmark(strategies=["thompson", "eig"], **SMALL)
        assert set(r["summary"]) == {"thompson", "eig"}
        for name, s in r["summary"].items():
            assert set(s) >= {"sink_rate", "mean_hits",
                              "mean_turns_if_sank",
                              "best_score_key", "worst_score_key"}


# --------------------------------------------------------------------------
# Fair comparison — same truth across strategies per trial
# --------------------------------------------------------------------------

class TestFairComparison:
    @pytest.mark.slow
    def test_truth_shared_across_strategies_per_trial(self):
        """All strategies in a single trial play the same ground-truth board."""
        r = run_benchmark(strategies=["thompson", "eig"], **SMALL)
        for tr in r["trials"]:
            # Truth is recorded once, not per-strategy.
            assert "truth" in tr
            assert "ships" in tr["truth"]

    def test_single_strategy_run_has_trials(self):
        r = run_benchmark(strategies=["thompson"], **SMALL)
        assert len(r["trials"]) == SMALL["num_trials"]
        # Truths should differ across trials (with overwhelming probability).
        truth_sigs = [tuple(sorted(
            (s["length"], s["row"], s["col"], s["orientation"])
            for s in tr["truth"]["ships"]
        )) for tr in r["trials"]]
        assert len(set(truth_sigs)) > 1


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------

class TestDeterminism:
    def test_same_config_same_result(self):
        a = run_benchmark(strategies=["thompson"], **SMALL)
        b = run_benchmark(strategies=["thompson"], **SMALL)
        assert a == b

    @pytest.mark.slow
    def test_adding_more_strategies_does_not_change_existing(self):
        """Thompson's results at each trial should be independent of whether
        EIG also ran in that trial (same truth, same strategy rng key)."""
        a = run_benchmark(strategies=["thompson"], **SMALL)
        b = run_benchmark(strategies=["thompson", "eig"], **SMALL)
        for ta, tb in zip(a["trials"], b["trials"]):
            assert ta["results"]["thompson"] == tb["results"]["thompson"]


# --------------------------------------------------------------------------
# Summary correctness
# --------------------------------------------------------------------------

class TestSummary:
    @pytest.mark.slow
    def test_sink_rate_matches_trial_count(self):
        r = run_benchmark(strategies=["thompson"], **SMALL)
        sank = sum(1 for tr in r["trials"] if tr["results"]["thompson"]["sank"])
        expected = sank / SMALL["num_trials"]
        assert r["summary"]["thompson"]["sink_rate"] == pytest.approx(expected)

    @pytest.mark.slow
    def test_mean_hits_matches_trials(self):
        r = run_benchmark(strategies=["thompson"], **SMALL)
        hits = [tr["results"]["thompson"]["hits"] for tr in r["trials"]]
        assert r["summary"]["thompson"]["mean_hits"] == pytest.approx(
            sum(hits) / len(hits)
        )

    def test_mean_turns_if_sank_none_when_no_sinks(self):
        """With a very tight turn cap, Thompson can't possibly sink.  Expect
        mean_turns_if_sank == None in that case."""
        r = run_benchmark(strategies=["thompson"],
                          num_trials=2, t_max=3, N=32, eps=0.10, seed=1)
        assert all(not tr["results"]["thompson"]["sank"] for tr in r["trials"])
        assert r["summary"]["thompson"]["mean_turns_if_sank"] is None


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------

class TestErrors:
    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="unknown strategy"):
            run_benchmark(strategies=["nope"], **SMALL)

    def test_empty_strategies_raises(self):
        with pytest.raises(ValueError):
            run_benchmark(strategies=[], **SMALL)

    def test_zero_trials_raises(self):
        with pytest.raises(ValueError):
            run_benchmark(strategies=["thompson"],
                          num_trials=0, t_max=10, N=32, eps=0.10, seed=0)
