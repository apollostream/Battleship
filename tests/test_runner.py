"""Tests for simulator.runner — single-game execution and JSON trajectory.

Runner contract:
    * run_game(strategy_name, truth, t_max, N, eps, seed, ask_rng=None)
        → Trajectory (a dict consumable by the mockup and by `json.dumps`).
    * Deterministic given (strategy_name, truth, N, eps, seed).
    * Terminates on hits == MAX_HITS or turn == t_max (whichever first).
    * Trajectory JSON contains:
        - strategy, seed, t_max, N, eps
        - truth.ships (list of ship dicts)
        - turns: list of per-turn records
            * {turn, action: {kind: "shot", cell: [r, c]}, observed, result}
            * {turn, action: {kind: "ask",  question_id}, observed}
        - terminal: {hits, turns, sank, score_key}

The runner is responsible for:
    1. Building the named strategy with a seeded rng.
    2. Looping: choose_action → apply to GameState → feed observation to strategy.
    3. For asks: flipping the truth answer with probability eps using ask_rng
       (the "wire" noise), producing the observed answer the strategy sees.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from engine.board import (
    Configuration,
    FLEET_LENGTHS,
    MAX_HITS,
    Orientation,
    Ship,
)
from simulator.runner import run_game, trajectory_to_json


@pytest.fixture
def truth() -> Configuration:
    return Configuration(ships=(
        Ship(4, 0, 0, Orientation.HORIZONTAL),
        Ship(3, 2, 0, Orientation.HORIZONTAL),
        Ship(3, 4, 4, Orientation.HORIZONTAL),
        Ship(2, 7, 6, Orientation.HORIZONTAL),
    ))


# --------------------------------------------------------------------------
# Basic shape
# --------------------------------------------------------------------------

class TestTrajectoryShape:
    def test_returns_dict_with_metadata(self, truth):
        traj = run_game(
            strategy_name="thompson", truth=truth,
            t_max=20, N=32, eps=0.10, seed=1,
        )
        assert traj["strategy"] == "thompson"
        assert traj["seed"] == 1
        assert traj["t_max"] == 20
        assert traj["N"] == 32
        assert traj["eps"] == pytest.approx(0.10)

    def test_truth_section_has_ships(self, truth):
        traj = run_game(
            strategy_name="thompson", truth=truth,
            t_max=5, N=32, eps=0.10, seed=2,
        )
        ships = traj["truth"]["ships"]
        assert len(ships) == len(FLEET_LENGTHS)
        lengths = sorted(s["length"] for s in ships)
        assert lengths == sorted(FLEET_LENGTHS)
        for s in ships:
            assert set(s.keys()) == {"length", "row", "col", "orientation"}
            assert s["orientation"] in ("H", "V")

    def test_turns_have_expected_fields(self, truth):
        traj = run_game(
            strategy_name="thompson", truth=truth,
            t_max=5, N=32, eps=0.10, seed=3,
        )
        assert len(traj["turns"]) > 0
        for rec in traj["turns"]:
            assert "turn" in rec
            assert "action" in rec
            assert "observed" in rec
            kind = rec["action"]["kind"]
            assert kind in ("shot", "ask")
            if kind == "shot":
                assert "cell" in rec["action"]
                assert "result" in rec  # "HIT" or "MISS"
                assert rec["result"] in ("HIT", "MISS")
            else:
                assert "question_id" in rec["action"]

    def test_terminal_block_shape(self, truth):
        traj = run_game(
            strategy_name="thompson", truth=truth,
            t_max=10, N=32, eps=0.10, seed=4,
        )
        term = traj["terminal"]
        assert set(term.keys()) >= {"hits", "turns", "sank", "score_key"}
        assert 0 <= term["hits"] <= MAX_HITS
        assert term["turns"] == len(traj["turns"])
        assert term["sank"] == (term["hits"] == MAX_HITS)
        assert len(term["score_key"]) == 3

    def test_final_posterior_is_8x8_probability_grid(self, truth):
        traj = run_game(
            strategy_name="thompson", truth=truth,
            t_max=5, N=32, eps=0.10, seed=14,
        )
        mu = traj["final_posterior"]
        assert len(mu) == 8
        assert all(len(row) == 8 for row in mu)
        for row in mu:
            for p in row:
                assert 0.0 <= p <= 1.0 + 1e-9


# --------------------------------------------------------------------------
# Execution semantics
# --------------------------------------------------------------------------

class TestExecutionSemantics:
    def test_respects_turn_cap(self, truth):
        traj = run_game(
            strategy_name="thompson", truth=truth,
            t_max=3, N=32, eps=0.10, seed=5,
        )
        assert len(traj["turns"]) <= 3
        assert traj["terminal"]["turns"] <= 3

    def test_shot_result_matches_truth(self, truth):
        """Noiseless shots: observed bit = truth.X(cell); result HIT iff observed=1."""
        traj = run_game(
            strategy_name="thompson", truth=truth,
            t_max=12, N=64, eps=0.10, seed=6,
        )
        occupied = truth.occupied_cells()
        for rec in traj["turns"]:
            if rec["action"]["kind"] == "shot":
                cell = tuple(rec["action"]["cell"])
                expected_obs = 1 if cell in occupied else 0
                assert rec["observed"] == expected_obs
                assert rec["result"] == ("HIT" if expected_obs == 1 else "MISS")

    def test_no_repeated_shots(self, truth):
        traj = run_game(
            strategy_name="thompson", truth=truth,
            t_max=12, N=64, eps=0.10, seed=7,
        )
        shots: list[tuple[int, int]] = []
        for rec in traj["turns"]:
            if rec["action"]["kind"] == "shot":
                shots.append(tuple(rec["action"]["cell"]))
        assert len(shots) == len(set(shots))

    def test_hits_count_matches_hit_results(self, truth):
        traj = run_game(
            strategy_name="thompson", truth=truth,
            t_max=12, N=64, eps=0.10, seed=8,
        )
        n_hits = sum(1 for r in traj["turns"]
                     if r["action"]["kind"] == "shot" and r["result"] == "HIT")
        assert traj["terminal"]["hits"] == n_hits


# --------------------------------------------------------------------------
# Determinism — same seed ⇒ identical trajectory
# --------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_trajectory(self, truth):
        kw = dict(strategy_name="thompson", truth=truth,
                  t_max=10, N=64, eps=0.10, seed=42)
        a = run_game(**kw)
        b = run_game(**kw)
        assert a == b

    def test_different_seeds_diverge(self, truth):
        # On a small random sampler, two different seeds should not produce
        # pixel-identical trajectories (probability of collision is ~0).
        a = run_game(strategy_name="thompson", truth=truth,
                     t_max=10, N=64, eps=0.10, seed=11)
        b = run_game(strategy_name="thompson", truth=truth,
                     t_max=10, N=64, eps=0.10, seed=22)
        assert a["turns"] != b["turns"]


# --------------------------------------------------------------------------
# Ask handling (BSC noise on the wire)
# --------------------------------------------------------------------------

class TestAskNoise:
    def test_ask_observation_is_0_or_1(self, truth):
        traj = run_game(
            strategy_name="eig", truth=truth,
            t_max=6, N=64, eps=0.10, seed=9,
        )
        for rec in traj["turns"]:
            if rec["action"]["kind"] == "ask":
                assert rec["observed"] in (0, 1)

    def test_ask_with_zero_eps_is_truthful(self, truth):
        """eps=0 ⇒ BSC is lossless, so observed equals truth answer."""
        from engine.questions import evaluate, question_by_id

        traj = run_game(
            strategy_name="eig", truth=truth,
            t_max=6, N=64, eps=0.0, seed=10,
        )
        for rec in traj["turns"]:
            if rec["action"]["kind"] == "ask":
                q = question_by_id(rec["action"]["question_id"])
                assert rec["observed"] == evaluate(q, truth)


# --------------------------------------------------------------------------
# JSON serialisation
# --------------------------------------------------------------------------

class TestJsonSerialisation:
    def test_trajectory_round_trips(self, truth):
        traj = run_game(
            strategy_name="thompson", truth=truth,
            t_max=5, N=32, eps=0.10, seed=13,
        )
        blob = trajectory_to_json(traj)
        restored = json.loads(blob)
        assert restored["strategy"] == traj["strategy"]
        assert restored["terminal"] == traj["terminal"]
        # Cells become lists after JSON, so compare via JSON-normalised original.
        assert json.loads(json.dumps(traj)) == restored


# --------------------------------------------------------------------------
# Unknown strategy raises cleanly
# --------------------------------------------------------------------------

class TestErrors:
    def test_unknown_strategy_raises(self, truth):
        with pytest.raises(ValueError, match="unknown strategy"):
            run_game(
                strategy_name="no_such_strategy",
                truth=truth, t_max=3, N=16, eps=0.10, seed=0,
            )
