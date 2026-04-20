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
            assert "decision_value" in rec
            assert "cum_net_reward" in rec
            kind = rec["action"]["kind"]
            assert kind in ("shot", "ask")
            if kind == "shot":
                assert "cell" in rec["action"]
                assert "result" in rec  # "HIT" or "MISS"
                assert rec["result"] in ("HIT", "MISS")
            else:
                assert "question_id" in rec["action"]

    def test_cum_net_reward_matches_terminal(self, truth):
        """cum_net_reward at the last turn should equal terminal.net_reward."""
        traj = run_game(
            strategy_name="thompson", truth=truth,
            t_max=10, N=32, eps=0.10, seed=5,
        )
        if traj["turns"]:
            last_cum = traj["turns"][-1]["cum_net_reward"]
            assert last_cum == traj["terminal"]["net_reward"]

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


# --------------------------------------------------------------------------
# Realized info gain + rationale per turn (dock "Why?" + journal Δ-info column)
# --------------------------------------------------------------------------

@pytest.mark.slow
class TestTurnRationaleAndInfoGain:
    """Every turn record must carry `info_gain_nats` + `rationale`.

    Rationale: snapshot of the strategy's decision at decision time — both
    branches (shot + ask), plus which one was chosen.  info_gain_nats: the
    realized KL of the posterior update, in nats; compared against each
    branch's expected score this is "how much did the coin land teach us?"
    """

    def test_thompson_records_info_gain_and_rationale(self, truth):
        traj = run_game(
            strategy_name="thompson", truth=truth,
            t_max=8, N=32, eps=0.10, seed=100,
        )
        for rec in traj["turns"]:
            assert "info_gain_nats" in rec
            assert isinstance(rec["info_gain_nats"], float)
            assert rec["info_gain_nats"] >= -1e-10
            rat = rec["rationale"]
            assert rat["chosen"] == "shot"
            assert rat["ask"] is None
            shot = rat["shot"]
            assert 0.0 <= shot["mu"] <= 1.0 + 1e-9
            # Thompson's action must match the rationale's shot cell.
            assert list(shot["cell"]) == rec["action"]["cell"]

    def test_eig_records_both_branches(self, truth):
        traj = run_game(
            strategy_name="eig", truth=truth,
            t_max=4, N=32, eps=0.10, seed=101,
        )
        for rec in traj["turns"]:
            rat = rec["rationale"]
            # Both branches present on every turn.
            assert rat["shot"] is not None and rat["ask"] is not None
            assert rat["shot"]["threshold"] == pytest.approx(2.0 / 3.0)
            assert rat["ask"]["metric"] == "eig"
            assert rat["chosen"] in ("shot", "ask")

    def test_info_gain_shot_matches_formula(self, truth):
        """GameSession.step: for a shot, info_gain_nats = -log μ_pre (hit) or
        -log(1-μ_pre) (miss) with μ_pre = cell_marginal at decision time."""
        from simulator.runner import GameSession
        from engine.metrics import info_gain_shot

        sess = GameSession(
            strategy_name="thompson", truth=truth,
            t_max=6, N=32, eps=0.10, seed=102,
        )
        # Peek at the cell marginal *before* stepping.
        mu_pre = sess.strategy.filter.cell_marginal_grid().copy()
        rec = sess.step()
        assert rec["action"]["kind"] == "shot"
        r, c = rec["action"]["cell"]
        expected = info_gain_shot(mu_c=float(mu_pre[r, c]), observed=rec["observed"])
        assert rec["info_gain_nats"] == pytest.approx(expected, abs=1e-10)

    def test_info_gain_ask_matches_formula(self, truth):
        from simulator.runner import GameSession
        from engine.metrics import info_gain_ask_bsc
        from engine.questions import question_by_id
        import numpy as np

        sess = GameSession(
            strategy_name="eig", truth=truth,
            t_max=4, N=32, eps=0.10, seed=103,
        )
        rec = sess.step()
        if rec["action"]["kind"] != "ask":
            pytest.skip("EIG shot this turn, not ask")
        qid = rec["action"]["question_id"]
        q = question_by_id(qid)
        # The ask happens first; after observe, the posterior has shifted.
        # We approximate p_hat by redoing the dot product on a fresh filter
        # at the SAME seed but without stepping — the pre-action state.
        fresh = GameSession(
            strategy_name="eig", truth=truth,
            t_max=4, N=32, eps=0.10, seed=103,
        )
        a = fresh.strategy.filter.answers_for(q)
        p_hat = float(np.dot(fresh.strategy.filter.weights, a))
        expected = info_gain_ask_bsc(
            p_hat=p_hat, eps=0.10, observed=rec["observed"],
        )
        assert rec["info_gain_nats"] == pytest.approx(expected, abs=1e-10)

    def test_trajectory_json_serializes_rationale(self, truth):
        traj = run_game(
            strategy_name="thompson", truth=truth,
            t_max=4, N=32, eps=0.10, seed=104,
        )
        blob = trajectory_to_json(traj)
        restored = json.loads(blob)
        for rec in restored["turns"]:
            assert "info_gain_nats" in rec
            assert "rationale" in rec
            # Cells come back as [r, c] lists.
            if rec["rationale"]["shot"] is not None:
                assert len(rec["rationale"]["shot"]["cell"]) == 2
