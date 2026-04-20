"""Tests for engine.board — Ship, Configuration, GameState.

TDD spec per battleship.md:
  - 8x8 grid, ships {4, 3, 3, 2}, horizontal or vertical, placed within bounds
  - Constraint set S: exactly 4 ships, lengths {2,3,3,4}, no overlap, no touching
    (orthogonally-adjacent cells of distinct ships are forbidden)
  - X_c(s) = 1 iff config s occupies cell c
  - Shot outcomes are noiseless: apply_shot returns exactly HIT/MISS
  - Shots cannot repeat a cell
  - Game terminates on (H=12) OR (T=T_max)
  - Score per trajectory is the pair (H, T)
"""
from __future__ import annotations

import pytest

from engine.board import (
    BOARD_SIZE,
    FLEET_LENGTHS,
    MAX_HITS,
    Ship,
    Orientation,
    Configuration,
    GameState,
    ShotResult,
    score_comparison_key,
)


# --------------------------------------------------------------------------
# Ship — a single oriented placement
# --------------------------------------------------------------------------

class TestShip:
    def test_horizontal_cells(self):
        s = Ship(length=3, row=2, col=4, orientation=Orientation.HORIZONTAL)
        assert s.cells() == [(2, 4), (2, 5), (2, 6)]

    def test_vertical_cells(self):
        s = Ship(length=4, row=1, col=0, orientation=Orientation.VERTICAL)
        assert s.cells() == [(1, 0), (2, 0), (3, 0), (4, 0)]

    def test_out_of_bounds_horizontal_rejected(self):
        with pytest.raises(ValueError):
            Ship(length=4, row=0, col=6, orientation=Orientation.HORIZONTAL).validate_in_bounds()

    def test_out_of_bounds_vertical_rejected(self):
        with pytest.raises(ValueError):
            Ship(length=4, row=6, col=0, orientation=Orientation.VERTICAL).validate_in_bounds()

    def test_length_one_allowed_as_primitive_but_not_in_fleet(self):
        s = Ship(length=1, row=0, col=0, orientation=Orientation.HORIZONTAL)
        assert s.cells() == [(0, 0)]
        assert s.length not in FLEET_LENGTHS or FLEET_LENGTHS.count(1) == 0


# --------------------------------------------------------------------------
# Configuration — 4-ship fleet satisfying S
# --------------------------------------------------------------------------

@pytest.fixture
def valid_config() -> Configuration:
    """A hand-authored valid placement with generous separation."""
    return Configuration(ships=(
        Ship(4, 0, 0, Orientation.HORIZONTAL),   # (0,0) (0,1) (0,2) (0,3)
        Ship(3, 2, 0, Orientation.HORIZONTAL),   # (2,0) (2,1) (2,2)
        Ship(3, 4, 4, Orientation.HORIZONTAL),   # (4,4) (4,5) (4,6)
        Ship(2, 7, 6, Orientation.HORIZONTAL),   # (7,6) (7,7)
    ))


class TestConfiguration:
    def test_valid_configuration_accepted(self, valid_config):
        assert valid_config.is_valid()

    def test_fleet_must_have_four_ships(self):
        # Only 3 ships — invalid fleet composition
        cfg = Configuration(ships=(
            Ship(4, 0, 0, Orientation.HORIZONTAL),
            Ship(3, 2, 0, Orientation.HORIZONTAL),
            Ship(2, 4, 0, Orientation.HORIZONTAL),
        ))
        assert not cfg.is_valid()

    def test_fleet_lengths_must_be_4_3_3_2(self):
        # Wrong lengths — {4,3,2,2} instead of {4,3,3,2}
        cfg = Configuration(ships=(
            Ship(4, 0, 0, Orientation.HORIZONTAL),
            Ship(3, 2, 0, Orientation.HORIZONTAL),
            Ship(2, 4, 0, Orientation.HORIZONTAL),
            Ship(2, 6, 0, Orientation.HORIZONTAL),
        ))
        assert not cfg.is_valid()

    def test_overlap_forbidden(self):
        cfg = Configuration(ships=(
            Ship(4, 0, 0, Orientation.HORIZONTAL),
            Ship(3, 0, 2, Orientation.HORIZONTAL),   # overlaps at (0,2) (0,3)
            Ship(3, 4, 0, Orientation.HORIZONTAL),
            Ship(2, 6, 0, Orientation.HORIZONTAL),
        ))
        assert not cfg.is_valid()

    def test_touching_orthogonally_forbidden(self):
        cfg = Configuration(ships=(
            Ship(4, 0, 0, Orientation.HORIZONTAL),
            Ship(3, 1, 0, Orientation.HORIZONTAL),   # row 1 touches row 0
            Ship(3, 4, 0, Orientation.HORIZONTAL),
            Ship(2, 6, 0, Orientation.HORIZONTAL),
        ))
        assert not cfg.is_valid()

    def test_diagonal_touch_allowed(self):
        """Diagonally-adjacent ships are permitted; only orthogonal touches forbidden."""
        cfg = Configuration(ships=(
            Ship(4, 0, 0, Orientation.HORIZONTAL),   # (0,0)..(0,3)
            Ship(3, 1, 4, Orientation.HORIZONTAL),   # (1,4) diag to (0,3) — OK
            Ship(3, 4, 0, Orientation.HORIZONTAL),
            Ship(2, 6, 0, Orientation.HORIZONTAL),
        ))
        assert cfg.is_valid()

    def test_occupied_cells_count_equals_12(self, valid_config):
        assert len(valid_config.occupied_cells()) == MAX_HITS == 12

    def test_X_c_indicator(self, valid_config):
        assert valid_config.X((0, 0)) == 1
        assert valid_config.X((0, 3)) == 1
        assert valid_config.X((0, 4)) == 0   # just past length-4 ship
        assert valid_config.X((7, 7)) == 1


# --------------------------------------------------------------------------
# GameState — shot/ask accounting, termination, (H, T) score
# --------------------------------------------------------------------------

class TestGameState:
    def test_initial_state(self, valid_config):
        gs = GameState(truth=valid_config, t_max=80)
        assert gs.turn == 0
        assert gs.hits == 0
        assert not gs.terminated
        assert gs.shots_fired == frozenset()

    def test_apply_shot_hit(self, valid_config):
        gs = GameState(truth=valid_config, t_max=80)
        res = gs.apply_shot((0, 0))
        assert res == ShotResult.HIT
        assert gs.turn == 1
        assert gs.hits == 1
        assert (0, 0) in gs.shots_fired

    def test_apply_shot_miss(self, valid_config):
        gs = GameState(truth=valid_config, t_max=80)
        res = gs.apply_shot((1, 1))   # empty cell
        assert res == ShotResult.MISS
        assert gs.turn == 1
        assert gs.hits == 0

    def test_cannot_shoot_same_cell_twice(self, valid_config):
        gs = GameState(truth=valid_config, t_max=80)
        gs.apply_shot((0, 0))
        with pytest.raises(ValueError):
            gs.apply_shot((0, 0))

    def test_apply_ask_advances_turn_but_not_hits(self, valid_config):
        gs = GameState(truth=valid_config, t_max=80)
        # Represent an "ask" as a turn-advancing action with no board state change.
        # Its noisy answer is a concern of questions.py; GameState just tracks turn count
        # and logs the (question_id, observed_answer) pair for replay / UI consumption.
        gs.apply_ask(question_id="row:3", observed_answer=1)
        assert gs.turn == 1
        assert gs.hits == 0
        assert gs.shots_fired == frozenset()
        assert gs.history[-1] == ("ask", "row:3", 1)

    def test_terminates_on_all_ship_cells_hit(self, valid_config):
        gs = GameState(truth=valid_config, t_max=80)
        for c in valid_config.occupied_cells():
            gs.apply_shot(c)
        assert gs.hits == 12
        assert gs.terminated
        assert gs.score == (12, 12)  # all 12 hits in exactly 12 shots (no misses, no asks)

    def test_terminates_on_turn_cap(self, valid_config):
        gs = GameState(truth=valid_config, t_max=3)
        gs.apply_ask("row:0", 1)
        gs.apply_ask("col:0", 0)
        gs.apply_shot((1, 1))   # miss
        assert gs.terminated
        assert gs.hits == 0
        assert gs.score == (0, 3)

    def test_cannot_act_after_termination(self, valid_config):
        gs = GameState(truth=valid_config, t_max=1)
        gs.apply_shot((1, 1))  # consumes the only turn
        with pytest.raises(RuntimeError):
            gs.apply_shot((2, 2))

    def test_score_primary_sort_key(self):
        """Pure function — ranks trajectories.  Lower key = better.

        Cases (all against t_max=80):
          A: H=12, T=30   — sank fleet fast
          B: H=12, T=50   — sank fleet slower
          C: H=10, T=80   — timed out with 10 hits
          D: H= 6, T=80   — timed out with 6 hits
          expected order: A < B < C < D
        """
        a = score_comparison_key(hits=12, turns=30, t_max=80)
        b = score_comparison_key(hits=12, turns=50, t_max=80)
        c = score_comparison_key(hits=10, turns=80, t_max=80)
        d = score_comparison_key(hits= 6, turns=80, t_max=80)
        assert a < b < c < d


# --------------------------------------------------------------------------
# MAX_HITS invariant
# --------------------------------------------------------------------------

def test_max_hits_equals_fleet_sum():
    """Structural invariant — MAX_HITS is always sum(FLEET_LENGTHS)."""
    assert MAX_HITS == sum(FLEET_LENGTHS)
