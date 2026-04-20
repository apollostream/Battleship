"""Tests for engine.questions — catalogue, Q(s) evaluator, BSC(ε) likelihood.

TDD spec per battleship.md:
  Question catalogue (binary questions Q: S → {0, 1}):
    * 64 cell queries:   "is there a ship at cell (r, c)?"       ids: cell:r-c
    * 8  row queries:    "does row r contain any ship cell?"     ids: row:r
    * 8  col queries:    "does col c contain any ship cell?"     ids: col:c
    * 4  quadrant queries: "does quadrant Q contain any ship cell?"
                           quadrants = {NW, NE, SW, SE} (4x4 blocks).
    * 3  hparity queries: "is any length-L ship placed horizontally?"
                           one per *distinct* ship length, L ∈ {2, 3, 4}.
  Total: 64 + 8 + 8 + 4 + 3 = 87 questions.

  Each question has a stable string id so it can be serialized into the
  trajectory log (e.g., "cell:3-5", "row:2", "col:7", "quad:NE", "hparity:3").

  BSC(ε) likelihood:
    p_ask(y | s, q) = (1-ε)^{1[y == Q(s)]} * ε^{1[y != Q(s)]}
"""
from __future__ import annotations

import math
import pytest

from engine.board import Configuration, Orientation, Ship
from engine.questions import (
    QUESTION_CATALOGUE,
    Question,
    bsc_likelihood,
    build_catalogue,
    evaluate,
    question_by_id,
)


# --------------------------------------------------------------------------
# Reusable fixture — same valid configuration from test_board
# --------------------------------------------------------------------------

@pytest.fixture
def cfg() -> Configuration:
    return Configuration(ships=(
        Ship(4, 0, 0, Orientation.HORIZONTAL),   # (0,0)..(0,3)
        Ship(3, 2, 0, Orientation.HORIZONTAL),   # (2,0)..(2,2)
        Ship(3, 4, 4, Orientation.HORIZONTAL),   # (4,4)..(4,6)
        Ship(2, 7, 6, Orientation.HORIZONTAL),   # (7,6)..(7,7)
    ))


# --------------------------------------------------------------------------
# Catalogue shape
# --------------------------------------------------------------------------

class TestCatalogue:
    def test_catalogue_size(self):
        # 64 cell + 8 row + 8 col + 4 quadrant + 3 hparity = 87
        assert len(QUESTION_CATALOGUE) == 87

    def test_catalogue_ids_unique(self):
        ids = [q.id for q in QUESTION_CATALOGUE]
        assert len(ids) == len(set(ids))

    def test_build_catalogue_idempotent(self):
        a = build_catalogue()
        b = build_catalogue()
        assert [q.id for q in a] == [q.id for q in b]

    def test_catalogue_contains_expected_counts(self):
        kinds = [q.kind for q in QUESTION_CATALOGUE]
        assert kinds.count("cell") == 64
        assert kinds.count("row") == 8
        assert kinds.count("col") == 8
        assert kinds.count("quadrant") == 4
        assert kinds.count("hparity") == 3

    def test_question_by_id_roundtrip(self):
        q = question_by_id("cell:3-5")
        assert q.kind == "cell"
        assert q.id == "cell:3-5"


# --------------------------------------------------------------------------
# Q(s) — evaluation against a known configuration
# --------------------------------------------------------------------------

class TestEvaluate:
    def test_cell_query_positive(self, cfg):
        assert evaluate(question_by_id("cell:0-0"), cfg) == 1
        assert evaluate(question_by_id("cell:7-7"), cfg) == 1

    def test_cell_query_negative(self, cfg):
        assert evaluate(question_by_id("cell:1-1"), cfg) == 0
        assert evaluate(question_by_id("cell:3-3"), cfg) == 0

    def test_row_query(self, cfg):
        # Row 0 has the length-4 horizontal ship.
        assert evaluate(question_by_id("row:0"), cfg) == 1
        # Row 1 is empty.
        assert evaluate(question_by_id("row:1"), cfg) == 0
        # Row 7 has the length-2 ship.
        assert evaluate(question_by_id("row:7"), cfg) == 1

    def test_col_query(self, cfg):
        # Col 0 crosses the length-4 and length-3 horizontal ships at their starts.
        assert evaluate(question_by_id("col:0"), cfg) == 1
        # Col 3 only has the length-4's rightmost cell (0,3).
        assert evaluate(question_by_id("col:3"), cfg) == 1
        # Col 7 has the length-2 ship's rightmost cell.
        assert evaluate(question_by_id("col:7"), cfg) == 1
        # Col 1 has (0,1) from the length-4 and (2,1) from the length-3.
        assert evaluate(question_by_id("col:1"), cfg) == 1
        # No ship cell in col that only the interior-3 skips?  Check (col that's empty).
        # Actually with this cfg, col 3 has (0,3); col 7 has (7,7).  Pick one that's empty:
        assert evaluate(question_by_id("cell:3-3"), cfg) == 0  # sanity

    def test_quadrant_query(self, cfg):
        # Quadrants are 4x4 sub-boards, ids: NW (rows 0-3, cols 0-3),
        #   NE (rows 0-3, cols 4-7), SW (rows 4-7, cols 0-3), SE (rows 4-7, cols 4-7).
        assert evaluate(question_by_id("quad:NW"), cfg) == 1   # length-4 + length-3 start
        assert evaluate(question_by_id("quad:NE"), cfg) == 0   # no ship cell here
        assert evaluate(question_by_id("quad:SW"), cfg) == 0   # no ship cell here
        assert evaluate(question_by_id("quad:SE"), cfg) == 1   # length-3 + length-2

    def test_hparity_all_horizontal_in_fixture(self, cfg):
        # Every ship in the fixture is horizontal.
        assert evaluate(question_by_id("hparity:2"), cfg) == 1
        assert evaluate(question_by_id("hparity:3"), cfg) == 1
        assert evaluate(question_by_id("hparity:4"), cfg) == 1

    def test_hparity_distinct_lengths(self):
        """One query per distinct fleet length: {2, 3, 4}."""
        kinds = [q.id for q in QUESTION_CATALOGUE if q.kind == "hparity"]
        assert sorted(kinds) == ["hparity:2", "hparity:3", "hparity:4"]

    def test_hparity_any_length3_horizontal(self):
        """If at least one length-3 ship is horizontal, the query returns 1."""
        cfg_one_vertical = Configuration(ships=(
            Ship(4, 0, 0, Orientation.HORIZONTAL),
            Ship(3, 2, 0, Orientation.HORIZONTAL),   # horizontal length-3
            Ship(3, 0, 5, Orientation.VERTICAL),     # vertical length-3
            Ship(2, 7, 6, Orientation.HORIZONTAL),
        ))
        # At least one length-3 is horizontal → answer 1.
        assert evaluate(question_by_id("hparity:3"), cfg_one_vertical) == 1

    def test_hparity_all_length3_vertical(self):
        cfg_both_vertical = Configuration(ships=(
            Ship(4, 0, 0, Orientation.HORIZONTAL),
            Ship(3, 2, 0, Orientation.VERTICAL),     # vertical length-3
            Ship(3, 2, 4, Orientation.VERTICAL),     # vertical length-3
            Ship(2, 7, 6, Orientation.HORIZONTAL),
        ))
        assert evaluate(question_by_id("hparity:3"), cfg_both_vertical) == 0


# --------------------------------------------------------------------------
# BSC(ε) likelihood
# --------------------------------------------------------------------------

class TestBSCLikelihood:
    def test_agreement_probability(self):
        # When observed matches truth, likelihood = 1 - ε.
        assert bsc_likelihood(observed=1, truth=1, eps=0.10) == pytest.approx(0.90)
        assert bsc_likelihood(observed=0, truth=0, eps=0.10) == pytest.approx(0.90)

    def test_disagreement_probability(self):
        # When observed flips the truth, likelihood = ε.
        assert bsc_likelihood(observed=1, truth=0, eps=0.10) == pytest.approx(0.10)
        assert bsc_likelihood(observed=0, truth=1, eps=0.10) == pytest.approx(0.10)

    def test_zero_noise_is_delta(self):
        assert bsc_likelihood(1, 1, eps=0.0) == 1.0
        assert bsc_likelihood(1, 0, eps=0.0) == 0.0

    def test_half_noise_is_uninformative(self):
        # ε = 0.5 → flat likelihood, same for every (observed, truth).
        for obs in (0, 1):
            for tru in (0, 1):
                assert bsc_likelihood(obs, tru, eps=0.5) == pytest.approx(0.5)

    def test_epsilon_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            bsc_likelihood(1, 1, eps=-0.01)
        with pytest.raises(ValueError):
            bsc_likelihood(1, 1, eps=1.0 + 1e-9)

    def test_log_likelihood_matches_log(self):
        # Convenience — if we expose a log variant:
        lik = bsc_likelihood(1, 1, eps=0.10)
        assert math.log(lik) == pytest.approx(math.log(0.90))
