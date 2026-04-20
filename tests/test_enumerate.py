"""Tests for engine.enumerate — exact enumeration over the constraint set S.

Provides:
    - enumerate_S() -> (count: int, mu_0: np.ndarray)
    - Optionally a stream-style generator for incremental use.

Canonical tie-breaking: the two length-3 ships are indistinguishable, so the
enumerator enforces index order (i_3a < i_3b) on the single-ship placement
list to count each configuration exactly once.  Same-length permutations are
not double-counted.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.board import BOARD_SIZE, FLEET_LENGTHS, MAX_HITS
from engine.enumerate import enumerate_S


# --------------------------------------------------------------------------
# Shape and basic invariants
# --------------------------------------------------------------------------

@pytest.mark.slow
class TestEnumerationShape:
    """Enumerating |S| on 8x8 with fleet {4,3,3,2} runs in O(|S|); mark slow."""

    def test_returns_positive_count_and_8x8_grid(self):
        count, mu = enumerate_S()
        assert isinstance(count, int)
        assert count > 0
        assert mu.shape == (BOARD_SIZE, BOARD_SIZE)
        assert mu.dtype == float

    def test_mu_values_in_unit_interval(self):
        _, mu = enumerate_S()
        assert float(mu.min()) >= 0.0
        assert float(mu.max()) <= 1.0

    def test_board_mean_equals_fleet_fraction(self):
        """Σ μ(c) = E[hits on a random cell × 64] = MAX_HITS.
        Equivalently, mean of μ over the 64 cells = 12/64 = 0.1875 exactly."""
        _, mu = enumerate_S()
        expected = MAX_HITS / (BOARD_SIZE * BOARD_SIZE)
        assert float(mu.mean()) == pytest.approx(expected, abs=1e-12)
        assert float(mu.sum()) == pytest.approx(float(MAX_HITS), abs=1e-10)


# --------------------------------------------------------------------------
# Board symmetries — μ_0 must respect the dihedral group of the square
# (90°/180°/270° rotation, horizontal/vertical/diagonal reflection)
# --------------------------------------------------------------------------

@pytest.mark.slow
class TestSymmetries:
    def test_horizontal_reflection(self):
        _, mu = enumerate_S()
        np.testing.assert_allclose(mu, mu[:, ::-1], atol=1e-12)

    def test_vertical_reflection(self):
        _, mu = enumerate_S()
        np.testing.assert_allclose(mu, mu[::-1, :], atol=1e-12)

    def test_180_rotation(self):
        _, mu = enumerate_S()
        np.testing.assert_allclose(mu, mu[::-1, ::-1], atol=1e-12)

    def test_main_diagonal_reflection(self):
        _, mu = enumerate_S()
        np.testing.assert_allclose(mu, mu.T, atol=1e-12)


# --------------------------------------------------------------------------
# Topology sanity — corner/middle-of-edge ordering from battleship.md
# --------------------------------------------------------------------------

@pytest.mark.slow
class TestTopology:
    def test_corner_is_global_minimum_class(self):
        """Corners have the fewest orientations that include them → smallest μ."""
        _, mu = enumerate_S()
        corners = [mu[0, 0], mu[0, 7], mu[7, 0], mu[7, 7]]
        non_corner = [mu[r, c] for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
                      if (r, c) not in {(0, 0), (0, 7), (7, 0), (7, 7)}]
        assert max(corners) <= min(non_corner)

    def test_middle_edge_exceeds_corner(self):
        """The middle-of-edge class carries strictly more mass than corners
        (no-touch constraint hurts corners most — they have fewer 'neighbor
        room' configurations where another ship can abut diagonally)."""
        _, mu = enumerate_S()
        edge_middle = [mu[0, 3], mu[0, 4], mu[3, 0], mu[4, 0],
                       mu[7, 3], mu[7, 4], mu[3, 7], mu[4, 7]]
        corners = [mu[0, 0], mu[0, 7], mu[7, 0], mu[7, 7]]
        assert min(edge_middle) > max(corners)


# --------------------------------------------------------------------------
# Determinism — enumerate_S is a pure function of FLEET_LENGTHS / BOARD_SIZE
# --------------------------------------------------------------------------

@pytest.mark.slow
class TestDeterminism:
    def test_two_calls_agree(self):
        a_count, a_mu = enumerate_S()
        b_count, b_mu = enumerate_S()
        assert a_count == b_count
        np.testing.assert_array_equal(a_mu, b_mu)
