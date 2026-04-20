"""Tests for engine.exact — exact posterior over the full |S|=5,174,944 configurations.

Contract
--------
ExactPosterior is the drop-in replacement for ParticleFilter when |S| is small
enough to enumerate in memory.  Unlike a sampled filter:

  * The weights vector has length |S| exactly; every configuration in S is
    present and uniquely represented.
  * Under uniform weights, cell_marginal_grid() reproduces the enumerated
    μ_0 to machine precision (not Monte-Carlo error).
  * reweight_shot applies a delta likelihood (noiseless); reweight_ask applies
    BSC(ε).  Both are vectorised O(|S|).
  * answers_for(q) returns a (|S|,) bool vector of Q(s) for every config s.
    Cheap enough to be cached lazily.
  * thompson_sample draws config i ~ weights, picks an unshot occupied cell.

Zero-mass recovery
------------------
If a reweight zeroes every weight (genuinely impossible under noiseless shots
against truth ∈ S; possible via test setup), fall back to uniform prior — the
filter becomes uninformed but downstream consumers still produce legal actions.

All tests here are @slow because constructing an ExactPosterior triggers the
~30s full enumeration of S (cached per process).
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.board import BOARD_SIZE, MAX_HITS
from engine.enumerate import enumerate_S
from engine.questions import question_by_id
from engine.exact import ExactPosterior


# --------------------------------------------------------------------------
# Prior ≡ enumeration — the defining property of exact inference
# --------------------------------------------------------------------------

@pytest.mark.slow
class TestPriorMatchesEnumeration:
    def test_n_equals_S(self):
        count, _ = enumerate_S()
        ep = ExactPosterior()
        assert ep.n_configurations == count

    def test_prior_weights_uniform(self):
        ep = ExactPosterior()
        n = ep.n_configurations
        assert ep.weights.shape == (n,)
        np.testing.assert_allclose(ep.weights, 1.0 / n, atol=1e-15)
        assert ep.weights.sum() == pytest.approx(1.0, abs=1e-10)

    def test_uniform_prior_mu_equals_mu_0_exactly(self):
        """Under uniform weights, cell_marginal_grid reproduces enumerate_S's μ_0."""
        _, mu_0 = enumerate_S()
        ep = ExactPosterior()
        mu = ep.cell_marginal_grid()
        assert mu.shape == (BOARD_SIZE, BOARD_SIZE)
        np.testing.assert_allclose(mu, mu_0, atol=1e-10)

    def test_mu_sum_equals_max_hits(self):
        ep = ExactPosterior()
        mu = ep.cell_marginal_grid()
        assert mu.sum() == pytest.approx(float(MAX_HITS), abs=1e-10)


# --------------------------------------------------------------------------
# Shot reweighting — delta likelihood, noiseless
# --------------------------------------------------------------------------

@pytest.mark.slow
class TestShotReweight:
    def test_hit_forces_cell_marginal_to_one(self):
        ep = ExactPosterior()
        ep.reweight_shot(cell=(0, 0), observed=1)
        mu = ep.cell_marginal_grid()
        assert mu[0, 0] == pytest.approx(1.0, abs=1e-10)

    def test_miss_forces_cell_marginal_to_zero(self):
        ep = ExactPosterior()
        ep.reweight_shot(cell=(0, 0), observed=0)
        mu = ep.cell_marginal_grid()
        assert mu[0, 0] == pytest.approx(0.0, abs=1e-10)

    def test_shot_preserves_normalisation(self):
        ep = ExactPosterior()
        ep.reweight_shot(cell=(3, 3), observed=1)
        assert ep.weights.sum() == pytest.approx(1.0, abs=1e-10)

    def test_hit_zeros_inconsistent_configs(self):
        """Every config with (0,0) unoccupied gets weight zero."""
        ep = ExactPosterior()
        a = ep.answers_for(question_by_id("cell:0-0"))   # True where (0,0) occupied
        ep.reweight_shot(cell=(0, 0), observed=1)
        # Configs with a=False must have zero weight.
        assert ep.weights[~a].sum() == 0.0
        # Configs with a=True must carry all the mass.
        assert ep.weights[a].sum() == pytest.approx(1.0, abs=1e-10)


# --------------------------------------------------------------------------
# Ask reweighting — BSC(ε), soft
# --------------------------------------------------------------------------

@pytest.mark.slow
class TestAskReweight:
    def test_ask_keeps_all_weights_positive(self):
        ep = ExactPosterior()
        ep.reweight_ask(question=question_by_id("row:0"), observed=1, eps=0.10)
        assert np.all(ep.weights > 0)
        assert ep.weights.sum() == pytest.approx(1.0, abs=1e-10)

    def test_ask_matching_configs_gain_weight(self):
        ep = ExactPosterior()
        q = question_by_id("row:0")
        a = ep.answers_for(q)
        prior_mass_match = float(a.mean())   # under uniform prior
        ep.reweight_ask(question=q, observed=1, eps=0.05)
        post_mass_match = ep.weights[a].sum()
        assert post_mass_match > prior_mass_match

    def test_ask_eps_zero_is_hard_filter(self):
        """ε=0 means the answer is infallible — configs that disagree get weight 0."""
        ep = ExactPosterior()
        q = question_by_id("row:0")
        a = ep.answers_for(q)
        ep.reweight_ask(question=q, observed=1, eps=0.0)
        assert ep.weights[~a].sum() == 0.0
        assert ep.weights[a].sum() == pytest.approx(1.0, abs=1e-10)

    def test_ask_eps_half_is_no_information(self):
        """ε=0.5 means the answer is pure noise — posterior equals prior."""
        ep = ExactPosterior()
        before = ep.weights.copy()
        ep.reweight_ask(question=question_by_id("col:3"), observed=1, eps=0.5)
        np.testing.assert_allclose(ep.weights, before, atol=1e-15)


# --------------------------------------------------------------------------
# Answer vectors — Q(s) for every config s
# --------------------------------------------------------------------------

@pytest.mark.slow
class TestAnswerVectors:
    def test_cell_answer_mean_equals_mu_0(self):
        """Under uniform prior, E[1[cell c occupied]] = μ_0[c]."""
        _, mu_0 = enumerate_S()
        ep = ExactPosterior()
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                a = ep.answers_for(question_by_id(f"cell:{r}-{c}"))
                assert a.shape == (ep.n_configurations,)
                assert a.dtype == bool
                assert float(a.mean()) == pytest.approx(float(mu_0[r, c]), abs=1e-10)

    def test_row_answer_positive_fraction(self):
        ep = ExactPosterior()
        a = ep.answers_for(question_by_id("row:3"))
        assert a.shape == (ep.n_configurations,)
        assert a.dtype == bool
        assert 0.0 < a.mean() < 1.0

    def test_quadrant_answer_positive_fraction(self):
        ep = ExactPosterior()
        a = ep.answers_for(question_by_id("quad:NW"))
        assert a.shape == (ep.n_configurations,)
        assert a.dtype == bool
        assert 0.0 < a.mean() < 1.0

    def test_hparity_symmetric_prior(self):
        """Under uniform prior, horizontal vs vertical are symmetric by rotation:
        Pr[hparity:L = 1] ≈ Pr[at least one length-L ship horizontal] — must be
        in (0, 1) and should be symmetric, but the exact value depends on fleet
        composition.  Sanity check: in (0, 1) and close to 0.5 for a single ship.
        """
        ep = ExactPosterior()
        a4 = ep.answers_for(question_by_id("hparity:4"))
        assert a4.dtype == bool
        # For the singleton length-4 ship, hparity:4 is symmetric ⇒ exactly 0.5.
        assert float(a4.mean()) == pytest.approx(0.5, abs=1e-10)

    def test_answer_vector_is_cached(self):
        """Same question returns the same object (or array) on repeat."""
        ep = ExactPosterior()
        a1 = ep.answers_for(question_by_id("row:3"))
        a2 = ep.answers_for(question_by_id("row:3"))
        np.testing.assert_array_equal(a1, a2)


# --------------------------------------------------------------------------
# Thompson sampling — posterior-conditioned shot selection
# --------------------------------------------------------------------------

@pytest.mark.slow
class TestThompsonSample:
    def test_returns_unshot_cell(self):
        rng = np.random.default_rng(0)
        ep = ExactPosterior()
        shots = frozenset([(0, 0), (1, 1)])
        cell = ep.thompson_sample(rng=rng, shots_fired=shots)
        assert cell not in shots
        r, c = cell
        assert 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

    def test_draws_ship_cell(self):
        """Under uniform prior, the drawn cell is a ship cell of some config."""
        rng = np.random.default_rng(1)
        ep = ExactPosterior()
        cell = ep.thompson_sample(rng=rng, shots_fired=frozenset())
        a = ep.answers_for(question_by_id(f"cell:{cell[0]}-{cell[1]}"))
        assert bool(a.any())

    def test_collapsed_posterior_draws_from_lone_config(self):
        """When all weight is on one config, TS must pick that config's ship cells."""
        ep = ExactPosterior()
        # Collapse onto the config with (0,0) occupied — pick any valid index.
        a00 = ep.answers_for(question_by_id("cell:0-0"))
        idx = int(np.where(a00)[0][0])
        ep.weights = np.zeros_like(ep.weights)
        ep.weights[idx] = 1.0
        rng = np.random.default_rng(0)
        cell = ep.thompson_sample(rng=rng, shots_fired=frozenset())
        # The cell must be one of config idx's occupied cells (12 of them).
        occupied = ep.occupied_cells_of(idx)   # list of 12 (r, c) tuples
        assert cell in occupied


# --------------------------------------------------------------------------
# Determinism — ExactPosterior's internal state is a pure function of history
# --------------------------------------------------------------------------

@pytest.mark.slow
class TestDeterminism:
    def test_repeated_reweight_sequence_matches(self):
        def play():
            ep = ExactPosterior()
            ep.reweight_shot(cell=(0, 0), observed=1)
            ep.reweight_ask(question=question_by_id("col:3"), observed=0, eps=0.10)
            ep.reweight_shot(cell=(4, 4), observed=0)
            return ep.weights.copy()
        a = play()
        b = play()
        np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------
# Zero-mass recovery
# --------------------------------------------------------------------------

@pytest.mark.slow
class TestRecovery:
    def test_zero_mass_reweight_falls_back_to_uniform_prior(self):
        """If weights go to zero (impossible under noiseless shots on legal truth,
        but constructible in tests), we restore uniform prior rather than crash."""
        ep = ExactPosterior()
        a00 = ep.answers_for(question_by_id("cell:0-0"))
        idx_no_00 = int(np.where(~a00)[0][0])
        ep.weights = np.zeros_like(ep.weights)
        ep.weights[idx_no_00] = 1.0
        # Now say we observed a HIT at (0,0) — the lone config has X(0,0)=0 ⇒
        # total mass goes to 0.  The filter should fall back to uniform.
        ep.reweight_shot(cell=(0, 0), observed=1)
        assert ep.weights.sum() == pytest.approx(1.0, abs=1e-10)
        np.testing.assert_allclose(ep.weights, 1.0 / ep.n_configurations, atol=1e-15)
