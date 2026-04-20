"""Tests for engine.smc — rejection sampler, particle filter, posterior updates.

TDD spec per battleship.md and CLAUDE.md:

Sampler must sample uniformly over S.  The induced cell marginal μ_0(c)
is non-uniform by symmetry:
    corners    (A1, A8, H1, H8)                  ≈ 0.13
    edges      (non-corner perimeter)            ≈ 0.16 – 0.18
    interior   (inner 4x4 block, rows/cols 2..5) ≈ 0.20 – 0.23
    board mean (all 64 cells averaged)           = 12/64 = 0.1875 exactly

Large-N rejection draw should recover these to within Monte-Carlo error.

Particle filter must:
    * reweight on shot observations using a *delta* likelihood (noiseless)
      — inconsistent particles get weight zero.
    * reweight on ask observations using BSC(ε) likelihood — all particles
      stay non-zero (soft).
    * expose effective sample size (ESS) and resample on low ESS.
    * provide cell marginal μ(c) = Σ_i w_i · X_c(s_i) on demand.
    * provide Thompson samples: draw a particle, pick a shot target on it.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.board import BOARD_SIZE, MAX_HITS, Configuration, Orientation, Ship
from engine.questions import evaluate, question_by_id
from engine.smc import (
    ParticleFilter,
    sample_configuration,
    sample_particles,
)


RNG_SEED = 20260419


# --------------------------------------------------------------------------
# Rejection sampler — uniform over S
# --------------------------------------------------------------------------

class TestSampler:
    def test_single_sample_is_valid(self):
        rng = np.random.default_rng(RNG_SEED)
        cfg = sample_configuration(rng)
        assert cfg.is_valid()
        assert len(cfg.occupied_cells()) == MAX_HITS

    def test_sample_particles_all_valid(self):
        rng = np.random.default_rng(RNG_SEED)
        particles = sample_particles(N=256, rng=rng)
        assert len(particles) == 256
        assert all(p.is_valid() for p in particles)

    @pytest.mark.slow
    def test_marginal_board_mean_exact(self):
        """Empirical board-average μ_0 should converge to 12/64 = 0.1875.

        With N draws, SE ≈ sqrt(p(1-p) / N) on a given cell, and the board
        average is MUCH tighter than any single cell (averaging 64 cells
        that sum to 12 on every draw — the board average is exactly 12/64
        on *every sample*, so its SE is literally zero).
        """
        rng = np.random.default_rng(RNG_SEED)
        particles = sample_particles(N=1_000, rng=rng)
        mu = np.zeros((BOARD_SIZE, BOARD_SIZE))
        for p in particles:
            for (r, c) in p.occupied_cells():
                mu[r, c] += 1
        mu /= len(particles)
        assert mu.mean() == pytest.approx(MAX_HITS / (BOARD_SIZE * BOARD_SIZE))

    @pytest.mark.slow
    def test_marginal_topology_matches_spec(self):
        """μ_0 topology per corrected spec table:

            corner              ≈ 0.13
            edge-near-corner    ≈ 0.17
            middle-of-edge      ≈ 0.21   ← max class
            near-corner interior≈ 0.17
            deep interior       ≈ 0.19

        Sampler SE per cell at N=4000 is ≈ 0.006; per-class mean SE ≈ 0.003.
        Using 3σ margins (≈ 0.01).
        """
        rng = np.random.default_rng(RNG_SEED)
        N = 4_000
        particles = sample_particles(N=N, rng=rng)
        mu = np.zeros((BOARD_SIZE, BOARD_SIZE))
        for p in particles:
            for (r, c) in p.occupied_cells():
                mu[r, c] += 1
        mu /= N

        corners = [mu[0, 0], mu[0, 7], mu[7, 0], mu[7, 7]]
        edge_near_corner = [mu[0, 1], mu[0, 6], mu[7, 1], mu[7, 6],
                            mu[1, 0], mu[6, 0], mu[1, 7], mu[6, 7]]
        middle_of_edge = [mu[0, 3], mu[0, 4], mu[7, 3], mu[7, 4],
                          mu[3, 0], mu[4, 0], mu[3, 7], mu[4, 7]]
        near_corner_interior = [mu[1, 1], mu[1, 6], mu[6, 1], mu[6, 6]]
        deep_interior = [mu[r, c] for r in (3, 4) for c in (3, 4)]

        assert 0.11 < np.mean(corners) < 0.15
        assert 0.15 < np.mean(edge_near_corner) < 0.19
        assert 0.19 < np.mean(middle_of_edge) < 0.23
        assert 0.15 < np.mean(near_corner_interior) < 0.19
        assert 0.17 < np.mean(deep_interior) < 0.21

        # Ordering relationships (per the corrected spec topology)
        assert np.mean(corners) < np.mean(edge_near_corner)
        assert np.mean(edge_near_corner) < np.mean(middle_of_edge)
        assert np.mean(deep_interior) < np.mean(middle_of_edge)  # the counter-intuitive one


# --------------------------------------------------------------------------
# ParticleFilter construction
# --------------------------------------------------------------------------

class TestParticleFilterInit:
    def test_initial_weights_uniform(self):
        rng = np.random.default_rng(RNG_SEED)
        pf = ParticleFilter(N=128, rng=rng)
        assert pf.N == 128
        assert pf.weights.shape == (128,)
        assert np.allclose(pf.weights, 1.0 / 128)
        assert pf.ess() == pytest.approx(128.0)

    def test_particles_valid(self):
        rng = np.random.default_rng(RNG_SEED)
        pf = ParticleFilter(N=64, rng=rng)
        assert all(p.is_valid() for p in pf.particles)

    def test_cell_marginal_grid_shape(self):
        rng = np.random.default_rng(RNG_SEED)
        pf = ParticleFilter(N=64, rng=rng)
        mu = pf.cell_marginal_grid()
        assert mu.shape == (BOARD_SIZE, BOARD_SIZE)
        assert mu.sum() == pytest.approx(MAX_HITS, abs=1e-9)
        assert 0.0 <= mu.min()
        assert mu.max() <= 1.0


# --------------------------------------------------------------------------
# Shot reweighting — noiseless delta likelihood
# --------------------------------------------------------------------------

class TestShotReweight:
    def _fixed_cfg(self) -> Configuration:
        return Configuration(ships=(
            Ship(4, 0, 0, Orientation.HORIZONTAL),
            Ship(3, 2, 0, Orientation.HORIZONTAL),
            Ship(3, 4, 4, Orientation.HORIZONTAL),
            Ship(2, 7, 6, Orientation.HORIZONTAL),
        ))

    def test_hit_zeros_inconsistent_particles(self):
        """Observing HIT at cell c gives zero weight to particles with X_c=0."""
        rng = np.random.default_rng(RNG_SEED)
        pf = ParticleFilter(N=256, rng=rng)
        # Pick a cell that's a hit in some particles, miss in others — (0,0).
        survivor_before = sum(1 for p in pf.particles if p.X((0, 0)) == 1)
        pf.reweight_shot(cell=(0, 0), observed=1)
        nonzero = np.count_nonzero(pf.weights)
        assert nonzero == survivor_before
        assert pf.weights.sum() == pytest.approx(1.0, abs=1e-9)

    def test_miss_zeros_hitting_particles(self):
        rng = np.random.default_rng(RNG_SEED)
        pf = ParticleFilter(N=256, rng=rng)
        survivor_before = sum(1 for p in pf.particles if p.X((0, 0)) == 0)
        pf.reweight_shot(cell=(0, 0), observed=0)
        nonzero = np.count_nonzero(pf.weights)
        assert nonzero == survivor_before
        assert pf.weights.sum() == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------
# Ask reweighting — BSC(ε), soft
# --------------------------------------------------------------------------

class TestAskReweight:
    def test_ask_keeps_all_weights_positive(self):
        rng = np.random.default_rng(RNG_SEED)
        pf = ParticleFilter(N=128, rng=rng)
        pf.reweight_ask(question=question_by_id("row:0"), observed=1, eps=0.10)
        assert np.all(pf.weights > 0)
        assert pf.weights.sum() == pytest.approx(1.0, abs=1e-9)

    def test_ask_agreement_favored(self):
        """Particles whose Q(s) matches observation should weigh more on average."""
        rng = np.random.default_rng(RNG_SEED)
        pf = ParticleFilter(N=512, rng=rng)
        q = question_by_id("row:0")
        pf.reweight_ask(question=q, observed=1, eps=0.05)
        w_match = sum(w for w, p in zip(pf.weights, pf.particles) if evaluate(q, p) == 1)
        w_mismatch = sum(w for w, p in zip(pf.weights, pf.particles) if evaluate(q, p) == 0)
        assert w_match > w_mismatch


# --------------------------------------------------------------------------
# ESS & resampling
# --------------------------------------------------------------------------

class TestESSResample:
    def test_ess_uniform_is_N(self):
        rng = np.random.default_rng(RNG_SEED)
        pf = ParticleFilter(N=256, rng=rng)
        assert pf.ess() == pytest.approx(256.0)

    def test_ess_drops_after_shot(self):
        rng = np.random.default_rng(RNG_SEED)
        pf = ParticleFilter(N=256, rng=rng)
        pf.reweight_shot(cell=(0, 0), observed=1)
        assert pf.ess() < 256.0

    def test_resample_if_low_ess_restores_uniform(self):
        rng = np.random.default_rng(RNG_SEED)
        pf = ParticleFilter(N=256, rng=rng)
        pf.reweight_shot(cell=(0, 0), observed=1)
        # Low threshold forces resample.
        resampled = pf.resample_if_low_ess(threshold=1.0)
        assert resampled is True
        assert np.allclose(pf.weights, 1.0 / 256)
        assert pf.ess() == pytest.approx(256.0)
        assert all(p.is_valid() for p in pf.particles)

    def test_resample_skipped_when_ess_high(self):
        rng = np.random.default_rng(RNG_SEED)
        pf = ParticleFilter(N=256, rng=rng)
        resampled = pf.resample_if_low_ess(threshold=0.5)
        assert resampled is False


# --------------------------------------------------------------------------
# Thompson sampling — posterior-conditioned shot selection
# --------------------------------------------------------------------------

class TestThompsonSample:
    def test_thompson_sample_returns_unshot_cell(self):
        rng = np.random.default_rng(RNG_SEED)
        pf = ParticleFilter(N=64, rng=rng)
        shot_set = {(0, 0), (1, 1)}
        cell = pf.thompson_sample(rng=rng, shots_fired=shot_set)
        assert cell not in shot_set
        r, c = cell
        assert 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

    def test_thompson_sample_prefers_unshot_ship_cells(self):
        """When a sampled particle has unshot ship cells, TS must pick one of them."""
        rng = np.random.default_rng(RNG_SEED)
        pf = ParticleFilter(N=64, rng=rng)
        # shots_fired leaves every ship cell available on every particle.
        cell = pf.thompson_sample(rng=rng, shots_fired=frozenset())
        # The drawn cell must be a ship cell on at least one particle (since TS
        # samples a particle and picks its ship cell).
        assert any(p.X(cell) == 1 for p in pf.particles)
