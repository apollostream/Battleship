"""Tests for engine.metrics — binary entropy, EIG, ELLR.

All quantities in NATS (natural log).

Key formulas:
    H(p)        = -p log p - (1-p) log(1-p)
    I(q; s|O)   = H(p̄) - E_π[H(p_s)]                                   (EIG)
    where p_s = (1-ε) if Q(s)=1 else ε   (BSC forward likelihood)
    and   p̄   = (1-ε)·μ_q + ε·(1-μ_q),   μ_q = Σ_i w_i · 1[Q(s_i)=1]

    E[log LR](q) = Σ_s w_s · KL(p_s || p̄_{-s})                          (ELLR)
    where p̄_{-s} excludes s from the mixture.

    EIG ≤ ELLR always.

Adaptive ask-vs-shoot uses:
    V_ask   = max_q I(q; s|O)   (in nats)
    V_shoot = max_c H(μ(c))     (binary entropy of the unshot cell marginal)
    Shoot iff V_shoot ≥ V_ask.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from engine.metrics import (
    binary_entropy,
    eig_of_all_asks,
    eig_of_ask,
    ellr_of_ask,
    info_gain_ask_bsc,
    info_gain_shot,
    shoot_information_value,
)


# --------------------------------------------------------------------------
# eig_of_all_asks — vectorised block matmul over a stack of questions
# --------------------------------------------------------------------------

class TestEIGOfAllAsks:
    def test_matches_per_question_dense(self):
        """Vectorised path agrees with the per-question scalar fn to ~1e-12."""
        rng = np.random.default_rng(0)
        n, q = 200, 17
        weights = rng.dirichlet(np.ones(n))
        A = rng.integers(0, 2, size=(n, q)).astype(bool)
        eps = 0.10
        v = eig_of_all_asks(answers_matrix=A, weights=weights, eps=eps)
        ref = np.array([
            eig_of_ask(answers=A[:, j], weights=weights, eps=eps) for j in range(q)
        ])
        np.testing.assert_allclose(v, ref, atol=1e-12)

    def test_matches_per_question_sparse(self):
        """Sparsified active path also agrees — exercises the masked branch."""
        rng = np.random.default_rng(1)
        n, q = 500, 13
        # Only 5% of weights are non-zero — forces the active-subset code path.
        w_full = np.zeros(n)
        idx = rng.choice(n, n // 20, replace=False)
        w_full[idx] = rng.dirichlet(np.ones(len(idx)))
        A = rng.integers(0, 2, size=(n, q)).astype(bool)
        eps = 0.05
        v = eig_of_all_asks(answers_matrix=A, weights=w_full, eps=eps)
        ref = np.array([
            eig_of_ask(answers=A[:, j], weights=w_full, eps=eps) for j in range(q)
        ])
        np.testing.assert_allclose(v, ref, atol=1e-12)

    def test_eps_validation(self):
        with pytest.raises(ValueError, match="epsilon"):
            eig_of_all_asks(
                answers_matrix=np.zeros((3, 2), dtype=bool),
                weights=np.array([0.5, 0.5, 0.0]),
                eps=1.5,
            )

    def test_eps_zero_and_one(self):
        """At ε=0 EIG = H(μ_q); at ε=1 the channel is a deterministic flip → also H(μ_q)."""
        rng = np.random.default_rng(2)
        n, q = 100, 7
        weights = rng.dirichlet(np.ones(n))
        A = rng.integers(0, 2, size=(n, q)).astype(bool)
        v0 = eig_of_all_asks(answers_matrix=A, weights=weights, eps=0.0)
        v1 = eig_of_all_asks(answers_matrix=A, weights=weights, eps=1.0)
        ref0 = np.array([eig_of_ask(answers=A[:, j], weights=weights, eps=0.0) for j in range(q)])
        ref1 = np.array([eig_of_ask(answers=A[:, j], weights=weights, eps=1.0) for j in range(q)])
        np.testing.assert_allclose(v0, ref0, atol=1e-12)
        np.testing.assert_allclose(v1, ref1, atol=1e-12)


# --------------------------------------------------------------------------
# binary_entropy — in nats
# --------------------------------------------------------------------------

class TestBinaryEntropy:
    def test_boundary_zero(self):
        assert binary_entropy(0.0) == 0.0
        assert binary_entropy(1.0) == 0.0

    def test_midpoint_is_log_2(self):
        assert binary_entropy(0.5) == pytest.approx(math.log(2))

    def test_symmetry(self):
        for p in (0.1, 0.25, 0.4, 0.73):
            assert binary_entropy(p) == pytest.approx(binary_entropy(1 - p))

    def test_monotone_on_halves(self):
        # Increasing on [0, 0.5], decreasing on [0.5, 1].
        xs = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
        hs = [binary_entropy(x) for x in xs]
        up = hs[:4]
        down = hs[3:]
        assert up == sorted(up)
        assert down == sorted(down, reverse=True)

    def test_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            binary_entropy(-0.01)
        with pytest.raises(ValueError):
            binary_entropy(1.0001)


# --------------------------------------------------------------------------
# eig_of_ask — expected information gain of a binary question
# --------------------------------------------------------------------------

class TestEIG:
    def test_zero_info_when_all_particles_agree(self):
        """If every particle has the same Q(s), asking gives no information."""
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        answers = np.array([1, 1, 1, 1])
        assert eig_of_ask(answers=answers, weights=weights, eps=0.10) == pytest.approx(0.0, abs=1e-12)

    def test_half_half_noiseless_is_log_2(self):
        """μ_q = 0.5, ε = 0 ⇒ EIG = H(0.5) - 0 = log 2."""
        weights = np.array([0.5, 0.5])
        answers = np.array([1, 0])
        assert eig_of_ask(answers=answers, weights=weights, eps=0.0) == pytest.approx(math.log(2))

    def test_half_half_noisy_is_less_than_log_2(self):
        weights = np.array([0.5, 0.5])
        answers = np.array([1, 0])
        eig_noisy = eig_of_ask(answers=answers, weights=weights, eps=0.10)
        assert 0.0 < eig_noisy < math.log(2)

    def test_eig_at_epsilon_half_is_zero(self):
        """ε = 0.5 ⇒ BSC is uninformative ⇒ EIG = 0 regardless of μ_q."""
        weights = np.array([0.3, 0.3, 0.4])
        answers = np.array([1, 0, 1])
        assert eig_of_ask(answers=answers, weights=weights, eps=0.5) == pytest.approx(0.0, abs=1e-12)

    def test_eig_non_negative(self):
        rng = np.random.default_rng(11)
        for _ in range(20):
            N = rng.integers(3, 20)
            w = rng.random(N)
            w /= w.sum()
            a = rng.integers(0, 2, size=N)
            eps = float(rng.uniform(0.01, 0.45))
            assert eig_of_ask(answers=a, weights=w, eps=eps) >= -1e-12


# --------------------------------------------------------------------------
# ellr_of_ask — expected log-likelihood ratio (weighted KL against mixture)
# --------------------------------------------------------------------------

class TestELLR:
    def test_zero_info_when_all_particles_agree(self):
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        answers = np.array([0, 0, 0, 0])
        assert ellr_of_ask(answers=answers, weights=weights, eps=0.10) == pytest.approx(0.0, abs=1e-12)

    def test_ellr_non_negative(self):
        rng = np.random.default_rng(23)
        for _ in range(20):
            N = rng.integers(3, 20)
            w = rng.random(N)
            w /= w.sum()
            a = rng.integers(0, 2, size=N)
            eps = float(rng.uniform(0.01, 0.45))
            assert ellr_of_ask(answers=a, weights=w, eps=eps) >= -1e-12

    def test_ellr_at_epsilon_half_is_zero(self):
        weights = np.array([0.4, 0.6])
        answers = np.array([1, 0])
        assert ellr_of_ask(answers=answers, weights=weights, eps=0.5) == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# Key invariant: EIG ≤ ELLR (from transcript.md / CLAUDE.md)
# --------------------------------------------------------------------------

class TestEIGVsELLR:
    def test_eig_le_ellr_random_cases(self):
        rng = np.random.default_rng(47)
        for _ in range(200):
            N = int(rng.integers(3, 40))
            w = rng.random(N)
            w /= w.sum()
            a = rng.integers(0, 2, size=N)
            eps = float(rng.uniform(0.01, 0.45))
            eig = eig_of_ask(answers=a, weights=w, eps=eps)
            ellr = ellr_of_ask(answers=a, weights=w, eps=eps)
            assert eig <= ellr + 1e-10, f"EIG={eig} > ELLR={ellr} at w={w}, a={a}, eps={eps}"


# --------------------------------------------------------------------------
# shoot_information_value — max_c H(μ(c)) over unshot cells
# --------------------------------------------------------------------------

class TestShootInformationValue:
    def test_empty_mu_grid_empty_shots_returns_zero_when_all_hit_or_miss(self):
        """If every unshot cell is deterministic, shoot value is zero."""
        mu = np.zeros((8, 8))   # every cell is μ=0 ⇒ H(0)=0
        assert shoot_information_value(mu=mu, shots_fired=frozenset()) == 0.0

    def test_excludes_shot_cells(self):
        mu = np.zeros((8, 8))
        mu[0, 0] = 0.5   # H(0.5)=log 2
        mu[1, 1] = 0.3   # H(0.3)=something smaller than log 2
        v = shoot_information_value(mu=mu, shots_fired=frozenset({(0, 0)}))
        # (0,0) is excluded, so the best unshot cell is (1,1) with μ=0.3
        assert v == pytest.approx(binary_entropy(0.3))

    def test_picks_max_entropy_cell(self):
        mu = np.zeros((8, 8))
        mu[3, 3] = 0.9
        mu[3, 4] = 0.5
        mu[3, 5] = 0.1
        v = shoot_information_value(mu=mu, shots_fired=frozenset())
        assert v == pytest.approx(math.log(2))  # H(0.5) = log 2

    def test_returns_zero_when_no_cells_available(self):
        mu = np.zeros((8, 8))
        mu[:] = 0.5
        all_cells = frozenset((r, c) for r in range(8) for c in range(8))
        assert shoot_information_value(mu=mu, shots_fired=all_cells) == 0.0


# --------------------------------------------------------------------------
# info_gain_shot — realized KL of posterior update for noiseless shot
# --------------------------------------------------------------------------

class TestInfoGainShot:
    def test_expected_miss(self):
        assert math.isclose(
            info_gain_shot(mu_c=0.1, observed=0),
            -math.log(0.9), abs_tol=1e-12,
        )

    def test_surprising_miss(self):
        assert math.isclose(
            info_gain_shot(mu_c=0.9, observed=0),
            -math.log(0.1), abs_tol=1e-12,
        )

    def test_hit_symmetric(self):
        assert math.isclose(
            info_gain_shot(mu_c=0.3, observed=1),
            -math.log(0.3), abs_tol=1e-12,
        )

    def test_entropy_identity(self):
        """E_y[info_gain_shot(μ, y)] = H(μ) — the prior's self-entropy."""
        mu = 0.4
        expected_h = -(mu * math.log(mu) + (1 - mu) * math.log(1 - mu))
        observed_ev = (
            mu * info_gain_shot(mu_c=mu, observed=1)
            + (1 - mu) * info_gain_shot(mu_c=mu, observed=0)
        )
        assert math.isclose(observed_ev, expected_h, abs_tol=1e-12)

    def test_invalid_observed_rejected(self):
        with pytest.raises(ValueError):
            info_gain_shot(mu_c=0.5, observed=2)

    def test_invalid_mu_rejected(self):
        with pytest.raises(ValueError):
            info_gain_shot(mu_c=-0.01, observed=1)


# --------------------------------------------------------------------------
# info_gain_ask_bsc — realized KL of posterior update for BSC(ε) ask
# --------------------------------------------------------------------------

class TestInfoGainAskBSC:
    def test_noiseless_limit(self):
        """ε=0 with p_hat=0.5, y=1 ⇒ KL = -log p_hat = log 2."""
        assert math.isclose(
            info_gain_ask_bsc(p_hat=0.5, eps=0.0, observed=1),
            math.log(2), abs_tol=1e-12,
        )

    def test_pure_noise_limit(self):
        """ε=0.5 ⇒ BSC is uninformative ⇒ realized KL = 0 for any p_hat, y."""
        assert math.isclose(
            info_gain_ask_bsc(p_hat=0.3, eps=0.5, observed=1),
            0.0, abs_tol=1e-12,
        )

    def test_expected_equals_eig(self):
        """E_y[info_gain_ask_bsc(p_hat, ε, y)] = H(p̄) - H(ε) = EIG."""
        p_hat, eps = 0.3, 0.1
        p_bar_1 = (1 - eps) * p_hat + eps * (1 - p_hat)
        H_bar = -(
            p_bar_1 * math.log(p_bar_1)
            + (1 - p_bar_1) * math.log(1 - p_bar_1)
        )
        H_eps = -(eps * math.log(eps) + (1 - eps) * math.log(1 - eps))
        expected_eig = H_bar - H_eps
        ev = (
            p_bar_1 * info_gain_ask_bsc(p_hat=p_hat, eps=eps, observed=1)
            + (1 - p_bar_1) * info_gain_ask_bsc(p_hat=p_hat, eps=eps, observed=0)
        )
        assert math.isclose(ev, expected_eig, abs_tol=1e-12)

    def test_invalid_observed_rejected(self):
        with pytest.raises(ValueError):
            info_gain_ask_bsc(p_hat=0.5, eps=0.1, observed=2)

    def test_invalid_eps_rejected(self):
        with pytest.raises(ValueError):
            info_gain_ask_bsc(p_hat=0.5, eps=1.5, observed=1)
