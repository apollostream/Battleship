"""engine.metrics — information-theoretic scoring of binary questions.

All quantities in NATS (natural log).

Given a particle ensemble with weights w_i summing to 1, and a question q
whose true answer on particle i is a_i ∈ {0, 1}, and a BSC(ε) channel:

  μ_q    = Σ_i w_i · a_i                          (marginal probability of answer=1)
  p_s    = (1 - ε) if a_s == 1 else ε             (prob observer sees "1" given truth s)
  p̄      = (1 - ε) μ_q + ε (1 - μ_q)              (mixture forward-predictive for "1")

EIG (expected information gain):
  I(q; s | O) = H(p̄) - Σ_s w_s · H(p_s)
  Under BSC, p_s is either (1-ε) or ε, so H(p_s) = H(ε); the expectation collapses:
  I(q; s) = H(p̄) - H(ε)
  (Valid only because p_s has the same distribution on both truth values: BSC is symmetric.)

ELLR (expected log-likelihood ratio):
  Σ_s w_s · KL(p_s || p̄_{-s}),  p̄_{-s} = ( Σ_{j≠s} w_j p_j ) / (1 - w_s)

Under BSC, p_s places mass (1-ε) on a_s and ε on 1-a_s.  p̄_{-s} places mass
  q_+^{-s} on 1 and 1 - q_+^{-s} on 0, where q_+^{-s} = (μ_q · 1[a_s = 0] + (μ_q - w_s)· 1[a_s=1])
                                                      / (1 - w_s)  * (1-ε) + ε·(complement).
Full derivation in transcript.md; implementation is direct.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from engine.board import BOARD_SIZE

# --------------------------------------------------------------------------
# Binary entropy — nats
# --------------------------------------------------------------------------

_FP_SLACK = 1e-9


def binary_entropy(p: float) -> float:
    if not -_FP_SLACK <= p <= 1.0 + _FP_SLACK:
        raise ValueError(f"probability must be in [0, 1], got {p}")
    p = min(1.0, max(0.0, p))
    if p == 0.0 or p == 1.0:
        return 0.0
    return -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)


def _h_vec(p: np.ndarray) -> np.ndarray:
    """Vectorised binary entropy.  p is an array of probabilities in [0, 1]."""
    out = np.zeros_like(p, dtype=float)
    safe = (p > 0.0) & (p < 1.0)
    ps = p[safe]
    out[safe] = -ps * np.log(ps) - (1.0 - ps) * np.log(1.0 - ps)
    return out


# --------------------------------------------------------------------------
# EIG — expected information gain under BSC(ε)
# --------------------------------------------------------------------------

def eig_of_ask(*, answers: np.ndarray, weights: np.ndarray, eps: float) -> float:
    if not 0.0 <= eps <= 1.0:
        raise ValueError(f"epsilon must be in [0, 1], got {eps}")
    # Sparsify: zero-weight entries don't contribute to the dot product but cost
    # memory bandwidth on fancy indices; skip the copy when no pruning is needed.
    mask = weights > 0.0
    if not bool(mask.all()):
        answers = answers[mask]
        weights = weights[mask]
    a = answers.astype(float)
    w = weights.astype(float)
    mu_q = float(np.dot(w, a))
    p_bar = (1.0 - eps) * mu_q + eps * (1.0 - mu_q)
    # Under BSC, p_s ∈ {ε, 1-ε} with H(ε) = H(1-ε), so E[H(p_s)] = H(ε).
    return binary_entropy(p_bar) - binary_entropy(eps)


def eig_of_all_asks(
    *, answers_matrix: np.ndarray, weights: np.ndarray, eps: float,
) -> np.ndarray:
    """Vectorised EIG over a stacked (|S|, |Q|) answer matrix.

    Returns a (|Q|,) float64 array of EIG values, one per column of
    ``answers_matrix``.  Replaces |Q| separate ``eig_of_ask`` calls with a
    single matmul; for |S|≈5M and |Q|=87, ~14× faster end-to-end (and dominated
    by the matmul build cost when active configs ≪ |S|).

    Sparsification policy: when most weights are zero (mid-game and later),
    cast the active rows to float64 and matmul those — cuts both memory and
    compute proportionally to the active fraction.  Cold (uniform) start falls
    through to the bool×float64 path which numpy handles in a single pass.
    """
    if not 0.0 <= eps <= 1.0:
        raise ValueError(f"epsilon must be in [0, 1], got {eps}")
    mask = weights > 0.0
    if bool(mask.all()):
        # Cold path: bool×float64 — numpy issues one C-level reduction per
        # column (~4 s for 5M×87).  No copy.
        mu_q = (weights @ answers_matrix).astype(np.float64)
    else:
        # Active path: copy active rows into a contiguous float64 matrix and
        # let BLAS handle it.  At ~10% active, the build+matmul together is
        # under 1 s; sub-ms once we're below ~50 K active configs.
        w = weights[mask]
        A = answers_matrix[mask].astype(np.float64)
        mu_q = w @ A
    p_bar = np.clip(mu_q, 0.0, 1.0)
    p_bar = (1.0 - eps) * p_bar + eps * (1.0 - p_bar)
    return _h_vec(p_bar) - binary_entropy(eps)


# --------------------------------------------------------------------------
# ELLR — expected log-likelihood ratio against leave-one-out mixture
# --------------------------------------------------------------------------

def ellr_of_ask(*, answers: np.ndarray, weights: np.ndarray, eps: float) -> float:
    """Vectorised expected log-likelihood-ratio against the leave-one-out mixture.

    For N up to ~|S| ≈ 5M we must avoid any Python-level per-config loop; the
    per-s arithmetic is expressed entirely as numpy vector ops.
    """
    if not 0.0 <= eps <= 1.0:
        raise ValueError(f"epsilon must be in [0, 1], got {eps}")
    # Sparsify: the log computations below are expensive (~200 ms per call on
    # 5 M entries); zero-weight configs contribute nothing to the final sum,
    # so prune them before expanding the per-s arithmetic.
    mask = weights > 0.0
    if not bool(mask.all()):
        answers = answers[mask]
        weights = weights[mask]
    a = answers.astype(np.float64)
    w = weights.astype(np.float64)
    mu_q = float(np.dot(w, a))

    # Leave-one-out answer-marginal μ_q^{-s} — invalid where w_s == 1 (full mass
    # on a single config); we mask those positions out of the sum below.
    denom = 1.0 - w
    valid = (w > 0.0) & (denom > 0.0)
    mu_q_los = np.zeros_like(w)
    np.divide(mu_q - w * a, denom, out=mu_q_los, where=valid)

    # p̄^{-s} for answer=1 under the BSC channel.  Clipped into (0, 1) so the
    # log below is defined even at numerical corners.
    p_bar_los = (1.0 - eps) * mu_q_los + eps * (1.0 - mu_q_los)
    q1 = np.clip(p_bar_los, 1e-300, 1.0 - 0.0)
    q0 = np.clip(1.0 - p_bar_los, 1e-300, 1.0 - 0.0)

    # p_s: mass (1-ε) on a_s and ε on 1-a_s.  a is in {0, 1}, so:
    p_s_1 = np.where(a > 0.5, 1.0 - eps, eps)
    p_s_0 = 1.0 - p_s_1

    kl = np.zeros_like(w)
    if 0.0 < eps < 1.0:   # both p_s_1 and p_s_0 strictly positive ⇒ both terms contribute
        kl = p_s_1 * np.log(p_s_1 / q1) + p_s_0 * np.log(p_s_0 / q0)
    elif eps == 0.0:
        # p_s has a 0 mass on the non-matching outcome; skip that log term to avoid 0·log(0/·).
        kl = np.where(
            a > 0.5,
            1.0 * np.log(1.0 / q1),
            1.0 * np.log(1.0 / q0),
        )
    elif eps == 1.0:
        kl = np.where(
            a > 0.5,
            1.0 * np.log(1.0 / q0),
            1.0 * np.log(1.0 / q1),
        )

    return float(np.sum(w * kl, where=valid))


# --------------------------------------------------------------------------
# Realized (per-observation) information gain — for post-hoc display.
#
# Whereas eig_of_ask / ellr_of_ask rank *candidate* questions before any
# answer is known, these return the KL divergence of the posterior update
# after the observation lands.  Used to surface "how much did we actually
# learn from that turn?" in the UI.  In expectation over the observation,
# info_gain_shot → H(μ) and info_gain_ask_bsc → EIG(q).
# --------------------------------------------------------------------------

def info_gain_shot(*, mu_c: float, observed: int) -> float:
    """KL(π_new ‖ π_old) in nats for a noiseless shot at cell c.

    μ_c is the prior probability that c contained a ship.  observed ∈ {0,1}
    with 1 = hit, 0 = miss.  Because shots are noiseless, the posterior
    update is a simple likelihood indicator, and the KL collapses to the
    Shannon self-information: -log P(observed).
    """
    if not -_FP_SLACK <= mu_c <= 1.0 + _FP_SLACK:
        raise ValueError(f"mu_c must be in [0, 1], got {mu_c}")
    if observed not in (0, 1):
        raise ValueError(f"observed must be 0 or 1, got {observed}")
    mu_c = min(1.0, max(0.0, mu_c))
    p = mu_c if observed == 1 else (1.0 - mu_c)
    if p <= 0.0:
        return math.inf
    return -math.log(p)


def info_gain_ask_bsc(*, p_hat: float, eps: float, observed: int) -> float:
    """KL(π_new ‖ π_old) in nats for a BSC(ε) ask with answer marginal p_hat.

    p_hat = P(a_s = 1 | π_old).  Uses the closed form
        KL = E_{π_new}[log p(y|s)] − log P̄(y)
    where π_new re-weights truth-mass by the BSC likelihood of ``observed``.
    """
    if not -_FP_SLACK <= p_hat <= 1.0 + _FP_SLACK:
        raise ValueError(f"p_hat must be in [0, 1], got {p_hat}")
    if not 0.0 <= eps <= 1.0:
        raise ValueError(f"epsilon must be in [0, 1], got {eps}")
    if observed not in (0, 1):
        raise ValueError(f"observed must be 0 or 1, got {observed}")
    p_hat = min(1.0, max(0.0, p_hat))

    p_y = p_hat if observed == 1 else (1.0 - p_hat)
    p_bar_y = (1.0 - eps) * p_y + eps * (1.0 - p_y)
    if p_bar_y <= 0.0:
        return math.inf

    new_match = (1.0 - eps) * p_y / p_bar_y
    new_flip = eps * (1.0 - p_y) / p_bar_y

    e_log = 0.0
    if new_match > 0.0:
        if eps >= 1.0:
            return math.inf
        e_log += new_match * math.log(1.0 - eps)
    if new_flip > 0.0:
        if eps <= 0.0:
            return math.inf
        e_log += new_flip * math.log(eps)

    return e_log - math.log(p_bar_y)


# --------------------------------------------------------------------------
# Shot information value — adaptive ask-vs-shoot comparator
# --------------------------------------------------------------------------

def shoot_information_value(*, mu: np.ndarray, shots_fired: Iterable) -> float:
    """Max binary entropy over unshot cells: max_{c unshot} H(μ(c)).

    This is the noiseless-shot analogue of EIG: since a shot observation has
    no BSC channel, its self-information equals H(μ(c)).
    """
    shot_set = frozenset(shots_fired)
    best = 0.0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if (r, c) in shot_set:
                continue
            h = binary_entropy(float(mu[r, c]))
            if h > best:
                best = h
    return best
