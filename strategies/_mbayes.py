"""Shared MBayes base — cost-aware ask-vs-shoot comparator.

Decision rule
-------------
Each shot has a fixed cost ``shot_cost`` and yields ``hit_reward`` on a hit,
``-miss_cost`` on a miss.  Each ask has zero cost and zero reward (information
only).  Expected net reward of shooting cell ``c`` is

    E[shot|μ(c)] = (R+ + M)·μ(c) − (M + C)         where R+ = hit_reward,
                                                         M = miss_cost,
                                                         C = shot_cost

so it is rational to shoot iff

    μ(c) ≥ (M + C) / (R+ + M)        ← parameter-free given (R+, M, C)

With defaults (R+, M, C) = (2, 1, 1), the threshold is **2/3**.  When no
unshot cell beats the threshold, the strategy asks ``argmax_q metric(q)`` over
the question catalogue.

Backed by engine.exact.ExactPosterior; question scoring uses the vectorised
``eig_of_all_asks`` block-matmul path (EIG) or an EIG-shortlist + full-score
top-K (ELLR).  See engine/metrics.py for the math.
"""
from __future__ import annotations

from typing import FrozenSet

import numpy as np

from engine.board import BOARD_SIZE, Cell
from engine.exact import ExactPosterior
from engine.metrics import (
    eig_of_all_asks,
    ellr_of_ask,
)
from engine.questions import QUESTION_CATALOGUE, Question, question_by_id
from strategies.base import Action, AskAction, ShotAction


class MBayesStrategy:
    """Cost-aware MBayes policy.

    Subclasses pick their own question — see ``_best_question`` — but share the
    ask-vs-shoot comparator and the observe path.
    """

    def __init__(
        self,
        eps: float,
        rng: np.random.Generator,
        *,
        hit_reward: float = 2.0,
        miss_cost: float = 1.0,
        shot_cost: float = 1.0,
    ):
        self.eps = eps
        self.rng = rng
        self.filter = ExactPosterior()
        self.hit_reward = float(hit_reward)
        self.miss_cost = float(miss_cost)
        self.shot_cost = float(shot_cost)
        denom = self.hit_reward + self.miss_cost
        if denom <= 0.0:
            raise ValueError(
                f"hit_reward + miss_cost must be > 0, got R+={hit_reward}, M={miss_cost}"
            )
        self.shoot_threshold: float = (self.miss_cost + self.shot_cost) / denom
        self.last_decision_value: float = 0.0

    # --- decision logic -------------------------------------------------

    def _best_unshot_cell(
        self, mu: np.ndarray, shots_fired: FrozenSet[Cell],
    ) -> tuple[Cell, float]:
        best_cell: Cell | None = None
        best_mu = -1.0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if (r, c) in shots_fired:
                    continue
                if mu[r, c] > best_mu:
                    best_mu, best_cell = float(mu[r, c]), (r, c)
        assert best_cell is not None
        return best_cell, best_mu

    def _best_question(self) -> tuple[Question, float]:   # subclasses override
        raise NotImplementedError

    def choose_action(self, shots_fired: FrozenSet[Cell], turn: int) -> Action:
        mu = self.filter.cell_marginal_grid()
        best_cell, max_mu = self._best_unshot_cell(mu, shots_fired)
        if max_mu >= self.shoot_threshold:
            self.last_decision_value = max_mu     # μ at chosen shot cell
            return ShotAction(cell=best_cell)
        best_q, best_v = self._best_question()
        self.last_decision_value = best_v         # EIG/ELLR score of chosen ask
        return AskAction(question_id=best_q.id)

    def observe(self, action: Action, observed: int) -> None:
        if isinstance(action, ShotAction):
            self.filter.reweight_shot(cell=action.cell, observed=observed)
        else:
            self.filter.reweight_ask(
                question=question_by_id(action.question_id),
                observed=observed, eps=self.eps,
            )


class EIGMBayesStrategy(MBayesStrategy):
    """EIG question pick — vectorised over the full catalogue."""

    def _best_question(self) -> tuple[Question, float]:
        A = self.filter.build_answer_matrix(QUESTION_CATALOGUE)
        scores = eig_of_all_asks(
            answers_matrix=A, weights=self.filter.weights, eps=self.eps,
        )
        i = int(np.argmax(scores))
        return QUESTION_CATALOGUE[i], float(scores[i])


class _SampleBackedMBayes(MBayesStrategy):
    """MBayes variant that scores questions *and* cell marginals from one K-sample
    posterior draw per turn.

    Both the ask-vs-shoot comparator (μ(c)) and the question rank (p̂(q)) are
    estimated from the same K configurations, which is both cheaper and more
    self-consistent than pairing a sampled metric with an exact marginal:
    per-turn cost collapses to one sampler call plus O(K·(|Q|+64)) indexing,
    dwarfed by the previously-dominant bool×f64 matmul in ``cell_marginal_grid``.

    Quality: p̂_q and μ̂(c) are unbiased with variance O(1/K).  For K=200 the
    stderr is ~3.5%, well inside the 2/3 shoot threshold margin most of the
    time; sampling noise is the cost we pay for dropping the |S|-scaling.
    Subclasses tune K per metric sensitivity (ELLR is tail-heavy → larger K).
    """

    K_SAMPLES: int = 200

    def _sample_block(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw the K-sample snapshot used for this turn's decision.

        Returns ``(p_hat, mu_grid, idx)`` — question-answer marginals (|Q|,),
        cell-occupancy marginals (8, 8), and the config indices that backed
        them.  Subclasses access ``p_hat`` for question ranking; the base
        class uses ``mu_grid`` for the shoot/ask comparator.
        """
        A = self.filter.build_answer_matrix(QUESTION_CATALOGUE)
        idx = self.filter.sample_configs(K=self.K_SAMPLES, rng=self.rng)
        p_hat = A[idx].mean(axis=0).astype(np.float64)
        cells_K = self.filter._cells_matrix[idx]
        mu_grid = cells_K.mean(axis=0).astype(np.float64).reshape(BOARD_SIZE, BOARD_SIZE)
        return p_hat, mu_grid, idx

    def choose_action(self, shots_fired: FrozenSet[Cell], turn: int) -> Action:
        self._p_hat, mu, _idx = self._sample_block()
        best_cell, max_mu = self._best_unshot_cell(mu, shots_fired)
        if max_mu >= self.shoot_threshold:
            self.last_decision_value = max_mu
            return ShotAction(cell=best_cell)
        best_q, best_v = self._best_question()
        self.last_decision_value = best_v
        return AskAction(question_id=best_q.id)


class ApproxEIGMBayesStrategy(_SampleBackedMBayes):
    """EIG pick via K-sample BALD ranking: argmax p̂(1−p̂).

    Under BSC(ε), EIG = H(p̄) − H(ε) is monotone in H(p̂) which is monotone
    in the Bernoulli variance p̂(1−p̂) — peaks at p̂=0.5, decreases toward
    the endpoints.  So we rank by sample variance and skip the entropy call.
    See Houlsby et al. 2011 (arXiv:1112.5745) for the BALD formulation.
    ``last_decision_value`` reports that variance at the chosen question.
    """

    K_SAMPLES: int = 200

    def _best_question(self) -> tuple[Question, float]:
        p_hat = self._p_hat
        score = p_hat * (1.0 - p_hat)
        i = int(np.argmax(score))
        return QUESTION_CATALOGUE[i], float(score[i])


class ApproxELLRMBayesStrategy(_SampleBackedMBayes):
    """ELLR pick via closed-form sample-mean KL: E_s[KL(p_s ‖ p̄)] under BSC(ε).

    The exact ELLR uses leave-one-out reference p̄_{-s}; the K-sample
    approximation replaces it with the K-sample mean p̄ (the 1/K correction
    vanishes as K grows).  Under BSC this reduces to a closed form in p̂(q)
    and ε, avoiding any per-sample loop.
    """

    K_SAMPLES: int = 500

    def _best_question(self) -> tuple[Question, float]:
        p_hat = self._p_hat
        eps = self.eps
        p_bar_1 = np.clip((1.0 - eps) * p_hat + eps * (1.0 - p_hat), 1e-300, 1.0)
        p_bar_0 = np.clip(1.0 - p_bar_1, 1e-300, 1.0)
        if 0.0 < eps < 1.0:
            kl_a1 = (1 - eps) * np.log((1 - eps) / p_bar_1) + eps * np.log(eps / p_bar_0)
            kl_a0 = eps * np.log(eps / p_bar_1) + (1 - eps) * np.log((1 - eps) / p_bar_0)
        elif eps == 0.0:
            kl_a1 = np.log(1.0 / p_bar_1)
            kl_a0 = np.log(1.0 / p_bar_0)
        else:  # eps == 1.0
            kl_a1 = np.log(1.0 / p_bar_0)
            kl_a0 = np.log(1.0 / p_bar_1)
        score = p_hat * kl_a1 + (1.0 - p_hat) * kl_a0
        i = int(np.argmax(score))
        return QUESTION_CATALOGUE[i], float(score[i])


class ELLRMBayesStrategy(MBayesStrategy):
    """ELLR question pick — shortlist top-K by EIG, then full-score with ELLR.

    EIG and ELLR are tightly correlated (their scores differ only by the
    leave-one-out reference distribution; the ranking agrees on most boards).
    Shortlisting cuts ELLR's wall by ~|Q|/K with negligible quality loss.
    """

    SHORTLIST_K: int = 10

    def _best_question(self) -> tuple[Question, float]:
        A = self.filter.build_answer_matrix(QUESTION_CATALOGUE)
        eig_scores = eig_of_all_asks(
            answers_matrix=A, weights=self.filter.weights, eps=self.eps,
        )
        k = min(self.SHORTLIST_K, len(QUESTION_CATALOGUE))
        # argpartition gives the top-k indices in arbitrary order — fine, we
        # full-score them all and take argmax of ELLR.
        top_idx = np.argpartition(-eig_scores, k - 1)[:k]
        best_q: Question | None = None
        best_v = -float("inf")
        for i in top_idx:
            q = QUESTION_CATALOGUE[int(i)]
            a = self.filter.answers_for(q)
            v = ellr_of_ask(answers=a, weights=self.filter.weights, eps=self.eps)
            if v > best_v:
                best_v, best_q = v, q
        assert best_q is not None
        return best_q, best_v
