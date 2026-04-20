"""Shared MBayes base — adaptive ask-vs-shoot comparator.

Both EIG and ELLR strategies have the same shape:
    * pick best question by their own metric;
    * compare metric(best_q) against max_c H(μ(c));
    * shoot argmax μ(c) if shooting wins, else ask best_q.

Only the question-scoring function differs.

Backed by engine.exact.ExactPosterior: answer vectors are (|S|,) bool arrays
permanent for the lifetime of the process, so there is no per-turn invalidation
logic — caching lives inside the posterior itself.
"""
from __future__ import annotations

from typing import Callable, FrozenSet

import numpy as np

from engine.board import BOARD_SIZE, Cell
from engine.exact import ExactPosterior
from engine.metrics import (
    eig_of_ask,
    ellr_of_ask,
    shoot_information_value,
)
from engine.questions import QUESTION_CATALOGUE, Question, question_by_id
from strategies.base import Action, AskAction, ShotAction


MetricFn = Callable[..., float]


class MBayesStrategy:
    """Generic adaptive MBayes policy parameterised by a question-metric."""

    metric_fn: MetricFn   # subclasses set this

    def __init__(self, eps: float, rng: np.random.Generator):
        self.eps = eps
        self.rng = rng
        self.filter = ExactPosterior()

    # --- decision logic -------------------------------------------------

    def _score_question(self, q: Question) -> float:
        a = self.filter.answers_for(q)
        return self.metric_fn(answers=a, weights=self.filter.weights, eps=self.eps)

    def _best_question(self) -> tuple[Question, float]:
        best_q: Question | None = None
        best_v = -float("inf")
        for q in QUESTION_CATALOGUE:
            v = self._score_question(q)
            if v > best_v:
                best_v, best_q = v, q
        assert best_q is not None
        return best_q, best_v

    def _best_shot_cell(self, mu: np.ndarray, shots_fired: FrozenSet[Cell]) -> Cell:
        best_cell: Cell | None = None
        best_mu = -1.0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if (r, c) in shots_fired:
                    continue
                if mu[r, c] > best_mu:
                    best_mu, best_cell = float(mu[r, c]), (r, c)
        assert best_cell is not None
        return best_cell

    def choose_action(self, shots_fired: FrozenSet[Cell], turn: int) -> Action:
        mu = self.filter.cell_marginal_grid()
        v_shoot = shoot_information_value(mu=mu, shots_fired=shots_fired)
        best_q, v_ask = self._best_question()
        if v_shoot >= v_ask:
            return ShotAction(cell=self._best_shot_cell(mu, shots_fired))
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
    metric_fn = staticmethod(eig_of_ask)


class ELLRMBayesStrategy(MBayesStrategy):
    metric_fn = staticmethod(ellr_of_ask)
