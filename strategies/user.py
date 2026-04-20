"""Interactive user "strategy" — action supplied externally each turn.

The HTTP bridge uses this to turn the step-based runner into a human-driven
game: the UI collects a click (shot) or question id (ask), calls
``set_next_action(action)``, then advances ``GameSession.step()``.  The
strategy's ``choose_action`` returns the pre-supplied action; the shared
posterior is updated via ``observe`` like any other strategy, so the UI can
show a live μ-heatmap.

Contract: exactly one ``set_next_action(...)`` call must precede each
``choose_action`` call.  Calling ``choose_action`` without a pending action
raises — this surfaces HTTP misuse rather than silently picking a default.
"""
from __future__ import annotations

from typing import FrozenSet

import numpy as np

from engine.board import Cell
from engine.exact import ExactPosterior
from engine.questions import question_by_id
from strategies.base import Action, AskAction, ShotAction


class UserStrategy:
    """Posterior-backed "strategy" whose action is supplied externally."""

    # Canonical (R+, M, C) = (2, 1, 1) → shoot iff μ ≥ 2/3.  The user's
    # "Why?" panel surfaces this threshold so the human can compare their
    # pick against the cost-aware policy.
    _SHOOT_THRESHOLD = 2.0 / 3.0

    def __init__(self, eps: float, rng: np.random.Generator):
        self.eps = eps
        self.rng = rng
        self.filter = ExactPosterior()
        self.last_decision_value: float = 0.0
        self.last_rationale: dict = {
            "chosen": None, "shot": None, "ask": None,
        }
        self._pending: Action | None = None

    def set_next_action(self, action: Action) -> None:
        """Queue the action to return on the next ``choose_action`` call."""
        if not isinstance(action, (ShotAction, AskAction)):
            raise TypeError(f"expected ShotAction or AskAction, got {type(action)!r}")
        self._pending = action

    def choose_action(self, shots_fired: FrozenSet[Cell], turn: int) -> Action:
        if self._pending is None:
            raise RuntimeError(
                "UserStrategy.choose_action called without a pending action; "
                "the caller must invoke set_next_action(...) first."
            )
        action = self._pending
        self._pending = None
        # Decision value: μ(cell) for shots, 0.0 for asks (no information metric
        # computed — the human's motivation is opaque to the posterior).
        if isinstance(action, ShotAction):
            mu = float(self.filter.cell_marginal(action.cell))
            self.last_decision_value = mu
            # Shot EV at canonical (R+, M, C) = (2, 1, 1) → (R++M)μ − (M+C) = 3μ−2.
            self.last_rationale = {
                "chosen": "shot",
                "shot": {
                    "cell": action.cell,
                    "mu": mu,
                    "ev": 3.0 * mu - 2.0,
                    "threshold": self._SHOOT_THRESHOLD,
                },
                "ask": None,
            }
        else:
            self.last_decision_value = 0.0
            self.last_rationale = {
                "chosen": "ask",
                "shot": None,
                "ask": {
                    "question_id": action.question_id,
                    "metric": "user",
                    "score": 0.0,
                },
            }
        return action

    def observe(self, action: Action, observed: int) -> None:
        if isinstance(action, ShotAction):
            self.filter.reweight_shot(cell=action.cell, observed=observed)
        else:
            self.filter.reweight_ask(
                question=question_by_id(action.question_id),
                observed=observed, eps=self.eps,
            )
