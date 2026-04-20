"""Thompson-sampling doctrine — always shoots; shot chosen by posterior draw."""
from __future__ import annotations

from typing import FrozenSet

import numpy as np

from engine.board import Cell
from engine.exact import ExactPosterior
from strategies.base import Action, AskAction, ShotAction


class ThompsonStrategy:
    def __init__(self, eps: float, rng: np.random.Generator):
        self.eps = eps
        self.rng = rng
        self.filter = ExactPosterior()
        self.last_decision_value: float = 0.0

    def choose_action(self, shots_fired: FrozenSet[Cell], turn: int) -> Action:
        cell = self.filter.thompson_sample(rng=self.rng, shots_fired=shots_fired)
        # μ at the sampled cell — Thompson's implicit "confidence" in the shot.
        self.last_decision_value = self.filter.cell_marginal(cell)
        return ShotAction(cell=cell)

    def observe(self, action: Action, observed: int) -> None:
        if isinstance(action, ShotAction):
            self.filter.reweight_shot(cell=action.cell, observed=observed)
        elif isinstance(action, AskAction):
            # Thompson doesn't ask; tolerate this for a uniform interface.
            from engine.questions import question_by_id
            self.filter.reweight_ask(
                question=question_by_id(action.question_id),
                observed=observed, eps=self.eps,
            )
