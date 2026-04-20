"""Strategy protocol and action types.

A Strategy maintains its own ParticleFilter posterior.  The simulator calls
`choose_action` to get the next move, applies it to the ground-truth
GameState to produce an observation, then feeds the observation back via
`observe` so the Strategy can reweight its posterior.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Protocol, Union

from engine.board import Cell


@dataclass(frozen=True)
class ShotAction:
    cell: Cell


@dataclass(frozen=True)
class AskAction:
    question_id: str


Action = Union[ShotAction, AskAction]


class Strategy(Protocol):
    def choose_action(self, shots_fired: FrozenSet[Cell], turn: int) -> Action: ...
    def observe(self, action: Action, observed: int) -> None: ...
