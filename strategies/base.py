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
    """Implementations must set ``last_decision_value`` during ``choose_action``.

    The value is the strategy's own expected-utility estimate at the chosen
    action: μ(c*) for shots (probability the cell is occupied) and the
    information-theoretic metric for asks (EIG or ELLR in nats).  The runner
    records it per turn so we can audit decision quality post-hoc.
    """
    last_decision_value: float

    def choose_action(self, shots_fired: FrozenSet[Cell], turn: int) -> Action: ...
    def observe(self, action: Action, observed: int) -> None: ...
