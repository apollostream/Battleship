"""engine.board — ship geometry, valid configurations, and game-state accounting.

Implements the constraint set S from battleship.md:
    1. exactly four ships
    2. fleet lengths {2, 3, 3, 4}
    3. no overlap, no orthogonal touching

Game-state semantics:
    * Shots are noiseless: apply_shot → HIT/MISS, deterministic from truth.
    * Asks are turn-advancing only at this layer; posterior update lives in smc.py.
    * Termination: hits == MAX_HITS OR turn == t_max.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, Iterable

BOARD_SIZE: int = 8
FLEET_LENGTHS: tuple[int, ...] = (4, 3, 3, 2)
MAX_HITS: int = sum(FLEET_LENGTHS)

Cell = tuple[int, int]   # (row, col), both in [0, BOARD_SIZE)


class Orientation(Enum):
    HORIZONTAL = "H"
    VERTICAL = "V"


class ShotResult(Enum):
    HIT = "HIT"
    MISS = "MISS"


# --------------------------------------------------------------------------
# Ship — a single oriented placement
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Ship:
    length: int
    row: int
    col: int
    orientation: Orientation

    def cells(self) -> list[Cell]:
        if self.orientation is Orientation.HORIZONTAL:
            return [(self.row, self.col + k) for k in range(self.length)]
        return [(self.row + k, self.col) for k in range(self.length)]

    def in_bounds(self) -> bool:
        return all(0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE for r, c in self.cells())

    def validate_in_bounds(self) -> None:
        if not self.in_bounds():
            raise ValueError(f"Ship {self} extends off the {BOARD_SIZE}x{BOARD_SIZE} board")


# --------------------------------------------------------------------------
# Configuration — a fleet of ships satisfying S
# --------------------------------------------------------------------------

def _orthogonal_neighbours(cell: Cell) -> list[Cell]:
    r, c = cell
    return [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]


@dataclass(frozen=True)
class Configuration:
    ships: tuple[Ship, ...]

    def is_valid(self) -> bool:
        # Fleet composition: exactly the multiset FLEET_LENGTHS.
        if sorted(s.length for s in self.ships) != sorted(FLEET_LENGTHS):
            return False
        # All in bounds.
        if not all(s.in_bounds() for s in self.ships):
            return False
        # No overlap, no orthogonal touching.
        per_ship_cells = [frozenset(s.cells()) for s in self.ships]
        all_cells: set[Cell] = set()
        for cells in per_ship_cells:
            if all_cells & cells:           # overlap
                return False
            all_cells |= cells
        for i, cells_i in enumerate(per_ship_cells):
            for j in range(i + 1, len(per_ship_cells)):
                cells_j = per_ship_cells[j]
                for c in cells_i:
                    if any(n in cells_j for n in _orthogonal_neighbours(c)):
                        return False
        return True

    def occupied_cells(self) -> FrozenSet[Cell]:
        out: set[Cell] = set()
        for s in self.ships:
            out.update(s.cells())
        return frozenset(out)

    def X(self, cell: Cell) -> int:
        return 1 if cell in self.occupied_cells() else 0


# --------------------------------------------------------------------------
# Scoring — pure function, lower = better
# --------------------------------------------------------------------------

def score_comparison_key(*, hits: int, turns: int, t_max: int) -> tuple[int, int, int]:
    """Sort key for ranking trajectories.  Lower is better.

    Composed as (sank_flag, turns_if_sank, -hits):
      * sank_flag: 0 if H == MAX_HITS else 1  — sinkers sort before timed-out runs.
      * turns_if_sank: T if sank else t_max   — among sinkers, fewer turns wins.
      * -hits: negative so that more hits sort lower among timed-out runs.
    """
    sank = hits == MAX_HITS
    return (0 if sank else 1, turns if sank else t_max, -hits)


# --------------------------------------------------------------------------
# GameState — turn accounting, history, termination
# --------------------------------------------------------------------------

ActionRecord = tuple   # ("shot", cell, ShotResult)  or  ("ask", question_id, answer)


@dataclass
class GameState:
    truth: Configuration
    t_max: int
    turn: int = 0
    hits: int = 0
    shots_fired: FrozenSet[Cell] = field(default_factory=frozenset)
    history: list[ActionRecord] = field(default_factory=list)

    @property
    def terminated(self) -> bool:
        return self.hits >= MAX_HITS or self.turn >= self.t_max

    @property
    def score(self) -> tuple[int, int]:
        return (self.hits, self.turn)

    def comparison_key(self) -> tuple[int, int, int]:
        return score_comparison_key(hits=self.hits, turns=self.turn, t_max=self.t_max)

    def _guard_live(self) -> None:
        if self.terminated:
            raise RuntimeError("GameState is terminated; no further actions allowed")

    def apply_shot(self, cell: Cell) -> ShotResult:
        self._guard_live()
        if cell in self.shots_fired:
            raise ValueError(f"cell {cell} already shot at this game")
        result = ShotResult.HIT if self.truth.X(cell) == 1 else ShotResult.MISS
        self.shots_fired = self.shots_fired | {cell}
        if result is ShotResult.HIT:
            self.hits += 1
        self.turn += 1
        self.history.append(("shot", cell, result))
        return result

    def apply_ask(self, question_id: str, observed_answer: int) -> None:
        self._guard_live()
        self.turn += 1
        self.history.append(("ask", question_id, observed_answer))
