"""engine.questions — binary-question catalogue, Q(s) evaluator, BSC(ε) likelihood.

The catalogue is built once at import time and exposed as QUESTION_CATALOGUE.
Each question has a stable string id suitable for trajectory logging.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from engine.board import BOARD_SIZE, FLEET_LENGTHS, Configuration, Orientation

# --------------------------------------------------------------------------
# Question — immutable, carries its own evaluator
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Question:
    id: str
    kind: str                      # "cell" | "row" | "col" | "quadrant" | "hparity"
    row: Optional[int] = None
    col: Optional[int] = None
    quadrant: Optional[str] = None # "NW" | "NE" | "SW" | "SE"
    length: Optional[int] = None   # for hparity


# --------------------------------------------------------------------------
# Catalogue factory
# --------------------------------------------------------------------------

_QUADRANT_BOUNDS = {
    "NW": (range(0, 4), range(0, 4)),
    "NE": (range(0, 4), range(4, 8)),
    "SW": (range(4, 8), range(0, 4)),
    "SE": (range(4, 8), range(4, 8)),
}


def build_catalogue() -> tuple[Question, ...]:
    out: list[Question] = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            out.append(Question(id=f"cell:{r}-{c}", kind="cell", row=r, col=c))
    for r in range(BOARD_SIZE):
        out.append(Question(id=f"row:{r}", kind="row", row=r))
    for c in range(BOARD_SIZE):
        out.append(Question(id=f"col:{c}", kind="col", col=c))
    for quad in ("NW", "NE", "SW", "SE"):
        out.append(Question(id=f"quad:{quad}", kind="quadrant", quadrant=quad))
    for L in sorted(set(FLEET_LENGTHS)):
        out.append(Question(id=f"hparity:{L}", kind="hparity", length=L))
    return tuple(out)


QUESTION_CATALOGUE: tuple[Question, ...] = build_catalogue()
_BY_ID: dict[str, Question] = {q.id: q for q in QUESTION_CATALOGUE}


def question_by_id(qid: str) -> Question:
    return _BY_ID[qid]


# --------------------------------------------------------------------------
# Evaluation — Q(s) ∈ {0, 1}
# --------------------------------------------------------------------------

def evaluate(q: Question, cfg: Configuration) -> int:
    occ = cfg.occupied_cells()
    if q.kind == "cell":
        return 1 if (q.row, q.col) in occ else 0
    if q.kind == "row":
        return 1 if any(r == q.row for r, _ in occ) else 0
    if q.kind == "col":
        return 1 if any(c == q.col for _, c in occ) else 0
    if q.kind == "quadrant":
        rows, cols = _QUADRANT_BOUNDS[q.quadrant]
        return 1 if any(r in rows and c in cols for r, c in occ) else 0
    if q.kind == "hparity":
        return 1 if any(
            s.length == q.length and s.orientation is Orientation.HORIZONTAL
            for s in cfg.ships
        ) else 0
    raise ValueError(f"unknown question kind: {q.kind}")


# --------------------------------------------------------------------------
# BSC(ε) likelihood
# --------------------------------------------------------------------------

def bsc_likelihood(observed: int, truth: int, eps: float) -> float:
    if not 0.0 <= eps <= 1.0:
        raise ValueError(f"epsilon must be in [0, 1], got {eps}")
    return (1.0 - eps) if observed == truth else eps
