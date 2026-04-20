"""engine.exact — exact posterior over the full constraint set S.

The cardinality |S|=5,174,944 for the 8×8 / {4,3,3,2} fleet is small enough to
enumerate and hold in memory.  ExactPosterior replaces the sampled particle
filter with:

  * a weights vector of length |S| (every configuration represented exactly),
  * vectorised reweighting under noiseless shots (delta likelihood) and noisy
    asks (BSC(ε)),
  * machine-precision cell marginals μ(c) (no Monte-Carlo error),
  * Thompson sampling via weighted draw over configurations.

Precomputed state is shared across instances through engine.enumerate.enumerate_all()
(cached lru_cache(1)); only the weights vector is per-instance.  Memory: 41 MB
masks + 5 MB hparity + 330 MB cells matrix ≈ 375 MB shared, plus ~40 MB weights
per instance.

Answer-vector cache (self._answers) is populated lazily per question id.  Each
entry is a (|S|,) bool array; the catalogue of 87 questions, fully materialised,
would add ~440 MB worst-case — typical benchmarks only touch a handful.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from engine.board import BOARD_SIZE, Cell
from engine.enumerate import HPARITY_BIT, enumerate_all
from engine.questions import Question


# Row / column / quadrant masks live at module scope — computed once, shared.
_ROW_MASK: np.ndarray = np.array(
    [sum(1 << (r * BOARD_SIZE + c) for c in range(BOARD_SIZE)) for r in range(BOARD_SIZE)],
    dtype=np.uint64,
)
_COL_MASK: np.ndarray = np.array(
    [sum(1 << (r * BOARD_SIZE + c) for r in range(BOARD_SIZE)) for c in range(BOARD_SIZE)],
    dtype=np.uint64,
)
_QUAD_BOUNDS: dict[str, tuple[range, range]] = {
    "NW": (range(0, 4), range(0, 4)),
    "NE": (range(0, 4), range(4, 8)),
    "SW": (range(4, 8), range(0, 4)),
    "SE": (range(4, 8), range(4, 8)),
}
_QUAD_MASK: dict[str, np.uint64] = {
    q: np.uint64(sum(1 << (r * BOARD_SIZE + c) for r in rows for c in cols))
    for q, (rows, cols) in _QUAD_BOUNDS.items()
}


class ExactPosterior:
    """Exact posterior over S.  Drop-in replacement for ParticleFilter.

    Shared precomputed state (masks, hparity, cells_matrix) is held by reference
    from the lru_cached enumerate_all(); cheap to construct repeatedly.
    """

    def __init__(self) -> None:
        (
            self._count,
            self._masks,
            self._hparity,
            self._cells_matrix,
            _mu_0,
        ) = enumerate_all()
        self.weights: np.ndarray = np.full(self._count, 1.0 / self._count, dtype=np.float64)
        # Lazy per-question answer cache — bool (|S|,) arrays.
        self._answers: dict[str, np.ndarray] = {}
        # Lazy stacked answer matrix for vectorised question scoring.
        # Keyed by tuple-of-question-ids so shortlists / partial catalogues can
        # cache independently of the full one.
        self._answer_matrices: dict[tuple[str, ...], np.ndarray] = {}

    # ------------------------------------------------------------------
    # Properties / accessors
    # ------------------------------------------------------------------

    @property
    def n_configurations(self) -> int:
        return self._count

    def occupied_cells_of(self, idx: int) -> list[Cell]:
        """The 12 occupied cells of configuration idx, as (r, c) tuples."""
        m = int(self._masks[idx])
        out: list[Cell] = []
        while m:
            low = m & -m
            bit = low.bit_length() - 1
            out.append((bit // BOARD_SIZE, bit % BOARD_SIZE))
            m ^= low
        return out

    # ------------------------------------------------------------------
    # Posterior summaries
    # ------------------------------------------------------------------

    def cell_marginal_grid(self) -> np.ndarray:
        """μ(r, c) = Σ_i w_i · 1[config i occupies (r, c)].  Shape (8, 8).

        Sparsified on zero-weight rows: once shots have collapsed the posterior,
        most weights are exactly 0 and we copy the active rows into a float64
        matrix for BLAS, turning a ~200 ms bool×f64 loop into a sub-10 ms
        f64×f64 matmul.  Cold start (uniform) falls through the bool path.
        """
        mask = self.weights > 0.0
        if bool(mask.all()):
            mu_flat = np.dot(self.weights, self._cells_matrix)
        else:
            w = self.weights[mask]
            cells = self._cells_matrix[mask].astype(np.float64)
            mu_flat = w @ cells
        return mu_flat.reshape(BOARD_SIZE, BOARD_SIZE)

    def cell_marginal(self, cell: Cell) -> float:
        r, c = cell
        column = self._cells_matrix[:, r * BOARD_SIZE + c]
        return float(np.dot(self.weights, column))

    # ------------------------------------------------------------------
    # Posterior sampling — supports the approximate strategies
    # ------------------------------------------------------------------

    def sample_configs(self, *, K: int, rng: np.random.Generator) -> np.ndarray:
        """Draw K configuration indices from the posterior (with replacement).

        Fast path once the posterior has collapsed: the cumulative sum under
        ``rng.choice`` scales with the input length, so after shots have
        zeroed most weights we sample from the active subset and remap —
        turning a per-turn 300 ms sampler into a sub-ms one.  Cold start
        samples the full length.
        """
        if K <= 0:
            raise ValueError(f"K must be positive, got {K}")
        mask = self.weights > 0.0
        if bool(mask.all()):
            return rng.choice(self._count, size=K, replace=True, p=self.weights)
        active_idx = np.nonzero(mask)[0]
        active_w = self.weights[active_idx]
        local_idx = rng.choice(len(active_idx), size=K, replace=True, p=active_w)
        return active_idx[local_idx]

    def ess(self) -> float:
        return float(1.0 / np.sum(self.weights ** 2))

    # ------------------------------------------------------------------
    # Answer vectors — Q(s) for every config s
    # ------------------------------------------------------------------

    def answers_for(self, question: Question) -> np.ndarray:
        """Return (|S|,) bool array of Q(s) for each configuration s.  Cached."""
        cached = self._answers.get(question.id)
        if cached is not None:
            return cached
        arr = self._compute_answers(question)
        self._answers[question.id] = arr
        return arr

    def _compute_answers(self, q: Question) -> np.ndarray:
        if q.kind == "cell":
            # Column of the precomputed cells matrix.
            return self._cells_matrix[:, q.row * BOARD_SIZE + q.col]
        if q.kind == "row":
            return (self._masks & _ROW_MASK[q.row]) != np.uint64(0)
        if q.kind == "col":
            return (self._masks & _COL_MASK[q.col]) != np.uint64(0)
        if q.kind == "quadrant":
            return (self._masks & _QUAD_MASK[q.quadrant]) != np.uint64(0)
        if q.kind == "hparity":
            bit = np.uint8(1 << HPARITY_BIT[q.length])
            return (self._hparity & bit) != np.uint8(0)
        raise ValueError(f"unknown question kind: {q.kind}")

    def build_answer_matrix(self, questions: tuple[Question, ...]) -> np.ndarray:
        """Stack of answer columns for the given questions.

        Returned as bool (|S|, |Q|).  Cached by question-id tuple so a
        shortlist's matrix doesn't invalidate the catalogue's, and vice versa.
        Build cost is dominated by the 64-iteration bit-shift in
        ``_compute_answers`` for cell questions; ~6 s for the full 87-question
        catalogue, then free for the lifetime of the posterior.
        """
        key = tuple(q.id for q in questions)
        cached = self._answer_matrices.get(key)
        if cached is not None:
            return cached
        cols = [self.answers_for(q) for q in questions]
        mat = np.column_stack(cols)
        self._answer_matrices[key] = mat
        return mat

    # ------------------------------------------------------------------
    # Reweighting
    # ------------------------------------------------------------------

    def reweight_shot(self, cell: Cell, observed: int) -> None:
        """Delta likelihood: retain only configs whose occupancy matches `observed`."""
        r, c = cell
        column = self._cells_matrix[:, r * BOARD_SIZE + c]   # True where occupied
        keep = column if observed == 1 else ~column
        new_w = self.weights * keep
        total = new_w.sum()
        if total == 0.0:
            self._reset_to_uniform_prior()
            return
        self.weights = new_w / total

    def reweight_ask(self, question: Question, observed: int, eps: float) -> None:
        """BSC(ε) likelihood: (1-ε) on configs whose Q(s) matches observed, ε otherwise."""
        if not 0.0 <= eps <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {eps}")
        truth = self.answers_for(question)                     # (|S|,) bool
        match = truth == (observed == 1)                       # (|S|,) bool
        lik = np.where(match, 1.0 - eps, eps)
        new_w = self.weights * lik
        total = new_w.sum()
        if total == 0.0:
            self._reset_to_uniform_prior()
            return
        self.weights = new_w / total

    def _reset_to_uniform_prior(self) -> None:
        self.weights = np.full(self._count, 1.0 / self._count, dtype=np.float64)

    # ------------------------------------------------------------------
    # Thompson sampling
    # ------------------------------------------------------------------

    def thompson_sample(self, rng: np.random.Generator, shots_fired: Iterable[Cell]) -> Cell:
        """Draw config i ~ weights; pick uniformly from its unshot occupied cells.

        Fallback: if config i has no unshot occupied cells, pick uniformly from
        any unshot board cell.
        """
        shot_set = frozenset(shots_fired)
        i = int(rng.choice(self._count, p=self.weights))
        cells = self.occupied_cells_of(i)
        candidates = [c for c in cells if c not in shot_set]
        if candidates:
            j = int(rng.integers(len(candidates)))
            return candidates[j]
        unshot = [
            (r, c)
            for r in range(BOARD_SIZE)
            for c in range(BOARD_SIZE)
            if (r, c) not in shot_set
        ]
        if not unshot:
            raise RuntimeError("no unshot cells left — game should have terminated")
        j = int(rng.integers(len(unshot)))
        return unshot[j]
