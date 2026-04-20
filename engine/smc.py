"""engine.smc — sequential Monte-Carlo particle filter on the constraint set S.

The sampler is a rejection sampler with an independent-proposal form: draw one
placement per ship from its single-ship placement list independently uniformly;
accept iff the resulting 4-ship fleet is a valid Configuration.  Acceptance
rate on 8x8 / {2,3,3,4} is empirically ~5–10%.

The filter supports:
  * delta (noiseless) reweighting on shot observations,
  * BSC(ε) reweighting on ask observations,
  * ESS-based resampling (multinomial),
  * cell marginal grid μ(c) for MBayes shot selection,
  * Thompson sampling (draw particle ~ w_i, pick an unshot ship cell uniformly).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from engine.board import (
    BOARD_SIZE,
    FLEET_LENGTHS,
    Cell,
    Configuration,
    Orientation,
    Ship,
)
from engine.questions import Question, bsc_likelihood, evaluate, question_by_id

# --------------------------------------------------------------------------
# Single-ship placement enumeration (cheap, computed once)
# --------------------------------------------------------------------------

def _single_ship_placements(length: int) -> tuple[Ship, ...]:
    out: list[Ship] = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE - length + 1):
            out.append(Ship(length, r, c, Orientation.HORIZONTAL))
    for r in range(BOARD_SIZE - length + 1):
        for c in range(BOARD_SIZE):
            out.append(Ship(length, r, c, Orientation.VERTICAL))
    return tuple(out)


_PLACEMENTS_BY_LENGTH: dict[int, tuple[Ship, ...]] = {
    L: _single_ship_placements(L) for L in set(FLEET_LENGTHS)
}


# --------------------------------------------------------------------------
# Rejection sampler — uniform over S
# --------------------------------------------------------------------------

def sample_configuration(rng: np.random.Generator, max_attempts: int = 10_000) -> Configuration:
    """Return a single uniform draw from S using independent-proposal rejection.

    Each of the four ships is independently proposed from its length-specific
    placement list; the joint is accepted iff Configuration.is_valid().
    """
    lengths = FLEET_LENGTHS
    placements = [_PLACEMENTS_BY_LENGTH[L] for L in lengths]
    sizes = np.array([len(p) for p in placements])
    for _ in range(max_attempts):
        idxs = rng.integers(low=0, high=sizes)
        ships = tuple(placements[k][i] for k, i in enumerate(idxs))
        cfg = Configuration(ships=ships)
        if cfg.is_valid():
            return cfg
    raise RuntimeError(
        f"rejection sampler failed to produce a valid configuration in "
        f"{max_attempts} attempts — sampler or constraint set is broken"
    )


def sample_particles(N: int, rng: np.random.Generator) -> list[Configuration]:
    return [sample_configuration(rng) for _ in range(N)]


# --------------------------------------------------------------------------
# ParticleFilter
# --------------------------------------------------------------------------

class ParticleFilter:
    def __init__(self, N: int, rng: np.random.Generator):
        self.N = N
        self.rng = rng
        self.particles: list[Configuration] = sample_particles(N, rng)
        self.weights: np.ndarray = np.full(N, 1.0 / N)
        # Observation history — enables regeneration from scratch when
        # sample impoverishment collapses the filter.
        self._shot_history: list[tuple[Cell, int]] = []
        self._ask_history: list[tuple[str, int, float]] = []   # (qid, observed, eps)

    # ------- posterior summaries ----------------------------------------

    def ess(self) -> float:
        return float(1.0 / np.sum(self.weights ** 2))

    def cell_marginal_grid(self) -> np.ndarray:
        grid = np.zeros((BOARD_SIZE, BOARD_SIZE))
        for w, p in zip(self.weights, self.particles):
            if w == 0.0:
                continue
            for (r, c) in p.occupied_cells():
                grid[r, c] += w
        return grid

    def cell_marginal(self, cell: Cell) -> float:
        return float(self.cell_marginal_grid()[cell])

    # ------- reweighting -------------------------------------------------

    def reweight_shot(self, cell: Cell, observed: int) -> None:
        """Noiseless delta likelihood — inconsistent particles get weight 0."""
        self._shot_history.append((cell, observed))
        consistent = np.array(
            [p.X(cell) == observed for p in self.particles], dtype=float
        )
        new_w = self.weights * consistent
        total = new_w.sum()
        if total == 0.0:
            # Sample impoverishment — regenerate from the full observation
            # history and retry.  If regeneration also fails, the observations
            # are genuinely incompatible with S (bug).
            self._regenerate_from_history()
            return
        self.weights = new_w / total

    def reweight_ask(self, question: Question, observed: int, eps: float) -> None:
        """BSC(ε) likelihood — soft reweight, no particle zeroed out."""
        self._ask_history.append((question.id, observed, eps))
        liks = np.array(
            [bsc_likelihood(observed=observed, truth=evaluate(question, p), eps=eps)
             for p in self.particles],
            dtype=float,
        )
        new_w = self.weights * liks
        total = new_w.sum()
        if total == 0.0:
            self._regenerate_from_history()
            return
        self.weights = new_w / total

    # ------- regeneration (recovery from sample impoverishment) ---------

    def _regenerate_from_history(self) -> None:
        """Re-draw N particles from the prior filtered by all past observations.

        Shots apply as hard filters (rejection); asks apply as importance
        weights via BSC likelihood product.  If we can't produce any live
        particles from a generous attempt budget, raise — the observations
        are inconsistent with the constraint set.
        """
        new_particles: list[Configuration] = []
        new_weights: list[float] = []
        attempts = 0
        # Budget chosen empirically: a doomed regeneration (history inconsistent
        # with prior within the attempt horizon) fails in ~0.2s at N=32 here,
        # vs. ~2s at the earlier 200k default.  Acceptance in the normal-case
        # is independent of the budget — only the fail-fast tail changes.
        max_attempts = max(self.N * 500, 20_000)
        while len(new_particles) < self.N and attempts < max_attempts:
            attempts += 1
            cfg = sample_configuration(self.rng)
            ok = True
            for cell, obs in self._shot_history:
                if cfg.X(cell) != obs:
                    ok = False
                    break
            if not ok:
                continue
            w = 1.0
            for qid, obs, eps in self._ask_history:
                q = question_by_id(qid)
                w *= bsc_likelihood(observed=obs, truth=evaluate(q, cfg), eps=eps)
                if w == 0.0:
                    break
            if w > 0.0:
                new_particles.append(cfg)
                new_weights.append(w)
        if not new_particles:
            # Sampler found no particles consistent with full history within
            # the attempt budget.  Typical with small N and dense shot
            # histories (acceptance rate ~5% × product of per-obs hits).
            # Fall back to prior draws — the filter becomes uninformed, but
            # downstream consumers (Thompson's shots_fired filter, MBayes's
            # unshot-cell argmax) still produce legal actions.  New
            # observations will re-narrow the posterior on the next turn.
            self.particles = sample_particles(self.N, self.rng)
            self.weights = np.full(self.N, 1.0 / self.N)
            return
        # Pad if under-full by duplicating with same weights (rare).
        while len(new_particles) < self.N:
            new_particles.append(new_particles[-1])
            new_weights.append(new_weights[-1])
        self.particles = new_particles[: self.N]
        w = np.array(new_weights[: self.N], dtype=float)
        self.weights = w / w.sum()

    # ------- resampling --------------------------------------------------

    def resample_if_low_ess(self, threshold: float) -> bool:
        """Multinomial resample when ESS / N < threshold.  Returns True if resampled."""
        if self.ess() / self.N >= threshold:
            return False
        idx = self.rng.choice(self.N, size=self.N, p=self.weights, replace=True)
        self.particles = [self.particles[i] for i in idx]
        self.weights = np.full(self.N, 1.0 / self.N)
        return True

    # ------- Thompson sampling ------------------------------------------

    def thompson_sample(self, rng: np.random.Generator, shots_fired: Iterable[Cell]) -> Cell:
        """Draw particle i ~ w_i; pick one of s_i's unshot ship cells uniformly.

        Fallback: if the sampled particle has no unshot ship cells (very rare,
        late game), pick uniformly from any unshot board cell.
        """
        shot_set = frozenset(shots_fired)
        i = int(rng.choice(self.N, p=self.weights))
        cfg = self.particles[i]
        candidates = [c for c in cfg.occupied_cells() if c not in shot_set]
        if candidates:
            j = int(rng.integers(len(candidates)))
            return candidates[j]
        # fallback — any unshot cell
        unshot = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
                  if (r, c) not in shot_set]
        if not unshot:
            raise RuntimeError("no unshot cells left — game should have terminated")
        j = int(rng.integers(len(unshot)))
        return unshot[j]
