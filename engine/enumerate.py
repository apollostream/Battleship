"""engine.enumerate — exact enumeration over the constraint set S.

Computes:
  * |S|: the number of valid fleet configurations (up to permutation of
    identical-length ships — the two length-3 ships are indistinguishable,
    so their index order is canonicalised to avoid double-counting).
  * μ_0: the exact 8×8 prior cell marginal grid, Pr[cell c occupied].
  * masks / hparity / cells_matrix: full per-configuration arrays used by
    engine.exact.ExactPosterior to run exact (non-sampled) inference.

Representation uses 64-bit bitmasks: cell (r, c) maps to bit index 8·r + c.
A ship's "cells mask" is the bits for its own squares; its "exclusion mask"
is cells ∪ orthogonal neighbours.  Two ships are compatible iff the second
ship's cells don't intersect the first's exclusion mask (and vice versa,
which is equivalent since exclusion is symmetric for this constraint).
"""
from __future__ import annotations

import functools

import numpy as np

from engine.board import BOARD_SIZE, FLEET_LENGTHS, Orientation, Ship
from engine.smc import _PLACEMENTS_BY_LENGTH


_BIT = lambda r, c: 1 << (r * BOARD_SIZE + c)


def _cells_mask(ship: Ship) -> int:
    m = 0
    for r, c in ship.cells():
        m |= _BIT(r, c)
    return m


def _exclusion_mask(ship: Ship) -> int:
    """Cells the ship itself occupies plus their orthogonal neighbours (clipped
    to board).  A later ship is compatible iff its cells mask AND this is 0."""
    m = 0
    for r, c in ship.cells():
        m |= _BIT(r, c)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                m |= _BIT(nr, nc)
    return m


def _precompute(length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    placements = _PLACEMENTS_BY_LENGTH[length]
    cells = np.array([_cells_mask(p) for p in placements], dtype=object)
    excl = np.array([_exclusion_mask(p) for p in placements], dtype=object)
    horiz = np.array(
        [p.orientation is Orientation.HORIZONTAL for p in placements], dtype=bool
    )
    return cells, excl, horiz


# hparity flag bit mapping: bit index = length - 2.  Length-2 → bit 0, length-3
# → bit 1, length-4 → bit 2.  A config's hparity byte has bit L set iff at least
# one ship of length L+2 is horizontal.  Covers the Question(kind="hparity")
# catalogue exactly.
HPARITY_BIT: dict[int, int] = {2: 0, 3: 1, 4: 2}


@functools.lru_cache(maxsize=1)
def enumerate_all() -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Full enumeration of S.  Returns:

        count: |S| (int)
        masks: (|S|,) uint64 — occupancy bitmask per configuration
        hparity: (|S|,) uint8 — horizontal-ship-of-length-L flags
                                 (bit 0: L=2, bit 1: L=3, bit 2: L=4)
        cells_matrix: (|S|, 64) bool — cells_matrix[i, 8r+c] iff config i occupies (r, c)
        mu_0: (8, 8) float — Pr[cell (r, c) occupied] under uniform prior

    Cached — the ~30s enumeration runs once per process.  Total memory footprint
    is ~375 MB (41 MB masks + 5 MB hparity + 330 MB bool cells matrix).
    """
    lengths = list(FLEET_LENGTHS)
    assert sorted(lengths) == [2, 3, 3, 4], (
        "enumerator assumes fleet {4,3,3,2}; generalise before changing fleet"
    )

    c4, e4, h4 = _precompute(4)
    c3, e3, h3 = _precompute(3)
    c2, e2, h2 = _precompute(2)

    # Pre-allocate generously; trim after the loop.  The true |S| for 8×8 /
    # {4,3,3,2} is 5,174,944 — 6 M is a safe ceiling.
    max_size = 6_000_000
    masks = np.empty(max_size, dtype=np.uint64)
    hparity = np.empty(max_size, dtype=np.uint8)
    count = 0

    h2_bit = 1 << HPARITY_BIT[2]
    h3_bit = 1 << HPARITY_BIT[3]
    h4_bit = 1 << HPARITY_BIT[4]

    n4, n3, n2 = len(c4), len(c3), len(c2)
    for i4 in range(n4):
        cells4 = c4[i4]
        excl4 = e4[i4]
        h4_flag = h4_bit if h4[i4] else 0
        for i3a in range(n3):
            cells3a = c3[i3a]
            if cells3a & excl4:
                continue
            excl_after_3a = excl4 | e3[i3a]
            h3a_horiz = bool(h3[i3a])
            # Canonical: i_3b > i_3a to avoid double-counting identical-length swaps.
            for i3b in range(i3a + 1, n3):
                cells3b = c3[i3b]
                if cells3b & excl_after_3a:
                    continue
                excl_after_3b = excl_after_3a | e3[i3b]
                combined_cells = cells4 | cells3a | cells3b
                h3_flag = h3_bit if (h3a_horiz or h3[i3b]) else 0
                for i2 in range(n2):
                    cells2 = c2[i2]
                    if cells2 & excl_after_3b:
                        continue
                    occupied = combined_cells | cells2
                    masks[count] = occupied
                    hparity[count] = h4_flag | h3_flag | (h2_bit if h2[i2] else 0)
                    count += 1

    if count == 0:
        raise RuntimeError("enumerator produced zero valid configurations — bug")
    if count > max_size:
        raise RuntimeError(
            f"pre-allocation exceeded: count={count} > max_size={max_size}"
        )

    masks = masks[:count].copy()
    hparity = hparity[:count].copy()

    # Unpack each uint64 mask into a (count, 64) bool matrix via per-bit shifts.
    # Portable across endianness (avoids byteorder assumptions from viewing uint64
    # as 8 uint8s).  ~1 s total for 64 passes over 41 MB.
    cells_matrix = np.empty((count, 64), dtype=bool)
    for b in range(64):
        cells_matrix[:, b] = ((masks >> np.uint64(b)) & np.uint64(1)) == np.uint64(1)

    mu_0 = cells_matrix.mean(axis=0, dtype=np.float64).reshape(BOARD_SIZE, BOARD_SIZE)
    return count, masks, hparity, cells_matrix, mu_0


def enumerate_S() -> tuple[int, np.ndarray]:
    """Backwards-compatible shim: (|S|, μ_0 grid)."""
    count, _, _, _, mu_0 = enumerate_all()
    return count, mu_0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _format_grid(mu: np.ndarray) -> str:
    rows = []
    rows.append("    " + " ".join(f"{c+1:>5}" for c in range(BOARD_SIZE)))
    for r in range(BOARD_SIZE):
        row = f"{'ABCDEFGH'[r]:>2}  " + " ".join(f"{mu[r,c]:.3f}" for c in range(BOARD_SIZE))
        rows.append(row)
    return "\n".join(rows)


def main() -> int:
    import time
    t0 = time.perf_counter()
    count, mu = enumerate_S()
    dt = time.perf_counter() - t0
    print(f"|S| = {count:,}   (enumerated in {dt:.2f}s)")
    print(f"mean μ_0 = {mu.mean():.6f}  (expected {MAX_HITS}/64 = {MAX_HITS/64:.6f})")
    print(f"min μ_0  = {mu.min():.4f}   max μ_0 = {mu.max():.4f}")
    print(f"Σ μ_0    = {mu.sum():.6f}   (expected {MAX_HITS})")
    print()
    print("μ_0(c) — exact:")
    print(_format_grid(mu))
    return 0


if __name__ == "__main__":
    from engine.board import MAX_HITS
    raise SystemExit(main())
