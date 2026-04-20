#!/usr/bin/env bash
# Benchmark Thompson + EIG + ELLR across 10 shared-truth trials.
#
# Each trial samples one ground-truth board; all three strategies play that
# same board with deterministically-derived per-strategy seeds. Output lands
# in results/benchmark_3x10.json; a one-line per-strategy summary is printed
# on stderr at the end.
#
# Configuration: 8x8 board, ships {4,3,3,2}, BSC(eps=0.10) on asks, hard
# turn cap T_max=80, master seed=0.

set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-results/benchmark_3x10.json}"

.venv/bin/python -m simulator.benchmark \
    --strategies thompson eig ellr \
    --trials 10 \
    --seed 0 \
    --t-max 80 \
    --eps 0.10 \
    --out "$OUT"
