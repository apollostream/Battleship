# Benchmark — Thompson vs EIG vs ELLR (10 trials, cost-aware comparator)

_© 2026, Michael L. Thompson. This work is licensed under [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/)._

---

## Configuration

| parameter | value |
|---|---|
| board | 8×8 |
| fleet | {4, 3, 3, 2} (12 ship cells) |
| BSC noise on asks | ε = 0.10 |
| hard turn cap | T_max = 80 |
| trials | 10 (shared truth across all strategies per trial) |
| master seed | 0 |
| backend | exact posterior over \|S\| = 5,174,944 |
| ask-vs-shoot rule | cost-aware: shoot iff max_c μ(c) ≥ 2/3 (R+=2, M=1, C=1) |
| scripts | `scripts/benchmark_3x10.sh` → `results/benchmark_3x10{,_approx}.json` |

Per-strategy seeds derived deterministically from `(master_seed, "strategy", trial_idx, name)` via blake2b, so adding a strategy doesn't perturb the others' results.

## Summary — exact variants (`benchmark_3x10.json`)

| strategy | sink rate | mean hits | mean T \| sank | mean asks | mean net_R | wall (s/game) |
|---|---:|---:|---:|---:|---:|---:|
| thompson | 1.00 | 12.00 | 32.8 | 0.0  | **−29.6** | ~8  |
| eig      | 1.00 | 12.00 | 43.4 | 28.6 | **+6.4**  | ~124 |
| ellr     | 1.00 | 12.00 | 41.4 | 26.7 | **+6.6**  | ~220 |

Total benchmark wall time: ~58 minutes.

## Summary — approximate variants (`benchmark_3x10_approx.json`)

Same cost-aware comparator; sample-backed question scoring (K=200 for EIG, K=500 for ELLR) and sample-based cell marginals replacing the |S|×|Q| block matmul.

| strategy     | sink rate | mean hits | mean T \| sank | mean asks | mean net_R | wall (s/game) |
|---|---:|---:|---:|---:|---:|---:|
| thompson     | 1.00 | 12.00 | 32.8 | 0.0  | **−29.6** | ~8  |
| eig_approx   | 1.00 | 12.00 | 38.9 | 24.6 | **+7.4**  | ~11 |
| ellr_approx  | 1.00 | 12.00 | 42.8 | 27.9 | **+6.2**  | ~11 |

Total benchmark wall time: ~5 minutes.

## Findings

1. **Cost-aware comparator makes asks fire.** Replacing the info-vs-action threshold with `μ ≥ (M + C) / (R⁺ + M) = 2/3` produces 20–30 asks per game for both EIG and ELLR.  Under the canonical reward scheme `(R⁺, M, C) = (2, 1, 1)`, Thompson's all-shoot policy is the worst in net_reward — every miss pays `−1` reward and `−1` shot cost.

2. **Asks buy net reward, not speed.**  EIG/ELLR run ~10 turns longer than Thompson but convert most of those turns into asks (zero cost), leaving fewer misses among the shots actually fired — net_R flips from −30 (thompson) to +6 (EIG/ELLR).

3. **EIG and ELLR are statistically tied at ε=0.10.** Mean net_reward +6.4 vs +6.6 (exact) and +7.4 vs +6.2 (approx).  The ELLR shortlist (top-K by EIG, then full-score) converges to near-EIG rankings in practice — expected given EIG and ELLR differ only by leave-one-out reference distribution and concentrate on similar questions when no single hypothesis dominates.

4. **Approximate variants win on the speed-quality trade.** Sample-based EIG/ELLR match (or slightly beat) their exact counterparts on net_reward while running **~11–20× faster** (~1.5× Thompson wall).  The BALD variance ranking `p̂(1−p̂)` is a sufficient statistic for EIG under BSC; the K-sample KL estimator (assuming `p̄_{-s} ≈ p̄`) matches exact ELLR to O(1/K).

## Performance breakdown

Exact EIG per-turn cost is dominated by `build_answer_matrix` build (one-time, ~6s) and the `|S|×|Q|` block matmul (~1s/turn once `|active| ≪ |S|`).  Exact ELLR adds per-question log computations over the active subset, ~3× EIG.

Approximate variants replace the block matmul with one posterior draw + K-row indexing:

| operation | exact | approx |
|---|---|---|
| sampler         | — | `rng.choice(|S|, K, p=w)` ≈ 200 ms early, sub-ms late |
| question score  | `weights @ A` over |S|×|Q| ≈ 0.5 s | `A[idx].mean(axis=0)` ≈ negligible |
| cell marginal   | `weights @ cells` over |S|×64 ≈ 0.6 s | `cells[idx].mean(axis=0)` ≈ negligible |
| per-turn wall   | ~1.2 s (EIG) / ~3 s (ELLR) | ~200 ms |

## Reproduction

```bash
# Exact (thompson, eig, ellr) — ~58 min
bash scripts/benchmark_3x10.sh results/benchmark_3x10.json

# Approximate (thompson, eig_approx, ellr_approx) — ~5 min
.venv/bin/python -m simulator.benchmark \
    --strategies thompson eig ellr --approx \
    --trials 10 --seed 0 --t-max 80 --eps 0.10 \
    --out results/benchmark_3x10_approx.json
```

Both trial sets are fully deterministic given `seed=0`; the JSON is the source of truth for all numbers above.

---

_© 2026, Michael L. Thompson. This work is licensed under [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/)._
