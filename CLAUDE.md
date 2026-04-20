# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status: implemented + playable

The simulator is past MVP.  Five strategies run against the exact posterior over |S|=5,174,944 configurations (`thompson`, `eig`, `ellr`, `eig_approx`, `ellr_approx`, plus the interactive `user` strategy) and the full stack — engine, strategies, runner/benchmark, FastAPI bridge, single-page UI — is wired end-to-end.

Layout:
- `engine/` — board, question catalogue, exact posterior (`exact.py`), SMC fallback, metrics.
- `strategies/` — one file per doctrine (`thompson.py`, `eig.py`, `ellr.py`, `_mbayes.py`, `user.py`).
- `simulator/runner.py` — `GameSession.step()` state machine + `run_game` wrapper + CLI.
- `simulator/benchmark.py` — multi-trial runner (shared-truth trials, deterministic per-strategy seeding, `--approx` flag).
- `web/app.py` — FastAPI bridge: `POST /session`, `POST /session/{id}/action|step`, `GET /session/{id}/state|trajectory`, `GET /strategies|questions`. Serves `mockups/admiralty_dashboard.html` at `/ui/` and results JSONs at `/results/`.
- `mockups/admiralty_dashboard.html` — single-page UI. Drop a trajectory JSON to replay; click the **Live** button to start an interactive session against the bridge. User strategy fires a shot on any board cell click; ask picker populates from `/questions`.
- `tests/` — pytest suite. `@pytest.mark.slow` on anything that builds `ExactPosterior` (~30s cold start, cached for the session via `enumerate_all`).
- `results/` — benchmark JSON + markdown reports. `benchmark_3x10.json` (exact), `benchmark_3x10_approx.json` (sample-backed).

Design background worth knowing:
- `battleship.md` — the design spec.
- `eig_vs_ellr.jsx`, `eig_vs_ellr_v2.jsx` — toy 4×4 reference implementations of the EIG/ELLR math. Treat as executable math specs for the formulas in `engine/metrics.py`.
- `transcript.md` / `.tex` / `.html` / `.log` — canonical derivation of EIG vs E[log LR] as weighted KL divergences.

Commands:
```bash
# Tests (slow = fixtures that enumerate |S|)
.venv/bin/pytest -q                       # fast suite
.venv/bin/pytest -q -m slow               # slow suite
.venv/bin/pytest tests/test_strategies.py::TestApproxStrategies -q

# One-off game
.venv/bin/python -m simulator.runner --strategy eig_approx --seed 0 --out /tmp/game.json

# Benchmarks
bash scripts/benchmark_3x10.sh results/benchmark_3x10.json   # exact (~58 min)
.venv/bin/python -m simulator.benchmark --strategies thompson eig ellr --approx \
    --trials 10 --seed 0 --out results/benchmark_3x10_approx.json   # ~5 min

# Dev server + UI
.venv/bin/uvicorn web.app:app --host 127.0.0.1 --port 8000 --reload
# Then open http://127.0.0.1:8000/ui/  (redirect from /)
```

## The project being built

A **Battleship strategy-comparison simulator** (not a game-first product). Four strategies compete on identical boards:

| Strategy | Question choice | Shot choice |
|---|---|---|
| EIG + MBayes | argmax EIG over question catalogue | MAP cell |
| ELLR + MBayes | argmax E[log LR] | MAP cell |
| BO + EI (or UCB) | none — shoots every turn | acquisition over per-cell posterior |
| User | interactive | interactive |

Core rules: 8×8 grid, ships {4,3,3,2}, each turn pick **ask** (binary question, no state change, BSC(ε) noisy answer) or **shoot** (marks cell, hit/miss). Game ends when all 12 ship cells are hit; score = turn count. The ask-vs-shoot decision is the pivotal design axis — currently resolved as a cost-aware expected-reward comparator (see "Architectural constraints" and the resolved open-question #5 below), not an entropy-balanced one as `battleship.md:106` originally argued.

## Architectural constraints that must carry through any implementation

- **Hypothesis space is enumerable after all.** `engine/enumerate.py` computes $|\mathcal{S}| = 5{,}174{,}944$ exactly in ~30s via bitmask backtracking. SMC is still present (`engine/smc.py`) as a conceptual baseline and a fallback for fleet/board changes that blow the budget, but for the current 8×8 / {4,3,3,2} problem, **exact inference is feasible** — a precomputed $|S|\times 64$ occupancy matrix plus $|S|\times|Q|$ question-truth matrix turns each observation into a vectorised $O(|\mathcal{S}|)$ weight update and each question-scoring pass into a matrix multiply. When adding new strategies or question types, prefer the exact path; only fall back to SMC if profile-driven.
- **Cell marginals come from the particle ensemble:** μ(c) = Σᵢ wᵢ · 𝟙[particle i has ship at c]. MBayes shot selection uses argmax μ(c); BO acquisition functions should use both μ(c) and its variance σ(c) across particles (otherwise EI on a Bernoulli degenerates to greedy MAP — see `battleship.md:45`).
- **EIG ≤ E[log LR] always.** The gap is O(Σ πₛ²) and the two reference distributions differ by whether board s is included in the mixture. Both must be implemented with natural log (nats) to match `transcript.md`; the JSX uses log₂ for EIG and natural log for ELLR, so a Python port should standardize.
- **Cost-aware ask-vs-shoot comparator.** The current rule (overriding the earlier info-vs-action threshold) is `shoot iff max_c μ(c) ≥ (M + C) / (R⁺ + M)`; canonical `(R⁺, M, C) = (2, 1, 1)` gives threshold 2/3. Lives in `strategies/_mbayes.py:MBayesStrategy` and applies to all four MBayes variants. Thompson still always shoots.
- **Session = one game driven one turn at a time.** `simulator.runner.GameSession.step()` consults the strategy, applies the action, feeds the observation back, returns the JSON-ready turn record. `run_game` just loops `step()` to termination. The FastAPI bridge and the interactive `UserStrategy` both rely on this — do not collapse back into a single monolithic loop.

## Open design questions (from `battleship.md:99`)

Flag these to the user before implementing — they change the math and the code:

1. ~~Noiseless shots, or BSC(ε) on shots too?~~ **Resolved: shots noiseless (delta likelihood); BSC(ε) on queries only.** Asks model uncertain intel, shots model direct observation. Update `engine/board.py` and `engine/questions.py` accordingly.
2. ~~Ships allowed to touch, or forbidden?~~ **Resolved: forbidden** (encoded in $\mathcal{S}$).
3. ~~BO-with-EI: pure greedy MAP, UCB with exploration bonus, or Thompson sampling?~~ **Resolved: Thompson sampling.** Doctrine III file is `strategies/thompson.py`, not `bo_ei.py`.
4. ~~Hard turn cap or no cap?~~ **Resolved: hard cap** $T_{\max}$ (default 80, user-configurable in UI). Trajectory score is the pair $(H, T)$ — hits in $\{0,\ldots,12\}$ and turns in $\{1,\ldots,T_{\max}\}$. Benchmark sort: primary by $T$ when $H=12$, else by $H$.
5. ~~Ask-vs-shoot threshold: fixed or adaptive?~~ **Resolved: cost-aware, expected-reward comparator** (supersedes the earlier entropy/EIG balance). Shoot iff $\max_c \mu(c) \ge (M + C) / (R^{+} + M)$; canonical $(R^{+}, M, C) = (2, 1, 1)$ gives threshold $2/3$. Equivalently: shoot iff the best-cell shot has non-negative expected net reward. Asks are zero-cost in the comparator — the question score (EIG/ELLR) only picks *which* ask to make if no shot pays. Applies to EIG + MBayes, ELLR + MBayes, and both approx variants; Thompson always shoots (no ask branch). Lives in `strategies/_mbayes.py`.

**Deferred (not MVP):** POMCP / determinized-rollout 5th doctrine as a *performance ceiling* baseline. Add only after the four primary doctrines are validated end-to-end.

**SMC sampler must sample uniformly over $\mathcal{S}$.** Naive sequential placement biases toward configurations where early ships didn't block later ones; use rejection sampling (~5–10% acceptance on 8×8/{2,3,3,4}) or importance-weighted sequential. Validate against the enumerated $\mu_0(c)$ — corners ≈ 0.13, edges ≈ 0.16–0.18, interior ≈ 0.20–0.23, board mean exactly $12/64 = 0.1875$. See `battleship.md` "Prior cell marginal" section for full derivation.

## Licensing convention

- **Code** is Apache 2.0 (see `LICENSE` at repo root). Any new source files inherit this; no per-file headers required.
- **Documents and manuals** (specs, derivations, reports, write-ups) are **CC BY-SA 4.0**. When creating a new prose document, include a top-of-file copyright line and a bottom-of-file license footer matching the pattern in `battleship.md` and `transcript.md`. Attribute to "Michael L. Thompson" with the current year.
- Operational tooling files (`CLAUDE.md`, memory files, `.gitignore`) are not "documents" in this sense and don't need CC footers.

## Working with the JSX prototypes

The prototypes run in a React sandbox (no local build config exists here). Treat them as executable math specs — when porting to Python, match the formulas in `eig_vs_ellr.jsx:27–65` exactly, then validate the port by reproducing the prototype's simulation outputs on the 4-hypothesis toy problem before scaling to the full 8×8 SMC version.
