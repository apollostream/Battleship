# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status: pre-implementation

This repo is currently a **design/research workspace**, not yet a codebase. Before writing code, confirm the open design questions below with the user — `battleship.md` lists five that shape the architecture and must be answered before implementation.

What's here:
- `battleship.md` — the design spec for the project. Read this first; everything else is supporting material.
- `eig_vs_ellr.jsx`, `eig_vs_ellr_v2.jsx` — **reference implementations** of the EIG and E[log LR] math on a toy 4×4 / 4-hypothesis problem. These are React prototypes; the math inside them (functions `computeEIG`, `computeELLR`, `pHYes`, `margYes`, `updateW`) is the ground truth any Python port must reproduce exactly.
- `transcript.md` / `.tex` / `.html` / `.log` — the derivation transcripts. `transcript.md` is the canonical derivation of EIG vs E[log LR] as weighted KL divergences against different reference distributions; cite this when explaining the math.
- `.venv/` — empty Python venv (only `pip` installed). No `requirements.txt`, `pyproject.toml`, or `package.json` yet.
- No git, no tests, no build system, no Makefile.

## The project being built

A **Battleship strategy-comparison simulator** (not a game-first product). Four strategies compete on identical boards:

| Strategy | Question choice | Shot choice |
|---|---|---|
| EIG + MBayes | argmax EIG over question catalogue | MAP cell |
| ELLR + MBayes | argmax E[log LR] | MAP cell |
| BO + EI (or UCB) | none — shoots every turn | acquisition over per-cell posterior |
| User | interactive | interactive |

Core rules: 8×8 grid, ships {4,3,3,2}, each turn pick **ask** (binary question, no state change, BSC(ε) noisy answer) or **shoot** (marks cell, hit/miss). Game ends when all 12 ship cells are hit; score = turn count. The ask-vs-shoot decision is the pivotal design axis — prefer an adaptive information-vs-action threshold over a fixed one (the spec argues for this explicitly at `battleship.md:106`).

## Architectural constraints that must carry through any implementation

- **Hypothesis space is not enumerable.** ~millions of legal placements on 8×8. Use sequential Monte Carlo with N≈2000–5000 particles; reweight on each observation; resample on low ESS. Do not attempt to enumerate — the toy JSX prototype enumerates only because it has 4 hypotheses, and that approach does not scale.
- **Cell marginals come from the particle ensemble:** μ(c) = Σᵢ wᵢ · 𝟙[particle i has ship at c]. MBayes shot selection uses argmax μ(c); BO acquisition functions should use both μ(c) and its variance σ(c) across particles (otherwise EI on a Bernoulli degenerates to greedy MAP — see `battleship.md:45`).
- **EIG ≤ E[log LR] always.** The gap is O(Σ πₛ²) and the two reference distributions differ by whether board s is included in the mixture. Both must be implemented with natural log (nats) to match `transcript.md`; the JSX uses log₂ for EIG and natural log for ELLR, so a Python port should standardize.
- **Proposed layout** (from `battleship.md:80`): `engine/` (board, questions, smc, metrics) + `strategies/` (one file per strategy, common `choose_action` interface) + `simulator/` (single-run + multi-trial benchmark) + `ui/` as a separate consumer. A Python-only benchmark-mode MVP is the fastest path to producing comparison data; add UI only after the strategies work.

## Open design questions (from `battleship.md:99`)

Flag these to the user before implementing — they change the math and the code:

1. Noiseless shots, or BSC(ε) on shots too?
2. Ships allowed to touch, or forbidden?
3. BO-with-EI: pure greedy MAP, UCB with exploration bonus, or Thompson sampling?
4. Hard turn cap or no cap?
5. Ask-vs-shoot threshold: fixed or adaptive? (Spec recommends adaptive.)

## Licensing convention

- **Code** is Apache 2.0 (see `LICENSE` at repo root). Any new source files inherit this; no per-file headers required.
- **Documents and manuals** (specs, derivations, reports, write-ups) are **CC BY-SA 4.0**. When creating a new prose document, include a top-of-file copyright line and a bottom-of-file license footer matching the pattern in `battleship.md` and `transcript.md`. Attribute to "Michael L. Thompson" with the current year.
- Operational tooling files (`CLAUDE.md`, memory files, `.gitignore`) are not "documents" in this sense and don't need CC footers.

## Working with the JSX prototypes

The prototypes run in a React sandbox (no local build config exists here). Treat them as executable math specs — when porting to Python, match the formulas in `eig_vs_ellr.jsx:27–65` exactly, then validate the port by reproducing the prototype's simulation outputs on the 4-hypothesis toy problem before scaling to the full 8×8 SMC version.
