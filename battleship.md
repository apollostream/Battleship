# Battleship Strategy-Comparison Simulator — Design Sketch

_© 2026, Michael L. Thompson. This work is licensed under [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/)._

---

Actually, no research needed here — this is a design-spec sketch that flows from our prior conversation and from the mathematical framework we already worked through. Let me just think it through.

---

## Core design tensions to resolve first

Before sketching the plan, I want to flag three decisions that shape everything downstream. You should make these calls:

**1. Player-vs-computer vs. computer-only benchmark.**
These are very different products. The first is a *game* where a human competes against one of the four strategies; the second is a *simulator* where strategies compete against each other on the same board configuration, run many times, and aggregated. I'd strongly recommend the **simulator** direction first, with the human-play mode as an optional layer added later — because the whole point of the earlier work was comparing strategies quantitatively, and a PvC game confounds that with human skill. If you go simulator-first, you can always launch a game window that replays a single simulated match later.

**2. What are "questions" in full Battleship?**
In the paper, questions are arbitrary binary predicates over ship configurations — row/column queries, counting queries, "do any ships touch the edge", etc. This is powerful but also abstract. In classic Battleship, the only "question" is a shot at a single cell. My recommendation: **keep both**. Each turn the agent chooses between (a) asking a non-destructive binary question and (b) firing a shot. Questions are free to ask but don't score; shots score hits/misses. This makes the EIG/ELLR question-selection logic meaningful while still producing a playable game. The question language can be: cell queries, row-has-any-ship, col-has-any-ship, and maybe "ship of length k present in region R" — a carefully chosen finite set that's rich enough to differentiate strategies but bounded enough to compute over.

**3. Hypothesis space: explicit or particle-based?**
With 4 ships on 8×8, the number of valid placements (accounting for no-overlap constraints) is in the millions. You cannot enumerate. This forces a decision: use **sequential Monte Carlo with N particles** (the paper's approach) from the start. This is also where Bayesian Optimization with Expected Improvement slots in naturally — EI operates over a probabilistic surrogate rather than an enumerated posterior, so it lives more comfortably with SMC.

---

## Proposed game rules

**Setup.** 8×8 grid. Ships of lengths {4, 3, 3, 2}. Ships placed horizontally or vertically, non-overlapping, non-touching (or touching allowed — pick one; non-touching is easier to reason about). Placement is fixed at game start.

**Turn structure.** On each turn the agent chooses exactly one action:
- **Ask**: a binary question from the question catalogue. Answer is returned under BSC(ε). No state change to the board.
- **Shoot**: fire at a cell. Answer is "hit" or "miss" (also under BSC(ε) if you want noise everywhere, or noiseless for shots — another design call). Cell is marked permanently; cannot shoot same cell twice.

**Game end.** When all ship cells have been hit (i.e., when the total number of hit cells equals 4+3+3+2 = 12). Score = total turns taken. Lower is better.

**Budget constraint.** Optionally cap total turns at some ceiling (say 80). This prevents strategies from asking endless questions without ever shooting.

**Ask vs. shoot trade-off.** Questions are information-rich but don't progress the game; shots progress the game but may miss. Rational strategies must learn when to stop asking and start shooting. This is where the ask-vs-shoot decision logic differs across the four strategies:

---

## The four strategies

| Strategy | Question selection | Ask-vs-shoot decision | Shot selection |
|---|---|---|---|
| **EIG + MBayes** | argmax EIG over question catalogue | ask while max-cell-posterior is below threshold; shoot when it exceeds | MAP cell (argmax posterior marginal probability of containing any ship part) |
| **ELLR + MBayes** | argmax E[log LR] over question catalogue | same threshold rule | MAP cell |
| **BO with EI** | no questions — always shoots | n/a | argmax Expected Improvement over cells, where "improvement" = probability of hit given posterior, and the surrogate is the per-cell posterior probability |
| **User** | interactive menu | user chooses | user clicks |

A couple of notes on strategy 3: BO-with-EI on a discrete board is somewhat degenerate — EI over a Bernoulli outcome reduces to picking the cell with highest posterior hit probability, which is just greedy MAP. To make EI meaningfully different, you'd want to **include uncertainty in the cell marginal** as part of the acquisition function, e.g., $\alpha(c) = \mu(c) + \kappa \sigma(c)$ style (UCB-like), where $\mu(c)$ is the marginal posterior probability and $\sigma(c)$ is its variance across the particle ensemble. This gives BO a genuine exploration bonus that pure MAP doesn't have, and makes for a more interesting comparison. Worth discussing whether to use pure EI, UCB, or Thompson sampling here — all are BO-family methods and give different behaviour.

---

## Hypothesis space and inference

**Prior.** Uniform over all legal ship placements. Approximated by sampling N particles (say N = 2000–5000) at initialization.

**Update.** On each observation (question answer or shot result), multiply each particle's weight by the likelihood under BSC(ε). Resample when effective sample size falls below threshold (standard SMC practice).

**Cell marginal.** For any cell c, $\mu(c) = \sum_i w_i \cdot \mathbb{1}[\text{particle } i \text{ has a ship at } c]$. This is the per-cell posterior probability used by MBayes shot selection.

**Question catalogue.** I'd suggest a manageable set:
- 64 cell queries (is there a ship at (r,c)?)
- 8 row queries, 8 column queries
- 4 "is ship of length k placed horizontally?" queries (one per ship length — noting that two ships have length 3)
- Maybe 8 quadrant queries (does quadrant Q contain any ship?)

That's ~92 candidate questions per turn — tractable to score all of them each turn.

---

## Simulator vs. game UI

Two modes, sharing the same engine:
- **Benchmark mode**: given a seed, run all four strategies on the same ground-truth board, record full trajectories, plot comparison (turns-to-finish, hit rate over time, posterior entropy over time). Run 100+ trials with different seeds to get distributions. This is the analytically interesting mode.
- **Play mode**: user plays against one selected strategy. The strategy's board is hidden; user's board is shown. Alternating turns is traditional but complicates the comparison — I'd actually suggest **same-board mode**: both the user and the strategy see the same ground-truth board, and the score is just "who finished in fewer turns" after they both play it independently.

---

## What I'd sketch for Claude Code

A plan document roughly like:

```
PROJECT: battleship_strategy_comparison/
├── engine/
│   ├── board.py           # Ship placement, shot resolution, game state
│   ├── questions.py       # Question catalogue + true-answer evaluator
│   ├── smc.py             # Particle filter: sample, reweight, resample
│   └── metrics.py         # EIG, E[log LR], EI/UCB acquisition functions
├── strategies/
│   ├── base.py            # Strategy interface: choose_action(state) -> Action
│   ├── eig_mbayes.py
│   ├── ellr_mbayes.py
│   ├── bo_ei.py
│   └── user.py            # Reads from UI
├── simulator/
│   ├── runner.py          # Single-game runner
│   ├── benchmark.py       # Multi-trial comparison
│   └── trajectory.py      # Logging
└── ui/                    # Separate; consumes engine
```

Key questions to resolve before coding:
1. Noiseless shots, or BSC(ε) on shots too?
2. Ships touch-allowed or touch-forbidden?
3. BO-with-EI: pure greedy MAP, or UCB-style with exploration bonus?
4. Hard turn cap or no cap?
5. Ask-vs-shoot threshold: fixed (e.g., 0.7) or adaptive (shoot when EIG of best question drops below EI of best shot)?

The last one is actually the most important: **an adaptive ask-vs-shoot threshold is the right theoretical approach**, because it turns the whole decision into a unified information-vs-action value comparison. I'd recommend going straight to that.

Want me to tighten this into a proper spec document, or is this sketch enough to hand to Claude Code?

---

**License:** *This work is licensed under [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/).*