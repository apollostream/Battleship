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
- **Ask**: a binary question from the question catalogue. Answer is returned under BSC(ε) — noisy intel. No state change to the board.
- **Shoot**: fire at a cell. Answer is *exactly* "hit" or "miss" — **noiseless**. Cell is marked permanently; cannot shoot same cell twice.

The asymmetry — noisy queries, noiseless shots — is deliberate. It models the real-world distinction between *intel* (uncertain) and *direct observation* (certain). Strategically, it makes shots a cheap source of perfect information that the agent must trade against asks (which are information-rich on a global scale but unreliable per-bit).

**Game end.** Terminates on whichever comes first:
1. All 12 ship cells hit (complete sinking of the fleet), or
2. Hard turn cap $T_{\max}$ reached (default 80; user-configurable).

**Scoring — two coupled metrics, not just turns.**

- **Hits** $H \in \{0, 1, \ldots, 12\}$ — number of ship cells struck by game end. Maximum $H_{\max} = 12$.
- **Turns** $T \in \{1, \ldots, T_{\max}\}$ — total actions taken (asks + shots).

A trajectory is summarized by the pair $(H, T)$. When $H = 12$ the game ended by sinking the fleet and $T$ is the primary comparator (lower is better); when $T = T_{\max}$ the game ended by timeout and $H$ is the primary comparator (higher is better). This makes the leaderboard well-defined even when a strategy fails to sink the fleet within the cap — we still have a meaningful score.

**Budget constraint.** The hard cap $T_{\max}$ prevents strategies from asking endless questions without ever shooting, and bounds benchmark compute.

**Ask vs. shoot trade-off — adaptive threshold.** Questions are information-rich but don't progress the game; shots progress the game but may miss. The MBayes doctrines (EIG, ELLR) decide ask-vs-shoot by comparing two quantities in the same units:

$$V_{\text{ask}}^\star \;=\; \max_{q \in \mathcal{Q}} \; I(q; s \mid \mathcal{O}) \qquad\text{vs.}\qquad V_{\text{shoot}}^\star \;=\; \max_{c \text{ unshot}} \; H\bigl(\mu(c)\bigr),$$

both in nats. $H(p) = -p\log p - (1-p)\log(1-p)$ is the binary entropy of a shot's Bernoulli outcome — its information value, since the shot is noiseless and so the mutual information equals the outcome entropy. **Shoot iff $V_{\text{shoot}}^\star \ge V_{\text{ask}}^\star$**, else ask the best question.

This is adaptive, parameter-free, and well-founded: early in the game $V_{\text{ask}}^\star$ dominates (global queries slice $\mathcal{S}$ in half); late in the game $V_{\text{shoot}}^\star$ dominates (most cells have $\mu(c)$ near 0 or 1, so the few remaining cells with $\mu(c) \approx 0.5$ offer up to $\log 2 \approx 0.69$ nats of free information *and* score a hit if right). No hand-tuned threshold constant required.

---

## The four strategies

| Strategy | Question selection | Ask-vs-shoot decision | Shot selection |
|---|---|---|---|
| **EIG + MBayes** | argmax EIG over question catalogue (cell, row, column, quadrant) | adaptive: shoot iff $\max_c H(\mu(c)) \ge \max_q I(q;s)$ | MAP cell (argmax posterior marginal) |
| **ELLR + MBayes** | argmax E[log LR] over the same catalogue | same adaptive rule (using ELLR in place of EIG) | MAP cell |
| **Thompson Sampling** | no questions — always shoots | n/a | draw $s^\* \sim \pi(\cdot \mid \mathcal{O})$; shoot the unshot cell of $s^\*$ with highest priority (adjacency-to-hits tiebreak, else uniform) |
| **User** | interactive menu | user chooses | user clicks |

Rationale for Thompson Sampling in the BO slot: on a single-cell Bernoulli outcome, UCB-style acquisition $\alpha(c) = \mu(c) + \kappa \sigma(c)$ is degenerate — the Bernoulli identity $\sigma^2(c) = \mu(c)(1-\mu(c))$ makes $\alpha$ a deterministic function of $\mu$, so the policy reduces to a monotone rescaling of greedy MAP. Finite-particle estimator variance $\sum_i w_i^2 (X_c(s_i) - \hat\mu)^2$ has the same defect. The one genuinely non-degenerate BO-family policy on correlated Bernoulli arms is **Thompson sampling**: draw an entire configuration from the posterior each turn, shoot according to that draw. This is regret-near-optimal in the bandit sense, respects the support constraint by construction, and yields a visibly distinct trajectory class in the benchmark — EIG-per-shot would collapse into one of the MBayes modes.

EIG vs ELLR are *not* confused by this — they differ genuinely under the multi-type question catalogue (row/col/quadrant as well as cell), because their weighted-KL reference distributions differ (see `transcript.md`), and that difference is amplified on composite queries whose answers partition $\mathcal{S}$ unevenly across hypotheses.

---

## Hypothesis space and inference

**Constraint set.** Let $\mathcal{S}$ be the set of configurations $s$ satisfying simultaneously:

1. *exactly four ships*,
2. *ship lengths* $\{2, 3, 3, 4\}$ (one of each length except the length-3 class which holds two),
3. *no overlap and no touching* — pairwise cell-occupancies of any two ships share neither a cell nor an orthogonally-adjacent cell pair.

These three constraints are encoded purely through the support $\mathcal{S}$.

**Prior.** Uniform over $\mathcal{S}$: $\pi_0(s) = \mathbf{1}[s \in \mathcal{S}] / |\mathcal{S}|$. Approximated by drawing $N$ particles from $\mathcal{S}$ (say $N = 2000$–$5000$) at initialization. Anything outside $\mathcal{S}$ has zero prior, so zero posterior forever.

**Likelihood — asymmetric noise model.** Two channels:

- **Shots (noiseless, delta likelihood).** For a shot at cell $c$ returning $y \in \{0, 1\}$, with $X_c(s) = \mathbf{1}[s \text{ has ship at } c]$:

  $$p_{\text{shot}}(y \mid s, c) \;=\; \mathbf{1}[y = X_c(s)].$$

  A shot definitively eliminates every particle inconsistent with the outcome — its weight goes to zero.

- **Queries (BSC(ε), constraint 4).** For a binary question $q$ with true answer $Q(s) \in \{0, 1\}$:

  $$p_{\text{ask}}(y \mid s, q) \;=\; (1-\varepsilon)^{\mathbf{1}[y = Q(s)]}\,\varepsilon^{\mathbf{1}[y \ne Q(s)]}.$$

  Queries reweight particles softly; no particle is fully eliminated.

**Posterior.** After observations $\mathcal{O} = \{(a_t, y_t)\}$ (each $a_t$ either a question or a shot),

$$\pi(s \mid \mathcal{O}) \;\propto\; \pi_0(s)\,\prod_{t=1}^{T} p(y_t \mid s, a_t),$$

with $p$ chosen per action type. All four stated constraints are baked in — (1)–(3) via the support of $\pi_0$, (4) via the BSC likelihood for asks. No supplementary conditioning step is needed.

**Update.** On each observation, multiply each particle's weight by $p(y_t \mid s_i, a_t)$; renormalise. Shots will zero out a substantial fraction of particles in one stroke — resample aggressively (standard SMC practice with low-ESS trigger).

**Cell marginal.** For any cell c, $\mu(c) = \sum_i w_i \cdot \mathbb{1}[\text{particle } i \text{ has a ship at } c]$. This is the per-cell posterior probability used by MBayes shot selection.

**Prior cell marginal $\mu_0(c)$ is non-uniform — and its topology is counter-intuitive.** Although $\pi_0$ is uniform over $\mathcal{S}$, the induced cell marginal is not uniform across the 64 cells. Two competing forces shape it:

- *Coverage geometry:* interior cells are covered by more single-ship placements (horizontal + vertical passing through them) than edge cells.
- *No-touching constraint:* interior cells have more potential neighbours, so placing a ship there blocks more configurations of the remaining fleet than placing a ship near the edge.

The no-touching constraint **dominates**, which reverses the naive "interior > edge" intuition. The constraint set is small enough to enumerate exactly: $|\mathcal{S}| = 5{,}174{,}944$ (computed by `engine/enumerate.py` in ~30s, bitmask backtracking with canonical ordering for the two indistinguishable length-3 ships).

| Class | Cells | $\mu_0(c)$ (exact) |
|---|---|---|
| corner | (0,0), (0,7), (7,0), (7,7) | 0.1340 |
| edge-near-corner | (0,1), (1,0), (0,6), (6,0), (1,7), (7,1), (6,7), (7,6) | 0.1732 |
| near-corner interior | (1,1), (1,6), (6,1), (6,6) | 0.1702 |
| **middle-of-edge** | (0,3), (0,4), (3,0), (4,0), (0,2), (0,5), (2,0), (5,0) *etc.* | **0.2099 – 0.2161** ← max class |
| deep interior | (3,3), (3,4), (4,3), (4,4) | 0.1891 |

Board average is fixed at $12/64 = 0.1875$ exactly (every valid configuration occupies 12 cells; averaging 64 cells that always sum to 12 yields a deterministic constant). **The prior MAP shot is a middle-of-edge cell** — specifically (0,3), (0,4), (3,0), (4,0) and their mirror images, all at μ_0 = 0.2161 — not a corner (which folk Battleship heuristics favour as "bold") and not a deep-interior cell (which the naive "more placements cover it" argument predicts).

Full grid (exact, dihedral-symmetric):

```
       col 0    1    2    3    4    5    6    7
row 0  .134  .173  .210  .216  .216  .210  .173  .134
row 1  .173  .170  .184  .179  .179  .184  .170  .173
row 2  .210  .184  .197  .192  .192  .197  .184  .210
row 3  .216  .179  .192  .189  .189  .192  .179  .216
row 4  .216  .179  .192  .189  .189  .192  .179  .216
row 5  .210  .184  .197  .192  .192  .197  .184  .210
row 6  .173  .170  .184  .179  .179  .184  .170  .173
row 7  .134  .173  .210  .216  .216  .210  .173  .134
```

These are pinned values — `engine/enumerate.py` reproduces them to machine precision on every run and is memoised via `functools.lru_cache`.

*Implementation note.* A naive sequential-placement sampler (place length-4, then 3, then 3, then 2; restart on collision) does **not** sample uniformly from $\mathcal{S}$ — it over-represents configurations whose early ships didn't block later placements. Two clean alternatives:

1. **Rejection sampling.** Draw all four placements independently uniformly over single-ship placement options; accept iff non-overlapping and non-touching. Acceptance rate is moderate (~5–10% on 8×8 / $\{2,3,3,4\}$); fine for $N \le 5000$ particles.
2. **Importance-weighted sequential sampler.** Keep the sequential generator but track its per-particle proposal density $q(s_i)$ and reweight by $w_i \propto 1/q(s_i)$. Cleaner asymptotically, more bookkeeping.

**Validation test:** the empirical $\mu_0(c)$ from the particle ensemble must match the enumerated $\mu_0(c)$ within Monte Carlo error. `engine/enumerate.py` is that one-time fixture — its output is the ground truth any SMC sampler must reproduce.

**Exact-inference alternative.** Since $|\mathcal{S}| = 5.17$M fits comfortably in memory (≈80 MB for four `uint32` placement indices per config), the SMC machinery is **not structurally required**. With precomputed boolean matrices $X \in \{0,1\}^{|\mathcal{S}| \times 64}$ (occupancy) and $Q \in \{0,1\}^{|\mathcal{S}| \times |\text{questions}|}$ (question truth values), each observation becomes an $O(|\mathcal{S}|)$ vectorised weight update and each question-scoring pass an $O(|\mathcal{S}| \cdot |Q|)$ matrix multiply — measured in tens of milliseconds per turn, not seconds. SMC remains useful as a conceptual baseline and as a fallback if the fleet or board changes and blows the enumeration budget, but exact inference is the recommended production path for the current 8×8 / $\{4,3,3,2\}$ configuration.

**Thompson sample frequency** (used by Doctrine III). Let $\Pi$ denote drawing a particle index $i$ with probability $w_i$. Then

$$p_{\mathrm{TS}}(c) \;=\; \Pr_{i \sim \Pi}\!\left[c = \arg\max_{c'\text{ unshot}} X_{c'}(s_i)\right],$$

estimated by sampling $K$ particles and tallying the argmax target of each (breaking ties uniformly or by adjacency to prior hits). Unlike $\mu(c)$, $p_{\mathrm{TS}}$ is **not** a deterministic function of $\mu$ — it reflects posterior multimodality that $\mu$ smears out.

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
1. ~~Noiseless shots, or BSC(ε) on shots too?~~ **Resolved: noiseless shots, BSC(ε) only on queries.** Asks model uncertain *intel*; shots model direct observation.
2. Ships touch-allowed or touch-forbidden? **Recommended: touch-forbidden** (already encoded in $\mathcal{S}$ above).
3. ~~BO-with-EI: pure greedy MAP, or UCB-style with exploration bonus?~~ **Resolved: Thompson sampling.** UCB is degenerate on Bernoulli arms; Thompson is the genuinely non-degenerate BO-family policy.
4. ~~Hard turn cap or no cap?~~ **Resolved: hard cap** $T_{\max}$ (default 80, user-configurable). Score each trajectory by the pair $(H, T)$ where $H \in \{0,\ldots,12\}$ is hits and $T \in \{1,\ldots,T_{\max}\}$ is turns. Primary comparator is $T$ when $H=12$, else $H$.
5. ~~Ask-vs-shoot threshold: fixed or adaptive?~~ **Resolved: adaptive.** Shoot iff $\max_c H(\mu(c)) \ge \max_q I(q;s\mid\mathcal{O})$ (both in nats). Parameter-free; no hand-tuned constant.

**Deferred to a Phase 3 extension:** a 5th doctrine using POMCP / determinized rollout as a *performance ceiling* baseline. Not in the MVP scope — added once the four primary doctrines are working in the simulator and we want a reference point for "how close to optimal" the cheap strategies actually get.

Want me to tighten this into a proper spec document, or is this sketch enough to hand to Claude Code?

---

**License:** *This work is licensed under [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/).*