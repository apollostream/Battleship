---
output:
  pdf_document: default
  html_document: default
---
# EIG and ELLR -- Theory & Battleship Simulator

---

## Prompt 1

> See ["SHOOT FIRST, ASK QUESTIONS LATER? BUILDING RATIONAL AGENTS THAT EXPLORE AND ACT LIKE PEOPLE"](https://arxiv.org/pdf/2510.20886).
> They use expected information gain, EIG(qt | x, H1:t)=EIGε(qt | x, H1:t) = Hb(ε + (1 − 2ε) pt) − Hb(ε), to choose question qt at time t given the partially observed board x and history H1:t of all resulting answers of questions and results of actions. Where Hb is the binary entropy.
> Show mathematically how EIG differs from using the expected log-likelihood ratio log(LR(H=s;E=a˜t |qt, x, H1:t)), where hypothesis H=board s and evidence E=answer to qt is a˜t. And the expectation is taken over all boards s∈S⊢x where S⊢x = { s ∈ S : s is consistent with x }, i.e. all boards consistent with partially observed board x, with probabiity πt(s) = Pr(S = s | x, H1:t).
LR(H=s;E=a˜t |qt, x, H1:t) = Pr(E=a˜t|qt, s, x, H1:t) / Pr(E=a˜t|qt, not(s), x, H1:t).
So log-likelihood ratio is log Pr(E=a˜t|qt, s, x, H1:t) - log Pr(E=a˜t|qt, not(s), x, H1:t).
> And not(s) is all boards other than s.

## Response, Claude Sonnet 4.6

This is a clean information-theoretic question I can answer directly. Here's the full derivation.

---

## Setup and Notation

Fix question $q_t$, board $x$, and history $H_{1:t}$. Write $\pi_s \equiv \pi_t(s)$ for brevity, and let $p_s(\tilde{a}) \equiv \Pr(\tilde{A}_t = \tilde{a} \mid s, q_t)$. Under the BSC($\varepsilon$) noise model,

$$p_s(1) = \varepsilon + (1-2\varepsilon)\,\mathbf{1}\{f_q(s)=1\}, \qquad p_s(0) = 1 - p_s(1).$$

The **marginal** answer distribution (over all boards) is

$$\bar{p}(\tilde{a}) \equiv \Pr(\tilde{A}_t = \tilde{a} \mid q_t, x, H_{1:t}) = \sum_{s'} \pi_{s'}\, p_{s'}(\tilde{a}),$$

and the **leave-$s$-out** marginal (over all boards except $s$) is

$$\bar{p}_{-s}(\tilde{a}) \equiv \Pr(\tilde{A}_t = \tilde{a} \mid q_t, \text{not}(s), x, H_{1:t}) = \frac{\bar{p}(\tilde{a}) - \pi_s\, p_s(\tilde{a})}{1 - \pi_s}.$$

---

## Both Quantities as Weighted KL Divergences

**EIG** is the mutual information $I(S;\tilde{A}_t \mid x, H_{1:t})$, which expands as

$$\boxed{\mathrm{EIG} = \sum_s \pi_s\; D_{\mathrm{KL}}\!\bigl(p_s \;\|\; \bar{p}\bigr)}$$

**Expected log-LR** is, by definition,

$$\mathrm{E}[\log \mathrm{LR}] = \sum_s \pi_s \sum_{\tilde{a}} p_s(\tilde{a})\;\log\frac{p_s(\tilde{a})}{\bar{p}_{-s}(\tilde{a})} = \sum_s \pi_s\; D_{\mathrm{KL}}\!\bigl(p_s \;\|\; \bar{p}_{-s}\bigr).$$

$$\boxed{\mathrm{E}[\log \mathrm{LR}] = \sum_s \pi_s\; D_{\mathrm{KL}}\!\bigl(p_s \;\|\; \bar{p}_{-s}\bigr)}$$

The **only structural difference** is the reference distribution inside the KL divergence: EIG references $\bar{p}$, the mixture over *all* boards; E[log LR] references $\bar{p}_{-s}$, the mixture *leaving out* board $s$ itself.

---

## Exact Relationship

Because $\bar{p}(\tilde{a}) = \pi_s\, p_s(\tilde{a}) + (1-\pi_s)\,\bar{p}_{-s}(\tilde{a})$, the ratio of the two reference distributions is

$$\frac{\bar{p}(\tilde{a})}{\bar{p}_{-s}(\tilde{a})} = (1-\pi_s) + \pi_s\,\underbrace{\frac{p_s(\tilde{a})}{\bar{p}_{-s}(\tilde{a})}}_{\displaystyle\equiv\, \mathrm{LR}_s(\tilde{a})}.$$

Therefore

$$D_{\mathrm{KL}}(p_s \| \bar{p}_{-s}) - D_{\mathrm{KL}}(p_s \| \bar{p}) = \sum_{\tilde{a}} p_s(\tilde{a})\,\log\frac{\bar{p}(\tilde{a})}{\bar{p}_{-s}(\tilde{a})} = \mathbb{E}_{\tilde{a}\sim p_s}\!\left[\log\bigl(1-\pi_s + \pi_s\,\mathrm{LR}_s(\tilde{A})\bigr)\right].$$

Summing over $s$ with weights $\pi_s$:

$$\boxed{\mathrm{E}[\log \mathrm{LR}] - \mathrm{EIG} = \sum_s \pi_s\;\mathbb{E}_{\tilde{a}\sim p_s}\!\left[\log\bigl(1 - \pi_s + \pi_s\,\mathrm{LR}_s(\tilde{A})\bigr)\right].}$$

---

## EIG $\leq$ E[log LR] Always

Since $\mathrm{KL}(P\|\cdot)$ is **convex** in its second argument, and $\bar{p}$ is the convex mixture

$$\bar{p} = \pi_s\, p_s + (1-\pi_s)\,\bar{p}_{-s},$$

we get

$$D_{\mathrm{KL}}(p_s \| \bar{p}) \leq \pi_s\, D_{\mathrm{KL}}(p_s \| p_s) + (1-\pi_s)\, D_{\mathrm{KL}}(p_s \| \bar{p}_{-s}) = (1-\pi_s)\, D_{\mathrm{KL}}(p_s \| \bar{p}_{-s}).$$

Hence, term-by-term and after taking the weighted sum:

$$\mathrm{EIG} \;\leq\; \sum_s \pi_s(1-\pi_s)\, D_{\mathrm{KL}}(p_s \| \bar{p}_{-s}) \;\leq\; \mathrm{E}[\log \mathrm{LR}].$$

EIG is always a **lower bound** on the expected log-LR, with the gap controlled by the self-weight $\pi_s$.

---

## When They Coincide

Taylor-expand the correction term for small $\pi_s$:

$$\log(1 - \pi_s + \pi_s\,\mathrm{LR}_s) = \pi_s(\mathrm{LR}_s - 1) - \tfrac{\pi_s^2}{2}(\mathrm{LR}_s-1)^2 + O(\pi_s^3).$$

Taking the expectation over $\tilde{a} \sim p_s$ and noting $\mathbb{E}_{p_s}[\mathrm{LR}_s] = \sum_{\tilde a} p_s(\tilde a)\,p_s(\tilde a)/\bar p_{-s}(\tilde a) \geq 1$:

$$\mathrm{E}[\log \mathrm{LR}] - \mathrm{EIG} = O\!\left(\sum_s \pi_s^2\right).$$

For a uniform prior over $N$ boards, $\pi_s = 1/N$, so the gap is $O(1/N)$. As $N\to\infty$ (or more generally as the belief $\pi_t$ becomes diffuse), **the two measures coincide**. With the SMC particle approximation used in the paper, this occurs as the number of particles grows.

---

## Intuitive Summary

| | Reference distribution | Board $s$ included? |
|---|---|---|
| **EIG** | $\bar p = \sum_{s'}\pi_{s'} p_{s'}$ | Yes — $s$ inflates its own answer probability |
| **E[log LR]** | $\bar p_{-s} = (\bar p - \pi_s p_s)/(1-\pi_s)$ | No — leave-one-out |

The log-LR measures how "surprising" the observed answer is *relative to every other hypothesis*, making it strictly larger. EIG instead compares each board to the **population-average** answer distribution, which is already partially "contaminated" by that board's own contribution. This self-contamination shrinks the apparent divergence, giving EIG $\leq$ E[log LR], with equality only when no single board dominates the belief (i.e., $\max_s \pi_s \to 0$).

---

## Prompt 2

> given the difference, can an argument be made to prefer either one over the other?

## Response, Claude Sonnet 4.6

This is actually a conceptual question I can address directly from first principles — no web research needed here.

There are principled arguments on both sides, and the right choice depends on what the question-asker cares about.

---

## Arguments for EIG

**1. Coherent Bayesian accounting.** EIG = $I(S;\tilde{A}_t)$ is the *expected reduction in Shannon entropy of the full belief* $\pi_t$. This means it accounts for the agent's **global epistemic state**: a question is good if it compresses the entire hypothesis space, not just if it discriminates one board from the rest. This is exactly what Bayesian Experimental Design (BED) optimizes for, and what the paper formalizes.

**2. Additivity over turns.** Because entropy is additive, EIG has a clean sequential interpretation: total information gained across turns is the sum of per-turn EIGs (modulo correlations from belief updates). This compositionality is very convenient for planning and for comparing strategies across games of different lengths.

**3. Answer-symmetric.** EIG is symmetric in $\tilde{a} \in \{0, 1\}$: a question is maximally informative when $p_t \approx 1/2$, i.e., when "yes" and "no" are equally likely. The log-LR is not symmetric in this way — it depends on which answer is observed, and the expectation over answers introduces asymmetries depending on the prior.

**4. Practical computability.** As shown in the paper (Eq. 4), EIG under BSC($\varepsilon$) has a **closed form** in terms of $p_t$, which is itself just a particle-weight average. This makes it tractable in the SMC approximation. The expected log-LR requires a leave-one-out marginal $\bar{p}_{-s}$ for every particle, which is $O(N)$ more expensive to compute.

**5. Robustness to degenerate beliefs.** When $\pi_s \to 1$ for some board $s$ (near-certainty), the log-LR diverges because $\bar{p}_{-s}$ can be very different from $p_s$, making the ratio extreme. EIG stays bounded in $[0, 1-H_b(\varepsilon)]$ regardless.

---

## Arguments for E[log LR]

**1. Hypothesis-discriminating interpretation.** The log-LR directly measures the **weight of evidence** (in the Bayesian/Peircean sense) that the answer provides *for a specific board $s$ against all alternatives*. If the goal is to confirm or rule out particular hypotheses — as in medical diagnosis or targeted scientific experiments — this is more semantically natural.

**2. Connection to hypothesis testing.** The log-LR is the central quantity in Neyman-Pearson testing and sequential probability ratio tests (SPRT). If the agent's downstream task is a **binary decision** (e.g., "is the ship at this location?") rather than full belief compression, optimizing expected log-LR aligns with minimizing expected type-I/II error more directly than EIG does.

**3. Penalizes overconfident priors less harshly.** Because $\bar{p}_{-s}$ excludes $s$'s self-contribution, it measures discriminability against the *rest of the world* rather than against a mixture that already includes $s$. This means the log-LR does not artificially deflate the value of a question just because the agent already assigns high probability to the correct board — it rewards questions that sharply separate the true board from alternatives even when the prior is concentrated.

**4. Closer to causal/interventional reasoning.** The log-LR can be interpreted as: "if I run this experiment and observe answer $\tilde{a}$, by how much should I update my log-odds on board $s$?" This maps cleanly onto practical scientific reasoning — researchers often think in terms of evidence strength for specific theories, not global entropy reduction.

---

## The Deeper Tension

The two criteria reflect a **fundamental trade-off** in active learning and experimental design:

| | EIG | E[log LR] |
|---|---|---|
| **Goal** | Compress the whole belief | Discriminate specific hypotheses |
| **Framework** | Information theory (Shannon) | Evidence theory (Bayes factors) |
| **Scales like** | $\sum_s \pi_s \log(\pi_s / \bar p)$ | $\sum_s \pi_s \log(\pi_s / \bar p_{-s})$ |
| **Coincide when** | Belief is diffuse ($\pi_s \ll 1$) | — |
| **Preferred when** | Many hypotheses, uncertain prior | Few hypotheses, or decision-focused |

The gap between them, $\sum_s \pi_s \mathbb{E}[\log(1 - \pi_s + \pi_s \mathrm{LR}_s)]$, is largest when the belief is **concentrated** — late in the game after many questions. This is precisely when the two criteria give the most different advice about what question to ask next.

In the Battleship setting specifically, EIG is the right choice for the reasons the paper emphasizes: the hypothesis space is huge, the prior starts diffuse, and the goal is efficient global belief compression. But in a setting like diagnostic medicine — where you have a short list of candidate diseases and a patient must be treated — the expected log-LR (or equivalently, optimizing Bayes factors) is arguably the more principled criterion, because the clinician cares about discrimination, not entropy.

---


## Prompt 3

> is there a scenario that you can represent in code and simulate that clearly illustrates the different path in queries given the same initial setting? Say, a simple board search problem: much smaller board than in paper? If so, before coding it, generate a detailed explanation of the scenario.

## Response, Claude Sonnet 4.6

This is a conceptual design question I can work through from our mathematical analysis — no research needed. Let me lay out the scenario carefully before any code is written.

---

## The Scenario

### Grid and Hypothesis Space

A **4×4 grid** with a single **length-2 ship** (domino). The full hypothesis space has 24 positions (12 horizontal, 12 vertical). After some initial observations — say, confirmed misses at several cells — the belief has been narrowed to exactly **4 remaining hypotheses**:

| ID | Ship position | Cells occupied | Weight $\pi_s$ |
|---|---|---|---|
| s1 | Horizontal, row 0 | (0,0) and (0,1) | **0.45** |
| s2 | Horizontal, row 1 | (1,0) and (1,1) | **0.45** |
| s3 | Vertical, col 2 | (0,2) and (1,2) | 0.05 |
| s4 | Vertical, col 2 | (1,2) and (2,2) | 0.05 |

The asymmetric prior reflects that earlier hits or revealed water tiles made the two horizontal positions in the top-left highly probable, leaving a small residual probability for two vertical positions in column 2. This is the critical state: belief is heavily concentrated on two equally plausible dominant boards, with two low-probability alternatives still in play.

### The Two Candidate Questions

From our analysis, the divergence between EIG and E[log LR] is sharpest when the belief is **concentrated and asymmetric with respect to the question structure**. The most revealing pair of candidate questions here is:

**Q_cell**: *"Is cell (0,1) occupied?"*

- s1 → Yes (cell (0,1) is the right half of the horizontal ship in row 0)
- s2, s3, s4 → No
- Marginal answer probability: $p_t \approx 0.45 \times 0.9 + 0.55 \times 0.1 = 0.46$ (slightly below 0.5)

**Q_row**: *"Is there any ship in row 0?"*

- s1 → Yes (occupies cells in row 0)
- s3 → Yes (has a cell at (0,2))
- s2, s4 → No
- Marginal answer probability: $p_t \approx (0.45 + 0.05) \times 0.9 + 0.50 \times 0.1 = 0.50$ (exactly balanced)

### Why EIG and E[log LR] Disagree Here

**EIG** measures $H_b(\varepsilon + (1-2\varepsilon)p_t) - H_b(\varepsilon)$, which is maximized at $p_t = 0.5$. Q_row achieves exactly this by combining s1 (weight 0.45) and s3 (weight 0.05) into the "Yes" partition, perfectly balancing the expected answer distribution. Q_cell falls just short, with $p_t = 0.46$. So **EIG prefers Q_row**.

**E[log LR]** computes $\sum_s \pi_s \log \frac{p_s(\tilde{a})}{\bar{p}_{-s}(\tilde{a})}$. For Q_cell and board s1:
$$\bar{p}_{-s1}(\text{Yes}) = \frac{\bar{p}(\text{Yes}) - \pi_{s1} \cdot p_{s1}(\text{Yes})}{1 - \pi_{s1}} = \frac{0.46 - 0.45 \times 0.9}{0.55} = \frac{0.055}{0.55} = 0.10$$

So the log-LR for s1 under Q_cell is $\log(0.9/0.10) = \log 9 \approx 2.20$ — nearly perfect discrimination. For Q_row, s3 "dilutes" the Yes partition: $\bar{p}_{-s1}(\text{Yes})$ is now larger (the leave-s1-out marginal is no longer near zero), so the log-LR for s1 drops to about 1.65. The 45% weight on s1 makes this difference decisive: **E[log LR] prefers Q_cell**.

The divergence in numbers (with $\varepsilon = 0.1$):

| Question | $p_t$ | EIG (bits) | E[log LR] |
|---|---|---|---|
| Q_row | 0.50 | **0.531** | 1.548 |
| Q_cell | 0.46 | 0.529 | **1.628** |

The gap in EIG is tiny (0.002 bits). The gap in E[log LR] is meaningful (0.080). They rank oppositely.

### The Structural Reason for Disagreement

Q_row achieves its perfect $p_t = 0.5$ by **bundling s1 with s3** in the Yes partition. This is informationally efficient in a global sense — the answer compresses the hypothesis space maximally. But if the answer is Yes, you still cannot distinguish s1 (the dominant board, $\pi = 0.45$) from s3 ($\pi = 0.05$). The posterior on s1 after a Yes answer is $\approx 0.81$, not 1.0.

Q_cell **perfectly isolates s1**. A Yes answer concentrates posterior mass on s1 at $\approx 0.88$. A No answer elevates s2 to $\approx 0.75$ while redistributing small weight to s3/s4 proportionally. In neither case does a minority board "contaminate" the signal about the dominant hypothesis.

E[log LR] rewards this because it explicitly weights the log-likelihood ratio for each board by that board's prior probability. The 2.20 log-LR for s1 (weight 0.45) dominates the sum, making Q_cell win despite its slightly worse global entropy reduction.

### What the Simulation Should Show

The simulation runs many full identification games from this belief state under each strategy, greedy over one criterion at each step:

- **Performance when true board is s1 or s2** (90% of games): Q_cell-based strategy identifies the board faster, because the first question provides extreme discrimination of the dominant board
- **Performance when true board is s3 or s4** (10% of games): Q_row-based strategy does slightly better here, because the row query gives useful information about the column-2 vertical ships too
- **Overall expected questions to identification**: E[log LR] strategy wins overall despite the rare-case disadvantage, because the 90% dominant case dominates the expectation

The simulation should also show the **belief trajectory** — how posterior entropy evolves step by step under each strategy — which will visually illustrate that EIG compresses entropy uniformly while E[log LR] collapses the dominant hypothesis faster at the cost of slower convergence in the tail.

---

Should I proceed to code this up now?

---

## Prompt 4

> yes. make it clear what candidate questions are available to pick from. i suspect Q_cell for each cell, Q_row for each row and Q_column for each column, right?

## Response, Claude Sonnet 4.6

Yes, exactly right — cell questions (one per cell = 16), row questions (4), column questions (4), giving 24 total. Let me code it up.Here's what the simulator contains and what to look for.

---

### Question catalogue (24 total)

| Type | Count | Example |
|---|---|---|
| **Cell** | 16 | `Cell (0,1)` — "Is (row 0, col 1) occupied?" |
| **Row** | 4 | `Row 0` — "Is any ship in row 0?" |
| **Col** | 4 | `Col 0` — "Is any ship in column 0?" |

Many are uninformative at the initial prior (e.g., `Col 3` — no hypothesis has a ship there). The checkbox "Include uninformative" reveals them; by default only the 11 informative questions are shown.

---

### What the ranking table shows at the initial belief state

**EIG #1 → `Row 0`** (P(Yes) = 0.50 exactly; perfectly balanced partition {H1, H3} vs {H2, H4})

**E[log LR] #1 → `Cell (0,0)`** (and Cell(0,1), Cell(1,0), Cell(1,1) all tie) — these perfectly isolate the dominant hypothesis (H1 or H2, weight 0.45) from all others, yielding a very high leave-one-out KL divergence despite P(Yes) = 0.46 rather than 0.50.

The `Rank Δ` column shows how dramatically the two criteria re-rank each question — `Row 0` will be EIG #1 but ELLR #3 or #4, while the cell questions are EIG #2 but ELLR #1.

---

### How to use it

1. **Click a hypothesis** to designate it as ground truth for interactive play
2. **"Ask EIG-best"** or **"Ask E[log LR]-best"** — the belief updates after a noisy answer, and the question rankings re-sort live
3. **Watch the divergence banner** — it turns yellow the moment the criteria disagree, green when they agree (tends to happen once the belief is already concentrated)
4. **"Simulate 500×4 games"** — runs the full greedy strategy for each criterion on each true hypothesis and reports average questions-to-convergence and accuracy

---

...

...

...

---

## Prompt 14

> based upon theory, is it wrong-headed to think of the ELLR metric as adopting a 0-1 loss function, thus seeking max a posteriori choices vs. the EIG metric as adopting a least-squares loss function, thus seeking a maximum expectation over the posterior? Seems like there's at least an analogous thinking between these.
> As such, it makes sense to me that for an explicitly enumerated set of hypotheses and tractable computation, for Battleship -- in which you're rewarded for hits and not for near misses -- that ELLR is preferred. But if the hypotheses are too many to enumerate all of them -- i.e., high-dimensional; large Battleship board -- that EIG might be preferred. Another way to say is that when there is lots of uncertainty, either due to lack of info or approximate computation or high epsilon noise parameter, then we opt for exploring as guided by EIG; but when the uncertainty becomes better resolved and posterior mass concentrates on few enough hypotheses to be explicitly computable, then we'd switch to exploiting as guided by ELLR.
> Do any of these ideas hold any water? Or have I just taken things too far based on near-similarities that don't hold up under careful theoretical computation?

## Response, Claude Sonnet 4.6

This question is conceptual and draws directly on our prior derivations — I can reason through it directly without research.

Your intuitions are genuinely interesting and some of them hold up well, but others don't quite survive careful theoretical inspection. Let me go through each thread.

---

## Thread 1: ELLR as 0-1 loss / MAP, EIG as squared-error loss / posterior mean

This analogy is *suggestive but not precise*. Here's what's actually happening:

**EIG** optimizes for the question that maximally compresses the posterior — it's agnostic about what you'll *do* with the information. The downstream decision rule is not specified by EIG itself. You could follow EIG question-selection with a MAP decision, a posterior-mean decision, or anything else. EIG is purely an epistemic criterion.

**ELLR** is also purely an epistemic criterion, not a decision criterion. It measures how well the observed answer discriminates the true hypothesis from all alternatives, weighted by prior. It doesn't directly encode a loss function either.

So neither criterion *is* a loss function. However, your analogy points at something real: ELLR is more sensitive to **what the dominant hypothesis is** (because it weights discrimination by $\pi_s$), while EIG is sensitive to **the shape of the whole distribution**. In that sense, ELLR behaves more like a criterion that "cares about identifying the winner," which is spiritually closer to MAP reasoning. EIG behaves more like a criterion that "cares about the whole distribution," which is spiritually closer to posterior-mean reasoning.

The precise connection is: ELLR is the expected **log posterior odds** update — after observing $\tilde{a}$, the log-odds on board $s$ shift by $\log \frac{p_s(\tilde{a})}{\bar{p}_{-s}(\tilde{a})}$, and ELLR is the expectation of this shift over answers and boards. So ELLR is directly the expected gain in **discriminative evidence** in the Bayes-factor sense, which is the right criterion if your goal is to concentrate posterior mass on one hypothesis as fast as possible. That is indeed structurally MAP-adjacent.

---

## Thread 2: Battleship rewards hits not near-misses, so ELLR is preferred

This is the sharpest and most defensible claim you've made, and I think it's essentially correct. The argument runs:

In Battleship the reward structure is **binary per cell**: you either hit or miss, and a miss at distance 1 from a ship is worth exactly the same as a miss at distance 10. This is precisely a 0-1 loss on the move decision. Given that, the right move criterion is MAP (fire at the highest-probability cell), and — as we saw in the paper with MBayes — explicitly marginalizing over the posterior to pick the MAP cell is exactly what works.

The question-selection criterion should then be chosen to *serve* that downstream MAP move decision. ELLR, by rewarding questions that sharply separate the dominant hypothesis from all others, is more directly aligned with driving the posterior toward a state where the MAP cell is reliably correct. EIG, by seeking balanced partitions, can "waste" a question finely splitting probability mass between two hypotheses that both point to the same region of the board — the entropy goes down but the MAP cell doesn't change.

This is a genuine and non-trivial insight. It parallels results in the Bayesian experimental design literature on the distinction between **entropy-reduction** designs (minimize posterior entropy) and **decision-theoretic** designs (minimize expected loss under the posterior given a specific action space). When your action space is discrete and your loss is 0-1, the decision-theoretic design and the entropy-reduction design can diverge, and the decision-theoretic one is preferable.

---

## Thread 3: EIG preferred in high-dimensional / intractable settings; ELLR preferred when posterior is concentrated and tractable

This is where I'd push back most carefully. The argument has two separable parts:

**Tractability:** It's true that ELLR requires computing a leave-one-out marginal $\bar{p}_{-s}$ for every particle, making it $O(N)$ more expensive than EIG. So in high-dimensional settings where you're running SMC with many particles, EIG is cheaper. This is a practical point, not a theoretical one.

**High uncertainty → EIG, concentrated posterior → ELLR:** This is more subtle and I don't think it holds up cleanly. The gap between EIG and ELLR is $\sum_s \pi_s \mathbb{E}[\log(1 - \pi_s + \pi_s \mathrm{LR}_s)]$, which is largest when the posterior is **concentrated** (large $\pi_s$ for some $s$). So ELLR and EIG diverge *most* when the posterior is already concentrated — meaning ELLR is most distinctively useful precisely when you already have a good idea of the answer.

When the posterior is diffuse (early in the game, high uncertainty), EIG $\approx$ ELLR to $O(1/N)$, so the choice barely matters. This somewhat inverts your framing: it's not that you switch *to* ELLR as uncertainty resolves — rather, ELLR is always better in principle for the 0-1 reward setting, but the difference only becomes practically significant as the posterior concentrates.

**The explore/exploit framing:** This is appealing but I'd restate it slightly. EIG is not really an exploration criterion and ELLR is not really an exploitation criterion — both are question-selection criteria that always ask questions (exploring). The exploit step is the *move decision* (firing at a cell), which is separate. What you're actually describing is the transition from asking questions to taking shots, which in the paper is handled by DBayes (the one-step lookahead decision). That's orthogonal to the EIG vs ELLR choice for question selection.

---

## Summary

| Your claim | Verdict |
|---|---|
| ELLR is MAP-adjacent, EIG is posterior-mean-adjacent | Approximately right, but neither is strictly a loss function |
| Battleship's 0-1 reward makes ELLR the better question criterion | Solid — this is a genuine decision-theoretic argument |
| EIG preferred when hypothesis space is too large to enumerate | Partially right as a *tractability* argument; not a theoretical superiority |
| High uncertainty → EIG, concentrated posterior → ELLR | Inverted: the gap between them is largest (ELLR most distinctively useful) when posterior is *already* concentrated |
| Explore with EIG, exploit with ELLR | The exploit step is the move decision, not question selection — so this conflates two separate choices |

The deepest correct insight in your thinking is the second one: given a 0-1 reward structure, ELLR is the more decision-theoretically coherent question criterion. The rest are useful analogies that need some refinement to hold precisely.

---

## Prompt 15

> good explanations -- though in thread 1, i didn't mean to imply that the metrics themselves were loss functions; i meant what you've called out in thread 2, that search guided by ellr aligns more closely when you have a 0-1 loss function than does search with eig. good clarification on thread 3. thanks, you've helped to refine my thinking on this topic of experimental searches & active learning which i've studied for a long time.

## Response, Claude Sonnet 4.6

Thank you — and I understood what you meant in thread 1 on reflection. The rephrasing is cleaner: EIG aligns with searching to minimize a squared-error-like loss on the posterior (reduce uncertainty uniformly), while ELLR aligns with searching to minimize 0-1 loss on the final identification decision. That's a crisp and defensible distinction.

What strikes me most about your framing is that it connects experimental design to the classic statistical question of what your estimator is *for*. The choice of search criterion is really a choice of implicit downstream loss, and the mismatch between criterion and loss is what creates the divergence we've been simulating — EIG wastes question budget achieving balanced partitions that don't move the MAP cell, exactly because it isn't "aware" that the game scores hits not entropy reductions.

The explore/exploit thread is also worth keeping in a refined form: not EIG vs ELLR as explore vs exploit, but rather the *question-asking phase itself* as exploration, and the transition from asking to shooting (DBayes) as the exploit decision. The paper's result that DBayes + ELLR-style move selection (MBayes) outperforms pure EIG-question selection alone is consistent with this — the two mechanisms operate at different levels of the decision hierarchy.

If you ever revisit this in a research context, the natural next question seems to be whether there's a unified criterion that interpolates between EIG and ELLR as a function of posterior concentration — essentially adapting the question-selection criterion dynamically as the game progresses, switching from entropy-reduction to discriminative-evidence mode as mass concentrates. That gap appears to be open in the literature.

---

## Prompt 16

> oh, and what if epsilon swaps p, i.e. low signal-to-noise ratio in answers or conversily epsilon goes to zero: what are impacts on metric choice?

## Response, Claude Sonnet 4.6

This is a clean theoretical question that falls directly out of our earlier derivations — I can work through it directly.

---

## What ε does to both metrics

Recall the BSC(ε) model: the noisy answer $\tilde{A}$ flips the true answer with probability ε. The signal-to-noise ratio of a single answer is essentially $1 - 2\varepsilon$, which runs from 1 (perfect) at ε=0 to 0 (pure noise) at ε=½.

Both metrics depend on ε through the same channel parameter, but they respond differently.

---

## EIG as a function of ε

$$\mathrm{EIG}_\varepsilon(q) = H_b\!\left(\varepsilon + (1-2\varepsilon)\,p_t\right) - H_b(\varepsilon)$$

The two terms behave as follows. $H_b(\varepsilon)$ is the entropy injected by the channel itself — it rises from 0 at ε=0 to 1 at ε=½. The first term $H_b(\varepsilon + (1-2\varepsilon)p_t)$ is the total entropy of the noisy answer. Their difference is the mutual information that survives the channel.

**At ε → 0:** EIG → $H_b(p_t)$, the binary entropy of the noiseless marginal. The metric works exactly as designed — a perfectly balanced question with $p_t = ½$ achieves the maximum of 1 bit.

**At ε → ½:** Both terms approach $H_b(½) = 1$ bit, so EIG → 0 for all questions regardless of $p_t$. The channel is pure noise; no question is worth asking. EIG correctly reflects that no information can be extracted.

**Critically:** the *ranking* of questions by EIG is preserved under ε. Since EIG is a monotone function of $|p_t - ½|$ for any fixed ε (questions closer to $p_t = ½$ always rank higher), and ε doesn't change which question achieves the most balanced partition, **ε scales the absolute EIG values uniformly downward but does not change the rank ordering of questions**. EIG's choice of best question is ε-invariant.

---

## E[log LR] as a function of ε

$$\mathrm{E}[\log\mathrm{LR}] = \sum_s \pi_s\, D_{\mathrm{KL}}(p_s \| \bar{p}_{-s})$$

where $p_s(\tilde{a}) \in \{\varepsilon,\, 1-\varepsilon\}$ depending on whether hypothesis $s$ answers Yes or No to $q$.

**At ε → 0:** $p_s(\tilde{a}) \in \{0, 1\}$ — answers are deterministic. For a hypothesis $s$ that answers Yes, $p_s(\text{Yes}) = 1$ and $p_s(\text{No}) = 0$, so the KL divergence from $\bar{p}_{-s}$ becomes $\log(1/\bar{p}_{-s}(\text{Yes}))$, which can be very large if the leave-one-out marginal assigns low probability to Yes. The metric becomes extremely sensitive to questions that would produce answers inconsistent with all other hypotheses — it strongly rewards questions that *perfectly isolate* individual hypotheses. The discrimination-maximizing character of ELLR is sharpest here.

**At ε → ½:** $p_s(\tilde{a}) → ½$ for all hypotheses and all questions — the noisy answer distribution becomes identical for all $s$, collapsing to Bernoulli(½). Therefore $p_s \to \bar{p}_{-s} \to $ Bernoulli(½) for all $s$, and every KL divergence goes to zero. So E[log LR] → 0 uniformly, just like EIG. But the path to zero is different.

**The key asymmetry:** Unlike EIG, the *ranking* of questions by ELLR **is sensitive to ε**. As ε increases, the contrast between $p_s(\text{Yes}) = 1-\varepsilon$ and $p_s(\text{No}) = \varepsilon$ shrinks. This compresses the differences in KL divergence across questions. Questions that were highly discriminating at low ε (large KL for the dominant hypothesis) shrink faster than questions with moderate discrimination, because the KL between two distributions close to each other in $\{ε, 1-ε\}$ depends on the ratio $\frac{1-\varepsilon}{\varepsilon}$, which is very sensitive to ε. Specifically:

$$D_{\mathrm{KL}}(p_s \| \bar{p}_{-s}) \propto (1-2\varepsilon)^2 \cdot \text{(noiseless discrimination)}$$

to first order in small $(1-2\varepsilon)$, meaning ELLR shrinks as the *square* of signal strength whereas EIG shrinks roughly linearly in mutual information. So **ELLR degrades faster than EIG as ε increases**.

---

## The crossover implication for metric choice

Putting it together:

| Regime | EIG ranking | ELLR ranking | Preferred |
|---|---|---|---|
| ε → 0 (perfect SNR) | stable, picks $p_t = ½$ question | most discriminating, picks dominant-isolating question | **ELLR** — discrimination maximally sharp |
| ε moderate | stable | compressed but still distinct from EIG | **ELLR** still, for 0-1 reward, but gap narrows |
| ε → ½ (pure noise) | all → 0, uniform | all → 0, but faster | **EIG** — degrades more gracefully; ELLR collapses first |

The crossover is real and has a clean interpretation. At low ε, answers are trustworthy, so it makes sense to ask the question that most sharply discriminates your best hypothesis — you'll act on the answer. At high ε, answers are unreliable, so the question ranking should reflect *robustness to noise*, which is better captured by EIG's balanced-partition criterion. A balanced question with $p_t = ½$ is maximally robust to noise because flipping the answer still gives you useful information (just in the other direction), whereas a highly discriminating question with $p_t \approx 1-\varepsilon$ for the dominant hypothesis becomes nearly useless when noise can flip it.

---

## Connection to your earlier insight

This reinforces your thread 2 point in a new way. The 0-1 loss argument for ELLR assumed answers are reliable enough to act on. As ε grows, that assumption weakens: the answer is no longer a clean signal about which hypothesis to bet on, so the MAP-adjacent reasoning behind ELLR loses its force, and the entropy-reduction reasoning behind EIG — which never assumed you'd act immediately on any single answer — becomes relatively more valuable. In the limit of pure noise, the only rational strategy is to ask balanced questions and aggregate many weak signals, exactly what EIG prescribes.

So ε can be read as a *trust parameter*: trust the answers and use ELLR; distrust them and use EIG.

---

