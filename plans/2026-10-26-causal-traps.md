# Week of 2026-10-26 — Causal Traps (causal inference)

**Theme:** data/statistics — the portfolio's first causal-inference project.
**Type:** new technique (causal DAGs, confounding/collider/mediator, adjustment);
user explicitly asked for "the fresh capstone."

## The question
"Correlation isn't causation" — but the usual fix, "control for everything," is
wrong. When does adjusting for a variable remove bias, and when does it create it?

## Scope (shipped)
- `dgp.py` — three canonical structures with KNOWN true effects (continuous T so the
  effect is a regression coefficient): confounder (C->T, C->Y), collider (T->Z<-Y),
  mediator (T->M->Y + direct). Each returns data + true_effect + adjusting_is_right.
- `estimate.py` — small numpy OLS (with classical SEs); `effect(df, adjust_for=)`
  returns the T coefficient naive or adjusted.
- `experiment.py` — Monte-Carlo each scenario (default 300 sims) → naive/adjusted
  means vs truth, so the bias is systematic not one-draw luck.
- `plots.py` — DAG diagrams, bias-summary bars (true/naive/adjusted with "correct"
  marker), and the collider-illusion scatter (independent overall, spurious within a
  high-Z stratum).
- CLI; no external data (pure simulation, known ground truth). 8 offline tests.

## Findings (300 sims each, n=4000)
- **Confounder:** true 0.50, naive 0.99 (≈2x biased), adjusted 0.50 → ADJUST.
- **Collider:** true 0.00, naive 0.00 (correct), adjusted -0.39 (spurious) → DON'T.
- **Mediator:** true total 0.62, naive 0.62 (total), adjusted 0.20 (direct) →
  depends what you're asking.

## The honest core
- You CANNOT distinguish these from the data alone — same correlation structure,
  opposite correct handling; the arrows come from domain knowledge, not the numbers.
- "Control for everything" is actively dangerous (will eventually include a collider
  or mediator). Cartoons, not a turnkey pipeline (real problems mix roles, have
  unmeasured confounders, use instruments/natural experiments).

## Why this matters for the portfolio story
First causal-inference project — a sophisticated, widely-misunderstood technique and
the deepest expression yet of the portfolio's "don't fool yourself" thread (the
collider/"control for everything" trap is genuinely counterintuitive and
essay-ready). Quantifiable via known ground truth. Pure numpy/pandas/matplotlib, no
new deps.

## Deliberately deferred / next
- Difference-in-differences / instrumental variables / a real natural experiment;
  unmeasured-confounder sensitivity analysis.
- GRADUATE one of the ungraduated projects (confirm first).

## Status
Built and shipped 2026-10-26. 8 tests pass; verified end-to-end (300-sim Monte
Carlo per scenario). Three charts.
