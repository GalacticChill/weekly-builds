# Week of 2026-08-17 — Bootstrap significance (the honesty trilogy, part 3)

**Theme:** finance + statistical rigor — iterating on the backtester.
**Type:** completes a deliberate three-part arc on how a backtest can fool you.

## The arc it completes
1. **No-lookahead** (wk5) — fixes *when* a strategy sees data.
2. **Walk-forward / overfitting tax** (wk8) — fixes *how its parameters are chosen*.
3. **Statistical significance** (this week) — even a clean out-of-sample Sharpe is
   one noisy estimate from a finite sample; is it distinguishable from zero?

## What it does
- **Block bootstrap** of the strategy's daily returns: resample in short contiguous
  blocks (default 10 days) to preserve autocorrelation / volatility clustering,
  recompute the Sharpe thousands of times, and read off a confidence interval.
- **p-value** for "true Sharpe > 0" via a separate bootstrap on the *mean-centered*
  returns (imposes the null of no edge), bias-corrected `(#>=obs + 1)/(B + 1)`.
- **Paired comparison** vs a benchmark: resample both return series with the *same*
  block indices each draw, so common market shocks stay aligned; CI + p-value for
  the Sharpe difference.
- **Multiple-testing correction**: Sidak `1-(1-p)^N` for having tried N strategies —
  the statistical echo of the overfitting tax.

## Scope (shipped)
- `significance.py` — `sharpe_from_returns`, `block_bootstrap_indices` (circular),
  `bootstrap_metric` -> `BootstrapResult` (CI, p, `.significant`), `compare_sharpe`
  -> `CompareResult`, `sidak_pvalue`.
- `plots.py` — `plot_bootstrap` (distribution histogram with observed line, shaded
  CI, and the zero "no edge" line).
- CLI — `--significance` with `--n-boot`, `--block`, `--trials`.
- 10 new offline tests (34 total): real edge is significant, pure noise is not,
  determinism under seed, block=1 == iid, paired comparison detects a true
  outperformer and clears twins, contiguous-block guard, and Sidak properties.

## Findings (real 9-name basket, 2012–2024, momentum)
- **Vs zero:** Sharpe +1.51, 95% CI [+0.99, +2.04], p = 0.0005 — solidly
  significant, and still significant (p = 0.0015) after a Sidak correction for 3
  strategies.
- **Vs equal-weight:** Sharpe difference only +0.29, CI [+0.00, +0.58], p = 0.025 —
  real but marginal. The honest reading: most of momentum's skill is just *being in
  the market*; the incremental skill over naive 1/N is modest. A backtest that only
  printed "Sharpe 1.5" would hide exactly that.

## Why this matters for the portfolio story
Turns the backtester into a genuinely rigorous, self-skeptical framework and closes
a memorable narrative — "here is every way a backtest can lie to you, and a tool to
defend against each." Sustained depth on one project, which reads as real
intellectual engagement. Pure numpy (bootstrap by hand); no new dependencies.

## Deliberately deferred / next
- Could graduate the backtester into its own standalone repo (now deep enough to be
  a strong second pinned repo, like star-signals).
- k-selection diagnostic / embedding view for market-structure.
- Another original capstone remains the highest-value college-app move — ask the
  user what they care about when starting one.

## Status
Built and shipped 2026-08-17. 34 tests pass; verified end-to-end on the 9-name
basket and on synthetic edge/noise fixtures. Bootstrap chart generated.
