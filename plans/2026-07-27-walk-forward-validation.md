# Week of 2026-07-27 — Walk-forward validation

**Theme:** finance + methodological rigor — iterating on the backtester.
**Type:** the natural discipline step after no-lookahead; carries last week's
out-of-sample lesson from the ML capstone back into the trading strategies.

## The gap it closes
No-lookahead (week 5) fixes *when* a strategy sees data. But choosing a strategy's
`lookback` by trying several over the whole history and reporting the best is a
subtler cheat: you searched for a number, so part of the reported Sharpe is luck,
not skill. Walk-forward validation removes that hindsight.

## What it does
- Slide a rolling **train** window; in each, score every candidate lookback and
  pick the best — using only that window's data.
- Apply the choice to the **test** window that immediately follows (never seen by
  the selection); record those out-of-sample returns.
- Test windows tile the timeline with no gaps/overlap, so stitching gives one
  continuous, honest OOS equity curve.
- Contrast three curves over the same OOS span: honest walk-forward, the
  in-sample-optimized param (chosen with full-sample hindsight), and equal-weight.
- Headline number: the **overfitting tax** = claimed (full-sample) Sharpe minus
  honest walk-forward OOS Sharpe.

## Scope (shipped)
- `walkforward.py` — `walk_forward` (rolling selection + stitched OOS),
  `best_fixed_param` (the tempting hindsight benchmark), `compare` (three-curve
  Comparison with `.overfitting_tax`). All reuse the engine's point-in-time
  simulation; a shared `_segment_returns` runs any strategy over an index range
  using only prior history.
- `plots.py` — `plot_curves` (comparison overlay) and `plot_param_choices` (step
  plot of the lookback picked each fold — instability is itself a finding).
- CLI — `--walk-forward` with `--grid`, `--train`, `--test`.
- 7 new offline tests (24 total): a no-future-leakage guard on `_segment_returns`,
  tiling/shape invariants, and the crux — **overfitting tax is positive on pure
  noise**, while a **genuine engineered edge still survives OOS**.

## Findings
- **Pure noise:** in-sample search brags Sharpe ~0.79; honest walk-forward ~0.13.
  Tax ~0.66. The "best" lookback lurches 252->126->63->21 across folds — the tell.
- **Real 9-name basket, 2012–2024:** claimed Sharpe 1.66 vs honest 1.54 (tax only
  0.13), and walk-forward still beats equal-weight (1.21). Here the lookback is
  fairly stable, so honesty costs little and the edge is largely real — an honest
  result in the other direction, and a good foil to the noise case.

## Why this matters for the portfolio story
Shows the discipline most amateur backtests skip and demonstrates intellectual
honesty as a *habit*, not a one-off: the same out-of-sample principle from the ML
capstone, now closing the loop on the finance track. Pure numpy/pandas; no new
dependencies.

## Deliberately deferred / next
- Graduate star-signals into its own standalone repo (still the strongest profile
  move).
- Richer selection metric (e.g. penalize turnover) or nested walk-forward.
- A README-length / CI-presence enrichment pass on star-signals.

## Status
Built and shipped 2026-07-27. 24 tests pass; verified end-to-end on a fresh
9-name sample and on synthetic noise/edge fixtures. Two charts generated.
