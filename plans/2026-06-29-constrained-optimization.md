# Week of 2026-06-29 — Long-only constrained optimization

**Theme:** finance / numerical optimization
**Builds on:** week 2's `portfolio-lab` (iterates on the same project)

## Goal
Fix the most visible flaw from week 2: the unconstrained max-Sharpe portfolio
returned +255% / −153% weights (shorting and leverage). Add the realistic version
— **long-only, fully-invested** portfolios — and trace the true efficient frontier.

## Scope (shipped)
- New `optimize.py` module (kept separate from the pure-NumPy `portfolio.py`):
  - `min_variance_long_only` / `max_sharpe_long_only` via SciPy SLSQP, with
    `w ≥ 0` and `Σw = 1` constraints (and an `allow_short` escape hatch).
  - `efficient_frontier` — minimum variance for each target return, the real
    frontier curve rather than a random cloud.
- Frontier chart upgraded: true frontier curve + capital market line overlaid on
  the random cloud, with long-only optima marked.
- CLI now reports long-only optima by default; `--allow-short` reproduces week 2's
  closed-form unconstrained results.
- 6 new offline tests (16 total), including a known-answer case where the
  unconstrained solver shorts but the long-only one binds at the boundary.
- Added `scipy` dependency (first time — installed into miniconda).

## Why this matters for the portfolio story
Shows *iterating on your own work*: spotting a limitation, understanding why it
happens (unconstrained optimization), and engineering the realistic fix. That
narrative reads better than a pile of unrelated one-off projects.

## Deliberately deferred (future weeks)
- Risk-parity and Black-Litterman allocations.
- Backtesting with rebalancing + transaction costs.
- Walk-forward / out-of-sample validation.

## Status
Built and shipped 2026-06-29. 16 tests pass; verified end-to-end on AAPL/MSFT/GOOG
(long-only max-Sharpe = 100% MSFT; `--allow-short` reproduces the +255% MSFT case).
