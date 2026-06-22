# Week of 2026-06-22 — Portfolio Lab (multi-asset analyzer)

**Theme:** finance / quantitative portfolio analysis
**Builds on:** week 1's `stock-toolkit` (single ticker → multi-asset basket)

## Goal
A toolkit that takes a basket of tickers and answers: how risky/rewarding is this
mix, how are the assets correlated, and what allocation is "optimal"?

## Scope (shipped)
- Fetch + date-align adjusted closes for N tickers (`yfinance`).
- Portfolio metrics: annualized return, volatility, Sharpe ratio, max drawdown.
- Correlation matrix.
- Mean-variance optimization (pure NumPy, closed form):
  - global minimum-variance portfolio  `w ∝ Σ⁻¹ 1`
  - maximum-Sharpe / tangency portfolio  `w ∝ Σ⁻¹ (μ − r_f)`
- Monte-Carlo random-portfolio cloud for the efficient-frontier visual.
- Charts: growth of $1, correlation heatmap, efficient frontier.
- CLI + library API, offline known-answer pytest suite.

## Deliberately deferred (good future weeks)
- **Long-only / constrained** optimization (quadratic programming) so the
  max-Sharpe portfolio respects no-shorting — would pull in `scipy` or `cvxpy`.
- Rolling/walk-forward analysis and rebalancing cost modelling.
- Risk-parity and Black-Litterman allocations.

## Status
Built and shipped 2026-06-22. 11 tests pass; verified end-to-end on AAPL/MSFT/GOOG.
