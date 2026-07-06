# Week of 2026-07-06 — Portfolio backtester

**Theme:** finance / time-series simulation
**Builds on:** the portfolio thread — `portfolio-lab` optimizes a *single period*;
this simulates holding an allocation *through time*.

## Goal
Answer the realistic follow-up to optimization: if you actually held a portfolio
and rebalanced it on a schedule, what would you have earned — after weight drift
and transaction costs?

## Scope (shipped)
- `engine.py` — day-by-day simulation: positions grow with returns, drift between
  rebalances, and on each rebalance day the engine trades back to target weights
  and charges a proportional transaction cost on the dollars traded.
- Rebalance rules: W / M / Q / Y / none (buy-and-hold).
- `metrics.py` — CAGR, annualized vol, Sharpe, max drawdown, drawdown path.
- Reports turnover and total cost so the cost drag is explicit.
- Every run compares the rebalancing rule vs. buy-and-hold of the same weights.
- `plots.py` — equity curve and underwater (drawdown) chart.
- CLI + library API; 10 offline known-answer tests.

## Why this matters for the portfolio story
Moves from *what's optimal* to *what actually happens* — introduces time-series
thinking, path dependence, and the real-world friction (costs, turnover) that a
lot of naive "my strategy returns 200%" projects ignore. Honest accounting is
itself a signal of maturity.

## Deliberately deferred (future weeks)
- Signal-driven strategies (momentum, moving-average crossover) on top of the
  engine — the engine already accepts any target weights, so a strategy layer is
  the natural next step.
- Walk-forward / out-of-sample validation.
- An ML return predictor feeding the weights (ties in the data/ML interest).
- **Original personal capstone** still on the roadmap before app deadlines.

## Status
Built and shipped 2026-07-06. 10 tests pass; verified end-to-end on
AAPL/MSFT/GOOG/AMZN 2018-2024 (quarterly rebalancing 3.71x vs buy-and-hold 3.65x,
~0.29% total cost over 23 rebalances).
