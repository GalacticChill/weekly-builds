# Week of 2026-07-13 — Signal-driven strategies (no lookahead)

**Theme:** finance / quantitative strategy + backtesting discipline
**Builds on:** week 4's `backtester` (iterates on the same project)

## Goal
The week-4 engine rebalanced to *fixed* weights. Add a **strategy layer** that
*chooses* weights from market signals — and enforce the single most important
backtesting rule: **point-in-time evaluation with no lookahead**.

## Scope (shipped)
- `strategies.py` — three signal functions that map return-history → weights:
  - `equal_weight` (1/N baseline)
  - `momentum` (cross-sectional relative strength, long-only, losers zeroed)
  - `inverse_volatility` (risk-parity-lite, weight ∝ 1/vol)
  - plus a `STRATEGIES` name registry for the CLI.
- `engine.run_strategy_backtest` — on each rebalance date, calls the strategy with
  `returns.iloc[:i]` (data strictly before the date). Warmup window before first
  trade. Records `weights_history` (weights chosen over time).
- CLI: `--strategy {equal,momentum,inverse-vol}`, `--lookback`, `--warmup`;
  benchmarks the strategy against a point-in-time equal-weight portfolio.
- 7 new offline tests (17 total), incl. a direct **no-lookahead** guard (a spy
  strategy asserts it never sees data on/after the rebalance date).

## Why this matters for the portfolio story
Lookahead bias is *the* classic backtesting mistake — building the discipline in,
and testing for it explicitly, is a strong maturity signal. Also introduces two
real factor strategies (momentum, risk parity) with honest caveats in the README
(the flattering 2016–2023 tech basket, higher turnover/costs, regime risk).

## Deliberately deferred (future weeks)
- Walk-forward / out-of-sample parameter selection.
- Moving-average timing with a cash position (needs a cash bucket in the engine).
- ML return predictor feeding the weights (ties in data/ML interest).
- **Original personal capstone** + GitHub profile README — still the top-priority
  moves for the college-app story before fall-2027 deadlines.

## Status
Built and shipped 2026-07-13. 17 tests pass; verified end-to-end on a 7-name
basket 2016–2023 (momentum 17.7x / Sharpe 1.46 vs equal-weight 6.6x / Sharpe 1.03,
with turnover rising 5%→36%).
