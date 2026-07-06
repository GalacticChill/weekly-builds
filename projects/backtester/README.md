# Backtester

A small, honest portfolio **backtester**. Give it a basket of tickers and target
weights, and it simulates actually *holding* that portfolio through history —
rebalancing on a schedule and paying transaction costs on every trade — then
reports the performance you'd really have earned.

Where [`portfolio-lab`](../portfolio-lab) finds the best allocation for a single
period, this project asks the follow-up question: **what happens when you live
with an allocation over time?** Weights drift with the market, rebalancing pulls
them back, and trading isn't free.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m backtester AAPL MSFT GOOG AMZN --start 2018-01-01 --end 2024-01-01 --rebalance Q --cost 0.001 --risk-free 0.04
```

Output:

```
Backtest: AAPL, MSFT, GOOG, AMZN   (2018-01-02 → 2023-12-29)
  Target weights: AAPL 25%, MSFT 25%, GOOG 25%, AMZN 25%
  Transaction cost: 0.10% per trade

Rebalanced (Q)  (rebalance=Q)
  Final value (start 1.0):   3.713
  Total return:           268.19%
  CAGR:                    24.35%
  Annualized volatility:   28.12%
  Sharpe (rf=4%):             0.72
  Max drawdown:           -39.30%
  Rebalances / avg turnover: 23  /  5.8%
  Total transaction cost:  0.0029

Buy & hold  (rebalance=none)
  Final value (start 1.0):   3.654
  ...
```

Every run compares your rebalancing rule against simply buying and holding the
same weights, so you can see whether rebalancing actually earned its costs.

Options:

- `--rebalance` — `W`, `M`, `Q`, `Y`, or `none` (buy-and-hold)
- `--cost` — proportional cost per dollar traded (`0.001` = 10 bps)
- `--weights` — custom target weights (default: equal weight)

## Charts

| Equity curve | Drawdown (underwater) |
|---|---|
| ![equity](assets/equity.png) | ![drawdown](assets/drawdown.png) |

## How the simulation works

Each trading day:

1. Every asset's dollar position grows by that day's return.
2. On a **rebalance day**, the engine computes the trades needed to return to the
   target weights, charges `cost × dollars_traded`, and resets the positions to
   the target on the remaining capital.

Between rebalances the weights **drift** with the market — exactly like a real
buy-and-mostly-hold portfolio. This drift-and-correct loop, plus honest cost
accounting, is what separates a backtest from just multiplying returns together.

**Turnover** (the fraction of the portfolio traded at each rebalance) and the
**total cost** are reported so you can see the drag directly. Rebalancing more
often controls risk but trades more; the tool lets you weigh that trade-off.

## What's inside

- `data.py` — fetch and date-align adjusted closes for many tickers (`yfinance`)
- `engine.py` — the day-by-day simulation with drift, rebalancing, and costs
- `metrics.py` — CAGR, annualized volatility, Sharpe, max drawdown, drawdown path
- `plots.py` — equity-curve and underwater charts
- `cli.py` — the command-line entry point

## Use it as a library

```python
from backtester import load_prices, daily_returns, run_backtest, summary

prices = load_prices(["SPY", "TLT"], start="2010-01-01")
r = daily_returns(prices)
res = run_backtest(r, [0.6, 0.4], rebalance="Q", cost=0.001)
print(summary(res.equity, risk_free=0.02))
print(f"Paid {res.total_cost:.4f} in costs over {res.n_rebalances} rebalances")
```

## Tests

```bash
python -m pytest
```

All 10 tests run offline against synthetic returns with hand-checked answers —
for example, a 100%-in-one-asset backtest must equal that asset's compounded
returns and incur zero cost, and a curve that doubles over exactly 252 trading
days must report a CAGR of 100%.
