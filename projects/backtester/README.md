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

## Signal-driven strategies

Instead of fixed weights, you can let a **strategy** pick the weights from market
signals, re-deciding on every rebalance date. Crucially, the engine only ever
shows a strategy the data from *before* the rebalance date — so there's **no
lookahead bias**, the mistake that makes most amateur backtests look brilliant and
then fail live.

```bash
python -m backtester AAPL MSFT GOOG AMZN NVDA JPM XOM \
    --strategy momentum --lookback 126 --rebalance M --cost 0.001 --risk-free 0.04
```

```
momentum (M)  (rebalance=M)
  Final value (start 1.0):  17.687
  CAGR:                    47.48%
  Sharpe (rf=4%):             1.46
  Max drawdown:           -31.72%
  Rebalances / avg turnover: 88  /  36.4%
  Total transaction cost:  0.1723

Equal-weight (M)  ...
  CAGR:                    29.09%
  Sharpe (rf=4%):             1.03
```

Built-in strategies (`--strategy`):

- **`equal`** — the 1/N baseline that's famously hard to beat.
- **`momentum`** — tilt toward assets with the strongest trailing return
  (cross-sectional relative strength); losers get zero weight.
- **`inverse-vol`** — weight by 1/volatility so each asset contributes similar
  risk (a simple risk-parity heuristic).

Each strategy run is benchmarked against a point-in-time equal-weight portfolio.

> **Read the results honestly.** The momentum result above looks spectacular partly
> because a tech-heavy basket over 2016–2023 is about the friendliest possible
> environment for momentum — and note the turnover jumps from ~5% to ~36%, so it
> only wins *after* paying much higher costs. Momentum also suffers sharp
> "crashes" in other regimes. A backtest is a hypothesis, not a promise; the point
> of the no-lookahead discipline is to make the hypothesis a fair one.

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

For a **strategy** run, the same loop applies, but the target weights are
recomputed on each rebalance date from `returns` up to (but not including) that
day. The simulation starts only once `--warmup` days of history are available, so
the signal has something to work with.

## What's inside

- `data.py` — fetch and date-align adjusted closes for many tickers (`yfinance`)
- `engine.py` — the day-by-day simulation with drift, rebalancing, and costs;
  `run_backtest` (fixed weights) and `run_strategy_backtest` (point-in-time)
- `strategies.py` — equal-weight, momentum, and inverse-volatility signals
- `metrics.py` — CAGR, annualized volatility, Sharpe, max drawdown, drawdown path
- `plots.py` — equity-curve and underwater charts
- `cli.py` — the command-line entry point

## Use it as a library

```python
from backtester import load_prices, daily_returns, run_strategy_backtest, momentum, summary

prices = load_prices(["SPY", "QQQ", "TLT", "GLD"], start="2010-01-01")
r = daily_returns(prices)
res = run_strategy_backtest(r, momentum, rebalance="M", cost=0.001, warmup=126)
print(summary(res.equity, risk_free=0.02))
print(res.weights_history.tail())  # the weights the strategy actually chose
```

## Tests

```bash
python -m pytest
```

All 17 tests run offline against synthetic returns with hand-checked answers —
for example, a 100%-in-one-asset backtest must equal that asset's compounded
returns and incur zero cost, and a curve that doubles over exactly 252 trading
days must report a CAGR of 100%. One test guards the no-lookahead rule directly: a
"spy" strategy records the last date it was shown on each call and the test
asserts every one lies strictly before the rebalance date it fed.
