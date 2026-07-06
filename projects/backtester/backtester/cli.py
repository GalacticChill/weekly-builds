"""Command-line entry point for the backtester."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import metrics
from .data import daily_returns, load_prices
from .engine import run_backtest
from .plots import plot_drawdown, plot_equity


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="backtester",
        description="Backtest a fixed-weight, periodically-rebalanced portfolio.",
    )
    p.add_argument("tickers", nargs="+", help="Ticker symbols, e.g. AAPL MSFT GOOG")
    p.add_argument("--start", default="2018-01-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    p.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Target weights (default: equal weight). Auto-normalized to sum to 1.",
    )
    p.add_argument(
        "--rebalance",
        default="M",
        help="Rebalance frequency: W, M, Q, Y, or none (buy-and-hold). Default M.",
    )
    p.add_argument(
        "--cost",
        type=float,
        default=0.001,
        help="Proportional transaction cost per dollar traded (0.001 = 10 bps).",
    )
    p.add_argument(
        "--risk-free", type=float, default=0.0, help="Annual risk-free rate, e.g. 0.04"
    )
    p.add_argument(
        "--assets-dir", default="assets", help="Directory to write charts into"
    )
    return p.parse_args(argv)


def _print_summary(label: str, res, risk_free: float) -> None:
    s = metrics.summary(res.equity, risk_free)
    print(f"\n{label}  (rebalance={res.rebalance})")
    print(f"  Final value (start 1.0): {res.equity.iloc[-1]:7.3f}")
    print(f"  Total return:           {s['total_return']:7.2%}")
    print(f"  CAGR:                   {s['cagr']:7.2%}")
    print(f"  Annualized volatility:  {s['annualized_volatility']:7.2%}")
    print(f"  Sharpe (rf={risk_free:.0%}):          {s['sharpe']:7.2f}")
    print(f"  Max drawdown:           {s['max_drawdown']:7.2%}")
    print(f"  Rebalances / avg turnover: {res.n_rebalances}  /  {res.avg_turnover:.1%}")
    print(f"  Total transaction cost: {res.total_cost:7.4f}")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    prices = load_prices(args.tickers, start=args.start, end=args.end)
    tickers = list(prices.columns)
    returns = daily_returns(prices)

    if args.weights is None:
        weights = np.repeat(1.0 / len(tickers), len(tickers))
    else:
        if len(args.weights) != len(tickers):
            raise SystemExit(
                f"Got {len(args.weights)} weights for {len(tickers)} tickers."
            )
        weights = np.array(args.weights)

    # The strategy under test vs. the same weights simply bought and held.
    strategy = run_backtest(returns, weights, rebalance=args.rebalance, cost=args.cost)
    hold = run_backtest(returns, weights, rebalance="none", cost=args.cost)

    span = f"{prices.index[0].date()} → {prices.index[-1].date()}"
    wtxt = ", ".join(f"{t} {w:.0%}" for t, w in zip(tickers, strategy.weights))
    print(f"\nBacktest: {', '.join(tickers)}   ({span})")
    print(f"  Target weights: {wtxt}")
    print(f"  Transaction cost: {args.cost:.2%} per trade")

    _print_summary(f"Rebalanced ({args.rebalance})", strategy, args.risk_free)
    _print_summary("Buy & hold", hold, args.risk_free)

    assets = Path(args.assets_dir)
    assets.mkdir(parents=True, exist_ok=True)
    results = {f"Rebalanced ({args.rebalance})": strategy, "Buy & hold": hold}
    eq = plot_equity(results, assets / "equity.png")
    dd = plot_drawdown(results, assets / "drawdown.png")

    print("\nSaved charts:")
    for path in (eq, dd):
        print(f"  {path}")
    print()
