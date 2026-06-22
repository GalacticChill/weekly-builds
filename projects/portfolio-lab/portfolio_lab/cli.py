"""Command-line entry point for portfolio-lab."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import portfolio as pf
from .data import load_prices
from .plots import plot_correlation, plot_efficient_frontier, plot_growth


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="portfolio_lab",
        description="Analyze and optimize a multi-asset portfolio.",
    )
    p.add_argument("tickers", nargs="+", help="Ticker symbols, e.g. AAPL MSFT GOOG")
    p.add_argument("--start", default="2021-01-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    p.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Portfolio weights (default: equal weight). Auto-normalized to sum to 1.",
    )
    p.add_argument(
        "--risk-free", type=float, default=0.0, help="Annual risk-free rate, e.g. 0.04"
    )
    p.add_argument(
        "--assets-dir", default="assets", help="Directory to write charts into"
    )
    return p.parse_args(argv)


def _fmt_weights(labels, weights) -> str:
    return ", ".join(f"{t} {w:+.1%}" for t, w in zip(labels, weights))


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    prices = load_prices(args.tickers, start=args.start, end=args.end)
    tickers = list(prices.columns)
    returns = pf.daily_returns(prices)

    mean_ann = pf.annualized_mean(returns)
    cov_ann = pf.annualized_cov(returns)

    if args.weights is None:
        weights = np.repeat(1.0 / len(tickers), len(tickers))
    else:
        if len(args.weights) != len(tickers):
            raise SystemExit(
                f"Got {len(args.weights)} weights for {len(tickers)} tickers."
            )
        weights = pf.normalize_weights(np.array(args.weights))

    ret, vol, sharpe = pf.portfolio_performance(weights, mean_ann, cov_ann, args.risk_free)
    growth = pf.portfolio_growth(returns, weights)
    mdd = pf.max_drawdown(growth)

    span = f"{prices.index[0].date()} → {prices.index[-1].date()}"
    print(f"\nPortfolio: {', '.join(tickers)}   ({span})")
    print(f"  Weights:                 {_fmt_weights(tickers, weights)}")
    print(f"  Annualized return:      {ret:7.2%}")
    print(f"  Annualized volatility:  {vol:7.2%}")
    print(f"  Sharpe (rf={args.risk_free:.0%}):          {sharpe:7.2f}")
    print(f"  Max drawdown:           {mdd:7.2%}")

    mv_w = pf.min_variance_weights(cov_ann)
    ms_w = pf.max_sharpe_weights(mean_ann, cov_ann, args.risk_free)
    mv = pf.portfolio_performance(mv_w, mean_ann, cov_ann, args.risk_free)
    ms = pf.portfolio_performance(ms_w, mean_ann, cov_ann, args.risk_free)

    print("\nOptimal portfolios (long/short allowed):")
    print(f"  Min variance:  ret {mv[0]:6.2%}  vol {mv[1]:6.2%}  sharpe {mv[2]:5.2f}")
    print(f"     weights →   {_fmt_weights(tickers, mv_w)}")
    print(f"  Max Sharpe:    ret {ms[0]:6.2%}  vol {ms[1]:6.2%}  sharpe {ms[2]:5.2f}")
    print(f"     weights →   {_fmt_weights(tickers, ms_w)}")

    assets = Path(args.assets_dir)
    assets.mkdir(parents=True, exist_ok=True)
    g = plot_growth(returns, weights, assets / "growth.png")
    c = plot_correlation(returns, assets / "correlation.png")
    f = plot_efficient_frontier(mean_ann, cov_ann, assets / "frontier.png", args.risk_free)

    print("\nSaved charts:")
    for path in (g, c, f):
        print(f"  {path}")
    print()
