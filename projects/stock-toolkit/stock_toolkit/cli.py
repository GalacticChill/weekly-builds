"""Command-line entry point: fetch a ticker, print metrics, save charts."""

from __future__ import annotations

import argparse

from .data import load_prices
from .metrics import summary
from .plots import plot_cumulative_return, plot_price_with_mas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stock data toolkit")
    parser.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--start", default="2020-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (optional)")
    parser.add_argument("--no-charts", action="store_true", help="Skip saving charts")
    args = parser.parse_args(argv)

    df = load_prices(args.ticker, start=args.start, end=args.end)
    close = df["Close"]

    stats = summary(close)
    print(f"\n{args.ticker} ({close.index[0].date()} → {close.index[-1].date()})")
    print(f"  Cumulative return:     {stats['cumulative_return'] * 100:7.2f}%")
    print(f"  Annualized volatility: {stats['annualized_volatility'] * 100:7.2f}%")
    print(f"  Max drawdown:          {stats['max_drawdown'] * 100:7.2f}%")

    if not args.no_charts:
        p1 = plot_price_with_mas(close, args.ticker)
        p2 = plot_cumulative_return(close, args.ticker)
        print(f"\nSaved charts:\n  {p1}\n  {p2}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
