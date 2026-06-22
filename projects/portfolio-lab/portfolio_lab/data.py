"""Fetch and align historical prices for several tickers at once."""

from __future__ import annotations

import pandas as pd


def load_prices(
    tickers: list[str], start: str, end: str | None = None
) -> pd.DataFrame:
    """Download adjusted close prices for multiple tickers.

    Returns a DataFrame with one column per ticker, a sorted DatetimeIndex, and
    only the dates for which every ticker has data (inner join), so the columns
    are directly comparable.
    """
    import yfinance as yf

    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False
    )
    if raw.empty:
        raise ValueError(f"No data returned for {tickers} between {start} and {end}.")

    # With multiple tickers yfinance returns MultiIndex columns (field, ticker).
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        # Single ticker: columns are plain fields.
        close = raw[["Close"]]
        close.columns = [tickers[0] if isinstance(tickers, list) else tickers]

    close = close.dropna(how="any").sort_index()
    close.index = pd.to_datetime(close.index)
    # Keep the caller's ticker order where possible.
    if isinstance(tickers, list):
        ordered = [t for t in tickers if t in close.columns]
        close = close[ordered]
    return close
