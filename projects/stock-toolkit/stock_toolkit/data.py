"""Fetch historical price data from Yahoo Finance."""

from __future__ import annotations

import pandas as pd


def load_prices(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Download daily OHLCV data for a ticker and return a clean DataFrame.

    The returned frame has a sorted DatetimeIndex and no missing rows.
    """
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker!r} between {start} and {end}.")

    # yfinance may return MultiIndex columns for a single ticker; flatten them.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna().sort_index()
    df.index = pd.to_datetime(df.index)
    return df
