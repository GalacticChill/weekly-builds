"""Risk and return metrics computed from a price series."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(prices: pd.Series) -> pd.Series:
    """Simple day-over-day percentage returns."""
    return prices.pct_change().dropna()


def cumulative_return(prices: pd.Series) -> float:
    """Total return over the whole series, e.g. 0.25 == +25%."""
    return float(prices.iloc[-1] / prices.iloc[0] - 1.0)


def annualized_volatility(prices: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Annualized standard deviation of daily returns."""
    return float(daily_returns(prices).std(ddof=0) * np.sqrt(periods))


def moving_average(prices: pd.Series, window: int) -> pd.Series:
    """Simple moving average over `window` periods."""
    return prices.rolling(window=window).mean()


def max_drawdown(prices: pd.Series) -> float:
    """Largest peak-to-trough decline, as a negative fraction (e.g. -0.4 == -40%)."""
    running_max = prices.cummax()
    drawdown = prices / running_max - 1.0
    return float(drawdown.min())


def summary(prices: pd.Series) -> dict[str, float]:
    """A compact dict of the headline metrics for a price series."""
    return {
        "cumulative_return": cumulative_return(prices),
        "annualized_volatility": annualized_volatility(prices),
        "max_drawdown": max_drawdown(prices),
    }
