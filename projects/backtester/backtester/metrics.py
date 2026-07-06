"""Performance metrics computed from a backtest equity curve."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(equity: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Compound annual growth rate implied by the equity curve.

    Uses the number of steps (len − 1) as the elapsed time in trading days.
    """
    steps = len(equity) - 1
    if steps <= 0:
        return 0.0
    years = steps / periods
    growth = equity.iloc[-1] / equity.iloc[0]
    return float(growth ** (1.0 / years) - 1.0)


def annualized_volatility(equity: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Annualized standard deviation of the equity curve's daily returns."""
    daily = equity.pct_change().dropna()
    return float(daily.std(ddof=0) * np.sqrt(periods))


def sharpe_ratio(
    equity: pd.Series, risk_free: float = 0.0, periods: int = TRADING_DAYS
) -> float:
    """Annualized Sharpe ratio of the equity curve (excess CAGR over volatility)."""
    vol = annualized_volatility(equity, periods)
    if vol == 0:
        return float("nan")
    return (cagr(equity, periods) - risk_free) / vol


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline as a negative fraction (e.g. -0.4 == -40%)."""
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    """The full drawdown path, for the underwater plot."""
    running_max = equity.cummax()
    return equity / running_max - 1.0


def summary(equity: pd.Series, risk_free: float = 0.0) -> dict[str, float]:
    """Headline metrics for an equity curve, as a compact dict."""
    return {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "cagr": cagr(equity),
        "annualized_volatility": annualized_volatility(equity),
        "sharpe": sharpe_ratio(equity, risk_free),
        "max_drawdown": max_drawdown(equity),
    }
