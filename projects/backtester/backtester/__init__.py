"""backtester: simulate a rebalanced portfolio through time, with costs."""

from .data import daily_returns, load_prices
from .engine import BacktestResult, run_backtest
from .metrics import (
    annualized_volatility,
    cagr,
    drawdown_series,
    max_drawdown,
    sharpe_ratio,
    summary,
)

__all__ = [
    "load_prices",
    "daily_returns",
    "run_backtest",
    "BacktestResult",
    "cagr",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "drawdown_series",
    "summary",
]
