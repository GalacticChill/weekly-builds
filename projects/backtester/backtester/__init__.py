"""backtester: simulate a rebalanced portfolio through time, with costs."""

from .data import daily_returns, load_prices
from .engine import BacktestResult, run_backtest, run_strategy_backtest
from .metrics import (
    annualized_volatility,
    cagr,
    drawdown_series,
    max_drawdown,
    sharpe_ratio,
    summary,
)
from .strategies import (
    STRATEGIES,
    equal_weight,
    inverse_volatility,
    momentum,
)

__all__ = [
    "load_prices",
    "daily_returns",
    "run_backtest",
    "run_strategy_backtest",
    "BacktestResult",
    "cagr",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "drawdown_series",
    "summary",
    "STRATEGIES",
    "equal_weight",
    "momentum",
    "inverse_volatility",
]
