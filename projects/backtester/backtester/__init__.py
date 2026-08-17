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
from .significance import (
    BootstrapResult,
    CompareResult,
    block_bootstrap_indices,
    bootstrap_metric,
    compare_sharpe,
    sharpe_from_returns,
    sidak_pvalue,
)
from .walkforward import (
    DEFAULT_GRID,
    Comparison,
    FixedParamFit,
    WalkForwardResult,
    best_fixed_param,
    compare,
    walk_forward,
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
    "walk_forward",
    "WalkForwardResult",
    "best_fixed_param",
    "FixedParamFit",
    "compare",
    "Comparison",
    "DEFAULT_GRID",
    "sharpe_from_returns",
    "block_bootstrap_indices",
    "bootstrap_metric",
    "BootstrapResult",
    "compare_sharpe",
    "CompareResult",
    "sidak_pvalue",
]
