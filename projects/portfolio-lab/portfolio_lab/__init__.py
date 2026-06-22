"""portfolio-lab: multi-asset portfolio analysis and mean-variance optimization."""

from .data import load_prices
from .portfolio import (
    annualized_cov,
    annualized_mean,
    correlation_matrix,
    daily_returns,
    max_drawdown,
    max_sharpe_weights,
    min_variance_weights,
    normalize_weights,
    portfolio_growth,
    portfolio_performance,
    random_portfolios,
)

__all__ = [
    "load_prices",
    "daily_returns",
    "annualized_mean",
    "annualized_cov",
    "correlation_matrix",
    "normalize_weights",
    "portfolio_performance",
    "min_variance_weights",
    "max_sharpe_weights",
    "random_portfolios",
    "portfolio_growth",
    "max_drawdown",
]
