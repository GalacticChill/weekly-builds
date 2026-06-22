"""Portfolio risk/return math: returns, weights, and mean-variance optimization.

Everything here works on plain NumPy/pandas objects so the maths can be tested
offline against series with known answers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple day-over-day returns for each asset (one column per ticker)."""
    return prices.pct_change().dropna(how="any")


def annualized_mean(returns: pd.DataFrame, periods: int = TRADING_DAYS) -> pd.Series:
    """Annualized mean return per asset."""
    return returns.mean() * periods


def annualized_cov(returns: pd.DataFrame, periods: int = TRADING_DAYS) -> pd.DataFrame:
    """Annualized covariance matrix of asset returns."""
    return returns.cov() * periods


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix of asset returns."""
    return returns.corr()


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Scale weights so they sum to 1."""
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if np.isclose(total, 0.0):
        raise ValueError("Weights sum to zero; cannot normalize.")
    return weights / total


def portfolio_performance(
    weights: np.ndarray,
    mean_returns: pd.Series,
    cov: pd.DataFrame,
    risk_free: float = 0.0,
) -> tuple[float, float, float]:
    """Return (annual_return, annual_volatility, sharpe_ratio) for a weight vector.

    `mean_returns` and `cov` are expected to be already annualized.
    """
    w = np.asarray(weights, dtype=float)
    mu = np.asarray(mean_returns, dtype=float)
    sigma = np.asarray(cov, dtype=float)

    ret = float(w @ mu)
    vol = float(np.sqrt(w @ sigma @ w))
    sharpe = (ret - risk_free) / vol if vol > 0 else np.nan
    return ret, vol, sharpe


def min_variance_weights(cov: pd.DataFrame) -> np.ndarray:
    """Global minimum-variance weights (long/short allowed).

    Closed form: w ∝ Σ⁻¹ 1, then normalized to sum to 1.
    """
    sigma = np.asarray(cov, dtype=float)
    ones = np.ones(sigma.shape[0])
    inv_dot_ones = np.linalg.solve(sigma, ones)
    return inv_dot_ones / inv_dot_ones.sum()


def max_sharpe_weights(
    mean_returns: pd.Series,
    cov: pd.DataFrame,
    risk_free: float = 0.0,
) -> np.ndarray:
    """Tangency (maximum-Sharpe) portfolio weights (long/short allowed).

    Closed form: w ∝ Σ⁻¹ (μ − r_f), then normalized to sum to 1.
    """
    mu = np.asarray(mean_returns, dtype=float)
    sigma = np.asarray(cov, dtype=float)
    excess = mu - risk_free
    inv_dot_excess = np.linalg.solve(sigma, excess)
    return inv_dot_excess / inv_dot_excess.sum()


def random_portfolios(
    mean_returns: pd.Series,
    cov: pd.DataFrame,
    n: int = 5000,
    risk_free: float = 0.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Monte-Carlo cloud of long-only portfolios for the efficient-frontier plot.

    Returns a DataFrame with columns return, volatility, sharpe and one column
    per asset weight.
    """
    rng = np.random.default_rng(seed)
    n_assets = len(mean_returns)
    weights = rng.dirichlet(np.ones(n_assets), size=n)

    rows = []
    for w in weights:
        ret, vol, sharpe = portfolio_performance(w, mean_returns, cov, risk_free)
        rows.append((ret, vol, sharpe, *w))

    cols = ["return", "volatility", "sharpe", *list(mean_returns.index)]
    return pd.DataFrame(rows, columns=cols)


def portfolio_growth(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Growth of $1 invested in the (rebalanced) portfolio over the period."""
    w = normalize_weights(weights)
    port_returns = returns.to_numpy() @ w
    growth = np.cumprod(1.0 + port_returns)
    return pd.Series(growth, index=returns.index, name="portfolio")


def max_drawdown(series: pd.Series) -> float:
    """Largest peak-to-trough decline as a negative fraction (e.g. -0.4 == -40%)."""
    running_max = series.cummax()
    drawdown = series / running_max - 1.0
    return float(drawdown.min())
