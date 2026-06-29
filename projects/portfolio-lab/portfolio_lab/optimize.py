"""Constrained (long-only) portfolio optimization via numerical methods.

Week 2 (`portfolio.py`) solved the *unconstrained* min-variance and max-Sharpe
portfolios in closed form. Those allow shorting and leverage, so they can return
weights like +255% / −153% — mathematically optimal but not something most people
can actually hold.

This module adds the realistic version: weights that are **long-only** (no
shorting) and **fully invested** (sum to 100%). There's no neat closed form for
that, so we minimize numerically with SciPy's SLSQP solver. It also traces the
true **efficient frontier** — the lowest possible volatility for each target
return — rather than just a cloud of random guesses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import portfolio as pf


def _as_arrays(mean_returns: pd.Series, cov: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(mean_returns, dtype=float), np.asarray(cov, dtype=float)


def _full_investment_constraint() -> dict:
    """The weights must sum to 1 (100% invested)."""
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


def _default_bounds(n: int, allow_short: bool) -> list[tuple[float, float]]:
    return [(-1.0, 1.0) if allow_short else (0.0, 1.0)] * n


def min_variance_long_only(
    cov: pd.DataFrame, allow_short: bool = False
) -> np.ndarray:
    """Minimum-variance weights subject to long-only, fully-invested constraints."""
    _, sigma = _as_arrays(pd.Series(np.zeros(len(cov))), cov)
    n = sigma.shape[0]

    def variance(w: np.ndarray) -> float:
        return float(w @ sigma @ w)

    x0 = np.repeat(1.0 / n, n)
    res = minimize(
        variance,
        x0,
        method="SLSQP",
        bounds=_default_bounds(n, allow_short),
        constraints=[_full_investment_constraint()],
    )
    if not res.success:
        raise RuntimeError(f"Min-variance optimization failed: {res.message}")
    return res.x


def max_sharpe_long_only(
    mean_returns: pd.Series,
    cov: pd.DataFrame,
    risk_free: float = 0.0,
    allow_short: bool = False,
) -> np.ndarray:
    """Maximum-Sharpe weights subject to long-only, fully-invested constraints."""
    mu, sigma = _as_arrays(mean_returns, cov)
    n = len(mu)

    def neg_sharpe(w: np.ndarray) -> float:
        ret = w @ mu
        vol = np.sqrt(w @ sigma @ w)
        if vol == 0:
            return 0.0
        return -(ret - risk_free) / vol

    x0 = np.repeat(1.0 / n, n)
    res = minimize(
        neg_sharpe,
        x0,
        method="SLSQP",
        bounds=_default_bounds(n, allow_short),
        constraints=[_full_investment_constraint()],
    )
    if not res.success:
        raise RuntimeError(f"Max-Sharpe optimization failed: {res.message}")
    return res.x


def efficient_frontier(
    mean_returns: pd.Series,
    cov: pd.DataFrame,
    n_points: int = 50,
    allow_short: bool = False,
) -> pd.DataFrame:
    """Trace the efficient frontier: minimum volatility for each target return.

    Returns a DataFrame with columns `return`, `volatility`, and one weight column
    per asset, sorted by target return. These points form the upper-left edge of
    what's achievable — the classic frontier curve.
    """
    mu, sigma = _as_arrays(mean_returns, cov)
    n = len(mu)
    targets = np.linspace(mu.min(), mu.max(), n_points)

    def variance(w: np.ndarray) -> float:
        return float(w @ sigma @ w)

    rows = []
    x0 = np.repeat(1.0 / n, n)
    for target in targets:
        constraints = [
            _full_investment_constraint(),
            {"type": "eq", "fun": lambda w, t=target: w @ mu - t},
        ]
        res = minimize(
            variance,
            x0,
            method="SLSQP",
            bounds=_default_bounds(n, allow_short),
            constraints=constraints,
        )
        if not res.success:
            continue
        w = res.x
        rows.append((float(w @ mu), float(np.sqrt(w @ sigma @ w)), *w))

    cols = ["return", "volatility", *list(mean_returns.index)]
    frontier = pd.DataFrame(rows, columns=cols)
    return frontier.sort_values("return").reset_index(drop=True)
