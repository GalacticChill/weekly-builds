"""Ordinary least squares, and the one number we care about: the effect of T.

Small enough to read: fit Y on an intercept, the treatment T, and optionally the
third variable, and return the coefficient on T. The whole study is the contrast
between that coefficient *with* and *without* the adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Fit:
    names: list[str]
    coef: np.ndarray
    se: np.ndarray

    def of(self, name: str) -> float:
        return float(self.coef[self.names.index(name)])

    def se_of(self, name: str) -> float:
        return float(self.se[self.names.index(name)])


def ols(y: np.ndarray, X: np.ndarray, names: list[str]) -> Fit:
    """Plain OLS with classical standard errors. X already includes any intercept."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(n - k, 1)
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    return Fit(names=names, coef=beta, se=se)


def _design(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, list[str]]:
    names = ["intercept"] + cols
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy() for c in cols])
    return X, names


def effect(df: pd.DataFrame, adjust_for: str | None = None) -> float:
    """The estimated causal effect of T on Y (the coefficient on T).

    With `adjust_for=None` this is the naive estimate; pass the third variable's name
    to adjust (control) for it.
    """
    cols = ["T"] + ([adjust_for] if adjust_for else [])
    X, names = _design(df, cols)
    return ols(df["Y"].to_numpy(), X, names).of("T")


def effect_fit(df: pd.DataFrame, adjust_for: str | None = None) -> Fit:
    """Like `effect`, but return the full fit (for standard errors)."""
    cols = ["T"] + ([adjust_for] if adjust_for else [])
    X, names = _design(df, cols)
    return ols(df["Y"].to_numpy(), X, names)
