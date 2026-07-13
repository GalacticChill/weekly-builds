"""Signal-driven strategies that choose target weights from past returns.

A strategy is just a function that takes the return history available *so far* and
returns a long-only target-weight vector (summing to 1). The engine calls it on
each rebalance date with data strictly *before* that date, so a strategy can never
peek at the future — the single most important discipline in honest backtesting.

Included:

- `equal_weight`      — the humble 1/N baseline that's famously hard to beat.
- `momentum`          — tilt toward assets with the strongest recent return
                        (relative strength / cross-sectional momentum).
- `inverse_volatility`— weight assets by 1/volatility so each contributes similar
                        risk (a simple "risk parity" heuristic).

Each takes a `history` DataFrame of daily returns (one column per asset) and an
optional `lookback` in trading days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def equal_weight(history: pd.DataFrame, lookback: int | None = None) -> np.ndarray:
    """Equal 1/N weight across all assets."""
    n = history.shape[1]
    return np.repeat(1.0 / n, n)


def momentum(history: pd.DataFrame, lookback: int = 126) -> np.ndarray:
    """Weight assets in proportion to positive trailing return.

    Assets with negative momentum get zero weight (long-only). If nothing has
    positive momentum, fall back to equal weight.
    """
    window = history.iloc[-lookback:]
    trailing = (1.0 + window).prod() - 1.0        # cumulative return per asset
    scores = trailing.clip(lower=0.0)
    total = scores.sum()
    if total <= 0:
        return equal_weight(history)
    return (scores / total).to_numpy()


def inverse_volatility(history: pd.DataFrame, lookback: int = 63) -> np.ndarray:
    """Weight assets by the inverse of their recent volatility (risk parity-lite)."""
    window = history.iloc[-lookback:]
    vol = window.std(ddof=0)
    inv = 1.0 / vol.replace(0.0, np.nan)
    inv = inv.fillna(0.0)
    total = inv.sum()
    if total <= 0:
        return equal_weight(history)
    return (inv / total).to_numpy()


# Registry so the CLI (and users) can select a strategy by name.
STRATEGIES = {
    "equal": equal_weight,
    "momentum": momentum,
    "inverse-vol": inverse_volatility,
}
