"""Offline tests for walk-forward validation — synthetic data, no network.

The headline test is `test_overfitting_tax_is_positive_on_noise`: on returns with
*no* real signal, an in-sample-optimized backtest manufactures a positive Sharpe
that walk-forward refuses to reproduce out of sample. That gap is the whole point
of the module, so we assert it directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtester import metrics
from backtester.strategies import momentum
from backtester.walkforward import (
    _segment_returns,
    best_fixed_param,
    compare,
    walk_forward,
)

GRID = (21, 63, 126, 252)


def _business_index(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2015-01-01", periods=n)


def _noise_returns(n=1400, k=4, seed=0, scale=0.01) -> pd.DataFrame:
    """Pure-noise daily returns: no asset has any persistent edge."""
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, scale, size=(n, k))
    cols = [f"A{i}" for i in range(k)]
    return pd.DataFrame(data, index=_business_index(n), columns=cols)


def _trending_returns(n=1400, k=4, seed=1, scale=0.01) -> pd.DataFrame:
    """One asset has a genuine, persistent positive drift the others lack."""
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, scale, size=(n, k))
    data[:, 0] += 0.0025     # a real, steady edge for asset 0, well above the noise
    cols = [f"A{i}" for i in range(k)]
    return pd.DataFrame(data, index=_business_index(n), columns=cols)


# --------------------------------------------------------------------------- #
# No lookahead: the strategy must never see data at or beyond the current day. #
# --------------------------------------------------------------------------- #

def test_segment_never_sees_future():
    returns = _noise_returns(n=800)
    start, end = 300, 500
    seen_max = {"i": -1}

    def spy(history: pd.DataFrame, lookback: int = 63) -> np.ndarray:
        # Record the last row index of history the strategy was handed.
        if len(history):
            seen_max["i"] = max(seen_max["i"], len(history) - 1)
        n = history.shape[1]
        return np.repeat(1.0 / n, n)

    _segment_returns(returns, spy, start, end, rebalance="M", cost=0.001)
    # The most recent history row index must be strictly below the last test day.
    assert seen_max["i"] < end - 1


# --------------------------------------------------------------------------- #
# Structural invariants.                                                       #
# --------------------------------------------------------------------------- #

def test_choices_and_equity_shapes():
    returns = _trending_returns(n=1400)
    wf = walk_forward(returns, momentum, GRID, train=504, test=126)
    # One row per fold, and every chosen parameter came from the grid.
    assert len(wf.choices) == wf.folds
    assert set(wf.choices["lookback"]).issubset(set(GRID))
    # OOS equity starts at the first post-train day and tiles to the end.
    assert wf.equity.index[0] == returns.index[504]
    assert len(wf.equity) == len(returns) - 504
    assert np.isfinite(wf.oos_sharpe)


def test_test_windows_tile_without_gaps_or_overlap():
    returns = _noise_returns(n=1300)
    wf = walk_forward(returns, momentum, GRID, train=504, test=126)
    # Concatenated OOS returns must have a strictly increasing, unique index.
    idx = wf.oos_returns.index
    assert idx.is_monotonic_increasing
    assert idx.is_unique
    assert len(idx) == len(returns) - 504


def test_rejects_train_shorter_than_lookback():
    returns = _noise_returns(n=800)
    # train must exceed the longest lookback (252) — 200 is too short.
    try:
        walk_forward(returns, momentum, GRID, train=200, test=100)
    except ValueError:
        return
    raise AssertionError("expected ValueError for train <= warmup")


# --------------------------------------------------------------------------- #
# The point of the whole exercise.                                            #
# --------------------------------------------------------------------------- #

def test_overfitting_tax_is_positive_on_noise():
    # No real signal exists, yet searching six lookbacks over the full sample will
    # find one that looks good in-sample. Walk-forward must not reproduce it OOS.
    returns = _noise_returns(n=1600, seed=7)
    cmp = compare(returns, momentum, GRID, train=504, test=126)

    naive_claim = cmp.naive_full_sample_sharpe   # what a naive backtest brags about
    honest = cmp.walk_forward.oos_sharpe         # what you'd actually have earned

    # The in-sample-optimized number is inflated relative to the honest one...
    assert naive_claim > honest
    # ...and the reported "overfitting tax" is exactly that gap.
    assert np.isclose(cmp.overfitting_tax, naive_claim - honest)


def test_real_edge_survives_out_of_sample():
    # When a genuine, persistent edge exists, walk-forward should still capture it:
    # the honest OOS curve ends up ahead of where it started.
    returns = _trending_returns(n=1600, seed=3)
    cmp = compare(returns, momentum, GRID, train=504, test=126)
    assert cmp.walk_forward.equity.iloc[-1] > 1.0


def test_best_fixed_param_is_in_grid():
    returns = _trending_returns(n=1200)
    fit = best_fixed_param(returns, momentum, GRID)
    assert fit.param in GRID
    assert np.isfinite(fit.full_sample_sharpe)
