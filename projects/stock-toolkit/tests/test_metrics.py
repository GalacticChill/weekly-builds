"""Metric tests on hand-made series with known answers — no network needed."""

import numpy as np
import pandas as pd
import pytest

from stock_toolkit.metrics import (
    cumulative_return,
    daily_returns,
    max_drawdown,
    moving_average,
)


def _series(values):
    idx = pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype="float64")


def test_cumulative_return():
    prices = _series([100, 110, 121])
    assert cumulative_return(prices) == pytest.approx(0.21)  # 100 -> 121 == +21%


def test_daily_returns():
    prices = _series([100, 110, 121])
    r = daily_returns(prices)
    assert len(r) == 2
    np.testing.assert_allclose(r.values, [0.1, 0.1])


def test_moving_average():
    prices = _series([1, 2, 3, 4])
    ma = moving_average(prices, window=2)
    assert np.isnan(ma.iloc[0])
    np.testing.assert_allclose(ma.iloc[1:].values, [1.5, 2.5, 3.5])


def test_max_drawdown():
    # Peak 100 -> trough 60 is a 40% drawdown.
    prices = _series([100, 120, 60, 80])
    assert max_drawdown(prices) == -0.5  # peak 120 -> trough 60 == -50%
