"""Offline tests for forecasting and honest evaluation — synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecastable import data as dt
from forecastable import forecast as fc


def _seasonal_series(n=180, trend=0.3, curve=0.0, noise=0.0, seed=0):
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    t = np.arange(n)
    season = np.array([0, 1, 3, 5, 6, 5, 3, 0, -2, -4, -3, -1], dtype=float)
    rng = np.random.default_rng(seed)
    y = 300 + trend * t + curve * t**2 + season[idx.month - 1] + rng.normal(0, noise, n)
    return pd.Series(y, index=idx, name="synthetic")


# --- metrics ---------------------------------------------------------------- #

def test_metrics_hand_values():
    actual = [10.0, 20.0, 30.0]
    forecast = [12.0, 18.0, 33.0]                 # errors: 2, 2, 3
    assert np.isclose(fc.mae(actual, forecast), 7 / 3)
    assert np.isclose(fc.rmse(actual, forecast), np.sqrt((4 + 4 + 9) / 3))
    assert fc.mape(actual, forecast) > 0


# --- baselines -------------------------------------------------------------- #

def test_naive_last_repeats_final_value():
    s = _seasonal_series(n=60)
    fcast = fc.naive_last(s, 5)
    assert len(fcast) == 5
    assert (fcast == s.iloc[-1]).all()


def test_seasonal_naive_tiles_last_period():
    s = _seasonal_series(n=48)
    fcast = fc.seasonal_naive(s, 12)
    # Forecast for month k should equal the value 12 months earlier.
    assert np.allclose(fcast, s.iloc[-12:].to_numpy())


# --- the model -------------------------------------------------------------- #

def test_trend_seasonal_recovers_structure_and_beats_naive():
    s = _seasonal_series(n=180, trend=0.3, noise=0.2)
    card = fc.evaluate(s, "synthetic", test_len=24)
    assert card.mae["trend+seasonal"] < card.mae["naive-last"]
    assert card.mae["trend+seasonal"] < card.mae["seasonal-naive"]
    assert card.model_beats_naive


def test_quadratic_trend_captures_curvature():
    # A strongly accelerating series (like CO2): the quadratic trend must forecast
    # it far better than a flat naive baseline.
    s = _seasonal_series(n=240, trend=0.1, curve=0.002, noise=0.2)
    card = fc.evaluate(s, "curved", test_len=36)
    assert card.mae["trend+seasonal"] < 0.3 * card.mae["naive-last"]


def test_evaluate_shapes():
    s = _seasonal_series(n=120)
    card = fc.evaluate(s, "synthetic", test_len=18)
    assert len(card.actual) == 18
    for arr in card.forecasts.values():
        assert len(arr) == 18


# --- the honest random-walk claim ------------------------------------------ #

def test_random_walk_defeats_the_flexible_model_on_average():
    mean_mae, win_rate = fc.random_walk_experiment(n_seeds=40, n=300, test_len=36)
    # Repeating the last value should have the lowest average error...
    assert mean_mae["naive-last"] == min(mean_mae.values())
    # ...and the flexible model should NOT reliably beat it.
    assert win_rate < 0.5
    assert mean_mae["trend+seasonal"] > mean_mae["naive-last"]


def test_evaluate_rejects_too_short_series():
    s = dt.random_walk(n=40, seed=0)
    try:
        fc.evaluate(s, "short", test_len=36)
    except ValueError:
        return
    raise AssertionError("expected ValueError for too-short series")
