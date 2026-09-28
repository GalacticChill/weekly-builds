"""Forecasting models, baselines, and an honest way to score them.

The cardinal rule of time-series evaluation: never shuffle. You may only ever train
on the past and test on the future, because in real life that's all you have. So we
hold out the *last* stretch of each series, forecast it from the end of the training
data, and compare every method on that untouched future.

And you must always beat a naive baseline before believing a model. Two are provided
-- repeat-the-last-value and repeat-last-season -- because on data with no real
structure, one of them is unbeatable, and a "good-looking" model that can't clear
that bar is fooling you.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SEASONAL_PERIOD = 12


# --- metrics ---------------------------------------------------------------- #

def mae(actual, forecast) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(forecast))))


def rmse(actual, forecast) -> float:
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    return float(np.sqrt(np.mean((a - f) ** 2)))


def mape(actual, forecast) -> float:
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    nz = a != 0
    return float(np.mean(np.abs((a[nz] - f[nz]) / a[nz])) * 100.0)


# --- forecasters: each maps (train_series, horizon) -> forecast values ------ #

def naive_last(train: pd.Series, h: int) -> np.ndarray:
    """Repeat the last observed value -- the optimal forecast for a random walk."""
    return np.repeat(train.iloc[-1], h)


def seasonal_naive(train: pd.Series, h: int, period: int = SEASONAL_PERIOD) -> np.ndarray:
    """Repeat the last full season (value from `period` steps ago)."""
    last = train.iloc[-period:].to_numpy()
    return np.array([last[i % period] for i in range(h)])


TREND_DEGREE = 2  # CO2's rise is accelerating, so a quadratic trend fits it honestly


def _design(t: np.ndarray, months: np.ndarray, period: int, degree: int) -> np.ndarray:
    """Intercept + polynomial trend (to `degree`) + one-hot month dummies."""
    cols = [np.ones_like(t, dtype=float)]
    for d in range(1, degree + 1):
        cols.append(t.astype(float) ** d)
    for m in range(2, period + 1):          # drop month 1 as the baseline
        cols.append((months == m).astype(float))
    return np.column_stack(cols)


def trend_seasonal(train: pd.Series, h: int, period: int = SEASONAL_PERIOD,
                   degree: int = TREND_DEGREE) -> np.ndarray:
    """Least-squares polynomial trend plus a fixed monthly seasonal pattern.

    Fit on the training window only, then project trend and season forward. The
    same flexibility that lets this track CO2's accelerating curve is exactly what
    makes it dangerous on a structureless series -- it will happily extrapolate a
    trend that was only ever noise.
    """
    n = len(train)
    t = np.arange(n)
    months = train.index.month.to_numpy()
    coef, *_ = np.linalg.lstsq(
        _design(t, months, period, degree), train.to_numpy(), rcond=None
    )

    future_index = pd.date_range(
        train.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS"
    )
    tf = np.arange(n, n + h)
    xf = _design(tf, future_index.month.to_numpy(), period, degree)
    return xf @ coef


FORECASTERS = {
    "naive-last": naive_last,
    "seasonal-naive": seasonal_naive,
    "trend+seasonal": trend_seasonal,
}


@dataclass
class Scorecard:
    """How every method did on the held-out future of one series."""

    name: str
    horizon: int
    train_end: pd.Timestamp
    actual: pd.Series
    forecasts: dict[str, np.ndarray]
    mae: dict[str, float]
    rmse: dict[str, float]
    mape: dict[str, float]

    @property
    def best(self) -> str:
        return min(self.mae, key=self.mae.get)

    @property
    def model_beats_naive(self) -> bool:
        return self.mae["trend+seasonal"] < self.mae["naive-last"]


def random_walk_experiment(n_seeds: int = 300, n: int = 822, test_len: int = 60):
    """Average forecast error across many random walks -- the honest way to judge.

    A single random walk can flatter any method by luck. Across hundreds of them,
    the truth emerges: repeating the last value should win on average, and the
    flexible model should beat it only about half the time (i.e. never reliably).
    Returns (mean MAE per method, fraction of walks where the model beats naive).
    """
    from .data import random_walk

    totals = {m: 0.0 for m in FORECASTERS}
    model_wins = 0
    for seed in range(n_seeds):
        walk = random_walk(n=n, seed=seed)
        card = evaluate(walk, "rw", test_len)
        for m in FORECASTERS:
            totals[m] += card.mae[m]
        if card.mae["trend+seasonal"] < card.mae["naive-last"]:
            model_wins += 1
    mean_mae = {m: totals[m] / n_seeds for m in FORECASTERS}
    return mean_mae, model_wins / n_seeds


def evaluate(series: pd.Series, name: str, test_len: int = 60) -> Scorecard:
    """Hold out the last `test_len` points and forecast them from the past only."""
    if len(series) <= test_len + SEASONAL_PERIOD:
        raise ValueError("series too short for this test_len")
    train, test = series.iloc[:-test_len], series.iloc[-test_len:]

    forecasts, mae_, rmse_, mape_ = {}, {}, {}, {}
    for label, fn in FORECASTERS.items():
        fc = np.asarray(fn(train, test_len), dtype=float)
        forecasts[label] = fc
        mae_[label] = mae(test.to_numpy(), fc)
        rmse_[label] = rmse(test.to_numpy(), fc)
        mape_[label] = mape(test.to_numpy(), fc)

    return Scorecard(
        name=name, horizon=test_len, train_end=train.index[-1], actual=test,
        forecasts=forecasts, mae=mae_, rmse=rmse_, mape=mape_,
    )
