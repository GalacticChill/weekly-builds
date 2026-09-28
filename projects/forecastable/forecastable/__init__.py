"""forecastable: what can you actually predict in a time series, and what only looks it?

A predictable series (NOAA Mauna Loa CO2) against a random walk, with baselines, a
trend+seasonal model, and honest forecast-the-future evaluation.
"""

from .data import fetch_co2, load, random_walk, save
from .forecast import (
    FORECASTERS,
    Scorecard,
    evaluate,
    mae,
    mape,
    naive_last,
    rmse,
    seasonal_naive,
    trend_seasonal,
)

__all__ = [
    "fetch_co2",
    "random_walk",
    "save",
    "load",
    "naive_last",
    "seasonal_naive",
    "trend_seasonal",
    "FORECASTERS",
    "evaluate",
    "Scorecard",
    "mae",
    "rmse",
    "mape",
]
