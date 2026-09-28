"""The two time series at the heart of the study.

- **Atmospheric CO2** from NOAA's Mauna Loa observatory: monthly since 1958, with a
  relentless upward trend and a clean yearly breathing cycle. About as forecastable
  as real-world data gets.
- **A random walk**: each step is the last value plus pure noise. It wanders, it
  trends by accident, it *looks* like it has structure — and it is, provably,
  unforecastable beyond "tomorrow looks like today".

The contrast is the whole point: the same models that nail CO2 should fail on the
random walk, and a study that can't tell the two apart isn't worth trusting.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

CO2_URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt"
_UA = {"User-Agent": "Mozilla/5.0 (time-series research)"}


def fetch_co2() -> pd.Series:
    """Download monthly mean CO2 (ppm) from NOAA, indexed by month start."""
    req = urllib.request.Request(CO2_URL, headers=_UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8", "replace")

    years, months, values = [], [], []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        year, month, average = int(parts[0]), int(parts[1]), float(parts[3])
        if average <= 0:            # historical missing-value sentinel
            continue
        years.append(year)
        months.append(month)
        values.append(average)

    idx = pd.to_datetime({"year": years, "month": months, "day": 1})
    return pd.Series(values, index=pd.DatetimeIndex(idx), name="co2").sort_index()


def random_walk(n: int = 820, start: float = 100.0, step_sd: float = 1.0,
                seed: int = 0) -> pd.Series:
    """A random walk of length n on a monthly index (drift-free)."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, step_sd, size=n)
    values = start + np.cumsum(steps)
    idx = pd.date_range("1958-03-01", periods=n, freq="MS")
    return pd.Series(values, index=idx, name="random_walk")


def save(series: pd.Series, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(path)
    return path


def load(path: str | Path) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.iloc[:, 0]
