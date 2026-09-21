"""The datasets we test against Benford's Law.

A fair test needs three kinds of data:

- **Real, wide-ranging** numbers that should conform — country populations, GDP,
  land area, CO2 emissions (pulled from the World Bank, cached to a CSV).
- **Mathematical** sequences known to follow Benford almost exactly — Fibonacci
  numbers and powers of two — as positive controls.
- **Things that should NOT conform** — uniform random draws, and a bounded quantity
  living within one order of magnitude — as negative controls, so a passing grade
  actually means something.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

API = "https://api.worldbank.org/v2"

# World Bank indicators that naturally span many orders of magnitude.
WB_INDICATORS = {
    "SP.POP.TOTL": "population",
    "NY.GDP.MKTP.CD": "gdp",
    "AG.LND.TOTL.K2": "land_area",
    "EN.GHG.CO2.MT.CE.AR5": "co2_emissions",
}


def _get_json(url: str, retries: int = 3, pause: float = 1.0):
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.load(resp)
        except Exception as exc:  # noqa: BLE001 - retry transient network errors
            last = exc
            time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"World Bank request failed: {url}\n{last}")


def collect_worldbank(year_start: int = 2015, year_end: int = 2022) -> pd.DataFrame:
    """One long column of real-world magnitudes, tagged by indicator.

    Takes each country's most recent value per indicator over the window, then
    stacks every indicator's values together — a few hundred numbers spanning from
    tiny island nations to the whole planet.
    """
    rows = []
    for code, name in WB_INDICATORS.items():
        url = f"{API}/country/all/indicator/{code}?format=json&date={year_start}:{year_end}&per_page=20000"
        data = _get_json(url)
        if len(data) < 2 or data[1] is None:
            continue
        best: dict[str, tuple[int, float]] = {}
        for r in data[1]:
            if r["value"] is None:
                continue
            iso = r["countryiso3code"]
            yr = int(r["date"])
            if iso and (iso not in best or yr > best[iso][0]):
                best[iso] = (yr, float(r["value"]))
        for iso, (yr, val) in best.items():
            rows.append({"indicator": name, "iso3": iso, "value": val})
        time.sleep(pause := 0.4)
    return pd.DataFrame(rows)


def save(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


# ---- Mathematical positive controls --------------------------------------- #

def fibonacci(n: int = 1000) -> np.ndarray:
    """First n Fibonacci numbers (as floats; they follow Benford almost exactly)."""
    a, b = 1.0, 1.0
    out = []
    for _ in range(n):
        out.append(a)
        a, b = b, a + b
    return np.array(out)


def powers_of_two(n: int = 1000) -> np.ndarray:
    """2**0 .. 2**(n-1); geometric growth also follows Benford."""
    return np.power(2.0, np.arange(n))


# ---- Negative controls ----------------------------------------------------- #

def uniform_noise(n: int = 5000, seed: int = 0) -> np.ndarray:
    """Uniform draws on [1, 100000): NOT scale-invariant, so NOT Benford."""
    return np.random.default_rng(seed).uniform(1.0, 1e5, size=n)


def bounded_quantity(n: int = 5000, seed: int = 0) -> np.ndarray:
    """A quantity confined to one order of magnitude (like adult heights in cm).

    Values clustered around a single scale can't produce Benford's spread of
    leading digits.
    """
    return np.random.default_rng(seed).normal(170.0, 10.0, size=n)


def build_datasets(wb: pd.DataFrame | None) -> dict[str, tuple[np.ndarray, str]]:
    """Assemble the labelled test suite: name -> (values, kind)."""
    sets: dict[str, tuple[np.ndarray, str]] = {}
    if wb is not None and len(wb):
        for name, grp in wb.groupby("indicator"):
            sets[f"World Bank: {name}"] = (grp["value"].to_numpy(), "real")
        sets["World Bank: all combined"] = (wb["value"].to_numpy(), "real")
    sets["Fibonacci numbers"] = (fibonacci(), "math (should conform)")
    sets["Powers of two"] = (powers_of_two(), "math (should conform)")
    sets["Uniform random"] = (uniform_noise(), "control (should NOT)")
    sets["Bounded (heights)"] = (bounded_quantity(), "control (should NOT)")
    return sets
