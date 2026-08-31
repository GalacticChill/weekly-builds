"""Pull cross-country development indicators from the World Bank Open Data API.

No API key needed. We ask for a small range of recent years and keep, for each
country, its most recent non-null value per indicator — which fills gaps that any
single year would leave. Aggregates ("World", "Arab World", income groups) are
dropped so we're left with actual countries, each tagged with its World Bank
region and income group.

The collected table is cached to `data/indicators.csv` so the analysis runs fully
offline and reproducibly; `--collect` refreshes it.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

API = "https://api.worldbank.org/v2"

# World Bank indicator code -> the short column name we'll use.
INDICATORS = {
    "SP.DYN.LE00.IN": "life_expectancy",     # target: life expectancy at birth (years)
    "NY.GDP.PCAP.PP.CD": "gdp_pc_ppp",       # GDP per capita, PPP (current international $)
    "SH.XPD.CHEX.GD.ZS": "health_exp_pct_gdp",  # current health expenditure (% of GDP)
    "SE.SEC.ENRR": "school_enroll_secondary",   # gross secondary school enrollment (%)
    "SH.H2O.BASW.ZS": "water_access",        # people using at least basic drinking water (%)
    "SH.STA.BASS.ZS": "sanitation_access",   # people using at least basic sanitation (%)
    "SP.URB.TOTL.IN.ZS": "urban_pct",        # urban population (% of total)
    "SH.IMM.MEAS": "measles_immunization",   # measles immunization, ages 12-23 months (%)
    "SP.DYN.TFRT.IN": "fertility_rate",      # total fertility rate (births per woman)
}

TARGET = "life_expectancy"
MONEY = "gdp_pc_ppp"


def _get_json(url: str, retries: int = 3, pause: float = 1.0):
    """Fetch and parse a World Bank JSON endpoint, with a couple of retries."""
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.load(resp)
        except Exception as exc:  # noqa: BLE001 - network hiccups; retry then raise
            last = exc
            time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"World Bank request failed: {url}\n{last}")


def _countries() -> pd.DataFrame:
    """Real countries only (aggregates dropped), with region and income group."""
    data = _get_json(f"{API}/country?format=json&per_page=400")
    rows = []
    for c in data[1]:
        if c["region"]["value"] == "Aggregates":
            continue
        rows.append(
            {
                "iso3": c["id"],
                "country": c["name"],
                "region": c["region"]["value"],
                "income_group": c["incomeLevel"]["value"],
            }
        )
    return pd.DataFrame(rows).set_index("iso3").sort_index()


def _fetch_indicator(code: str, start: int, end: int) -> dict[str, float]:
    """Most recent non-null value per country for one indicator over [start, end]."""
    url = (
        f"{API}/country/all/indicator/{code}"
        f"?format=json&date={start}:{end}&per_page=20000"
    )
    data = _get_json(url)
    if len(data) < 2 or data[1] is None:
        return {}
    best: dict[str, tuple[int, float]] = {}
    for row in data[1]:
        val = row["value"]
        if val is None:
            continue
        iso3 = row["countryiso3code"]
        year = int(row["date"])
        if iso3 not in best or year > best[iso3][0]:
            best[iso3] = (year, float(val))
    return {iso3: v for iso3, (yr, v) in best.items()}


def collect(start: int = 2015, end: int = 2021, pause: float = 0.5) -> pd.DataFrame:
    """Assemble a country x indicator table from the World Bank API."""
    countries = _countries()
    frame = countries.copy()
    for code, name in INDICATORS.items():
        series = _fetch_indicator(code, start, end)
        frame[name] = pd.Series(series)
        time.sleep(pause)
    frame.index.name = "iso3"
    return frame


def save(df: pd.DataFrame, path: str | Path) -> Path:
    """Write the indicator table to CSV (index = iso3)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return path
