"""Clean the raw indicator table into a modeling frame.

The single most important transform here is taking the **logarithm of income**.
Life expectancy doesn't rise linearly with GDP per capita — an extra $1,000 means
years of life in a poor country and almost nothing in a rich one. On a log-income
axis that curved relationship straightens out, which is both easier to model and
the whole point of the story.

We deliberately *exclude* infant and child mortality as predictors even though the
World Bank publishes them: they are arithmetic components of life expectancy, so
using them would be predicting the answer from itself (the same leakage trap as
using forks to predict a repo's stars). Every predictor kept is a plausible *cause*
of longevity — money, health spending, schooling, water, sanitation — not a
restatement of it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .collect import MONEY, TARGET

# Predictors fed to the multivariate model. `log_gdp` is engineered below.
PREDICTORS = [
    "log_gdp",
    "health_exp_pct_gdp",
    "school_enroll_secondary",
    "water_access",
    "sanitation_access",
    "urban_pct",
    "measles_immunization",
    "fertility_rate",
]

# Kept out on purpose — mechanical components of life expectancy, not causes.
EXCLUDED_AS_LEAKAGE = ["infant_mortality", "child_mortality", "adult_mortality"]

_META = ["country", "region", "income_group"]


def load(path: str | Path) -> pd.DataFrame:
    """Read the cached indicator CSV (index = iso3)."""
    return pd.read_csv(path, index_col="iso3")


def missing_report(raw: pd.DataFrame) -> pd.Series:
    """Fraction missing per indicator among countries that have target + income."""
    base = raw.dropna(subset=[TARGET, MONEY])
    cols = [c for c in raw.columns if c not in _META]
    return base[cols].isna().mean().sort_values(ascending=False)


def build_features(raw: pd.DataFrame, impute: bool = True) -> pd.DataFrame:
    """Assemble the modeling frame: target, log-income, and imputed predictors.

    Countries missing life expectancy or income are dropped (both are essential).
    Remaining gaps in the other predictors are filled with the column median so the
    country still contributes; the count of fills is available via `missing_report`.
    """
    df = raw.dropna(subset=[TARGET, MONEY]).copy()
    df["log_gdp"] = np.log10(df[MONEY].astype(float))

    for col in PREDICTORS:
        if col == "log_gdp":
            continue
        if col not in df.columns:
            raise KeyError(f"expected predictor {col!r} missing from raw data")
        if impute:
            df[col] = df[col].fillna(df[col].median())

    keep = _META + [TARGET, MONEY] + PREDICTORS
    return df[[c for c in keep if c in df.columns]]
