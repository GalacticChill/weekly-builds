"""Offline tests for feature building — no network, hand-checked answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from life_and_money import features as ft


def _raw(n=40, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gdp = rng.uniform(1000, 80000, size=n)
    df = pd.DataFrame(
        {
            "country": [f"Country {i}" for i in range(n)],
            "region": "Somewhere",
            "income_group": "Lower middle income",
            "life_expectancy": 40 + 13 * np.log10(gdp) + rng.normal(0, 1, n),
            "gdp_pc_ppp": gdp,
            "health_exp_pct_gdp": rng.uniform(2, 12, n),
            "school_enroll_secondary": rng.uniform(30, 110, n),
            "water_access": rng.uniform(40, 100, n),
            "sanitation_access": rng.uniform(20, 100, n),
            "urban_pct": rng.uniform(15, 95, n),
            "measles_immunization": rng.uniform(50, 99, n),
            "fertility_rate": rng.uniform(1.2, 6.5, n),
        },
        index=[f"C{i:02d}" for i in range(n)],
    )
    df.index.name = "iso3"
    return df


def test_log_gdp_is_log10_of_income():
    raw = _raw()
    raw.loc["C00", "gdp_pc_ppp"] = 1000.0
    df = ft.build_features(raw)
    assert np.isclose(df.loc["C00", "log_gdp"], 3.0)   # log10(1000) == 3


def test_rows_missing_target_or_income_are_dropped():
    raw = _raw(n=10)
    raw.loc["C00", "life_expectancy"] = np.nan
    raw.loc["C01", "gdp_pc_ppp"] = np.nan
    df = ft.build_features(raw)
    assert "C00" not in df.index and "C01" not in df.index
    assert len(df) == 8


def test_median_imputation_fills_missing_predictor():
    raw = _raw(n=11)
    raw.loc["C05", "water_access"] = np.nan
    med = raw["water_access"].median()   # median of the remaining (non-null) values
    df = ft.build_features(raw)
    assert not df["water_access"].isna().any()
    assert np.isclose(df.loc["C05", "water_access"], med)


def test_leakage_columns_are_not_predictors():
    # Mortality components of life expectancy must never be predictors.
    for bad in ft.EXCLUDED_AS_LEAKAGE:
        assert bad not in ft.PREDICTORS


def test_all_predictors_present_after_build():
    df = ft.build_features(_raw())
    for col in ft.PREDICTORS:
        assert col in df.columns


def test_missing_report_is_a_fraction():
    raw = _raw(n=20)
    raw.loc["C03", "water_access"] = np.nan
    rep = ft.missing_report(raw)
    assert (rep >= 0).all() and (rep <= 1).all()
    assert np.isclose(rep["water_access"], 1 / 20)
