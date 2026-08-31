"""Offline tests for the models — synthetic data with a known 'right answer'."""

from __future__ import annotations

import numpy as np
import pandas as pd

from life_and_money import features as ft
from life_and_money import model as ml


def _raw(n=180, seed=1, noise=1.0):
    """Life expectancy is genuinely a log function of income, plus noise."""
    rng = np.random.default_rng(seed)
    gdp = np.power(10, rng.uniform(3.0, 5.0, size=n))   # $1k .. $100k
    df = pd.DataFrame(
        {
            "country": [f"Country {i}" for i in range(n)],
            "region": "Somewhere",
            "income_group": "Lower middle income",
            "life_expectancy": 40 + 13 * np.log10(gdp) + rng.normal(0, noise, n),
            "gdp_pc_ppp": gdp,
            "health_exp_pct_gdp": rng.uniform(2, 12, n),
            "school_enroll_secondary": rng.uniform(30, 110, n),
            "water_access": rng.uniform(40, 100, n),
            "sanitation_access": rng.uniform(20, 100, n),
            "urban_pct": rng.uniform(15, 95, n),
            "measles_immunization": rng.uniform(50, 99, n),
            "fertility_rate": rng.uniform(1.2, 6.5, n),
        },
        index=[f"C{i:03d}" for i in range(n)],
    )
    df.index.name = "iso3"
    return df


def test_log_income_beats_linear_income():
    # The Preston curve's signature: log-income fits far better than raw income.
    df = ft.build_features(_raw(noise=1.0))
    fit = ml.preston_fit(df)
    assert fit.r2_log > fit.r2_linear
    assert fit.slope > 0            # more income -> more life


def test_residual_flags_an_engineered_overperformer():
    raw = _raw(noise=0.5)
    raw.loc["C000", "life_expectancy"] += 12.0     # make one country beat the odds
    df = ft.build_features(raw)
    resid = ml.income_residuals(df)
    # It should have a clearly positive residual and rank near the very top.
    assert resid.loc["C000", "residual"] > 5
    assert "C000" in resid.head(3).index


def test_residuals_are_centered_near_zero():
    df = ft.build_features(_raw())
    resid = ml.income_residuals(df)
    # An OLS fit's residuals sum to ~0 by construction.
    assert abs(resid["residual"].mean()) < 1e-6


def test_full_fit_has_real_out_of_sample_skill():
    df = ft.build_features(_raw(noise=1.5))
    res = ml.full_fit(df, seed=0)
    assert res.forest_r2 > 0.5
    assert res.linear_r2 > 0.5


def test_importances_are_nonnegative_and_sum_to_one():
    res = ml.full_fit(ft.build_features(_raw()), seed=0)
    assert (res.importances >= 0).all()
    assert np.isclose(res.importances.sum(), 1.0, atol=1e-6)
    # Income is the engineered driver, so it should lead the importances.
    assert res.top_feature == "log_gdp"
