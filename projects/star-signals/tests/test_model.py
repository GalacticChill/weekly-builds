"""Offline tests for the model — synthetic data with a known 'right answer'."""

import numpy as np
import pandas as pd

from star_signals import features as ft
from star_signals import model as ml


def _synthetic(n=600, seed=0, driver="size_kb", noise=0.3) -> pd.DataFrame:
    """Build a feature frame where the target is driven by ONE substance feature
    plus noise, and every other feature is pure noise. A good model should then
    attribute most importance to the substance group.
    """
    rng = np.random.default_rng(seed)
    cols = {}
    for f in ft.ALL_FEATURES:
        cols[f] = rng.normal(size=n)
    df = pd.DataFrame(cols)
    signal = (df[driver] - df[driver].mean()) / df[driver].std()
    df[ft.TARGET] = 5.0 + 2.0 * signal + rng.normal(scale=noise, size=n)
    df["stars"] = np.expm1(df[ft.TARGET].clip(lower=0))
    return df


def test_substance_driver_is_attributed_to_substance():
    res = ml.fit(_synthetic(driver="size_kb"))
    assert res.group_importance["substance"] > res.group_importance["signal"]
    assert "substance" in res.verdict


def test_signal_driver_is_attributed_to_signal():
    # Flip it: when a SIGNAL feature drives the target, the model must say so.
    res = ml.fit(_synthetic(driver="topics_count"))
    assert res.group_importance["signal"] > res.group_importance["substance"]
    assert "signal" in res.verdict


def test_model_recovers_signal_out_of_sample():
    res = ml.fit(_synthetic(noise=0.2))
    assert res.forest_r2 > 0.5   # real out-of-sample skill on a learnable target


def test_result_shapes_and_indices():
    data = _synthetic()
    res = ml.fit(data, test_size=0.25)
    assert list(res.importances.index.sort_values()) == sorted(ft.ALL_FEATURES)
    assert list(res.coefficients.index.sort_values()) == sorted(ft.ALL_FEATURES)
    assert len(res.y_test) == len(res.y_pred) == int(round(len(data) * 0.25))
    # Group importances are a proper split summing to ~1.
    assert np.isclose(res.group_importance.sum(), 1.0, atol=1e-6)


def test_importances_are_nonnegative_and_sum_to_one():
    res = ml.fit(_synthetic())
    assert (res.importances >= 0).all()
    assert np.isclose(res.importances.sum(), 1.0, atol=1e-6)
