"""Offline tests for data helpers — no network (CO2 fetch omitted)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecastable import data as dt


def test_random_walk_shape_and_index():
    s = dt.random_walk(n=100, seed=1)
    assert len(s) == 100
    assert isinstance(s.index, pd.DatetimeIndex)
    assert s.index.freqstr in ("MS", "ME", "M")  # monthly


def test_random_walk_is_reproducible():
    a = dt.random_walk(n=50, seed=7)
    b = dt.random_walk(n=50, seed=7)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_random_walk_steps_are_unit_ish():
    # First differences should look like the configured step noise, not a trend.
    s = dt.random_walk(n=5000, step_sd=1.0, seed=3)
    diffs = np.diff(s.to_numpy())
    assert abs(diffs.mean()) < 0.1        # drift-free
    assert 0.9 < diffs.std() < 1.1        # ~ unit step
