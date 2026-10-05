"""Offline tests for the tolerance sweep."""

from __future__ import annotations

import numpy as np

from schelling import experiment as ex


def test_sweep_shapes_and_baseline():
    s = ex.sweep(tolerances=[0.0, 0.3, 0.5], trials=2, size=30, seed=0)
    assert s.tolerances.shape == s.final_segregation.shape == (3,)
    assert 0.45 < s.baseline < 0.55            # random 50/50 start


def test_sweep_zero_tolerance_stays_near_baseline():
    s = ex.sweep(tolerances=[0.0], trials=3, size=30, seed=0)
    # Nobody ever wants to move, so the city stays as mixed as it started.
    assert abs(s.final_segregation[0] - s.baseline) < 0.05


def test_sweep_preference_raises_segregation():
    s = ex.sweep(tolerances=[0.0, 0.5], trials=3, size=30, seed=0)
    assert s.final_segregation[1] > s.final_segregation[0] + 0.2


def test_tipping_tolerance_is_reported():
    s = ex.sweep(tolerances=[0.0, 0.2, 0.4, 0.6], trials=3, size=30, seed=0)
    # A divided city (>=0.70) should appear at some moderate tolerance, not at 0.
    assert 0.0 < s.tipping_tolerance <= 0.6
