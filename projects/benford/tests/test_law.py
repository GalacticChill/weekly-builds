"""Offline tests for the Benford core — hand-checked and known-answer."""

from __future__ import annotations

import numpy as np

from benford import datasets as ds
from benford import law


def test_benford_probs_known_values():
    p = law.benford_probs()
    assert p.shape == (9,)
    assert np.isclose(p.sum(), 1.0)
    assert np.isclose(p[0], 0.30103, atol=1e-4)   # P(1)
    assert np.isclose(p[8], 0.045757, atol=1e-4)  # P(9)
    assert (np.diff(p) < 0).all()                 # strictly decreasing


def test_first_digit_extraction_ignores_scale_and_sign():
    vals = [3.14, -3140.0, 0.00031, 100.0, 900000.0]
    assert list(law.first_digits(vals)) == [3, 3, 3, 1, 9]


def test_first_digit_drops_zero_and_nonfinite():
    vals = [0.0, np.nan, np.inf, 5.0, -7.0]
    assert list(law.first_digits(vals)) == [5, 7]


def test_digit_distribution_sums_to_one():
    dist = law.digit_distribution(ds.powers_of_two(500))
    assert np.isclose(dist.sum(), 1.0)
    assert dist.shape == (9,)


def test_conformity_bands():
    assert law.conformity(0.003) == "close conformity"
    assert law.conformity(0.010) == "acceptable conformity"
    assert law.conformity(0.013) == "marginal conformity"
    assert law.conformity(0.050) == "nonconformity"


def test_fibonacci_conforms_closely():
    r = law.analyze(ds.fibonacci(1000))
    assert r.mad < 0.006          # close conformity
    assert r.p_value > 0.05       # not rejected


def test_powers_of_two_conform():
    r = law.analyze(ds.powers_of_two(1000))
    assert r.mad < 0.006


def test_uniform_noise_does_not_conform():
    r = law.analyze(ds.uniform_noise(5000, seed=1))
    assert r.mad > 0.02
    assert r.p_value < 0.01       # clearly rejected


def test_all_same_leading_digit_is_extreme_nonconformity():
    # Every value starts with 1 -> observed[0] = 1.0, huge deviation.
    r = law.analyze(np.array([1.0, 15.0, 1234.0, 0.019, 180.0]))
    assert np.isclose(r.observed[0], 1.0)
    assert r.conforms == "nonconformity"
