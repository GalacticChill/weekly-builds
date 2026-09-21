"""Benford's Law: the leading digit of real data isn't uniform.

Count the first digit of a big pile of naturally-occurring numbers — populations,
river lengths, stock prices — and 1 shows up about 30% of the time while 9 shows up
about 5%. The reason is scale-invariance: data that spans many orders of magnitude
is (roughly) uniform on a *logarithmic* axis, and on that axis the stretch of
numbers beginning with 1 is far wider than the stretch beginning with 9.

The expected frequency of leading digit d is::

    P(d) = log10(1 + 1/d)

This module extracts leading digits and measures how closely a dataset follows that
law, using two standard yardsticks: Pearson's chi-square (with a p-value) and
Nigrini's mean absolute deviation (MAD), whose conformity bands are the ones used
in accounting forensics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import chi2

DIGITS = np.arange(1, 10)


def benford_probs() -> np.ndarray:
    """Expected proportions for leading digits 1..9 under Benford's Law."""
    return np.log10(1.0 + 1.0 / DIGITS)


def first_digits(values) -> np.ndarray:
    """Leading (most-significant) decimal digit of each nonzero, finite value.

    Sign is ignored and scale doesn't matter: 3.14, -3140 and 0.00031 all lead
    with 3. Zeros and non-finite values are dropped (they have no leading digit).
    """
    x = np.abs(np.asarray(values, dtype=float))
    x = x[np.isfinite(x) & (x > 0)]
    # Scale each value into [1, 10) via its order of magnitude, then take floor.
    scaled = x / np.power(10.0, np.floor(np.log10(x)))
    digits = np.floor(scaled).astype(int)
    # Guard against floating-point landing on 10 (e.g. 9.9999999 -> 10).
    digits[digits == 10] = 9
    return digits


def digit_distribution(values) -> np.ndarray:
    """Observed proportion of each leading digit 1..9."""
    d = first_digits(values)
    counts = np.array([(d == k).sum() for k in DIGITS], dtype=float)
    total = counts.sum()
    return counts / total if total else counts


# Nigrini's MAD conformity bands for the first-digit test.
_MAD_BANDS = [
    (0.006, "close conformity"),
    (0.012, "acceptable conformity"),
    (0.015, "marginal conformity"),
    (float("inf"), "nonconformity"),
]


def conformity(mad: float) -> str:
    """Map a MAD value to Nigrini's verbal conformity band."""
    for threshold, label in _MAD_BANDS:
        if mad < threshold:
            return label
    return "nonconformity"


@dataclass
class BenfordResult:
    """How well one dataset follows Benford's Law."""

    n: int
    observed: np.ndarray = field(repr=False)     # proportions, digits 1..9
    expected: np.ndarray = field(repr=False)     # Benford proportions
    chi_square: float
    p_value: float
    mad: float

    @property
    def conforms(self) -> str:
        return conformity(self.mad)


def analyze(values) -> BenfordResult:
    """Full Benford analysis of a dataset of numbers."""
    d = first_digits(values)
    n = int(d.size)
    observed_counts = np.array([(d == k).sum() for k in DIGITS], dtype=float)
    observed = observed_counts / n if n else observed_counts
    expected = benford_probs()

    expected_counts = expected * n
    chi_sq = float(np.sum((observed_counts - expected_counts) ** 2 / expected_counts))
    p = float(chi2.sf(chi_sq, df=8))
    mad = float(np.mean(np.abs(observed - expected)))

    return BenfordResult(
        n=n, observed=observed, expected=expected,
        chi_square=chi_sq, p_value=p, mad=mad,
    )
