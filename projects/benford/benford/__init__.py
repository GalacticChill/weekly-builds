"""benford: does real data follow Benford's Law, and can it catch fabricated numbers?

Extract the leading digit of a pile of numbers, compare the distribution to
Benford's log law, and measure conformity with chi-square and Nigrini's MAD.
"""

from .datasets import build_datasets, collect_worldbank, fibonacci, powers_of_two
from .law import (
    BenfordResult,
    analyze,
    benford_probs,
    conformity,
    digit_distribution,
    first_digits,
)

__all__ = [
    "benford_probs",
    "first_digits",
    "digit_distribution",
    "analyze",
    "BenfordResult",
    "conformity",
    "build_datasets",
    "collect_worldbank",
    "fibonacci",
    "powers_of_two",
]
