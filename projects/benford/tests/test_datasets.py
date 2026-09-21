"""Offline tests for dataset assembly — no network (World Bank omitted)."""

from __future__ import annotations

import numpy as np

from benford import datasets as ds


def test_fibonacci_sequence_start():
    fib = ds.fibonacci(6)
    assert list(fib) == [1, 1, 2, 3, 5, 8]


def test_powers_of_two_values():
    p = ds.powers_of_two(5)
    assert list(p) == [1, 2, 4, 8, 16]


def test_build_datasets_without_worldbank_has_controls():
    suite = ds.build_datasets(None)
    # The math + control sets are always present even with no network data.
    assert "Fibonacci numbers" in suite
    assert "Powers of two" in suite
    assert "Uniform random" in suite
    assert "Bounded (heights)" in suite
    for name, (values, kind) in suite.items():
        assert len(values) > 0
        assert isinstance(kind, str)


def test_bounded_quantity_is_within_one_order_of_magnitude():
    vals = ds.bounded_quantity(2000, seed=0)
    # Heights around 170cm: essentially all lead with digit 1, by construction.
    lead_ones = np.mean([str(abs(v))[0] == "1" for v in vals])
    assert lead_ones > 0.95
