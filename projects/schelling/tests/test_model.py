"""Offline tests for the Schelling model — deterministic, hand-checked."""

from __future__ import annotations

import numpy as np

from schelling import model as m


def test_make_grid_counts_and_reproducible():
    g = m.make_grid(size=10, empty_frac=0.2, share_a=0.5, seed=1)
    assert g.shape == (10, 10)
    assert (g == m.EMPTY).sum() == 20                 # 20% of 100
    agents = (g != m.EMPTY).sum()
    assert agents == 80
    assert (g == m.A).sum() == 40 and (g == m.B).sum() == 40
    assert np.array_equal(g, m.make_grid(size=10, empty_frac=0.2, seed=1))


def test_same_type_share_hand_computed():
    # Center (1,1) is A; of its 6 occupied neighbours, 3 are A -> share 0.5.
    grid = np.array([
        [m.A, m.A, m.EMPTY],
        [m.A, m.A, m.B],
        [m.EMPTY, m.B, m.B],
    ])
    share = m.same_type_share(grid)
    assert np.isclose(share[1, 1], 0.5)
    assert np.isnan(share[0, 2])                      # empty cell -> NaN


def test_fully_surrounded_by_same_is_share_one():
    grid = np.full((3, 3), m.A)
    share = m.same_type_share(grid)
    assert np.isclose(share[1, 1], 1.0)


def test_discontent_mask_threshold_behavior():
    grid = np.array([
        [m.A, m.A, m.EMPTY],
        [m.A, m.A, m.B],
        [m.EMPTY, m.B, m.B],
    ])
    # Center A has same-share 0.5: discontent at tol 0.6, content at tol 0.5.
    assert m.discontent_mask(grid, 0.6)[1, 1]
    assert not m.discontent_mask(grid, 0.5)[1, 1]


def test_isolated_agent_is_content():
    grid = np.full((3, 3), m.EMPTY)
    grid[1, 1] = m.A                                   # no occupied neighbours
    assert not m.discontent_mask(grid, 0.99)[1, 1]


def test_segregation_blocks_high_random_mid():
    blocks = np.array([[m.A] * 10] * 10 + [[m.B] * 10] * 10)  # top A, bottom B
    rand = m.make_grid(size=50, empty_frac=0.0, seed=2)
    assert m.segregation(blocks) > 0.7
    assert 0.45 < m.segregation(rand) < 0.55
    assert m.segregation(blocks) > m.segregation(rand)


def test_tolerance_zero_no_one_moves():
    grid = m.make_grid(size=20, seed=3)
    res = m.run(grid, tolerance=0.0, max_rounds=50)
    assert res.rounds == 0 and res.converged
    assert np.array_equal(res.initial, res.final)


def test_moderate_tolerance_increases_segregation():
    grid = m.make_grid(size=40, seed=4)
    res = m.run(grid, tolerance=0.4, max_rounds=100, seed=4)
    assert res.final_segregation > res.initial_segregation + 0.1


def test_run_is_reproducible():
    g = m.make_grid(size=30, seed=5)
    a = m.run(g, tolerance=0.35, seed=5)
    b = m.run(g, tolerance=0.35, seed=5)
    assert np.isclose(a.final_segregation, b.final_segregation)
    assert np.array_equal(a.final, b.final)
