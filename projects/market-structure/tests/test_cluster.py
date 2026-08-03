"""Offline tests for clustering — synthetic block-correlated returns, no network.

The centerpiece is `test_recovers_known_blocks`: we build returns where assets in
the same group share a common factor (so they genuinely co-move) and assets in
different groups don't. A correct pipeline must rediscover those groups from the
returns alone — perfect purity and an adjusted Rand index of 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_structure import cluster, evaluate


def _block_returns(group_sizes=(5, 5, 5), n=600, noise=1.0, seed=0):
    """Returns where each group shares a latent factor plus idiosyncratic noise.

    Within-group correlation is high; cross-group correlation is ~0. Returns the
    frame plus the true group id of each column.
    """
    rng = np.random.default_rng(seed)
    columns, true = [], []
    series = []
    for gi, size in enumerate(group_sizes):
        factor = rng.normal(0.0, 1.0, size=n)      # the shared group driver
        for j in range(size):
            series.append(factor + rng.normal(0.0, noise, size=n))
            columns.append(f"g{gi}_{j}")
            true.append(gi)
    df = pd.DataFrame(np.array(series).T, columns=columns)
    return df, true


def test_correlation_distance_properties():
    df, _ = _block_returns()
    dist = cluster.correlation_distance(df)
    m = dist.to_numpy()
    # Zero diagonal, symmetric, and bounded in [0, 2].
    assert np.allclose(np.diag(m), 0.0)
    assert np.allclose(m, m.T)
    assert m.min() >= 0.0 and m.max() <= 2.0 + 1e-9


def test_identical_series_are_distance_zero():
    rng = np.random.default_rng(1)
    x = rng.normal(size=300)
    df = pd.DataFrame({"A": x, "B": x, "C": rng.normal(size=300)})
    dist = cluster.correlation_distance(df)
    assert dist.loc["A", "B"] < 1e-9        # perfectly correlated -> distance 0
    assert dist.loc["A", "C"] > 0.5         # independent -> clearly positive


def test_recovers_known_blocks():
    df, true = _block_returns(group_sizes=(5, 5, 5), n=800, noise=1.0, seed=3)
    dist = cluster.correlation_distance(df)
    pred = cluster.cluster_labels(dist, k=3).tolist()
    assert evaluate.purity(true, pred) == 1.0
    assert evaluate.adjusted_rand(true, pred) == 1.0


def test_leaf_order_is_a_permutation_grouping_blocks():
    df, true = _block_returns(group_sizes=(4, 4, 4), n=600, noise=0.8, seed=5)
    dist = cluster.correlation_distance(df)
    order = cluster.leaf_order(dist)
    # Every ticker appears exactly once.
    assert sorted(order) == sorted(df.columns)
    # Same-group tickers are contiguous in the leaf order (blocks aren't interleaved).
    true_of = dict(zip(df.columns, true))
    seq = [true_of[t] for t in order]
    # Count how many times the group id changes as we walk the order; with 3 clean
    # blocks it should change exactly twice (one boundary between each pair).
    switches = sum(a != b for a, b in zip(seq, seq[1:]))
    assert switches == 2


def test_uncorrelated_noise_scores_near_chance():
    # No block structure at all: clustering should NOT find real agreement with an
    # arbitrary "true" labeling — ARI should sit near zero.
    rng = np.random.default_rng(9)
    df = pd.DataFrame(rng.normal(size=(600, 12)), columns=[f"x{i}" for i in range(12)])
    dist = cluster.correlation_distance(df)
    pred = cluster.cluster_labels(dist, k=3).tolist()
    fake_true = [i % 3 for i in range(12)]     # a meaningless grouping
    assert abs(evaluate.adjusted_rand(fake_true, pred)) < 0.35
