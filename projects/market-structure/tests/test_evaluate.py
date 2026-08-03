"""Offline tests for the scoring helpers — hand-checked answers."""

from __future__ import annotations

from market_structure import evaluate


def test_purity_perfect_match():
    true = ["tech", "tech", "energy", "energy"]
    pred = [1, 1, 2, 2]
    assert evaluate.purity(true, pred) == 1.0


def test_purity_partial():
    # Cluster 1 = {tech, tech, energy} -> majority tech, 2/3 right.
    # Cluster 2 = {energy}             -> 1/1 right.  Total 3/4.
    true = ["tech", "tech", "energy", "energy"]
    pred = [1, 1, 1, 2]
    assert evaluate.purity(true, pred) == 0.75


def test_purity_is_label_invariant():
    # Renaming the cluster ids must not change purity.
    true = ["a", "a", "b", "b"]
    assert evaluate.purity(true, [7, 7, 9, 9]) == evaluate.purity(true, [0, 0, 1, 1])


def test_adjusted_rand_perfect_and_relabeled():
    true = ["a", "a", "b", "b", "c", "c"]
    # A pure relabeling is still a perfect clustering: ARI == 1.
    assert evaluate.adjusted_rand(true, [2, 2, 0, 0, 1, 1]) == 1.0


def test_contingency_shape_and_counts():
    true = ["tech", "tech", "energy"]
    pred = [1, 1, 2]
    table = evaluate.contingency(true, pred)
    assert table.loc["tech", 1] == 2
    assert table.loc["energy", 2] == 1
    assert int(table.to_numpy().sum()) == 3


def test_cluster_composition_columns():
    comp = evaluate.cluster_composition(["AAPL", "XOM"], ["tech", "energy"], [1, 2])
    assert list(comp.columns) == ["ticker", "sector", "cluster"]
    assert set(comp["ticker"]) == {"AAPL", "XOM"}
