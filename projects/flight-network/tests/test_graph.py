"""Offline tests for graph metrics and robustness — hand-built graphs, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from flight_network import data as dt
from flight_network import graph as gr
from flight_network import robustness as rb


def _from_edges(n, edges):
    rows, cols = [], []
    for a, b in edges:
        rows += [a, b]
        cols += [b, a]
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def _star(n):
    return _from_edges(n, [(0, j) for j in range(1, n)])


# --- degrees & components --------------------------------------------------- #

def test_degrees_of_a_star():
    deg = gr.degrees(_star(6))
    assert deg[0] == 5                       # the hub
    assert list(deg[1:]) == [1, 1, 1, 1, 1]  # the spokes


def test_giant_component_fraction():
    assert gr.giant_component_fraction(_star(10)) == 1.0           # fully connected
    two = _from_edges(4, [(0, 1), (2, 3)])                         # two separate edges
    assert gr.giant_component_fraction(two) == 0.5


def test_degree_distribution_counts_all_nodes():
    vals, counts = gr.degree_distribution(_star(8))
    assert counts.sum() == 8


# --- small world ------------------------------------------------------------ #

def test_average_path_length_on_a_path_graph():
    # 0-1-2: ordered-pair hop distances are 1,2,1,1,1,2 -> mean 8/6 = 4/3, max 2.
    path = _from_edges(3, [(0, 1), (1, 2)])
    res = gr.average_path_length(path, sample=10)
    assert np.isclose(res["mean"], 4 / 3)
    assert res["max"] == 2


# --- robust yet fragile ----------------------------------------------------- #

def test_removing_the_hub_first_shatters_a_star():
    star = _star(11)
    center_first = rb.attack(star, order=np.arange(11))           # hub removed first
    center_last = rb.attack(star, order=np.array([*range(1, 11), 0]))  # hub last
    assert rb.collapse_point(center_first) < rb.collapse_point(center_last)
    assert rb.collapse_point(center_first) < 0.2


def test_targeted_beats_random_on_a_hub_network():
    star = _star(201)                        # one giant hub, 200 spokes
    targeted = rb.targeted_attack(star)
    random = rb.random_attack(star, seed=0)
    assert rb.collapse_point(targeted) < rb.collapse_point(random)
    assert rb.collapse_point(targeted) < 0.05   # removing the lone hub ends it at once


# --- data layer (no network) ------------------------------------------------ #

def test_build_graph_from_frames_is_symmetric():
    edges = pd.DataFrame({"src": ["AAA", "AAA", "BBB"], "dst": ["BBB", "CCC", "CCC"]})
    airports = pd.DataFrame(
        {"iata": ["AAA", "BBB", "CCC"], "name": ["a", "b", "c"],
         "city": ["x", "y", "z"], "country": ["U", "U", "U"],
         "lat": [0.0, 1.0, 2.0], "lon": [0.0, 1.0, 2.0]}
    ).set_index("iata")
    adj, nodes, meta = dt.build_graph(edges, airports)
    assert nodes == ["AAA", "BBB", "CCC"]
    dense = adj.toarray()
    assert (dense == dense.T).all()          # undirected
    assert dense.diagonal().sum() == 0       # no self-loops
    assert meta.loc["AAA", "degree"] == 2    # AAA connects to BBB and CCC
