# Week of 2026-11-02 — Flight Network (network science)

**Theme:** data/network-science — the portfolio's first graph-theory project.
**Type:** new technique (networks: components, shortest paths, degree distributions,
targeted vs random attack); user asked for "the fresh capstone."

## The question
What shape is the world air-route network, and why is it both a small world and
"robust yet fragile"?

## Scope (shipped)
- `data.py` — download + parse OpenFlights (airports.dat, routes.dat) into an
  undirected airport graph; cache compact `data/edges.csv` + `data/airports.csv`
  (committed, offline). networkx NOT installed → all graph work via
  `scipy.sparse.csgraph`.
- `graph.py` — degrees, giant-component fraction (connected_components), degree
  distribution, sampled average shortest-path length (shortest_path, unweighted).
- `robustness.py` — `attack(adj, order)` removes nodes in a given order tracking the
  giant-component share; `random_attack` vs `targeted_attack` (hubs first);
  `collapse_point` (fraction removed to fall below 50%).
- `plots.py` — log-log degree distribution (hubs annotated), robustness curves
  (random vs targeted), world hub map (lon/lat scatter sized by degree).
- CLI; committed snapshot. 7 offline tests (hand-built graphs, known answers).

## Findings (3257 airports, 18930 routes)
- Heavy-tailed degree: mega-hubs AMS/FRA/CDG/IST/ATL (~200-250 direct dests) vs a
  long tail of tiny airports.
- Small world: 99.2% in one giant component; **average ~3.97 hops** between airports
  ("four degrees").
- **Robust yet fragile: random removal needs 40% to break the network below 50%
  connected; targeted hub removal needs only 9%** (near-shattered by ~20%).

## The honest core
- An edge = "a route exists," unweighted by traffic (traffic-weighting would make
  hubs even more dominant); paths are in flights not time/price; undirected ~2014
  snapshot. Structural lessons travel; exact numbers don't.

## Why this matters for the portfolio story
First network-science project — genuinely new technique (graph theory via
scipy.csgraph, no networkx), a canonical and striking result (robust-yet-fragile),
and excellent visuals (the world draws itself). Real reliable data + committed
snapshot. Pure numpy/pandas/scipy/matplotlib, no new deps.

## Deliberately deferred / next
- Betweenness centrality / community detection; traffic-weighted edges; a financial
  correlation network.
- GRADUATE one of the ungraduated projects (confirm first).

## Status
Built and shipped 2026-11-02. 7 tests pass; verified end-to-end on the OpenFlights
snapshot. Three charts.
