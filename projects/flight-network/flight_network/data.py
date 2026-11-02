"""Build the world air-route network from the OpenFlights dataset.

Nodes are airports; an undirected edge joins two airports if at least one direct
(non-stop) route connects them. We keep only airports that both appear in a route
and carry metadata (IATA code, name, coordinates). The processed edge and node lists
are cached as small CSVs so the analysis runs offline and reproducibly; `--collect`
refreshes them from OpenFlights.

Data vintage note: OpenFlights is a community snapshot (routes last bulk-updated
around 2014), so this is the shape of the network then, not today's exact schedule.
"""

from __future__ import annotations

import csv
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

_BASE = "https://raw.githubusercontent.com/jpatokal/openflights/master/data"
_UA = {"User-Agent": "Mozilla/5.0 (network research)"}


def _fetch(name: str) -> str:
    req = urllib.request.Request(f"{_BASE}/{name}", headers=_UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", "replace")


def _parse_airports(text: str) -> dict[str, dict]:
    """IATA code -> metadata (name, city, country, lat, lon)."""
    meta = {}
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 8:
            continue
        iata = row[4].strip()
        if not iata or iata == "\\N" or len(iata) != 3:
            continue
        try:
            lat, lon = float(row[6]), float(row[7])
        except ValueError:
            continue
        meta[iata] = {"name": row[1], "city": row[2], "country": row[3],
                      "lat": lat, "lon": lon}
    return meta


def _parse_edges(text: str, valid: set[str]) -> set[tuple[str, str]]:
    """Unique undirected airport pairs from non-stop routes."""
    edges = set()
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 8:
            continue
        src, dst, stops = row[2].strip(), row[4].strip(), row[7].strip()
        if stops != "0" or src == dst or src not in valid or dst not in valid:
            continue
        edges.add((src, dst) if src < dst else (dst, src))
    return edges


def build_snapshot() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download and process OpenFlights into (edges_df, airports_df)."""
    meta = _parse_airports(_fetch("airports.dat"))
    edges = _parse_edges(_fetch("routes.dat"), set(meta))
    edges_df = pd.DataFrame(sorted(edges), columns=["src", "dst"])
    used = sorted(set(edges_df["src"]) | set(edges_df["dst"]))
    airports_df = pd.DataFrame(
        [{"iata": k, **meta[k]} for k in used]
    ).set_index("iata")
    return edges_df, airports_df


def save(edges_df: pd.DataFrame, airports_df: pd.DataFrame, data_dir: str | Path) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    edges_df.to_csv(data_dir / "edges.csv", index=False)
    airports_df.to_csv(data_dir / "airports.csv")


def load(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    edges_df = pd.read_csv(data_dir / "edges.csv")
    airports_df = pd.read_csv(data_dir / "airports.csv", index_col="iata")
    return edges_df, airports_df


def build_graph(edges_df: pd.DataFrame, airports_df: pd.DataFrame):
    """Return (adjacency CSR, node list, airports-with-degree DataFrame).

    The adjacency is symmetric 0/1 with no self-loops; nodes are the airports that
    appear in at least one edge, in a stable sorted order.
    """
    nodes = sorted(set(edges_df["src"]) | set(edges_df["dst"]))
    idx = {code: i for i, code in enumerate(nodes)}
    n = len(nodes)

    rows, cols = [], []
    for s, d in zip(edges_df["src"], edges_df["dst"]):
        i, j = idx[s], idx[d]
        rows += [i, j]
        cols += [j, i]
    data = np.ones(len(rows), dtype=np.int8)
    adj = csr_matrix((data, (rows, cols)), shape=(n, n))
    adj.data[:] = 1  # guard against any accidental duplicate summing

    degree = np.asarray(adj.sum(axis=1)).ravel()
    meta = airports_df.reindex(nodes).copy()
    meta["degree"] = degree
    return adj, nodes, meta
