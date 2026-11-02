"""Command-line entry point for the flight-network study."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import data as dt
from . import graph as gr
from . import plots
from . import robustness as rb

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data"


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="flight-network",
        description="The world air-route network: hubs, six degrees, and why it is "
        "robust to random failure but fragile to hub attacks.",
    )
    p.add_argument("--data", default=str(DEFAULT_DATA), help="Directory with the CSVs.")
    p.add_argument("--collect", action="store_true",
                   help="Re-download and re-process OpenFlights first.")
    p.add_argument("--seed", type=int, default=0, help="Seed for the random attack.")
    p.add_argument("--assets-dir", default="assets", help="Directory for charts.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    if args.collect:
        print("Downloading OpenFlights...")
        edges, airports = dt.build_snapshot()
        dt.save(edges, airports, args.data)
        print(f"Saved {len(edges)} edges, {len(airports)} airports to {args.data}")
    else:
        edges, airports = dt.load(args.data)

    adj, nodes, meta = dt.build_graph(edges, airports)
    print(f"\nFlight network: {len(nodes)} airports, {len(edges)} routes")

    print("\nBusiest hubs (direct destinations):")
    for code, r in meta.sort_values("degree", ascending=False).head(8).iterrows():
        print(f"  {code}  {r['degree']:4d}  {r['city']}, {r['country']}")

    paths = gr.average_path_length(adj, seed=args.seed)
    giant = gr.giant_component_fraction(adj)
    print(f"\nConnectedness:")
    print(f"  Giant component: {giant:.1%} of all airports")
    print(f"  Average hops between airports: {paths['mean']:.2f}  "
          f"(longest shortest-path sampled: {int(paths['max'])})")
    print(f"  -> you can get between almost any two airports in about "
          f"{paths['mean']:.0f} flights.")

    random_curve = rb.random_attack(adj, seed=args.seed)
    targeted_curve = rb.targeted_attack(adj)
    r_collapse = rb.collapse_point(random_curve)
    t_collapse = rb.collapse_point(targeted_curve)
    print("\nRobust yet fragile (fraction of airports removed to break the network "
          "below 50% connected):")
    print(f"  random failures:      {r_collapse:.0%}" if r_collapse == r_collapse
          else "  random failures:      never within 60%")
    print(f"  targeted (hubs first): {t_collapse:.0%}")

    assets = Path(args.assets_dir)
    assets.mkdir(parents=True, exist_ok=True)
    charts = [
        plots.plot_degree_distribution(adj, meta, assets / "degree_distribution.png"),
        plots.plot_robustness(random_curve, targeted_curve, assets / "robustness.png"),
        plots.plot_hub_map(meta, assets / "hub_map.png"),
    ]
    print("\nSaved charts:")
    for c in charts:
        print(f"  {c}")
    print()
