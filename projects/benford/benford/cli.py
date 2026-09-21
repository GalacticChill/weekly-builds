"""Command-line entry point for the Benford's Law study."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import datasets as ds
from . import plots
from .law import analyze

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "worldbank.csv"


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="benford",
        description="Does real data follow Benford's Law, and can the leading digit "
        "flag numbers that were made up?",
    )
    p.add_argument("--data", default=str(DEFAULT_DATA), help="Path to worldbank.csv")
    p.add_argument("--collect", action="store_true",
                   help="Re-download the World Bank figures first.")
    p.add_argument("--assets-dir", default="assets", help="Directory for charts.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    if args.collect:
        print("Downloading World Bank figures...")
        wb = ds.collect_worldbank()
        ds.save(wb, args.data)
        print(f"Saved {len(wb)} values to {args.data}")
    else:
        path = Path(args.data)
        wb = ds.load(path) if path.exists() else None
        if wb is None:
            print("(no cached World Bank data; testing mathematical sets and controls)")

    suite = ds.build_datasets(wb)

    print("\nHow closely does each dataset follow Benford's Law?")
    print(f"  {'dataset':30s} {'kind':24s} {'n':>6s}  {'MAD':>6s}  verdict")
    for name, (values, kind) in suite.items():
        r = analyze(values)
        print(f"  {name:30s} {kind:24s} {r.n:6d}  {r.mad:6.3f}  {r.conforms}")

    print("\nThe first-digit '1' share (Benford predicts 30.1%):")
    for name, (values, kind) in suite.items():
        r = analyze(values)
        print(f"  {name:30s} {r.observed[0]:5.1%}")

    assets = Path(args.assets_dir)
    assets.mkdir(parents=True, exist_ok=True)
    grid = plots.plot_grid(suite, assets / "benford_grid.png")
    charts = [grid]
    if wb is not None and len(wb):
        charts.append(
            plots.plot_single(wb["value"].to_numpy(), assets / "worldbank.png",
                              title="World Bank figures (populations, GDP, area, CO2)")
        )
    print("\nSaved charts:")
    for c in charts:
        print(f"  {c}")
    print()
