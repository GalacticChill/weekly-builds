"""Command-line entry point for the Schelling segregation study."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import experiment as ex
from . import plots
from .model import make_grid, run, segregation


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="schelling",
        description="How much personal preference does it take to segregate a city? "
        "An agent-based simulation.",
    )
    p.add_argument("--size", type=int, default=50, help="Grid side length.")
    p.add_argument("--empty", type=float, default=0.1, help="Fraction of empty cells.")
    p.add_argument("--tolerance", type=float, default=0.33,
                   help="Same-type neighbour share each agent wants. Default 0.33.")
    p.add_argument("--trials", type=int, default=5, help="Trials per tolerance (sweep).")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--assets-dir", default="assets", help="Directory for charts.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    grid = make_grid(size=args.size, empty_frac=args.empty, seed=args.seed)
    result = run(grid, tolerance=args.tolerance, seed=args.seed)

    print(f"\nSchelling model: {args.size}x{args.size} grid, "
          f"tolerance = {args.tolerance:.0%}")
    print(f"  Start segregation:  {result.initial_segregation:.2f}  (random mix)")
    print(f"  Final segregation:  {result.final_segregation:.2f}")
    print(f"  Rounds to settle:   {result.rounds}"
          f"{' (converged)' if result.converged else ' (hit cap)'}")
    print(f"  Agents want only {args.tolerance:.0%} same-type neighbours — happy to be "
          f"a {1 - args.tolerance:.0%} minority — yet the city splits.")

    print("\nSweeping how fussy the agents are...")
    sweep = ex.sweep(trials=args.trials, size=min(args.size, 40), seed=args.seed)
    print(f"  {'tolerance':>10s} {'final segregation':>20s}")
    for tol, seg in zip(sweep.tolerances, sweep.final_segregation):
        bar = "#" * int(round(seg * 40))
        print(f"  {tol:10.0%} {seg:20.2f}  {bar}")
    print(f"\n  Random-start baseline: {sweep.baseline:.2f}")
    print(f"  A 'divided city' (segregation >= 0.70) appears once agents want just "
          f"{sweep.tipping_tolerance:.0%} of their neighbours to match.")

    assets = Path(args.assets_dir)
    assets.mkdir(parents=True, exist_ok=True)
    charts = [
        plots.plot_before_after(result.initial, result.final, args.tolerance,
                                assets / "before_after.png"),
        plots.plot_tipping_curve(sweep, assets / "tipping_curve.png"),
        plots.plot_convergence(result, assets / "convergence.png"),
    ]
    print("\nSaved charts:")
    for c in charts:
        print(f"  {c}")
    print()
