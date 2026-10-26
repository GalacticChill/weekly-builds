"""Command-line entry point for the causal-traps study."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import plots
from .experiment import run_all


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="causal-traps",
        description="Correlation, causation, and the control-for-everything trap: "
        "when adjusting for a variable fixes bias, and when it creates it.",
    )
    p.add_argument("--trials", type=int, default=300, help="Monte-Carlo trials each.")
    p.add_argument("--n", type=int, default=4000, help="Sample size per trial.")
    p.add_argument("--seed", type=int, default=0, help="Base random seed.")
    p.add_argument("--assets-dir", default="assets", help="Directory for charts.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    results = run_all(trials=args.trials, n=args.n, seed=args.seed)

    print(f"\nCausal traps: {args.trials} simulations each, n={args.n} per simulation\n")
    print(f"  {'scenario':12s} {'3rd var':8s} {'true':>7s} {'naive':>8s} "
          f"{'adjusted':>9s}  what to do")
    for name, r in results.items():
        action = (f"adjust for {r.extra}" if r.adjusting_is_right
                  else f"do NOT adjust for {r.extra}")
        print(f"  {name:12s} {r.extra:8s} {r.true_effect:7.2f} {r.naive_mean:8.2f} "
              f"{r.adjusted_mean:9.2f}  {action}")

    print("\nRead it as:")
    for name, r in results.items():
        print(f"  - {name}: {r.note}")
        print(f"      correct estimate {r.correct_estimate:+.2f} vs the truth "
              f"{r.true_effect:+.2f}; the other approach gives {r.wrong_estimate:+.2f}.")

    print("\nThe lesson: you cannot tell a confounder from a collider from the data")
    print("alone -- the same T, Y, and third column can demand opposite handling.")
    print("'Control for everything' is not a strategy; the causal graph is.")

    assets = Path(args.assets_dir)
    assets.mkdir(parents=True, exist_ok=True)
    charts = [
        plots.plot_bias_summary(results, assets / "bias_summary.png"),
        plots.plot_collider_illusion(assets / "collider_illusion.png", seed=args.seed),
        plots.plot_dags(assets / "dags.png"),
    ]
    print("\nSaved charts:")
    for c in charts:
        print(f"  {c}")
    print()
