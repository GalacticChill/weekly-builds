"""Sweep the one knob that matters: how fussy are the agents?

Run the model across a range of `tolerance` values, several random trials each, and
record the segregation the city settles into. The shape of that curve is the whole
point — it shows how little preference it takes to tip a mixed city into a divided
one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import make_grid, run, segregation


@dataclass
class SweepResult:
    tolerances: np.ndarray
    final_segregation: np.ndarray      # mean over trials
    final_segregation_std: np.ndarray
    baseline: float                    # segregation of the random starting grid

    @property
    def tipping_tolerance(self) -> float:
        """Lowest tolerance at which mean segregation clears 0.7 (a divided city)."""
        over = self.tolerances[self.final_segregation >= 0.7]
        return float(over.min()) if over.size else float("nan")


def sweep(tolerances=None, trials: int = 5, size: int = 40, empty_frac: float = 0.1,
          share_a: float = 0.5, max_rounds: int = 100, seed: int = 0) -> SweepResult:
    """Average final segregation across trials for each tolerance value."""
    if tolerances is None:
        tolerances = np.round(np.arange(0.0, 0.90, 0.1), 3)
    tolerances = np.asarray(tolerances, dtype=float)

    means, stds, baselines = [], [], []
    for ti, tol in enumerate(tolerances):
        finals = []
        for trial in range(trials):
            s = seed + 1000 * ti + trial
            grid = make_grid(size=size, empty_frac=empty_frac, share_a=share_a, seed=s)
            baselines.append(segregation(grid))
            result = run(grid, tolerance=float(tol), max_rounds=max_rounds, seed=s)
            finals.append(result.final_segregation)
        means.append(np.mean(finals))
        stds.append(np.std(finals))

    return SweepResult(
        tolerances=tolerances,
        final_segregation=np.array(means),
        final_segregation_std=np.array(stds),
        baseline=float(np.mean(baselines)),
    )
