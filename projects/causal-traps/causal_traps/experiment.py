"""Repeat each scenario many times, so a result is a pattern, not a fluke.

One simulated dataset can mislead; averaging the naive and adjusted estimates over
hundreds of fresh draws shows the *systematic* bias of each approach and compares it
to the known truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dgp import SCENARIOS
from .estimate import effect


@dataclass
class ScenarioResult:
    name: str
    true_effect: float
    extra: str
    adjusting_is_right: bool
    naive_mean: float
    naive_std: float
    adjusted_mean: float
    adjusted_std: float
    note: str

    @property
    def correct_estimate(self) -> float:
        return self.adjusted_mean if self.adjusting_is_right else self.naive_mean

    @property
    def wrong_estimate(self) -> float:
        return self.naive_mean if self.adjusting_is_right else self.adjusted_mean


def run_scenario(name: str, trials: int = 300, n: int = 4000,
                 seed: int = 0) -> ScenarioResult:
    """Monte-Carlo the naive and adjusted effect estimates for one scenario."""
    maker = SCENARIOS[name]
    naive, adjusted = [], []
    extra = true = None
    adj_right = True
    note = ""
    for t in range(trials):
        sc = maker(n=n, seed=seed + t)
        extra, true, adj_right, note = (
            sc.extra, sc.true_effect, sc.adjusting_is_right, sc.note
        )
        naive.append(effect(sc.data, adjust_for=None))
        adjusted.append(effect(sc.data, adjust_for=sc.extra))
    naive = np.array(naive)
    adjusted = np.array(adjusted)
    return ScenarioResult(
        name=name, true_effect=float(true), extra=extra, adjusting_is_right=adj_right,
        naive_mean=float(naive.mean()), naive_std=float(naive.std()),
        adjusted_mean=float(adjusted.mean()), adjusted_std=float(adjusted.std()),
        note=note,
    )


def run_all(trials: int = 300, n: int = 4000, seed: int = 0) -> dict[str, ScenarioResult]:
    return {name: run_scenario(name, trials=trials, n=n, seed=seed) for name in SCENARIOS}
