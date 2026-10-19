"""A population of 'players' with known true rates but unequal sample sizes.

The whole point of the project is that we can only ever *observe* someone's raw
success rate, never their true skill. So here we play God: draw each player a true
success probability from a shared Beta distribution, give them a wildly varying
number of trials (a handful for some, hundreds for others), and simulate their
successes. Because we kept the true rates, we can later check which estimator
recovers them best -- the honest test a real dataset can never give you.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate(
    n_players: int = 1000,
    prior_a: float = 8.0,
    prior_b: float = 22.0,
    min_trials: int = 1,
    max_trials: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """Draw true rates ~ Beta(prior_a, prior_b), then observe noisy successes.

    Trial counts are drawn on a log scale so the population is dominated by
    small-sample players (the interesting, dangerous case) with a long tail of
    well-measured ones. Returns one row per player.
    """
    rng = np.random.default_rng(seed)

    true_rate = rng.beta(prior_a, prior_b, size=n_players)
    log_lo, log_hi = np.log10(min_trials), np.log10(max_trials)
    trials = np.round(10 ** rng.uniform(log_lo, log_hi, size=n_players)).astype(int)
    trials = np.clip(trials, min_trials, max_trials)
    successes = rng.binomial(trials, true_rate)

    return pd.DataFrame(
        {
            "player": [f"P{i:04d}" for i in range(n_players)],
            "true_rate": true_rate,
            "trials": trials,
            "successes": successes,
            "raw_rate": successes / trials,
        }
    )


def prior_mean(prior_a: float = 8.0, prior_b: float = 22.0) -> float:
    """The mean of the Beta prior used by `simulate` (handy for tests/plots)."""
    return prior_a / (prior_a + prior_b)
