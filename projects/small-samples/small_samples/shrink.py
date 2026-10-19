"""Empirical Bayes: borrow strength from the crowd to tame small samples.

If you had a good prior belief about how success rates are distributed, Bayes' rule
would tell you to blend each player's noisy raw rate with that prior -- trusting the
data more when there's a lot of it, and the prior more when there's little. Empirical
Bayes does exactly that, with one twist: it *estimates the prior from the data
itself*, by fitting a single Beta distribution to the whole population's results.

Concretely, with a Beta(alpha, beta) prior the posterior for a player with `s`
successes in `n` trials is Beta(alpha + s, beta + n - s), whose mean is::

    (alpha + s) / (alpha + beta + n)

Read that formula and the whole idea falls out: it's as if every player starts with
`alpha` phantom successes and `beta` phantom failures. For a 5-for-5 player those
phantom failures drag 100% sharply down to earth; for a 150-of-500 player they barely
move the needle. The prior is a crowd of imaginary trials, and small samples are the
ones it outvotes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist
from scipy.stats import betabinom


@dataclass
class Prior:
    """The fitted Beta prior shared across all players."""

    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def strength(self) -> float:
        """alpha + beta -- the prior's weight, in units of phantom trials."""
        return self.alpha + self.beta


def fit_prior(trials, successes) -> Prior:
    """Fit Beta(alpha, beta) by maximum likelihood on the Beta-Binomial marginal.

    This is the principled empirical-Bayes step: the marginal likelihood already
    accounts for binomial noise, so (unlike naively fitting a Beta to the raw rates)
    it doesn't mistake small-sample jitter for real spread in the true rates.
    """
    n = np.asarray(trials, dtype=int)
    s = np.asarray(successes, dtype=int)

    def neg_log_likelihood(log_params):
        a, b = np.exp(log_params)           # optimize in log-space to keep a,b > 0
        return -np.sum(betabinom.logpmf(s, n, a, b))

    # Start from a method-of-moments guess based on the pooled rate.
    p0 = max(min(s.sum() / n.sum(), 0.99), 0.01)
    start = np.log([p0 * 10.0, (1 - p0) * 10.0])
    res = minimize(neg_log_likelihood, start, method="Nelder-Mead")
    a, b = np.exp(res.x)
    return Prior(alpha=float(a), beta=float(b))


def shrink(df: pd.DataFrame, prior: Prior, cred: float = 0.90) -> pd.DataFrame:
    """Add posterior (shrunken) estimates and a credible interval per player."""
    s = df["successes"].to_numpy()
    n = df["trials"].to_numpy()
    post_a = prior.alpha + s
    post_b = prior.beta + (n - s)

    lo = (1 - cred) / 2
    out = df.copy()
    out["shrunk_rate"] = post_a / (post_a + post_b)
    out["ci_low"] = beta_dist.ppf(lo, post_a, post_b)
    out["ci_high"] = beta_dist.ppf(1 - lo, post_a, post_b)
    return out


def empirical_bayes(df: pd.DataFrame, cred: float = 0.90) -> tuple[pd.DataFrame, Prior]:
    """Fit the prior from the data and return shrunken estimates for every player."""
    prior = fit_prior(df["trials"], df["successes"])
    return shrink(df, prior, cred=cred), prior
