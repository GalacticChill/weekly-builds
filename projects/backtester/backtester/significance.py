"""Is the edge real, or is it luck? Bootstrap significance for a backtest.

No-lookahead fixes *when* a strategy sees data. Walk-forward fixes *how you pick
its parameters*. This module closes the third gap: even a clean, out-of-sample
Sharpe is a single number estimated from a noisy, finite sample — so how sure are
we it isn't zero?

The tool is the **block bootstrap**. We resample the strategy's daily returns in
short contiguous blocks (not one day at a time), which preserves the volatility
clustering and autocorrelation that make financial returns *not* independent.
Recomputing the Sharpe on thousands of resamples traces out its sampling
distribution, which gives:

- a **confidence interval** for the Sharpe, and
- a bootstrap **p-value** for the hypothesis that the true Sharpe is > 0, obtained
  by re-centering the returns to impose the null of no edge.

We also compare a strategy against a benchmark with a *paired* bootstrap (same
resampled days for both), and provide a multiple-testing correction, because the
surest way to manufacture a significant Sharpe is to try many strategies and
report the winner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sharpe_from_returns(
    returns, periods: int = TRADING_DAYS, risk_free: float = 0.0
) -> float:
    """Annualized Sharpe computed directly from a daily-return series.

    Order-independent (uses only the mean and standard deviation), which is exactly
    what a bootstrap needs. ``risk_free`` is an annual rate.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size < 2:
        return np.nan
    excess = r - risk_free / periods
    sd = excess.std(ddof=1)
    if sd == 0:
        return np.nan
    return float(np.sqrt(periods) * excess.mean() / sd)


def block_bootstrap_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Indices for one circular block-bootstrap resample of length ``n``.

    Draws ceil(n/block) random block start points and lays down ``block``
    consecutive indices from each (wrapping around the end), then trims to ``n``.
    ``block=1`` recovers the ordinary i.i.d. bootstrap.
    """
    if block < 1:
        raise ValueError("block must be >= 1")
    if n <= 0:
        raise ValueError("n must be positive")
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
    return idx[:n]


@dataclass
class BootstrapResult:
    """Sampling distribution of a metric under the block bootstrap."""

    metric: str
    observed: float
    ci_low: float
    ci_high: float
    p_value: float          # bootstrap p-value that the true metric is > 0
    n_boot: int
    block: int
    alpha: float
    samples: np.ndarray = field(repr=False)

    @property
    def significant(self) -> bool:
        """True if the (1-alpha) CI excludes zero."""
        return self.ci_low > 0.0 or self.ci_high < 0.0


def bootstrap_metric(
    returns,
    metric_fn=sharpe_from_returns,
    n_boot: int = 2000,
    block: int = 10,
    seed: int = 42,
    alpha: float = 0.05,
    metric_name: str = "sharpe",
) -> BootstrapResult:
    """Block-bootstrap a metric of a daily-return series.

    The confidence interval comes from the ordinary (percentile) bootstrap; the
    p-value comes from a separate bootstrap on the *mean-centered* returns, which
    imposes the null hypothesis "no edge" (expected return zero) while keeping the
    return distribution's shape and volatility intact.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = r.size
    if n < 2:
        raise ValueError("Need at least 2 returns to bootstrap.")

    observed = float(metric_fn(r))

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        boot[b] = metric_fn(r[block_bootstrap_indices(n, block, rng)])
    ci_low = float(np.nanpercentile(boot, 100 * alpha / 2))
    ci_high = float(np.nanpercentile(boot, 100 * (1 - alpha / 2)))

    # Null distribution: re-center to zero mean, then bootstrap.
    r0 = r - r.mean()
    rng_null = np.random.default_rng(seed + 1)
    null = np.empty(n_boot)
    for b in range(n_boot):
        null[b] = metric_fn(r0[block_bootstrap_indices(n, block, rng_null)])
    # One-sided, bias-corrected: how often does the no-edge world beat what we saw?
    p_value = float((np.sum(null >= observed) + 1) / (n_boot + 1))

    return BootstrapResult(
        metric=metric_name,
        observed=observed,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        n_boot=n_boot,
        block=block,
        alpha=alpha,
        samples=boot,
    )


@dataclass
class CompareResult:
    """Paired-bootstrap comparison of two strategies' Sharpe ratios."""

    observed_diff: float    # sharpe(a) - sharpe(b)
    ci_low: float
    ci_high: float
    p_value: float          # bootstrap p that A does NOT beat B (diff <= 0)
    n_boot: int
    block: int
    alpha: float
    samples: np.ndarray = field(repr=False)

    @property
    def significant(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0


def compare_sharpe(
    returns_a,
    returns_b,
    n_boot: int = 2000,
    block: int = 10,
    seed: int = 42,
    alpha: float = 0.05,
    periods: int = TRADING_DAYS,
) -> CompareResult:
    """Paired block bootstrap of the Sharpe *difference* between two strategies.

    Both series are resampled with the *same* block indices on each draw, so common
    market shocks stay aligned and we measure the difference that survives them.
    The p-value is the bootstrap mass at or below zero (strategy A failing to beat
    B), one-sided.
    """
    a = np.asarray(returns_a, dtype=float)
    b = np.asarray(returns_b, dtype=float)
    m = min(a.size, b.size)
    a, b = a[-m:], b[-m:]           # align on the common tail
    if m < 2:
        raise ValueError("Need at least 2 aligned returns to compare.")

    observed = sharpe_from_returns(a, periods) - sharpe_from_returns(b, periods)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = block_bootstrap_indices(m, block, rng)     # shared -> paired
        diffs[i] = sharpe_from_returns(a[idx], periods) - sharpe_from_returns(
            b[idx], periods
        )

    ci_low = float(np.nanpercentile(diffs, 100 * alpha / 2))
    ci_high = float(np.nanpercentile(diffs, 100 * (1 - alpha / 2)))
    p_value = float((np.sum(diffs <= 0) + 1) / (n_boot + 1))

    return CompareResult(
        observed_diff=float(observed),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        n_boot=n_boot,
        block=block,
        alpha=alpha,
        samples=diffs,
    )


def sidak_pvalue(p: float, n_trials: int) -> float:
    """Multiple-testing correction: the chance at least one of ``n_trials``
    independent attempts looks this good by luck. ``1 - (1 - p)**n_trials``.

    If you quietly tried 20 strategies and reported the best one's p = 0.03, its
    honest p is 1 - 0.97**20 = 0.46 — a coin flip. This is the statistical echo of
    the walk-forward overfitting tax: searching inflates significance.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    return float(1.0 - (1.0 - p) ** n_trials)
