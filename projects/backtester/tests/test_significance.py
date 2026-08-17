"""Offline tests for bootstrap significance — synthetic returns, fixed seeds.

The statistical claims are the product here, so the tests pin them down: a real
edge must come back significant, pure noise must not, and a strategy that only
looks better by luck must fail the paired comparison.
"""

from __future__ import annotations

import numpy as np

from backtester import significance as sig


def _returns(mean_daily, vol_daily, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(mean_daily, vol_daily, size=n)


# --------------------------------------------------------------------------- #
# Sharpe-from-returns and the block sampler.                                   #
# --------------------------------------------------------------------------- #

def test_sharpe_from_returns_hand_value():
    # Constant positive returns -> zero std -> undefined Sharpe (nan), by design.
    assert np.isnan(sig.sharpe_from_returns([0.01, 0.01, 0.01]))
    # A known series: mean and std computed by hand, annualized by sqrt(252).
    r = np.array([0.01, -0.01, 0.02, -0.02, 0.03])
    expected = np.sqrt(252) * r.mean() / r.std(ddof=1)
    assert np.isclose(sig.sharpe_from_returns(r), expected)


def test_block_indices_shape_and_range():
    rng = np.random.default_rng(0)
    idx = sig.block_bootstrap_indices(100, block=7, rng=rng)
    assert idx.shape == (100,)
    assert idx.min() >= 0 and idx.max() < 100


def test_block_indices_are_contiguous_within_blocks():
    rng = np.random.default_rng(1)
    block = 5
    idx = sig.block_bootstrap_indices(50, block=block, rng=rng)
    # Within each laid-down block, indices step by +1 (mod n).
    for start in range(0, 50, block):
        chunk = idx[start:start + block]
        steps = (np.diff(chunk)) % 50
        assert np.all(steps == 1)


# --------------------------------------------------------------------------- #
# The core claims.                                                             #
# --------------------------------------------------------------------------- #

def test_real_edge_is_significant():
    # Strong positive mean, modest vol -> Sharpe well above zero and detectable.
    r = _returns(mean_daily=0.0009, vol_daily=0.01, n=1500, seed=3)
    res = sig.bootstrap_metric(r, n_boot=1000, block=10, seed=7)
    assert res.observed > 0
    assert res.significant            # 95% CI excludes zero
    assert res.ci_low > 0
    assert res.p_value < 0.05


def test_pure_noise_is_not_significant():
    # Zero-mean returns: no edge exists, so the CI must straddle zero and the
    # p-value must not clear the 0.05 bar.
    r = _returns(mean_daily=0.0, vol_daily=0.01, n=1500, seed=11)
    res = sig.bootstrap_metric(r, n_boot=1000, block=10, seed=5)
    assert res.ci_low < 0 < res.ci_high
    assert not res.significant
    assert res.p_value > 0.05


def test_bootstrap_is_deterministic_under_seed():
    r = _returns(0.0005, 0.01, n=800, seed=2)
    a = sig.bootstrap_metric(r, n_boot=500, block=8, seed=99)
    b = sig.bootstrap_metric(r, n_boot=500, block=8, seed=99)
    assert a.observed == b.observed
    assert a.ci_low == b.ci_low and a.ci_high == b.ci_high
    assert a.p_value == b.p_value


def test_block_one_is_iid_bootstrap():
    # With block=1 the sampler is the ordinary i.i.d. bootstrap; it should still run
    # and return a finite CI.
    r = _returns(0.0006, 0.01, n=600, seed=4)
    res = sig.bootstrap_metric(r, n_boot=400, block=1, seed=1)
    assert np.isfinite(res.ci_low) and np.isfinite(res.ci_high)


# --------------------------------------------------------------------------- #
# Paired comparison and multiple-testing correction.                          #
# --------------------------------------------------------------------------- #

def test_compare_detects_a_real_outperformer():
    a = _returns(mean_daily=0.0012, vol_daily=0.008, n=2000, seed=100)  # better
    b = _returns(mean_daily=0.0000, vol_daily=0.008, n=2000, seed=101)  # worse
    res = sig.compare_sharpe(a, b, n_boot=1000, block=10, seed=8)
    assert res.observed_diff > 0
    assert res.significant
    assert res.p_value < 0.05


def test_compare_finds_no_difference_between_twins():
    # Same distribution, different draws: the difference should not be significant.
    a = _returns(0.0004, 0.01, n=1500, seed=30)
    b = _returns(0.0004, 0.01, n=1500, seed=31)
    res = sig.compare_sharpe(a, b, n_boot=1000, block=10, seed=9)
    assert res.ci_low < 0 < res.ci_high
    assert res.p_value > 0.05


def test_sidak_correction_properties():
    p = 0.03
    # One trial changes nothing; more trials only inflate the p-value; bounded <= 1.
    assert np.isclose(sig.sidak_pvalue(p, 1), p)
    assert sig.sidak_pvalue(p, 20) > p
    assert sig.sidak_pvalue(p, 20) <= 1.0
    # Twenty shots at p=0.03 is roughly a coin flip.
    assert 0.4 < sig.sidak_pvalue(p, 20) < 0.5
