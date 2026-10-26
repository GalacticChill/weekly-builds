"""Offline tests for the causal-traps study — synthetic, known-answer."""

from __future__ import annotations

import numpy as np

from causal_traps import (
    collider,
    confounded,
    effect,
    mediated,
    ols,
    run_scenario,
)


# --- OLS sanity ------------------------------------------------------------- #

def test_ols_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=500)
    x2 = rng.normal(size=500)
    y = 2.0 + 3.0 * x1 - 1.0 * x2            # no noise -> exact recovery
    X = np.column_stack([np.ones(500), x1, x2])
    fit = ols(y, X, ["intercept", "x1", "x2"])
    assert np.isclose(fit.of("intercept"), 2.0)
    assert np.isclose(fit.of("x1"), 3.0)
    assert np.isclose(fit.of("x2"), -1.0)


# --- true-effect bookkeeping ------------------------------------------------ #

def test_true_effects_and_flags():
    assert np.isclose(confounded().true_effect, 0.5)
    assert confounded().adjusting_is_right is True
    assert np.isclose(collider().true_effect, 0.0)
    assert collider().adjusting_is_right is False
    med = mediated(direct=0.2, a=0.7, b=0.6)
    assert np.isclose(med.true_effect, 0.2 + 0.7 * 0.6)   # total = direct + a*b
    assert med.adjusting_is_right is False


# --- the three lessons ------------------------------------------------------ #

def test_confounder_adjustment_recovers_truth():
    r = run_scenario("confounder", trials=40, n=4000, seed=0)
    assert abs(r.adjusted_mean - r.true_effect) < 0.05    # adjusting fixes it
    assert r.naive_mean - r.true_effect > 0.2             # naive is biased upward


def test_collider_adjustment_creates_bias():
    r = run_scenario("collider", trials=40, n=4000, seed=0)
    assert abs(r.naive_mean - r.true_effect) < 0.05       # naive is already correct
    assert abs(r.adjusted_mean - r.true_effect) > 0.2     # adjusting invents an effect


def test_mediator_adjustment_gives_direct_not_total():
    r = run_scenario("mediator", trials=40, n=4000, seed=0)
    assert abs(r.naive_mean - r.true_effect) < 0.05       # naive = total effect
    assert r.adjusted_mean < r.true_effect - 0.2          # adjusted = smaller direct


def test_correct_estimate_matches_truth_each_scenario():
    for name in ("confounder", "collider", "mediator"):
        r = run_scenario(name, trials=40, n=4000, seed=1)
        assert abs(r.correct_estimate - r.true_effect) < 0.06
        # ...and the other approach is meaningfully off.
        assert abs(r.wrong_estimate - r.true_effect) > 0.15


# --- plumbing --------------------------------------------------------------- #

def test_effect_single_draw_runs():
    sc = confounded(n=1000, seed=3)
    naive = effect(sc.data)
    adjusted = effect(sc.data, adjust_for="C")
    assert naive > adjusted                                # confounding inflates naive


def test_run_scenario_reproducible():
    a = run_scenario("collider", trials=20, n=2000, seed=5)
    b = run_scenario("collider", trials=20, n=2000, seed=5)
    assert a.naive_mean == b.naive_mean
    assert a.adjusted_mean == b.adjusted_mean
