"""Offline tests for simulation and empirical-Bayes shrinkage."""

from __future__ import annotations

import numpy as np
import pandas as pd

from small_samples import (
    Prior,
    empirical_bayes,
    fit_prior,
    naive_leader,
    rank_correlation,
    rmse,
    rmse_by_sample_size,
    shrink,
    simulate,
)


# --- simulation ------------------------------------------------------------- #

def test_simulate_shapes_and_consistency():
    df = simulate(n_players=200, seed=1)
    assert len(df) == 200
    assert np.allclose(df["raw_rate"], df["successes"] / df["trials"])
    assert (df["true_rate"] >= 0).all() and (df["true_rate"] <= 1).all()
    assert (df["successes"] <= df["trials"]).all()


def test_simulate_reproducible():
    a = simulate(n_players=100, seed=5)
    b = simulate(n_players=100, seed=5)
    assert a.equals(b)


# --- prior fitting ---------------------------------------------------------- #

def test_fit_prior_recovers_truth():
    # Simulated from Beta(8, 22) (mean 0.267); MLE should land close on enough data.
    df = simulate(n_players=3000, prior_a=8.0, prior_b=22.0, seed=2)
    prior = fit_prior(df["trials"], df["successes"])
    assert abs(prior.mean - 8.0 / 30.0) < 0.04
    assert 15 < prior.strength < 70          # ballpark of the true 30


# --- shrinkage direction ---------------------------------------------------- #

def test_posterior_mean_lies_between_raw_and_prior():
    df = simulate(n_players=300, seed=3)
    shrunk, prior = empirical_bayes(df)
    lo = np.minimum(shrunk["raw_rate"], prior.mean)
    hi = np.maximum(shrunk["raw_rate"], prior.mean)
    assert (shrunk["shrunk_rate"] >= lo - 1e-9).all()
    assert (shrunk["shrunk_rate"] <= hi + 1e-9).all()


def test_small_samples_shrink_more_than_large():
    # Two perfect records, 1/1 vs 50/50: the larger sample stays closer to raw (1.0).
    df = pd.DataFrame({
        "player": ["tiny", "big"],
        "true_rate": [0.5, 0.5],
        "trials": [1, 50],
        "successes": [1, 50],
        "raw_rate": [1.0, 1.0],
    })
    prior = Prior(alpha=8.0, beta=22.0)          # mean 0.26, well below 1.0
    out = shrink(df, prior)
    tiny = out.set_index("player").loc["tiny", "shrunk_rate"]
    big = out.set_index("player").loc["big", "shrunk_rate"]
    assert tiny < big                            # tiny pulled harder toward prior
    assert tiny < 0.5 and big > 0.7


def test_credible_interval_brackets_estimate():
    df = simulate(n_players=200, seed=4)
    shrunk, _ = empirical_bayes(df)
    assert (shrunk["ci_low"] <= shrunk["shrunk_rate"] + 1e-9).all()
    assert (shrunk["shrunk_rate"] <= shrunk["ci_high"] + 1e-9).all()
    assert (shrunk["ci_low"] >= 0).all() and (shrunk["ci_high"] <= 1).all()


# --- the headline claim ----------------------------------------------------- #

def test_shrinkage_beats_raw_against_truth():
    df = simulate(n_players=1000, seed=0)
    shrunk, _ = empirical_bayes(df)
    raw_rmse = rmse(shrunk["raw_rate"], shrunk["true_rate"])
    shrunk_rmse = rmse(shrunk["shrunk_rate"], shrunk["true_rate"])
    assert shrunk_rmse < raw_rmse                # shrinkage is closer to the truth


def test_improvement_is_largest_for_small_samples():
    df = simulate(n_players=1500, seed=0)
    shrunk, _ = empirical_bayes(df)
    table = rmse_by_sample_size(shrunk)
    small = table.iloc[0]["improvement"]         # fewest trials
    large = table.iloc[-1]["improvement"]        # most trials
    assert small > large


def test_shrinkage_improves_rank_correlation():
    df = simulate(n_players=1000, seed=0)
    shrunk, _ = empirical_bayes(df)
    corr = rank_correlation(shrunk)
    assert corr["shrunk"] > corr["raw"]


def test_naive_leader_has_highest_raw_rate():
    df = simulate(n_players=500, seed=7)
    shrunk, _ = empirical_bayes(df)
    leader = naive_leader(shrunk)
    assert leader["raw_rate"] == shrunk["raw_rate"].max()
