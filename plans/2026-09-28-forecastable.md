# Week of 2026-09-28 — Forecastable? (time-series forecasting)

**Theme:** data/ML — the portfolio's first dedicated TIME-SERIES project.
**Type:** new technique (temporal forecasting + honest out-of-sample validation);
user delegated ("do the next one"), chosen for standalone merit.

## The question
What can you actually forecast in a time series, and what only looks forecastable?
Contrast a genuinely predictable series (NOAA Mauna Loa CO2 — strong trend +
seasonality) against a random walk (structure by accident only).

## Scope (shipped)
- `data.py` — fetch monthly CO2 from NOAA (cached `data/co2.csv`, 822 months
  1958-2026); reproducible drift-free random walks.
- `forecast.py` — baselines (`naive_last`, `seasonal_naive`), model `trend_seasonal`
  (least-squares QUADRATIC trend + monthly dummies — CO2 accelerates, so degree-2 is
  honest), MAE/RMSE/MAPE, held-out `evaluate` (train on past, forecast the untouched
  future — never shuffle), and `random_walk_experiment` (average over 300 seeds).
- `plots.py` — forecast-vs-actual for CO2 and a representative random walk, plus a
  log-scale error-comparison bar chart.
- CLI + committed CO2 snapshot.
- 11 offline tests (synthetic known-answer + the multi-seed honesty claim).

## Findings
- **CO2 is genuinely predictable:** trend+seasonal MAE 1.15 ppm (MAPE 0.27%)
  forecasting 60 months ahead, ~8x better than naive-last (8.9) and seasonal-naive
  (7.58). The model tracks trend + seasonal cycle years out.
- **Random walk is not:** averaged over 300 seeds, naive-last has the LOWEST mean MAE
  (4.30) while trend+seasonal is WORST (8.83) — it extrapolates spurious trends. The
  flexible model beats naive only 23% of the time (no reliable edge).
- **One example can lie:** on seed 0 the model happens to win; that's why the claim
  is made over 300 seeds, not one. The plotted walk (seed 6) is representative.

## The honest core
- Never shuffle time series; always forecast forward; always beat a naive baseline.
- The quadratic trend is disclosed as a choice that's right for CO2 but dangerous on
  structureless data (extrapolates parabolas) — flexibility helps only when there's
  real structure.
- Financial prices behave like the random walk, not CO2 — the efficient-market
  lesson, tying back to the backtester's no-lookahead ethos.

## Why this matters for the portfolio story
Adds the missing time-series technique with proper temporal validation, and a
memorable, honest "predictable vs. only-looks-predictable" result. Reliable real
data (NOAA) + committed snapshot. No new dependencies.

## Deliberately deferred / next
- Rolling-origin (expanding-window) multi-horizon backtest; prediction intervals.
- GRADUATE word-prints / benford / this to standalone repos.
- Another original capstone (user enjoys steering direction).

## Status
Built and shipped 2026-09-28. 11 tests pass; verified end-to-end on a freshly
fetched 822-month CO2 series plus random walks. Three charts.
