# Week of 2026-08-31 — Capstone: Life & Money

**Theme:** data/ML — an original "society & the real world" capstone (user's pick).
**Type:** a fresh, personal, differentiating capstone in a new domain (global
development), distinct from the finance body of work.

## Origin
The user asked for an original capstone and chose the "society & the real world"
direction. I framed it as a genuinely meaningful, non-obvious question with a
reliable public dataset (World Bank Open Data, no API key).

## The question
Does money buy a longer life? Three parts:
1. **Preston curve** — is the income->longevity relationship linear or diminishing?
2. **Beyond income** — what else predicts life expectancy, out-of-sample?
3. **Residuals** — which countries live far longer (or shorter) than their wealth
   predicts, and what do those groups have in common?

## Scope (shipped)
- `collect.py` — pull 9 indicators for ~217 countries from the World Bank API
  (most-recent-value-per-country over 2015-2021), drop aggregates, tag region +
  income group; cache to `data/indicators.csv` (committed, runs offline).
- `features.py` — log-transform income (the crux), median-impute predictor gaps,
  drop rows missing target/income, and DELIBERATELY exclude infant/child/adult
  mortality as leakage (arithmetic components of life expectancy).
- `model.py` — `preston_fit` (linear vs log R² + fitted curve), `full_fit`
  (standardized linear + random forest, out-of-sample R²), `income_residuals`
  (actual - income-predicted per country, ranked).
- `plots.py` — Preston curve (log-x scatter colored by income group, fitted curve,
  annotated outliers), RF importances, diverging beats-the-odds residual bars.
- CLI + library API + committed dataset for reproducibility.
- 11 offline tests (synthetic, known-answer).

## Findings (real World Bank data, ~200 countries)
- **Preston curve confirmed:** life ~ income R² = 0.56; life ~ log(income) R² =
  0.71. About **+13 years of life per 10x of income** — steep diminishing returns.
- **Water & sanitation nearly rival income:** RF importances log_gdp 39%,
  water_access 28%, sanitation_access 22%. Out-of-sample R² ~0.70. A hopeful result:
  these are buildable without first getting rich.
- **Residuals tell a human story:** biggest shortfalls are the Southern African
  HIV/AIDS belt (Eswatini, Botswana, South Africa, Namibia) and resource-curse
  states (Equatorial Guinea -9.7 despite oil wealth, Nigeria, Guyana).

## Why this matters for the portfolio story
Original, personal, socially meaningful, and in a NEW domain (global development,
not finance) — the most differentiating kind of project for admissions. Reuses and
extends the star-signals discipline (leakage-awareness, out-of-sample scoring,
signal decomposition) on a subject with real human weight. Reliable public data
source; committed snapshot for reproducibility.

## Deliberately deferred / next
- Could graduate this to its own standalone repo (strong candidate — vivid visuals,
  meaningful topic) like star-signals and honest-backtester.
- Could add a time dimension (has the curve shifted over decades?) or a causal-ish
  angle (fixed effects, instrument for income).

## Status
Built and shipped 2026-08-31. 11 tests pass; verified end-to-end on a freshly
collected ~200-country World Bank sample. Three charts generated. New dependency:
none beyond the already-installed scikit-learn stack.
