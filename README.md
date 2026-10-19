# weekly-builds

One real, finished Python project a week — across **finance, machine learning,
statistics, and simulation** — built in the open. No tutorials recopied, no hollow
filler: each one asks a genuine question, answers it with data, and is honest about
what the answer is worth.

If there's a single thread running through all of them, it's a stubborn refusal to
let a result look better than it really is — out-of-sample everywhere, baselines
before models, and leakage hunted down and removed. That theme has its own section
[below](#the-through-line-not-fooling-myself).

This repo is the incubator where projects start. The strongest ones graduate into
their own [standalone repositories](#graduated-standalone-repos).

---

## At a glance

| Project | Field | Core technique |
|---|---|---|
| [Stock Toolkit](projects/stock-toolkit) | Finance | returns, volatility, drawdown |
| [Portfolio Lab](projects/portfolio-lab) | Finance | mean-variance optimization |
| [Backtester](projects/backtester) → [repo](https://github.com/GalacticChill/honest-backtester) | Finance | backtesting + walk-forward + bootstrap |
| [Star Signals](projects/star-signals) → [repo](https://github.com/GalacticChill/star-signals) | Machine learning | supervised regression, feature importance |
| [Market Structure](projects/market-structure) | Machine learning | unsupervised hierarchical clustering |
| [Word Prints](projects/word-prints) | Machine learning | NLP / stylometry, text classification |
| [Life & Money](projects/life-and-money) → [repo](https://github.com/GalacticChill/life-and-money) | Data & society | regression + residual analysis |
| [Slang Half-Life](https://github.com/GalacticChill/slang-half-life) | Data & society | survival analysis (Kaplan-Meier) |
| [Benford's Law](projects/benford) | Statistics | distributional goodness-of-fit |
| [Forecastable?](projects/forecastable) | Forecasting | time-series, honest validation |
| [Schelling](projects/schelling) | Simulation | agent-based modeling |
| [Small Samples](projects/small-samples) | Statistics | empirical-Bayes shrinkage |

---

## The projects

### Finance & markets

- **[Stock Toolkit](projects/stock-toolkit)** — where it started: pull a stock's
  price history and measure its return, volatility, and worst drawdown.
- **[Portfolio Lab](projects/portfolio-lab)** — from one stock to a basket:
  correlation, the mean-variance efficient frontier, and the realistic long-only
  (no-shorting) min-variance and max-Sharpe portfolios.
- **[Backtester](projects/backtester)** — what *actually* happens if you hold and
  rebalance a portfolio through history, after drift and trading costs. Grew over
  four weeks into a three-part honesty toolkit — no-lookahead strategies,
  walk-forward validation with an "overfitting tax," and block-bootstrap
  significance tests — and graduated to
  **[honest-backtester](https://github.com/GalacticChill/honest-backtester)**.

### Machine learning

- **[Star Signals](projects/star-signals)** — does GitHub reward real *substance* or
  surface *signal*? A model predicts a repo's stars from features split into those
  two camps; substance wins, and the biggest predictor is simply whether a project
  is still being worked on. Graduated to
  **[star-signals](https://github.com/GalacticChill/star-signals)**.
- **[Market Structure](projects/market-structure)** — hand a clustering algorithm
  nothing but daily returns and watch it *rediscover the market's sectors*, then
  score the match honestly (it finds risk factors that cut across the tidy sector
  labels).
- **[Word Prints](projects/word-prints)** — stylometry: identify an author from the
  tiny function words they don't think about (*the, of, upon*), content stripped
  away. 87% accuracy on unseen books — beating the full vocabulary.

### Data & society

- **[Life & Money](projects/life-and-money)** — does wealth buy a longer life? The
  Preston curve across ~200 countries, where clean water and sanitation turn out to
  rival income, and a residual analysis of who beats the odds. Graduated to
  **[life-and-money](https://github.com/GalacticChill/life-and-money)**.
- **[Slang Half-Life](https://github.com/GalacticChill/slang-half-life)** — how long
  does a slang term live? Survival analysis of a decade of Wiktionary lookups, with
  an installable dashboard that refreshes weekly. Built as its own repo from day one.

### Statistics, forecasting & simulation

- **[Benford's Law](projects/benford)** — real data starts with the digit 1 about
  30% of the time, not 11%. A forensic-statistics study with chi-square and MAD
  conformity tests — and an honest look at why it's a *screen*, not a fraud verdict.
- **[Forecastable?](projects/forecastable)** — what you can and can't predict: NOAA's
  CO2 forecasts five years out to ~0.3% error, while a random walk beats every model
  with the humble "repeat the last value."
- **[Schelling](projects/schelling)** — how little personal preference it takes to
  segregate a city. An agent-based model where wanting just ~30% of your neighbours
  to match tips a mixed city into a divided one — a mechanism, not a map.
- **[Small Samples](projects/small-samples)** — why a 100% success rate usually means
  nothing. Empirical-Bayes shrinkage pulls noisy small-sample rates toward the crowd;
  against known true rates it cuts error threefold, and a 3-for-3 "best player" turns
  out to be average.

---

## The through-line: not fooling myself

The most interesting part of each project is the honesty check — the specific way it
refuses to overstate its result:

- **Leakage, hunted and removed.** Star Signals drops forks and open-issues (they're
  *consequences* of popularity); Life & Money drops infant mortality (it's an
  arithmetic *component* of life expectancy). Both would have made the models look
  great by predicting the answer from itself.
- **Out-of-sample, always.** Every model is scored on data it never trained on —
  held-out repos, held-out countries, and for Word Prints, held-out whole *books*,
  so an author's style has to carry into a work the model never saw.
- **Beat a baseline before believing a model.** Forecastable? shows a fancy model
  losing to "repeat the last value" on a random walk — averaged over 300 seeds,
  because one lucky seed can flatter anything.
- **Searching inflates significance.** The backtester's walk-forward "overfitting
  tax" and its Šidák correction both measure how much of a good-looking result is
  just the luckiest of many tries.
- **The right test, read the right way.** Benford's Law leans on MAD rather than
  chi-square (which rejects everything at large *n*) and insists conformity isn't
  honesty and non-conformity isn't fraud.
- **A model is a mechanism, not a map.** Schelling proves mild preference is
  *sufficient* to segregate — not that it's the only cause — and even flags the
  regime where the model has no equilibrium.
- **Don't trust a rate from a tiny sample.** Small Samples shows a "perfect" 3-for-3
  record is noise, and leans on shrinkage and credible intervals to say how much of
  an estimate is really earned.

## Graduated standalone repos

The projects polished into their own pip-installable, CI-tested repositories:

- **[honest-backtester](https://github.com/GalacticChill/honest-backtester)** — a
  backtester built around the ways a backtest can fool you.
- **[star-signals](https://github.com/GalacticChill/star-signals)** — does GitHub
  reward substance or surface signal?
- **[life-and-money](https://github.com/GalacticChill/life-and-money)** — does wealth
  buy a longer life?
- **[slang-half-life](https://github.com/GalacticChill/slang-half-life)** — how long
  does slang live? (with a live, offline-capable dashboard).

## How I work

- **Real over flashy.** Something modest that runs and is tested beats something
  impressive that only works in a screenshot.
- **Every project ships three things:** a clear README with the finding up front,
  charts generated from the code, and an **offline test suite** (synthetic or
  hand-checked, no network) so the result is reproducible and the claims are pinned
  down.
- **Reproducible data.** Projects that pull from an API (World Bank, NOAA, Project
  Gutenberg) commit a data snapshot so the analysis runs the same way next year.
- **Tools:** Python, NumPy, pandas, SciPy, scikit-learn, and matplotlib — with
  pytest keeping it honest.

---

## Build log

Each week's project starts with a short plan in [`plans/`](plans):

| Week | Plan |
|---|---|
| 2026-06-16 | [Stock Data Toolkit](plans/2026-06-16-stock-data-toolkit.md) |
| 2026-06-22 | [Portfolio Lab](plans/2026-06-22-portfolio-analyzer.md) |
| 2026-06-29 | [Long-only constrained optimization](plans/2026-06-29-constrained-optimization.md) |
| 2026-07-06 | [Portfolio backtester](plans/2026-07-06-backtester.md) |
| 2026-07-13 | [Signal-driven strategies](plans/2026-07-13-signal-strategies.md) |
| 2026-07-20 | [Capstone: Star Signals](plans/2026-07-20-star-signals-capstone.md) |
| 2026-07-27 | [Walk-forward validation](plans/2026-07-27-walk-forward-validation.md) |
| 2026-08-03 | [Market Structure (unsupervised)](plans/2026-08-03-market-structure.md) |
| 2026-08-17 | [Bootstrap significance](plans/2026-08-17-bootstrap-significance.md) |
| 2026-08-31 | [Capstone: Life & Money](plans/2026-08-31-life-and-money.md) |
| 2026-09-14 | [Capstone: Word Prints (stylometry)](plans/2026-09-14-word-prints.md) |
| 2026-09-21 | [Benford's Law](plans/2026-09-21-benford.md) |
| 2026-09-28 | [Forecastable? (time series)](plans/2026-09-28-forecastable.md) |
| 2026-10-05 | [Schelling segregation (agent-based)](plans/2026-10-05-schelling.md) |
| 2026-10-19 | [Small Samples (empirical Bayes)](plans/2026-10-19-small-samples.md) |
