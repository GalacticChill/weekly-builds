# Week of 2026-10-19 — Small Samples (empirical Bayes)

**Theme:** data/statistics — the portfolio's first Bayesian project.
**Type:** new technique (empirical-Bayes shrinkage / Beta-Binomial); user delegated
("do the next one"), so a BUILD in weekly-builds (no new public repo on an ambiguous
ask).

## The question
Who's really the best when sample sizes differ wildly? A 5-for-5 (100%) vs a
150-for-500 (30%). Raw rates from tiny samples are noise; empirical Bayes shrinks
them toward the crowd. Does that actually recover the truth better?

## Scope (shipped)
- `simulate.py` — draw true rates ~ Beta(a,b), give players log-uniform trial counts
  (many small, few large), observe Binomial successes; keep the true rates for
  scoring (the honest test a real leaderboard can't give).
- `shrink.py` — fit the Beta prior by **Beta-Binomial MLE** (scipy betabinom +
  Nelder-Mead in log-space); posterior mean (α+s)/(α+β+n) + Beta credible intervals.
- `evaluate.py` — RMSE vs truth overall and by sample-size bucket; top-k overlap and
  Spearman rank correlation vs the true order.
- `plots.py` — shrinkage scatter (raw vs shrunk, sized/colored by trials), RMSE-by-
  size bars, and a raw/shrunk/true + credible-interval example chart.
- CLI; no external data (pure simulation, fully reproducible, known ground truth).
- 10 offline deterministic tests.

## Findings (1000 players, known truth)
- Fitted prior Beta(8.7, 25.4) ~ recovers the true Beta(8, 22) (mean 0.26).
- **RMSE vs truth: raw 0.173 -> shrunk 0.058 (3x better).** By bucket: 76% better for
  1-4 trials, down to 7% for 100+ (helps most where data is thinnest).
- Naive "best" = 3/3 (100%); true rate 37%, shrunk to 32%. Raw-rate top-10 overlap
  with the true top-10 = **0%**; shrunk = 20%. Spearman vs truth: 0.49 -> 0.63.

## The honest core
- Shrinkage trades a little bias for a lot of variance — deliberately under-estimates
  a genuine small-sample superstar; the win is lower average error, not perfect
  ranking.
- Assumes one shared population/prior (hierarchical models are the next step).
- Ranking stays hard even shrunken — honest message is "raw rates definitely don't
  find the best," not "shrinkage does."

## Why this matters for the portfolio story
First Bayesian method (empirical Bayes, Beta-Binomial conjugacy, credible intervals),
and a textbook fit for the portfolio's "don't fool yourself" thread — the whole
project is about not trusting a flashy rate from a tiny sample. Quantifiable
ground-truth win (simulation). Pure numpy/pandas/scipy/matplotlib, no new deps.

## Deliberately deferred / next
- A real dataset example (batting averages, conversion rates); a hierarchical
  (multi-group) prior; a Bayesian A/B "stopping rule" companion.
- GRADUATE one of the ungraduated projects (confirm first).

## Status
Built and shipped 2026-10-19. 10 tests pass; verified end-to-end (1000-player
simulation). Three charts.
