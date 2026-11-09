# Week of 2026-11-09 — Recommender (collaborative filtering)

**Theme:** data/ML — the portfolio's first recommender-systems project.
**Type:** new technique (collaborative filtering / matrix factorization); user asked
me to pick "the next one."

## The question
Does matrix factorization actually beat recommending popular movies? Build the
textbook recsys on MovieLens and test it honestly on rating error AND ranking.

## Scope (shipped)
- `data.py` — load MovieLens latest-small (~100k ratings; cached `data/ratings.csv.gz`
  + `data/movies.csv`); **per-user** train/test split (hold out 20% of each user's
  ratings — never a global split that could orphan a user).
- `models.py` — `GlobalMean`, `ItemMean` (shrunk toward global for rare items — a
  small-samples callback), `BiasModel` (mu + user bias + item bias, regularized), and
  a from-scratch `MatrixFactorization` (biases + latent factors, SGD, ~5s fit).
- `evaluate.py` — RMSE, RMSE by user-activity bucket (cold start), and **recall@10**
  of held-out liked movies (MF vs popularity, vectorized MF scoring).
- `plots.py` — RMSE ladder, cold-start bars, ranking comparison.
- CLI; committed snapshot. 9 offline synthetic tests (incl. MF beats bias beats
  global on data with known latent structure).

## Findings (MovieLens, per-user holdout)
- **RMSE ladder:** GlobalMean 1.045 -> ItemMean 0.960 -> BiasModel 0.872 -> MF 0.865.
  The plain bias model does most of the work; MF adds just 0.7%.
- **Cold start:** MF RMSE 0.95 (<20 ratings) -> 0.84 (200+). Less history, worse guess.
- **THE TWIST — ranking:** recall@10 MF **0.026** vs popularity **0.105**. The model
  with better RMSE is the WORSE recommender: RMSE-trained MF fills its top-10 with
  obscure overfit items; popularity surfaces broadly-loved films. Predicting ratings
  well != recommending well; popularity is a strong baseline.

## The honest core
- MF isn't "failing" — it's optimized for the wrong objective; ranking-specific models
  (implicit ALS, BPR) close the gap. Evaluate on the objective you actually care about.
- Popularity scores well but is one-size-fits-all (no personalization) — a trap too.
- Small, dated, EXPLICIT-ratings data; real systems use implicit signals at scale.

## Why this matters for the portfolio story
First recommender/collaborative-filtering project (matrix factorization from scratch),
and a textbook-perfect "don't fool yourself" result: the fancy model wins the metric
everyone reports (RMSE) and loses the one that matters (recall) — the baseline-
skepticism thread, again. Real reliable data (MovieLens) + committed snapshot. Pure
numpy/pandas/matplotlib, no new deps.

## Deliberately deferred / next
- A ranking-trained model (BPR/implicit ALS) to actually beat popularity; implicit
  feedback; a genre/diversity analysis.
- GRADUATE one of the ungraduated projects (confirm first).

## Status
Built and shipped 2026-11-09. 9 tests pass; verified end-to-end on MovieLens
(~100k ratings). Three charts.
