# Week of 2026-07-20 — Capstone: Star Signals

**Theme:** data/ML — an original, personal capstone (not a textbook exercise)
**Type:** the standout project flagged for weeks; first real machine-learning build.

## Origin
Grew out of the user's own candid answer about what motivates them ("looking good
on paper"). Reframed that instinct into a genuine research question using the
economics idea of **signaling** — and pointed it, self-referentially, at GitHub
itself: does a repo's popularity come from real substance or surface polish?

## The question
Predict a repository's star count from features split into two camps —
**signal** (description length, topic tags, homepage, name) vs **substance**
(push recency, age, code size, license, wiki) — then measure how much predictive
power each camp contributes.

## Scope (shipped)
- `collect.py` — sample 2,400 repos across 6 star buckets × 4 languages via the
  GitHub API (avoids survivorship bias); cached to `data/repos.csv`.
- `features.py` — engineer signal/substance features; target = log(stars).
  Deliberately excludes forks & open-issues as popularity *consequences* (leakage).
- `model.py` — train/test split; standardized linear regression (coefficients) +
  random forest (importances), both scored out-of-sample; rolled up by group.
- `plots.py` — group split, feature importances, predicted-vs-actual fit.
- CLI (`--collect` to refresh data) + library API; committed dataset for repro.
- 12 offline tests, incl. a flip test (swap the true driver signal↔substance and
  confirm the verdict flips).

## Finding
Substance dominates: random forest attributes ~82% of predictive power to
substance (linear model a more modest 61% — same direction). Top predictor is
`days_since_push` (active maintenance). Out-of-sample R² ≈ 0.59. README documents
the honest caveats (age-vs-time confound, sampling banding, model disagreement).

## Why this matters for the portfolio story
First ML project; original and personal rather than textbook; and it models the
data/ML + finance-adjacent "signaling" idea. The self-aware framing ("I built this
to look good on paper, and the data told me to do the real work") is genuinely
memorable and essay-ready. New dependency introduced: scikit-learn.

## Deliberately deferred / next
- Could graduate this into its own standalone repo (strong candidate — good pinned
  capstone for the profile).
- Walk-forward validation of the backtester still open.
- Possible follow-up: add richer substance features (README length, CI presence)
  via per-repo fetches.

## Status
Built and shipped 2026-07-20. 12 tests pass; verified end-to-end on a freshly
collected 2,400-repo sample.
