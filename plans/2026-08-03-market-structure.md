# Week of 2026-08-03 — Market Structure (unsupervised)

**Theme:** data/ML — a *second, methodologically distinct* ML project.
**Type:** unsupervised learning, chosen to broaden a finance-heavy portfolio that
had only one ML project (star-signals, supervised regression).

## The question
Give a clustering algorithm nothing but daily price movements — no names, no
sector labels — and ask: does it rediscover the market's sectors on its own?

## Why this project
The portfolio was 3 finance tools + 1 supervised-ML capstone. This adds genuine ML
breadth (unsupervised, correlation-based) with admissions-friendly visuals, stays
finance-themed, and keeps the through-line of intellectual honesty (report the real
agreement, including where and why it falls short of a perfect match).

## Scope (shipped)
- `universe.py` — curated 32-ticker, six-sector universe; sector labels are the
  held-back ground truth the algorithm never sees.
- `cluster.py` — correlation -> true metric `d = sqrt(2(1-corr))` -> SciPy
  hierarchical `linkage` -> `fcluster` cut into k groups; dendrogram leaf order for
  the heatmap reordering.
- `evaluate.py` — purity AND adjusted Rand index (two metrics that fail differently;
  ARI is the honest headline since purity inflates with cluster count), plus
  contingency and per-ticker composition tables.
- `plots.py` — dendrogram with leaf labels colored by true sector (links drawn
  neutral gray so sector color is the only semantic encoding) + correlation heatmap
  reordered by the clustering (the block-structure "aha").
- CLI + library API. Works on the labeled universe (with scoring) or arbitrary
  tickers (clustering only).
- 11 offline tests, incl. recovery of a known synthetic block structure (perfect
  purity + ARI = 1), distance-metric properties, noise-scores-near-chance, and
  hand-checked purity/ARI with label-invariance.

## Finding
On 2019–2024 the discovered groups match real sectors at **ARI ~0.51 / purity
~0.69** — far above chance, honestly short of perfect *for a real reason*:
- Financials, Energy, Technology, Utilities fall out as clean unsupervised groups.
- The algorithm merges banks + oil into one **cyclical** super-group and pools
  utilities with defensive staples and health care into a **defensive** one —
  co-movement tracks *risk factors* that cut across GICS sectors.
- Pharma (PFE/MRK/ABBV) barely clusters — it hangs as near-independent branches,
  because drug stocks move on trial/FDA news, not the market factor. A feature.

## Why this matters for the portfolio story
The "algorithm rediscovers structure it was never taught" result is memorable and
essay-ready, and the honest gap (0.51, not 1.0) becomes the *interesting* part:
return co-movement is about risk factors, not tidy taxonomies. New technique on the
portfolio: unsupervised hierarchical clustering, correlation distance, ARI. No new
dependencies beyond the already-installed scipy + scikit-learn.

## Deliberately deferred / next
- GRADUATE star-signals into its own standalone pinned repo (still the strongest
  single profile move; deferred three weeks running).
- Could add a `k`-selection diagnostic (silhouette / gap statistic) instead of
  fixing k = number of sectors.
- Could show a 2-D MDS/t-SNE embedding colored by sector as a third view.

## Status
Built and shipped 2026-08-03. 11 tests pass; verified end-to-end on the 32-name
universe (ARI 0.51) and on synthetic block/noise fixtures. Two charts generated.
