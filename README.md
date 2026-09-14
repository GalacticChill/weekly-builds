# weekly-builds
Weekly data/ML and finance projects in Python. An incubator for new builds.

## Projects
- [Stock Data Toolkit](projects/stock-toolkit) — pull, analyze, and chart historical stock data (returns, volatility, drawdown, moving averages).
- [Portfolio Lab](projects/portfolio-lab) — multi-asset portfolio analysis with correlation and mean-variance optimization: closed-form and long-only constrained min-variance & max-Sharpe portfolios, plus the efficient frontier.
- [Backtester](projects/backtester) — simulate holding a portfolio through time with periodic rebalancing and transaction costs, plus point-in-time signal strategies (momentum, inverse-volatility) with no lookahead; equity-curve and drawdown charts. A three-part honesty toolkit: no-lookahead, walk-forward validation with an "overfitting tax," and block-bootstrap significance testing (is the Sharpe distinguishable from luck?). **Graduated to its own repo → [GalacticChill/honest-backtester](https://github.com/GalacticChill/honest-backtester)** (pip-installable, CI-tested).
- [Star Signals](projects/star-signals) — does GitHub reward substance or surface signal? A machine-learning capstone predicting a repo's stars from features split into signal vs substance, measuring how much predictive power each camp contributes. **Graduated to its own repo → [GalacticChill/star-signals](https://github.com/GalacticChill/star-signals)** (pip-installable, CI-tested).
- [Market Structure](projects/market-structure) — unsupervised learning on daily returns: can hierarchical clustering rediscover the market's sectors from price co-movement alone? Dendrogram, reordered correlation heatmap, and an honest adjusted-Rand score against real sectors.
- [Life & Money](projects/life-and-money) — does wealth buy a longer life? A World Bank cross-country study: the Preston curve (diminishing returns of income), an out-of-sample model where clean water and sanitation nearly rival income, and a residual analysis of which countries beat the odds. **Graduated to its own repo → [GalacticChill/life-and-money](https://github.com/GalacticChill/life-and-money)** (pip-installable, CI-tested).
- [Word Prints](projects/word-prints) — stylometry / NLP: can we identify an author from function words alone (the, of, upon), with all content stripped away? Leave-one-book-out attribution across 20 novels hits 87% on function words alone — beating the full vocabulary. Confusion matrix, per-author signatures, and a PCA style map.

## Plans
- [Stock Data Toolkit](plans/2026-06-16-stock-data-toolkit.md)
- [Portfolio Lab](plans/2026-06-22-portfolio-analyzer.md)
- [Long-only constrained optimization](plans/2026-06-29-constrained-optimization.md)
- [Portfolio backtester](plans/2026-07-06-backtester.md)
- [Signal-driven strategies](plans/2026-07-13-signal-strategies.md)
- [Capstone: Star Signals](plans/2026-07-20-star-signals-capstone.md)
- [Walk-forward validation](plans/2026-07-27-walk-forward-validation.md)
- [Market Structure (unsupervised)](plans/2026-08-03-market-structure.md)
- [Bootstrap significance](plans/2026-08-17-bootstrap-significance.md)
- [Capstone: Life & Money](plans/2026-08-31-life-and-money.md)
- [Capstone: Word Prints (stylometry)](plans/2026-09-14-word-prints.md)
