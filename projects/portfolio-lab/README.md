# Portfolio Lab

A Python toolkit for analyzing and optimizing a **multi-asset** stock portfolio.
Give it a basket of tickers and it reports portfolio risk/return, finds the
minimum-variance and maximum-Sharpe portfolios via mean-variance optimization,
and saves charts you can drop straight into a README.

This is the multi-asset follow-up to [`stock-toolkit`](../stock-toolkit), which
analyzes a single ticker.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m portfolio_lab AAPL MSFT GOOG --start 2022-01-01 --end 2024-01-01 --risk-free 0.04
```

Output:

```
Portfolio: AAPL, MSFT, GOOG   (2022-01-03 → 2023-12-29)
  Weights:                 AAPL +33.3%, MSFT +33.3%, GOOG +33.3%
  Annualized return:        7.92%
  Annualized volatility:   28.56%
  Sharpe (rf=4%):             0.14
  Max drawdown:           -33.49%

Optimal portfolios (long/short allowed):
  Min variance:  ret  8.83%  vol 27.77%  sharpe  0.17
     weights →   AAPL +57.8%, MSFT +36.0%, GOOG +6.2%
  Max Sharpe:    ret 22.00%  vol 53.61%  sharpe  0.34
     weights →   AAPL -2.4%, MSFT +255.5%, GOOG -153.2%
```

Pass your own allocation with `--weights` (auto-normalized to sum to 1):

```bash
python -m portfolio_lab AAPL MSFT GOOG --weights 0.5 0.3 0.2
```

## Charts

| Growth of $1 | Efficient frontier | Correlation |
|---|---|---|
| ![growth](assets/growth.png) | ![frontier](assets/frontier.png) | ![correlation](assets/correlation.png) |

The frontier plot scatters thousands of random long-only portfolios coloured by
Sharpe ratio, then marks the two closed-form optima.

## The maths

Given annualized expected returns **μ** and the annualized covariance matrix **Σ**,
a portfolio with weights **w** has return `wᵀμ` and volatility `√(wᵀΣw)`. Two
classic portfolios have closed-form solutions:

- **Minimum variance:**  `w ∝ Σ⁻¹ 1`
- **Maximum Sharpe (tangency):**  `w ∝ Σ⁻¹ (μ − r_f)`

both then normalized so the weights sum to 1. These are solved directly with
`numpy.linalg.solve` — no optimizer required.

> **Note on the leverage you'll see:** these closed forms are *unconstrained*, so
> they allow short positions and leverage (weights below 0% or above 100%). That's
> why the max-Sharpe portfolio above shorts GOOG to pile into MSFT — and why its
> star sits outside the long-only random cloud on the frontier plot. It's a
> feature worth seeing: unconstrained optimizers chase return aggressively. A
> long-only version requires constrained optimization (e.g. quadratic programming),
> a natural next extension.

## What's inside

- `data.py` — fetch and date-align adjusted closes for many tickers (`yfinance`)
- `portfolio.py` — returns, annualization, correlation, portfolio performance,
  min-variance & max-Sharpe weights, Monte-Carlo random portfolios, max drawdown
- `plots.py` — growth, correlation heatmap, efficient-frontier charts
- `cli.py` — the command-line entry point

## Use it as a library

```python
import numpy as np
from portfolio_lab import load_prices, daily_returns, annualized_mean, annualized_cov, max_sharpe_weights

prices = load_prices(["AAPL", "MSFT", "GOOG"], start="2022-01-01")
r = daily_returns(prices)
w = max_sharpe_weights(annualized_mean(r), annualized_cov(r), risk_free=0.04)
print(dict(zip(prices.columns, np.round(w, 3))))
```

## Tests

```bash
python -m pytest
```

Every test runs offline against synthetic series with hand-checked answers
(e.g. the min-variance weights of two uncorrelated assets with variances 0.04
and 0.01 must be 20% / 80%).
