# Stock Data Toolkit

> A clean, reusable Python toolkit for pulling, analyzing, and visualizing historical stock data — your first weekly build.

## Why it's interesting
Almost every finance/ML project starts by getting price data into a tidy shape and computing basic risk/return metrics. Building this well once gives you a reusable foundation for every future project in `weekly-builds`, and it produces nice charts that look great in a README. It's small enough to finish in a focused week but real enough to be genuinely useful.

## Difficulty
**Beginner–Intermediate** · roughly 4–8 focused hours

## Tech stack & key libraries
- **Python 3.11+**
- `yfinance` — free historical price data
- `pandas` — data wrangling
- `matplotlib` — charts
- `pytest` — a couple of sanity tests
- (optional) `typer` or `argparse` — a small CLI

## Data source
**Yahoo Finance via `yfinance`** — free, no API key. Example:
```python
import yfinance as yf
df = yf.download("AAPL", start="2020-01-01", end="2024-12-31")
