"""A curated universe of liquid stocks with known sector labels.

These sector labels are the *ground truth* we hold back: the clustering only ever
sees daily returns, never these tags. At the end we check how well the groups it
discovered line up with the sectors it was never shown. Names were chosen to be
large, liquid, and unambiguously in one sector.
"""

from __future__ import annotations

# ticker -> sector. Six sectors with 5-6 names each.
SECTORS: dict[str, str] = {
    # Technology
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "ORCL": "Technology",
    "ADBE": "Technology",
    "CSCO": "Technology",
    # Financials (banks / broker-dealers)
    "JPM": "Financials",
    "BAC": "Financials",
    "WFC": "Financials",
    "GS": "Financials",
    "MS": "Financials",
    "C": "Financials",
    # Energy
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    "SLB": "Energy",
    "EOG": "Energy",
    # Health care
    "JNJ": "Health Care",
    "PFE": "Health Care",
    "MRK": "Health Care",
    "UNH": "Health Care",
    "ABBV": "Health Care",
    # Consumer staples
    "PG": "Staples",
    "KO": "Staples",
    "PEP": "Staples",
    "WMT": "Staples",
    "COST": "Staples",
    # Utilities
    "NEE": "Utilities",
    "DUK": "Utilities",
    "SO": "Utilities",
    "AEP": "Utilities",
    "D": "Utilities",
}


def tickers() -> list[str]:
    """All tickers in the universe, in a stable order."""
    return list(SECTORS.keys())


def sectors_for(symbols) -> list[str]:
    """The sector label for each symbol, in the given order."""
    return [SECTORS[s] for s in symbols]


def n_sectors() -> int:
    """How many distinct sectors the universe spans."""
    return len(set(SECTORS.values()))
