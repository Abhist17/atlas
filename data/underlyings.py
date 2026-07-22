"""F&O underlying universe — the instruments we generate directional signals on
(you then trade their options). Indices + liquid F&O single stocks that have
active option chains and clean intraday moves.

yfinance tickers: indices use ^ symbols, stocks use .NS suffix.
"""
from __future__ import annotations

# Index underlyings (most liquid options in India)
INDICES = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
}

# Liquid F&O single stocks (high option volume, clean trends)
FNO_STOCKS = [
    "RELIANCE", "ONGC", "SBIN", "TATASTEEL", "TATAMOTORS", "ICICIBANK",
    "HDFCBANK", "AXISBANK", "INFY", "TCS", "ITC", "LT", "HINDALCO",
    "BHARTIARTL", "MARUTI", "BAJFINANCE", "ADANIENT", "COALINDIA",
    "POWERGRID", "NTPC", "WIPRO", "SUNPHARMA", "TITAN", "ULTRACEMCO",
]


def yf_ticker(name: str) -> str:
    """Map an underlying name to its yfinance ticker."""
    if name in INDICES:
        return INDICES[name]
    return name if name.endswith(".NS") else f"{name.upper()}.NS"


def universe() -> list[str]:
    """All underlying names (indices first, then stocks)."""
    return list(INDICES.keys()) + FNO_STOCKS
