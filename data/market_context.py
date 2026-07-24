"""Broad-market / sector context for intraday confirmation.

A directional intraday trade should not fight the tape: buying a CALL while
NIFTY is selling off is a low-quality trade. This module gives the current
direction of the broad market (NIFTY) and the banking sector (BANKNIFTY) so the
signal engine can require alignment before it says ENTER.

Bias is +1 (up) / -1 (down) / 0 (flat), from 5-min EMA structure. Cached ~60s.
"""
from __future__ import annotations

import time

import pandas as pd
import yfinance as yf

from utils.logger import get_logger

log = get_logger("data.market_context")

_INDEX = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK"}

# Bank / financial F&O names → judged against BANKNIFTY instead of NIFTY.
_BANKING = {
    "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "INDUSINDBK",
    "BANKBARODA", "PNB", "FEDERALBNK", "IDFCFIRSTB", "AUBANK", "BANDHANBNK",
    "BAJFINANCE", "BAJAJFINSV", "SBILIFE", "HDFCLIFE", "ICICIPRULI",
    "ICICIGI", "CHOLAFIN", "SHRIRAMFIN", "LICHSGFIN", "MUTHOOTFIN",
}

_cache: dict[str, tuple[float, dict]] = {}
_TTL = 60


def _flat(col: pd.Series | pd.DataFrame) -> pd.Series:
    return col.iloc[:, 0] if hasattr(col, "columns") else col


def _index_bias(ticker: str) -> dict:
    hit = _cache.get(ticker)
    if hit and time.time() - hit[0] < _TTL:
        return hit[1]
    out = {"bias": 0, "chg": 0.0}
    try:
        df = yf.download(ticker, period="2d", interval="5m",
                         progress=False, auto_adjust=True)
        if df is not None and len(df) > 25:
            close = _flat(df["Close"]).astype(float).reset_index(drop=True)
            ema9 = close.ewm(span=9, adjust=False).mean()
            ema21 = close.ewm(span=21, adjust=False).mean()
            c, e9, e21 = close.iloc[-1], ema9.iloc[-1], ema21.iloc[-1]
            if c > e21 and e9 >= e21:
                out["bias"] = 1
            elif c < e21 and e9 <= e21:
                out["bias"] = -1
            # intraday change vs the last session's first bar
            out["chg"] = round(float(c / close.iloc[max(0, len(close) - 75)] - 1) * 100, 2)
    except Exception as e:
        log.debug("index bias failed for %s: %s", ticker, e)
    _cache[ticker] = (time.time(), out)
    return out


def index_change(name: str = "NIFTY") -> float:
    """Intraday % change of an index (NIFTY / BANKNIFTY)."""
    return _index_bias(_INDEX.get(name, "^NSEI"))["chg"]


def get_context(symbol: str) -> dict:
    """Return the relevant market/sector context for a stock.

    {name, bias, chg} — BANKNIFTY for banking/financial names, else NIFTY.
    """
    symbol = symbol.upper()
    name = "BANKNIFTY" if symbol in _BANKING else "NIFTY"
    ctx = _index_bias(_INDEX[name])
    return {"name": name, "bias": ctx["bias"], "chg": ctx["chg"]}
