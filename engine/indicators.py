"""Technical indicators for screening & signals.

Operates on an OHLCV DataFrame (columns: timestamp, open, high, low, close,
volume) as returned by data.dhan_client. All functions are pure — they add
columns and return a new frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add the indicator columns the screener/strategy rely on."""
    if df.empty:
        return df
    out = df.copy()

    out["rsi"] = ta.rsi(out["close"], length=14)
    out["atr"] = ta.atr(out["high"], out["low"], out["close"], length=14)
    out["atr_pct"] = out["atr"] / out["close"] * 100

    # EMAs for trend (9/15 crossover is the quant entry core; 21 kept for legacy)
    out["ema9"] = ta.ema(out["close"], length=9)
    out["ema15"] = ta.ema(out["close"], length=15)
    out["ema21"] = ta.ema(out["close"], length=21)

    # ADX = trend-strength / regime filter (high = trending, low = chop)
    try:
        adx = ta.adx(out["high"], out["low"], out["close"], length=14)
        out["adx"] = adx[f"ADX_14"] if adx is not None else np.nan
    except Exception:
        out["adx"] = np.nan

    # VWAP resets each trading day (intraday anchor)
    out["vwap"] = _session_vwap(out)

    # Rolling average volume (in bars)
    out["avg_volume"] = out["volume"].rolling(20, min_periods=1).mean()
    return out


def _session_vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-weighted average price, reset per calendar day."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    day = pd.to_datetime(df["timestamp"]).dt.date
    cum_pv = (tp * df["volume"]).groupby(day).cumsum()
    cum_vol = df["volume"].groupby(day).cumsum().replace(0, np.nan)
    return cum_pv / cum_vol
