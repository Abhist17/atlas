"""Directional conviction engine — the core of the options system.

Produces a SIGNED conviction score per bar in [-1, +1]:
    +1  strong bullish  -> buy CALL
    -1  strong bearish  -> buy PUT
     0  no clear edge    -> stay flat (most of the time)

The philosophy is selectivity: only when several independent factors align in
the SAME direction does conviction get large. A high threshold means few trades
— only A+ setups — which is what keeps option-buying (theta-negative) viable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _dir_score(row: pd.Series) -> float:
    """Combine confirmations into a signed [-1,1] conviction for one bar."""
    votes = []

    # 1. Trend: EMA9 vs EMA21
    votes.append(1.0 if row["ema9"] > row["ema21"] else -1.0)

    # 2. VWAP side (institutional reference)
    votes.append(1.0 if row["close"] > row["vwap"] else -1.0)

    # 3. RSI momentum around 50, scaled
    votes.append(float(np.clip((row["rsi"] - 50) / 20, -1, 1)))

    # 4. Price vs opening-range (breakout direction)
    if not np.isnan(row.get("or_high", np.nan)):
        if row["close"] > row["or_high"]:
            votes.append(1.0)
        elif row["close"] < row["or_low"]:
            votes.append(-1.0)
        else:
            votes.append(0.0)

    # 5. Volume confirmation gates conviction (no volume -> weak)
    vol_conf = float(np.clip(row["volume"] / row["avg_volume"], 0, 2) / 2)

    raw = np.mean(votes) if votes else 0.0
    return float(raw * vol_conf)


def add_opening_range(df: pd.DataFrame, opening_bars: int = 6) -> pd.DataFrame:
    """Add per-session opening-range high/low (first `opening_bars` bars)."""
    out = df.copy()
    day = pd.to_datetime(out["timestamp"]).dt.date
    bar_no = out.groupby(day).cumcount()
    mask = bar_no < opening_bars
    out["or_high"] = out["high"].where(mask).groupby(day).transform("max")
    out["or_low"] = out["low"].where(mask).groupby(day).transform("min")
    return out


def directional_score(df: pd.DataFrame) -> pd.Series:
    """Signed conviction series for an indicator-augmented frame."""
    d = df if "or_high" in df.columns else add_opening_range(df)
    return d.apply(_dir_score, axis=1)
