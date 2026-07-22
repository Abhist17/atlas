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


def explain_score(row: pd.Series) -> list[dict]:
    """Human-readable breakdown of WHY the signal is what it is.

    Returns one entry per factor: its direction (+1 bull / -1 bear / 0 neutral),
    a strength in [0,1], and a plain-English reason. Mirrors _dir_score exactly.
    """
    out = []

    # 1. Trend
    up = row["ema9"] > row["ema21"]
    out.append({
        "factor": "Trend (EMA 9/21)", "dir": 1 if up else -1, "strength": 1.0,
        "text": f"EMA9 {'above' if up else 'below'} EMA21 → {'up' if up else 'down'}trend",
    })

    # 2. VWAP
    above = row["close"] > row["vwap"]
    out.append({
        "factor": "VWAP", "dir": 1 if above else -1, "strength": 1.0,
        "text": f"Price {'above' if above else 'below'} VWAP ({row['vwap']:.2f}) — "
                f"{'buyers' if above else 'sellers'} in control",
    })

    # 3. RSI momentum
    rsi = float(row["rsi"])
    rsi_v = float(np.clip((rsi - 50) / 20, -1, 1))
    out.append({
        "factor": "Momentum (RSI)", "dir": 1 if rsi_v > 0 else -1 if rsi_v < 0 else 0,
        "strength": abs(rsi_v),
        "text": f"RSI {rsi:.0f} → {'bullish' if rsi_v>0.1 else 'bearish' if rsi_v<-0.1 else 'neutral'} momentum",
    })

    # 4. Opening-range breakout
    orh, orl = row.get("or_high", np.nan), row.get("or_low", np.nan)
    if not np.isnan(orh):
        if row["close"] > orh:
            out.append({"factor": "Opening-range", "dir": 1, "strength": 1.0,
                        "text": f"Broke above OR high ({orh:.2f}) — bullish breakout"})
        elif row["close"] < orl:
            out.append({"factor": "Opening-range", "dir": -1, "strength": 1.0,
                        "text": f"Broke below OR low ({orl:.2f}) — bearish breakdown"})
        else:
            out.append({"factor": "Opening-range", "dir": 0, "strength": 0.0,
                        "text": f"Inside opening range ({orl:.2f}–{orh:.2f}) — no breakout yet"})

    # 5. Volume confirmation (gate)
    vx = float(row["volume"] / row["avg_volume"]) if row["avg_volume"] else 0.0
    conf = float(np.clip(vx, 0, 2) / 2)
    out.append({
        "factor": "Volume confirmation", "dir": 0, "strength": conf,
        "text": f"Volume {vx:.1f}× average — {'strong' if conf>0.6 else 'weak' if conf<0.35 else 'moderate'} "
                f"conviction gate",
    })
    return out


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
