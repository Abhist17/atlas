"""Trade-plan levels for a chosen stock.

Given the live intraday frame we derive a directional bias and an ATR-based
plan: entry, stop-loss and layered take-profit targets (1R/2R/3R), plus the
key intraday levels (VWAP, day range, opening range, prev close). We also map
the plan onto the ATM option using a ~0.5 delta approximation so the user sees
roughly where to book/stop on the CE/PE they'd actually buy.

These are decision-support levels, NOT predictions — Atlas has no proven edge
on direction; the plan just makes risk/reward explicit around the user's call.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.yf_client import yfc
from engine.directional import _dir_score, add_opening_range, explain_score
from engine.indicators import add_indicators
from utils.logger import get_logger

log = get_logger("engine.levels")

_ATR_STOP = 1.5          # stop distance = 1.5 * ATR
_ATM_DELTA = 0.5         # rough delta of an ATM option


def compute_levels(symbol: str, interval: int = 5) -> dict:
    symbol = symbol.upper()
    df = yfc.batch_intraday([f"{symbol}.NS"], days=2, interval=interval).get(f"{symbol}.NS")
    if df is None or len(df) < 25:
        return {"symbol": symbol, "ok": False, "error": "Not enough data."}

    ind = add_opening_range(add_indicators(df)).dropna(subset=["atr", "vwap", "ema21"])
    if ind.empty:
        return {"symbol": symbol, "ok": False, "error": "Indicators unavailable."}
    ind = ind.copy()
    if "timestamp" in ind:
        ind["date"] = pd.to_datetime(ind["timestamp"]).dt.date
    else:
        ind["date"] = None
    last = ind.iloc[-1]

    ltp = float(last["close"])
    atr = float(last["atr"])
    score = float(_dir_score(last))
    bias = "LONG" if score >= 0 else "SHORT"
    opt = "CALL" if score >= 0 else "PUT"
    conf = min(abs(score), 1.0)

    risk = _ATR_STOP * atr
    if bias == "LONG":
        entry, stop = ltp, ltp - risk
        tps = [ltp + risk, ltp + 2 * risk, ltp + 3 * risk]
    else:
        entry, stop = ltp, ltp + risk
        tps = [ltp - risk, ltp - 2 * risk, ltp - 3 * risk]

    today = ind[ind["date"] == last["date"]] if ind["date"].iloc[0] is not None else ind
    prev = ind[ind["date"] < last["date"]] if ind["date"].iloc[0] is not None else ind.iloc[:0]

    def _lvl(v):
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 2)

    levels = {
        "VWAP": _lvl(last.get("vwap")),
        "Day high": _lvl(today["high"].max()),
        "Day low": _lvl(today["low"].min()),
        "OR high": _lvl(last.get("or_high")),
        "OR low": _lvl(last.get("or_low")),
        "Prev close": _lvl(prev["close"].iloc[-1]) if not prev.empty else None,
    }

    # option mapping (ATM ~0.5 delta): option move ≈ delta * underlying move
    opt_move_sl = round(abs(entry - stop) * _ATM_DELTA, 2)
    opt_move_tp = [round(abs(t - entry) * _ATM_DELTA, 2) for t in tps]

    return {
        "symbol": symbol, "ok": True, "ltp": round(ltp, 2),
        "bias": bias, "option": opt, "confidence": round(conf, 2),
        "atr": round(atr, 2), "atr_pct": round(float(last["atr_pct"]), 2),
        "entry": round(entry, 2), "stop": round(stop, 2),
        "risk_pts": round(abs(entry - stop), 2),
        "risk_pct": round(abs(entry - stop) / ltp * 100, 2),
        "targets": [
            {"px": round(tps[i], 2), "rr": i + 1,
             "pct": round((tps[i] / ltp - 1) * 100, 2),
             "opt_gain": opt_move_tp[i]}
            for i in range(3)
        ],
        "opt_stop_move": opt_move_sl,
        "levels": {k: v for k, v in levels.items() if v is not None},
        "factors": explain_score(last),
        "note": "ATR-based plan · option moves use ~0.5 ATM delta · not a prediction",
    }
