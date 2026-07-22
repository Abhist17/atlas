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


_MIN_CONF = 0.30      # below this: no clean edge, stay out
_AT_SUPPORT = 0.45    # within this many ATRs of VWAP = "at value"
_EXTENDED = 1.20      # beyond this many ATRs from VWAP = chasing


def _entry_decision(long, conf, ext, ltp, vwap, ema9, atr, orh, orl,
                    swing_hi, swing_lo, vol_x):
    """Decide WHEN to enter. Returns (status, headline, trigger, entry_price).

    status ∈ {ENTER, WAIT, AVOID}. entry_price is where you'd actually get in
    (now, a pullback level, or a breakout trigger) — TP/SL are measured from it.
    """
    d = "up" if long else "down"
    breakout_lvl = orh if not np_isnan(orh) else swing_hi
    breakdown_lvl = orl if not np_isnan(orl) else swing_lo
    strong_vol = vol_x >= 1.3
    value_lo, value_hi = (min(vwap, ema9), max(vwap, ema9))

    # 1. No clean edge — the factors don't line up.
    if conf < _MIN_CONF:
        return ("AVOID", "No trade — signals are mixed, no clean edge.",
                "Wait for factors to align (trend + VWAP + momentum same way).", ltp)

    # 2. Fresh breakout in the trade direction, with volume → take it now.
    if long and ltp >= breakout_lvl and ext < _EXTENDED and strong_vol:
        return ("ENTER", f"Breakout above {breakout_lvl:.2f} on volume — enter now.",
                "Momentum breakout confirmed.", ltp)
    if not long and ltp <= breakdown_lvl and ext < _EXTENDED and strong_vol:
        return ("ENTER", f"Breakdown below {breakdown_lvl:.2f} on volume — enter now.",
                "Momentum breakdown confirmed.", ltp)

    # 3. Sitting at value (VWAP/EMA) with the trend → good entry now.
    if abs(ext) <= _AT_SUPPORT:
        ref = "support" if long else "resistance"
        return ("ENTER", f"At VWAP {ref} in a clean {d}trend — enter now.",
                f"Price at value ({value_lo:.2f}–{value_hi:.2f}); low-risk entry.", ltp)

    # 4. Too extended from VWAP → don't chase, wait for a pullback.
    if ext >= _EXTENDED:
        pull = value_hi if long else value_lo
        return ("WAIT", f"Extended {ext:.1f} ATR from VWAP — don't chase.",
                f"Wait for a pullback toward {pull:.2f} (VWAP/EMA), then enter.", pull)

    # 5. In-between: bias is right but not at a trigger yet.
    if long:
        return ("WAIT", "Bias is up but no trigger yet.",
                f"Enter on a break above {breakout_lvl:.2f}, or a dip to {value_hi:.2f}.",
                breakout_lvl)
    return ("WAIT", "Bias is down but no trigger yet.",
            f"Enter on a break below {breakdown_lvl:.2f}, or a pop to {value_lo:.2f}.",
            breakdown_lvl)


def np_isnan(x):
    return isinstance(x, float) and np.isnan(x)


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
    vwap = float(last["vwap"])
    ema9 = float(last["ema9"])
    score = float(_dir_score(last))
    bias = "LONG" if score >= 0 else "SHORT"
    long = bias == "LONG"
    opt = "CALL" if long else "PUT"
    conf = min(abs(score), 1.0)

    today = ind[ind["date"] == last["date"]] if ind["date"].iloc[0] is not None else ind
    prev = ind[ind["date"] < last["date"]] if ind["date"].iloc[0] is not None else ind.iloc[:0]

    # recent swing (last 12 bars) for breakout triggers
    recent = ind.iloc[-12:]
    swing_hi = float(recent["high"].max())
    swing_lo = float(recent["low"].min())
    orh = float(last.get("or_high", np.nan))
    orl = float(last.get("or_low", np.nan))
    vol_x = float(last["volume"] / last["avg_volume"]) if last["avg_volume"] else 0.0

    # how extended is price from VWAP, in ATR units (signed toward the trade)
    ext = ((ltp - vwap) / atr) if long else ((vwap - ltp) / atr)

    # ---- entry-timing decision: ENTER / WAIT / AVOID ----
    status, headline, trigger, entry_price = _entry_decision(
        long, conf, ext, ltp, vwap, ema9, atr, orh, orl, swing_hi, swing_lo, vol_x)

    # SL/TP measured from the PLANNED entry price (not just current LTP)
    risk = _ATR_STOP * atr
    if long:
        stop = entry_price - risk
        tps = [entry_price + risk, entry_price + 2 * risk, entry_price + 3 * risk]
    else:
        stop = entry_price + risk
        tps = [entry_price - risk, entry_price - 2 * risk, entry_price - 3 * risk]

    def _lvl(v):
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 2)

    levels = {
        "VWAP": _lvl(vwap),
        "Day high": _lvl(today["high"].max()),
        "Day low": _lvl(today["low"].min()),
        "OR high": _lvl(orh),
        "OR low": _lvl(orl),
        "Prev close": _lvl(prev["close"].iloc[-1]) if not prev.empty else None,
    }

    # option mapping (ATM ~0.5 delta): option move ≈ delta * underlying move
    opt_move_sl = round(abs(entry_price - stop) * _ATM_DELTA, 2)
    opt_move_tp = [round(abs(t - entry_price) * _ATM_DELTA, 2) for t in tps]

    return {
        "symbol": symbol, "ok": True, "ltp": round(ltp, 2),
        "bias": bias, "option": opt, "confidence": round(conf, 2),
        "atr": round(atr, 2), "atr_pct": round(float(last["atr_pct"]), 2),
        "status": status, "headline": headline, "trigger": trigger,
        "extension": round(ext, 2),
        "entry": round(entry_price, 2), "stop": round(stop, 2),
        "risk_pts": round(abs(entry_price - stop), 2),
        "risk_pct": round(abs(entry_price - stop) / ltp * 100, 2),
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
