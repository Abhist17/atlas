"""Quant entry engine — EMA 9/15 crossover with a regime filter.

A mechanical, rule-based signal (no black boxes):

  1. Core trigger: EMA(9) crossing EMA(15). Cross up = long bias, cross down = short.
  2. Regime filter: ADX(14). Crossovers only work in a trending tape, so we stand
     aside (AVOID) when ADX is low (chop) — this is where naive crossover systems
     bleed out.
  3. Location filter: VWAP side must agree with the bias, and we don't chase price
     that is stretched far from EMA15 (WAIT for a pullback instead).
  4. Risk: ATR-based stop and 1R/2R/3R targets, measured from the planned entry.

Output matches the dashboard's signal schema (status / headline / trigger / entry /
stop / targets / factors / levels), so the UI is unchanged.

Honest note: a clean rule set does not manufacture edge — short-term direction is
still noisy. The value is discipline: it keeps you out of chop and off chases.
"""
from __future__ import annotations

import numpy as np

from data.live_feed import get_bars
from engine.directional import add_opening_range
from engine.indicators import add_indicators
from utils.logger import get_logger

log = get_logger("engine.quant_signal")

ADX_MIN = 20.0        # below this = choppy, crossover unreliable
FRESH_BARS = 3        # a cross this recent counts as a fresh trigger
EXTENDED = 1.5        # ATRs from EMA15 beyond which we don't chase
AT_EMA = 0.5          # within this many ATRs of EMA15 = "at the moving average"
_ATR_STOP = 1.3
_ATM_DELTA = 0.5


def compute_signal(symbol: str, interval: int = 5) -> dict:
    symbol = symbol.upper()
    feed = get_bars(symbol, days=5, interval=interval)
    if not feed.get("ok"):
        return {"symbol": symbol, "ok": False, "error": feed.get("error", "No data.")}

    ind = add_opening_range(add_indicators(feed["bars"])).dropna(
        subset=["ema9", "ema15", "atr", "vwap"]).reset_index(drop=True)
    if len(ind) < 20:
        return {"symbol": symbol, "ok": False, "error": "Not enough clean bars."}

    last = ind.iloc[-1]
    ltp = float(feed["ltp"])
    atr = float(last["atr"])
    ema9 = float(last["ema9"])
    ema15 = float(last["ema15"])
    vwap = float(last["vwap"])
    rsi = float(last["rsi"]) if not np.isnan(last["rsi"]) else 50.0
    adx = float(last["adx"]) if not np.isnan(last.get("adx", np.nan)) else 0.0
    vol_x = float(last["volume"] / last["avg_volume"]) if last["avg_volume"] else 0.0

    # --- EMA 9/15 crossover state ---
    diff = (ind["ema9"] - ind["ema15"]).to_numpy()
    sign = np.sign(diff)
    long = sign[-1] >= 0
    bias = "LONG" if long else "SHORT"
    opt = "CALL" if long else "PUT"
    # bars since the most recent sign change
    bars_since, cross_dir = _bars_since_cross(sign)

    ext = ((ltp - ema15) / atr) if long else ((ema15 - ltp) / atr)
    vwap_ok = (ltp > vwap) if long else (ltp < vwap)
    trending = adx >= ADX_MIN
    vol_ok = vol_x >= 1.0
    fresh = bars_since is not None and bars_since <= FRESH_BARS

    # --- confidence: regime strength + confirmations ---
    adx_norm = float(np.clip((adx - 15) / 25, 0, 1))     # 15..40 -> 0..1
    conf = float(np.clip(
        0.45 * adx_norm + 0.25 * (1 if vwap_ok else 0)
        + 0.15 * (1 if fresh else 0) + 0.15 * min(vol_x / 2, 1), 0, 1))

    # --- entry decision ---
    status, headline, trigger, entry = _decide(
        long, trending, adx, fresh, bars_since, cross_dir, vwap_ok,
        ext, ltp, ema15, vwap, vol_ok)

    risk = _ATR_STOP * atr
    if long:
        stop = entry - risk
        tps = [entry + risk, entry + 2 * risk, entry + 3 * risk]
    else:
        stop = entry + risk
        tps = [entry - risk, entry - 2 * risk, entry - 3 * risk]

    today = ind[ind["timestamp"].astype(str).str[:10] == str(last["timestamp"])[:10]]
    if today.empty:
        today = ind

    def _r(v):
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 2)

    levels = {
        "EMA 9": _r(ema9), "EMA 15": _r(ema15), "VWAP": _r(vwap),
        "ADX": _r(adx), "Day high": _r(today["high"].max()), "Day low": _r(today["low"].min()),
    }
    opt_move_sl = round(abs(entry - stop) * _ATM_DELTA, 2)
    opt_move_tp = [round(abs(t - entry) * _ATM_DELTA, 2) for t in tps]

    return {
        "symbol": symbol, "ok": True, "ltp": round(ltp, 2),
        "bias": bias, "option": opt, "confidence": round(conf, 2),
        "atr": round(atr, 2), "atr_pct": round(atr / ltp * 100, 2),
        "status": status, "headline": headline, "trigger": trigger,
        "extension": round(ext, 2),
        "entry": round(entry, 2), "stop": round(stop, 2),
        "risk_pts": round(abs(entry - stop), 2),
        "risk_pct": round(abs(entry - stop) / ltp * 100, 2),
        "targets": [
            {"px": round(tps[i], 2), "rr": i + 1,
             "pct": round((tps[i] / ltp - 1) * 100, 2), "opt_gain": opt_move_tp[i]}
            for i in range(3)
        ],
        "opt_stop_move": opt_move_sl,
        "levels": {k: v for k, v in levels.items() if v is not None},
        "factors": _factors(long, cross_dir, bars_since, adx, trending, vwap_ok,
                             vwap, rsi, vol_x, ema9, ema15),
        "live": {"is_live": feed["is_live"], "source": feed["source"]},
        "note": "Quant EMA 9/15 crossover · ADX regime filter · ATR risk · not a prediction",
    }


def _bars_since_cross(sign: np.ndarray):
    """Bars since the last EMA sign change, and its direction (up/down)."""
    for i in range(len(sign) - 1, 0, -1):
        if sign[i] != sign[i - 1] and sign[i] != 0:
            return (len(sign) - 1 - i), ("up" if sign[i] > 0 else "down")
    return None, ("up" if sign[-1] >= 0 else "down")


def _decide(long, trending, adx, fresh, bars_since, cross_dir, vwap_ok,
            ext, ltp, ema15, vwap, vol_ok):
    d = "up" if long else "down"

    if not trending:
        return ("AVOID", f"Choppy tape (ADX {adx:.0f}) — EMA crossover unreliable.",
                "Wait for ADX to rise above 20 (a real trend).", ltp)

    if not vwap_ok:
        return ("WAIT", f"EMA bias is {d} but price is on the wrong side of VWAP.",
                f"Enter once price reclaims VWAP at {vwap:.2f}.", ltp)

    if fresh and vol_ok:
        return ("ENTER", f"EMA 9/15 crossed {cross_dir} {bars_since} bar(s) ago in a trend — enter now.",
                "Fresh crossover confirmed with volume.", ltp)

    if ext >= EXTENDED:
        return ("WAIT", f"Extended {ext:.1f} ATR beyond EMA15 — don't chase.",
                f"Wait for a pullback toward EMA15 at {ema15:.2f}.", ema15)

    if abs(ext) <= AT_EMA:
        return ("ENTER", f"Pullback to EMA15 in a clean {d}trend — enter now.",
                f"Price at the moving average ({ema15:.2f}); low-risk entry.", ltp)

    return ("WAIT", f"Trend is {d} but no fresh trigger yet.",
            f"Enter on the next EMA 9/15 cross or a dip to EMA15 {ema15:.2f}.", ema15)


def _factors(long, cross_dir, bars_since, adx, trending, vwap_ok, vwap,
             rsi, vol_x, ema9, ema15):
    since = f"{bars_since} bar(s) ago" if bars_since is not None else "earlier"
    rsi_dir = 1 if rsi > 55 else -1 if rsi < 45 else 0
    return [
        {"factor": "EMA 9/15 cross", "dir": 1 if cross_dir == "up" else -1,
         "text": f"EMA9 {'above' if ema9 >= ema15 else 'below'} EMA15 — crossed {cross_dir} {since}"},
        {"factor": "Regime (ADX)", "dir": 0,
         "text": f"ADX {adx:.0f} — {'trending, crossover valid' if trending else 'choppy, stand aside'}"},
        {"factor": "VWAP", "dir": 1 if (vwap_ok and long) else -1 if (vwap_ok and not long) else 0,
         "text": f"Price {'on the right side of' if vwap_ok else 'against'} VWAP ({vwap:.2f})"},
        {"factor": "Momentum (RSI)", "dir": rsi_dir,
         "text": f"RSI {rsi:.0f} — {'bullish' if rsi_dir > 0 else 'bearish' if rsi_dir < 0 else 'neutral'}"},
        {"factor": "Volume", "dir": 0,
         "text": f"Volume {vol_x:.1f}× average — {'confirming' if vol_x >= 1 else 'light'}"},
    ]
