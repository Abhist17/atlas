"""Alpha engine — multi-timeframe confluence for higher-quality entries.

This is the upgrade over a single EMA crossover. It does NOT try to predict the
market better (short-term direction is noise); it improves entries by being
*selective* — the way disciplined quant/discretionary traders actually filter:

  1. Higher-timeframe gate (15m): the 15m trend sets the ONLY direction allowed.
     We never take a 5m long when the 15m trend is down. This kills most bad trades.
  2. Regime gate (ADX): crossovers/momentum only work in a trending tape. Low ADX
     -> AVOID (chop).
  3. 5m confluence stack: EMA 9/15 cross, MACD histogram, RSI, VWAP side, volume,
     opening-range. Each is an independent vote in the candidate direction.
  4. Grade + timing: count the aligned votes -> A+/A/B. ENTER only on a high-grade,
     fresh, non-extended setup; otherwise WAIT (right idea, wrong moment) or AVOID.

Output matches the dashboard signal schema, with two extra fields: `grade` and
`confluence` (aligned votes / total). Honest note stays: selectivity, not prophecy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.live_feed import get_bars
from data.market_context import get_context
from data.yf_client import yfc
from engine.directional import add_opening_range
from engine.indicators import add_indicators
from engine.prob_model import win_probability
from utils.logger import get_logger

log = get_logger("engine.alpha_signal")

ADX_MIN = 20.0
FRESH_BARS = 4
EXTENDED = 1.6      # ATRs from EMA15 beyond which we don't chase
AT_EMA = 0.5
_ATR_STOP = 1.3
_ATM_DELTA = 0.5
GRADE_MIN = 4       # min aligned votes (of 6) to allow ENTER
OPEN_NOISE_END = (9, 30)    # first 15 min of the session: let the auction settle
LATE_ENTRY_CUTOFF = (15, 0)  # no fresh intraday entries in the last half hour


def compute_signal(symbol: str, interval: int = 5) -> dict:
    symbol = symbol.upper()
    feed = get_bars(symbol, days=5, interval=interval)
    if not feed.get("ok"):
        return {"symbol": symbol, "ok": False, "error": feed.get("error", "No data.")}
    ind = add_opening_range(add_indicators(feed["bars"])).dropna(
        subset=["ema9", "ema15", "atr", "vwap"]).reset_index(drop=True)
    if len(ind) < 25:
        return {"symbol": symbol, "ok": False, "error": "Not enough clean bars."}

    htf_bias = _htf_bias(symbol)          # +1 up / -1 down / 0 neutral (15m)
    last = ind.iloc[-1]
    ltp = float(feed["ltp"])
    atr = float(last["atr"])
    ema9, ema15 = float(last["ema9"]), float(last["ema15"])
    vwap = float(last["vwap"])
    rsi = float(last["rsi"]) if not np.isnan(last["rsi"]) else 50.0
    adx = float(last["adx"]) if not np.isnan(last.get("adx", np.nan)) else 0.0
    macd_h = float(last.get("macd_hist", np.nan)) if not np.isnan(last.get("macd_hist", np.nan)) else 0.0
    vol_x = float(last["volume"] / last["avg_volume"]) if last["avg_volume"] else 0.0

    # candidate direction: prefer the EMA cross, tie-broken by HTF
    diff = (ind["ema9"] - ind["ema15"]).to_numpy()
    sign = np.sign(diff)
    long = (sign[-1] >= 0) if sign[-1] != 0 else (htf_bias >= 0)
    bias = "LONG" if long else "SHORT"
    opt = "CALL" if long else "PUT"
    bars_since, cross_dir = _bars_since_cross(sign)
    fresh = bars_since is not None and bars_since <= FRESH_BARS

    # --- confluence votes (each +1 if it confirms the candidate direction) ---
    orh = float(last.get("or_high", np.nan))
    orl = float(last.get("or_low", np.nan))
    votes = _votes(long, ema9, ema15, macd_h, rsi, ltp, vwap, vol_x, orh, orl)
    aligned = sum(1 for v in votes if v["ok"])
    total = len(votes)
    grade = "A+" if aligned >= 6 else "A" if aligned == 5 else "B" if aligned == 4 else "C"

    htf_ok = (htf_bias > 0 and long) or (htf_bias < 0 and not long)
    trending = adx >= ADX_MIN
    vwap_ok = (ltp > vwap) if long else (ltp < vwap)
    ext = ((ltp - ema15) / atr) if long else ((ema15 - ltp) / atr)

    # broad-market / sector context (NIFTY or BANKNIFTY)
    mkt = get_context(symbol)
    mkt_dir = mkt["bias"]
    mkt_against = mkt_dir != 0 and ((mkt_dir > 0) != long)
    mkt_aligned = mkt_dir != 0 and ((mkt_dir > 0) == long)

    # relative strength: stock's intraday move vs the market (excess return)
    _tday = ind[ind["timestamp"].astype(str).str[:10] == str(last["timestamp"])[:10]]
    day_open = float(_tday["open"].iloc[0]) if not _tday.empty else ltp
    stock_chg = (ltp / day_open - 1) * 100 if day_open else 0.0
    rs = round(stock_chg - mkt["chg"], 2)                 # + = outperforming
    rs_ok = (rs > 0.15) if long else (rs < -0.15)          # a leader in its direction
    rs_bad = (rs < -0.15) if long else (rs > 0.15)         # a laggard (against)

    # heuristic confluence confidence (fallback)
    heuristic = round(min(0.10 + aligned / total * 0.56
                          + (0.14 if htf_ok else 0)
                          + (0.12 if mkt_aligned else -0.12 if mkt_against else 0)
                          + (0.08 if rs_ok else -0.08 if rs_bad else 0), 1.0), 2)

    # calibrated win probability from the learned model (if trained)
    ema50 = float(last["ema50"]) if not np.isnan(last.get("ema50", np.nan)) else ema15
    win_prob = win_probability({
        "close": ltp, "ema9": ema9, "ema15": ema15, "ema50": ema50, "atr": atr,
        "macd_hist": macd_h, "rsi": rsi, "vwap": vwap, "adx": adx, "vol_x": vol_x})
    if win_prob is not None:
        # market/relative-strength context nudges the calibrated probability
        confidence = win_prob + (0.05 if mkt_aligned else -0.05 if mkt_against else 0) \
                     + (0.03 if rs_ok else -0.03 if rs_bad else 0)
        confidence = round(min(max(confidence, 0.0), 1.0), 2)
    else:
        confidence = max(heuristic, 0.0)

    status, headline, trigger, entry = _decide(
        long, htf_bias, htf_ok, trending, adx, aligned, grade, fresh, bars_since,
        cross_dir, vwap_ok, ext, ltp, ema15, vwap)

    # market gate: never ENTER straight into a broad market moving against you
    if status == "ENTER" and mkt_against:
        status = "WAIT"
        headline = f"{mkt['name']} is moving {'up' if mkt_dir>0 else 'down'} — against this {('long' if long else 'short')}."
        trigger = f"Wait for {mkt['name']} to turn, or stand aside."

    # time-of-day gate: the opening 15 min is noise, the last half hour has no runway
    phase, phase_txt = _session_phase(last["timestamp"])
    if status == "ENTER" and phase == "open":
        status = "WAIT"
        headline = "Opening 15 minutes — let the auction settle before entering."
        trigger = f"Re-check after {OPEN_NOISE_END[0]:02d}:{OPEN_NOISE_END[1]:02d}."
    elif status == "ENTER" and phase == "late":
        status = "AVOID"
        headline = "Too late in the session for a fresh intraday entry."
        trigger = "No new positions this close to the close."

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

    levels = {"EMA 9": _r(ema9), "EMA 15": _r(ema15), "VWAP": _r(vwap), "ADX": _r(adx),
              "Day high": _r(today["high"].max()), "Day low": _r(today["low"].min())}
    opt_move_sl = round(abs(entry - stop) * _ATM_DELTA, 2)
    opt_move_tp = [round(abs(t - entry) * _ATM_DELTA, 2) for t in tps]

    htf_txt = {1: "up", -1: "down", 0: "flat"}[htf_bias]
    factors = [{"factor": "15m trend", "dir": htf_bias,
                "text": f"Higher timeframe is {htf_txt} — "
                        f"{'aligned' if htf_ok else 'NOT aligned, caution'}"}]
    factors += [{"factor": v["name"], "dir": (1 if long else -1) if v["ok"] else 0,
                 "text": v["text"]} for v in votes]
    factors.append({"factor": "Regime (ADX)", "dir": 0,
                    "text": f"ADX {adx:.0f} — {'trending' if trending else 'choppy'}"})
    mkt_txt = {1: "up", -1: "down", 0: "flat"}[mkt_dir]
    factors.append({"factor": f"Market ({mkt['name']})",
                    "dir": mkt_dir if mkt_aligned else -1 if mkt_against else 0,
                    "text": f"{mkt['name']} {mkt_txt} ({mkt['chg']:+.2f}%) — "
                            f"{'aligned' if mkt_aligned else 'against, caution' if mkt_against else 'neutral'}"})
    factors.append({"factor": "Time of day", "dir": 0 if phase == "core" else -1,
                    "text": phase_txt})
    factors.append({"factor": "Relative strength",
                    "dir": (1 if long else -1) if rs_ok else (-1 if rs_bad else 0),
                    "text": f"{'Out' if rs > 0 else 'Under'}performing {mkt['name']} by {rs:+.2f}% — "
                            f"{'leader' if rs_ok else 'laggard, caution' if rs_bad else 'in line'}"})

    return {
        "symbol": symbol, "ok": True, "ltp": round(ltp, 2),
        "bias": bias, "option": opt, "confidence": confidence,
        "grade": grade, "confluence": f"{aligned}/{total}",
        "atr": round(atr, 2), "atr_pct": round(atr / ltp * 100, 2),
        "status": status, "headline": headline, "trigger": trigger,
        "extension": round(ext, 2),
        "entry": round(entry, 2), "stop": round(stop, 2),
        "risk_pts": round(abs(entry - stop), 2),
        "risk_pct": round(abs(entry - stop) / ltp * 100, 2),
        "targets": [{"px": round(tps[i], 2), "rr": i + 1,
                     "pct": round((tps[i] / ltp - 1) * 100, 2), "opt_gain": opt_move_tp[i]}
                    for i in range(3)],
        "opt_stop_move": opt_move_sl,
        "levels": {k: v for k, v in levels.items() if v is not None},
        "factors": factors,
        "live": {"is_live": feed["is_live"], "source": feed["source"]},
        "market": {"name": mkt["name"], "bias": mkt_dir, "chg": mkt["chg"]},
        "rel_strength": rs,
        "win_prob": win_prob,
        "note": (("Win prob from a model trained on historical outcomes · "
                  if win_prob is not None else "")
                 + f"Grade {grade} · {aligned}/{total} aligned · 15m + {mkt['name']} filtered · not a prediction"),
    }


def _htf_bias(symbol: str) -> int:
    """15-min trend bias: +1 up / -1 down / 0 neutral."""
    try:
        df = yfc.intraday(symbol, days=15, interval=15)
        if df is None or len(df) < 55:
            return 0
        h = add_indicators(df).dropna(subset=["ema21"])
        if h.empty:
            return 0
        r = h.iloc[-1]
        # price vs 15m EMA21 with a small deadband to avoid whipsaw at the line
        band = 0.0015 * float(r["ema21"])   # ~0.15%
        if float(r["close"]) > float(r["ema21"]) + band:
            return 1
        if float(r["close"]) < float(r["ema21"]) - band:
            return -1
        return 0
    except Exception as e:
        log.debug("HTF bias failed for %s: %s", symbol, e)
        return 0


def _session_phase(ts) -> tuple[str, str]:
    """Where in the trading day are we? -> (phase, human text).

    phase: "open" (first 15 min, auction noise), "late" (last half hour, no time
    for the trade to work + squaring-off flow), or "core" (tradeable).
    """
    try:
        t = pd.Timestamp(ts)
        hm = (int(t.hour), int(t.minute))
    except Exception:
        return "core", "Session time unknown"
    if hm < OPEN_NOISE_END:
        return "open", f"{hm[0]:02d}:{hm[1]:02d} — opening 15 min, prices still unsettled"
    if hm >= LATE_ENTRY_CUTOFF:
        return "late", f"{hm[0]:02d}:{hm[1]:02d} — late session, too little time left to work"
    return "core", f"{hm[0]:02d}:{hm[1]:02d} — core session"


def _votes(long, ema9, ema15, macd_h, rsi, ltp, vwap, vol_x, orh, orl):
    up = long
    ema_ok = (ema9 >= ema15) if up else (ema9 < ema15)
    macd_ok = (macd_h > 0) if up else (macd_h < 0)
    rsi_ok = (rsi >= 52) if up else (rsi <= 48)
    vwap_ok = (ltp > vwap) if up else (ltp < vwap)
    vol_ok = vol_x >= 1.0
    if not np.isnan(orh):
        or_ok = (ltp >= orh) if up else (ltp <= orl)
        or_txt = (f"{'Above' if up else 'Below'} opening range — "
                  f"{'breakout' if or_ok else 'inside range'}")
    else:
        or_ok, or_txt = False, "Opening range not set"
    return [
        {"name": "EMA 9/15", "ok": ema_ok,
         "text": f"EMA9 {'above' if ema9 >= ema15 else 'below'} EMA15"},
        {"name": "MACD", "ok": macd_ok,
         "text": f"MACD histogram {'positive' if macd_h > 0 else 'negative'} ({macd_h:+.2f})"},
        {"name": "RSI", "ok": rsi_ok, "text": f"RSI {rsi:.0f}"},
        {"name": "VWAP", "ok": vwap_ok,
         "text": f"Price {'above' if ltp > vwap else 'below'} VWAP ({vwap:.2f})"},
        {"name": "Volume", "ok": vol_ok, "text": f"Volume {vol_x:.1f}x average"},
        {"name": "Opening range", "ok": or_ok, "text": or_txt},
    ]


def _decide(long, htf_bias, htf_ok, trending, adx, aligned, grade, fresh,
            bars_since, cross_dir, vwap_ok, ext, ltp, ema15, vwap):
    d = "up" if long else "down"

    if not trending:
        return ("AVOID", f"Choppy tape (ADX {adx:.0f}) — no clean trend to trade.",
                "Wait for ADX above 20.", ltp)
    if htf_bias == 0:
        return ("AVOID", "15-min trend is flat — no directional edge.",
                "Wait for the higher timeframe to pick a side.", ltp)
    if not htf_ok:
        return ("AVOID", f"Setup fights the 15-min trend ({'up' if htf_bias>0 else 'down'}).",
                "Only trade with the higher timeframe. Stand aside.", ltp)
    if aligned < GRADE_MIN:
        return ("AVOID", f"Only {aligned}/6 factors aligned (grade {grade}) — too weak.",
                "Wait for a higher-confluence setup.", ltp)
    if not vwap_ok:
        return ("WAIT", f"Grade {grade} {d} setup, but price is on the wrong side of VWAP.",
                f"Enter once price reclaims VWAP {vwap:.2f}.", ltp)
    if ext >= EXTENDED:
        return ("WAIT", f"Grade {grade} but extended {ext:.1f} ATR from EMA15 — don't chase.",
                f"Wait for a pullback toward EMA15 {ema15:.2f}.", ema15)
    if fresh or abs(ext) <= AT_EMA:
        why = (f"fresh EMA cross {cross_dir} {bars_since} bar(s) ago"
               if fresh else "pullback to EMA15")
        return ("ENTER", f"Grade {grade} {d} setup, {aligned}/6 aligned with 15m — enter now.",
                f"High-confluence entry ({why}).", ltp)
    return ("WAIT", f"Grade {grade} {d} setup, but no fresh trigger yet.",
            f"Enter on the next EMA cross or a dip to EMA15 {ema15:.2f}.", ema15)


def _bars_since_cross(sign):
    for i in range(len(sign) - 1, 0, -1):
        if sign[i] != sign[i - 1] and sign[i] != 0:
            return (len(sign) - 1 - i), ("up" if sign[i] > 0 else "down")
    return None, ("up" if sign[-1] >= 0 else "down")
