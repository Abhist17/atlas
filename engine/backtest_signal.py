"""Backtest the live entry-timing strategy on historical intraday bars.

Replays exactly what the dashboard does: at each 5-min bar we compute the same
5-factor conviction and entry decision. When the signal says ENTER, we open a
trade at that bar's close with an ATR stop and an R-multiple target, then walk
forward bar-by-bar until the stop or target is hit (intrabar, using high/low),
or the session ends (intraday square-off). No look-ahead — every decision uses
only information available at that bar.

This backtests the SAME code path used live, so the numbers are honest about
what the signal would actually have done.
"""
from __future__ import annotations

import numpy as np

from data.yf_client import yfc
from engine.directional import _dir_score, add_opening_range
from engine.indicators import add_indicators
from engine.levels import _ATR_STOP, _entry_decision
from utils.logger import get_logger

log = get_logger("engine.backtest_signal")


def backtest_symbol(symbol: str, days: int = 30, interval: int = 5,
                    target_r: float = 2.0) -> dict:
    """Return backtest stats + trade list + equity curve for one symbol."""
    symbol = symbol.upper()
    df = yfc.intraday(symbol, days=days, interval=interval)
    if df is None or len(df) < 60:
        return {"symbol": symbol, "ok": False, "error": "Not enough history."}

    ind = add_opening_range(add_indicators(df)).dropna(
        subset=["atr", "vwap", "ema9", "ema21", "rsi", "avg_volume"]).reset_index(drop=True)
    if len(ind) < 60:
        return {"symbol": symbol, "ok": False, "error": "Not enough clean bars."}

    date = np.array([d.date() for d in _dt(ind)])
    close = ind["close"].to_numpy()
    high = ind["high"].to_numpy()
    low = ind["low"].to_numpy()
    atr = ind["atr"].to_numpy()

    trades = []
    n = len(ind)
    i = 0
    while i < n - 1:
        row = ind.iloc[i]
        recent = ind.iloc[max(0, i - 11):i + 1]
        status, entry_px = _signal_at(row, recent)
        if status != "ENTER":
            i += 1
            continue

        long = _dir_score(row) >= 0
        a = float(atr[i])
        risk = _ATR_STOP * a
        entry = float(close[i])
        if long:
            stop, target = entry - risk, entry + target_r * risk
        else:
            stop, target = entry + risk, entry - target_r * risk

        # walk forward within the same session
        exit_px, outcome, j = None, None, i + 1
        while j < n and date[j] == date[i]:
            hi, lo = float(high[j]), float(low[j])
            if long:
                if lo <= stop:
                    exit_px, outcome = stop, "SL"; break
                if hi >= target:
                    exit_px, outcome = target, "TP"; break
            else:
                if hi >= stop:
                    exit_px, outcome = stop, "SL"; break
                if lo <= target:
                    exit_px, outcome = target, "TP"; break
            j += 1
        if exit_px is None:  # square off at last bar of session
            j = min(j, n - 1)
            exit_px, outcome = float(close[j]), "EOD"

        r = ((exit_px - entry) if long else (entry - exit_px)) / risk
        trades.append({
            "side": "LONG" if long else "SHORT",
            "entry": round(entry, 2), "exit": round(exit_px, 2),
            "stop": round(stop, 2), "target": round(target, 2),
            "outcome": outcome, "r": round(float(r), 2),
            "when": str(_dt(ind).iloc[i]),
        })
        i = j + 1  # resume after the trade closes

    return _summarise(symbol, trades, days, target_r)


def _signal_at(row, recent):
    """Run the same ENTER/WAIT/AVOID decision as live, for one bar."""
    score = float(_dir_score(row))
    long = score >= 0
    conf = min(abs(score), 1.0)
    ltp = float(row["close"])
    vwap = float(row["vwap"])
    ema9 = float(row["ema9"])
    a = float(row["atr"])
    orh = float(row.get("or_high", np.nan))
    orl = float(row.get("or_low", np.nan))
    swing_hi = float(recent["high"].max())
    swing_lo = float(recent["low"].min())
    vol_x = float(row["volume"] / row["avg_volume"]) if row["avg_volume"] else 0.0
    ext = ((ltp - vwap) / a) if long else ((vwap - ltp) / a)
    status, _, _, entry_px = _entry_decision(
        long, conf, ext, ltp, vwap, ema9, a, orh, orl, swing_hi, swing_lo, vol_x)
    return status, entry_px


def _summarise(symbol, trades, days, target_r):
    if not trades:
        return {"symbol": symbol, "ok": True, "trades": 0,
                "note": "No ENTER signals fired in this window."}
    rs = np.array([t["r"] for t in trades])
    wins = rs[rs > 0]
    losses = rs[rs < 0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    equity, cum = [], 0.0
    for t in trades:
        cum += t["r"]
        equity.append(round(cum, 2))
    return {
        "symbol": symbol, "ok": True, "days": days, "target_r": target_r,
        "trades": len(trades),
        "win_rate": round(len(wins) / len(trades) * 100, 1),
        "total_r": round(float(rs.sum()), 2),
        "avg_r": round(float(rs.mean()), 2),
        "expectancy": round(float(rs.mean()), 3),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else None,
        "best_r": round(float(rs.max()), 2),
        "worst_r": round(float(rs.min()), 2),
        "wins": int(len(wins)), "losses": int(len(losses)),
        "equity": equity,
        "recent_trades": trades[-12:][::-1],
    }


def _dt(ind):
    import pandas as pd
    return pd.to_datetime(ind["timestamp"])
