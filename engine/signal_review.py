"""Forward-scoring for journaled signals — did the call actually work?

`storage/live_journal.py` records what the app said. This scores it against what
the market then did, so the dashboard can show a real hit rate instead of a
remembered one.

Two rules keep the numbers honest rather than flattering:

1. **Only bars after the decision bar count.** The signal was decided on a closed
   bar; scoring it on that same bar's range would be reading the data the
   decision was made from.
2. **When a bar touches both the stop and a target, the stop wins.** Intraday
   OHLC does not say which came first, and assuming the good one is how
   backtests end up describing a strategy nobody could have traded.

A trade that is still alive at the square-off time is closed there at that bar's
close (`TIMEOUT`) — Atlas is an intraday tool, so an open position is not a
pending winner, it is a position you had to flatten.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("engine.signal_review")

SQUARE_OFF = (15, 15)     # IST — flatten intraday positions here


def score_signal(row: dict, bars: pd.DataFrame,
                 square_off: tuple[int, int] = SQUARE_OFF) -> dict:
    """Score one journaled signal against the bars that followed it.

    `row` needs: bar_time, bias, entry, stop (tp1/tp2/tp3 optional — derived
    from the 1R risk when absent). `bars` needs: timestamp, high, low, close.

    Returns {outcome, r, exit_px, exit_time, bars_held, mfe_r, mae_r}.
    `outcome` is one of TP3, TP2, TP1, STOP, TIMEOUT, OPEN.
    """
    entry, stop = float(row["entry"]), float(row["stop"])
    risk = abs(entry - stop)
    long = str(row.get("bias", "LONG")).upper() == "LONG"
    blank = {"outcome": "OPEN", "r": None, "exit_px": None, "exit_time": None,
             "bars_held": 0, "mfe_r": None, "mae_r": None}
    if risk <= 0 or bars is None or bars.empty:
        return blank

    fwd = bars[pd.to_datetime(bars["timestamp"]) > pd.Timestamp(row["bar_time"])]
    if fwd.empty:
        return blank

    tps = [row.get(f"tp{i}") for i in (1, 2, 3)]
    tps = [float(t) if t is not None and not _isnan(t) else
           (entry + i * risk if long else entry - i * risk)
           for i, t in enumerate(tps, start=1)]

    mfe = mae = 0.0
    held = 0
    for _, b in fwd.iterrows():
        held += 1
        hi, lo = float(b["high"]), float(b["low"])
        fav = (hi - entry) if long else (entry - lo)
        adv = (entry - lo) if long else (hi - entry)
        mfe = max(mfe, fav / risk)
        mae = max(mae, adv / risk)

        stopped = (lo <= stop) if long else (hi >= stop)
        if stopped:                       # stop wins a same-bar tie, always
            return _result("STOP", -1.0, stop, b["timestamp"], held, mfe, mae)

        for level, tp in zip((3, 2, 1), reversed(tps)):
            hit = (hi >= tp) if long else (lo <= tp)
            if hit:
                return _result(f"TP{level}", float(level), tp,
                               b["timestamp"], held, mfe, mae)

        if _at_or_past(b["timestamp"], square_off):
            px = float(b["close"])
            r = (px - entry) / risk if long else (entry - px) / risk
            return _result("TIMEOUT", r, px, b["timestamp"], held, mfe, mae)

    return {"outcome": "OPEN", "r": None, "exit_px": None, "exit_time": None,
            "bars_held": held, "mfe_r": round(mfe, 2), "mae_r": round(mae, 2)}


def _result(outcome, r, px, ts, held, mfe, mae) -> dict:
    return {"outcome": outcome, "r": round(float(r), 2), "exit_px": round(float(px), 2),
            "exit_time": str(ts), "bars_held": held,
            "mfe_r": round(mfe, 2), "mae_r": round(mae, 2)}


def _isnan(v) -> bool:
    try:
        return bool(np.isnan(float(v)))
    except (TypeError, ValueError):
        return True


def _at_or_past(ts, hm: tuple[int, int]) -> bool:
    t = pd.Timestamp(ts)
    return (int(t.hour), int(t.minute)) >= hm


def summarise(scored: pd.DataFrame) -> dict:
    """Aggregate scored ENTER calls into the numbers worth trusting."""
    if scored is None or scored.empty:
        return {"scored": 0, "open": 0}
    done = scored[scored["outcome"].isin(["TP1", "TP2", "TP3", "STOP", "TIMEOUT"])]
    if done.empty:
        return {"scored": 0, "open": int((scored["outcome"] == "OPEN").sum())}
    r = done["r"].astype(float)
    wins = r[r > 0]
    losses = r[r <= 0]
    return {
        "scored": int(len(done)),
        "open": int((scored["outcome"] == "OPEN").sum()),
        "win_rate": round(len(wins) / len(done) * 100, 1),
        "avg_r": round(float(r.mean()), 2),
        "total_r": round(float(r.sum()), 2),
        "expectancy_r": round(float(r.mean()), 2),
        "best_r": round(float(r.max()), 2),
        "worst_r": round(float(r.min()), 2),
        "avg_win_r": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss_r": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "target_hits": int(done["outcome"].str.startswith("TP").sum()),
        "stopped": int((done["outcome"] == "STOP").sum()),
        "timed_out": int((done["outcome"] == "TIMEOUT").sum()),
    }


def review(interval: int = 5, limit: int = 200) -> dict:
    """Score every journaled ENTER call that has bars available to score it.

    Fetches recent intraday bars per symbol (the free feed only reaches back a
    few days), so older calls simply stay unscored rather than being guessed at.
    """
    from data.live_feed import get_bars
    from storage.live_journal import load

    df = load()
    if df.empty:
        return {"rows": [], "summary": {"scored": 0, "open": 0}, "logged": 0}

    enters = df[df["status"] == "ENTER"].tail(limit).copy()
    if enters.empty:
        return {"rows": [], "summary": {"scored": 0, "open": 0}, "logged": int(len(df))}

    cache: dict[str, pd.DataFrame] = {}
    scored = []
    for _, row in enters.iterrows():
        sym = row["symbol"]
        if sym not in cache:
            feed = get_bars(sym, days=5, interval=interval)
            bars = feed["bars"] if feed.get("ok") else pd.DataFrame()
            if not bars.empty and feed.get("forming"):
                bars = bars.iloc[:-1]      # only closed bars can score anything
            cache[sym] = bars
        scored.append({**row.to_dict(), **score_signal(row.to_dict(), cache[sym])})

    out = pd.DataFrame(scored)
    return {"rows": out.to_dict("records"), "summary": summarise(out),
            "logged": int(len(df))}
