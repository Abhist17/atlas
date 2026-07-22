"""Live scanner — the tool that watches every F&O underlying at once and flags
the setups aligning right now, so you never miss a move across names you can't
watch simultaneously.

Produces a ranked snapshot: for each underlying, its latest price, signed
directional conviction, suggested side (CALL/PUT), and the confirming factors.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from data.underlyings import universe, yf_ticker
from data.yf_client import yfc
from engine.directional import _dir_score, add_opening_range
from engine.indicators import add_indicators
from utils.logger import get_logger

log = get_logger("engine.scanner")


def _factors(row: pd.Series) -> str:
    """Human-readable list of which confirmations are active."""
    f = []
    f.append("trend↑" if row["ema9"] > row["ema21"] else "trend↓")
    f.append("vwap↑" if row["close"] > row["vwap"] else "vwap↓")
    if row["rsi"] >= 55:
        f.append(f"rsi{int(row['rsi'])}")
    elif row["rsi"] <= 45:
        f.append(f"rsi{int(row['rsi'])}")
    if not np.isnan(row.get("or_high", np.nan)):
        if row["close"] > row["or_high"]:
            f.append("ORB↑")
        elif row["close"] < row["or_low"]:
            f.append("ORB↓")
    if row["volume"] > 1.5 * row["avg_volume"]:
        f.append("vol⚡")
    return " ".join(f)


def scan(threshold: float = 0.4, names: list[str] | None = None,
         interval: int = 5) -> pd.DataFrame:
    """Return current setups ranked by absolute conviction (strongest first).

    columns = [underlying, price, conviction, side, factors, rsi, at]
    """
    names = names or universe()
    rows = []
    for name in names:
        try:
            df = yfc.intraday(name if name.startswith("^") else yf_ticker(name),
                              days=2, interval=interval, use_cache=False)
            if df is None or len(df) < 25:
                continue
            ind = add_opening_range(add_indicators(df)).dropna(
                subset=["rsi", "atr", "vwap", "ema21"])
            if ind.empty:
                continue
            last = ind.iloc[-1]
            score = _dir_score(last)
            if abs(score) < threshold:
                continue
            rows.append({
                "underlying": name,
                "price": round(float(last["close"]), 2),
                "conviction": round(float(score), 3),
                "side": "CALL" if score > 0 else "PUT",
                "factors": _factors(last),
                "rsi": round(float(last["rsi"]), 1),
                "at": pd.to_datetime(last["timestamp"]).strftime("%H:%M"),
            })
        except Exception as e:
            log.warning("scan skip %s: %s", name, e)

    snap = pd.DataFrame(rows)
    if not snap.empty:
        snap = snap.reindex(
            snap["conviction"].abs().sort_values(ascending=False).index
        ).reset_index(drop=True)
    log.info("Scan: %d setups >= %.2f conviction", len(snap), threshold)
    return snap


if __name__ == "__main__":
    print(f"Atlas scan @ {datetime.now():%H:%M}")
    s = scan(threshold=0.4)
    print(s.to_string(index=False) if not s.empty else "No setups right now.")
