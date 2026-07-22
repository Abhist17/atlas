"""NSE intraday screener — scans the full F&O universe with market-style
columns (LTP, % change, day OHLC, volume) plus technical signals, using bulk
data fetch for speed.

Returns one rich row per stock; filtering/preset-scans are applied by the
caller (dashboard).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.nse_universe import FNO_UNIVERSE
from data.yf_client import yfc
from engine.directional import _dir_score, add_opening_range
from engine.indicators import add_indicators
from utils.logger import get_logger

log = get_logger("engine.nse_screener")


def _row(symbol: str, df: pd.DataFrame) -> dict | None:
    """Build a screener row from a symbol's intraday frame."""
    if df is None or len(df) < 25:
        return None
    ind = add_opening_range(add_indicators(df)).dropna(
        subset=["rsi", "atr", "vwap", "ema21"])
    if ind.empty:
        return None

    ind = ind.copy()
    ind["date"] = pd.to_datetime(ind["timestamp"]).dt.date
    today = ind["date"].iloc[-1]
    day = ind[ind["date"] == today]
    prev = ind[ind["date"] < today]

    last = ind.iloc[-1]
    ltp = float(last["close"])
    day_open = float(day["open"].iloc[0])
    prev_close = float(prev["close"].iloc[-1]) if not prev.empty else day_open
    chg_pct = (ltp / prev_close - 1) * 100
    score = _dir_score(last)

    return {
        "symbol": symbol,
        "ltp": round(ltp, 2),
        "chg_%": round(chg_pct, 2),
        "open": round(day_open, 2),
        "high": round(float(day["high"].max()), 2),
        "low": round(float(day["low"].min()), 2),
        "prev_close": round(prev_close, 2),
        "volume": int(day["volume"].sum()),
        "vol_x_avg": round(float(last["volume"] / last["avg_volume"]), 2),
        "rsi": round(float(last["rsi"]), 1),
        "atr_%": round(float(last["atr_pct"]), 2),
        "vs_vwap_%": round((ltp / last["vwap"] - 1) * 100, 2),
        "trend": "up" if last["ema9"] > last["ema21"] else "down",
        "orb": ("↑" if ltp > last.get("or_high", np.inf)
                else "↓" if ltp < last.get("or_low", -np.inf) else "·"),
        "conviction": round(float(score), 3),
        "side": "CALL" if score > 0 else "PUT",
    }


def screen_nse(interval: int = 5, symbols: list[str] | None = None,
               progress=None) -> pd.DataFrame:
    """Screen the F&O universe. `progress` is an optional callback(frac, text)."""
    symbols = symbols or FNO_UNIVERSE
    tickers = [f"{s}.NS" for s in symbols]
    if progress:
        progress(0.1, f"Fetching {len(tickers)} stocks…")
    data = yfc.batch_intraday(tickers, days=2, interval=interval)

    rows = []
    for i, sym in enumerate(symbols):
        if progress and i % 20 == 0:
            progress(0.1 + 0.9 * i / len(symbols), f"Analysing {sym}…")
        df = data.get(f"{sym}.NS")
        r = _row(sym, df) if df is not None else None
        if r:
            rows.append(r)

    out = pd.DataFrame(rows)
    log.info("NSE screen: %d/%d stocks analysed", len(out), len(symbols))
    return out


if __name__ == "__main__":
    df = screen_nse()
    if not df.empty:
        top = df.reindex(df["conviction"].abs().sort_values(ascending=False).index)
        print(top.head(15).to_string(index=False))
