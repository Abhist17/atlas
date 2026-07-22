"""Directional backtest for the options system.

For each underlying we take at most one position at a time. Entries fire only on
high-conviction signals (selective). Exits use an ATR-based trailing stop so
winners ride and noise doesn't shake us out, plus an EOD square-off.

We measure the thing that actually matters for option buying: directional
accuracy and the size of the underlying move captured. Option P&L is then
ESTIMATED via a leverage factor (near-ATM intraday options move roughly this
many times the underlying's % move on the premium) — clearly an approximation,
since we have no free option price data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.yf_client import yfc
from data.underlyings import universe, yf_ticker
from engine.directional import add_opening_range, _dir_score
from engine.indicators import add_indicators
from utils.logger import get_logger

log = get_logger("backtest.directional")

# Rough leverage of a near-ATM weekly option's premium vs the underlying % move,
# intraday. Conservative; real value varies with moneyness/IV/theta.
OPTION_LEVERAGE = 6.0


def _prep(name: str, days: int, interval: int) -> pd.DataFrame | None:
    df = yfc.intraday(name if name.startswith("^") else yf_ticker(name),
                      days=days, interval=interval)
    if df is None or len(df) < 25:
        return None
    ind = add_opening_range(add_indicators(df)).dropna(
        subset=["rsi", "atr", "vwap", "ema21"])
    return ind if not ind.empty else None


def _backtest_one(df: pd.DataFrame, threshold: float, atr_mult: float,
                  square_off="15:15") -> list[dict]:
    """Run the trailing-stop directional logic on one underlying."""
    so_h, so_m = map(int, square_off.split(":"))
    df = df.reset_index(drop=True)
    scores = df.apply(_dir_score, axis=1)

    trades, pos = [], None
    prev_day = None
    for i, row in df.iterrows():
        ts = pd.to_datetime(row["timestamp"])
        day = ts.date()
        price, atr = row["close"], row["atr"]

        # Manage open position
        if pos is not None:
            if pos["dir"] == 1:
                pos["stop"] = max(pos["stop"], price - atr_mult * atr)
                hit = price <= pos["stop"]
            else:
                pos["stop"] = min(pos["stop"], price + atr_mult * atr)
                hit = price >= pos["stop"]
            eod = (ts.hour, ts.minute) >= (so_h, so_m) or day != prev_day
            if hit or eod:
                ret = (price / pos["entry"] - 1) * 100 * pos["dir"]
                trades.append({"dir": pos["dir"], "entry": pos["entry"],
                               "exit": price, "under_ret_pct": round(ret, 3),
                               "win": ret > 0})
                pos = None

        # New entry (only if flat, high conviction, and not at square-off)
        if pos is None and (ts.hour, ts.minute) < (so_h, so_m):
            s = scores.iloc[i]
            if abs(s) >= threshold and not np.isnan(atr) and atr > 0:
                d = 1 if s > 0 else -1
                pos = {"dir": d, "entry": price,
                       "stop": price - d * atr_mult * atr}
        prev_day = day
    return trades


def run(days: int = 30, interval: int = 5, threshold: float = 0.5,
        atr_mult: float = 1.5, names: list[str] | None = None) -> dict:
    names = names or universe()
    all_trades = []
    for name in names:
        df = _prep(name, days, interval)
        if df is None:
            continue
        all_trades += _backtest_one(df, threshold, atr_mult)

    if not all_trades:
        return {"trades": 0}
    t = pd.DataFrame(all_trades)
    wins = t[t["win"]]
    est_opt = t["under_ret_pct"] * OPTION_LEVERAGE  # approximate option % return
    return {
        "trades": len(t),
        "accuracy_pct": round(len(wins) / len(t) * 100, 2),
        "avg_under_move_pct": round(t["under_ret_pct"].mean(), 3),
        "avg_win_move_pct": round(wins["under_ret_pct"].mean(), 3) if len(wins) else 0,
        "avg_loss_move_pct": round(t[~t["win"]]["under_ret_pct"].mean(), 3),
        "est_option_ret_per_trade_pct": round(est_opt.mean(), 2),
        "est_option_total_pct": round(est_opt.sum(), 2),
        "trades_log": t,
    }


if __name__ == "__main__":
    for thr in (0.4, 0.5, 0.6):
        r = run(days=30, threshold=thr, atr_mult=1.5)
        print(f"\n=== threshold={thr} (selectivity) ===")
        for k, v in r.items():
            if k != "trades_log":
                print(f"  {k:30} {v}")
