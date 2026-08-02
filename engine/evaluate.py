"""Signal-quality evaluation — does the alpha engine's selectivity actually work?

This is NOT a P&L backtest. It answers one question, honestly:

    Of all the bars where the engine says ENTER, what fraction reached +1R
    before -1R? And is that better than the base rate of every bar?

If ENTER does not beat the base rate, every gate in alpha_signal.py is
decoration and the confidence numbers in the UI are unearned.

Method
------
* Replay `engine/alpha_signal._decide` (plus the market and time-of-day gates)
  bar by bar over the whole F&O universe, vectorised.
* Decisions are made on **closed bars only** — the decision at bar i uses data
  through bar i and enters at close[i]. Outcomes are resolved from bar i+1
  onward. (Production currently decides on the forming bar; that repaints. This
  replay measures the intended behaviour, not that bug.)
* No lookahead anywhere: the 15m higher-timeframe gate uses the last *closed*
  15m bar, and the opening range expands causally instead of using
  `add_opening_range`'s session-wide `transform("max")`.
* Label: +1R before -1R within the same session, risk = 1.3 x ATR (matching
  alpha_signal._ATR_STOP). Unresolved by the close is reported as its own
  bucket, and counted as a loss in the headline number — a trade that goes
  nowhere and gets squared off at 15:30 is not a win. If both levels are
  touched on the same bar we cannot tell the order intrabar, so it counts as a
  loss (conservative).

Statistics
----------
Bars overlap heavily, so raw n wildly overstates the evidence. Every rate gets
a Wilson interval on the raw n *and* a session-clustered t-test (each trading
session is one independent observation). Trust the clustered number.

Run
---
    python -m engine.evaluate                     # full universe, 60 days
    python -m engine.evaluate --limit 40          # quick pass
    python -m engine.evaluate --days 30 --json out.json
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import CACHE_DIR
from data.nse_universe import FNO_UNIVERSE
from data.yf_client import yfc, _normalise
from engine.alpha_signal import (ADX_MIN, AT_EMA, EXTENDED, FRESH_BARS,
                                 GRADE_MIN, LATE_ENTRY_CUTOFF, OPEN_NOISE_END,
                                 _ATR_STOP)
from engine.indicators import add_indicators
from utils.logger import get_logger

log = get_logger("engine.evaluate")

OR_BARS = 6
_INDEX = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK"}
# mirrors data.market_context._BANKING
_BANKING = {
    "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "INDUSINDBK",
    "BANKBARODA", "PNB", "FEDERALBNK", "IDFCFIRSTB", "AUBANK", "BANDHANBNK",
    "BAJFINANCE", "BAJAJFINSV", "SBILIFE", "HDFCLIFE", "ICICIPRULI",
    "ICICIGI", "CHOLAFIN", "SHRIRAMFIN", "LICHSGFIN", "MUTHOOTFIN",
}

_OPEN_MIN = OPEN_NOISE_END[0] * 60 + OPEN_NOISE_END[1]
_LATE_MIN = LATE_ENTRY_CUTOFF[0] * 60 + LATE_ENTRY_CUTOFF[1]


# --------------------------------------------------------------------- data
def _fetch_universe(symbols: list[str], days: int) -> dict[str, pd.DataFrame]:
    """Bulk 5m bars, cached to one parquet so re-runs are instant."""
    cache = CACHE_DIR / f"eval_{len(symbols)}sym_{days}d_5m.parquet"
    if cache.exists():
        big = pd.read_parquet(cache)
        log.info("loaded %d cached bars for %d symbols",
                 len(big), big["symbol"].nunique())
        return {s: g.drop(columns="symbol").reset_index(drop=True)
                for s, g in big.groupby("symbol")}

    log.info("downloading %d symbols x %dd of 5m bars…", len(symbols), days)
    raw = yfc.batch_intraday([f"{s}.NS" for s in symbols], days=days, interval=5)
    out, frames = {}, []
    for s in symbols:
        df = raw.get(f"{s}.NS")
        if df is None or len(df) < 200:
            continue
        out[s] = df
        frames.append(df.assign(symbol=s))
    if frames:
        pd.concat(frames, ignore_index=True).to_parquet(cache, index=False)
    log.info("got %d/%d symbols", len(out), len(symbols))
    return out


def _fetch_index(ticker: str, days: int) -> pd.DataFrame | None:
    """Index 5m bars (yf_client appends .NS, which breaks ^NSEI)."""
    cache = CACHE_DIR / f"eval_idx_{ticker.strip('^')}_{days}d_5m.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    try:
        df = _normalise(yf.download(ticker, period=f"{days}d", interval="5m",
                                    progress=False, auto_adjust=True))
    except Exception as e:
        log.warning("index fetch failed for %s: %s", ticker, e)
        return None
    if df.empty:
        return None
    df.to_parquet(cache, index=False)
    return df


# ------------------------------------------------------- causal derivations
def _causal_opening_range(df: pd.DataFrame, day: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Opening-range high/low that only ever looks backward.

    `engine.directional.add_opening_range` broadcasts the session-wide max of
    the first OR_BARS bars to every bar of the day — inside those first bars
    that is lookahead. Here the range expands bar by bar and freezes once the
    window closes.
    """
    g = df.groupby(day)
    bar_no = g.cumcount().to_numpy()
    in_win = bar_no < OR_BARS
    hi = df["high"].where(in_win)
    lo = df["low"].where(in_win)
    orh = hi.groupby(day).cummax().to_numpy()
    orl = lo.groupby(day).cummin().to_numpy()
    # after the window, carry the frozen range forward
    s_h = pd.Series(orh).groupby(day.to_numpy()).ffill().to_numpy()
    s_l = pd.Series(orl).groupby(day.to_numpy()).ffill().to_numpy()
    return s_h, s_l


def _htf_bias_series(df: pd.DataFrame) -> np.ndarray:
    """15m trend bias (+1/-1/0) mapped onto 5m bars with no lookahead.

    Resamples the 5m bars to 15m, takes EMA21 with the same 0.15% deadband as
    `alpha_signal._htf_bias`, then attaches to each 5m bar the most recent 15m
    bar that had already *closed* by that 5m bar's close.
    """
    ts = pd.to_datetime(df["timestamp"])
    h15 = (df.assign(ts=ts).set_index("ts")
             .resample("15min", origin="start_day", offset="15min")
             .agg({"close": "last"}).dropna())
    if len(h15) < 25:
        return np.zeros(len(df), dtype=int)
    ema21 = h15["close"].ewm(span=21, adjust=False).mean()
    band = 0.0015 * ema21
    bias = np.where(h15["close"] > ema21 + band, 1,
                    np.where(h15["close"] < ema21 - band, -1, 0))
    right = pd.DataFrame({"close_at": h15.index + pd.Timedelta(minutes=15),
                          "htf": bias})
    left = pd.DataFrame({"close_at": ts + pd.Timedelta(minutes=5)})
    merged = pd.merge_asof(left.sort_values("close_at"),
                           right.sort_values("close_at"),
                           on="close_at", direction="backward")
    return merged["htf"].fillna(0).to_numpy().astype(int)


def _index_frame(days: int) -> dict[str, pd.DataFrame]:
    """Per-index bias and session-anchored % change, keyed by timestamp.

    NOTE the session-anchored change: `data/market_context.py:57` measures the
    index from 75 bars back, while the stock side measures from the session
    open — mismatched baselines. This uses the session open for both, which is
    what the engine intends.
    """
    out = {}
    for name, tk in _INDEX.items():
        df = _fetch_index(tk, days)
        if df is None or df.empty:
            continue
        c = df["close"].astype(float)
        e9 = c.ewm(span=9, adjust=False).mean()
        e21 = c.ewm(span=21, adjust=False).mean()
        bias = np.where((c > e21) & (e9 >= e21), 1,
                        np.where((c < e21) & (e9 <= e21), -1, 0))
        day = pd.to_datetime(df["timestamp"]).dt.date
        day_open = df.groupby(day)["open"].transform("first").astype(float)
        out[name] = pd.DataFrame({
            "timestamp": pd.to_datetime(df["timestamp"]),
            f"bias": bias,
            f"chg": (c / day_open - 1) * 100,
        })
    return out


# ------------------------------------------------------------------ labels
def _first_touch(high, low, entry, risk, is_long, session_id) -> np.ndarray:
    """1 = +1R first, 0 = -1R first, -1 = unresolved by session close.

    Both levels touched on the same bar -> 0, since intrabar order is unknown.
    """
    out = np.full(len(entry), -1, dtype=int)
    for _, idx in pd.Series(np.arange(len(entry))).groupby(session_id):
        i = idx.to_numpy()
        n = len(i)
        if n < 2:
            continue
        h, l = high[i], low[i]
        e, r, lg = entry[i], risk[i], is_long[i]
        tgt = np.where(lg, e + r, e - r)[:, None]
        stp = np.where(lg, e - r, e + r)[:, None]
        future = np.arange(n)[None, :] > np.arange(n)[:, None]
        lgc = lg[:, None]
        hit_t = np.where(lgc, h[None, :] >= tgt, l[None, :] <= tgt) & future
        hit_s = np.where(lgc, l[None, :] <= stp, h[None, :] >= stp) & future
        big = n + 1
        ft = np.where(hit_t.any(1), hit_t.argmax(1), big)
        fs = np.where(hit_s.any(1), hit_s.argmax(1), big)
        res = np.full(n, -1, dtype=int)
        res[ft < fs] = 1
        res[fs <= ft] = 0
        res[(ft == big) & (fs == big)] = -1
        out[i] = res
    return out


# ------------------------------------------------------------------ replay
def replay_symbol(symbol: str, df: pd.DataFrame, idx: dict) -> pd.DataFrame | None:
    """One symbol -> a frame of per-bar decisions, gates and outcomes."""
    if df is None or len(df) < 120:
        return None
    ind = add_indicators(df.reset_index(drop=True))
    ts = pd.to_datetime(ind["timestamp"])
    day = ts.dt.date
    ind = ind.assign(_day=day)

    orh, orl = _causal_opening_range(ind, ind["_day"])
    htf = _htf_bias_series(ind)

    ok = (~ind[["ema9", "ema15", "ema50", "atr", "vwap", "rsi", "adx"]]
          .isna().any(axis=1)).to_numpy() & (ind["atr"].to_numpy() > 0)

    close = ind["close"].to_numpy(float)
    ema9 = ind["ema9"].to_numpy(float)
    ema15 = ind["ema15"].to_numpy(float)
    atr = ind["atr"].to_numpy(float)
    vwap = ind["vwap"].to_numpy(float)
    rsi = ind["rsi"].to_numpy(float)
    adx = ind["adx"].to_numpy(float)
    mh = np.nan_to_num(ind["macd_hist"].to_numpy(float))
    avgv = ind["avg_volume"].to_numpy(float)
    vol_x = np.divide(ind["volume"].to_numpy(float), avgv,
                      out=np.zeros(len(ind)), where=avgv > 0)

    is_long = ema9 >= ema15

    # --- six confluence votes -------------------------------------------
    v_ema = np.where(is_long, ema9 >= ema15, ema9 < ema15)
    v_macd = np.where(is_long, mh > 0, mh < 0)
    v_rsi = np.where(is_long, rsi >= 52, rsi <= 48)
    v_vwap = np.where(is_long, close > vwap, close < vwap)
    v_vol = vol_x >= 1.0
    v_or = np.where(np.isnan(orh), False,
                    np.where(is_long, close >= orh, close <= orl))
    aligned = (v_ema.astype(int) + v_macd + v_rsi + v_vwap + v_vol + v_or)

    # --- bars since the last EMA 9/15 cross ------------------------------
    sign = np.sign(ema9 - ema15)
    changed = np.zeros(len(sign), bool)
    changed[1:] = (sign[1:] != sign[:-1]) & (sign[1:] != 0)
    pos = np.where(changed, np.arange(len(sign)), -1)
    last_cross = np.maximum.accumulate(pos)
    bars_since = np.where(last_cross >= 0, np.arange(len(sign)) - last_cross, 10**6)
    fresh = bars_since <= FRESH_BARS

    # --- context ---------------------------------------------------------
    name = "BANKNIFTY" if symbol in _BANKING else "NIFTY"
    ctx = idx.get(name)
    if ctx is None:
        mkt_bias = np.zeros(len(ind), int)
        mkt_chg = np.zeros(len(ind))
    else:
        m = pd.merge_asof(pd.DataFrame({"timestamp": ts}).sort_values("timestamp"),
                          ctx.sort_values("timestamp"), on="timestamp",
                          direction="backward", tolerance=pd.Timedelta(minutes=5))
        mkt_bias = m["bias"].fillna(0).to_numpy().astype(int)
        mkt_chg = m["chg"].fillna(0.0).to_numpy()

    day_open = ind.groupby("_day")["open"].transform("first").to_numpy(float)
    stock_chg = np.divide(close, day_open, out=np.ones(len(ind)),
                          where=day_open > 0) * 100 - 100
    rs = stock_chg - mkt_chg
    rs_ok = np.where(is_long, rs > 0.15, rs < -0.15)

    mkt_aligned = (mkt_bias != 0) & ((mkt_bias > 0) == is_long)
    mkt_against = (mkt_bias != 0) & ((mkt_bias > 0) != is_long)

    # --- gates (order mirrors alpha_signal._decide) ----------------------
    trending = adx >= ADX_MIN
    htf_ok = ((htf > 0) & is_long) | ((htf < 0) & ~is_long)
    vwap_ok = np.where(is_long, close > vwap, close < vwap)
    ext = np.where(is_long, (close - ema15) / atr, (ema15 - close) / atr)

    hm = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()
    phase = np.where(hm < _OPEN_MIN, "open",
                     np.where(hm >= _LATE_MIN, "late", "core"))

    status = np.select(
        [~trending, htf == 0, ~htf_ok, aligned < GRADE_MIN, ~vwap_ok,
         ext >= EXTENDED, fresh | (np.abs(ext) <= AT_EMA)],
        ["AVOID", "AVOID", "AVOID", "AVOID", "WAIT", "WAIT", "ENTER"],
        default="WAIT")
    # market gate, then time-of-day gate
    status = np.where((status == "ENTER") & mkt_against, "WAIT", status)
    status = np.where((status == "ENTER") & (phase == "open"), "WAIT", status)
    status = np.where((status == "ENTER") & (phase == "late"), "AVOID", status)

    grade = np.select([aligned >= 6, aligned == 5, aligned == 4],
                      ["A+", "A", "B"], default="C")

    outcome = _first_touch(ind["high"].to_numpy(float), ind["low"].to_numpy(float),
                           close, _ATR_STOP * atr, is_long, day.to_numpy())

    res = pd.DataFrame({
        "symbol": symbol, "timestamp": ts, "session": day.astype(str),
        "status": status, "grade": grade, "aligned": aligned,
        "is_long": is_long, "outcome": outcome,
        "trending": trending, "htf_ok": htf_ok, "htf": htf,
        "vwap_ok": vwap_ok, "ext": ext, "fresh": fresh,
        "mkt_against": mkt_against, "mkt_aligned": mkt_aligned,
        "rs_ok": rs_ok, "phase": phase, "atr_pct": atr / close * 100,
    })
    return res[ok].reset_index(drop=True)


# -------------------------------------------------------------- statistics
def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _clustered(df: pd.DataFrame, base: float) -> tuple[float, float, int]:
    """Per-session hit rate: mean, t-stat vs `base`, number of sessions."""
    per = df.groupby("session")["win"].mean()
    per = per[df.groupby("session").size() >= 5]
    k = len(per)
    if k < 3:
        return (float(per.mean()) if k else 0.0, 0.0, k)
    sd = per.std(ddof=1)
    t = 0.0 if sd == 0 else (per.mean() - base) / (sd / math.sqrt(k))
    return (float(per.mean()), float(t), k)


def _bucket(df: pd.DataFrame, base: float) -> dict:
    n = len(df)
    if n == 0:
        return {"n": 0}
    wins = int(df["win"].sum())
    lo, hi = _wilson(wins, n)
    mean, t, k = _clustered(df, base)
    res = df["outcome"].value_counts()
    return {
        "n": n, "hit": wins / n, "ci": [lo, hi],
        "timeout_pct": float((df["outcome"] == -1).mean()),
        "excl_timeout": float((res.get(1, 0) / (res.get(1, 0) + res.get(0, 0)))
                              if (res.get(1, 0) + res.get(0, 0)) else 0.0),
        "expectancy_R": 2 * (wins / n) - 1,
        "session_hit": mean, "t_stat": t, "sessions": k,
        "edge_pp": (wins / n - base) * 100,
    }


# ------------------------------------------------------------------ report
def _fmt(label: str, b: dict, base: float) -> str:
    if not b.get("n"):
        return f"  {label:<22} —"
    return (f"  {label:<24} n={b['n']:>7,}  hit {b['hit']*100:5.1f}%  "
            f"[{b['ci'][0]*100:4.1f}–{b['ci'][1]*100:4.1f}]  "
            f"{b['edge_pp']:+5.2f}pp  E {b['expectancy_R']:+5.2f}R  "
            f"t={b['t_stat']:+5.2f}({b['sessions']:>2}s)  "
            f"unres {b['timeout_pct']*100:4.1f}%  resolved {b['excl_timeout']*100:5.1f}%")


def evaluate(symbols: list[str], days: int) -> dict:
    idx = _index_frame(days)
    bars = _fetch_universe(symbols, days)
    if not bars:
        raise SystemExit("No data downloaded — check the network / yfinance.")

    frames = []
    for i, (s, df) in enumerate(bars.items(), 1):
        try:
            r = replay_symbol(s, df, idx)
            if r is not None and not r.empty:
                frames.append(r)
        except Exception as e:
            log.warning("replay failed for %s: %s", s, e)
        if i % 40 == 0:
            log.info("replayed %d/%d symbols", i, len(bars))
    all_df = pd.concat(frames, ignore_index=True)
    # timeout counts as a loss in the headline rate
    all_df["win"] = (all_df["outcome"] == 1).astype(int)

    base = float(all_df["win"].mean())
    out: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbols": all_df["symbol"].nunique(),
        "sessions": all_df["session"].nunique(),
        "from": str(all_df["timestamp"].min())[:16],
        "to": str(all_df["timestamp"].max())[:16],
        "bars": len(all_df),
        "base_rate": base,
        "by_status": {}, "by_grade": {}, "ablation": {}, "by_phase": {},
    }

    for st in ["ENTER", "WAIT", "AVOID"]:
        out["by_status"][st] = _bucket(all_df[all_df["status"] == st], base)
    ent = all_df[all_df["status"] == "ENTER"]
    for g in ["A+", "A", "B"]:
        out["by_grade"][g] = _bucket(ent[ent["grade"] == g], base)
    for ph in ["open", "core", "late"]:
        out["by_phase"][ph] = _bucket(all_df[all_df["phase"] == ph], base)

    # --- ablation: which gate is actually carrying the signal? -----------
    core = (all_df["vwap_ok"] & (all_df["ext"] < EXTENDED)
            & (all_df["fresh"] | (all_df["ext"].abs() <= AT_EMA)))
    gates = {
        "full engine": (all_df["trending"] & (all_df["htf"] != 0) & all_df["htf_ok"]
                        & (all_df["aligned"] >= GRADE_MIN) & core
                        & ~all_df["mkt_against"] & (all_df["phase"] == "core")),
        "-ADX regime": ((all_df["htf"] != 0) & all_df["htf_ok"]
                        & (all_df["aligned"] >= GRADE_MIN) & core
                        & ~all_df["mkt_against"] & (all_df["phase"] == "core")),
        "-15m gate": (all_df["trending"] & (all_df["aligned"] >= GRADE_MIN) & core
                      & ~all_df["mkt_against"] & (all_df["phase"] == "core")),
        "-confluence min": (all_df["trending"] & (all_df["htf"] != 0) & all_df["htf_ok"]
                            & core & ~all_df["mkt_against"] & (all_df["phase"] == "core")),
        "-market gate": (all_df["trending"] & (all_df["htf"] != 0) & all_df["htf_ok"]
                         & (all_df["aligned"] >= GRADE_MIN) & core
                         & (all_df["phase"] == "core")),
        "-time gate": (all_df["trending"] & (all_df["htf"] != 0) & all_df["htf_ok"]
                       & (all_df["aligned"] >= GRADE_MIN) & core & ~all_df["mkt_against"]),
        "-timing (vwap/ext/fresh)": (all_df["trending"] & (all_df["htf"] != 0)
                                     & all_df["htf_ok"] & (all_df["aligned"] >= GRADE_MIN)
                                     & ~all_df["mkt_against"] & (all_df["phase"] == "core")),
        "ONLY confluence>=4": (all_df["aligned"] >= GRADE_MIN),
        "ONLY 15m aligned": ((all_df["htf"] != 0) & all_df["htf_ok"]),
        "ONLY ADX>=20": all_df["trending"],
    }
    for k, mask in gates.items():
        out["ablation"][k] = _bucket(all_df[mask], base)

    out["direction_split"] = {
        "long": _bucket(ent[ent["is_long"]], base),
        "short": _bucket(ent[~ent["is_long"]], base),
    }
    return out


def render(r: dict) -> str:
    base = r["base_rate"]
    L = []
    L.append("=" * 96)
    L.append("ATLAS SIGNAL EVALUATION — does the selectivity beat the base rate?")
    L.append("=" * 96)
    L.append(f"{r['symbols']} symbols · {r['sessions']} sessions · "
             f"{r['from']} → {r['to']} · {r['bars']:,} decision bars")
    L.append(f"Risk {_ATR_STOP} ATR · label = +1R before −1R, same session · "
             f"timeout counts as a loss · closed bars only, no lookahead")
    L.append("")
    L.append(f"BASE RATE (every bar, direction = EMA9 vs EMA15): {base*100:.2f}%")
    L.append("  A coin-flip engine scores this. Everything below must beat it.")
    L.append("")
    hdr = ("n         hit     [95% CI]     edge    expect   clustered-t   "
           "unresolved  hit-if-resolved")
    L.append("BY STATUS" + " " * 16 + hdr)
    for k, v in r["by_status"].items():
        L.append(_fmt(k, v, base))
    L.append("")
    L.append("ENTER BY GRADE")
    for k, v in r["by_grade"].items():
        L.append(_fmt(k, v, base))
    L.append("")
    L.append("ENTER BY DIRECTION")
    for k, v in r["direction_split"].items():
        L.append(_fmt(k, v, base))
    L.append("")
    L.append("BY SESSION PHASE (all bars)")
    for k, v in r["by_phase"].items():
        L.append(_fmt(k, v, base))
    L.append("")
    L.append("GATE ABLATION — remove one gate at a time; does the hit rate fall?")
    for k, v in r["ablation"].items():
        L.append(_fmt(k, v, base))
    L.append("")
    L.append("HOW TO READ THIS")
    L.append("  · 'hit' is the raw rate; the 95% interval assumes independent bars,")
    L.append("    which they are NOT — overlapping windows make it far too narrow.")
    L.append("  · 't' clusters by session (one trading day = one observation).")
    L.append("    |t| < 2 means the edge is indistinguishable from noise.")
    L.append("  · 'unresolved' = neither +1R nor −1R was touched before the close;")
    L.append("    these count as losses in 'hit'. Late-session bars are mechanically")
    L.append("    unresolved (no time left), so compare them on 'hit-if-resolved' —")
    L.append("    a low 'hit' late in the day is mostly this artifact, not an edge.")
    L.append("  · expectancy is in R, at a 1:1 target/stop: E = 2·hit − 1.")
    L.append("    It ignores brokerage, slippage, option spread and theta — all of")
    L.append("    which are negative. A positive E here is necessary, not sufficient.")
    L.append("=" * 96)
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="Atlas signal-quality evaluation")
    ap.add_argument("--days", type=int, default=60,
                    help="calendar days of 5m history (yfinance caps at 60)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N universe symbols (quick pass)")
    ap.add_argument("--json", type=str, default="",
                    help="also write the raw numbers to this path")
    a = ap.parse_args()

    syms = FNO_UNIVERSE[:a.limit] if a.limit else list(FNO_UNIVERSE)
    r = evaluate(syms, a.days)
    print(render(r))
    if a.json:
        from pathlib import Path
        Path(a.json).write_text(json.dumps(r, indent=2, default=float))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
