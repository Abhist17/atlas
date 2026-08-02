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
HORIZON = 12        # bars (1 hour on 5m) for the fixed-horizon R metric
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
            # as_unit: parquet round-trips as ms while a fresh fetch is s, and
            # merge_asof refuses to join mismatched datetime resolutions
            "timestamp": pd.to_datetime(df["timestamp"]).dt.as_unit("ns"),
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


def _horizon_outcome(high, low, close, entry, risk, is_long, session_id,
                     horizon: int = HORIZON):
    """R-multiple of the trade over a fixed forward horizon. No timeout hole.

    Simulates the actual trade for `horizon` bars: stop first -> -1R, target
    first -> +1R, neither -> mark to market at the horizon bar. Every bar
    therefore gets a number, which is what first-touch labelling cannot do.

    Also returns an `evaluable` mask: bars with a full horizon left in the
    session. Restricting to those makes time-of-day buckets comparable —
    otherwise late bars are penalised simply for running out of clock.
    """
    n_all = len(entry)
    r = np.full(n_all, np.nan)
    ok = np.zeros(n_all, bool)
    for _, idx in pd.Series(np.arange(n_all)).groupby(session_id):
        i = idx.to_numpy()
        n = len(i)
        if n < 2:
            continue
        h, l, c = high[i], low[i], close[i]
        e, rk, lg = entry[i], risk[i], is_long[i]
        ar = np.arange(n)
        tgt = np.where(lg, e + rk, e - rk)[:, None]
        stp = np.where(lg, e - rk, e + rk)[:, None]
        window = (ar[None, :] > ar[:, None]) & (ar[None, :] <= (ar + horizon)[:, None])
        lgc = lg[:, None]
        hit_t = np.where(lgc, h[None, :] >= tgt, l[None, :] <= tgt) & window
        hit_s = np.where(lgc, l[None, :] <= stp, h[None, :] >= stp) & window
        big = n + 1
        ft = np.where(hit_t.any(1), hit_t.argmax(1), big)
        fs = np.where(hit_s.any(1), hit_s.argmax(1), big)
        end = np.minimum(ar + horizon, n - 1)
        d = np.where(lg, 1.0, -1.0)
        out = d * (c[end] - e) / np.maximum(rk, 1e-9)      # mark to market
        out = np.where((ft < big) & (ft < fs), 1.0, out)   # target first
        out = np.where((fs < big) & (fs <= ft), -1.0, out)  # stop first (ties -> loss)
        r[i] = out
        ok[i] = (ar + horizon) <= (n - 1)
    return r, ok


# ------------------------------------------------------------------ replay
def replay_symbol(symbol: str, df: pd.DataFrame, idx: dict) -> pd.DataFrame | None:
    """One symbol -> a frame of per-bar decisions, gates and outcomes."""
    if df is None or len(df) < 120:
        return None
    ind = add_indicators(df.reset_index(drop=True))
    ts = pd.to_datetime(ind["timestamp"]).dt.as_unit("ns")
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

    high = ind["high"].to_numpy(float)
    low = ind["low"].to_numpy(float)
    risk = _ATR_STOP * atr
    outcome = _first_touch(high, low, close, risk, is_long, day.to_numpy())
    r_h, evaluable = _horizon_outcome(high, low, close, close, risk, is_long,
                                      day.to_numpy())
    # Same bar scored BOTH ways, regardless of the direction the engine picked.
    # Direction and regime are otherwise the same variable (is_long is defined
    # by ema9 vs ema15), so they cannot be told apart in the main table: a short
    # bias in the sample looks identical to momentum failing in uptrends.
    yes = np.ones(len(close), bool)
    r_long, _ = _horizon_outcome(high, low, close, close, risk, yes, day.to_numpy())
    r_short, _ = _horizon_outcome(high, low, close, close, risk, ~yes, day.to_numpy())

    res = pd.DataFrame({
        "symbol": symbol, "timestamp": ts, "session": day.astype(str),
        "status": status, "grade": grade, "aligned": aligned,
        "is_long": is_long, "outcome": outcome,
        "r_h": r_h, "evaluable": evaluable,
        "r_long": r_long, "r_short": r_short,
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


def _clustered(df: pd.DataFrame, col: str, base: float) -> tuple[float, float, int]:
    """Per-session mean of `col`: mean, t-stat vs `base`, number of sessions.

    One trading session = one observation. Bars inside a session overlap
    heavily, so this is the only standard error worth quoting.
    """
    sub = df[["session", col]].dropna()
    if sub.empty:
        return (0.0, 0.0, 0)
    sizes = sub.groupby("session").size()
    per = sub.groupby("session")[col].mean()[sizes >= 5]
    k = len(per)
    if k < 3:
        return (float(per.mean()) if k else 0.0, 0.0, k)
    sd = per.std(ddof=1)
    t = 0.0 if sd == 0 else (per.mean() - base) / (sd / math.sqrt(k))
    return (float(per.mean()), float(t), k)


def _bucket(df: pd.DataFrame, base: float, base_r: float) -> dict:
    n = len(df)
    if n == 0:
        return {"n": 0}
    wins = int(df["win"].sum())
    lo, hi = _wilson(wins, n)
    _, t, k = _clustered(df, "win", base)
    res = df["outcome"].value_counts()

    # fixed-horizon R, on bars that actually have a full horizon left
    ev = df[df["evaluable"]]
    r_mean = float(ev["r_h"].mean()) if len(ev) else 0.0
    _, t_r, k_r = _clustered(ev, "r_h", base_r)
    return {
        "n": n, "hit": wins / n, "ci": [lo, hi],
        "timeout_pct": float((df["outcome"] == -1).mean()),
        "excl_timeout": float((res.get(1, 0) / (res.get(1, 0) + res.get(0, 0)))
                              if (res.get(1, 0) + res.get(0, 0)) else 0.0),
        "expectancy_R": 2 * (wins / n) - 1,
        "t_stat": t, "sessions": k,
        "edge_pp": (wins / n - base) * 100,
        "n_eval": len(ev), "r_h": r_mean, "r_edge": r_mean - base_r,
        "t_r": t_r, "sessions_r": k_r,
    }


def _gate_masks(d: pd.DataFrame) -> dict[str, pd.Series]:
    """Each entry = the engine with exactly one gate removed (or one kept)."""
    core = (d["vwap_ok"] & (d["ext"] < EXTENDED)
            & (d["fresh"] | (d["ext"].abs() <= AT_EMA)))
    adx = d["trending"]
    htf = (d["htf"] != 0) & d["htf_ok"]
    conf = d["aligned"] >= GRADE_MIN
    mkt = ~d["mkt_against"]
    tod = d["phase"] == "core"
    return {
        "full engine": adx & htf & conf & core & mkt & tod,
        "-ADX regime": htf & conf & core & mkt & tod,
        "-15m gate": adx & conf & core & mkt & tod,
        "-confluence min": adx & htf & core & mkt & tod,
        "-market gate": adx & htf & conf & core & tod,
        "-time gate": adx & htf & conf & core & mkt,
        "-timing (vwap/ext/fresh)": adx & htf & conf & mkt & tod,
        "ONLY confluence>=4": conf,
        "ONLY 15m aligned": htf,
        "ONLY ADX>=20": adx,
        "everything (no gates)": pd.Series(True, index=d.index),
    }


# ------------------------------------------------------------------ report
def _fmt(label: str, b: dict, base: float) -> str:
    if not b.get("n"):
        return f"  {label:<22} —"
    return (f"  {label:<24} n={b['n']:>7,}  hit {b['hit']*100:5.1f}% "
            f"({b['edge_pp']:+5.2f}pp) t={b['t_stat']:+5.2f}  │  "
            f"R{b['r_h']:+6.3f} ({b['r_edge']:+6.3f}) t={b['t_r']:+5.2f}  "
            f"n={b['n_eval']:>7,}  unres {b['timeout_pct']*100:4.1f}%")


def _table(title: str, rows: dict, base: float) -> list[str]:
    L = [title]
    for k, v in rows.items():
        L.append(_fmt(k, v, base))
    return L


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
    base_r = float(all_df.loc[all_df["evaluable"], "r_h"].mean())
    out: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbols": all_df["symbol"].nunique(),
        "sessions": all_df["session"].nunique(),
        "from": str(all_df["timestamp"].min())[:16],
        "to": str(all_df["timestamp"].max())[:16],
        "bars": len(all_df),
        "horizon_bars": HORIZON,
        "base_rate": base, "base_r": base_r,
        "by_status": {}, "by_grade": {}, "by_phase": {},
        "ablation": {}, "ablation_long": {}, "ablation_short": {},
    }

    def B(d):
        return _bucket(d, base, base_r)

    for st in ["ENTER", "WAIT", "AVOID"]:
        out["by_status"][st] = B(all_df[all_df["status"] == st])
    ent = all_df[all_df["status"] == "ENTER"]
    for g in ["A+", "A", "B"]:
        out["by_grade"][g] = B(ent[ent["grade"] == g])
    for ph in ["open", "core", "late"]:
        out["by_phase"][ph] = B(all_df[all_df["phase"] == ph])

    # --- ablation: which gate is actually carrying the signal? -----------
    masks = _gate_masks(all_df)
    # Within-direction baselines: "do the gates beat shorting everything?" is a
    # different question from "do they beat the overall base rate", and only the
    # first one tells you whether the gates are doing work.
    lg, sh_m = all_df["is_long"], ~all_df["is_long"]
    b_long = float(all_df.loc[lg, "win"].mean())
    b_short = float(all_df.loc[sh_m, "win"].mean())
    br_long = float(all_df.loc[lg & all_df["evaluable"], "r_h"].mean())
    br_short = float(all_df.loc[sh_m & all_df["evaluable"], "r_h"].mean())
    out["base_long"], out["base_short"] = b_long, b_short
    out["base_r_long"], out["base_r_short"] = br_long, br_short
    for k, m in masks.items():
        out["ablation"][k] = B(all_df[m])
        out["ablation_long"][k] = _bucket(all_df[m & lg], b_long, br_long)
        out["ablation_short"][k] = _bucket(all_df[m & sh_m], b_short, br_short)

    out["direction_split"] = {
        "ENTER long": B(ent[ent["is_long"]]),
        "ENTER short": B(ent[~ent["is_long"]]),
        "all bars long": B(all_df[all_df["is_long"]]),
        "all bars short": B(all_df[~all_df["is_long"]]),
    }
    # --- regime x direction: structure, or just a short-biased sample? ------
    ev = all_df[all_df["evaluable"]]
    matrix = {}
    for reg, sub in (("uptrend (ema9>=ema15)", ev[ev["is_long"]]),
                     ("downtrend (ema9<ema15)", ev[~ev["is_long"]])):
        row = {"n": len(sub)}
        for side in ("r_long", "r_short"):
            _, t, k = _clustered(sub, side, 0.0)
            row[side] = float(sub[side].mean()) if len(sub) else 0.0
            row[f"t_{side}"] = t
            row[f"sessions_{side}"] = k
        matrix[reg] = row
    out["regime_matrix"] = matrix

    # is the short result a few outlier sessions, or broad?
    sh = ent[~ent["is_long"] & ent["evaluable"]]
    per = sh.groupby("session")["r_h"].mean()
    out["short_session_spread"] = {
        "sessions": int(len(per)),
        "positive_sessions": int((per > 0).sum()),
        "median": float(per.median()) if len(per) else 0.0,
        "best": float(per.max()) if len(per) else 0.0,
        "worst": float(per.min()) if len(per) else 0.0,
        "mean_ex_best3": float(per.sort_values()[:-3].mean()) if len(per) > 3 else 0.0,
    }
    return out


def render(r: dict) -> str:
    base, base_r = r["base_rate"], r["base_r"]
    W = 118
    L = ["=" * W,
         "ATLAS SIGNAL EVALUATION — does the selectivity beat the base rate?",
         "=" * W,
         f"{r['symbols']} symbols · {r['sessions']} sessions · "
         f"{r['from']} → {r['to']} · {r['bars']:,} decision bars",
         f"Risk {_ATR_STOP} ATR · closed bars only, no lookahead",
         "",
         "Two independent metrics, because each has a blind spot:",
         f"  hit = +1R before −1R within the session (unresolved counts as a loss)"
         f"   → base {base*100:.2f}%",
         f"  R   = mean R-multiple over a fixed {r['horizon_bars']}-bar horizon, "
         f"only on bars with a full horizon left"
         f"   → base {base_r:+.4f}R",
         "The R metric is the honest one for time-of-day questions: it cannot",
         "penalise a bar for merely running out of clock.",
         ""]
    hdr = (" " * 26 + "n          hit    (edge)   t     │      R      (edge)    t"
           "      n(eval)   unresolved")
    L.append(hdr)
    L += _table("BY STATUS", r["by_status"], base)
    L.append("")
    L += _table("ENTER BY GRADE", r["by_grade"], base)
    L.append("")
    L += _table("BY DIRECTION", r["direction_split"], base)
    L.append("")
    L += _table("BY SESSION PHASE (all bars)", r["by_phase"], base)
    L.append("")
    L += _table("GATE ABLATION — remove one gate; does quality fall?", r["ablation"], base)
    L.append("")
    L += _table(f"GATE ABLATION — LONG ONLY (vs long base {r['base_long']*100:.2f}% / "
                f"{r['base_r_long']:+.4f}R)", r["ablation_long"], base)
    L.append("")
    L += _table(f"GATE ABLATION — SHORT ONLY (vs short base {r['base_short']*100:.2f}% / "
                f"{r['base_r_short']:+.4f}R)", r["ablation_short"], base)
    L.append("")
    L.append("REGIME × DIRECTION — every bar scored BOTH ways (mean R, "
             "clustered t vs zero)")
    L.append(" " * 26 + "go LONG                 go SHORT")
    for reg, v in r["regime_matrix"].items():
        L.append(f"  {reg:<24} R{v['r_long']:+.4f} t={v['t_r_long']:+5.2f}   "
                 f"R{v['r_short']:+.4f} t={v['t_r_short']:+5.2f}   n={v['n']:,}")
    L.append("  If shorting wins in BOTH rows, the engine has no momentum/reversion")
    L.append("  structure to exploit — the sample simply favoured shorts, and that")
    L.append("  will not survive into a different market.")
    L.append("")
    s = r["short_session_spread"]
    L.append("IS THE SHORT RESULT BROAD OR A FEW OUTLIER DAYS?")
    L.append(f"  ENTER-short sessions: {s['sessions']}  ·  positive: "
             f"{s['positive_sessions']} ({s['positive_sessions']/max(s['sessions'],1)*100:.0f}%)"
             f"  ·  median {s['median']:+.3f}R")
    L.append(f"  best session {s['best']:+.3f}R  ·  worst {s['worst']:+.3f}R  ·  "
             f"mean excluding the 3 best sessions: {s['mean_ex_best3']:+.3f}R")
    L.append("")
    L.append("HOW TO READ THIS")
    L.append("  · 't' clusters by session (one trading day = one observation), which")
    L.append("    is the only honest standard error here — bars overlap heavily, so")
    L.append("    raw n and its confidence interval are far too optimistic.")
    L.append("    |t| < 2 means indistinguishable from noise.")
    L.append("  · 'unresolved' matters only for the hit column; the R column excludes")
    L.append("    bars without a full horizon, so it is immune to that artifact.")
    L.append("  · Both metrics ignore brokerage, slippage, option spread and theta —")
    L.append("    all negative. A positive number here is necessary, not sufficient.")
    L.append("=" * W)
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
