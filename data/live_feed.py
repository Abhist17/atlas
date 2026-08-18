"""Live price feed.

Free real-time data for NSE is scarce: yfinance intraday bars lag ~15 min, and
broker live feeds (Dhan Data API) are paid. NSE's own public quote API, however,
is near-real-time and free from an Indian residential IP. So we use a hybrid:

  - Candles / indicator history: yfinance 5-min OHLC (slightly delayed, fine for EMAs).
  - Live last price (LTP): NSE quote API, patched onto the latest bar so the price
    and the freshest candle match your broker.

If NSE is unreachable (e.g. datacenter IP), we fall back to the yfinance close and
flag the data as delayed, so the UI can tell the user honestly.
"""
from __future__ import annotations

import time

import pandas as pd
import requests

from data.yf_client import yfc
from utils.logger import get_logger

log = get_logger("data.live_feed")

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/get-quotes/equity",
}

_IST = "Asia/Kolkata"

_session: requests.Session | None = None
_session_ts = 0.0
_ltp_cache: dict[str, tuple[float, float]] = {}   # symbol -> (ts, ltp)
_LTP_TTL = 5   # seconds — keep it snappy but avoid hammering NSE


def _nse_session() -> requests.Session:
    global _session, _session_ts
    if _session is None or time.time() - _session_ts > 3600:
        s = requests.Session()
        s.headers.update(_HEADERS)
        try:
            s.get("https://www.nseindia.com", timeout=6)
            s.get("https://www.nseindia.com/get-quotes/equity?symbol=RELIANCE", timeout=6)
        except requests.RequestException as e:
            log.warning("NSE warmup failed: %s", e)
        _session, _session_ts = s, time.time()
    return _session


def live_ltp(symbol: str) -> float | None:
    """Near-real-time last traded price from NSE (None if unavailable)."""
    symbol = symbol.upper()
    hit = _ltp_cache.get(symbol)
    if hit and time.time() - hit[0] < _LTP_TTL:
        return hit[1]
    url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
    try:
        r = _nse_session().get(url, timeout=6)
        if r.status_code == 200 and len(r.text) > 100:
            price = r.json().get("priceInfo", {}).get("lastPrice")
            if price:
                ltp = float(price)
                _ltp_cache[symbol] = (time.time(), ltp)
                return ltp
    except (requests.RequestException, ValueError) as e:
        log.debug("NSE LTP failed for %s: %s", symbol, e)
    return None


def bar_is_forming(last_ts, interval: int) -> bool:
    """Is the bar stamped `last_ts` still being built right now?

    A bar stamped T on an `interval`-minute series closes at T + interval. Any
    consumer that makes a decision on a bar which has not closed yet is reading a
    number that can still change — the signal repaints.
    """
    ts = pd.Timestamp(last_ts)
    if ts.tz is not None:
        now = pd.Timestamp.now(tz=ts.tz)
    else:   # NSE bars are IST; compare in the same wall clock
        now = pd.Timestamp.now(tz=_IST).tz_localize(None)
    return now < ts + pd.Timedelta(minutes=interval)


def get_bars(symbol: str, days: int = 5, interval: int = 5) -> dict:
    """OHLC bars for charts + indicators, with the forming bar patched to live LTP.

    Returns {ok, symbol, bars: DataFrame, ltp, is_live, forming, source}.

    `forming` says whether the final row is an unclosed bar. Only that bar is
    ever patched with the live price: overwriting a *closed* candle rewrites
    history, and every indicator downstream then disagrees with the backtest.
    Consumers that gate on indicators must decide on closed bars only.
    """
    symbol = symbol.upper()
    df = yfc.intraday(symbol, days=days, interval=interval)
    if df is None or df.empty:
        return {"ok": False, "symbol": symbol, "error": "No bar data."}

    df = df.copy().reset_index(drop=True)
    forming = bar_is_forming(df["timestamp"].iloc[-1], interval)
    ltp = live_ltp(symbol)
    is_live = ltp is not None
    if not is_live:
        ltp = float(df["close"].iloc[-1])
    elif forming:
        # patch the forming candle so the chart's freshest bar tracks the live price
        i = df.index[-1]
        df.at[i, "close"] = ltp
        df.at[i, "high"] = max(float(df.at[i, "high"]), ltp)
        df.at[i, "low"] = min(float(df.at[i, "low"]), ltp)

    return {"ok": True, "symbol": symbol, "bars": df, "ltp": round(float(ltp), 2),
            "is_live": is_live, "forming": bool(forming),
            "source": "NSE live" if is_live else "yfinance (delayed)"}
