"""Option-chain data for the trading dashboard.

Primary source is NSE's free option-chain API (works from an Indian
residential IP; blocked from datacenter/cloud IPs). When NSE returns nothing
we fall back to a **theoretical** Black-Scholes chain built around the live
spot from yfinance, so the UI always has strikes to trade on paper.
"""
from __future__ import annotations

import math
import time
from datetime import date, datetime

import requests

from data.yf_client import yfc
from utils.logger import get_logger

log = get_logger("data.option_chain")

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/option-chain",
}
_INDICES = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}

# Common NSE lot sizes (fallback = 1); extend as needed.
LOT_SIZES = {
    "NIFTY": 75, "BANKNIFTY": 35, "FINNIFTY": 65, "MIDCPNIFTY": 140,
    "RELIANCE": 500, "TCS": 175, "INFY": 400, "HDFCBANK": 550, "ICICIBANK": 700,
    "SBIN": 750, "ONGC": 3850, "TATASTEEL": 5500, "ITC": 1600, "AXISBANK": 625,
}

_session: requests.Session | None = None
_session_ts = 0.0


def _nse_session() -> requests.Session:
    """A warmed NSE session (cookies from the homepage), refreshed hourly."""
    global _session, _session_ts
    if _session is None or time.time() - _session_ts > 3600:
        s = requests.Session()
        s.headers.update(_HEADERS)
        try:
            s.get("https://www.nseindia.com", timeout=8)
            s.get("https://www.nseindia.com/option-chain", timeout=8)
        except requests.RequestException as e:
            log.warning("NSE warmup failed: %s", e)
        _session, _session_ts = s, time.time()
    return _session


def _nse_raw(symbol: str) -> dict | None:
    kind = "indices" if symbol in _INDICES else "equities"
    url = f"https://www.nseindia.com/api/option-chain-{kind}?symbol={symbol}"
    try:
        r = _nse_session().get(url, timeout=8)
        if r.status_code == 200 and len(r.text) > 100:
            return r.json()
    except (requests.RequestException, ValueError) as e:
        log.warning("NSE chain fetch failed for %s: %s", symbol, e)
    return None


def lot_size(symbol: str) -> int:
    return LOT_SIZES.get(symbol.upper(), 1)


# ------------------------------------------------------------------ public
def get_expiries(symbol: str) -> list[str]:
    raw = _nse_raw(symbol.upper())
    if raw:
        exp = raw.get("records", {}).get("expiryDates", [])
        if exp:
            return exp
    return _synthetic_expiries()


def get_chain(symbol: str, expiry: str | None = None) -> dict:
    """Return {symbol, spot, expiry, source, rows:[{strike, ce{...}, pe{...}}]}."""
    symbol = symbol.upper()
    raw = _nse_raw(symbol)
    if raw:
        chain = _parse_nse(raw, expiry)
        if chain["rows"]:
            return chain
    return _synthetic_chain(symbol, expiry)


# ------------------------------------------------------------------ NSE parse
def _parse_nse(raw: dict, expiry: str | None) -> dict:
    rec = raw.get("records", {})
    spot = rec.get("underlyingValue") or 0
    expiries = rec.get("expiryDates", [])
    expiry = expiry if expiry in expiries else (expiries[0] if expiries else None)
    rows = []
    for item in rec.get("data", []):
        if item.get("expiryDate") != expiry:
            continue
        ce, pe = item.get("CE", {}), item.get("PE", {})
        rows.append({
            "strike": item["strikePrice"],
            "ce": _leg(ce), "pe": _leg(pe),
        })
    rows.sort(key=lambda r: r["strike"])
    return {"symbol": rec.get("underlying") or "", "spot": round(spot, 2),
            "expiry": expiry, "source": "NSE", "rows": rows}


def _leg(d: dict) -> dict:
    return {
        "ltp": round(d.get("lastPrice", 0) or 0, 2),
        "oi": int(d.get("openInterest", 0) or 0),
        "chg_oi": int(d.get("changeinOpenInterest", 0) or 0),
        "iv": round(d.get("impliedVolatility", 0) or 0, 1),
        "volume": int(d.get("totalTradedVolume", 0) or 0),
    }


# ------------------------------------------------------------ Black-Scholes
def _bs(spot: float, strike: float, t: float, vol: float, call: bool) -> float:
    if t <= 0 or vol <= 0:
        return max(0.0, (spot - strike) if call else (strike - spot))
    d1 = (math.log(spot / strike) + 0.5 * vol * vol * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    nd = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    if call:
        return spot * nd(d1) - strike * nd(d2)
    return strike * nd(-d2) - spot * nd(-d1)


def _synthetic_expiries() -> list[str]:
    """Next few Thursdays, NSE date format e.g. '24-Jul-2026'."""
    out, d = [], date.today()
    while len(out) < 4:
        d = date.fromordinal(d.toordinal() + 1)
        if d.weekday() == 3:  # Thursday
            out.append(d.strftime("%d-%b-%Y"))
    return out


def _synthetic_chain(symbol: str, expiry: str | None) -> dict:
    df = yfc.batch_intraday([f"{symbol}.NS"], days=5, interval=5).get(f"{symbol}.NS")
    if df is None or df.empty:
        return {"symbol": symbol, "spot": 0, "expiry": expiry,
                "source": "unavailable", "rows": []}
    spot = float(df["close"].iloc[-1])
    # crude annualised vol from 5-min returns
    rets = df["close"].pct_change().dropna()
    vol = float(rets.std()) * math.sqrt(75 * 252) if len(rets) > 5 else 0.3
    vol = min(max(vol, 0.12), 1.5)

    expiries = _synthetic_expiries()
    expiry = expiry if expiry in expiries else expiries[0]
    exp_d = datetime.strptime(expiry, "%d-%b-%Y").date()
    t = max((exp_d - date.today()).days, 0) / 365 or 1 / 365

    step = _strike_step(spot)
    atm = round(spot / step) * step
    strikes = [atm + i * step for i in range(-8, 9)]
    rows = []
    for k in strikes:
        rows.append({
            "strike": round(k, 1),
            "ce": {"ltp": round(_bs(spot, k, t, vol, True), 2), "oi": 0,
                   "chg_oi": 0, "iv": round(vol * 100, 1), "volume": 0},
            "pe": {"ltp": round(_bs(spot, k, t, vol, False), 2), "oi": 0,
                   "chg_oi": 0, "iv": round(vol * 100, 1), "volume": 0},
        })
    return {"symbol": symbol, "spot": round(spot, 2), "expiry": expiry,
            "source": "theoretical", "rows": rows}


def _strike_step(spot: float) -> float:
    if spot < 100:
        return 2.5
    if spot < 500:
        return 5
    if spot < 2000:
        return 10
    if spot < 10000:
        return 50
    return 100
