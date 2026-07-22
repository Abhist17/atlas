"""Free market-data client via yfinance (NSE equities).

Returns the same OHLCV schema as data.dhan_client so the screener/backtest are
source-agnostic. Dhan stays reserved for order execution; data comes from here
until (or unless) a Dhan Data API subscription is added.

Note: yfinance intraday history is limited (~60 days for >=2m bars, ~7 days for
1m). Fine for intraday screening.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import CACHE_DIR
from utils.logger import get_logger

log = get_logger("data.yf_client")

_OHLC_COLS = ["timestamp", "open", "high", "low", "close", "volume"]
_INTERVAL_MAP = {1: "1m", 5: "5m", 15: "15m", 30: "30m", 60: "60m"}


def _to_yf_symbol(symbol: str) -> str:
    """RELIANCE -> RELIANCE.NS (NSE suffix). Pass-through if already suffixed."""
    return symbol if symbol.endswith(".NS") else f"{symbol.upper()}.NS"


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance output into our OHLCV schema."""
    if df is None or df.empty:
        return pd.DataFrame(columns=_OHLC_COLS)
    out = df.copy()
    # Single-ticker downloads come back with a MultiIndex column level
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out = out.reset_index()
    ts_col = "Datetime" if "Datetime" in out.columns else "Date"
    out = out.rename(
        columns={
            ts_col: "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    return out[_OHLC_COLS].dropna().reset_index(drop=True)


class YFClient:
    """yfinance-backed intraday/daily fetcher with Parquet caching."""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def intraday(
        self, symbol: str, days: int = 5, interval: int = 5, use_cache: bool = True
    ) -> pd.DataFrame:
        yf_interval = _INTERVAL_MAP.get(interval, "5m")
        cache = CACHE_DIR / f"yf_{symbol}_{days}d_{yf_interval}.parquet"
        if use_cache and cache.exists():
            return pd.read_parquet(cache)

        raw = yf.download(
            _to_yf_symbol(symbol),
            period=f"{days}d",
            interval=yf_interval,
            progress=False,
            auto_adjust=True,
        )
        df = _normalise(raw)
        if use_cache and not df.empty:
            df.to_parquet(cache, index=False)
        return df

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def daily(self, symbol: str, period: str = "6mo") -> pd.DataFrame:
        raw = yf.download(
            _to_yf_symbol(symbol),
            period=period,
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        return _normalise(raw)


yfc = YFClient()
