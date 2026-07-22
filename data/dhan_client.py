"""Dhan data client — wraps DhanContext/HistoricalData to fetch OHLCV bars
and return clean pandas DataFrames. Intraday bars are cached as Parquet.

Credentials come from config (env). If unconfigured, fetch calls raise a
clear error so paper-mode dev can still import everything.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import CACHE_DIR, config
from utils.logger import get_logger

log = get_logger("data.dhan_client")

EXCHANGE_SEGMENT = "NSE_EQ"
INSTRUMENT_TYPE = "EQUITY"

# Columns Dhan returns in historical responses
_OHLC_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


class DhanClient:
    """Thin wrapper around dhanhq for market data."""

    def __init__(self) -> None:
        self._hist = None  # lazy — only build when a fetch is requested

    def _historical(self):
        if self._hist is None:
            if not config.dhan.is_configured:
                raise RuntimeError(
                    "Dhan credentials missing. Set DHAN_CLIENT_ID and "
                    "DHAN_ACCESS_TOKEN in .env (see .env.example)."
                )
            from dhanhq import DhanContext, HistoricalData

            ctx = DhanContext(config.dhan.client_id, config.dhan.access_token)
            self._hist = HistoricalData(ctx)
        return self._hist

    @staticmethod
    def _parse(resp: dict) -> pd.DataFrame:
        """Normalise a Dhan historical response into an OHLCV DataFrame."""
        if not resp or resp.get("status") != "success":
            msg = resp.get("remarks") if resp else "empty response"
            raise RuntimeError(f"Dhan data error: {msg}")
        d = resp["data"]
        df = pd.DataFrame(
            {
                "timestamp": d.get("timestamp") or d.get("start_Time"),
                "open": d["open"],
                "high": d["high"],
                "low": d["low"],
                "close": d["close"],
                "volume": d["volume"],
            }
        )
        # Dhan timestamps are epoch seconds (IST-based); convert to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        return df[_OHLC_COLS]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=8),
        retry=retry_if_not_exception_type(RuntimeError),
    )
    def intraday(
        self,
        security_id: str,
        from_date: str,
        to_date: str,
        interval: int = 5,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch intraday minute bars (interval in minutes: 1/5/15/…).

        Dates are 'YYYY-MM-DD' strings. Cached per (security, dates, interval).
        """
        cache = (
            CACHE_DIR
            / f"intraday_{security_id}_{from_date}_{to_date}_{interval}m.parquet"
        )
        if use_cache and cache.exists():
            return pd.read_parquet(cache)

        resp = self._historical().intraday_minute_data(
            security_id=str(security_id),
            exchange_segment=EXCHANGE_SEGMENT,
            instrument_type=INSTRUMENT_TYPE,
            from_date=from_date,
            to_date=to_date,
            interval=interval,
        )
        df = self._parse(resp)
        if use_cache and not df.empty:
            df.to_parquet(cache, index=False)
        return df

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=8),
        retry=retry_if_not_exception_type(RuntimeError),
    )
    def daily(
        self, security_id: str, from_date: str, to_date: str
    ) -> pd.DataFrame:
        """Fetch daily OHLCV bars — used for liquidity/ATR screening context."""
        resp = self._historical().historical_daily_data(
            security_id=str(security_id),
            exchange_segment=EXCHANGE_SEGMENT,
            instrument_type=INSTRUMENT_TYPE,
            from_date=from_date,
            to_date=to_date,
        )
        return self._parse(resp)

    def recent_intraday(
        self, security_id: str, days: int = 5, interval: int = 5
    ) -> pd.DataFrame:
        """Convenience: last `days` calendar days of intraday bars."""
        to_d = date.today()
        from_d = to_d - timedelta(days=days)
        return self.intraday(
            security_id, from_d.isoformat(), to_d.isoformat(), interval
        )


# Shared instance
dhan = DhanClient()
