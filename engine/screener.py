"""Intraday screener — scans the NSE equity universe, applies liquidity +
momentum filters from ScreenerConfig, and returns ranked long candidates.

Each candidate is evaluated on its most recent bar's indicators. The output is
a DataFrame ordered by a momentum score (best first).
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from config.settings import ScreenerConfig, config
from data.instruments import load_nse_equities
from data.yf_client import yfc
from engine.indicators import add_indicators
from utils.logger import get_logger

log = get_logger("engine.screener")

# A fetcher maps a trading symbol -> OHLCV DataFrame. Injectable for tests.
Fetcher = Callable[[str], pd.DataFrame]


def _default_fetcher(symbol: str) -> pd.DataFrame:
    return yfc.intraday(symbol, days=5, interval=5)


def _passes(row: pd.Series, cfg: ScreenerConfig) -> bool:
    """Apply the screener rules to a symbol's latest indicator row."""
    price = row["close"]
    checks = [
        cfg.min_price <= price <= cfg.max_price,
        row["avg_volume"] >= cfg.min_avg_volume,
        row["atr_pct"] >= cfg.min_atr_pct,
        cfg.rsi_lower <= row["rsi"] <= cfg.rsi_upper,
        (row["close"] > row["vwap"]) if cfg.vwap_side == "above"
        else (row["close"] < row["vwap"]),
        row["ema9"] > row["ema21"],  # short-term uptrend
    ]
    return all(bool(c) for c in checks)


def _score(row: pd.Series) -> float:
    """Momentum score for ranking — higher is stronger.
    Combines RSI strength, distance above VWAP, and volatility (ATR%).
    """
    vwap_gap = (row["close"] - row["vwap"]) / row["vwap"] * 100
    return round(row["rsi"] * 0.5 + vwap_gap * 2.0 + row["atr_pct"], 3)


def screen(
    fetcher: Fetcher | None = None,
    cfg: ScreenerConfig | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run the screener over the universe. Returns ranked candidates:
    columns = [security_id, symbol, close, rsi, atr_pct, vwap, score].
    """
    cfg = cfg or config.screener
    fetcher = fetcher or _default_fetcher
    universe = load_nse_equities().head(limit or cfg.max_universe)

    rows = []
    for rec in universe.itertuples(index=False):
        try:
            df = fetcher(rec.symbol)
            if df is None or len(df) < 21:  # need enough bars for indicators
                continue
            last = add_indicators(df).iloc[-1]
            if _passes(last, cfg):
                rows.append(
                    {
                        "security_id": rec.security_id,
                        "symbol": rec.symbol,
                        "close": round(last["close"], 2),
                        "rsi": round(last["rsi"], 1),
                        "atr_pct": round(last["atr_pct"], 2),
                        "vwap": round(last["vwap"], 2),
                        "score": _score(last),
                    }
                )
        except Exception as e:  # one bad symbol shouldn't kill the scan
            log.warning("skip %s: %s", rec.symbol, e)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("score", ascending=False).reset_index(drop=True)
    log.info("Screened %d symbols → %d candidates", len(universe), len(result))
    return result
