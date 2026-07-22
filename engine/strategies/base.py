"""Base class for pluggable trading signals.

Each strategy consumes a per-symbol OHLCV+indicator frame (from
engine.indicators.add_indicators, timestamp-indexed) and returns a per-bar
long-conviction score in [0, 1]:
    0.0 = no signal / flat
    1.0 = strongest long conviction

Scores are backward-looking (bar t uses only data up to t) so they are safe to
precompute once and replay in the backtester without look-ahead.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    name: str = "base"

    @abstractmethod
    def signal(self, df: pd.DataFrame) -> pd.Series:
        """Return a [0,1] long-conviction score per bar, indexed like df."""

    @staticmethod
    def _session(df: pd.DataFrame) -> pd.Series:
        """Calendar-day grouping key for intraday session logic."""
        return pd.to_datetime(df.index.to_series()).dt.date
