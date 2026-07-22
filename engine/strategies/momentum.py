"""Trend momentum.

Long when short-term trend is up (EMA9 > EMA21), price holds above VWAP, and
RSI shows strength without being exhausted. This is the original Atlas signal,
now one voice in the ensemble rather than the sole decision-maker.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.strategies.base import Strategy


class Momentum(Strategy):
    name = "momentum"

    def __init__(self, rsi_lower: float = 55.0, rsi_upper: float = 72.0) -> None:
        self.rsi_lower = rsi_lower
        self.rsi_upper = rsi_upper

    def signal(self, df: pd.DataFrame) -> pd.Series:
        trigger = (
            (df["ema9"] > df["ema21"])
            & (df["close"] > df["vwap"])
            & (df["rsi"] >= self.rsi_lower)
            & (df["rsi"] <= self.rsi_upper)
        )
        # Conviction peaks mid-band, fades near the exhaustion edge
        mid = (self.rsi_lower + self.rsi_upper) / 2
        span = (self.rsi_upper - self.rsi_lower) / 2
        conviction = (1 - (df["rsi"] - mid).abs() / span).clip(0, 1)
        return pd.Series(np.where(trigger, conviction, 0.0), index=df.index).fillna(0.0)
