"""VWAP mean-reversion.

Buy dips: when price stretches below VWAP by a meaningful multiple of ATR but
the short-term trend (EMA9 > EMA21) is still up — a pullback in an uptrend that
tends to revert toward VWAP. Complements momentum/breakout signals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.strategies.base import Strategy


class VWAPReversion(Strategy):
    name = "vwap_reversion"

    def __init__(self, min_atr_stretch: float = 1.0, max_atr_stretch: float = 3.0) -> None:
        self.min_atr_stretch = min_atr_stretch   # how far below VWAP to trigger
        self.max_atr_stretch = max_atr_stretch    # beyond this, trend may be broken

    def signal(self, df: pd.DataFrame) -> pd.Series:
        atr = df["atr"].replace(0, np.nan)
        stretch = (df["vwap"] - df["close"]) / atr   # ATRs below VWAP
        in_uptrend = df["ema9"] > df["ema21"]
        trigger = (
            in_uptrend
            & (stretch >= self.min_atr_stretch)
            & (stretch <= self.max_atr_stretch)
        )
        # Conviction grows with stretch within the valid band
        conviction = (
            (stretch - self.min_atr_stretch)
            / (self.max_atr_stretch - self.min_atr_stretch)
        ).clip(0, 1)
        return pd.Series(np.where(trigger, conviction, 0.0), index=df.index).fillna(0.0)
