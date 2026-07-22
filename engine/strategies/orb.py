"""Opening Range Breakout (ORB).

Mark the high of the first `opening_bars` of each session. Go long when price
breaks above that high with above-average volume. A staple intraday edge on
liquid names. Only active after the opening range is formed and before an
optional cutoff (avoid late-day breakouts).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.strategies.base import Strategy


class OpeningRangeBreakout(Strategy):
    name = "orb"

    def __init__(self, opening_bars: int = 6, volume_mult: float = 1.2,
                 max_bar: int = 60) -> None:
        self.opening_bars = opening_bars       # 6 x 5min = first 30 min
        self.volume_mult = volume_mult
        self.max_bar = max_bar                 # no new breakouts after this bar #

    def signal(self, df: pd.DataFrame) -> pd.Series:
        day = self._session(df)
        # Bar index within each session (0-based)
        bar_no = df.groupby(day).cumcount()
        # Rolling opening-range high: max of first `opening_bars` closes per day
        or_high = (
            df["high"].where(bar_no < self.opening_bars)
            .groupby(day).transform("max")
        )
        breakout = (
            (bar_no >= self.opening_bars)
            & (bar_no <= self.max_bar)
            & (df["close"] > or_high)
            & (df["volume"] > self.volume_mult * df["avg_volume"])
        )
        # Conviction scales with how decisively volume confirms
        vol_ratio = (df["volume"] / df["avg_volume"]).clip(upper=3) / 3
        return pd.Series(np.where(breakout, vol_ratio, 0.0), index=df.index)
