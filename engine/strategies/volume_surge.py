"""Volume-surge continuation.

A sudden volume spike on an up-bar signals institutional interest / fresh
demand. Long when volume >> its recent average and the bar closed green above
VWAP. Fast-reacting; pairs well with slower trend signals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.strategies.base import Strategy


class VolumeSurge(Strategy):
    name = "volume_surge"

    def __init__(self, surge_mult: float = 2.0) -> None:
        self.surge_mult = surge_mult   # volume vs rolling average

    def signal(self, df: pd.DataFrame) -> pd.Series:
        up_bar = df["close"] > df["open"]
        surge = df["volume"] >= self.surge_mult * df["avg_volume"]
        trigger = up_bar & surge & (df["close"] > df["vwap"])
        conviction = (df["volume"] / df["avg_volume"] / (2 * self.surge_mult)).clip(0, 1)
        return pd.Series(np.where(trigger, conviction, 0.0), index=df.index).fillna(0.0)
