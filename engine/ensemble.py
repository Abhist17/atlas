"""Signal ensemble — combines multiple strategies into one composite long-
conviction score per bar.

The edge in quant trading comes from blending weakly-correlated signals: when
several agree, conviction (and position size) rises; when they disagree, we
stay out. Each strategy contributes a [0,1] score; the ensemble takes a
weighted average and gates it with a minimum-agreement threshold.
"""
from __future__ import annotations

import pandas as pd

from engine.strategies.base import Strategy
from engine.strategies.momentum import Momentum
from engine.strategies.orb import OpeningRangeBreakout
from engine.strategies.volume_surge import VolumeSurge
from engine.strategies.vwap_reversion import VWAPReversion


class Ensemble:
    def __init__(
        self,
        strategies: list[Strategy] | None = None,
        weights: dict[str, float] | None = None,
        threshold: float = 0.25,
        min_agree: int = 1,
    ) -> None:
        self.strategies = strategies or [
            OpeningRangeBreakout(),
            VWAPReversion(),
            Momentum(),
            VolumeSurge(),
        ]
        # Equal weight unless overridden
        self.weights = weights or {s.name: 1.0 for s in self.strategies}
        self.threshold = threshold
        self.min_agree = min_agree   # require at least this many active signals

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a frame with each strategy's signal, the composite score, and
        the count of active signals, indexed like df.
        """
        out = pd.DataFrame(index=df.index)
        total_w = 0.0
        weighted = pd.Series(0.0, index=df.index)
        active = pd.Series(0, index=df.index)
        for strat in self.strategies:
            sig = strat.signal(df).fillna(0.0)
            out[strat.name] = sig
            w = self.weights.get(strat.name, 1.0)
            weighted += w * sig
            total_w += w
            active += (sig > 0).astype(int)
        out["composite"] = weighted / total_w if total_w else weighted
        out["active"] = active
        return out

    def candidates(self, df: pd.DataFrame) -> pd.Series:
        """Boolean per-bar: does this symbol qualify as a long candidate?"""
        s = self.score(df)
        return (s["composite"] >= self.threshold) & (s["active"] >= self.min_agree)
