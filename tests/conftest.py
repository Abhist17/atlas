"""Shared fixtures: synthetic NSE-shaped intraday bars, no network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

IST = "Asia/Kolkata"
SESSION_BARS = 75          # 09:15 → 15:30 on a 5-minute series


def make_session(day: str, bars: int = SESSION_BARS, start: float = 100.0,
                 drift: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """One trading session of 5-minute OHLCV bars with a mild upward drift."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(f"{day} 09:15", periods=bars, freq="5min", tz=IST)
    close = start + np.cumsum(drift + rng.normal(0, 0.15, bars))
    open_ = np.concatenate([[start], close[:-1]])
    noise = np.abs(rng.normal(0, 0.12, bars))
    return pd.DataFrame({
        "timestamp": ts,
        "open": open_,
        "high": np.maximum(open_, close) + noise,
        "low": np.minimum(open_, close) - noise,
        "close": close,
        "volume": rng.integers(40_000, 90_000, bars).astype(float),
    })


@pytest.fixture
def bars() -> pd.DataFrame:
    """Three consecutive sessions of clean bars."""
    frames, start = [], 100.0
    for i, day in enumerate(["2026-08-12", "2026-08-13", "2026-08-14"]):
        f = make_session(day, start=start, seed=i)
        start = float(f["close"].iloc[-1])
        frames.append(f)
    return pd.concat(frames, ignore_index=True)
