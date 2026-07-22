"""Cached data services for the web API — avoids rescanning on every request."""
from __future__ import annotations

import time

import pandas as pd

from engine.nse_screener import screen_nse

_CACHE: dict[int, tuple[float, pd.DataFrame]] = {}
_TTL = 120  # seconds


def get_screen(interval: int = 5) -> pd.DataFrame:
    """Return the latest NSE screen, cached per interval for _TTL seconds."""
    now = time.time()
    hit = _CACHE.get(interval)
    if hit and now - hit[0] < _TTL:
        return hit[1]
    df = screen_nse(interval=interval)
    _CACHE[interval] = (now, df)
    return df


def refresh(interval: int = 5) -> pd.DataFrame:
    _CACHE.pop(interval, None)
    return get_screen(interval)
