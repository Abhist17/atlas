"""Cached data services for the web API — screener + option chain."""
from __future__ import annotations

import time

import pandas as pd

from data.option_chain import get_chain, get_expiries, lot_size
from engine.nse_screener import screen_nse

_CACHE: dict[int, tuple[float, pd.DataFrame]] = {}
_CHAIN_CACHE: dict[tuple, tuple[float, dict]] = {}
_TTL = 120       # screener cache seconds
_CHAIN_TTL = 20  # chain cache seconds


def get_screen(interval: int = 5) -> pd.DataFrame:
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


def expiries(symbol: str) -> list[str]:
    return get_expiries(symbol)


def chain(symbol: str, expiry: str | None = None) -> dict:
    key = (symbol.upper(), expiry)
    now = time.time()
    hit = _CHAIN_CACHE.get(key)
    if hit and now - hit[0] < _CHAIN_TTL:
        return hit[1]
    data = get_chain(symbol, expiry)
    data["lot_size"] = lot_size(symbol)
    _CHAIN_CACHE[key] = (now, data)
    return data
