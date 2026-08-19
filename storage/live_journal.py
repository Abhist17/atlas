"""Live signal journal — what the app actually told you, when.

`storage/journal.py` records the *scanner's* snapshots and closed paper trades.
This module records the thing the product is actually judged on: every
ENTER / WAIT / AVOID call the dashboard served, with the entry, stop and targets
that were on screen at the time. Without it the calls evaporate the moment the
page refreshes, and the honest win rate the README promises can never be
computed from anything but memory.

One row per (symbol, bar_time). Decisions are made on closed bars, so a symbol
cannot legitimately produce two different calls for the same bar — re-opening
the page mid-bar must not inflate the log with duplicates.

Stored as Parquet under data_store/ alongside the rest of the journal.
"""
from __future__ import annotations

import threading
from datetime import datetime

import pandas as pd

from config.settings import DATA_STORE
from utils.logger import get_logger

log = get_logger("storage.live_journal")

LIVE_SIGNALS_FILE = DATA_STORE / "live_signals.parquet"

# Columns we keep. Deliberately flat (no nested dicts) so the file stays a
# clean table that pandas, DuckDB or Excel can all read.
COLUMNS = [
    "logged_at", "symbol", "bar_time", "status", "bias", "option", "grade",
    "confluence", "confidence", "win_prob", "entry", "stop", "tp1", "tp2", "tp3",
    "atr", "risk_pts", "signal_px", "ltp", "drift_atr", "extension",
    "market", "market_bias", "rel_strength", "headline",
]

_lock = threading.Lock()


def _flatten(sig: dict, when: datetime) -> dict:
    tps = [t.get("px") for t in (sig.get("targets") or [])] + [None, None, None]
    mkt = sig.get("market") or {}
    return {
        "logged_at": when.isoformat(timespec="seconds"),
        "symbol": sig.get("symbol"),
        "bar_time": str(sig.get("bar_time")),
        "status": sig.get("status"),
        "bias": sig.get("bias"),
        "option": sig.get("option"),
        "grade": sig.get("grade"),
        "confluence": sig.get("confluence"),
        "confidence": sig.get("confidence"),
        "win_prob": sig.get("win_prob"),
        "entry": sig.get("entry"),
        "stop": sig.get("stop"),
        "tp1": tps[0], "tp2": tps[1], "tp3": tps[2],
        "atr": sig.get("atr"),
        "risk_pts": sig.get("risk_pts"),
        "signal_px": sig.get("signal_px"),
        "ltp": sig.get("ltp"),
        "drift_atr": sig.get("drift_atr"),
        "extension": sig.get("extension"),
        "market": mkt.get("name"),
        "market_bias": mkt.get("bias"),
        "rel_strength": sig.get("rel_strength"),
        "headline": sig.get("headline"),
    }


def load() -> pd.DataFrame:
    """Every journaled decision, oldest first. Empty frame if nothing logged."""
    if not LIVE_SIGNALS_FILE.exists():
        return pd.DataFrame(columns=COLUMNS)
    try:
        return pd.read_parquet(LIVE_SIGNALS_FILE)
    except (OSError, ValueError) as e:          # corrupt/partial file
        log.warning("Could not read %s: %s", LIVE_SIGNALS_FILE.name, e)
        return pd.DataFrame(columns=COLUMNS)


def record(sig: dict, when: datetime | None = None) -> bool:
    """Journal one signal. Returns True if it was written, False if deduped.

    Never raises: journaling is a side effect of serving a signal, and a broken
    disk must not take the signal endpoint down with it.
    """
    if not sig or not sig.get("ok") or not sig.get("symbol") or not sig.get("bar_time"):
        return False
    row = _flatten(sig, when or datetime.now())
    try:
        with _lock:
            df = load()
            if not df.empty and (
                (df["symbol"] == row["symbol"]) & (df["bar_time"] == row["bar_time"])
            ).any():
                return False
            out = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            out = out.reindex(columns=COLUMNS)
            LIVE_SIGNALS_FILE.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(LIVE_SIGNALS_FILE, index=False)
        return True
    except Exception as e:                      # noqa: BLE001 - never break the API
        log.warning("Failed to journal %s: %s", row.get("symbol"), e)
        return False
