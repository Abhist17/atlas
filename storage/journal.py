"""Persistence layer — the honest trade journal.

Everything the system sees and does is logged to disk so you can review your
REAL win rate (not the remembered one). Two logs:
  - signals: every setup the scanner flagged (whether traded or not)
  - trades:  every closed paper (or live) trade with net P&L

Stored as Parquet under data_store/ so they survive restarts and load fast.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from config.settings import DATA_STORE
from utils.logger import get_logger

log = get_logger("storage.journal")

SIGNALS_FILE = DATA_STORE / "signals_log.parquet"
TRADES_FILE = DATA_STORE / "trades_log.parquet"


def _append(path, rows: list[dict]) -> None:
    if not rows:
        return
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_parquet(path)
        new = pd.concat([old, new], ignore_index=True)
    new.to_parquet(path, index=False)


def log_signals(snapshot: pd.DataFrame, when: datetime | None = None) -> None:
    """Record a scanner snapshot (one row per flagged setup)."""
    if snapshot is None or snapshot.empty:
        return
    when = when or datetime.now()
    rows = snapshot.assign(logged_at=when.isoformat()).to_dict("records")
    _append(SIGNALS_FILE, rows)
    log.info("Journaled %d signals", len(rows))


def log_trade(trade: dict) -> None:
    """Record one closed trade."""
    _append(TRADES_FILE, [trade])


def load_signals() -> pd.DataFrame:
    return pd.read_parquet(SIGNALS_FILE) if SIGNALS_FILE.exists() else pd.DataFrame()


def load_trades() -> pd.DataFrame:
    return pd.read_parquet(TRADES_FILE) if TRADES_FILE.exists() else pd.DataFrame()


def stats() -> dict:
    """Real performance summary from the trade journal."""
    t = load_trades()
    if t.empty or "pnl" not in t:
        return {"trades": 0}
    wins = t[t["pnl"] > 0]
    return {
        "trades": len(t),
        "win_rate": round(len(wins) / len(t) * 100, 2),
        "total_pnl": round(t["pnl"].sum(), 2),
        "avg_pnl": round(t["pnl"].mean(), 2),
        "best": round(t["pnl"].max(), 2),
        "worst": round(t["pnl"].min(), 2),
    }
