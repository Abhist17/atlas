"""The journal is the record the win rate is computed from — it must not lie.

Duplicates inflate the sample with the same decision counted twice; a write
failure must never take the signal endpoint down with it.
"""
from __future__ import annotations

import pandas as pd
import pytest

from storage import live_journal


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(live_journal, "LIVE_SIGNALS_FILE",
                        tmp_path / "live_signals.parquet")


def _sig(symbol="RELIANCE", bar_time="2026-08-14 10:15:00+05:30", status="ENTER"):
    return {
        "symbol": symbol, "ok": True, "bar_time": bar_time, "status": status,
        "bias": "LONG", "option": "CALL", "grade": "A", "confluence": "5/6",
        "confidence": 0.62, "win_prob": 0.55, "entry": 100.0, "stop": 98.0,
        "atr": 1.5, "risk_pts": 2.0, "signal_px": 99.8, "ltp": 100.0,
        "drift_atr": 0.1, "extension": 0.4, "rel_strength": 0.3,
        "headline": "Grade A up setup",
        "targets": [{"px": 102.0, "rr": 1}, {"px": 104.0, "rr": 2},
                    {"px": 106.0, "rr": 3}],
        "market": {"name": "NIFTY", "bias": 1, "chg": 0.4},
    }


def test_a_signal_is_recorded_flat():
    assert live_journal.record(_sig()) is True
    df = live_journal.load()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["symbol"] == "RELIANCE"
    assert row["tp1"] == 102.0 and row["tp3"] == 106.0
    assert row["market"] == "NIFTY"


def test_same_bar_is_not_journaled_twice():
    """Re-opening the page mid-bar must not double-count one decision."""
    assert live_journal.record(_sig()) is True
    assert live_journal.record(_sig()) is False
    assert len(live_journal.load()) == 1


def test_a_new_bar_for_the_same_symbol_is_a_new_row():
    live_journal.record(_sig())
    live_journal.record(_sig(bar_time="2026-08-14 10:20:00+05:30"))
    assert len(live_journal.load()) == 2


def test_different_symbols_on_the_same_bar_both_land():
    live_journal.record(_sig(symbol="RELIANCE"))
    live_journal.record(_sig(symbol="INFY"))
    assert set(live_journal.load()["symbol"]) == {"RELIANCE", "INFY"}


def test_wait_and_avoid_calls_are_journaled_too():
    """The WAIT and AVOID calls are the product's actual value — keep them."""
    live_journal.record(_sig(status="WAIT"))
    live_journal.record(_sig(bar_time="2026-08-14 10:20:00+05:30", status="AVOID"))
    assert sorted(live_journal.load()["status"]) == ["AVOID", "WAIT"]


def test_a_failed_signal_is_not_journaled():
    assert live_journal.record({"symbol": "X", "ok": False}) is False
    assert live_journal.load().empty


def test_a_broken_disk_does_not_break_the_endpoint(monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    assert live_journal.record(_sig()) is False      # returns, does not raise


def test_a_corrupt_file_reads_as_empty(tmp_path):
    live_journal.LIVE_SIGNALS_FILE.write_bytes(b"not parquet")
    assert live_journal.load().empty
