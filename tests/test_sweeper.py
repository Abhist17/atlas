"""The sweeper decides what the track record is a sample OF.

If it journals only what you clicked, the hit rate measures your attention.
If it double-writes, the sample inflates. If one bad symbol kills the pass, the
record silently stops growing. These pin all three.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from engine import sweeper


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    from storage import live_journal
    monkeypatch.setattr(live_journal, "LIVE_SIGNALS_FILE", tmp_path / "j.parquet")


def _screen(*symbols_and_conv):
    return pd.DataFrame({"symbol": [s for s, _ in symbols_and_conv],
                         "conviction": [c for _, c in symbols_and_conv]})


def _sig(symbol, status="ENTER", bar="2026-08-19 10:15:00+05:30"):
    return {"symbol": symbol, "ok": True, "bar_time": bar, "status": status,
            "bias": "LONG", "entry": 100.0, "stop": 98.0, "grade": "A",
            "targets": [{"px": 102.0}, {"px": 104.0}, {"px": 106.0}],
            "market": {"name": "NIFTY", "bias": 1}}


def _wire(monkeypatch, screen, signals):
    monkeypatch.setattr("web.data_service.get_screen", lambda i=5: screen)
    monkeypatch.setattr("engine.alpha_signal.compute_signal",
                        lambda s, interval=5: signals[s])


# ---------------------------------------------------------------- ranking
def test_candidates_rank_by_absolute_conviction(monkeypatch):
    """A strong short is as tradable as a strong long — rank on magnitude."""
    monkeypatch.setattr("web.data_service.get_screen",
                        lambda i=5: _screen(("AAA", 0.2), ("BBB", -0.9), ("CCC", 0.5)))
    assert sweeper.candidates(limit=2) == ["BBB", "CCC"]


def test_an_empty_screen_yields_no_candidates(monkeypatch):
    monkeypatch.setattr("web.data_service.get_screen", lambda i=5: pd.DataFrame())
    assert sweeper.candidates() == []


# ---------------------------------------------------------------- journaling
def test_only_enter_calls_are_journaled(monkeypatch):
    """WAIT and AVOID dominate every pass; journaling them buries the scorable rows."""
    _wire(monkeypatch, _screen(("AAA", 0.9), ("BBB", 0.8), ("CCC", 0.7)),
          {"AAA": _sig("AAA", "ENTER"), "BBB": _sig("BBB", "WAIT"),
           "CCC": _sig("CCC", "AVOID")})
    res = sweeper.sweep(limit=3, force=True)
    assert res["enters"] == 1 and res["journaled"] == 1

    from storage.live_journal import load
    assert list(load()["symbol"]) == ["AAA"]


def test_the_same_bar_is_not_journaled_twice_across_passes(monkeypatch):
    """Sweeps run every 5 min on 5-min bars — re-seeing a bar is the normal case."""
    _wire(monkeypatch, _screen(("AAA", 0.9)), {"AAA": _sig("AAA")})
    assert sweeper.sweep(limit=1, force=True)["journaled"] == 1
    assert sweeper.sweep(limit=1, force=True)["journaled"] == 0

    from storage.live_journal import load
    assert len(load()) == 1


def test_a_new_bar_is_journaled(monkeypatch):
    _wire(monkeypatch, _screen(("AAA", 0.9)), {"AAA": _sig("AAA")})
    sweeper.sweep(limit=1, force=True)
    _wire(monkeypatch, _screen(("AAA", 0.9)),
          {"AAA": _sig("AAA", bar="2026-08-19 10:20:00+05:30")})
    assert sweeper.sweep(limit=1, force=True)["journaled"] == 1


# ---------------------------------------------------------------- robustness
def test_one_broken_symbol_does_not_end_the_pass(monkeypatch):
    def compute(sym, interval=5):
        if sym == "BBB":
            raise RuntimeError("yfinance exploded")
        return _sig(sym)
    monkeypatch.setattr("web.data_service.get_screen",
                        lambda i=5: _screen(("AAA", 0.9), ("BBB", 0.8), ("CCC", 0.7)))
    monkeypatch.setattr("engine.alpha_signal.compute_signal", compute)
    res = sweeper.sweep(limit=3, force=True)
    assert res["errors"] == 1
    assert res["journaled"] == 2        # AAA and CCC still recorded


def test_a_failed_signal_counts_as_an_error_not_a_call(monkeypatch):
    _wire(monkeypatch, _screen(("AAA", 0.9)),
          {"AAA": {"symbol": "AAA", "ok": False, "error": "No bar data."}})
    res = sweeper.sweep(limit=1, force=True)
    assert res["errors"] == 1 and res["journaled"] == 0


# ---------------------------------------------------------------- market hours
def test_a_closed_market_is_skipped(monkeypatch):
    monkeypatch.setattr(sweeper, "market_open", lambda now=None: False)
    assert sweeper.sweep()["skipped"] == "market closed"


def test_force_overrides_the_market_hours_guard(monkeypatch):
    monkeypatch.setattr(sweeper, "market_open", lambda now=None: False)
    monkeypatch.setattr("web.data_service.get_screen", lambda i=5: pd.DataFrame())
    assert sweeper.sweep(force=True)["skipped"] == "no candidates"


@pytest.mark.parametrize("when,expected", [
    (datetime(2026, 8, 19, 10, 0), True),    # Wednesday, mid-session
    (datetime(2026, 8, 19, 8, 0), False),    # before the open
    (datetime(2026, 8, 19, 16, 0), False),   # after the close
    (datetime(2026, 8, 22, 10, 0), False),   # Saturday
    (datetime(2026, 8, 23, 10, 0), False),   # Sunday
])
def test_market_hours(when, expected):
    assert sweeper.market_open(when) is expected
