"""The scorer must be pessimistic, forward-only, and intraday-honest.

These are the invariants that decide whether the hit rate on the dashboard is
worth anything. Each one, broken, makes the number look better than the trades
you could actually have taken.
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine.signal_review import score_signal, summarise

IST = "Asia/Kolkata"


def _bars(rows, day="2026-08-14", start="09:20"):
    """rows = [(high, low, close), ...] on a 5-minute series."""
    ts = pd.date_range(f"{day} {start}", periods=len(rows), freq="5min", tz=IST)
    return pd.DataFrame({
        "timestamp": ts,
        "open": [r[2] for r in rows],
        "high": [r[0] for r in rows],
        "low": [r[1] for r in rows],
        "close": [r[2] for r in rows],
        "volume": [50_000.0] * len(rows),
    })


LONG = {"bar_time": "2026-08-14 09:15:00+05:30", "bias": "LONG",
        "entry": 100.0, "stop": 98.0, "tp1": 102.0, "tp2": 104.0, "tp3": 106.0}
SHORT = {"bar_time": "2026-08-14 09:15:00+05:30", "bias": "SHORT",
         "entry": 100.0, "stop": 102.0, "tp1": 98.0, "tp2": 96.0, "tp3": 94.0}


def test_target_hit_pays_its_r():
    res = score_signal(LONG, _bars([(101, 99.5, 100.5), (102.5, 100, 102.2)]))
    assert res["outcome"] == "TP1"
    assert res["r"] == 1.0
    assert res["bars_held"] == 2


def test_stop_hit_is_minus_one_r():
    res = score_signal(LONG, _bars([(101, 99.5, 100.5), (100.5, 97.8, 98.1)]))
    assert res["outcome"] == "STOP"
    assert res["r"] == -1.0


def test_stop_wins_a_same_bar_tie():
    """One bar touching both is ambiguous — intraday OHLC has no ordering.

    Assuming the target came first is exactly how a backtest ends up describing
    a strategy nobody could have traded.
    """
    res = score_signal(LONG, _bars([(106.5, 97.0, 100.0)]))
    assert res["outcome"] == "STOP"
    assert res["r"] == -1.0


def test_furthest_target_in_a_bar_wins():
    res = score_signal(LONG, _bars([(106.5, 99.5, 106.0)]))
    assert res["outcome"] == "TP3"
    assert res["r"] == 3.0


def test_short_side_is_mirrored():
    assert score_signal(SHORT, _bars([(100.5, 97.5, 98.0)]))["outcome"] == "TP1"
    assert score_signal(SHORT, _bars([(102.5, 100, 102.2)]))["outcome"] == "STOP"


def test_bars_at_or_before_the_decision_bar_are_ignored():
    """Scoring the decision bar itself reads the data the decision came from."""
    stale = _bars([(106.0, 97.0, 105.0)], start="09:15")   # == bar_time
    stale["timestamp"] = pd.date_range("2026-08-14 09:10", periods=1,
                                       freq="5min", tz=IST)
    assert score_signal(LONG, stale)["outcome"] == "OPEN"


def test_still_running_is_open_not_a_winner():
    res = score_signal(LONG, _bars([(101, 99.5, 100.5), (101.5, 100, 101.0)]))
    assert res["outcome"] == "OPEN"
    assert res["r"] is None
    assert res["mfe_r"] == pytest.approx(0.75)


def test_square_off_closes_the_trade_at_that_bar():
    res = score_signal(LONG, _bars([(101, 99.5, 101.0)] * 3, start="15:05"))
    assert res["outcome"] == "TIMEOUT"
    assert res["exit_time"].startswith("2026-08-14 15:15")
    assert res["r"] == 0.5          # (101 - 100) / 2.0


def test_mfe_and_mae_track_the_worst_and_best_excursions():
    res = score_signal(LONG, _bars([(101.0, 99.0, 100.0), (103.0, 99.4, 102.5)]))
    assert res["mae_r"] == pytest.approx(0.5)     # 1.0 pt against / 2.0 risk
    assert res["mfe_r"] == pytest.approx(1.5)


def test_zero_risk_signal_cannot_be_scored():
    bad = {**LONG, "stop": 100.0}
    assert score_signal(bad, _bars([(106, 94, 100)]))["outcome"] == "OPEN"


def test_summary_counts_only_closed_calls():
    scored = pd.DataFrame([
        {"outcome": "TP1", "r": 1.0}, {"outcome": "TP2", "r": 2.0},
        {"outcome": "STOP", "r": -1.0}, {"outcome": "STOP", "r": -1.0},
        {"outcome": "OPEN", "r": None},
    ])
    s = summarise(scored)
    assert s["scored"] == 4
    assert s["open"] == 1
    assert s["win_rate"] == 50.0
    assert s["total_r"] == 1.0
    assert s["avg_r"] == 0.25
    assert s["stopped"] == 2


def test_summary_of_nothing_is_not_a_crash():
    assert summarise(pd.DataFrame())["scored"] == 0
