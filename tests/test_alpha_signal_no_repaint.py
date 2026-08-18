"""The signal must not repaint: an unfinished bar cannot change the decision.

`compute_signal` gates on indicators (EMA cross, MACD, RSI, ADX, the six votes).
Those keep moving until a bar closes, so a decision taken on the forming bar can
flip — or vanish — while the user is looking at it. These tests feed the engine
the *same* closed history twice, once with a wild in-progress bar glued on, and
require every gated field to come out identical.
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine import alpha_signal
from tests.conftest import IST, make_session

# fields that are decided by the gates — none may move with the forming bar
GATED = ["status", "bias", "option", "grade", "confluence", "signal_px",
         "atr", "stop", "extension", "headline", "trigger", "levels", "factors"]


def _history() -> pd.DataFrame:
    frames, start = [], 100.0
    for i, day in enumerate(["2026-08-12", "2026-08-13"]):
        f = make_session(day, start=start, drift=0.08, seed=i)
        start = float(f["close"].iloc[-1])
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Pin every external input so only the bars vary."""
    monkeypatch.setattr(alpha_signal, "_htf_bias", lambda s: 1)
    monkeypatch.setattr(alpha_signal, "get_context",
                        lambda s: {"name": "NIFTY", "bias": 1, "chg": 0.10})
    monkeypatch.setattr(alpha_signal, "win_probability", lambda f: None)


def _signal(monkeypatch, bars, ltp, forming):
    monkeypatch.setattr(alpha_signal, "get_bars", lambda *a, **k: {
        "ok": True, "symbol": "TEST", "bars": bars.reset_index(drop=True),
        "ltp": round(float(ltp), 2), "is_live": True, "forming": forming,
        "source": "test"})
    return alpha_signal.compute_signal("TEST")


def _forming_bar(prev: pd.Series, px: float) -> pd.DataFrame:
    """One in-progress bar at a wildly different price."""
    return pd.DataFrame([{
        "timestamp": pd.Timestamp(prev["timestamp"]) + pd.Timedelta(minutes=5),
        "open": float(prev["close"]), "high": max(float(prev["close"]), px),
        "low": min(float(prev["close"]), px), "close": px, "volume": 5_000.0,
    }])


@pytest.mark.parametrize("shock", [1.06, 0.94])
def test_forming_bar_cannot_move_the_decision(monkeypatch, shock):
    hist = _history()
    closed = _signal(monkeypatch, hist, hist["close"].iloc[-1], forming=False)
    assert closed["ok"], closed.get("error")

    px = float(hist["close"].iloc[-1]) * shock
    with_forming = pd.concat([hist, _forming_bar(hist.iloc[-1], px)],
                             ignore_index=True)
    live = _signal(monkeypatch, with_forming, px, forming=True)
    assert live["ok"], live.get("error")

    for k in GATED:
        assert live[k] == closed[k], f"{k} repainted on a ±6% forming bar"


def test_decision_bar_is_reported_and_is_a_closed_bar(monkeypatch):
    hist = _history()
    px = float(hist["close"].iloc[-1]) * 1.03
    with_forming = pd.concat([hist, _forming_bar(hist.iloc[-1], px)],
                             ignore_index=True)
    r = _signal(monkeypatch, with_forming, px, forming=True)

    # the signal is stamped with the last CLOSED bar, not the forming one
    assert r["bar_time"] == str(hist["timestamp"].iloc[-1])
    assert r["signal_px"] == pytest.approx(round(float(hist["close"].iloc[-1]), 2))
    assert r["ltp"] == pytest.approx(round(px, 2))
    assert r["drift_atr"] > 0          # price ran up since the bar closed
    assert r["bar_closed"] is False    # ...and we say so


def test_live_price_is_still_the_fill_when_we_say_enter(monkeypatch):
    """Gating on the closed bar must not hand the user an unfillable price."""
    hist = _history()
    px = float(hist["close"].iloc[-1]) * 1.001
    with_forming = pd.concat([hist, _forming_bar(hist.iloc[-1], px)],
                             ignore_index=True)
    r = _signal(monkeypatch, with_forming, px, forming=True)
    if r["status"] == "ENTER":
        assert r["entry"] == pytest.approx(round(px, 2))
        # stop and targets are measured from that fill
        assert abs(abs(r["entry"] - r["stop"]) - r["risk_pts"]) < 0.02
