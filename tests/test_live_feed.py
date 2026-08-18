"""The feed must know which bar is still forming, and never rewrite a closed one."""
from __future__ import annotations

import pandas as pd
import pytest

from data import live_feed
from tests.conftest import IST, make_session


def test_bar_is_forming_tracks_the_bar_clock():
    now = pd.Timestamp.now(tz=IST)
    # stamped one minute ago on a 5-minute series → still building
    assert live_feed.bar_is_forming(now - pd.Timedelta(minutes=1), 5)
    # stamped six minutes ago → closed
    assert not live_feed.bar_is_forming(now - pd.Timedelta(minutes=6), 5)
    # the same stamp on a 15-minute series is still forming
    assert live_feed.bar_is_forming(now - pd.Timedelta(minutes=6), 15)


def test_naive_timestamps_are_read_as_ist():
    naive = pd.Timestamp.now(tz=IST).tz_localize(None)
    assert live_feed.bar_is_forming(naive - pd.Timedelta(minutes=1), 5)
    assert not live_feed.bar_is_forming(naive - pd.Timedelta(minutes=30), 5)


def _feed(monkeypatch, df, ltp):
    monkeypatch.setattr(live_feed.yfc, "intraday", lambda *a, **k: df)
    monkeypatch.setattr(live_feed, "live_ltp", lambda s: ltp)
    return live_feed.get_bars("TEST", days=1, interval=5)


def test_closed_final_bar_is_never_patched(monkeypatch):
    """A closed candle is history. Overwriting it desyncs every indicator."""
    df = make_session("2026-08-14", bars=30, seed=3)          # long past
    out = _feed(monkeypatch, df, ltp=999.0)
    assert out["forming"] is False
    assert out["ltp"] == 999.0                                # still reported
    pd.testing.assert_frame_equal(out["bars"], df.reset_index(drop=True))


def test_forming_final_bar_is_patched(monkeypatch):
    df = make_session("2026-08-14", bars=30, seed=3)
    now = pd.Timestamp.now(tz=IST).floor("5min")
    df.loc[df.index[-1], "timestamp"] = now                   # last bar = right now
    out = _feed(monkeypatch, df, ltp=999.0)
    assert out["forming"] is True
    last = out["bars"].iloc[-1]
    assert last["close"] == 999.0 and last["high"] == 999.0   # high stretched up


def test_no_live_price_falls_back_to_the_bar_close(monkeypatch):
    df = make_session("2026-08-14", bars=30, seed=3)
    out = _feed(monkeypatch, df, ltp=None)
    assert out["is_live"] is False
    assert out["ltp"] == pytest.approx(round(float(df["close"].iloc[-1]), 2))
