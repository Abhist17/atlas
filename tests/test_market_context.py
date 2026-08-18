"""Relative strength subtracts two percentages — they must share an anchor."""
from __future__ import annotations

import pandas as pd
import pytest

from data.market_context import session_change
from tests.conftest import IST


def _two_sessions(day1_open, day1_close, day2_open, day2_close):
    """Two 5-minute sessions, deliberately gapped between them."""
    idx = (list(pd.date_range("2026-08-13 09:15", periods=3, freq="5min", tz=IST))
           + list(pd.date_range("2026-08-14 09:15", periods=3, freq="5min", tz=IST)))
    opens = [day1_open, day1_open, day1_close, day2_open, day2_open, day2_close]
    return pd.Series(opens, index=pd.DatetimeIndex(idx))


def test_change_is_measured_from_todays_open():
    opens = _two_sessions(100.0, 110.0, 111.0, 112.0)
    # last price 113 vs today's open 111 → +1.8%, NOT +13% off yesterday
    assert session_change(113.0, opens, opens.index) == pytest.approx(1.8, abs=0.01)


def test_an_overnight_gap_does_not_leak_into_the_number():
    flat_day = _two_sessions(100.0, 130.0, 130.0, 130.0)   # huge move yesterday
    assert session_change(130.0, flat_day, flat_day.index) == pytest.approx(0.0)


def test_a_zero_open_does_not_blow_up():
    opens = _two_sessions(100.0, 110.0, 0.0, 0.0)
    assert session_change(113.0, opens, opens.index) == 0.0
