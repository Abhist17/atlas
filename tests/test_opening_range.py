"""The opening range must never see a bar that has not happened yet."""
from __future__ import annotations

import pandas as pd
import pytest

from engine.directional import add_opening_range
from tests.conftest import make_session

OR_BARS = 6


def test_range_expands_inside_the_window_and_freezes_after():
    df = make_session("2026-08-14", seed=7)
    out = add_opening_range(df, opening_bars=OR_BARS)

    # inside the window each bar sees only the bars up to and including itself
    for i in range(OR_BARS):
        assert out["or_high"].iloc[i] == pytest.approx(df["high"].iloc[: i + 1].max())
        assert out["or_low"].iloc[i] == pytest.approx(df["low"].iloc[: i + 1].min())

    # after the window it is frozen at the window's extremes
    frozen_hi = df["high"].iloc[:OR_BARS].max()
    frozen_lo = df["low"].iloc[:OR_BARS].min()
    assert (out["or_high"].iloc[OR_BARS:] == frozen_hi).all()
    assert (out["or_low"].iloc[OR_BARS:] == frozen_lo).all()


def test_truncating_the_future_does_not_change_the_past(bars):
    """The signature of lookahead: recomputing on a shorter frame changes values."""
    full = add_opening_range(bars, opening_bars=OR_BARS)
    for cut in (3, 5, 8, 40):
        partial = add_opening_range(bars.iloc[:cut], opening_bars=OR_BARS)
        pd.testing.assert_series_equal(
            partial["or_high"], full["or_high"].iloc[:cut], check_names=False)
        pd.testing.assert_series_equal(
            partial["or_low"], full["or_low"].iloc[:cut], check_names=False)


def test_range_resets_each_session(bars):
    out = add_opening_range(bars, opening_bars=OR_BARS)
    day = pd.to_datetime(out["timestamp"]).dt.date
    per_day = out.groupby(day)["or_high"].nunique(dropna=True)
    assert len(per_day) == 3
    # each session has its own frozen level, not one level shared across days
    assert out.groupby(day)["or_high"].last().nunique() == 3
