"""The intraday parquet cache must expire — a stale bar is a wrong signal."""
from __future__ import annotations

import os
import time

import pandas as pd

from data import yf_client
from tests.conftest import make_session


def test_cache_fresh_respects_the_ttl(tmp_path):
    p = tmp_path / "c.parquet"
    make_session("2026-08-14", bars=5).to_parquet(p, index=False)
    assert yf_client._cache_fresh(p, ttl=60)
    # age the file past the TTL
    old = time.time() - 300
    os.utime(p, (old, old))
    assert not yf_client._cache_fresh(p, ttl=60)
    assert yf_client._cache_fresh(p, ttl=600)


def test_missing_cache_is_not_fresh(tmp_path):
    assert not yf_client._cache_fresh(tmp_path / "nope.parquet", ttl=60)


def test_stale_cache_triggers_a_refetch(tmp_path, monkeypatch):
    monkeypatch.setattr(yf_client, "CACHE_DIR", tmp_path)
    stale = make_session("2026-08-12", bars=5, start=100.0)
    fresh = make_session("2026-08-14", bars=5, start=200.0)

    calls = []

    def fake_download(*a, **k):
        calls.append(1)
        out = fresh.rename(columns={"timestamp": "Datetime", "open": "Open",
                                    "high": "High", "low": "Low",
                                    "close": "Close", "volume": "Volume"})
        return out.set_index("Datetime")

    monkeypatch.setattr(yf_client.yf, "download", fake_download)

    cache = tmp_path / "yf_TEST_1d_5m.parquet"
    stale.to_parquet(cache, index=False)
    old = time.time() - 3600
    os.utime(cache, (old, old))

    got = yf_client.yfc.intraday("TEST", days=1, interval=5)
    assert calls, "a stale cache was served instead of refetching"
    assert float(got["close"].iloc[0]) > 150     # the fresh session, not the stale one

    # a second call inside the TTL is served from the (now fresh) cache
    got2 = yf_client.yfc.intraday("TEST", days=1, interval=5)
    assert len(calls) == 1
    pd.testing.assert_frame_equal(got.reset_index(drop=True),
                                  got2.reset_index(drop=True))
