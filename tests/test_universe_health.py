"""Dead tickers must be pruned — but only the genuinely dead ones.

NSE's F&O list changes under a static Python list, so symbols rot. The risk in
fixing that is over-correcting: one flaky afternoon at the data provider must not
quietly shrink the tradable universe, and a symbol that recovers must come back.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from data import universe_health as uh


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(uh, "HEALTH_FILE", tmp_path / "universe_health.json")
    monkeypatch.setattr(uh, "FNO_UNIVERSE", ["AAA", "BBB", "CCC"])


def test_a_fresh_universe_is_entirely_tradable():
    assert uh.tradable() == ["AAA", "BBB", "CCC"]
    assert uh.dead_symbols() == set()


def test_one_failure_does_not_evict_a_symbol():
    """A dropped connection is not a delisting."""
    uh.record_failure("BBB")
    assert "BBB" in uh.tradable()


def test_repeated_failures_mark_it_dead():
    for _ in range(uh.DEAD_AFTER):
        uh.record_failure("BBB")
    assert uh.dead_symbols() == {"BBB"}
    assert uh.tradable() == ["AAA", "CCC"]


def test_a_success_clears_the_history_immediately():
    for _ in range(uh.DEAD_AFTER):
        uh.record_failure("BBB")
    uh.record_success("BBB")
    assert uh.dead_symbols() == set()
    assert uh.tradable() == ["AAA", "BBB", "CCC"]


def test_failures_must_be_consecutive_to_count():
    uh.record_failure("BBB")
    uh.record_failure("BBB")
    uh.record_success("BBB")        # resets the streak
    uh.record_failure("BBB")
    assert uh.dead_symbols() == set()


def test_a_dead_mark_expires_so_the_symbol_gets_retried():
    """A symbol killed by a bad day at the provider has to be able to come back."""
    for _ in range(uh.DEAD_AFTER):
        uh.record_failure("BBB")
    later = datetime.now() + timedelta(days=uh.RECHECK_DAYS + 1)
    assert uh.dead_symbols(now=later) == set()


def test_a_dead_mark_stands_inside_the_recheck_window():
    for _ in range(uh.DEAD_AFTER):
        uh.record_failure("BBB")
    soon = datetime.now() + timedelta(days=uh.RECHECK_DAYS - 1)
    assert uh.dead_symbols(now=soon) == {"BBB"}


def test_order_is_preserved_when_pruning():
    for _ in range(uh.DEAD_AFTER):
        uh.record_failure("AAA")
    assert uh.tradable() == ["BBB", "CCC"]


def test_symbols_are_matched_case_insensitively():
    for _ in range(uh.DEAD_AFTER):
        uh.record_failure("bbb")
    assert uh.tradable() == ["AAA", "CCC"]


def test_report_separates_dead_from_merely_ailing():
    uh.record_failure("AAA")                      # one strike: ailing
    for _ in range(uh.DEAD_AFTER):
        uh.record_failure("BBB")                  # dead
    r = uh.report()
    assert r["dead"] == ["BBB"]
    assert r["ailing"] == ["AAA"]
    assert r["tradable"] == 2


def test_a_corrupt_health_file_does_not_shrink_the_universe():
    """Failing open matters here: failing closed would scan nothing at all."""
    uh.HEALTH_FILE.write_text("{ not json")
    assert uh.tradable() == ["AAA", "BBB", "CCC"]


def test_an_unparseable_timestamp_is_retried_not_trusted():
    uh.HEALTH_FILE.write_text('{"BBB": {"failures": 9, "dead_since": "whenever"}}')
    assert uh.dead_symbols() == set()
