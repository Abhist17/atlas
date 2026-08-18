"""The evaluator must measure the engine that actually runs.

`engine/evaluate.py` reimplements the gate cascade vectorised, for speed. That
is only legitimate while the two stay identical — the moment they drift, the
published hit rates describe an engine nobody trades. This drives both over
random gate states and requires the same verdict every time.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.alpha_signal import _context_gates, _decide
from engine.evaluate import status_cascade

N = 4000
PHASES = np.array(["open", "core", "late"])


def _live(long, trending, htf, htf_ok, aligned, vwap_ok, ext, fresh,
          mkt_against, phase) -> str:
    """The production path: the cascade, then the context overrides."""
    status, headline, trigger, _ = _decide(
        long, htf, htf_ok, trending, 25.0, aligned, "A", fresh, 1, "up",
        vwap_ok, ext, 100.0, 99.0, 99.5)
    status, _, _ = _context_gates(status, headline, trigger, long, "NIFTY",
                                  1 if mkt_against else 0, mkt_against, phase)
    return status


def test_cascade_matches_the_live_engine_on_random_states():
    rng = np.random.default_rng(11)
    htf = rng.choice([-1, 0, 1], N)
    long = rng.random(N) < 0.5
    trending = rng.random(N) < 0.6
    aligned = rng.integers(0, 7, N)
    vwap_ok = rng.random(N) < 0.6
    ext = rng.normal(0.6, 1.2, N)
    fresh = rng.random(N) < 0.3
    mkt_against = rng.random(N) < 0.3
    phase = rng.choice(PHASES, N)
    # htf_ok is derived, exactly as both sides derive it
    htf_ok = ((htf > 0) & long) | ((htf < 0) & ~long)

    vec = status_cascade(trending, htf, htf_ok, aligned, vwap_ok, ext, fresh,
                         mkt_against, phase)
    for i in range(N):
        got = _live(bool(long[i]), bool(trending[i]), int(htf[i]), bool(htf_ok[i]),
                    int(aligned[i]), bool(vwap_ok[i]), float(ext[i]),
                    bool(fresh[i]), bool(mkt_against[i]), str(phase[i]))
        assert vec[i] == got, (
            f"replay says {vec[i]}, live engine says {got} for "
            f"htf={htf[i]} long={long[i]} adx_ok={trending[i]} aligned={aligned[i]} "
            f"vwap_ok={vwap_ok[i]} ext={ext[i]:.2f} fresh={fresh[i]} "
            f"mkt_against={mkt_against[i]} phase={phase[i]}")


@pytest.mark.parametrize("phase,expected", [("open", "WAIT"), ("late", "AVOID"),
                                            ("core", "ENTER")])
def test_time_of_day_overrides_a_clean_enter(phase, expected):
    """A textbook ENTER, downgraded only by the clock."""
    assert _live(True, True, 1, True, 6, True, 0.1, True, False, phase) == expected
    assert status_cascade(np.array([True]), np.array([1]), np.array([True]),
                          np.array([6]), np.array([True]), np.array([0.1]),
                          np.array([True]), np.array([False]),
                          np.array([phase]))[0] == expected


def test_market_gate_outranks_the_clock():
    """Both fire: the market reason is the one reported, and it is a WAIT."""
    assert _live(True, True, 1, True, 6, True, 0.1, True, True, "core") == "WAIT"
