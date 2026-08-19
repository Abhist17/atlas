"""Universe sweep — build the track record from the market, not from your clicks.

`/api/levels` journals a signal when you open a stock, which means the recorded
sample is whatever you happened to look at. That is the worst possible sample to
measure yourself on: you click what looks interesting, so the log inherits every
bias you were hoping to check for.

The sweeper fixes that by journaling on a schedule instead of on attention. Each
pass it screens the tradable universe, takes the strongest candidates by absolute
conviction, and runs the full signal on those — journaling every ENTER it finds
whether or not anyone is watching.

Why the top-N rather than all ~180: `compute_signal` fetches per symbol, so a
full sweep costs minutes and cannot fit a 5-minute cadence. The screener's batch
fetch already ranks the universe cheaply, and a setup that isn't in the top of
that ranking was never going to be an ENTER worth recording. `limit` is the knob
between coverage and cost.

Only ENTER calls are journaled here. WAIT and AVOID are the majority of every
pass, and writing ~180 of them every five minutes would bury the rows that can
actually be scored under a hundred thousand that never will be. The calls you
personally see are still journaled in full by the API.

Run:  python -m engine.sweeper --once
"""
from __future__ import annotations

from datetime import datetime, time

import pandas as pd

from config.settings import config
from storage.live_journal import record
from utils.logger import get_logger

log = get_logger("engine.sweeper")

DEFAULT_LIMIT = 25


def market_open(now: datetime | None = None) -> bool:
    """Is the NSE cash session running right now?"""
    now = now or datetime.now()
    o = time(*map(int, config.market_open.split(":")))
    c = time(*map(int, config.market_close.split(":")))
    return now.weekday() < 5 and o <= now.time() <= c


def candidates(limit: int = DEFAULT_LIMIT, interval: int = 5) -> list[str]:
    """The strongest symbols this pass, by absolute directional conviction."""
    from web.data_service import get_screen

    df = get_screen(interval)
    if df is None or df.empty or "conviction" not in df:
        return []
    ranked = df.reindex(df["conviction"].abs().sort_values(ascending=False).index)
    return [str(s) for s in ranked["symbol"].head(limit)]


def sweep(limit: int = DEFAULT_LIMIT, interval: int = 5,
          force: bool = False) -> dict:
    """One pass: screen, score the leaders, journal every ENTER.

    Never raises — this runs unattended on a timer, and one bad symbol must not
    kill the loop that records everything else.
    """
    from engine.alpha_signal import compute_signal

    if not force and not market_open():
        log.info("Market closed — skipping sweep")
        return {"skipped": "market closed", "journaled": 0}

    syms = candidates(limit, interval)
    if not syms:
        return {"skipped": "no candidates", "journaled": 0}

    journaled, entered, errors = 0, 0, 0
    for sym in syms:
        try:
            sig = compute_signal(sym, interval=interval)
        except Exception as e:                      # noqa: BLE001 — keep sweeping
            log.debug("Sweep failed for %s: %s", sym, e)
            errors += 1
            continue
        if not sig.get("ok"):
            errors += 1
            continue
        if sig.get("status") != "ENTER":
            continue
        entered += 1
        if record(sig):
            journaled += 1

    log.info("Sweep: %d scanned, %d ENTER, %d newly journaled, %d failed",
             len(syms), entered, journaled, errors)
    return {"scanned": len(syms), "enters": entered, "journaled": journaled,
            "errors": errors, "at": datetime.now().isoformat(timespec="seconds")}


def run_scheduled(every_min: int = 5, limit: int = DEFAULT_LIMIT,
                  interval: int = 5) -> None:
    """Blocking scheduler, for running the sweeper as its own process."""
    from apscheduler.schedulers.blocking import BlockingScheduler

    log.info("Sweeper started (top %d, every %dm). Ctrl-C to stop.", limit, every_min)
    sweep(limit, interval)
    sched = BlockingScheduler(timezone="Asia/Kolkata")
    sched.add_job(lambda: sweep(limit, interval), "interval", minutes=every_min)
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Sweeper stopped")


if __name__ == "__main__":
    import argparse
    import json as _json

    ap = argparse.ArgumentParser(description="Journal ENTER signals on a schedule.")
    ap.add_argument("--once", action="store_true", help="single forced sweep now")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    ap.add_argument("--interval", type=int, default=5)
    ap.add_argument("--every", type=int, default=5, help="minutes between sweeps")
    a = ap.parse_args()
    if a.once:
        print(_json.dumps(sweep(a.limit, a.interval, force=True), indent=2))
    else:
        run_scheduled(a.every, a.limit, a.interval)
