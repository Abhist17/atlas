"""Live daily runner — the heartbeat of Atlas.

During market hours it scans every `interval_min` minutes, journals the flagged
setups, and prints the top ones so you can act (buy the call/put) with your own
judgment. Outside market hours it idles.

Run:  python -m engine.runner          # scheduled loop
      python -m engine.runner --once    # single scan now (for testing)
"""
from __future__ import annotations

import argparse
from datetime import datetime, time

from apscheduler.schedulers.blocking import BlockingScheduler

from config.settings import config
from engine.scanner import scan
from storage.journal import log_signals
from utils.logger import get_logger

log = get_logger("engine.runner")


def _market_open(now: datetime) -> bool:
    o = time(*map(int, config.market_open.split(":")))
    c = time(*map(int, config.market_close.split(":")))
    return now.weekday() < 5 and o <= now.time() <= c


def cycle(threshold: float = 0.4, force: bool = False) -> None:
    """One scan cycle: skip if market closed (unless forced)."""
    now = datetime.now()
    if not force and not _market_open(now):
        log.info("Market closed — skipping scan")
        return
    snap = scan(threshold=threshold)
    log_signals(snap, now)
    if snap.empty:
        log.info("No setups this cycle")
    else:
        top = snap.head(5)
        log.info("Top setups:\n%s", top.to_string(index=False))


def run_scheduled(interval_min: int = 5, threshold: float = 0.4) -> None:
    log.info("Atlas runner started (mode=%s, every %dm). Ctrl-C to stop.",
             config.mode, interval_min)
    cycle(threshold)  # immediate first scan
    sched = BlockingScheduler(timezone="Asia/Kolkata")
    sched.add_job(lambda: cycle(threshold), "interval", minutes=interval_min)
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Runner stopped")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="single forced scan now")
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--interval", type=int, default=5)
    args = ap.parse_args()
    if args.once:
        cycle(threshold=args.threshold, force=True)
    else:
        run_scheduled(interval_min=args.interval, threshold=args.threshold)
