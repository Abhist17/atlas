"""Which symbols in the F&O universe can actually be traded today.

`FNO_UNIVERSE` is a static list, and NSE's F&O universe is not: symbols get
delisted, renamed, merged and demerged. TATAMOTORS is the live example — the
ticker no longer returns data, so every screener pass wasted a network round
trip on it and every user who clicked it got "No bar data."

Rather than hand-editing the list each time (which goes stale again immediately),
this tracks failures in `data_store/universe_health.json`:

  - a symbol is only marked dead after `DEAD_AFTER` *consecutive* failures, so a
    dropped connection or a rate limit doesn't evict a perfectly good stock;
  - a dead mark expires after `RECHECK_DAYS`, so a symbol that comes back — or
    was killed by a bad afternoon at the data provider — gets another chance;
  - any success clears the record immediately.

`tradable()` is what callers should scan.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta

from config.settings import DATA_STORE
from data.nse_universe import FNO_UNIVERSE
from utils.logger import get_logger

log = get_logger("data.universe_health")

HEALTH_FILE = DATA_STORE / "universe_health.json"
DEAD_AFTER = 3          # consecutive failures before we stop scanning a symbol
RECHECK_DAYS = 7        # how long a dead mark stands before we retry the symbol

_lock = threading.Lock()


def _load() -> dict:
    if not HEALTH_FILE.exists():
        return {}
    try:
        data = json.loads(HEALTH_FILE.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError) as e:
        log.warning("Could not read %s: %s", HEALTH_FILE.name, e)
        return {}


def _save(data: dict) -> None:
    try:
        HEALTH_FILE.parent.mkdir(parents=True, exist_ok=True)
        HEALTH_FILE.write_text(json.dumps(data, indent=2, sort_keys=True))
    except OSError as e:
        log.warning("Could not write %s: %s", HEALTH_FILE.name, e)


def record_failure(symbol: str, reason: str = "no data") -> None:
    """Count one failed fetch. Marks the symbol dead once it repeats."""
    symbol = symbol.upper()
    with _lock:
        data = _load()
        rec = data.get(symbol, {"failures": 0})
        rec["failures"] = int(rec.get("failures", 0)) + 1
        rec["last_failure"] = datetime.now().isoformat(timespec="seconds")
        rec["reason"] = reason
        if rec["failures"] >= DEAD_AFTER and not rec.get("dead_since"):
            rec["dead_since"] = rec["last_failure"]
            log.warning("%s marked dead after %d consecutive failures (%s)",
                        symbol, rec["failures"], reason)
        data[symbol] = rec
        _save(data)


def record_success(symbol: str) -> None:
    """A good fetch clears the symbol's history — it is alive, full stop."""
    symbol = symbol.upper()
    with _lock:
        data = _load()
        if data.pop(symbol, None) is not None:
            _save(data)


def dead_symbols(now: datetime | None = None) -> set[str]:
    """Symbols currently considered untradable (dead mark still in date)."""
    now = now or datetime.now()
    out = set()
    for sym, rec in _load().items():
        since = rec.get("dead_since")
        if not since:
            continue
        try:
            if now - datetime.fromisoformat(since) < timedelta(days=RECHECK_DAYS):
                out.add(sym)
        except ValueError:
            continue        # unparseable timestamp: treat as expired, retry it
    return out


def tradable(symbols: list[str] | None = None) -> list[str]:
    """The universe worth scanning right now, in the original order."""
    dead = dead_symbols()
    return [s for s in (symbols or FNO_UNIVERSE) if s.upper() not in dead]


def report() -> dict:
    """Health summary — what's excluded and why."""
    data = _load()
    dead = dead_symbols()
    return {
        "universe": len(FNO_UNIVERSE),
        "tradable": len(tradable()),
        "dead": sorted(dead),
        "ailing": sorted(s for s, r in data.items()
                         if s not in dead and r.get("failures", 0) > 0),
    }


def validate(symbols: list[str] | None = None, interval: int = 5) -> dict:
    """Fetch every symbol once and update the health record from the result.

    Run it after an F&O reshuffle (NSE revises the list periodically) to prune
    what no longer trades. Failures here count like any other, so one bad run
    cannot evict a symbol on its own.
    """
    from data.yf_client import yfc

    symbols = list(symbols or FNO_UNIVERSE)
    tickers = [f"{s}.NS" for s in symbols]
    data = yfc.batch_intraday(tickers, days=2, interval=interval)
    alive, failed = [], []
    for sym in symbols:
        df = data.get(f"{sym}.NS")
        if df is not None and not df.empty:
            record_success(sym)
            alive.append(sym)
        else:
            record_failure(sym, "validate: no intraday data")
            failed.append(sym)
    log.info("Validated %d symbols: %d alive, %d failed", len(symbols),
             len(alive), len(failed))
    return {"checked": len(symbols), "alive": len(alive), "failed": sorted(failed),
            **report()}


if __name__ == "__main__":
    import argparse
    import json as _json

    ap = argparse.ArgumentParser(description="Check which F&O symbols still trade.")
    ap.add_argument("--validate", action="store_true",
                    help="fetch every symbol and update the health record")
    a = ap.parse_args()
    print(_json.dumps(validate() if a.validate else report(), indent=2))
