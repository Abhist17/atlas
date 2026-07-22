"""Backtest engine — replays historical intraday bars through the SAME screener
rules and paper-trading logic used live, then reports quant metrics.

No look-ahead: every indicator is backward-looking, so we precompute indicators
once per symbol and index by timestamp. At each bar we screen across symbols,
rank candidates, and step the paper trader.
"""
from __future__ import annotations

import pandas as pd

from backtest.metrics import summarise
from config.settings import ScreenerConfig, config
from data.yf_client import yfc
from engine.indicators import add_indicators
from engine.paper_trader import PaperTrader
from engine.screener import _passes, _score
from utils.logger import get_logger

log = get_logger("backtest.engine")


def _prepare(symbols: list[str], days: int, interval: int) -> dict[str, pd.DataFrame]:
    """Fetch + precompute indicators once per symbol. Skips thin series."""
    data: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = yfc.intraday(sym, days=days, interval=interval)
        if df is None or len(df) < 21:
            continue
        ind = add_indicators(df).dropna(subset=["rsi", "atr", "vwap", "ema21"])
        if not ind.empty:
            data[sym] = ind.set_index("timestamp")
    return data


def run(
    symbols: list[str],
    days: int = 30,
    interval: int = 5,
    cfg: ScreenerConfig | None = None,
) -> dict:
    """Backtest the screener+strategy over `symbols`. Returns a metrics dict
    and attaches the trade log + equity curve under 'trades'/'equity'.
    """
    cfg = cfg or config.screener
    data = _prepare(symbols, days, interval)
    if not data:
        log.warning("No usable data for backtest")
        return {"trades": 0}

    # Master timeline = sorted union of all bar timestamps
    timeline = sorted(set().union(*[set(df.index) for df in data.values()]))
    pt = PaperTrader()

    equity_by_day: dict = {}
    current_day = None
    for ts in timeline:
        day = ts.date()
        if day != current_day:
            # New session: square-off carried by start_day equity snapshot
            prices_open = {s: df.loc[ts, "close"] for s, df in data.items() if ts in df.index}
            pt.start_day(prices_open)
            current_day = day

        # Prices + screener rows available at this bar
        prices: dict[str, float] = {}
        rows = []
        for sym, df in data.items():
            if ts not in df.index:
                continue
            row = df.loc[ts]
            prices[sym] = float(row["close"])
            if _passes(row, cfg):
                rows.append(
                    {"symbol": sym, "close": float(row["close"]), "score": _score(row)}
                )

        candidates = pd.DataFrame(rows)
        if not candidates.empty:
            candidates = candidates.sort_values("score", ascending=False)
        pt.step(candidates, prices, ts.to_pydatetime())
        equity_by_day[day] = pt.pf.equity(prices)  # last snapshot of the day wins

    trades = pd.DataFrame([vars(t) for t in pt.pf.trades])
    if not trades.empty:
        trades["pnl"] = [t.pnl for t in pt.pf.trades]
    equity = pd.Series(equity_by_day).sort_index()

    report = summarise(trades, equity, pt.risk.starting_capital)
    report["trades_log"] = trades
    report["equity"] = equity
    return report


if __name__ == "__main__":
    # Quick demo on a handful of liquid NSE names
    universe = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
                "SBIN", "AXISBANK", "ITC", "LT", "BHARTIARTL"]
    rep = run(universe, days=30, interval=5)
    print("\n=== Backtest summary ===")
    for k, v in rep.items():
        if k not in ("trades_log", "equity"):
            print(f"  {k:18} {v}")
