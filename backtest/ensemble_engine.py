"""Backtest for the multi-signal ensemble, with transaction costs.

Precomputes each symbol's ensemble score frame once (all signals are
backward-looking), then replays the timeline: at each bar the long candidates
are the symbols whose composite conviction clears the ensemble threshold,
ranked by conviction. Net P&L is after costs.
"""
from __future__ import annotations

import pandas as pd

from backtest.engine import _prepare
from backtest.metrics import summarise
from engine.ensemble import Ensemble
from engine.paper_trader import PaperTrader
from utils.logger import get_logger

log = get_logger("backtest.ensemble")


def _score_all(data: dict[str, pd.DataFrame], ens: Ensemble) -> dict[str, pd.DataFrame]:
    """Attach composite conviction + active-signal count to each symbol frame."""
    scored = {}
    for sym, df in data.items():
        s = ens.score(df)
        merged = df.copy()
        merged["composite"] = s["composite"]
        merged["active"] = s["active"]
        scored[sym] = merged
    return scored


def run(symbols: list[str], days: int = 30, interval: int = 5,
        ens: Ensemble | None = None) -> dict:
    ens = ens or Ensemble()
    data = _prepare(symbols, days, interval)
    if not data:
        log.warning("No usable data for backtest")
        return {"trades": 0}
    data = _score_all(data, ens)

    timeline = sorted(set().union(*[set(df.index) for df in data.values()]))
    pt = PaperTrader()

    equity_by_day, current_day = {}, None
    for ts in timeline:
        day = ts.date()
        if day != current_day:
            pt.start_day({s: df.loc[ts, "close"] for s, df in data.items() if ts in df.index})
            current_day = day

        prices, rows = {}, []
        for sym, df in data.items():
            if ts not in df.index:
                continue
            row = df.loc[ts]
            prices[sym] = float(row["close"])
            if row["composite"] >= ens.threshold and row["active"] >= ens.min_agree:
                rows.append({"symbol": sym, "close": float(row["close"]),
                             "score": float(row["composite"])})
        cand = pd.DataFrame(rows)
        if not cand.empty:
            cand = cand.sort_values("score", ascending=False)
        pt.step(cand, prices, ts.to_pydatetime())
        equity_by_day[day] = pt.pf.equity(prices)

    trades = pd.DataFrame([{"pnl": t.pnl, "gross": t.gross_pnl, "cost": t.cost,
                            "reason": t.reason, "symbol": t.symbol}
                           for t in pt.pf.trades])
    equity = pd.Series(equity_by_day).sort_index()
    report = summarise(trades, equity, pt.risk.starting_capital)
    report["trades_log"] = trades
    report["equity"] = equity
    report["total_costs"] = round(sum(t.cost for t in pt.pf.trades), 2)
    return report


if __name__ == "__main__":
    universe = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
                "SBIN", "AXISBANK", "ITC", "LT", "BHARTIARTL"]
    rep = run(universe, days=30, interval=5)
    print("\n=== Ensemble backtest (net of costs) ===")
    for k, v in rep.items():
        if k not in ("trades_log", "equity"):
            print(f"  {k:18} {v}")
