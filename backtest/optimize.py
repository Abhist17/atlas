"""Strategy parameter optimizer.

Sweeps screener/risk parameters through the backtester to search for a
profitable configuration. Guards against overfitting with a train/test split:
parameters are ranked on the TRAIN window, then the best is re-scored on the
unseen TEST window. A config that only works in-sample is a red flag.
"""
from __future__ import annotations

import itertools
from copy import deepcopy

import pandas as pd

from backtest.engine import _prepare
from backtest.metrics import summarise
from config.settings import ScreenerConfig, config
from engine.paper_trader import PaperTrader
from engine.screener import _passes, _score
from utils.logger import get_logger

log = get_logger("backtest.optimize")


def _replay(data: dict[str, pd.DataFrame], cfg: ScreenerConfig,
            stop_pct: float, target_pct: float) -> dict:
    """Replay precomputed data under a given screener cfg + stop/target."""
    risk = deepcopy(config.risk)
    risk.stop_loss_pct = stop_pct
    risk.take_profit_pct = target_pct
    pt = PaperTrader(risk=risk)

    timeline = sorted(set().union(*[set(df.index) for df in data.values()]))
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
            if _passes(row, cfg):
                rows.append({"symbol": sym, "close": float(row["close"]), "score": _score(row)})
        cand = pd.DataFrame(rows)
        if not cand.empty:
            cand = cand.sort_values("score", ascending=False)
        pt.step(cand, prices, ts.to_pydatetime())
        equity_by_day[day] = pt.pf.equity(prices)

    trades = pd.DataFrame([{"pnl": t.pnl} for t in pt.pf.trades])
    equity = pd.Series(equity_by_day).sort_index()
    return summarise(trades, equity, risk.starting_capital)


# Search space — kept small to limit overfitting surface
GRID = {
    "stop_pct": [0.5, 0.75, 1.0],
    "target_pct": [1.0, 1.5, 2.0],
    "rsi_lower": [50.0, 55.0, 60.0],
}


def optimize(symbols: list[str], days: int = 30, interval: int = 5,
             train_frac: float = 0.7) -> pd.DataFrame:
    """Grid-search on the train window, rank by expectancy, then report each
    config's out-of-sample TEST performance. Returns a ranked DataFrame.
    """
    data = _prepare(symbols, days, interval)
    if not data:
        log.warning("No data to optimize on")
        return pd.DataFrame()

    # Time-based split (no shuffling — respects chronology)
    all_days = sorted({ts.date() for df in data.values() for ts in df.index})
    split = all_days[int(len(all_days) * train_frac)]
    train = {s: df[df.index.map(lambda t: t.date() < split)] for s, df in data.items()}
    test = {s: df[df.index.map(lambda t: t.date() >= split)] for s, df in data.items()}
    train = {s: d for s, d in train.items() if not d.empty}
    test = {s: d for s, d in test.items() if not d.empty}
    log.info("Split: %d train days / %d test days", len([d for d in all_days if d < split]),
             len([d for d in all_days if d >= split]))

    results = []
    for stop, target, rsi_lo in itertools.product(*GRID.values()):
        cfg = deepcopy(config.screener)
        cfg.rsi_lower = rsi_lo
        tr = _replay(train, cfg, stop, target)
        te = _replay(test, cfg, stop, target)
        results.append({
            "stop_pct": stop, "target_pct": target, "rsi_lower": rsi_lo,
            "train_expectancy": tr.get("expectancy", 0),
            "train_pf": tr.get("profit_factor", 0),
            "test_expectancy": te.get("expectancy", 0),
            "test_pf": te.get("profit_factor", 0),
            "test_sharpe": te.get("sharpe", 0),
            "test_trades": te.get("trades", 0),
        })

    df = pd.DataFrame(results).sort_values("train_expectancy", ascending=False)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    universe = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
                "SBIN", "AXISBANK", "ITC", "LT", "BHARTIARTL"]
    ranked = optimize(universe, days=30, interval=5)
    if not ranked.empty:
        print("\n=== Top configs by TRAIN expectancy (check TEST holds up) ===")
        print(ranked.head(10).to_string(index=False))
