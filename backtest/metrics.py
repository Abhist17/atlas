"""Quant performance metrics from a trade log / equity curve.

Kept dependency-light (numpy/pandas only) so it can score both backtests and
live paper-trading results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# NSE ~ 252 trading days/year; intraday equity sampled daily here.
TRADING_DAYS = 252


def trade_stats(trades: pd.DataFrame) -> dict:
    """Win rate, avg win/loss, profit factor, expectancy from a trade log.

    trades needs a 'pnl' column (one row per closed trade).
    """
    if trades is None or trades.empty:
        return {"trades": 0}
    pnl = trades["pnl"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_win = wins.sum()
    gross_loss = abs(losses.sum())
    return {
        "trades": int(len(pnl)),
        "win_rate": round(len(wins) / len(pnl) * 100, 2),
        "avg_win": round(wins.mean(), 2) if len(wins) else 0.0,
        "avg_loss": round(losses.mean(), 2) if len(losses) else 0.0,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else float("inf"),
        "expectancy": round(pnl.mean(), 2),
        "total_pnl": round(pnl.sum(), 2),
    }


def equity_stats(equity: pd.Series, starting_capital: float) -> dict:
    """Total return, Sharpe, and max drawdown from a daily equity curve."""
    if equity is None or len(equity) < 2:
        return {}
    rets = equity.pct_change().dropna()
    sharpe = 0.0
    if rets.std() > 0:
        sharpe = np.sqrt(TRADING_DAYS) * rets.mean() / rets.std()

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return {
        "total_return_pct": round((equity.iloc[-1] / starting_capital - 1) * 100, 2),
        "sharpe": round(float(sharpe), 2),
        "max_drawdown_pct": round(float(drawdown.min()) * 100, 2),
    }


def summarise(
    trades: pd.DataFrame, equity: pd.Series, starting_capital: float
) -> dict:
    """Combined report: trade stats + equity stats."""
    return {**trade_stats(trades), **equity_stats(equity, starting_capital)}
