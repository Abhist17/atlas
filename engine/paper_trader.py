"""Paper-trading engine — turns screener candidates into simulated positions
under the risk rules in RiskConfig, and manages exits (stop / target / EOD
square-off) each cycle.

Designed to be driven either live (scheduler calls step() every N minutes) or
by the backtester (feeds historical bars through the same logic).
"""
from __future__ import annotations

from datetime import datetime, time

import pandas as pd

from config.settings import RiskConfig, config
from engine.costs import DEFAULT_COSTS, CostModel
from engine.portfolio import Portfolio, Position
from utils.logger import get_logger

log = get_logger("engine.paper_trader")


def _parse_hhmm(s: str) -> time:
    h, m = map(int, s.split(":"))
    return time(h, m)


class PaperTrader:
    def __init__(self, risk: RiskConfig | None = None,
                 costs: CostModel | None = None) -> None:
        self.risk = risk or config.risk
        self.costs = costs or DEFAULT_COSTS
        self.pf = Portfolio(self.risk.starting_capital)
        self._day_start_equity = self.risk.starting_capital
        self._halted = False  # daily kill-switch tripped
        self._traded_today: set[str] = set()  # no re-entry same day

    # --- position sizing -------------------------------------------------
    def _size(self, entry: float, stop: float) -> int:
        """Shares to buy: risk-based qty, capped by a per-slot capital budget
        (so one name can't consume all capital) and by available cash.
        """
        risk_amount = self.risk.starting_capital * self.risk.risk_per_trade_pct / 100
        per_share_risk = max(entry - stop, 0.01)
        qty = int(risk_amount / per_share_risk)
        # Cap allocation to an equal slice of capital per position slot
        slot_capital = self.risk.starting_capital / self.risk.max_positions
        max_by_slot = int(slot_capital / entry)
        max_affordable = int(self.pf.cash / entry)
        return max(0, min(qty, max_by_slot, max_affordable))

    # --- daily lifecycle -------------------------------------------------
    def start_day(self, prices: dict[str, float] | None = None) -> None:
        self._day_start_equity = self.pf.equity(prices or {})
        self._halted = False
        self._traded_today.clear()

    def _daily_loss_breached(self, prices: dict[str, float]) -> bool:
        dd = self._day_start_equity - self.pf.equity(prices)
        limit = self._day_start_equity * self.risk.max_daily_loss_pct / 100
        return dd >= limit

    # --- core step -------------------------------------------------------
    def step(
        self,
        candidates: pd.DataFrame,
        prices: dict[str, float],
        now: datetime,
    ) -> None:
        """One decision cycle: manage exits first, then consider new entries.

        candidates: screener output (symbol, close, score, …), best-ranked first.
        prices: latest price per symbol (must cover held + candidate symbols).
        now: current timestamp (drives square-off).
        """
        self._manage_exits(prices, now)

        # Kill-switch: stop opening new trades once daily loss cap is hit
        if self._daily_loss_breached(prices):
            if not self._halted:
                log.warning("Daily loss limit hit — halting new entries for the day")
            self._halted = True

        square_off = _parse_hhmm(config.square_off)
        if self._halted or now.time() >= square_off:
            return

        self._consider_entries(candidates, prices, now)

    def _manage_exits(self, prices: dict[str, float], now: datetime) -> None:
        square_off = _parse_hhmm(config.square_off)
        for symbol in list(self.pf.positions):
            pos = self.pf.positions[symbol]
            price = prices.get(symbol)
            if price is None:
                continue
            reason = None
            if price <= pos.stop:
                reason = "stop"
            elif price >= pos.target:
                reason = "target"
            elif now.time() >= square_off:
                reason = "square_off"
            if reason:
                cost = self.costs.round_trip(pos.entry_price, price, pos.qty)
                t = self.pf.close(symbol, price, now, reason, cost=cost)
                self._traded_today.add(symbol)
                log.info("EXIT %s %s @ %.2f net_pnl=%.2f cost=%.2f (%s)",
                         symbol, t.qty, price, t.pnl, cost, reason)

    def _consider_entries(
        self, candidates: pd.DataFrame, prices: dict[str, float], now: datetime
    ) -> None:
        if candidates is None or candidates.empty:
            return
        for rec in candidates.itertuples(index=False):
            if len(self.pf.positions) >= self.risk.max_positions:
                break
            symbol = rec.symbol
            if self.pf.holds(symbol) or symbol in self._traded_today:
                continue
            entry = prices.get(symbol)
            if entry is None or entry <= 0:
                continue
            stop = round(entry * (1 - self.risk.stop_loss_pct / 100), 2)
            target = round(entry * (1 + self.risk.take_profit_pct / 100), 2)
            qty = self._size(entry, stop)
            if qty <= 0:
                continue
            self.pf.open(Position(symbol, qty, entry, stop, target, now))
            log.info("ENTRY %s %s @ %.2f stop=%.2f target=%.2f",
                     symbol, qty, entry, stop, target)
