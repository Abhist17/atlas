"""Virtual portfolio for paper trading — tracks cash, open positions, and a
closed-trade log. All prices in INR. Pure bookkeeping; no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Position:
    symbol: str
    qty: int
    entry_price: float
    stop: float
    target: float
    entry_time: datetime

    def unrealised(self, price: float) -> float:
        return (price - self.entry_price) * self.qty


@dataclass
class Trade:
    symbol: str
    qty: int
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    reason: str  # 'target' | 'stop' | 'square_off'

    @property
    def pnl(self) -> float:
        return round((self.exit_price - self.entry_price) * self.qty, 2)

    @property
    def pnl_pct(self) -> float:
        return round((self.exit_price / self.entry_price - 1) * 100, 3)


@dataclass
class Portfolio:
    starting_capital: float
    cash: float = field(init=False)
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cash = self.starting_capital

    # --- queries ---------------------------------------------------------
    def holds(self, symbol: str) -> bool:
        return symbol in self.positions

    def equity(self, prices: dict[str, float]) -> float:
        """Cash + marked-to-market value of open positions."""
        mtm = sum(
            p.qty * prices.get(sym, p.entry_price)
            for sym, p in self.positions.items()
        )
        return round(self.cash + mtm, 2)

    @property
    def realised_pnl(self) -> float:
        return round(sum(t.pnl for t in self.trades), 2)

    # --- mutations -------------------------------------------------------
    def open(self, pos: Position) -> None:
        cost = pos.qty * pos.entry_price
        if cost > self.cash:
            raise ValueError(f"Insufficient cash for {pos.symbol}: {cost} > {self.cash}")
        self.cash -= cost
        self.positions[pos.symbol] = pos

    def close(self, symbol: str, price: float, when: datetime, reason: str) -> Trade:
        pos = self.positions.pop(symbol)
        self.cash += pos.qty * price
        trade = Trade(
            symbol=symbol,
            qty=pos.qty,
            entry_price=pos.entry_price,
            exit_price=price,
            entry_time=pos.entry_time,
            exit_time=when,
            reason=reason,
        )
        self.trades.append(trade)
        return trade
