"""Transaction-cost model for NSE intraday equity (MIS).

A strategy must clear these costs to be real. Rates reflect typical discount-
broker intraday charges (Dhan-style). Slippage is modelled separately in bps
since it dominates for fast intraday entries.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    # Brokerage: min(flat, pct of turnover) per executed order
    brokerage_flat: float = 20.0
    brokerage_pct: float = 0.0003          # 0.03%
    # Statutory (intraday equity)
    stt_sell_pct: float = 0.00025          # 0.025% on sell side only
    exch_txn_pct: float = 0.0000297        # NSE ~0.00297%
    sebi_pct: float = 0.000001             # 0.0001%
    stamp_buy_pct: float = 0.00003         # 0.003% on buy side only
    gst_pct: float = 0.18                  # 18% on (brokerage + txn charges)
    # Execution slippage, per side, in basis points (1 bp = 0.01%)
    slippage_bps: float = 5.0

    def _brokerage(self, turnover: float) -> float:
        return min(self.brokerage_flat, turnover * self.brokerage_pct)

    def round_trip(self, entry: float, exit: float, qty: int) -> float:
        """Total cost (INR) for a buy+sell intraday round trip."""
        buy_val = entry * qty
        sell_val = exit * qty

        brokerage = self._brokerage(buy_val) + self._brokerage(sell_val)
        exch = (buy_val + sell_val) * self.exch_txn_pct
        sebi = (buy_val + sell_val) * self.sebi_pct
        gst = (brokerage + exch + sebi) * self.gst_pct
        stt = sell_val * self.stt_sell_pct
        stamp = buy_val * self.stamp_buy_pct

        slip = (buy_val + sell_val) * (self.slippage_bps / 10_000)
        return round(brokerage + exch + sebi + gst + stt + stamp + slip, 2)


DEFAULT_COSTS = CostModel()
