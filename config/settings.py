"""Central configuration for Atlas.

Secrets come from environment variables (.env, never committed). Tunable
strategy/screener parameters live here so the whole system reads one source
of truth.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root if present
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# --- Paths ---------------------------------------------------------------
DATA_STORE = ROOT / "data_store"
CACHE_DIR = DATA_STORE / "cache"
LOG_DIR = ROOT / "logs"
for _d in (CACHE_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class DhanConfig:
    """Dhan broker credentials, pulled from environment."""

    client_id: str = field(default_factory=lambda: os.getenv("DHAN_CLIENT_ID", ""))
    access_token: str = field(default_factory=lambda: os.getenv("DHAN_ACCESS_TOKEN", ""))

    @property
    def is_configured(self) -> bool:
        return bool(self.client_id and self.access_token)


@dataclass
class ScreenerConfig:
    """Intraday screening universe + filter thresholds."""

    # Universe: which instruments to scan. Start with NSE equity.
    exchange: str = "NSE_EQ"
    # Cap the universe while developing; raise later.
    max_universe: int = 200

    # Liquidity / tradability filters
    min_price: float = 50.0
    max_price: float = 5000.0
    min_avg_volume: int = 200_000      # shares/day
    min_atr_pct: float = 1.0           # min intraday volatility (ATR as % of price)

    # Momentum thresholds (screener rules)
    rsi_lower: float = 55.0
    rsi_upper: float = 75.0
    vwap_side: str = "above"           # only long candidates trading above VWAP


@dataclass
class RiskConfig:
    """Risk / money-management limits for paper + live trading."""

    starting_capital: float = 100_000.0
    max_positions: int = 5
    risk_per_trade_pct: float = 1.0    # % of capital risked per trade
    stop_loss_pct: float = 1.0         # per-position hard stop
    take_profit_pct: float = 2.0       # target (2:1 reward:risk)
    max_daily_loss_pct: float = 3.0    # kill-switch for the day


@dataclass
class Config:
    dhan: DhanConfig = field(default_factory=DhanConfig)
    screener: ScreenerConfig = field(default_factory=ScreenerConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    # Global mode: "paper" or "live". Start on paper.
    mode: str = os.getenv("ATLAS_MODE", "paper")

    # Intraday session (IST)
    market_open: str = "09:15"
    market_close: str = "15:30"
    square_off: str = "15:15"          # exit all before close


# Singleton-style accessor
config = Config()
