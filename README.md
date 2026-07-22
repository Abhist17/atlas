# Atlas

Free intraday **options scanner + discipline tool** for NSE F&O trading.

Atlas watches every liquid F&O underlying (NIFTY, BANKNIFTY, and 24 stocks) in
real time, flags high-conviction directional setups (CALL / PUT), journals every
signal and trade, and enforces risk discipline — so you trade *your* edge with
more reach and honest feedback.

> **Honest note:** Extensive backtesting showed that simple technical signals do
> *not* reliably predict short-term direction (see `backtest/`). Atlas is a
> **decision-support and discipline tool**, not an automated money-maker. The
> edge stays with your judgment; Atlas gives you reach, journaling, and risk
> control. Options carry theta decay + direction risk — trade carefully.

## What it does

- **Live scanner** — scans all F&O underlyings every 5 min, ranks setups by
  directional conviction (trend + VWAP + RSI + opening-range breakout + volume)
- **CALL/PUT signals** — signed conviction tells you which side + why (factors)
- **Trade journal** — logs every signal and trade to disk; shows your *real* win rate
- **Risk engine** — position sizing, stop/target, daily loss kill-switch, square-off
- **Backtester** — no-look-ahead replay with realistic costs + quant metrics
- **Dashboard** — Streamlit live view of setups, P&L, and signal history

## Quickstart

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # optional: Dhan creds for future order execution

# Web app (login + screener) — the main product
uvicorn web.app:app --port 8000       # open http://localhost:8000

# Alternatives
streamlit run dashboard/app.py        # quick Streamlit dashboard
python -m engine.scanner              # one-off scan of current setups
python -m engine.runner --once        # single scan + journal
python -m engine.runner               # scheduled loop through market hours
python -m backtest.directional_bt     # validate the directional engine
```

## Web app

A proper multi-user web application (FastAPI):

- **Login / signup** — PBKDF2-hashed passwords, signed session cookies
- **Live screener dashboard** — ~180 NSE F&O stocks, one-click preset scans
  (Bullish / Bearish / Gainers / Losers / Volume-surge / Breakout / RSI extremes),
  sortable columns, side filter, timeframe toggle
- **JSON API** — `/api/screen` (auth-protected, cached)

## Modules

| Module        | Role                                                          |
|---------------|---------------------------------------------------------------|
| `config/`     | Settings, Dhan creds (env), screener/risk params              |
| `data/`       | yfinance data feed, NSE universe, F&O underlyings, Dhan client |
| `engine/`     | Indicators, strategies, ensemble, directional engine, scanner, runner, paper trader, costs |
| `backtest/`   | Backtest engines, quant metrics, parameter optimizer          |
| `storage/`    | Trade + signal journal (Parquet)                              |
| `dashboard/`  | Streamlit UI                                                  |
| `utils/`      | Logging                                                        |

## Data

Free market data via **yfinance** (NSE). Dhan is reserved for order execution.
Options-chain / OI / IV data would require Dhan's paid Data API — not used here.
