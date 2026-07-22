# Atlas

Intraday stock screening + trading engine for NSE equities via the
[Dhan](https://dhanhq.co/) API. Atlas screens a stock universe each session
using technical filters, then trades the shortlisted candidates — starting in
**paper-trading mode** before going live.

## Pipeline

```
Universe → Screener → Signals → Execution (paper/live) → Risk mgmt → Dashboard
                                                              ↑
                                                   Backtest (validate on history)
```

## Modules

| Module        | Role                                                        |
|---------------|-------------------------------------------------------------|
| `config/`     | Settings, credentials (env), strategy/risk parameters       |
| `data/`       | Dhan data feed — instruments, OHLCV, live quotes; Parquet cache |
| `engine/`     | Screener + strategy signals + order execution               |
| `backtest/`   | Historical replay + quant metrics (Sharpe, drawdown, etc.)  |
| `storage/`    | Trade log / portfolio persistence                           |
| `dashboard/`  | Streamlit UI — candidates, positions, P&L                   |
| `utils/`      | Logging, shared helpers                                     |

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in Dhan credentials
```

## Status

🚧 Under active development. Currently: foundation + data feed.
Mode defaults to **paper trading** (`ATLAS_MODE=paper`).
