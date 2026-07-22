"""Instrument universe — loads Dhan's scrip master and builds the tradable
NSE equity list (symbol → security_id). Cached locally as Parquet so we don't
re-download every run.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from config.settings import CACHE_DIR
from utils.logger import get_logger

log = get_logger("data.instruments")

SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
_CACHE = CACHE_DIR / "nse_equity_universe.parquet"

# Dhan scrip-master column names
COL_EXCH = "SEM_EXM_EXCH_ID"
COL_SEGMENT = "SEM_SEGMENT"
COL_SECID = "SEM_SMST_SECURITY_ID"
COL_INSTR = "SEM_INSTRUMENT_NAME"
COL_SYMBOL = "SEM_TRADING_SYMBOL"
COL_SERIES = "SEM_SERIES"
COL_NAME = "SM_SYMBOL_NAME"


def load_nse_equities(refresh: bool = False) -> pd.DataFrame:
    """Return NSE cash-equity instruments as a DataFrame:
    columns = [security_id, symbol, name]. Cached daily.
    """
    if _CACHE.exists() and not refresh:
        cached = pd.read_parquet(_CACHE)
        # Refresh once a day
        if cached.attrs.get("date") == date.today().isoformat():
            return cached

    log.info("Downloading Dhan scrip master…")
    raw = pd.read_csv(SCRIP_MASTER_URL, low_memory=False)

    mask = (
        (raw[COL_EXCH] == "NSE")
        & (raw[COL_SEGMENT] == "E")            # E = equity cash segment
        & (raw[COL_INSTR] == "EQUITY")
        & (raw[COL_SERIES] == "EQ")
    )
    eq = raw.loc[mask, [COL_SECID, COL_SYMBOL, COL_NAME]].copy()
    eq.columns = ["security_id", "symbol", "name"]
    eq["security_id"] = eq["security_id"].astype(str)
    eq = eq.drop_duplicates("security_id").reset_index(drop=True)
    eq.attrs["date"] = date.today().isoformat()

    eq.to_parquet(_CACHE, index=False)
    log.info("Loaded %d NSE equities", len(eq))
    return eq


def security_id_for(symbol: str) -> str | None:
    """Look up a security_id by trading symbol (e.g. 'RELIANCE')."""
    uni = load_nse_equities()
    hit = uni.loc[uni["symbol"] == symbol.upper(), "security_id"]
    return hit.iloc[0] if len(hit) else None


if __name__ == "__main__":
    df = load_nse_equities(refresh=True)
    print(f"{len(df)} NSE equities")
    print(df.head(10).to_string(index=False))
