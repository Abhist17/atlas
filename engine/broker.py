"""Order execution — Paper (simulated) and Live (Dhan) in one interface.

Paper fills are instant at the quoted LTP and stored as open positions in
data_store/positions.parquet; closing one computes net P&L (via the cost model)
and writes a closed trade to the journal. Live orders go through the Dhan API,
resolving the option's security_id from Dhan's scrip master on demand.
"""
from __future__ import annotations

import uuid
from datetime import datetime

import pandas as pd

from config.settings import DATA_STORE, config
from data.option_chain import lot_size
from storage import journal
from utils.logger import get_logger

log = get_logger("engine.broker")

POSITIONS_FILE = DATA_STORE / "positions.parquet"


# ------------------------------------------------------------------ storage
def _load() -> pd.DataFrame:
    if POSITIONS_FILE.exists():
        return pd.read_parquet(POSITIONS_FILE)
    return pd.DataFrame(columns=[
        "id", "mode", "symbol", "expiry", "strike", "opt_type", "side",
        "lots", "qty", "entry", "opened_at", "status", "dhan_order_id"])


def _save(df: pd.DataFrame) -> None:
    df.to_parquet(POSITIONS_FILE, index=False)


def open_positions() -> list[dict]:
    df = _load()
    if df.empty:
        return []
    return df[df["status"] == "OPEN"].to_dict("records")


# ------------------------------------------------------------------ orders
def place_order(mode: str, symbol: str, expiry: str, strike: float,
                opt_type: str, side: str, lots: int, ltp: float) -> dict:
    """Open a position. side = BUY/SELL, opt_type = CE/PE, mode = paper/live."""
    symbol, opt_type, side = symbol.upper(), opt_type.upper(), side.upper()
    lot = lot_size(symbol)
    qty = lots * lot
    dhan_id = None

    if mode == "live":
        ok, dhan_id, msg = _dhan_place(symbol, expiry, strike, opt_type, side, qty)
        if not ok:
            return {"ok": False, "error": msg}

    pos = {
        "id": uuid.uuid4().hex[:8], "mode": mode, "symbol": symbol,
        "expiry": expiry, "strike": float(strike), "opt_type": opt_type,
        "side": side, "lots": int(lots), "qty": int(qty), "entry": float(ltp),
        "opened_at": datetime.now().isoformat(timespec="seconds"),
        "status": "OPEN", "dhan_order_id": dhan_id or "",
    }
    df = pd.concat([_load(), pd.DataFrame([pos])], ignore_index=True)
    _save(df)
    log.info("%s %s %s %s%s x%d @ %.2f", mode.upper(), side, symbol, int(strike), opt_type, lots, ltp)
    return {"ok": True, "position": pos}


def close_position(pos_id: str, ltp: float) -> dict:
    """Square off an open position at ltp and journal the realised P&L."""
    df = _load()
    m = (df["id"] == pos_id) & (df["status"] == "OPEN")
    if not m.any():
        return {"ok": False, "error": "Position not found."}
    p = df[m].iloc[0]

    if p["mode"] == "live":
        exit_side = "SELL" if p["side"] == "BUY" else "BUY"
        ok, _, msg = _dhan_place(p["symbol"], p["expiry"], p["strike"],
                                 p["opt_type"], exit_side, int(p["qty"]))
        if not ok:
            return {"ok": False, "error": msg}

    direction = 1 if p["side"] == "BUY" else -1
    gross = (ltp - p["entry"]) * p["qty"] * direction
    try:
        from engine.costs import DEFAULT_COSTS
        cost = DEFAULT_COSTS.round_trip(p["entry"], ltp, int(p["qty"]))
    except Exception:
        cost = p["entry"] * p["qty"] * 0.0012 + ltp * p["qty"] * 0.0012
    net = gross - cost

    journal.log_trade({
        "symbol": f"{p['symbol']} {int(p['strike'])}{p['opt_type']}",
        "mode": p["mode"], "side": p["side"], "lots": int(p["lots"]),
        "qty": int(p["qty"]), "entry": float(p["entry"]), "exit": float(ltp),
        "gross": round(float(gross), 2), "cost": round(float(cost), 2),
        "net_pnl": round(float(net), 2), "pnl": round(float(net), 2),
        "opened_at": p["opened_at"],
        "closed_at": datetime.now().isoformat(timespec="seconds"),
    })
    df.loc[m, "status"] = "CLOSED"
    _save(df)
    return {"ok": True, "net_pnl": round(float(net), 2)}


# ------------------------------------------------------------------ Dhan live
_scrip_master: pd.DataFrame | None = None


def _load_scrip_master() -> pd.DataFrame | None:
    """Dhan detailed scrip master (cached to disk), used to resolve security_id."""
    global _scrip_master
    if _scrip_master is not None:
        return _scrip_master
    path = DATA_STORE / "dhan_scrip_master.csv"
    try:
        if not path.exists():
            log.info("Downloading Dhan scrip master…")
            url = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"
            pd.read_csv(url).to_csv(path, index=False)
        _scrip_master = pd.read_csv(path)
    except Exception as e:
        log.error("Scrip master load failed: %s", e)
        return None
    return _scrip_master


def _resolve_security_id(symbol, expiry, strike, opt_type) -> str | None:
    sm = _load_scrip_master()
    if sm is None:
        return None
    cols = {c.upper(): c for c in sm.columns}
    try:
        sym_c = cols.get("UNDERLYING_SYMBOL") or cols.get("SYMBOL_NAME")
        typ_c = cols.get("OPTION_TYPE")
        strike_c = cols.get("STRIKE_PRICE")
        exp_c = cols.get("SM_EXPIRY_DATE") or cols.get("EXPIRY_DATE")
        sid_c = cols.get("SECURITY_ID")
        exp_iso = datetime.strptime(expiry, "%d-%b-%Y").strftime("%Y-%m-%d")
        f = sm[(sm[sym_c].astype(str).str.upper() == symbol)
               & (sm[typ_c].astype(str).str.upper() == opt_type)
               & (abs(sm[strike_c].astype(float) - float(strike)) < 0.01)
               & (sm[exp_c].astype(str).str.startswith(exp_iso))]
        if not f.empty:
            return str(f.iloc[0][sid_c])
    except Exception as e:
        log.error("security_id resolve failed: %s", e)
    return None


def _dhan_place(symbol, expiry, strike, opt_type, side, qty):
    """Place a live market order via Dhan. Returns (ok, order_id, message)."""
    cfg = config
    if not cfg.dhan.is_configured:
        return False, None, "Dhan credentials missing in .env."
    sid = _resolve_security_id(symbol, expiry, strike, opt_type)
    if not sid:
        return False, None, "Could not resolve Dhan security_id for this option."
    try:
        from dhanhq import dhanhq
        d = dhanhq(cfg.dhan.client_id, cfg.dhan.access_token)
        resp = d.place_order(
            security_id=sid, exchange_segment=d.NSE_FNO,
            transaction_type=(d.BUY if side == "BUY" else d.SELL),
            quantity=int(qty), order_type=d.MARKET, product_type=d.INTRA,
            price=0, tag="atlas")
        if resp.get("status") == "success":
            return True, resp["data"].get("orderId"), "ok"
        return False, None, str(resp.get("remarks") or resp)
    except Exception as e:
        return False, None, f"Dhan error: {e}"
