"""Atlas dashboard — live view of setups, signal history, and your real
trade-journal stats.

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from engine.scanner import scan
from storage.journal import load_signals, load_trades, stats

st.set_page_config(page_title="Atlas", page_icon="📈", layout="wide")
st.title("📈 Atlas — Intraday Options Scanner")
st.caption("Free directional scanner for F&O underlyings. Signals support your "
           "judgment — they don't replace it. Options carry theta + direction risk.")

# --- Sidebar controls ---------------------------------------------------
with st.sidebar:
    st.header("Controls")
    threshold = st.slider("Conviction threshold", 0.2, 1.0, 0.4, 0.05)
    run_scan = st.button("🔄 Scan now", use_container_width=True)
    st.divider()
    st.caption("Higher threshold = fewer, stronger (A+) setups.")

# --- Live scan ----------------------------------------------------------
st.subheader("🎯 Live setups")
if run_scan:
    with st.spinner("Scanning all F&O underlyings…"):
        snap = scan(threshold=threshold)
    if snap.empty:
        st.info("No setups clear the threshold right now.")
    else:
        calls = snap[snap["side"] == "CALL"]
        puts = snap[snap["side"] == "PUT"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🟢 CALL setups")
            st.dataframe(calls, use_container_width=True, hide_index=True)
        with c2:
            st.markdown("### 🔴 PUT setups")
            st.dataframe(puts, use_container_width=True, hide_index=True)
else:
    st.info("Press **Scan now** in the sidebar to fetch current setups.")

st.divider()

# --- Journal stats ------------------------------------------------------
st.subheader("📓 Your real trade journal")
s = stats()
if s.get("trades", 0) == 0:
    st.info("No trades journaled yet. Log trades to see your *real* win rate.")
else:
    cols = st.columns(5)
    cols[0].metric("Trades", s["trades"])
    cols[1].metric("Win rate", f"{s['win_rate']}%")
    cols[2].metric("Total P&L", f"₹{s['total_pnl']:,}")
    cols[3].metric("Best", f"₹{s['best']:,}")
    cols[4].metric("Worst", f"₹{s['worst']:,}")

    trades = load_trades()
    if "pnl" in trades:
        trades = trades.copy()
        trades["cum_pnl"] = trades["pnl"].cumsum()
        fig = px.line(trades, y="cum_pnl", title="Cumulative P&L (net of costs)",
                      labels={"cum_pnl": "₹", "index": "trade #"})
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Signal history -----------------------------------------------------
st.subheader("🕒 Signal history")
hist = load_signals()
if hist.empty:
    st.info("No signals journaled yet. Run the scanner (or the daily runner).")
else:
    st.caption(f"{len(hist)} signals logged")
    side_counts = hist["side"].value_counts().reset_index()
    side_counts.columns = ["side", "count"]
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(side_counts, hide_index=True, use_container_width=True)
    with c2:
        top = hist["underlying"].value_counts().head(10).reset_index()
        top.columns = ["underlying", "signals"]
        st.bar_chart(top.set_index("underlying"))
    st.dataframe(hist.tail(20).iloc[::-1], use_container_width=True, hide_index=True)
