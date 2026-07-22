"""Atlas dashboard — a proper intraday stock screener.

Interactive filters, ranked results, and per-stock intraday charts (price +
VWAP + EMAs + volume). Free data via yfinance.

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data.underlyings import FNO_STOCKS, INDICES, yf_ticker
from data.yf_client import yfc
from engine.directional import _dir_score, add_opening_range
from engine.indicators import add_indicators
from storage.journal import load_signals, load_trades, stats

st.set_page_config(page_title="Atlas Screener", page_icon="📈", layout="wide")


# ---------------------------------------------------------------- helpers
@st.cache_data(ttl=120, show_spinner=False)
def get_frame(name: str, days: int = 2, interval: int = 5) -> pd.DataFrame:
    """Indicator-augmented intraday frame for one symbol (cached 2 min)."""
    ticker = name if name.startswith("^") else yf_ticker(name)
    df = yfc.intraday(ticker, days=days, interval=interval, use_cache=False)
    if df is None or len(df) < 25:
        return pd.DataFrame()
    ind = add_opening_range(add_indicators(df)).dropna(
        subset=["rsi", "atr", "vwap", "ema21"])
    return ind


def screen(names: list[str], interval: int) -> pd.DataFrame:
    """Build a full screener table (all indicators) for a universe."""
    rows = []
    prog = st.progress(0.0, text="Screening…")
    for i, name in enumerate(names):
        prog.progress((i + 1) / len(names), text=f"Screening {name}…")
        ind = get_frame(name, interval=interval)
        if ind.empty:
            continue
        last = ind.iloc[-1]
        score = _dir_score(last)
        rows.append({
            "symbol": name,
            "price": round(float(last["close"]), 2),
            "conviction": round(float(score), 3),
            "side": "CALL" if score > 0 else "PUT",
            "rsi": round(float(last["rsi"]), 1),
            "atr_pct": round(float(last["atr_pct"]), 2),
            "vs_vwap_%": round((last["close"] / last["vwap"] - 1) * 100, 2),
            "vol_x_avg": round(float(last["volume"] / last["avg_volume"]), 2),
            "trend": "up" if last["ema9"] > last["ema21"] else "down",
        })
    prog.empty()
    return pd.DataFrame(rows)


def price_chart(name: str, ind: pd.DataFrame) -> go.Figure:
    """Candlestick + VWAP + EMAs with a volume subplot."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)
    x = ind["timestamp"]
    fig.add_trace(go.Candlestick(
        x=x, open=ind["open"], high=ind["high"], low=ind["low"],
        close=ind["close"], name="price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=ind["vwap"], name="VWAP",
                             line=dict(color="orange", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=ind["ema9"], name="EMA9",
                             line=dict(color="#22c55e", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=ind["ema21"], name="EMA21",
                             line=dict(color="#ef4444", width=1)), row=1, col=1)
    colors = ["#22c55e" if c >= o else "#ef4444"
              for c, o in zip(ind["close"], ind["open"])]
    fig.add_trace(go.Bar(x=x, y=ind["volume"], name="vol",
                         marker_color=colors, showlegend=False), row=2, col=1)
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=30, b=0),
                      xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.02))
    return fig


# ---------------------------------------------------------------- sidebar
st.sidebar.title("📈 Atlas")
page = st.sidebar.radio("View", ["Screener", "Charts", "Journal"])
st.sidebar.divider()

universe_choice = st.sidebar.selectbox(
    "Universe", ["F&O stocks", "Indices", "F&O stocks + Indices"])
if universe_choice == "Indices":
    NAMES = list(INDICES.keys())
elif universe_choice == "F&O stocks + Indices":
    NAMES = list(INDICES.keys()) + FNO_STOCKS
else:
    NAMES = FNO_STOCKS

interval = st.sidebar.selectbox("Timeframe", [5, 15], format_func=lambda m: f"{m} min")


# ================================================================ SCREENER
if page == "Screener":
    st.title("Intraday Stock Screener")
    st.caption("Ranked by directional conviction. Signals support your judgment — "
               "they don't replace it.")

    with st.sidebar:
        st.subheader("Filters")
        min_conv = st.slider("Min |conviction|", 0.0, 1.0, 0.3, 0.05)
        side_f = st.multiselect("Side", ["CALL", "PUT"], default=["CALL", "PUT"])
        rsi_lo, rsi_hi = st.slider("RSI range", 0, 100, (0, 100))
        min_atr = st.slider("Min ATR %", 0.0, 2.0, 0.0, 0.05)
        min_vol = st.slider("Min volume × avg", 0.0, 5.0, 0.0, 0.25)
        trend_f = st.multiselect("Trend", ["up", "down"], default=["up", "down"])
        run = st.button("🔄 Run screen", use_container_width=True, type="primary")

    if run or "screen_df" not in st.session_state:
        st.session_state.screen_df = screen(NAMES, interval)
        st.session_state.screen_time = datetime.now()

    df = st.session_state.screen_df
    if df.empty:
        st.warning("No data returned. Market may be closed or data unavailable.")
        st.stop()

    # Apply filters
    f = df[
        (df["conviction"].abs() >= min_conv)
        & (df["side"].isin(side_f))
        & (df["rsi"].between(rsi_lo, rsi_hi))
        & (df["atr_pct"] >= min_atr)
        & (df["vol_x_avg"] >= min_vol)
        & (df["trend"].isin(trend_f))
    ].copy()
    f = f.reindex(f["conviction"].abs().sort_values(ascending=False).index)

    c = st.columns(4)
    c[0].metric("Matches", len(f))
    c[1].metric("CALL", int((f["side"] == "CALL").sum()))
    c[2].metric("PUT", int((f["side"] == "PUT").sum()))
    c[3].metric("Scanned", len(df))
    st.caption(f"Last run: {st.session_state.screen_time:%H:%M:%S} · {interval}-min bars")

    st.dataframe(
        f, use_container_width=True, hide_index=True,
        column_config={
            "conviction": st.column_config.ProgressColumn(
                "conviction", min_value=-1, max_value=1, format="%.2f"),
            "rsi": st.column_config.NumberColumn("RSI", format="%.1f"),
            "vs_vwap_%": st.column_config.NumberColumn("vs VWAP %", format="%.2f"),
            "vol_x_avg": st.column_config.NumberColumn("vol ×avg", format="%.2f"),
        },
    )
    st.download_button("⬇ Download CSV", f.to_csv(index=False),
                       f"atlas_screen_{datetime.now():%Y%m%d_%H%M}.csv")


# ================================================================== CHARTS
elif page == "Charts":
    st.title("Stock Charts")
    sym = st.selectbox("Symbol", NAMES)
    ind = get_frame(sym, days=5, interval=interval)
    if ind.empty:
        st.warning("No data for this symbol.")
    else:
        last = ind.iloc[-1]
        score = _dir_score(last)
        c = st.columns(5)
        c[0].metric("Price", f"₹{last['close']:.2f}")
        c[1].metric("Conviction", f"{score:.2f}", "CALL" if score > 0 else "PUT")
        c[2].metric("RSI", f"{last['rsi']:.1f}")
        c[3].metric("vs VWAP", f"{(last['close']/last['vwap']-1)*100:.2f}%")
        c[4].metric("ATR %", f"{last['atr_pct']:.2f}")
        st.plotly_chart(price_chart(sym, ind), use_container_width=True)


# ================================================================= JOURNAL
else:
    st.title("Trade Journal")
    s = stats()
    if s.get("trades", 0) == 0:
        st.info("No trades journaled yet — your *real* win rate shows up here.")
    else:
        c = st.columns(5)
        c[0].metric("Trades", s["trades"])
        c[1].metric("Win rate", f"{s['win_rate']}%")
        c[2].metric("Total P&L", f"₹{s['total_pnl']:,}")
        c[3].metric("Best", f"₹{s['best']:,}")
        c[4].metric("Worst", f"₹{s['worst']:,}")
        trades = load_trades()
        if "pnl" in trades:
            trades = trades.copy()
            trades["cum_pnl"] = trades["pnl"].cumsum()
            st.line_chart(trades["cum_pnl"])

    st.divider()
    st.subheader("Signal history")
    hist = load_signals()
    if hist.empty:
        st.info("Run the scanner or daily runner to build signal history.")
    else:
        st.caption(f"{len(hist)} signals logged")
        st.dataframe(hist.tail(30).iloc[::-1], use_container_width=True, hide_index=True)
