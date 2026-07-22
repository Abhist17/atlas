"""Atlas — a proper NSE intraday stock screener.

Scans the full F&O universe (~180 liquid stocks) with market-style columns and
one-click preset scans, plus per-stock intraday charts and a trade journal.

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data.nse_universe import FNO_UNIVERSE, INDEX_TICKERS
from data.yf_client import yfc
from engine.directional import _dir_score, add_opening_range
from engine.indicators import add_indicators
from engine.nse_screener import screen_nse
from storage.journal import load_signals, load_trades, stats

st.set_page_config(page_title="Atlas NSE Screener", page_icon="📈", layout="wide")


# ---------------------------------------------------------------- data
@st.cache_data(ttl=120, show_spinner=False)
def run_screen(interval: int) -> pd.DataFrame:
    bar = st.progress(0.0)
    df = screen_nse(interval=interval, progress=lambda f, t: bar.progress(min(f, 1.0), text=t))
    bar.empty()
    return df


@st.cache_data(ttl=120, show_spinner=False)
def get_frame(ticker: str, days: int = 5, interval: int = 5) -> pd.DataFrame:
    df = yfc.intraday(ticker, days=days, interval=interval, use_cache=False)
    if df is None or len(df) < 25:
        return pd.DataFrame()
    return add_opening_range(add_indicators(df)).dropna(subset=["rsi", "atr", "vwap", "ema21"])


# ---------------------------------------------------------------- presets
PRESETS = {
    "All stocks": lambda d: d,
    "🟢 Top Bullish": lambda d: d[(d.conviction > 0.4) & (d.trend == "up") & (d["vs_vwap_%"] > 0)],
    "🔴 Top Bearish": lambda d: d[(d.conviction < -0.4) & (d.trend == "down") & (d["vs_vwap_%"] < 0)],
    "📈 Day Gainers": lambda d: d[d["chg_%"] > 1].sort_values("chg_%", ascending=False),
    "📉 Day Losers": lambda d: d[d["chg_%"] < -1].sort_values("chg_%"),
    "⚡ Volume Surge": lambda d: d[d.vol_x_avg >= 2].sort_values("vol_x_avg", ascending=False),
    "🚀 Breakout Up (ORB)": lambda d: d[(d.orb == "↑") & (d.trend == "up")],
    "🔻 Breakdown (ORB)": lambda d: d[(d.orb == "↓") & (d.trend == "down")],
    "🥶 Oversold (RSI<30)": lambda d: d[d.rsi < 30].sort_values("rsi"),
    "🥵 Overbought (RSI>70)": lambda d: d[d.rsi > 70].sort_values("rsi", ascending=False),
}


def chart(ticker: str, ind: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)
    x = ind["timestamp"]
    fig.add_trace(go.Candlestick(x=x, open=ind["open"], high=ind["high"],
                  low=ind["low"], close=ind["close"], name="price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=ind["vwap"], name="VWAP",
                  line=dict(color="orange", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=ind["ema9"], name="EMA9",
                  line=dict(color="#22c55e", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=ind["ema21"], name="EMA21",
                  line=dict(color="#ef4444", width=1)), row=1, col=1)
    colors = ["#22c55e" if c >= o else "#ef4444" for c, o in zip(ind["close"], ind["open"])]
    fig.add_trace(go.Bar(x=x, y=ind["volume"], marker_color=colors,
                  showlegend=False), row=2, col=1)
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=30, b=0),
                      xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.02))
    return fig


COLCFG = {
    "chg_%": st.column_config.NumberColumn("Chg %", format="%.2f"),
    "conviction": st.column_config.ProgressColumn("Conviction", min_value=-1,
                                                  max_value=1, format="%.2f"),
    "vol_x_avg": st.column_config.NumberColumn("Vol ×avg", format="%.2f"),
    "vs_vwap_%": st.column_config.NumberColumn("vs VWAP %", format="%.2f"),
    "atr_%": st.column_config.NumberColumn("ATR %", format="%.2f"),
    "volume": st.column_config.NumberColumn("Volume", format="%d"),
}

# ---------------------------------------------------------------- sidebar
st.sidebar.title("📈 Atlas NSE")
page = st.sidebar.radio("View", ["Screener", "Charts", "Journal"])
interval = st.sidebar.selectbox("Timeframe", [5, 15], format_func=lambda m: f"{m} min")


# ================================================================ SCREENER
if page == "Screener":
    st.title("NSE Intraday Stock Screener")
    st.caption(f"Scanning {len(FNO_UNIVERSE)} liquid F&O stocks · free data · "
               "signals support your judgment, they don't replace it.")

    top = st.columns([2, 1, 1])
    preset = top[0].selectbox("Preset scan", list(PRESETS.keys()))
    if top[1].button("🔄 Refresh", use_container_width=True):
        run_screen.clear()
    sort_by = top[2].selectbox("Sort by", ["conviction", "chg_%", "vol_x_avg", "rsi"])

    with st.spinner("Screening NSE F&O universe…"):
        df = run_screen(interval)
    if df.empty:
        st.warning("No data — market may be closed or data unavailable.")
        st.stop()

    with st.sidebar:
        st.subheader("Fine filters")
        rsi_lo, rsi_hi = st.slider("RSI range", 0, 100, (0, 100))
        min_chg = st.slider("Min |Chg %|", 0.0, 10.0, 0.0, 0.25)
        min_vol = st.slider("Min Vol ×avg", 0.0, 5.0, 0.0, 0.25)
        sides = st.multiselect("Side", ["CALL", "PUT"], ["CALL", "PUT"])

    f = PRESETS[preset](df).copy()
    f = f[
        (f.rsi.between(rsi_lo, rsi_hi))
        & (f["chg_%"].abs() >= min_chg)
        & (f.vol_x_avg >= min_vol)
        & (f.side.isin(sides))
    ]
    if sort_by == "conviction":
        f = f.reindex(f["conviction"].abs().sort_values(ascending=False).index)
    else:
        f = f.sort_values(sort_by, ascending=False)

    m = st.columns(5)
    m[0].metric("Matches", len(f))
    m[1].metric("🟢 CALL", int((f.side == "CALL").sum()))
    m[2].metric("🔴 PUT", int((f.side == "PUT").sum()))
    m[3].metric("Avg Chg %", f"{f['chg_%'].mean():.2f}" if len(f) else "—")
    m[4].metric("Scanned", len(df))
    st.caption(f"Updated {datetime.now():%H:%M:%S} · {interval}-min bars · cached 2 min")

    st.dataframe(f, use_container_width=True, hide_index=True, column_config=COLCFG,
                 height=560)
    st.download_button("⬇ Download CSV", f.to_csv(index=False),
                       f"atlas_nse_{datetime.now():%Y%m%d_%H%M}.csv")


# ================================================================== CHARTS
elif page == "Charts":
    st.title("Stock / Index Charts")
    c1, c2 = st.columns([1, 1])
    kind = c1.radio("Type", ["Stock", "Index"], horizontal=True)
    if kind == "Index":
        name = c2.selectbox("Index", list(INDEX_TICKERS.keys()))
        ticker = INDEX_TICKERS[name]
    else:
        name = c2.selectbox("Stock", FNO_UNIVERSE)
        ticker = f"{name}.NS"

    ind = get_frame(ticker, days=5, interval=interval)
    if ind.empty:
        st.warning("No data for this symbol.")
    else:
        last = ind.iloc[-1]
        score = _dir_score(last)
        k = st.columns(5)
        k[0].metric("Price", f"₹{last['close']:.2f}")
        k[1].metric("Conviction", f"{score:.2f}", "CALL" if score > 0 else "PUT")
        k[2].metric("RSI", f"{last['rsi']:.1f}")
        k[3].metric("vs VWAP", f"{(last['close']/last['vwap']-1)*100:.2f}%")
        k[4].metric("ATR %", f"{last['atr_pct']:.2f}")
        st.plotly_chart(chart(name, ind), use_container_width=True)


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
            st.line_chart(trades["pnl"].cumsum())

    st.divider()
    st.subheader("Signal history")
    hist = load_signals()
    if hist.empty:
        st.info("Run the scanner or daily runner to build signal history.")
    else:
        st.caption(f"{len(hist)} signals logged")
        st.dataframe(hist.tail(30).iloc[::-1], use_container_width=True, hide_index=True)
