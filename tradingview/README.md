# Atlas on TradingView

`atlas_alpha.pine` is the Atlas alpha engine (`engine/alpha_signal.py`) ported to
Pine Script v6, so you can run the same entry-timing logic as a custom indicator
on any TradingView chart — with TradingView's real-time data instead of
yfinance's delayed feed.

## Install

1. TradingView → **Pine Editor** (bottom panel) → **Open → New indicator**.
2. Paste the whole of `atlas_alpha.pine`, replacing the template.
3. **Save**, then **Add to chart**.
4. Set the chart to **5 minutes** on an NSE symbol (e.g. `NSE:RELIANCE`).
5. For a bank/financial name, open the indicator settings and change
   **Index** to `NSE:BANKNIFTY` (the Python side does this automatically from
   its `_BANKING` list; Pine can't, so it's an input).

To get alerts: **Alert → Condition → Atlas Alpha → "Atlas ENTER long (CALL)"**
(or the short/PUT one). It fires on the bar where the status flips to ENTER.

## What it shows

- **Panel** (top right): status (ENTER / WAIT / AVOID), grade + aligned votes,
  win probability, 15m trend, ADX, index change, relative strength, session
  phase, and the entry / stop / 1R-2R-3R levels.
- **Lines**: entry (blue), ATR stop (red dashed), three targets (green dotted).
  Drawn whenever the status isn't AVOID.
- **Triangles + background tint** on bars where the status becomes ENTER.
- **Plots**: EMA 9, EMA 15, VWAP.

All of the gates are inputs, so you can loosen the ADX floor, the grade
minimum, the extension cap, or the time-of-day window and see the effect
immediately on history.

## Win-probability model

The logistic-regression coefficients from `data_store/prob_model.json` are
inlined in the script — Pine can't read files. After retraining:

```bash
python -m engine.prob_model        # retrain, rewrites prob_model.json
python -m tradingview.export_model # rewrite the constants in atlas_alpha.pine
```

Then re-paste the script into TradingView. The port was checked against the
Python model over random feature vectors: max probability deviation ~1.5e-10.

## Known differences from the Python engine

These are small but real — don't expect the two to agree to the last digit:

- **Data**: TradingView uses exchange real-time data; the app uses yfinance 5m
  bars patched with an NSE LTP quote. Bar boundaries and volume can differ.
- **15m gate timing**: Pine requests the higher timeframe with
  `lookahead_off`, so it uses the last *closed* 15m bar and never repaints —
  `_htf_bias()` in Python reads the forming 15m bar. On a bar where the 15m
  trend has just flipped, Pine can lag the app by up to one 15m bar. That's
  deliberate: a repainting gate would show entries on history that never
  existed live.
- **Index change**: Pine measures the index from the daily open;
  `data/market_context.py` measures it from ~75 bars back. Close, not identical.
- **BANKNIFTY routing** is manual here (see install step 5).
- **Option premium projection** (the read-only ATM-delta panel in the web app)
  is not ported — TradingView has no NSE option-chain access from Pine.
- Pine's `ta.atr` / `ta.dmi` use RMA smoothing, matching `pandas_ta`; EMA, RSI,
  MACD and session VWAP match too, so the confluence votes line up.

Same honest framing as the app: this is a **selectivity** tool. It filters out
most setups; it does not predict direction. Training accuracy of the win-prob
model is ~0.52 against a 0.49 base rate.
