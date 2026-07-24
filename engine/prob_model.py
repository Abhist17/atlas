"""Win-probability model — a calibrated conviction number.

Instead of hand-weighting the factors, we learn from history: at every past bar
we take the factor snapshot and label it 1 if a 1R move happened before a 1R
stop (same session), else 0. A logistic regression (pure NumPy, no heavy deps)
maps the factor vector to P(win). Live, we feed the current snapshot in and get
a calibrated probability — "setups like this won ~X% of the time."

This is honest about uncertainty: it does not predict direction, it estimates
how often a given *setup* has paid off. Retrain any time with:

    python -m engine.prob_model            # train + save
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from config.settings import DATA_STORE
from data.yf_client import yfc
from engine.directional import add_opening_range
from engine.indicators import add_indicators
from utils.logger import get_logger

log = get_logger("engine.prob_model")

MODEL_FILE = DATA_STORE / "prob_model.json"
FEATURES = ["trend_str", "macd", "rsi", "vwap", "adx", "vol", "ext", "htf"]
_ATR_STOP = 1.3          # risk unit (must match alpha_signal)

# symbols used to train (liquid, diverse) — training is offline/occasional
TRAIN_SYMBOLS = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK",
    "KOTAKBANK", "ITC", "LT", "BHARTIARTL", "HINDUNILVR", "MARUTI", "TITAN",
    "SUNPHARMA", "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "ONGC", "NTPC",
    "POWERGRID", "COALINDIA", "BAJFINANCE", "ADANIENT", "WIPRO", "TECHM",
    "HCLTECH", "ULTRACEMCO", "GRASIM", "CIPLA", "DRREDDY", "BPCL",
]

_model: dict | None = None


# ------------------------------------------------------------------ features
def feature_vector(p: dict) -> list[float]:
    """Direction-relative factor vector (so the model learns 'does it win')."""
    d = 1.0 if p["ema9"] >= p["ema15"] else -1.0
    atr = max(float(p["atr"]), 1e-6)
    return [
        abs(p["ema9"] - p["ema15"]) / atr,               # trend separation
        d * float(p.get("macd_hist", 0.0)) / atr,        # macd agreement
        d * (float(p["rsi"]) - 50.0) / 10.0,             # momentum agreement
        d * (p["close"] - p["vwap"]) / atr,              # vwap side
        float(p["adx"]) / 25.0,                          # regime strength
        min(float(p["vol_x"]), 3.0),                     # volume
        d * (p["close"] - p["ema15"]) / atr,             # extension in dir
        d * (p["close"] - p["ema50"]) / atr,             # longer-trend agreement
    ]


# ------------------------------------------------------------------ training
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _label_bars(ind) -> tuple[list, list]:
    """Build (features, label) pairs: 1 if +1R before -1R that session."""
    close = ind["close"].to_numpy()
    high = ind["high"].to_numpy()
    low = ind["low"].to_numpy()
    atr = ind["atr"].to_numpy()
    date = np.array([str(t)[:10] for t in ind["timestamp"]])
    X, y = [], []
    n = len(ind)
    for i in range(n - 1):
        a = float(atr[i])
        if a <= 0 or np.isnan(a):
            continue
        row = ind.iloc[i]
        if np.isnan(row.get("ema50", np.nan)) or np.isnan(row.get("adx", np.nan)):
            continue
        d = 1.0 if row["ema9"] >= row["ema15"] else -1.0
        risk = _ATR_STOP * a
        entry = float(close[i])
        tgt, stp = entry + d * risk, entry - d * risk
        outcome = None
        j = i + 1
        while j < n and date[j] == date[i]:
            hi, lo = float(high[j]), float(low[j])
            if d > 0:
                if lo <= stp:
                    outcome = 0; break
                if hi >= tgt:
                    outcome = 1; break
            else:
                if hi >= stp:
                    outcome = 0; break
                if lo <= tgt:
                    outcome = 1; break
            j += 1
        if outcome is None:      # unresolved by session end — drop (clean labels)
            continue
        prim = {"close": entry, "ema9": float(row["ema9"]), "ema15": float(row["ema15"]),
                "ema50": float(row["ema50"]), "atr": a, "macd_hist": float(row.get("macd_hist", 0.0)),
                "rsi": float(row["rsi"]) if not np.isnan(row["rsi"]) else 50.0,
                "vwap": float(row["vwap"]), "adx": float(row["adx"]),
                "vol_x": float(row["volume"] / row["avg_volume"]) if row["avg_volume"] else 0.0}
        X.append(feature_vector(prim))
        y.append(outcome)
    return X, y


def train_and_save(symbols=None, days: int = 30) -> dict:
    symbols = symbols or TRAIN_SYMBOLS
    log.info("Training win-prob model on %d symbols (%dd)…", len(symbols), days)
    data = yfc.batch_intraday([f"{s}.NS" for s in symbols], days=days, interval=5)
    X, y = [], []
    for s in symbols:
        df = data.get(f"{s}.NS")
        if df is None or len(df) < 60:
            continue
        ind = add_opening_range(add_indicators(df)).dropna(
            subset=["ema9", "ema15", "ema50", "atr", "vwap", "rsi", "adx", "avg_volume"]).reset_index(drop=True)
        if len(ind) < 60:
            continue
        xs, ys = _label_bars(ind)
        X += xs; y += ys
    X = np.array(X, float); y = np.array(y, float)
    if len(y) < 200:
        raise RuntimeError(f"Not enough labeled samples ({len(y)}).")

    mean, std = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mean) / std
    w, b = np.zeros(X.shape[1]), 0.0
    lr, lam, n = 0.2, 1e-3, len(y)
    for _ in range(4000):
        p = _sigmoid(Xs @ w + b)
        g = p - y
        w -= lr * (Xs.T @ g / n + lam * w)
        b -= lr * g.mean()

    pred = _sigmoid(Xs @ w + b)
    acc = float(((pred > 0.5) == (y > 0.5)).mean())
    model = {
        "features": FEATURES, "mean": mean.tolist(), "std": std.tolist(),
        "weights": w.tolist(), "bias": float(b),
        "n_samples": int(n), "base_rate": round(float(y.mean()), 4),
        "train_acc": round(acc, 4), "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    MODEL_FILE.write_text(json.dumps(model, indent=2))
    log.info("Model saved: %d samples, base_rate %.3f, acc %.3f", n, y.mean(), acc)
    return model


# ------------------------------------------------------------------ inference
def load_model() -> dict | None:
    global _model
    if _model is None and MODEL_FILE.exists():
        try:
            _model = json.loads(MODEL_FILE.read_text())
        except Exception as e:
            log.error("model load failed: %s", e)
    return _model


def win_probability(prim: dict) -> float | None:
    """P(win) for the current factor snapshot, or None if no model."""
    m = load_model()
    if not m:
        return None
    x = np.array(feature_vector(prim), float)
    xs = (x - np.array(m["mean"])) / np.array(m["std"])
    z = float(np.dot(xs, m["weights"]) + m["bias"])
    return round(float(_sigmoid(z)), 3)


if __name__ == "__main__":
    print(json.dumps(train_and_save(), indent=2))
