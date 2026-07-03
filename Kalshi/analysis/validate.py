import numpy as np
import pandas as pd
from scipy.stats import norm

from load_data import load
df = load()
df["CLOSE_TIME"] = pd.to_datetime(df["CLOSE_TIME"], utc=True)
d = df[df.RTI_INDEX_PRICE > 0].copy()
d["TS_UTC"] = d.TIMESTAMP.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
d["TTE"] = (d.CLOSE_TIME - d.TS_UTC).dt.total_seconds()
d = d[(d.TTE > 0) & (d.TTE <= 900)].copy()
d["MID"] = ((1 - d.KALSHI_NO_BID) + d.KALSHI_YES_BID) / 2

# --- Realized outcome per window: 60s-avg RTI near close vs strike ---
outcome = {}
for tk, w in d.groupby("TICKER"):
    tail = w[w.TTE <= 75]
    if len(tail) == 0:
        continue
    settle = tail.RTI_INDEX_PRICE.mean()
    outcome[tk] = 1 if settle >= w.STRIKE.iloc[0] else 0
d = d[d.TICKER.isin(outcome)].copy()
d["Y"] = d.TICKER.map(outcome)
print(f"windows with known outcome: {len(outcome)}  (YES resolved: {sum(outcome.values())})")

# --- Model fair prob ---
rets = []
for tk, w in d.groupby("TICKER"):
    w = w.sort_values("TS_UTC")
    lr = np.log(w.RTI_INDEX_PRICE).diff()
    dt = w.TS_UTC.diff().dt.total_seconds()
    rets.extend(lr[dt <= 3].dropna().tolist())
sigma_s = np.std(rets)
d["FAIR"] = norm.cdf(np.log(d.RTI_INDEX_PRICE / d.STRIKE) / (sigma_s * np.sqrt(d.TTE)))

# --- Brier scores (lower = better predictor of the actual outcome) ---
brier_k = np.mean((d.MID - d.Y) ** 2)
brier_m = np.mean((d.FAIR - d.Y) ** 2)
print(f"\nBrier score  Kalshi mid : {brier_k:.4f}")
print(f"Brier score  model fair: {brier_m:.4f}")
print("-> lower wins. If Kalshi is lower, the 'edges' are model error (don't trade).")

# --- Reliability: does Kalshi mid match realized YES rate? ---
print("\n=== CALIBRATION: predicted prob vs actual YES rate ===")
d["kbin"] = pd.cut(d.MID, np.arange(0, 1.01, 0.1))
rel = d.groupby("kbin", observed=True).agg(kalshi=("MID", "mean"),
                                           model=("FAIR", "mean"),
                                           actual=("Y", "mean"),
                                           n=("Y", "size"))
print((rel * [1, 1, 1, 1]).round(3).to_string())

# --- Hold-to-expiry backtest: trade toward the model, ONE trade per window ---
# (independent outcomes; costs = cross spread on entry + ~2c fee)
FEE = 0.02
for thr in (0.05, 0.10, 0.15):
    trades = []
    for tk, w in d.groupby("TICKER"):
        w = w.sort_values("TS_UTC")
        edge = w.MID - w.FAIR
        sig = w[edge.abs() > thr]
        if len(sig) == 0:
            continue
        r = sig.iloc[0]
        y = r.Y
        if r.MID < r.FAIR:          # Kalshi too cheap on YES -> buy YES at ask
            entry = 1 - r.KALSHI_NO_BID
            pnl = y - entry - FEE
        else:                        # Kalshi too rich on YES -> buy NO at ask
            entry = 1 - r.KALSHI_YES_BID
            pnl = (1 - y) - entry - FEE
        trades.append(pnl)
    trades = np.array(trades)
    if len(trades):
        print(f"\nthr={thr:.2f}: {len(trades)} trades  win%={100*(trades>0).mean():.0f}  "
              f"avg P&L={100*trades.mean():+.1f}c/trade  total={100*trades.sum():+.0f}c")
