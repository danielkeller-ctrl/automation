import numpy as np
import pandas as pd

from load_data import load
df = load()

print("=== OVERVIEW ===")
print(f"rows: {len(df)}   span: {df.TIMESTAMP.min()} -> {df.TIMESTAMP.max()}")
print(f"windows (distinct tickers): {df.TICKER.nunique()}")

# Data quality
zero_rti = (df.RTI_INDEX_PRICE == 0).sum()
print(f"rows with RTI==0 (feed warmup/stale): {zero_rti} ({100*zero_rti/len(df):.1f}%)")

# Mid price for Kalshi implied prob
df["KALSHI_MID"] = (df.KALSHI_YES_BID + (1 - df.KALSHI_NO_BID)) / 2
df["MONEYNESS"] = df.RTI_INDEX_PRICE - df.STRIKE  # spot - strike ($)

# Drop warmup/stale zero-RTI rows for analysis
d = df[df.RTI_INDEX_PRICE > 0].copy()

print("\n=== PER-WINDOW ROW COUNTS ===")
g = d.groupby("TICKER")
summ = g.agg(rows=("TIMESTAMP", "size"),
             start=("TIMESTAMP", "min"),
             end=("TIMESTAMP", "max"),
             strike=("STRIKE", "first"))
summ["dur_min"] = (summ.end - summ.start).dt.total_seconds() / 60
print(summ[["rows", "dur_min", "strike"]].to_string())

# --- Moneyness vs Kalshi price (sanity: does the market price track spot-vs-strike?) ---
corr_money = d["MONEYNESS"].corr(d["KALSHI_MID"])
print(f"\n=== SANITY: corr(moneyness, Kalshi mid) = {corr_money:+.3f} ===")
print("(should be strongly positive: spot above strike -> YES more likely)")

# --- LEAD/LAG: does spot lead the Kalshi price? ---
# Resample each window to a regular 1s grid, then cross-correlate
# spot change at t with Kalshi mid change at t+lag.
print("\n=== LEAD/LAG (spot change_t vs Kalshi change_{t+lag}) ===")
print("lag>0 means SPOT LEADS Kalshi by that many seconds (the edge).")

lags = range(-5, 11)
agg = {lag: [] for lag in lags}

for tk, w in d.groupby("TICKER"):
    w = w.set_index("TIMESTAMP").sort_index()
    # regular 1s grid, forward-fill short gaps
    w = w[~w.index.duplicated(keep="last")]
    grid = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="1s")).ffill(limit=5)
    spot = grid.RTI_INDEX_PRICE.diff()
    kal = grid.KALSHI_MID.diff()
    if spot.notna().sum() < 60:
        continue
    for lag in lags:
        c = spot.corr(kal.shift(-lag))
        if not np.isnan(c):
            agg[lag].append(c)

print(f"{'lag(s)':>7} {'mean_corr':>10} {'windows':>8}")
best_lag, best_val = None, -1
for lag in lags:
    vals = agg[lag]
    if not vals:
        continue
    m = float(np.mean(vals))
    print(f"{lag:>7} {m:>+10.3f} {len(vals):>8}")
    if m > best_val:
        best_val, best_lag = m, lag
print(f"\nPeak mean correlation at lag = {best_lag}s (corr={best_val:+.3f})")
