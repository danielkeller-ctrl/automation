import numpy as np
import pandas as pd

from load_data import load
df = load()
df["CLOSE_TIME"] = pd.to_datetime(df["CLOSE_TIME"], utc=True)
df["KALSHI_MID"] = (df.KALSHI_YES_BID + (1 - df.KALSHI_NO_BID)) / 2
d = df[df.RTI_INDEX_PRICE > 0].copy()

# seconds to expiry. TIMESTAMP is naive LOCAL time (America/New_York, EDT);
# CLOSE_TIME is UTC. Convert local -> UTC before differencing.
d["TS_UTC"] = d.TIMESTAMP.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
d["TTE"] = (d.CLOSE_TIME - d.TS_UTC).dt.total_seconds()
print("TTE sanity: min=%.0fs max=%.0fs (should be ~0..900)" % (d.TTE.min(), d.TTE.max()))

# Build per-window 1s grids once
grids = []
for tk, w in d.groupby("TICKER"):
    w = w.set_index("TS_UTC").sort_index()
    w = w[~w.index.duplicated(keep="last")]
    grid = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="1s")).ffill(limit=5)
    grid["d_spot"] = grid.RTI_INDEX_PRICE.diff()
    grid["d_kal"] = grid.KALSHI_MID.diff()
    grids.append(grid)

# ---- 1. Lead/lag conditioned on time-to-expiry ----
print("=== LEAD/LAG BY TIME-TO-EXPIRY ===")
print("lag>0 = spot leads Kalshi. (contemporaneous=lag 0)")
bins = [("early (>=5min left)", lambda g: g.TTE >= 300),
        ("mid (2-5min)",        lambda g: (g.TTE >= 120) & (g.TTE < 300)),
        ("late (<2min)",        lambda g: g.TTE < 120)]
lags = [-2, -1, 0, 1, 2, 3]
for name, cond in bins:
    row = {}
    for lag in lags:
        cs = []
        for g in grids:
            m = cond(g)
            s = g.d_spot[m]; k = g.d_kal.shift(-lag)[m]
            if s.notna().sum() > 30:
                c = s.corr(k)
                if not np.isnan(c):
                    cs.append(c)
        row[lag] = np.mean(cs) if cs else np.nan
    print(f"  {name:22} " + "  ".join(f"lag{lag:+d}:{row[lag]:+.3f}" for lag in lags))

# ---- 2. Event study: response to large spot jumps ----
print("\n=== EVENT STUDY: Kalshi response to large 1s spot moves ===")
all_ds = np.concatenate([g.d_spot.dropna().values for g in grids])
thr = np.nanpercentile(np.abs(all_ds), 98)  # top 2% moves
print(f"large-move threshold (98th pct |dspot|): ${thr:.2f}/s")

# For each large spot move at t (sign s), measure cumulative signed Kalshi
# move from t-1 baseline through t+H. If Kalshi keeps moving after t, there's
# a reaction window to exploit.
H = 4
cum = {h: [] for h in range(0, H + 1)}
n_events = 0
for g in grids:
    ds = g.d_spot.values
    kmid = g.KALSHI_MID.values
    for i in range(1, len(g) - H):
        if np.isnan(ds[i]) or abs(ds[i]) < thr:
            continue
        sign = np.sign(ds[i])
        base = kmid[i - 1]
        if np.isnan(base):
            continue
        ok = True
        vals = []
        for h in range(0, H + 1):
            v = kmid[i + h]
            if np.isnan(v):
                ok = False
                break
            vals.append(sign * (v - base))
        if ok:
            n_events += 1
            for h in range(0, H + 1):
                cum[h].append(vals[h])
print(f"events: {n_events}")
print("cumulative signed Kalshi move from t-1 baseline (cents), by seconds after the jump:")
for h in range(0, H + 1):
    arr = np.array(cum[h])
    print(f"  t{h:+d}: mean={100*np.mean(arr):+.2f}c  median={100*np.median(arr):+.2f}c")
# fraction of total move already done at t (the jump second)
final = np.mean(cum[H])
at_t = np.mean(cum[0])
if final != 0:
    print(f"\nFraction of {H}s move already priced by the jump second (t+0): {at_t/final*100:.0f}%")
