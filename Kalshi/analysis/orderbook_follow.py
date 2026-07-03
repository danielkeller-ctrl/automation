"""Follow large resting orders as they appear -- do they lead price, or spoof?

Signal: a large net increase in depth on one side (YES vs NO) in one second.
Follow it (buy that side). Measure (1) forward price move in the followed
direction, (2) whether the added depth persists or gets pulled (spoof tell),
(3) tradeable P&L after costs.
"""

import numpy as np
import pandas as pd
from load_data import load

PCTL = 95          # "large" = top 5% of |net depth change| per second
FEE = 0.02

df = load()
d = df[df.RTI_INDEX_PRICE > 0].copy()
d["YES_ASK"] = 1 - d.KALSHI_NO_BID
d["YES_BID"] = d.KALSHI_YES_BID
d["MID"] = (d.YES_BID + d.YES_ASK) / 2
yes_cols = [c for c in d.columns if c.startswith("YES - ")]
no_cols = [c for c in d.columns if c.startswith("NO - ")]
d["YES_DEPTH"] = d[yes_cols].sum(axis=1)
d["NO_DEPTH"] = d[no_cols].sum(axis=1)

# Build per-window 1s grids
grids = []
for tk, w in d.groupby("TICKER"):
    w = w.set_index("TIMESTAMP").sort_index()
    w = w[~w.index.duplicated(keep="last")]
    g = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="1s")).ffill(limit=5)
    g = g.dropna(subset=["MID", "YES_DEPTH", "NO_DEPTH", "YES_ASK", "YES_BID"])
    if len(g) > 60:
        g["dYES"] = g.YES_DEPTH.diff()
        g["dNO"] = g.NO_DEPTH.diff()
        g["net"] = g.dYES - g.dNO   # >0: net YES depth added; <0: net NO added
        grids.append(g)

net_all = np.concatenate([g.net.dropna().values for g in grids])
thr = np.percentile(np.abs(net_all), PCTL)
print(f"windows={len(grids)}  1s obs={len(net_all)}  large-order threshold |net|>={thr:.0f} contracts\n")

# ---------- 1. EVENT STUDY: forward price move when a large order appears ----------
print("=== FOLLOW large orders: forward MID move in the followed direction ===")
print("(signal = sign of net depth added; + move => following the order pays)")
H = [1, 3, 5, 10, 30]
agg = {h: [] for h in H}
hitrate = {h: [0, 0] for h in H}
for g in grids:
    mid = g.MID.values; net = g.net.values
    n = len(g)
    for i in range(1, n):
        if abs(net[i]) < thr:
            continue
        s = np.sign(net[i])
        for h in H:
            if i + h < n:
                fwd = mid[i + h] - mid[i]
                agg[h].append(s * fwd)
                hitrate[h][0] += 1 if np.sign(fwd) == s else 0
                hitrate[h][1] += 1
for h in H:
    a = np.array(agg[h]); hits, tot = hitrate[h]
    print(f"  +{h:>2}s: mean_followed_move={100*a.mean():+.3f}c  dir.acc={100*hits/tot:4.1f}%  n={tot}")

# ---------- 2. SPOOF TELL: does the added depth persist, and where does price go? ----------
print("\n=== SPOOF DIAGNOSTIC (large YES-add vs large NO-add events) ===")
for side, sign in (("large YES-add", +1), ("large NO-add", -1)):
    persist, moves = [], []
    n_ev = 0
    for g in grids:
        yd, nd = g.YES_DEPTH.values, g.NO_DEPTH.values
        net = g.net.values; mid = g.MID.values
        n = len(g)
        for i in range(1, n - 10):
            if np.sign(net[i]) != sign or abs(net[i]) < thr:
                continue
            n_ev += 1
            added = abs(net[i])
            depth = yd if sign > 0 else nd
            # fraction of the added size still present 5s later
            still = (depth[i + 5] - depth[i - 1]) / added if added > 0 else np.nan
            persist.append(np.clip(still, -1, 2))
            moves.append(mid[i + 5] - mid[i])          # raw MID move (YES perspective)
    persist = np.array(persist); moves = np.array(moves)
    exp = "MID up" if sign > 0 else "MID down"
    print(f"  {side}: n={n_ev}  depth still there @+5s={100*np.nanmean(persist):4.0f}%  "
          f"MID move @+5s={100*moves.mean():+.3f}c  (informed => {exp})")

# ---------- 3. TRADEABLE: follow the order, hold, exit at bid, net of costs ----------
print("\n=== TRADEABLE: follow large order, hold H sec, exit at bid, net of fees ===")
for h in (5, 10, 30):
    pnl = []
    for g in grids:
        net = g.net.values
        ask, bid = g.YES_ASK.values, g.YES_BID.values
        n = len(g); i = 1
        while i < n - h:
            if abs(net[i]) >= thr:
                if net[i] > 0:   # follow YES
                    p = bid[i + h] - ask[i] - FEE
                else:            # follow NO
                    p = bid[i] - ask[i + h] - FEE
                pnl.append(p)
                i += h            # non-overlapping
            else:
                i += 1
    pnl = np.array(pnl)
    print(f"  hold {h:>2}s: n={len(pnl):4d}  win%={100*(pnl>0).mean():4.1f}  "
          f"avgP&L={100*pnl.mean():+.2f}c  total={100*pnl.sum():+.0f}c")
