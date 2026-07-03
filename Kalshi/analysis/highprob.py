import numpy as np
import pandas as pd
from load_data import load

df = load()
df["CLOSE_TIME"] = pd.to_datetime(df["CLOSE_TIME"], utc=True)
d = df[df.RTI_INDEX_PRICE > 0].copy()
d["TS_UTC"] = d.TIMESTAMP.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
d["TTE"] = (d.CLOSE_TIME - d.TS_UTC).dt.total_seconds()
d = d[(d.TTE > 0) & (d.TTE <= 900)].copy()
d["YES_ASK"] = 1 - d.KALSHI_NO_BID
d["MID"] = (d.KALSHI_YES_BID + d.YES_ASK) / 2

# --- per-window realized outcome: 60s-avg RTI near close vs strike ---
outcome, dipped = {}, {}
for tk, w in d.groupby("TICKER"):
    tail = w[w.TTE <= 75]
    if len(tail) == 0:
        continue
    settle = tail.RTI_INDEX_PRICE.mean()
    outcome[tk] = 1 if settle >= w.STRIKE.iloc[0] else 0
d = d[d.TICKER.isin(outcome)].copy()
d["Y"] = d.TICKER.map(outcome)
print(f"windows: {len(outcome)}   YES-resolved: {sum(outcome.values())}  "
      f"NO-resolved: {len(outcome)-sum(outcome.values())}")

FEE = 0.02

# --- ROW-LEVEL: whenever price sits in a band, how often does it resolve YES? ---
print("\n=== ROW-LEVEL: when YES mid is in the band, what actually happens? ===")
print(f"{'band':>12} {'rows':>6} {'avg_price':>9} {'resolved_YES':>12} {'REVERSED(NO)':>12} {'buy&hold P&L':>13}")
bands = [(0.80, 0.90), (0.90, 0.95), (0.95, 1.001), (0.80, 1.001)]
for lo, hi in bands:
    m = (d.MID >= lo) & (d.MID < hi)
    s = d[m]
    if len(s) == 0:
        continue
    yes_rate = s.Y.mean()
    price = s.YES_ASK.mean()
    # buy YES at ask, hold to expiry, pay fee
    pnl = (s.Y - s.YES_ASK - FEE).mean()
    print(f"{f'{lo:.2f}-{hi if hi<=1 else 1.0:.2f}':>12} {len(s):>6} "
          f"{price:>9.3f} {yes_rate:>11.1%} {1-yes_rate:>11.1%} {100*pnl:>+11.2f}c")

# --- Intra-window reversal: after being >=X, does price later fall below 0.50? ---
print("\n=== INTRA-WINDOW: once price >= X, does it later dip below 0.50 in that window? ===")
for X in (0.80, 0.90, 0.95):
    hit = tot = 0
    for tk, w in d.groupby("TICKER"):
        w = w.sort_values("TS_UTC")
        mid = w.MID.values
        idx = np.where(mid >= X)[0]
        if len(idx) == 0:
            continue
        first = idx[0]
        tot += 1
        if (mid[first:] < 0.50).any():
            hit += 1
    print(f"  first time mid>={X:.2f}: {hit}/{tot} windows later dipped below 0.50 "
          f"({100*hit/tot:.0f}%)" if tot else f"  mid>={X}: no windows")

# --- TRADE-LEVEL: one buy per window at first cross above X, hold to expiry ---
print("\n=== TRADE-LEVEL: buy once per window at first mid>=X, hold to expiry ===")
for X in (0.80, 0.90, 0.95):
    pnls = []
    for tk, w in d.groupby("TICKER"):
        w = w.sort_values("TS_UTC")
        cross = w[w.MID >= X]
        if len(cross) == 0:
            continue
        r = cross.iloc[0]
        pnls.append(r.Y - r.YES_ASK - FEE)
    pnls = np.array(pnls)
    if len(pnls):
        print(f"  X>={X:.2f}: {len(pnls):2d} trades  win%={100*(pnls>0).mean():4.0f}  "
              f"avg P&L={100*pnls.mean():+.2f}c  total={100*pnls.sum():+.0f}c")
