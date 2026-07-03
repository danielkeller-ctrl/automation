"""Out-of-sample test of the high-cushion favorite filter.

Split windows chronologically: TRAIN (first half) fixes sigma and the z
threshold; TEST (second half) is scored with those frozen values -- no peeking.
If high-cushion favorites beat low-cushion in TEST too, the edge is real;
if it evaporates, it was in-sample fitting.
"""

import numpy as np
import pandas as pd
from load_data import load

BAND = (0.80, 0.90)
def fee(p):
    return np.ceil(0.07 * p * (1 - p) * 100) / 100

df = load()
df["CLOSE_TIME"] = pd.to_datetime(df["CLOSE_TIME"], utc=True)
d = df[df.RTI_INDEX_PRICE > 0].copy()
d["TS_UTC"] = d.TIMESTAMP.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
d["TTE"] = (d.CLOSE_TIME - d.TS_UTC).dt.total_seconds()
d = d[(d.TTE > 0) & (d.TTE <= 900)].copy()
d["YES_ASK"] = 1 - d.KALSHI_NO_BID
d["MID"] = (d.KALSHI_YES_BID + d.YES_ASK) / 2

outcome, first_ts = {}, {}
for tk, w in d.groupby("TICKER"):
    tail = w[w.TTE <= 75]
    if len(tail):
        outcome[tk] = 1 if tail.RTI_INDEX_PRICE.mean() >= w.STRIKE.iloc[0] else 0
        first_ts[tk] = w.TS_UTC.min()

# chronological split of windows
order = sorted(first_ts, key=lambda t: first_ts[t])
half = len(order) // 2
train_tk, test_tk = set(order[:half]), set(order[half:])

def sigma_from(tks):
    r = []
    for tk in tks:
        w = d[d.TICKER == tk].sort_values("TS_UTC")
        lr = np.log(w.RTI_INDEX_PRICE).diff(); dt = w.TS_UTC.diff().dt.total_seconds()
        r.extend(lr[dt <= 3].dropna().tolist())
    return np.std(r)

sigma_train = sigma_from(train_tk)   # freeze vol from TRAIN only

def entries(tks, sigma):
    rows = []
    for tk in tks:
        if tk not in outcome:
            continue
        w = d[d.TICKER == tk].sort_values("TS_UTC").reset_index(drop=True)
        hit = w.index[(w.MID >= BAND[0]) & (w.MID < BAND[1])]
        if len(hit) == 0:
            continue
        i = hit[0]; r = w.loc[i]
        z = np.log(r.RTI_INDEX_PRICE / r.STRIKE) / (sigma * np.sqrt(r.TTE)) if r.TTE > 0 else np.nan
        pnl = outcome[tk] - r.YES_ASK - fee(r.YES_ASK)
        rows.append(dict(z=z, pnl=pnl))
    return pd.DataFrame(rows).dropna()

tr = entries(train_tk, sigma_train)
te = entries(test_tk, sigma_train)
z_thr = tr.z.median()                # freeze threshold from TRAIN only

print(f"windows: train={len(train_tk)} test={len(test_tk)}")
print(f"frozen from TRAIN:  sigma={sigma_train:.2e}  z_threshold={z_thr:.3f}\n")

def show(tag, f):
    hi, lo = f[f.z >= z_thr], f[f.z < z_thr]
    for name, s in (("all", f), ("high cushion", hi), ("low cushion", lo)):
        if len(s):
            print(f"  {tag} {name:14} n={len(s):3d}  win%={100*(s.pnl>0).mean():4.0f}  "
                  f"avgP&L={100*s.pnl.mean():+6.2f}c  total={100*s.pnl.sum():+6.0f}c")

print("IN-SAMPLE (train):")
show("train", tr)
print("\nOUT-OF-SAMPLE (test) -- the honest number:")
show("test ", te)
