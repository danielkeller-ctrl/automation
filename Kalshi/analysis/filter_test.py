"""Does a cushion / momentum filter salvage the favorite (0.80-0.90) bet?

Enter once per window at first mid in the band, buy YES at ask, hold to expiry.
Compare P&L of the full set vs subsets filtered on the (pre-registered)
reversal-risk features. Small samples -> read as directional, not proof.
"""

import numpy as np
import pandas as pd
from load_data import load

BAND = (0.80, 0.90)
FEE = 0.02

df = load()
df["CLOSE_TIME"] = pd.to_datetime(df["CLOSE_TIME"], utc=True)
d = df[df.RTI_INDEX_PRICE > 0].copy()
d["TS_UTC"] = d.TIMESTAMP.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
d["TTE"] = (d.CLOSE_TIME - d.TS_UTC).dt.total_seconds()
d = d[(d.TTE > 0) & (d.TTE <= 900)].copy()
d["YES_ASK"] = 1 - d.KALSHI_NO_BID
d["MID"] = (d.KALSHI_YES_BID + d.YES_ASK) / 2

allr = []
for _, w in d.groupby("TICKER"):
    w = w.sort_values("TS_UTC")
    lr = np.log(w.RTI_INDEX_PRICE).diff(); dt = w.TS_UTC.diff().dt.total_seconds()
    allr.extend(lr[dt <= 3].dropna().tolist())
sigma_s = np.std(allr)

outcome = {}
for tk, w in d.groupby("TICKER"):
    tail = w[w.TTE <= 75]
    if len(tail):
        outcome[tk] = 1 if tail.RTI_INDEX_PRICE.mean() >= w.STRIKE.iloc[0] else 0

rows = []
for tk, w in d.groupby("TICKER"):
    if tk not in outcome:
        continue
    w = w.sort_values("TS_UTC").reset_index(drop=True)
    hit = w.index[(w.MID >= BAND[0]) & (w.MID < BAND[1])]
    if len(hit) == 0:
        continue
    i = hit[0]; r = w.loc[i]
    past = w.RTI_INDEX_PRICE.iloc[max(0, i-30):i+1]
    spot_mom = (r.RTI_INDEX_PRICE - past.iloc[0]) / past.iloc[0] if len(past) > 1 else 0.0
    z = np.log(r.RTI_INDEX_PRICE / r.STRIKE) / (sigma_s * np.sqrt(r.TTE)) if r.TTE > 0 else np.nan
    pnl = outcome[tk] - r.YES_ASK - FEE
    rows.append(dict(z=z, spot_mom=spot_mom, pnl=pnl, y=outcome[tk]))

f = pd.DataFrame(rows).dropna()
zmed = f.z.median()

def show(name, mask):
    s = f[mask]
    if len(s) == 0:
        print(f"  {name:34} n= 0"); return
    print(f"  {name:34} n={len(s):3d}  win%={100*(s.y>0).mean():4.0f}  "
          f"avgP&L={100*s.pnl.mean():+6.2f}c  total={100*s.pnl.sum():+6.0f}c")

print(f"favorite entries in {BAND}: {len(f)}  (z median={zmed:.2f})\n")
show("ALL (no filter)", f.index >= 0)
print("  -- filter: cushion --")
show(f"high cushion (z >= {zmed:.2f})", f.z >= zmed)
show(f"low cushion  (z <  {zmed:.2f})", f.z < zmed)
print("  -- filter: spot momentum --")
show("spot_mom >= 0 (rising)", f.spot_mom >= 0)
show("spot_mom <  0 (falling)", f.spot_mom < 0)
print("  -- combined (pre-registered 'safe') --")
show("z>=median AND spot_mom>=0", (f.z >= zmed) & (f.spot_mom >= 0))
show("z<median  AND spot_mom<0", (f.z < zmed) & (f.spot_mom < 0))
