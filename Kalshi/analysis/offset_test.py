"""Test the 'offset' idea: long YES at ~0.86, and if a reversal drives NO up to
a trigger price, buy NO (optionally larger size) to offset. Hold both to expiry.

Does adding the offset improve P&L, or does it just add a negative-EV bet plus
blow-up risk when the reversal itself reverses?
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
d["YES_BID"] = d.KALSHI_YES_BID
d["MID"] = (d.YES_BID + d.YES_ASK) / 2

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

# Build high-cushion favorite entries with their forward NO-ask path
entries = []
for tk, w in d.groupby("TICKER"):
    if tk not in outcome:
        continue
    w = w.set_index("TS_UTC").sort_index()
    w = w[~w.index.duplicated(keep="last")]
    g = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="1s")).ffill(limit=5)
    g = g.dropna(subset=["MID", "YES_ASK", "YES_BID", "RTI_INDEX_PRICE"])
    hit = np.where((g.MID.values >= BAND[0]) & (g.MID.values < BAND[1]))[0]
    if len(hit) == 0:
        continue
    i = hit[0]
    tte_i = g.TTE.values[i]
    z = np.log(g.RTI_INDEX_PRICE.values[i] / g.STRIKE.values[i]) / (sigma_s * np.sqrt(tte_i)) if tte_i > 0 else np.nan
    entries.append(dict(
        y=outcome[tk], entry_yes=g.YES_ASK.values[i], z=z,
        no_ask_fwd=(1 - g.YES_BID.values[i + 1:]),  # NO ask path after entry
    ))

E = pd.DataFrame(entries).dropna(subset=["z"])
zmed = E.z.median()
hi = E[E.z >= zmed]
print(f"favorite entries={len(E)}, high-cushion (z>={zmed:.2f})={len(hi)}\n")

def sim(subset, trigger, size_mode):
    pnls, triggered, rereversed, worst = [], 0, 0, 0.0
    for r in subset.itertuples():
        pnl_yes = r.y - r.entry_yes - fee(r.entry_yes)
        pnl_no = 0.0
        path = r.no_ask_fwd
        idx = np.where(path >= trigger)[0]
        if len(idx):
            triggered += 1
            no_entry = path[idx[0]]
            if size_mode == "1x":
                M = 1.0
            else:  # breakeven: size NO so a NO win offsets the YES loss
                M = r.entry_yes / max(1 - no_entry, 0.01)
            pnl_no = M * ((1 - r.y) - no_entry - fee(no_entry))
            if r.y == 1:            # YES won after we bought NO -> offset backfired
                rereversed += 1
        total = pnl_yes + pnl_no
        worst = min(worst, total)
        pnls.append(total)
    pnls = np.array(pnls)
    tr = f"{triggered:2d}" if triggered else " 0"
    rr = f"{rereversed}/{triggered}" if triggered else "-"
    print(f"  trigger NO>={trigger:.2f} size={size_mode:9} "
          f"avgP&L={100*pnls.mean():+7.2f}c  win%={100*(pnls>0).mean():4.0f}  "
          f"worst={100*worst:+8.0f}c  triggered={tr}  backfired={rr}")

for name, sub in (("HIGH-CUSHION favorites", hi), ("ALL favorites", E)):
    print(f"=== {name} (n={len(sub)}) ===")
    base = np.array([r.y - r.entry_yes - fee(r.entry_yes) for r in sub.itertuples()])
    print(f"  BASELINE (no offset)          "
          f"avgP&L={100*base.mean():+7.2f}c  win%={100*(base>0).mean():4.0f}  "
          f"worst={100*base.min():+8.0f}c")
    for trig in (0.70, 0.90):
        for mode in ("1x", "breakeven"):
            sim(sub, trig, mode)
    print()
