"""Reversal-risk feature study for high-probability (favorite) YES bets.

For every window that trades up into the favorite band, snapshot candidate
risk features at the entry moment, then label whether the bet reversed
(resolved NO, or dipped below 0.50 intra-window). As reversal events
accumulate over multi-day collection, compare features of reversed vs
survived favorites to see if technicals + order book flag the fragile ones.

No pipeline change needed -- all features derive from logged columns.
"""

import numpy as np
import pandas as pd
from load_data import load

ENTRY = 0.86          # favorite band center we care about
BAND = (0.83, 0.90)   # count entries whose mid falls in here

df = load()
df["CLOSE_TIME"] = pd.to_datetime(df["CLOSE_TIME"], utc=True)
d = df[df.RTI_INDEX_PRICE > 0].copy()
d["TS_UTC"] = d.TIMESTAMP.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
d["TTE"] = (d.CLOSE_TIME - d.TS_UTC).dt.total_seconds()
d = d[(d.TTE > 0) & (d.TTE <= 900)].copy()
d["YES_ASK"] = 1 - d.KALSHI_NO_BID
d["MID"] = (d.KALSHI_YES_BID + d.YES_ASK) / 2
yes_cols = [c for c in d.columns if c.startswith("YES - ")]
no_cols = [c for c in d.columns if c.startswith("NO - ")]
d["YES_DEPTH"] = d[yes_cols].sum(axis=1)
d["NO_DEPTH"] = d[no_cols].sum(axis=1)

# pooled per-second index vol (for the cushion z-score)
allr = []
for _, w in d.groupby("TICKER"):
    w = w.sort_values("TS_UTC")
    lr = np.log(w.RTI_INDEX_PRICE).diff()
    dt = w.TS_UTC.diff().dt.total_seconds()
    allr.extend(lr[dt <= 3].dropna().tolist())
sigma_s = np.std(allr)

# per-window settlement outcome (60s-avg RTI near close vs strike)
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
    i = hit[0]                                  # first entry into the band
    r = w.loc[i]
    past = w.RTI_INDEX_PRICE.iloc[max(0, i-30):i+1]
    spot_mom = (r.RTI_INDEX_PRICE - past.iloc[0]) / past.iloc[0] if len(past) > 1 else 0.0
    lr = np.log(w.RTI_INDEX_PRICE.iloc[max(0, i-60):i+1]).diff().dropna()
    rvol = lr.std() if len(lr) > 5 else np.nan
    cushion_usd = r.RTI_INDEX_PRICE - r.STRIKE
    z = np.log(r.RTI_INDEX_PRICE / r.STRIKE) / (sigma_s * np.sqrt(r.TTE)) if r.TTE > 0 else np.nan
    tot_depth = r.YES_DEPTH + r.NO_DEPTH
    ob_imb = (r.YES_DEPTH - r.NO_DEPTH) / tot_depth if tot_depth > 0 else 0.0
    dipped = bool((w.MID.iloc[i:] < 0.50).any())
    rows.append(dict(
        ticker=tk, tte=r.TTE, price=r.MID, cushion_usd=cushion_usd, z_cushion=z,
        spot_mom_30s=spot_mom, rvol_60s=rvol, ob_imbalance=ob_imb,
        resolved_no=1 - outcome[tk], dipped_below_50=int(dipped),
    ))

f = pd.DataFrame(rows)
print(f"favorite entries ({BAND[0]}-{BAND[1]} band): {len(f)} windows")
print(f"  settlement reversals (resolved NO): {f.resolved_no.sum()}")
print(f"  intra-window dips below 0.50:       {f.dipped_below_50.sum()}")

if len(f):
    feats = ["tte", "cushion_usd", "z_cushion", "spot_mom_30s", "rvol_60s", "ob_imbalance"]
    for label in ("resolved_no", "dipped_below_50"):
        n_ev = f[label].sum()
        print(f"\n=== feature means by outcome ({label}); events={n_ev} ===")
        if n_ev == 0 or n_ev == len(f):
            print("  (need both reversed AND survived examples -- keep collecting)")
            continue
        print(f.groupby(label)[feats].mean().round(4).to_string())
        # crude separation: point-biserial corr of each feature with the label
        print("  corr(feature, reversal):",
              {ft: round(f[ft].corr(f[label]), 2) for ft in feats})

print("\nHypotheses to confirm as reversals accumulate:")
print("  - lower z_cushion / cushion_usd  -> higher reversal (thin margin)")
print("  - higher tte                     -> higher reversal (more time to flip)")
print("  - negative spot_mom_30s          -> higher reversal (spot sliding toward strike)")
print("  - higher rvol_60s                -> higher reversal (volatile regime)")
print("  - negative ob_imbalance          -> higher reversal (NO depth stacking)")
