"""Does an early exit (stop) help the favorite (0.80-0.90) buy-and-hold bet?

Buy YES at ask on first entry into the band, then either hold to expiry or
sell at the bid if mid falls to a stop level. Key diagnostic: of the trades
the stop closes, how many WOULD have recovered to YES (reversion harvested).
Entry fee always applies; an early sell adds a second fee; settlement is free.
"""

import numpy as np
import pandas as pd
from load_data import load

BAND = (0.80, 0.90)
STOPS = [None, 0.70, 0.60, 0.50]

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

# collect one trade per window
trades = []  # each: dict(z, results={stop: (pnl, stopped, would_yes)})
for tk, w in d.groupby("TICKER"):
    if tk not in outcome:
        continue
    w = w.set_index("TS_UTC").sort_index()
    w = w[~w.index.duplicated(keep="last")]
    g = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="1s")).ffill(limit=5)
    g = g.dropna(subset=["MID", "YES_ASK", "YES_BID", "RTI_INDEX_PRICE"])
    mid, ask, bid = g.MID.values, g.YES_ASK.values, g.YES_BID.values
    hit = np.where((mid >= BAND[0]) & (mid < BAND[1]))[0]
    if len(hit) == 0:
        continue
    i = hit[0]
    entry = ask[i]
    y = outcome[tk]
    tte_i = g.TTE.values[i]
    z = np.log(g.RTI_INDEX_PRICE.values[i] / g.STRIKE.values[i]) / (sigma_s * np.sqrt(tte_i)) if tte_i > 0 else np.nan
    res = {}
    for stop in STOPS:
        pnl, stopped, would_yes = None, False, None
        if stop is not None:
            for j in range(i + 1, len(g)):
                if mid[j] <= stop:
                    pnl = bid[j] - entry - fee(entry) - fee(bid[j])
                    stopped, would_yes = True, y
                    break
        if pnl is None:  # held to expiry
            pnl = y - entry - fee(entry)
        res[stop] = (pnl, stopped, would_yes)
    trades.append(dict(z=z, res=res))

tr = pd.DataFrame(trades).dropna(subset=["z"])
zmed = tr.z.median()
print(f"favorite entries in {BAND}: {len(tr)}   (z median={zmed:.2f})\n")

def report(subset, title):
    print(f"=== {title} (n={len(subset)}) ===")
    print(f"{'exit rule':>16} {'avgP&L':>8} {'total':>8} {'win%':>6} {'stopped':>8} {'stop cut a winner':>18}")
    for stop in STOPS:
        vals = [r[stop] for r in subset.res]
        pnl = np.array([v[0] for v in vals])
        nstop = sum(1 for v in vals if v[1])
        cut = sum(1 for v in vals if v[1] and v[2] == 1)  # stopped but would've been YES
        name = "hold to expiry" if stop is None else f"stop @ {stop:.2f}"
        cut_str = f"{cut}/{nstop}" if nstop else "-"
        print(f"{name:>16} {100*pnl.mean():>+7.2f}c {100*pnl.sum():>+7.0f}c "
              f"{100*(pnl>0).mean():>5.0f} {nstop:>8} {cut_str:>18}")
    print()

report(tr, "ALL favorites")
report(tr[tr.z >= zmed], f"HIGH cushion only (z >= {zmed:.2f})")
report(tr[tr.z < zmed], f"LOW cushion only (z < {zmed:.2f})")
