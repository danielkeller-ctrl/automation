import numpy as np
import pandas as pd
from load_data import load

df = load()
d = df[df.RTI_INDEX_PRICE > 0].copy()

yes_cols = [c for c in d.columns if c.startswith("YES - ")]
no_cols = [c for c in d.columns if c.startswith("NO - ")]
d["YES_DEPTH"] = d[yes_cols].sum(axis=1)
d["NO_DEPTH"] = d[no_cols].sum(axis=1)
d["YES_ASK"] = 1 - d.KALSHI_NO_BID
d["YES_BID"] = d.KALSHI_YES_BID
d["MID"] = (d.YES_BID + d.YES_ASK) / 2

LOOKBACK = 30
UP_TRIG = 0.05
TARGET = 0.10
FEE = 0.02

def kalshi_fee(p):
    return np.ceil(0.07 * p * (1 - p) * 100) / 100

def run(stop):
    """stop=None means hold to window end; else exit at -stop relative to entry."""
    pnl, wins, losses, exits = [], [], [], {"target": 0, "stop": 0, "end": 0}
    for tk, w in d.groupby("TICKER"):
        w = w.set_index("TIMESTAMP").sort_index()
        w = w[~w.index.duplicated(keep="last")]
        g = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="1s")).ffill(limit=5)
        g = g.dropna(subset=["MID", "YES_ASK", "YES_BID"])
        mid, ask, bid = g.MID.values, g.YES_ASK.values, g.YES_BID.values
        yd, nd = g.YES_DEPTH.values, g.NO_DEPTH.values
        n = len(g); i = LOOKBACK
        while i < n - 1:
            past = mid[i - LOOKBACK]
            if past > 0 and (mid[i] - past) / past >= UP_TRIG and yd[i] > nd[i]:
                entry = ask[i]
                if entry <= 0 or entry >= 1:
                    i += 1; continue
                tgt = mid[i] * (1 + TARGET)
                stp = mid[i] * (1 - stop) if stop is not None else -1
                exit_px, held, why = bid[-1], n - 1 - i, "end"
                for j in range(i + 1, n):
                    if mid[j] >= tgt:
                        exit_px, held, why = bid[j], j - i, "target"; break
                    if stop is not None and mid[j] <= stp:
                        exit_px, held, why = bid[j], j - i, "stop"; break
                p = (exit_px - entry) - (kalshi_fee(entry) + kalshi_fee(exit_px))
                pnl.append(p); exits[why] += 1
                (wins if p > 0 else losses).append(p)
                i += held + 1
            else:
                i += 1
    pnl = np.array(pnl)
    aw = np.mean(wins) if wins else 0.0
    al = np.mean(losses) if losses else 0.0
    label = f"stop {int(stop*100)}%" if stop is not None else "no stop (hold to end)"
    print(f"{label:22} n={len(pnl):3d}  win%={100*(pnl>0).mean():4.1f}  "
          f"avgP&L={100*pnl.mean():+5.2f}c  total={100*pnl.sum():+6.0f}c  "
          f"avgWin={100*aw:+5.2f}c avgLoss={100*al:+5.2f}c  "
          f"exits[tgt/stop/end]={exits['target']}/{exits['stop']}/{exits['end']}")

print("Momentum entry (+5% rise AND yes depth>no depth), target +10%, varying the stop:\n")
for s in (0.05, 0.10, 0.25, None):
    run(s)
