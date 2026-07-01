import numpy as np
import pandas as pd

from load_data import load
df = load()
d = df[df.RTI_INDEX_PRICE > 0].copy()

yes_cols = [c for c in d.columns if c.startswith("YES - ")]
no_cols = [c for c in d.columns if c.startswith("NO - ")]
d["YES_DEPTH"] = d[yes_cols].sum(axis=1)
d["NO_DEPTH"] = d[no_cols].sum(axis=1)

# Realistic tradeable prices:
#   to BUY yes you cross to the ask  = 1 - no_bid
#   to SELL yes you hit the bid      = yes_bid
d["YES_ASK"] = 1 - d.KALSHI_NO_BID
d["YES_BID"] = d.KALSHI_YES_BID
d["MID"] = (d.YES_BID + d.YES_ASK) / 2

# Strategy params
LOOKBACK = 30       # seconds to measure the "recent rise"
UP_TRIG = 0.05      # +5% relative rise triggers a signal
TARGET = 0.10       # take profit at +10% relative to entry mid
STOP = 0.10         # symmetric stop at -10% (else hold to window end)
FEE = 0.01          # ~1c/contract round-trip fee estimate (Kalshi ~0.07*p*(1-p) per side)

def kalshi_fee(p):
    # Kalshi trading fee per contract per side, rounded up to the cent.
    return np.ceil(0.07 * p * (1 - p) * 100) / 100

def backtest(signal_fn, label):
    trades = []
    for tk, w in d.groupby("TICKER"):
        w = w.set_index("TIMESTAMP").sort_index()
        w = w[~w.index.duplicated(keep="last")]
        g = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="1s")).ffill(limit=5)
        g = g.dropna(subset=["MID", "YES_ASK", "YES_BID"])
        mid = g.MID.values
        ask = g.YES_ASK.values
        bid = g.YES_BID.values
        yd = g.YES_DEPTH.values
        nd = g.NO_DEPTH.values
        n = len(g)
        i = LOOKBACK
        while i < n - 1:
            if signal_fn(mid, yd, nd, i):
                entry = ask[i]                      # pay the ask
                if entry <= 0 or entry >= 1:
                    i += 1
                    continue
                tgt = mid[i] * (1 + TARGET)
                stp = mid[i] * (1 - STOP)
                exit_px = bid[-1]                    # default: window-end bid
                held = n - 1 - i
                for j in range(i + 1, n):
                    if mid[j] >= tgt:
                        exit_px = bid[j]            # sell at bid on target
                        held = j - i
                        break
                    if mid[j] <= stp:
                        exit_px = bid[j]
                        held = j - i
                        break
                fees = kalshi_fee(entry) + kalshi_fee(exit_px)
                pnl = (exit_px - entry) - fees
                trades.append((pnl, held, entry, exit_px))
                i = i + held + 1                    # no overlapping trades
            else:
                i += 1
    if not trades:
        print(f"{label}: no trades")
        return
    pnl = np.array([t[0] for t in trades])
    held = np.array([t[1] for t in trades])
    print(f"\n=== {label} ===")
    print(f"trades: {len(pnl)}")
    print(f"win rate: {100*(pnl>0).mean():.1f}%")
    print(f"avg P&L/trade: {100*pnl.mean():+.2f}c   median: {100*np.median(pnl):+.2f}c")
    print(f"total P&L (1 contract/trade): {100*pnl.sum():+.0f}c over the sample")
    print(f"avg hold: {held.mean():.0f}s")
    # expectancy vs a coin flip on the same trades w/o costs
    gross = np.array([t[3]-t[2] for t in trades])
    print(f"gross avg (no fees): {100*gross.mean():+.2f}c   fees drag: {100*(gross.mean()-pnl.mean()):.2f}c/trade")

# The user's strategy: recent +5% AND yes depth > no depth
def momentum_signal(mid, yd, nd, i):
    past = mid[i - LOOKBACK]
    if past <= 0:
        return False
    rose = (mid[i] - past) / past >= UP_TRIG
    imbalance = yd[i] > nd[i]
    return rose and imbalance

# Baseline A: recent +5% only (no depth filter)
def momentum_only(mid, yd, nd, i):
    past = mid[i - LOOKBACK]
    return past > 0 and (mid[i] - past) / past >= UP_TRIG

# Baseline B: enter every LOOKBACK seconds regardless (random-ish timing)
_counter = {"c": 0}
def always(mid, yd, nd, i):
    return (i % LOOKBACK) == 0

backtest(momentum_signal, "STRATEGY: +5% rise AND YES depth > NO depth")
backtest(momentum_only,   "BASELINE: +5% rise only")
backtest(always,          "BASELINE: enter every 30s (no signal)")
