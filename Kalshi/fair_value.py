import numpy as np
import pandas as pd
from scipy.stats import norm

from load_data import load
df = load()
df["CLOSE_TIME"] = pd.to_datetime(df["CLOSE_TIME"], utc=True)
d = df[df.RTI_INDEX_PRICE > 0].copy()

d["TS_UTC"] = d.TIMESTAMP.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
d["TTE"] = (d.CLOSE_TIME - d.TS_UTC).dt.total_seconds()
d = d[(d.TTE > 0) & (d.TTE <= 900)].copy()

# Tradeable prices
d["YES_ASK"] = 1 - d.KALSHI_NO_BID
d["YES_BID"] = d.KALSHI_YES_BID
d["MID"] = (d.YES_BID + d.YES_ASK) / 2

# --- 1. Estimate per-second volatility of the BTC index (pooled across windows) ---
rets = []
for tk, w in d.groupby("TICKER"):
    w = w.sort_values("TS_UTC")
    lr = np.log(w.RTI_INDEX_PRICE).diff().dropna()
    # guard against gap-induced multi-second jumps: keep only ~1s steps
    dt = w.TS_UTC.diff().dt.total_seconds().values[1:]
    lr = lr.values[dt <= 3]
    rets.extend(lr.tolist())
rets = np.array(rets)
sigma_s = rets.std()  # per-second log-return vol
print(f"per-second index vol: {sigma_s:.2e}   (annualized ~{sigma_s*np.sqrt(365*24*3600):.0%})")

# --- 2. Theoretical fair YES probability (zero-drift diffusion over TTE) ---
# P(S_T >= K) = N( ln(S/K) / (sigma_s * sqrt(TTE_seconds)) )
S = d.RTI_INDEX_PRICE.values
K = d.STRIKE.values
tau = d.TTE.values
vol_to_exp = sigma_s * np.sqrt(tau)
z = np.log(S / K) / vol_to_exp
d["FAIR"] = norm.cdf(z)
d["EDGE"] = d.MID - d.FAIR   # + => Kalshi rich (overpricing YES), - => cheap

print(f"\noverall: mean|edge| = {d.EDGE.abs().mean()*100:.2f}c   mean edge = {d.EDGE.mean()*100:+.2f}c")

# --- 3. Where does Kalshi diverge? Bucket by moneyness and time-to-expiry ---
d["MNY"] = (S - K)  # dollars in/out of the money
d["mny_bin"] = pd.cut(d.MNY, [-1e9, -40, -15, -5, 5, 15, 40, 1e9],
                      labels=["<<-40", "-40..-15", "-15..-5", "-5..5", "5..15", "15..40", ">>40"])
d["tte_bin"] = pd.cut(d.TTE, [0, 120, 300, 900], labels=["<2min", "2-5min", ">5min"])

print("\n=== mean Kalshi EDGE (mid - fair), in cents ===")
print("(+ = Kalshi overprices YES; - = underprices. |edge| must beat ~3.5c cost to trade)")
piv = d.pivot_table(index="mny_bin", columns="tte_bin", values="EDGE",
                    aggfunc="mean", observed=False) * 100
print(piv.round(1).to_string())

print("\n=== sample counts per bucket ===")
cnt = d.pivot_table(index="mny_bin", columns="tte_bin", values="EDGE",
                    aggfunc="size", observed=False)
print(cnt.to_string())

# --- 4. How often is the mispricing bigger than the spread you'd pay? ---
d["SPREAD"] = d.YES_ASK - d.YES_BID
tradeable = (d.EDGE.abs() > d.SPREAD + 0.02)  # edge must beat spread + ~2c fees
print(f"\nrows where |edge| > spread+2c fees: {tradeable.mean()*100:.1f}%")
print(f"median spread: {d.SPREAD.median()*100:.1f}c")
