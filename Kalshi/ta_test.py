import numpy as np
import pandas as pd

from load_data import load
df = load()
d = df[df.RTI_INDEX_PRICE > 0].copy()
d["MID"] = ((1 - d.KALSHI_NO_BID) + d.KALSHI_YES_BID) / 2

BAR = 5  # seconds per bar

def bars(series_col):
    """Resample each window to BAR-second bars; return list of per-window arrays."""
    out = []
    for tk, w in d.groupby("TICKER"):
        w = w.set_index("TIMESTAMP").sort_index()
        w = w[~w.index.duplicated(keep="last")]
        g = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="1s")).ffill(limit=5)
        b = g[series_col].resample(f"{BAR}s").last().dropna()
        if len(b) > 40:
            out.append(b.values.astype(float))
    return out

# ============ 1. RANDOM-WALK TESTS on the BTC index ============
print("=== RANDOM-WALK TESTS (BTC RTI index, %ds bars) ===" % BAR)
series = bars("RTI_INDEX_PRICE")
all_ret = np.concatenate([np.diff(np.log(s)) for s in series])
print(f"bars: {sum(len(s) for s in series)}   return samples: {len(all_ret)}")

# Return autocorrelation
print("\nreturn autocorrelation (~0 => random walk):")
for lag in (1, 2, 3, 5, 10):
    ac = np.corrcoef(all_ret[:-lag], all_ret[lag:])[0, 1]
    print(f"  lag {lag:>2}: {ac:+.3f}")

# Variance ratio: VR(q)=Var(q-step)/(q*Var(1-step)); ~1 random, <1 revert, >1 trend
print("\nvariance ratio (1.0 => random walk, <1 mean-revert, >1 trending):")
var1 = np.var(all_ret)
for q in (2, 5, 10, 20):
    vq = []
    for s in series:
        r = np.diff(np.log(s))
        if len(r) > q:
            qr = np.array([r[i:i+q].sum() for i in range(len(r)-q)])
            vq.append(np.var(qr))
    vr = np.mean(vq) / (q * var1)
    print(f"  VR({q:>2}) = {vr:.3f}")

# Runs test (sign sequence randomness)
signs = np.sign(all_ret); signs = signs[signs != 0]
runs = 1 + np.sum(signs[1:] != signs[:-1])
n_pos = np.sum(signs > 0); n_neg = np.sum(signs < 0); n = len(signs)
exp_runs = 1 + 2*n_pos*n_neg/n
std_runs = np.sqrt(2*n_pos*n_neg*(2*n_pos*n_neg-n)/(n**2*(n-1)))
z = (runs - exp_runs)/std_runs
print(f"\nruns test: z = {z:+.2f}  (|z|<1.96 => consistent with randomness)")

# ============ 2. CLASSIC INDICATORS: directional accuracy + P&L ============
def rsi(x, n=14):
    delta = np.diff(x); up = np.clip(delta, 0, None); dn = -np.clip(delta, None, 0)
    ru = pd.Series(up).rolling(n).mean().values; rd = pd.Series(dn).rolling(n).mean().values
    rs = ru/(rd+1e-12); r = 100 - 100/(1+rs)
    return np.concatenate([[np.nan], r])

def sma(x, n): return pd.Series(x).rolling(n).mean().values
def ema(x, n): return pd.Series(x).ewm(span=n, adjust=False).mean().values

def eval_signal(name, sigfun):
    hits = tot = 0; pnl = 0.0; ntr = 0
    for s in series:
        sig = sigfun(s)                    # +1 long / -1 short / 0 flat, per bar
        r = np.diff(np.log(s))             # forward return from bar i to i+1
        for i in range(len(r)):
            if i >= len(sig) or np.isnan(sig[i]) or sig[i] == 0:
                continue
            fwd = r[i]
            tot += 1; ntr += 1
            if np.sign(fwd) == np.sign(sig[i]): hits += 1
            pnl += sig[i]*fwd
    acc = 100*hits/tot if tot else float('nan')
    print(f"  {name:26} dir.acc={acc:5.1f}%  signals={ntr:5d}  gross_logret={pnl:+.4f}")

print("\n=== CLASSIC INDICATORS on BTC index (dir.acc ~50% => no edge) ===")
# RSI mean-reversion: long when oversold(<30), short when overbought(>70)
eval_signal("RSI(14) reversion", lambda s: np.where(rsi(s)<30,1,np.where(rsi(s)>70,-1,0)))
# RSI trend-follow (opposite)
eval_signal("RSI(14) trend", lambda s: np.where(rsi(s)>70,1,np.where(rsi(s)<30,-1,0)))
# SMA crossover trend: fast>slow => long
eval_signal("SMA(5/20) crossover", lambda s: np.sign(sma(s,5)-sma(s,20)))
# MACD: ema12-ema26 vs its signal ema9
def macd_sig(s):
    m = ema(s,12)-ema(s,26); sig = pd.Series(m).ewm(span=9,adjust=False).mean().values
    return np.sign(m-sig)
eval_signal("MACD(12/26/9)", macd_sig)
# Bollinger %b reversion: below lower band => long, above upper => short
def boll(s, n=20, k=2):
    ma = sma(s,n); sd = pd.Series(s).rolling(n).std().values
    return np.where(s < ma-k*sd, 1, np.where(s > ma+k*sd, -1, 0))
eval_signal("Bollinger(20,2) revert", boll)
# ROC momentum: sign of 10-bar rate of change
eval_signal("ROC(10) momentum", lambda s: np.sign(s - np.concatenate([[np.nan]*10, s[:-10]])))

print("\n(baseline: pure coin-flip = 50.0% directional accuracy)")
