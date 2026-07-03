# Kalshi BTC 15-Minute Market — Efficiency Study

A data pipeline and analysis suite built to find a retail trading edge in Kalshi's
**KXBTC15M** market ("Will BTC be up in the next 15 minutes?").

## TL;DR

**The market is efficient to a 1-second retail participant. No tradeable edge was found.**
Eight distinct strategies were tested on ~160 fifteen-minute windows (2026-07-01 → 07-03,
~130k rows of 1-second data). Every one was net-negative after costs, and the single
promising thread (buying high-cushion favorites) was shown to be **in-sample overfitting** —
it disappeared under a proper out-of-sample test.

This is a genuine, useful result: it saved real capital from eight losing ideas, and it's
confirmed with enough data (~160 windows, out-of-sample validated) to trust.

---

## Folder structure

```
Kalshi/
├─ README.md              <- this file
├─ kalshi_pipeline.py     <- LIVE data collector (run this to gather data)
├─ rti_feed.py            <- background WebSocket feed for the BTC index (BRTI)
├─ data/                  <- daily CSVs: btc_orderbook_data_YYYY-MM-DD.csv
├─ analysis/              <- all analysis scripts + load_data.py (shared loader)
└─ archive/               <- early throwaway scripts (kalshi_btc, kalshi_orderbook)
```

## Data pipeline

`kalshi_pipeline.py` polls the active KXBTC15M market once per second and logs, per row:
`TIMESTAMP, TICKER, STRIKE, CLOSE_TIME, KALSHI_YES_BID, KALSHI_NO_BID, RTI_INDEX_PRICE`,
plus bucketed order-book depth (`YES - x` / `NO - y` columns).

Key facts:
- The **BTC index price ("RTI")** is *not* a REST field — it only streams over the
  authenticated `cfbenchmarks_value` WebSocket channel (index id `BRTI`, CF Benchmarks).
  `rti_feed.py` runs this in a background thread. This was the main non-obvious build hurdle.
- REST endpoints (`markets.list`, `orderbook`) are effectively **public** — they return data
  even with a bad key, so a broken key only surfaces as a WebSocket 401.
- Output **rotates daily** into `data/`. Restart-safe (appends).
- `TIMESTAMP` is naive **local time (America/New_York)**; `CLOSE_TIME` is UTC. Convert before
  computing time-to-expiry.

### Running it

```bash
# from d:/automation  (relative path to Tools/Access/.env)
C:/Users/kelle/anaconda3/python.exe Kalshi/kalshi_pipeline.py
```
Credentials live in `d:/automation/Tools/Access/.env`
(`KALSHI_API_KEY_ID` = UUID, `KALSHI_KEY_FILE_PATH` = RSA PEM).
Use the Anaconda Python — the Windows Store `python` on PATH is a stub.

### Running the analysis

```bash
set PYTHONIOENCODING=utf-8      # ta_test.py prints non-ASCII
C:/Users/kelle/anaconda3/python.exe Kalshi/analysis/<script>.py
```
Every analysis script auto-loads *all* CSVs in `data/` via `analysis/load_data.py`.

---

## Strategies tested & results

| # | Strategy | Script | Verdict |
|---|----------|--------|---------|
| 1 | Order-book momentum (buy after a rise + depth imbalance) | `backtest_momentum.py` | Loses **even before fees** — price mean-reverts |
| 2 | + stop-losses (5/10/25%) | `stop_sweep.py` | All worse — stops harvest reversions |
| 3 | Spot → Kalshi lead/lag | `analyze.py`, `analyze2.py` | No lag at 1s; ~91% of a spot jump is priced within the same second |
| 4 | Technical analysis (RSI/MACD/SMA/Bollinger/…) | `ta_test.py` | Price ≈ random walk; every indicator ~50% directional |
| 5 | Fair-value mispricing (diffusion model vs price) | `fair_value.py`, `validate.py` | Model is a **worse** predictor than Kalshi (Brier 0.134 vs 0.126) |
| 6 | Order-book following ("follow the big orders") | `orderbook_follow.py` | **~67% of large orders are pulled within 5s (spoofs)**; following is <50% accurate and loses |
| 7 | High-probability favorites (buy at 0.80–0.95, hold) | `highprob.py` | Well-calibrated (0.86 → resolves YES ~88%); net loss |
| 8 | + cushion filter, + early exit, + offset/hedge | `filter_test.py`, `favorite_stop.py`, `offset_test.py`, `filter_oos.py` | Filter **overfit** (fails out-of-sample); exits/hedges make it far worse |

### The decisive test (strategy 8)

The high-cushion filter looked profitable in-sample (+6.4¢/trade) but was validated
out-of-sample by fixing the volatility and z-threshold on the first half of the data and
scoring the second half untouched (`filter_oos.py`):

| Bucket | In-sample | **Out-of-sample** |
|--------|-----------|-------------------|
| High cushion | +6.43¢ | **−8.17¢** |
| Low cushion | −3.10¢ | −8.32¢ |

The separation vanishes out-of-sample → the edge was fitting noise.

---

## Key technical findings

- **Mean reversion, not momentum.** Short-horizon returns reverse (variance ratios < 1;
  5–10s return autocorrelation negative). This is why every reactive overlay (stops, early
  exits, hedges) *loses* — in a mean-reverting market they systematically sell the bottom /
  buy right before the bounce.
- **The order book is adversarial.** ~2/3 of large resting orders vanish within 5 seconds
  *without the price moving* → cancelled, not filled → spoofing/layering. The visible depth
  is part fake (spoofs) and doesn't predict price.
- **Kalshi prices are well-calibrated probabilities.** A 0.86 contract resolves YES ~88% of
  the time; a simple diffusion model is a worse forecaster than the market price itself.
- **Regime dependence dominates small samples.** The favorite strategy showed +9¢ on the
  first 23 windows (a calm stretch) and turned negative once volatile sessions arrived — a
  reminder that anything under ~100+ windows here is noise.

## What "efficient" means here (and the caveats)

Prices already reflect available information, so no strategy using **public info + past
prices** beats the market after the ~1¢ spread + fees. Efficiency is **relative**:
- **To a 1-second retail taker on flagship BTC** → efficient (this study).
- **To an HFT with tick data + colocation** → microstructure edges may remain (they enforce
  the efficiency we observe).
- **In illiquid / niche Kalshi markets** → possibly *not* efficient; thinner competition can
  let mispricings survive.

## Where an edge might still exist (untested)

1. **Less-liquid Kalshi markets** — fewer sharp participants than flagship BTC.
2. **Longer horizons** than 15 minutes — outside the HFT reaction game.
3. **Genuine information edges** — domain knowledge others don't have.
4. Sub-second/tick microstructure — requires infrastructure retail can't buy.

---

*Data collected 2026-07-01 → 2026-07-03. ~160 windows, ~130k rows of 1-second observations.*
