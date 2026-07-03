"""Shared loader for the analysis scripts.

Concatenates every daily CSV in data/ plus the legacy single-file sample,
de-duplicates on (TIMESTAMP, TICKER), and parses TIMESTAMP. All analysis
scripts import this so a multi-day collection run is picked up automatically.
"""

import glob
import os

import pandas as pd

# Absolute so this loader works regardless of where it's imported from.
DATA_DIR = "d:/automation/Kalshi/data"
LEGACY = "d:/automation/Kalshi/btc_orderbook_data.csv"  # old sample location (now under data/)


def load():
    # Any CSV in data/ counts (daily files + a moved sample); dedup handles overlap.
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    # Back-compat: also pick up the sample if it's still in an old location.
    for extra in ("d:/automation/btc_orderbook_data.csv", LEGACY):
        if os.path.exists(extra) and extra not in files:
            files.append(extra)
    if not files:
        raise SystemExit(
            f"No data files found in {DATA_DIR} (or {LEGACY}). "
            "Run kalshi_pipeline.py first to collect data."
        )
    frames = [pd.read_csv(fp) for fp in files]
    df = pd.concat(frames, ignore_index=True)
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], format="%m-%d-%Y %H:%M:%S")
    # A ticker is unique per 15-min window, so (TIMESTAMP, TICKER) is a safe key
    # to drop any overlap between the legacy file and dated files.
    df = df.drop_duplicates(subset=["TIMESTAMP", "TICKER"]).reset_index(drop=True)
    print(f"[load_data] {len(files)} file(s), {len(df)} rows, "
          f"{df.TICKER.nunique()} windows")
    return df


if __name__ == "__main__":
    load()
