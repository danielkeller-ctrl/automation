"""Shared loader for the analysis scripts.

Concatenates every daily CSV in data/ plus the legacy single-file sample,
de-duplicates on (TIMESTAMP, TICKER), and parses TIMESTAMP. All analysis
scripts import this so a multi-day collection run is picked up automatically.
"""

import glob
import os

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data")
LEGACY = os.path.join(_HERE, "btc_orderbook_data.csv")  # the original ~4h sample


def load():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "btc_orderbook_data_*.csv")))
    if os.path.exists(LEGACY):
        files.append(LEGACY)
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
