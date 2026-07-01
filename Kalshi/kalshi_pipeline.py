import os
import sys
import time
import csv
from datetime import datetime
from dotenv import load_dotenv
from kalshi import KalshiClient

from rti_feed import RTIFeed

# --- Config -----------------------------------------------------------------
SERIES_TICKER = "KXBTC15M"
RTI_INDEX_ID = "BRTI"    # CF Benchmarks Bitcoin Real-Time Index
CSV_FILENAME = "btc_orderbook_data.csv"
POLL_INTERVAL = 1        # seconds between successful logs
RETRY_INTERVAL = 5       # seconds to wait after an error / no market

# --- Load credentials -------------------------------------------------------
load_dotenv(dotenv_path="Tools/Access/.env")
api_key_id = os.getenv("KALSHI_API_KEY_ID")
key_file_path = os.getenv("KALSHI_KEY_FILE_PATH")

if not api_key_id or not key_file_path:
    sys.exit("Error: KALSHI_API_KEY_ID / KALSHI_KEY_FILE_PATH not set. Check Tools/Access/.env")

# Bypass the Windows SSL key-log permission error
os.environ.pop("SSLKEYLOGFILE", None)

# --- Build the column schema (buckets 0.01 to 0.99) -------------------------
target_buckets = [0.01] + [round(i * 0.05, 2) for i in range(1, 20)] + [0.99]
columns = [
    "TIMESTAMP", "TICKER", "STRIKE", "CLOSE_TIME",
    "KALSHI_YES_BID", "KALSHI_NO_BID", "RTI_INDEX_PRICE",
]
for p in target_buckets:
    columns.extend([f"YES - {p:g}", f"NO - {round(1 - p, 2):g}"])


def get_bucket(price):
    """Round a price to its nearest 0.05 bucket, clamped to [0.01, 0.99]."""
    b = round(float(price) * 20) / 20
    return 0.01 if b < 0.05 else (0.99 if b > 0.95 else b)


def find_active_market(client):
    """Return the first open market whose ticker matches the series, else None."""
    open_markets = list(client.markets.list(series_ticker=SERIES_TICKER, status="open"))
    for m in open_markets:
        if SERIES_TICKER in m.ticker:
            return m
    return None


def extract_bids(ob):
    """Normalize an orderbook object into (yes_bids, no_bids) lists."""
    if hasattr(ob, "model_dump"):
        ob_data = ob.model_dump()
    elif hasattr(ob, "dict"):
        ob_data = ob.dict()
    else:
        ob_data = vars(ob)
    return ob_data.get("yes") or [], ob_data.get("no") or []


def build_row(market, ob, rti_price):
    """Build a single CSV row dict from the market, orderbook + underlying price."""
    yes_bids, no_bids = extract_bids(ob)

    # Best bid = highest price level (order book is sorted ascending)
    best_yes = round(float(yes_bids[-1]["price"]), 4) if yes_bids else 0.0
    best_no = round(float(no_bids[-1]["price"]), 4) if no_bids else 0.0

    strike = getattr(market, "floor_strike", None)
    close_time = getattr(market, "close_time", None)

    row = {col: 0 for col in columns}
    row.update({
        "TIMESTAMP": datetime.now().strftime("%m-%d-%Y %H:%M:%S"),
        "TICKER": market.ticker,
        "STRIKE": float(strike) if strike is not None else "",
        "CLOSE_TIME": close_time.isoformat() if close_time is not None else "",
        "KALSHI_YES_BID": best_yes,
        "KALSHI_NO_BID": best_no,
        "RTI_INDEX_PRICE": rti_price,
    })

    for bid in yes_bids:
        col_name = f"YES - {get_bucket(bid['price']):g}"
        if col_name in row:
            row[col_name] += float(bid["quantity"])
    for bid in no_bids:
        col_name = f"NO - {get_bucket(bid['price']):g}"
        if col_name in row:
            row[col_name] += float(bid["quantity"])

    return row


def main():
    print("Starting pipeline using Dynamic Market Discovery...")

    # Start the RTI index feed (WebSocket, runs in a background thread).
    # The BTC index price is not on the REST market object -- it only streams
    # over the cfbenchmarks_value channel.
    rti_feed = RTIFeed(api_key_id, key_file_path, index_id=RTI_INDEX_ID)
    rti_feed.start()
    print(f"Waiting for {RTI_INDEX_ID} feed to warm up...")

    # Open the CSV once and stream rows to it (append if it already exists)
    file_exists = os.path.exists(CSV_FILENAME)
    with open(CSV_FILENAME, "a", newline="") as f, \
         KalshiClient(key_id=api_key_id, private_key_path=key_file_path) as client:

        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
            f.flush()

        while True:
            try:
                market = find_active_market(client)
                if market is None:
                    print("No active market found. Waiting...")
                    time.sleep(RETRY_INTERVAL)
                    continue

                ticker = market.ticker
                ob = client.markets.orderbook(ticker)

                # Live BTC index from the WebSocket feed (0.0 until it warms up)
                rti_price = rti_feed.price
                if rti_feed.updates == 0:
                    print("Warning: RTI feed not warmed up yet; logging 0.0")
                elif rti_feed.age > 15:
                    print(f"Warning: RTI feed stale ({rti_feed.age:.0f}s old); "
                          f"logging last value ${rti_price:,.2f}")

                row = build_row(market, ob, rti_price)
                writer.writerow(row)
                f.flush()

                print(f"Logged {ticker} | RTI: ${float(rti_price):,.2f}")
                time.sleep(POLL_INTERVAL)

            except KeyboardInterrupt:
                print("\nStopped.")
                break
            except Exception as e:
                # Survive transient API / network hiccups instead of crashing
                print(f"Error: {e!r} - retrying in {RETRY_INTERVAL}s")
                time.sleep(RETRY_INTERVAL)


if __name__ == "__main__":
    main()
