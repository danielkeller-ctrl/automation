import os
from dotenv import load_dotenv
from kalshi import KalshiClient

# 1. Load the variables from your secure .env file
load_dotenv(dotenv_path="Tools/Access/.env")

api_key_id = os.getenv("KALSHI_API_KEY_ID")
key_file_path = os.getenv("KALSHI_KEY_FILE_PATH")

if not api_key_id or not key_file_path:
    print("Error: Could not load keys from .env file. Check your file paths.")
    exit()

# --> THE FIX: Force Python to ignore the restricted SSL key log file <--
if "SSLKEYLOGFILE" in os.environ:
    del os.environ["SSLKEYLOGFILE"]

print("Authenticating with Kalshi...")

# 2. Connect to Kalshi using your secured credentials
with KalshiClient(
    key_id=api_key_id,
    private_key_path=key_file_path
) as client:
    
    print("Connected successfully!\n")
    print("Fetching active BTC 15-Minute Markets...\n")

    # 3. Request the active markets for the BTC 15-minute series
    btc_markets = client.markets.list(series_ticker="KXBTC15M", status="open")
    
    # 4. Loop through the response and print the market info
    for market in btc_markets:
        print(f"Title:  {market.title}")
        print(f"Ticker: {market.ticker}")
        print(f"Yes Bid: ${market.yes_bid}")
        print(f"No Bid:  ${market.no_bid}")
        print("-" * 50)