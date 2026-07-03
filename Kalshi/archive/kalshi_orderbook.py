import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from kalshi import KalshiClient

# 1. Load the variables
load_dotenv(dotenv_path="Tools/Access/.env")
api_key_id = os.getenv("KALSHI_API_KEY_ID")
key_file_path = os.getenv("KALSHI_KEY_FILE_PATH")

# 2. Bypass the Windows SSL log permission error
if "SSLKEYLOGFILE" in os.environ:
    del os.environ["SSLKEYLOGFILE"]

# 3. Generate the target ticker dynamically for the ACTIVE expiration window
now = datetime.now()

# Round UP to the next 15-minute block
minutes_to_add = 15 - (now.minute % 15)
expiration_time = now + timedelta(minutes=minutes_to_add)

yy = expiration_time.strftime("%y")
mmm = expiration_time.strftime("%b").upper()
dd = expiration_time.strftime("%d")
hhmm = expiration_time.strftime("%H%M")

target_ticker = f"KXBTC15M-{yy}{mmm}{dd}{hhmm}-15"

# 4. Connect to Kalshi and pull the order book
with KalshiClient(
    key_id=api_key_id,
    private_key_path=key_file_path
) as client:
    
    print(f"Pulling order book for {target_ticker}...\n")
    
    # Fetch the order book data
    try:
        ob = client.markets.get_orderbook(target_ticker)
    except AttributeError:
        ob = client.markets.orderbook(target_ticker)

    # Safely convert the Pydantic object into a dictionary to expose the arrays
    if hasattr(ob, "model_dump"):
        ob_data = ob.model_dump()
    elif hasattr(ob, "dict"):
        ob_data = ob.dict()
    else: 
        ob_data = vars(ob)

    # Safely grab the arrays
    yes_bids = ob_data.get('yes') or ob_data.get('yes_dollars') or ob_data.get('bids') or []
    no_bids = ob_data.get('no') or ob_data.get('no_dollars') or ob_data.get('asks') or []
    
    print("--- YES BIDS ---")
    if yes_bids:
        # Loop backwards to show the best (highest) bids at the top
        for bid in reversed(yes_bids):
            # Target the dictionary keys directly to get the actual values
            print(f"Price: ${bid['price']} | Quantity: {bid['quantity']}")
    else:
        print("No active YES bids.")

    print("\n--- NO BIDS ---")
    if no_bids:
        # Loop backwards to show the best (highest) bids at the top
        for bid in reversed(no_bids):
            # Target the dictionary keys directly to get the actual values
            print(f"Price: ${bid['price']} | Quantity: {bid['quantity']}")
    else:
        print("No active NO bids.")