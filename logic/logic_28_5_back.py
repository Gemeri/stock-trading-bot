# logic.py

import os
import pandas as pd
from dotenv import load_dotenv
from forest import api, buy_shares, sell_shares, short_shares, close_short

load_dotenv()

BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME")
TICKERS = os.getenv("TICKERS").split(',')
DISABLED_FEATURES = os.getenv("DISABLED_FEATURES").split(',')

TIMEFRAME_MAP = {
    "4Hour": "H4",
    "2Hour": "H2",
    "1Hour": "H1",
    "30Min": "M30",
    "15Min": "M15"
}

def load_filtered_csv(ticker):
    suffix = TIMEFRAME_MAP.get(BAR_TIMEFRAME, "H1")
    csv_filename = f"{ticker}_{suffix}.csv"
    df = pd.read_csv(csv_filename)
    df = df.drop(columns=[col for col in DISABLED_FEATURES if col in df.columns], errors='ignore')
    return df

def run_logic(current_price, predicted_price, ticker):
    df = load_filtered_csv(ticker)

    # Get price 4 candles ago from the bottom (live)
    price_4_candles_ago = df['close'].iloc[-4]
    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception as e:
        print(f"No open position for {ticker}: {e}")
        position_qty = 0.0

    account = api.get_account()
    cash = float(account.cash)

    if predicted_price > price_4_candles_ago:
        if position_qty < 0:
            qty_to_close = abs(position_qty)
            print("cover")
            close_short(ticker, qty_to_close, current_price)
        if position_qty <= 0:
            max_shares = int(cash // current_price)
            print("buy")
            buy_shares(ticker, max_shares, current_price, predicted_price)

    elif predicted_price < price_4_candles_ago:
        if position_qty > 0:
            print("sell")
            sell_shares(ticker, position_qty, current_price, predicted_price)
        if position_qty >= 0:
            max_shares = int(cash // current_price)
            print("short")
            short_shares(ticker, max_shares, current_price, predicted_price)

def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    ticker = TICKERS[0]
    df = load_filtered_csv(ticker)

    # Get the index of the current timestamp
    current_index = df.index[df['timestamp'] == current_timestamp].tolist()

    if not current_index:
        return "NONE"

    current_index = current_index[0]

    if current_index < 4:
        return "NONE"

    # Get price 4 candles ago relative to current_timestamp
    price_4_candles_ago = df['close'].iloc[current_index - 4]

    if predicted_price > price_4_candles_ago:
        if position_qty < 0:
            return "COVER"
        if position_qty <= 0:
            return "BUY"

    elif predicted_price < price_4_candles_ago:
        if position_qty > 0:
            return "SELL"
        if position_qty >= 0:
            return "SHORT"

    return "NONE"
