"""
logic_5_classification_style.py

Implements a Classification-Style Strategy (UP / FLAT / DOWN).

- We define a small function to classify the predicted vs. current price difference
  into one of three categories: UP, DOWN, or FLAT (within some threshold).
- If classified as UP, we go or stay long.
- If classified as DOWN, we go or stay short.
- If classified as FLAT, we do nothing or close any open positions.
"""

import logging

def run_logic(current_price, predicted_price, ticker):
    """
    Main entry point called by the 'router' in the main script.
    
    This function:
      1. Classifies the market direction (UP/DOWN/FLAT) using a 1% threshold.
      2. Retrieves the current position via the API.
      3. Based on the classification and current position:
           - For an UP signal:
               • If already long, do nothing.
               • If short, flip from short to long.
               • If flat, open a long position.
           - For a DOWN signal:
               • If already short, do nothing.
               • If long, flip from long to short.
               • If flat, open a short position.
           - For a FLAT signal:
               • If long, close the long position.
               • If short, cover the short position.
               • If flat, do nothing.
    """
    from forest import api, buy_shares, sell_shares, short_shares, close_short

    threshold_pct = 1.0  # 1% threshold for classification
    signal = classify_signal(predicted_price, current_price, threshold_pct)

    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
        if position_qty > 0:
            current_position = "long"
        elif position_qty < 0:
            current_position = "short"
        else:
            current_position = "none"
    except Exception:
        position_qty = 0
        current_position = "none"

    if signal == "UP":
        # Go or stay long
        if current_position == "long":
            logging.info(f"[{ticker}] Classification: Signal=UP but already long, no change.")
        elif current_position == "short":
            logging.info(f"[{ticker}] Classification: Signal=UP, flipping from short to long.")
            close_short(ticker, abs(position_qty), current_price)
            account = api.get_account()
            cash = float(account.cash)
            shares_to_buy = int(cash // current_price)
            if shares_to_buy > 0:
                buy_shares(ticker, shares_to_buy, current_price, predicted_price)
            else:
                logging.info(f"[{ticker}] Classification: Not enough cash to flip to long.")
        else:
            logging.info(f"[{ticker}] Classification: Signal=UP, opening new long.")
            account = api.get_account()
            cash = float(account.cash)
            shares_to_buy = int(cash // current_price)
            if shares_to_buy > 0:
                buy_shares(ticker, shares_to_buy, current_price, predicted_price)
            else:
                logging.info(f"[{ticker}] Classification: Not enough cash to go long.")

    elif signal == "DOWN":
        # Go or stay short
        if current_position == "short":
            logging.info(f"[{ticker}] Classification: Signal=DOWN but already short, no change.")
        elif current_position == "long":
            logging.info(f"[{ticker}] Classification: Signal=DOWN, flipping from long to short.")
            sell_shares(ticker, position_qty, current_price, predicted_price)
            account = api.get_account()
            cash = float(account.cash)
            shares_to_short = int(cash // current_price)
            if shares_to_short > 0:
                short_shares(ticker, shares_to_short, current_price, predicted_price)
            else:
                logging.info(f"[{ticker}] Classification: Not enough funds to flip to short.")
        else:
            logging.info(f"[{ticker}] Classification: Signal=DOWN, opening new short.")
            account = api.get_account()
            cash = float(account.cash)
            shares_to_short = int(cash // current_price)
            if shares_to_short > 0:
                short_shares(ticker, shares_to_short, current_price, predicted_price)
            else:
                logging.info(f"[{ticker}] Classification: Not enough funds to go short.")

    else:
        # signal == "FLAT"
        logging.info(f"[{ticker}] Classification: Signal=FLAT.")
        # Optionally close any open position if we only want to be in the market for UP or DOWN
        if current_position == "long":
            logging.info(f"[{ticker}] Classification: Closing existing long due to FLAT signal.")
            sell_shares(ticker, position_qty, current_price, predicted_price)
        elif current_position == "short":
            logging.info(f"[{ticker}] Classification: Covering existing short due to FLAT signal.")
            close_short(ticker, abs(position_qty), current_price)
        else:
            logging.info(f"[{ticker}] Classification: Already flat, no action.")

def classify_signal(predicted_close, current_close, threshold_pct=1.0):
    """
    Convert numeric prediction into a simple classification: UP, DOWN, or FLAT.
    A threshold_pct of 1.0 means a 1% band around current_close is considered "FLAT."
    """
    if current_close == 0:
        return "FLAT"  # Avoid division by zero edge case
    ratio = (predicted_close - current_close) / current_close
    if ratio > threshold_pct / 100.0:
        return "UP"
    elif ratio < -threshold_pct / 100.0:
        return "DOWN"
    else:
        return "FLAT"

def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    """
    Backtest version of the Classification-Style Strategy.

    Parameters:
      - current_price (float): The current market price.
      - predicted_price (float): The model's predicted price.
      - position_qty (float): The current position quantity (positive for long, negative for short, 0 for none).

    Returns:
      - action (str): One of the following:
            • 'BUY'   - Open a new long position.
            • 'SELL'  - Exit an existing long position.
            • 'SHORT' - Open a new short position.
            • 'COVER' - Exit an existing short position.
            • 'NONE'  - Hold the current position / take no action.

    Trading Logic (mirrors run_logic):
      1. Classify the signal (UP, DOWN, or FLAT) using a 1% threshold.
      2. Determine the current position based on position_qty:
           - Positive: long.
           - Negative: short.
           - Zero: none.
      3. For an UP signal:
           - If already long, return "NONE".
           - If short, return "COVER" (to flip from short to long).
           - If flat, return "BUY".
      4. For a DOWN signal:
           - If already short, return "NONE".
           - If long, return "SELL" (to flip from long to short).
           - If flat, return "SHORT".
      5. For a FLAT signal:
           - If long, return "SELL" (to exit the long position).
           - If short, return "COVER" (to exit the short position).
           - If none, return "NONE".
    """
    threshold_pct = 1.0  # 1% threshold for classification
    signal = classify_signal(predicted_price, current_price, threshold_pct)

    # Determine current position based on position_qty.
    if position_qty > 0:
        current_position = "long"
    elif position_qty < 0:
        current_position = "short"
    else:
        current_position = "none"

    if signal == "UP":
        if current_position == "long":
            return "NONE"
        elif current_position == "short":
            return "COVER"
        else:
            return "BUY"

    elif signal == "DOWN":
        if current_position == "short":
            return "NONE"
        elif current_position == "long":
            return "SELL"
        else:
            return "SHORT"

    else:  # signal == "FLAT"
        if current_position == "long":
            return "SELL"
        elif current_position == "short":
            return "COVER"
        else:
            return "NONE"
