"""
logic_7_cooldown.py

Implements a "Cooldown" or "Time in Trade" Filter:
- After opening a position, we hold it for X bars (unless a stop is hit).
- This prevents rapid flipping on each bar.

Example approach:
    - We maintain a global bar_index that increments each call.
    - We also track last_trade_bar[ticker] to know when we last changed positions.
    - If bar_index - last_trade_bar[ticker] < HOLD_BARS, we skip opening or flipping.
"""

import logging
import os
from dotenv import load_dotenv

# Load environment variables from the .env file.
load_dotenv()

# ---------------------------
# Live Trading Global State
# ---------------------------
bar_index = 0
last_trade_bar = {}  # Dictionary to track each ticker's last trade bar

# ---------------------------
# Backtesting Global State
# ---------------------------
bt_bar_index = 0
bt_last_trade_bar = {}  # Separate backtesting state (per ticker)


def run_logic(current_price, predicted_price, ticker):
    """
    Live trading function called every time a new bar (or cycle) arrives.

    Trading logic:
      - If no position is open, open one if the signal is strong.
      - If a position is open and the cooldown period has elapsed, then
        re-evaluate and possibly flip the position.
    """
    from forest import api, buy_shares, sell_shares, short_shares, close_short

    global bar_index, last_trade_bar
    bar_index += 1

    # Define a minimal hold period in bars and a directional threshold (in percent)
    HOLD_BARS = 3  # Must hold at least 3 bars after opening a position
    threshold_pct = 0.5

    # Fetch current position information.
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

    # Initialize last_trade_bar for this ticker if not already present.
    if ticker not in last_trade_bar:
        last_trade_bar[ticker] = -9999  # A very low number to allow immediate trade if needed

    # Compute the percent difference between predicted and current price.
    diff_pct = (predicted_price - current_price) / (current_price if current_price else 1e-6) * 100

    # Check if the cooldown period has passed.
    can_flip = (bar_index - last_trade_bar[ticker]) >= HOLD_BARS

    # If no position is held, open a new one if the signal is strong.
    if current_position == "none":
        if diff_pct > threshold_pct:
            logging.info(f"[{ticker}] Cooldown: Opening LONG (no position).")
            account = api.get_account()
            cash = float(account.cash)
            shares_to_buy = int(cash // current_price)
            if shares_to_buy > 0:
                buy_shares(ticker, shares_to_buy, current_price, predicted_price)
                last_trade_bar[ticker] = bar_index
        elif diff_pct < -threshold_pct:
            logging.info(f"[{ticker}] Cooldown: Opening SHORT (no position).")
            account = api.get_account()
            cash = float(account.cash)
            shares_to_short = int(cash // current_price)
            if shares_to_short > 0:
                short_shares(ticker, shares_to_short, current_price, predicted_price)
                last_trade_bar[ticker] = bar_index
        else:
            logging.info(f"[{ticker}] Cooldown: No position and signal is FLAT. Doing nothing.")
    else:
        # A position is already held.
        if not can_flip:
            # Still within the hold period: do nothing.
            logging.info(f"[{ticker}] Cooldown: Holding {current_position} (bar_index={bar_index}, last_trade_bar={last_trade_bar[ticker]}).")
        else:
            # Cooldown period has passed: check if we need to flip the position.
            if current_position == "long":
                if diff_pct < -threshold_pct:
                    logging.info(f"[{ticker}] Cooldown: Model indicates DOWN, flipping to SHORT.")
                    sell_shares(ticker, position_qty, current_price, predicted_price)
                    account = api.get_account()
                    cash = float(account.cash)
                    shares_to_short = int(cash // current_price)
                    if shares_to_short > 0:
                        short_shares(ticker, shares_to_short, current_price, predicted_price)
                    last_trade_bar[ticker] = bar_index
                else:
                    logging.info(f"[{ticker}] Cooldown: Model not strongly negative => remain LONG.")
            elif current_position == "short":
                if diff_pct > threshold_pct:
                    logging.info(f"[{ticker}] Cooldown: Model indicates UP, flipping to LONG.")
                    close_short(ticker, abs(position_qty), current_price)
                    account = api.get_account()
                    cash = float(account.cash)
                    shares_to_buy = int(cash // current_price)
                    if shares_to_buy > 0:
                        buy_shares(ticker, shares_to_buy, current_price, predicted_price)
                    last_trade_bar[ticker] = bar_index
                else:
                    logging.info(f"[{ticker}] Cooldown: Model not strongly positive => remain SHORT.")


def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    """
    Backtest function that applies the same cooldown trading logic as run_logic.

    Parameters:
        current_price (float): The current price.
        predicted_price (float): The predicted price.
        position_qty (float): Current position quantity (positive for long, negative for short).

    Returns:
        str: One of the following actions: BUY, SELL, SHORT, COVER, or NONE.
             In this cooldown logic:
               - "BUY" indicates opening (or flipping to) a long position.
               - "SHORT" indicates opening (or flipping to) a short position.
               - "NONE" indicates no action taken.
             (The actions SELL and COVER are reserved for other strategies.)
    """
    global bt_bar_index, bt_last_trade_bar
    bt_bar_index += 1

    # Obtain the ticker from environment variables (assuming a single-ticker backtest)
    ticker = os.getenv("TICKERS")
    if not ticker:
        ticker = "DEFAULT"

    HOLD_BARS = 3  # Minimal hold period in bars
    threshold_pct = 0.5

    # Determine current position from position_qty.
    if position_qty > 0:
        current_position = "long"
    elif position_qty < 0:
        current_position = "short"
    else:
        current_position = "none"

    # Initialize the backtest last_trade_bar for this ticker if not already set.
    if ticker not in bt_last_trade_bar:
        bt_last_trade_bar[ticker] = -9999

    # Calculate the percent difference between predicted and current price.
    diff_pct = (predicted_price - current_price) / (current_price if current_price else 1e-6) * 100

    # Check if the cooldown period has passed.
    can_flip = (bt_bar_index - bt_last_trade_bar[ticker]) >= HOLD_BARS

    # Default action.
    action = "NONE"

    if current_position == "none":
        # No open position: open one if signal is strong.
        if diff_pct > threshold_pct:
            action = "BUY"
            bt_last_trade_bar[ticker] = bt_bar_index
        elif diff_pct < -threshold_pct:
            action = "SHORT"
            bt_last_trade_bar[ticker] = bt_bar_index
        else:
            action = "NONE"
    else:
        # A position is already held.
        if not can_flip:
            action = "NONE"
        else:
            if current_position == "long":
                if diff_pct < -threshold_pct:
                    # Flip from long to short.
                    action = "SHORT"
                    bt_last_trade_bar[ticker] = bt_bar_index
                else:
                    action = "NONE"
            elif current_position == "short":
                if diff_pct > threshold_pct:
                    # Flip from short to long.
                    action = "BUY"
                    bt_last_trade_bar[ticker] = bt_bar_index
                else:
                    action = "NONE"

    return action
