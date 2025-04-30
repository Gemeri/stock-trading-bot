"""
logic_10_classification_stop_loss_multi_bar.py

Combines:
 (5) Classification: Up, Down, (or Flat)
 (3) Stop Loss: Fixed % stop
 (4) Multi-bar horizon: Trade only once every N bars

Supports both:
 1) A bar-by-bar incremental approach for live Alpaca trades (run_logic(...))
 2) Offline backtesting using run_backtest(...).

DEPENDENCIES:
 - pandas, numpy
"""

import logging
import math
import numpy as np
import pandas as pd


########################################
# Module-level containers for live usage & backtesting
########################################
price_data = {}      # { ticker: DataFrame of historical bars for incremental usage }
position_state = {}  # { ticker: {'position': 'LONG'/'SHORT'/None, 'entry_price': float, 'last_trade_bar': int } }
bar_counter = {}     # { ticker: number of bars processed so far (to manage multi-bar horizon skipping) }


########################################
# 1) HELPER FUNCTIONS
########################################
def classify_signal_by_threshold(current_price, predicted_price, threshold_pct=0.5):
    """
    Classification logic (#5):
     - If predicted > current*(1+threshold_pct/100) => UP
     - If predicted < current*(1-threshold_pct/100) => DOWN
     - Else => FLAT
    Return +1 for UP, -1 for DOWN, 0 for FLAT
    """
    if current_price <= 0:
        return 0
    diff_pct = (predicted_price - current_price) / current_price * 100
    if diff_pct > threshold_pct:
        return 1
    elif diff_pct < -threshold_pct:
        return -1
    else:
        return 0


########################################
# 2) LIVE / INCREMENTAL run_logic(...)
########################################
def run_logic(current_price, predicted_price, ticker):
    """
    Called by main script each new bar for Alpaca trading.
    1) Track multi-bar horizon => only trade if enough bars passed.
    2) Classify predicted vs current => +1 (UP), -1 (DOWN), else 0 (FLAT).
    3) Stop loss => if price moves stop_loss_pct% against entry, exit.
    4) Open position only if we haven't traded within 'trade_every_n_bars' bars.
    """
    from forest import api, buy_shares, sell_shares, short_shares, close_short

    # Customizable parameters:
    threshold_pct = 0.5       # classification threshold
    stop_loss_pct = 3.0       # fixed percentage stop loss
    trade_every_n_bars = 4    # multi-bar horizon (only trade if enough bars have passed)

    # Setup dictionary structures if not present for this ticker
    if ticker not in price_data:
        cols = ["Close"]  # tracking only Close price for this example
        price_data[ticker] = pd.DataFrame([], columns=cols)
        position_state[ticker] = {
            'position': None,        # "LONG", "SHORT", or None
            'entry_price': 0.0,
            'last_trade_bar': -9999  # last bar index when a trade was made
        }
        bar_counter[ticker] = -1

    bar_counter[ticker] += 1
    current_bar_index = bar_counter[ticker]

    # Append new bar data (replacing .append with pd.concat)
    df = price_data[ticker]
    new_row_df = pd.DataFrame([{"Close": current_price}])
    df = pd.concat([df, new_row_df], ignore_index=True)

    # Retrieve position information
    pos_info = position_state[ticker]
    current_position = pos_info['position']
    entry_price = pos_info['entry_price']

    # Check for stop-loss conditions:
    if current_position == "LONG":
        if current_price < entry_price * (1 - stop_loss_pct / 100):
            logging.info(f"[{ticker}] CLASS+STOP+MULTI: Stop-loss triggered on LONG. Exiting.")
            try:
                pos = api.get_position(ticker)
                qty = float(pos.qty)
                if qty > 0:
                    sell_shares(ticker, qty, current_price, predicted_price)
            except Exception:
                pass
            pos_info['position'] = None
            pos_info['entry_price'] = 0.0

    elif current_position == "SHORT":
        if current_price > entry_price * (1 + stop_loss_pct / 100):
            logging.info(f"[{ticker}] CLASS+STOP+MULTI: Stop-loss triggered on SHORT. Covering.")
            try:
                pos = api.get_position(ticker)
                qty = float(pos.qty)
                if qty < 0:
                    close_short(ticker, abs(qty), current_price)
            except Exception:
                pass
            pos_info['position'] = None
            pos_info['entry_price'] = 0.0

    # Determine the classification signal: +1 (UP), -1 (DOWN), or 0 (FLAT)
    classification = classify_signal_by_threshold(current_price, predicted_price, threshold_pct=threshold_pct)

    # Enforce multi-bar horizon: only trade if enough bars have passed since the last trade
    if (current_bar_index - pos_info['last_trade_bar']) < trade_every_n_bars:
        price_data[ticker] = df
        position_state[ticker] = pos_info
        return

    # If no current position, open one based on classification
    if pos_info['position'] is None:
        if classification == 1:
            # Open a long position
            logging.info(f"[{ticker}] CLASS+STOP+MULTI: classification=UP => opening LONG.")
            try:
                account = api.get_account()
                cash = float(account.cash)
                shares_to_buy = int(cash // current_price)
                if shares_to_buy > 0:
                    buy_shares(ticker, shares_to_buy, current_price, predicted_price)
                    pos_info['position'] = "LONG"
                    pos_info['entry_price'] = current_price
                    pos_info['last_trade_bar'] = current_bar_index
            except Exception:
                pass

        elif classification == -1:
            # Open a short position
            logging.info(f"[{ticker}] CLASS+STOP+MULTI: classification=DOWN => opening SHORT.")
            try:
                account = api.get_account()
                cash = float(account.cash)
                shares_to_short = int(cash // current_price)
                if shares_to_short > 0:
                    short_shares(ticker, shares_to_short, current_price, predicted_price)
                    pos_info['position'] = "SHORT"
                    pos_info['entry_price'] = current_price
                    pos_info['last_trade_bar'] = current_bar_index
            except Exception:
                pass
        # If classification is 0, do nothing

    # Update global storage
    price_data[ticker] = df
    position_state[ticker] = pos_info


########################################
# 3) OFFLINE BACKTEST run_backtest(...)
########################################
def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    """
    Backtest function to be called externally with:
      - current_price: current bar's price.
      - predicted_price: predicted price (for evaluation/logging).
      - position_qty: current position size (positive for long, negative for short, 0 for no position).

    Implements the same trading logic as run_logic.
    Returns one of the following action strings:
        "BUY"   - Open a long position.
        "SELL"  - Exit a long position.
        "SHORT" - Open a short position.
        "COVER" - Exit a short position.
        "NONE"  - No action taken.
    """
    # Use the same parameters as in run_logic
    threshold_pct = 0.5       # classification threshold
    stop_loss_pct = 3.0       # stop loss percentage
    trade_every_n_bars = 4    # multi-bar horizon

    # Use a fixed ticker for backtesting purposes
    ticker = "BACKTEST"

    # Initialize global structures for backtesting if not already set
    if ticker not in price_data:
        cols = ["Close"]
        price_data[ticker] = pd.DataFrame([], columns=cols)
        # Initialize position based on position_qty
        if position_qty > 0:
            init_position = "LONG"
        elif position_qty < 0:
            init_position = "SHORT"
        else:
            init_position = None
        position_state[ticker] = {
            'position': init_position,
            'entry_price': 0.0,
            'last_trade_bar': -9999
        }
        bar_counter[ticker] = -1

    bar_counter[ticker] += 1
    current_bar_index = bar_counter[ticker]

    # Append new bar to historical data (replacing .append with pd.concat)
    df = price_data[ticker]
    new_row_df = pd.DataFrame([{"Close": current_price}])
    df = pd.concat([df, new_row_df], ignore_index=True)

    # Retrieve current position information
    pos_info = position_state[ticker]
    current_position = pos_info['position']
    entry_price = pos_info['entry_price']

    # Check stop-loss conditions
    if current_position == "LONG":
        if current_price < entry_price * (1 - stop_loss_pct / 100):
            # Stop-loss for LONG is triggered; action is to sell the position.
            logging.info("[BACKTEST] CLASS+STOP+MULTI: Stop-loss triggered on LONG. Exiting.")
            pos_info['position'] = None
            pos_info['entry_price'] = 0.0
            price_data[ticker] = df
            position_state[ticker] = pos_info
            return "SELL"

    elif current_position == "SHORT":
        if current_price > entry_price * (1 + stop_loss_pct / 100):
            # Stop-loss for SHORT is triggered; action is to cover the short position.
            logging.info("[BACKTEST] CLASS+STOP+MULTI: Stop-loss triggered on SHORT. Covering.")
            pos_info['position'] = None
            pos_info['entry_price'] = 0.0
            price_data[ticker] = df
            position_state[ticker] = pos_info
            return "COVER"

    # Classification:
    classification = classify_signal_by_threshold(current_price, predicted_price, threshold_pct=threshold_pct)

    # Enforce multi-bar horizon: only open a new position if enough bars have passed since the last trade
    if (current_bar_index - pos_info['last_trade_bar']) < trade_every_n_bars:
        price_data[ticker] = df
        position_state[ticker] = pos_info
        return "NONE"

    action = "NONE"  # Default action

    # If no position, then open a new one based on the classification signal
    if pos_info['position'] is None:
        if classification == 1:
            # Open a long position
            logging.info("[BACKTEST] CLASS+STOP+MULTI: classification=UP => opening LONG.")
            pos_info['position'] = "LONG"
            pos_info['entry_price'] = current_price
            pos_info['last_trade_bar'] = current_bar_index
            action = "BUY"
        elif classification == -1:
            # Open a short position
            logging.info("[BACKTEST] CLASS+STOP+MULTI: classification=DOWN => opening SHORT.")
            pos_info['position'] = "SHORT"
            pos_info['entry_price'] = current_price
            pos_info['last_trade_bar'] = current_bar_index
            action = "SHORT"
    # If already in a position and no stop loss or horizon conditions are met, then do nothing.

    price_data[ticker] = df
    position_state[ticker] = pos_info
    return action
