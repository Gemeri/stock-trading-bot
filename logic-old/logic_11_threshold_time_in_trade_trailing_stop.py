"""
logic_11_threshold_time_in_trade_trailing_stop.py

Combines:
 1) Threshold / No-Trade Zone (#1)
 7) Time-in-Trade / Cooldown (#7)
 3) Trailing Stop (#3 variation)

A. run_logic(...) => incremental approach for live Alpaca usage
   - We track the last trade bar index to enforce a time-in-trade hold.
   - We only enter if predicted move beyond threshold => open position.
   - We track a trailing stop that updates as price moves in our favor.
   - If the trailing stop triggers, we exit. If we've held enough bars, we can
     reevaluate flipping or exit if the predicted move is no longer strong.

B. run_backtest(...) => offline approach
   - Called externally with: current_price, predicted_price, position_qty
   - Implements the same logic as run_logic.
   - Returns one of the following action strings: "BUY", "SELL", "SHORT", "COVER", or "NONE"

DEPENDENCIES:
  pip install pandas numpy
"""

import logging
import math
import numpy as np
import pandas as pd

##############################################
# Module-level containers for the live usage & backtesting
##############################################
price_data = {}      # {ticker: DataFrame storing the bars, e.g. columns ["Close", "PredictedClose"]}
position_state = {}  # {ticker: dict with keys: "position", "entry_price", "best_price_for_stop", "bars_in_trade", "last_trade_bar"}
bar_counter = {}     # {ticker: integer counting how many bars have passed (for multi-bar horizon hold)}

##############################################
# 1) Helper Functions
##############################################
def predicted_move_pct(current_close, predicted_close):
    """
    Return the predicted move as a percentage:
      ((predicted_close - current_close) / current_close) * 100
    """
    if current_close <= 0:
        return 0
    return (predicted_close - current_close) / current_close * 100

##############################################
# 2) Live / Incremental run_logic(...)
##############################################
def run_logic(current_price, predicted_price, ticker):
    """
    Called each bar by your main script for real-time Alpaca usage.
    Implements:
      - Threshold / no-trade zone (#1)
      - Trailing stop (#3)
      - Time-in-trade / cooldown (#7)

    Steps:
      1) Maintain a DataFrame price_data[ticker] with columns ["Close", "PredictedClose"].
      2) Maintain position_state[ticker] = { "position": "LONG"/"SHORT"/None,
                                             "entry_price": float,
                                             "best_price_for_stop": float,
                                             "bars_in_trade": int,
                                             "last_trade_bar": int }
      3) Only enter a new trade if (bar_counter[ticker] - last_trade_bar) >= hold_bars.
      4) If in a position, update trailing stop logic as price moves in your favor.
      5) If trailing stop triggers, exit. If hold time is up & predicted move is not strong, exit or flip.
    """
    from forest import api, buy_shares, sell_shares, short_shares, close_short

    # Strategy parameters
    threshold_pct = 0.5      # (#1) minimum % difference to open trades
    hold_bars = 3            # (#7) must hold for at least 3 bars
    trailing_stop_pct = 2.0  # (#3) trailing stop in %

    # Prepare the data containers if not already present
    if ticker not in price_data:
        df_init = pd.DataFrame([], columns=["Close", "PredictedClose"])
        price_data[ticker] = df_init
        position_state[ticker] = {
            "position": None,
            "entry_price": 0.0,
            "best_price_for_stop": None,
            "bars_in_trade": 0,
            "last_trade_bar": -9999
        }
        bar_counter[ticker] = -1

    bar_counter[ticker] += 1
    current_bar_index = bar_counter[ticker]

    # Append the new row to the DataFrame (replace .append with pd.concat)
    df = price_data[ticker]
    new_row = pd.DataFrame([{
        "Close": current_price,
        "PredictedClose": predicted_price
    }])
    df = pd.concat([df, new_row], ignore_index=True)

    # Retrieve current position state
    pos_info = position_state[ticker]
    current_position = pos_info["position"]
    entry_price = pos_info["entry_price"]
    best_price_for_stop = pos_info["best_price_for_stop"]
    bars_in_trade = pos_info["bars_in_trade"]
    last_trade_bar = pos_info["last_trade_bar"]

    # If in a position, increment bars_in_trade
    if current_position in ("LONG", "SHORT"):
        bars_in_trade += 1

    # 1) Update trailing stop logic
    if current_position == "LONG":
        # Update best_price_for_stop if current price is higher
        if best_price_for_stop is None or current_price > best_price_for_stop:
            best_price_for_stop = current_price

        # Compute trailing stop price
        stop_price = best_price_for_stop * (1 - trailing_stop_pct / 100)
        if current_price < stop_price:
            logging.info(f"[{ticker}] THR+TIME+TRAIL: Trailing stop triggered on LONG.")
            try:
                pos = api.get_position(ticker)
                qty = float(pos.qty)
                if qty > 0:
                    sell_shares(ticker, qty, current_price, predicted_price)
            except Exception:
                pass
            pos_info["position"] = None
            pos_info["entry_price"] = 0.0
            pos_info["best_price_for_stop"] = None
            pos_info["bars_in_trade"] = 0
            pos_info["last_trade_bar"] = current_bar_index

    elif current_position == "SHORT":
        # Update best_price_for_stop if current price is lower (in our favor)
        if best_price_for_stop is None or current_price < best_price_for_stop:
            best_price_for_stop = current_price

        # Compute trailing stop price for shorts
        stop_price = best_price_for_stop * (1 + trailing_stop_pct / 100)
        if current_price > stop_price:
            logging.info(f"[{ticker}] THR+TIME+TRAIL: Trailing stop triggered on SHORT.")
            try:
                pos = api.get_position(ticker)
                qty = float(pos.qty)
                if qty < 0:
                    close_short(ticker, abs(qty), current_price)
            except Exception:
                pass
            pos_info["position"] = None
            pos_info["entry_price"] = 0.0
            pos_info["best_price_for_stop"] = None
            pos_info["bars_in_trade"] = 0
            pos_info["last_trade_bar"] = current_bar_index

    # 2) If in a position and held for at least hold_bars, re-check if predicted move is still strong
    if current_position in ("LONG", "SHORT") and bars_in_trade >= hold_bars:
        predicted_now = predicted_move_pct(current_price, predicted_price)
        if current_position == "LONG":
            if predicted_now < threshold_pct:
                logging.info(f"[{ticker}] THR+TIME+TRAIL: Time up on LONG, predicted move not strong => exit.")
                try:
                    pos = api.get_position(ticker)
                    qty = float(pos.qty)
                    if qty > 0:
                        sell_shares(ticker, qty, current_price, predicted_price)
                except Exception:
                    pass
                pos_info["position"] = None
                pos_info["entry_price"] = 0.0
                pos_info["best_price_for_stop"] = None
                pos_info["bars_in_trade"] = 0
                pos_info["last_trade_bar"] = current_bar_index
        else:  # SHORT
            if predicted_now > -threshold_pct:
                logging.info(f"[{ticker}] THR+TIME+TRAIL: Time up on SHORT, predicted move not strong => exit.")
                try:
                    pos = api.get_position(ticker)
                    qty = float(pos.qty)
                    if qty < 0:
                        close_short(ticker, abs(qty), current_price)
                except Exception:
                    pass
                pos_info["position"] = None
                pos_info["entry_price"] = 0.0
                pos_info["best_price_for_stop"] = None
                pos_info["bars_in_trade"] = 0
                pos_info["last_trade_bar"] = current_bar_index

    # 3) If no current position, check if we can open a new trade
    if pos_info["position"] is None:
        if (current_bar_index - last_trade_bar) >= hold_bars:
            pmove = predicted_move_pct(current_price, predicted_price)
            if pmove > 0 and pmove >= threshold_pct:
                logging.info(f"[{ticker}] THR+TIME+TRAIL: Opening LONG (predicted move = {pmove:.2f}%).")
                try:
                    account = api.get_account()
                    cash = float(account.cash)
                    shares = int(cash // current_price)
                    if shares > 0:
                        buy_shares(ticker, shares, current_price, predicted_price)
                        pos_info["position"] = "LONG"
                        pos_info["entry_price"] = current_price
                        pos_info["best_price_for_stop"] = current_price
                        pos_info["bars_in_trade"] = 0
                except Exception:
                    pass
            elif pmove < 0 and pmove <= -threshold_pct:
                logging.info(f"[{ticker}] THR+TIME+TRAIL: Opening SHORT (predicted move = {pmove:.2f}%).")
                try:
                    account = api.get_account()
                    cash = float(account.cash)
                    shares_to_short = int(cash // current_price)
                    if shares_to_short > 0:
                        short_shares(ticker, shares_to_short, current_price, predicted_price)
                        pos_info["position"] = "SHORT"
                        pos_info["entry_price"] = current_price
                        pos_info["best_price_for_stop"] = current_price
                        pos_info["bars_in_trade"] = 0
                except Exception:
                    pass

    # Update state variables
    pos_info["bars_in_trade"] = bars_in_trade
    price_data[ticker] = df
    position_state[ticker] = pos_info

##############################################
# 3) Offline BACKTEST run_backtest(...)
##############################################
def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    """
    Backtest function to be called externally with:
      - current_price: the current bar's price.
      - predicted_price: the predicted price.
      - position_qty: current position size (positive for LONG, negative for SHORT, 0 for no position).

    Implements the same trading logic as run_logic:
      - Threshold / no-trade zone, trailing stop, and time-in-trade/cooldown.
    Returns one of the following action strings:
         "BUY"   - Open a long position.
         "SELL"  - Exit a long position.
         "SHORT" - Open a short position.
         "COVER" - Exit a short position.
         "NONE"  - No action.
    """
    # Strategy parameters (must match run_logic)
    threshold_pct = 0.5      # minimum % difference to trigger trade
    hold_bars = 3            # must hold for at least 3 bars before reevaluation
    trailing_stop_pct = 2.0  # trailing stop in %

    # Use a fixed ticker for backtesting purposes
    ticker = "BACKTEST"

    # Initialize data containers for backtesting if not already set
    if ticker not in price_data:
        df_init = pd.DataFrame([], columns=["Close", "PredictedClose"])
        price_data[ticker] = df_init
        # Determine initial position from position_qty
        if position_qty > 0:
            init_position = "LONG"
        elif position_qty < 0:
            init_position = "SHORT"
        else:
            init_position = None
        position_state[ticker] = {
            "position": init_position,
            "entry_price": 0.0,
            "best_price_for_stop": None,
            "bars_in_trade": 0,
            "last_trade_bar": -9999
        }
        bar_counter[ticker] = -1

    bar_counter[ticker] += 1
    current_bar_index = bar_counter[ticker]

    # Append new bar data (replace .append with pd.concat)
    df = price_data[ticker]
    new_row = pd.DataFrame([{
        "Close": current_price,
        "PredictedClose": predicted_price
    }])
    df = pd.concat([df, new_row], ignore_index=True)

    # Retrieve current backtest state
    pos_info = position_state[ticker]
    current_position = pos_info["position"]
    entry_price = pos_info["entry_price"]
    best_price_for_stop = pos_info["best_price_for_stop"]
    bars_in_trade = pos_info["bars_in_trade"]
    last_trade_bar = pos_info["last_trade_bar"]

    # Increment bars_in_trade if in a position
    if current_position in ("LONG", "SHORT"):
        bars_in_trade += 1

    # 1) Update trailing stop logic (simulate exit)
    if current_position == "LONG":
        if best_price_for_stop is None or current_price > best_price_for_stop:
            best_price_for_stop = current_price
        stop_price = best_price_for_stop * (1 - trailing_stop_pct / 100)
        if current_price < stop_price:
            logging.info("[BACKTEST] THR+TIME+TRAIL: Trailing stop triggered on LONG. Exiting trade.")
            pos_info["position"] = None
            pos_info["entry_price"] = 0.0
            pos_info["best_price_for_stop"] = None
            pos_info["bars_in_trade"] = 0
            pos_info["last_trade_bar"] = current_bar_index
            price_data[ticker] = df
            position_state[ticker] = pos_info
            return "SELL"

    elif current_position == "SHORT":
        if best_price_for_stop is None or current_price < best_price_for_stop:
            best_price_for_stop = current_price
        stop_price = best_price_for_stop * (1 + trailing_stop_pct / 100)
        if current_price > stop_price:
            logging.info("[BACKTEST] THR+TIME+TRAIL: Trailing stop triggered on SHORT. Covering trade.")
            pos_info["position"] = None
            pos_info["entry_price"] = 0.0
            pos_info["best_price_for_stop"] = None
            pos_info["bars_in_trade"] = 0
            pos_info["last_trade_bar"] = current_bar_index
            price_data[ticker] = df
            position_state[ticker] = pos_info
            return "COVER"

    # 2) If in a position and held for at least hold_bars, re-check predicted move strength
    if current_position in ("LONG", "SHORT") and bars_in_trade >= hold_bars:
        predicted_now = predicted_move_pct(current_price, predicted_price)
        if current_position == "LONG":
            if predicted_now < threshold_pct:
                logging.info("[BACKTEST] THR+TIME+TRAIL: Time up on LONG and predicted move not strong. Exiting trade.")
                pos_info["position"] = None
                pos_info["entry_price"] = 0.0
                pos_info["best_price_for_stop"] = None
                pos_info["bars_in_trade"] = 0
                pos_info["last_trade_bar"] = current_bar_index
                price_data[ticker] = df
                position_state[ticker] = pos_info
                return "SELL"
        else:  # SHORT
            if predicted_now > -threshold_pct:
                logging.info("[BACKTEST] THR+TIME+TRAIL: Time up on SHORT and predicted move not strong. Exiting trade.")
                pos_info["position"] = None
                pos_info["entry_price"] = 0.0
                pos_info["best_price_for_stop"] = None
                pos_info["bars_in_trade"] = 0
                pos_info["last_trade_bar"] = current_bar_index
                price_data[ticker] = df
                position_state[ticker] = pos_info
                return "COVER"

    # 3) If no open position, check if we can open a new trade
    action = "NONE"
    if pos_info["position"] is None:
        if (current_bar_index - last_trade_bar) >= hold_bars:
            pmove = predicted_move_pct(current_price, predicted_price)
            if pmove > 0 and pmove >= threshold_pct:
                logging.info(f"[BACKTEST] THR+TIME+TRAIL: Opening LONG (predicted move = {pmove:.2f}%).")
                pos_info["position"] = "LONG"
                pos_info["entry_price"] = current_price
                pos_info["best_price_for_stop"] = current_price
                pos_info["bars_in_trade"] = 0
                action = "BUY"
            elif pmove < 0 and pmove <= -threshold_pct:
                logging.info(f"[BACKTEST] THR+TIME+TRAIL: Opening SHORT (predicted move = {pmove:.2f}%).")
                pos_info["position"] = "SHORT"
                pos_info["entry_price"] = current_price
                pos_info["best_price_for_stop"] = current_price
                pos_info["bars_in_trade"] = 0
                action = "SHORT"
    else:
        action = "NONE"

    # Update state variables
    pos_info["bars_in_trade"] = bars_in_trade
    price_data[ticker] = df
    position_state[ticker] = pos_info

    return action
