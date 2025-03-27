"""
logic_12_risk_to_reward_classification_stoploss.py

Combines:
 6) Risk‐to‐Reward Filter => predicted move >= some multiple * ATR
 5) Classification => direction is up (+1) or down (-1)
 3) Stop‐Loss => exit if price moves against you by stop_loss_pct%

Two usage modes:
 A) run_logic(...) => incremental for live Alpaca usage
 B) run_backtest(...) => offline reading CSV with columns [Date,Open,High,Low,Close,...],
    plus optional 'PredictedClose' or label to classify direction

DEPENDENCIES:
 - pandas, numpy
 - You should have the main script with buy_shares, sell_shares, short_shares, close_short, etc.
"""

import logging
import math
import numpy as np
import pandas as pd

#############################
# Module-level for live usage
#############################
price_data = {}      # {ticker: DataFrame with columns ["Open", "High", "Low", "Close", "ATR", ...]}
position_state = {}  # {ticker: {"position": "LONG"/"SHORT"/None, "entry_price": float, "bars_in_trade": int}}
bar_counter = {}     # for each ticker, count how many bars processed

#############################
# 1) Helper Functions
#############################
def compute_atr(df, window=14):
    """
    Compute ATR using High, Low, and previous Close.
    'df' must have columns: 'High', 'Low', 'Close'.
    Returns a Series.
    """
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr

def classification_direction(current_price, predicted_price, threshold_pct=0.0):
    """
    Basic classification (#5):
      - If predicted is above current by more than threshold_pct => +1
      - If predicted is below current by more than threshold_pct => -1
      - Otherwise, returns 0.
    """
    if current_price <= 0:
        return 0
    diff = (predicted_price - current_price) / current_price * 100
    if diff > threshold_pct:
        return 1
    elif diff < -threshold_pct:
        return -1
    else:
        return 0

def risk_to_reward_ok(current_price, predicted_price, atr_value, rr_multiple=1.0):
    """
    (#6) Check if the absolute predicted move is at least rr_multiple * ATR.
    Returns True if risk-to-reward condition is met; otherwise, False.
    """
    if pd.isna(atr_value) or atr_value <= 0:
        return False
    diff_abs = abs(predicted_price - current_price)
    return diff_abs >= (rr_multiple * atr_value)

#############################
# 2) run_logic(...) for live usage
#############################
def run_logic(current_price, predicted_price, ticker):
    """
    Called each bar by the main script for live trading.
    Steps:
      1) Maintain a DataFrame in price_data[ticker]. (Here, we fabricate bar data from current_price.)
      2) Compute or update ATR, classify direction, and check risk-to-reward.
      3) If classification is +1, we aim for LONG; if -1, SHORT.
      4) Apply stop-loss: exit if price moves against you by stop_loss_pct%.
    """
    from forest import api, buy_shares, sell_shares, short_shares, close_short

    # Strategy configuration
    atr_window = 14
    rr_multiple = 1.0
    stop_loss_pct = 3.0

    # Prepare data structures if not present
    if ticker not in price_data:
        cols = ["Open", "High", "Low", "Close", "ATR"]
        price_data[ticker] = pd.DataFrame([], columns=cols)
        position_state[ticker] = {
            "position": None,
            "entry_price": 0.0,
            "bars_in_trade": 0
        }
        bar_counter[ticker] = -1

    bar_counter[ticker] += 1
    idx = bar_counter[ticker]

    # Fabricate a bar (replace with real data if available)
    new_open = current_price
    new_high = current_price * 1.01
    new_low  = current_price * 0.99
    new_close = current_price

    df = price_data[ticker]
    # Replace .append with pd.concat
    new_row_df = pd.DataFrame([{
        "Open": new_open,
        "High": new_high,
        "Low":  new_low,
        "Close": new_close,
        "ATR": np.nan  # will update below
    }])
    df = pd.concat([df, new_row_df], ignore_index=True)

    # Compute ATR for the DataFrame if there is enough data
    if len(df) > 1:
        df['ATR'] = compute_atr(df, window=atr_window)

    # Determine direction and risk-to-reward condition
    direct = classification_direction(current_price, predicted_price, threshold_pct=0.0)
    atr_value = df['ATR'].iloc[-1] if len(df) > 0 else np.nan
    rr_ok = risk_to_reward_ok(current_price, predicted_price, atr_value, rr_multiple=rr_multiple)

    # Retrieve current state
    state = position_state[ticker]
    pos = state["position"]
    e_price = state["entry_price"]
    bars_in_trade = state["bars_in_trade"]
    if pos in ("LONG", "SHORT"):
        bars_in_trade += 1

    # Apply stop-loss if in position
    if pos == "LONG":
        if current_price < e_price * (1 - stop_loss_pct / 100):
            logging.info(f"[{ticker}] R2R+Class+Stop: LONG Stopped out.")
            try:
                position_obj = api.get_position(ticker)
                qty = float(position_obj.qty)
                if qty > 0:
                    sell_shares(ticker, qty, current_price, predicted_price)
            except Exception:
                pass
            pos = None
            e_price = 0.0
            bars_in_trade = 0
    elif pos == "SHORT":
        if current_price > e_price * (1 + stop_loss_pct / 100):
            logging.info(f"[{ticker}] R2R+Class+Stop: SHORT Stopped out.")
            try:
                position_obj = api.get_position(ticker)
                qty = float(position_obj.qty)
                if qty < 0:
                    close_short(ticker, abs(qty), current_price)
            except Exception:
                pass
            pos = None
            e_price = 0.0
            bars_in_trade = 0

    # If no position, open new trade if conditions are met
    if pos is None:
        if direct == 1 and rr_ok:
            logging.info(f"[{ticker}] R2R+Class+Stop: Opening LONG.")
            try:
                account = api.get_account()
                cash = float(account.cash)
                shares = int(cash // current_price)
                if shares > 0:
                    buy_shares(ticker, shares, current_price, predicted_price)
                    pos = "LONG"
                    e_price = current_price
                    bars_in_trade = 0
            except Exception:
                pass
        elif direct == -1 and rr_ok:
            logging.info(f"[{ticker}] R2R+Class+Stop: Opening SHORT.")
            try:
                account = api.get_account()
                cash = float(account.cash)
                shares_to_short = int(cash // current_price)
                if shares_to_short > 0:
                    short_shares(ticker, shares_to_short, current_price, predicted_price)
                    pos = "SHORT"
                    e_price = current_price
                    bars_in_trade = 0
            except Exception:
                pass

    # Update state and global data
    state["position"] = pos
    state["entry_price"] = e_price
    state["bars_in_trade"] = bars_in_trade
    price_data[ticker] = df
    position_state[ticker] = state

#############################
# 3) Offline BACKTEST run_backtest(...)
#############################
def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    """
    Backtest function to be called externally with:
      - current_price: the current bar's price.
      - predicted_price: the predicted price.
      - position_qty: current position size (positive for LONG, negative for SHORT, 0 for no position).

    Implements the same trading logic as run_logic:
      - It computes ATR, applies a risk-to-reward filter, classifies direction,
        and checks for stop-loss conditions.
    Returns one of the following action strings:
         "BUY"   - Open a long position.
         "SELL"  - Exit a long position.
         "SHORT" - Open a short position.
         "COVER" - Exit a short position.
         "NONE"  - No action.
    """
    # Strategy configuration (must match run_logic)
    atr_window = 14
    rr_multiple = 1.0
    stop_loss_pct = 3.0

    # Use a fixed ticker for backtesting
    ticker = "BACKTEST"

    # Initialize backtest data structures if not present
    if ticker not in price_data:
        cols = ["Open", "High", "Low", "Close", "ATR"]
        price_data[ticker] = pd.DataFrame([], columns=cols)
        # Set initial position based on position_qty
        if position_qty > 0:
            init_position = "LONG"
        elif position_qty < 0:
            init_position = "SHORT"
        else:
            init_position = None
        position_state[ticker] = {
            "position": init_position,
            "entry_price": 0.0,
            "bars_in_trade": 0
        }
        bar_counter[ticker] = -1

    bar_counter[ticker] += 1
    idx = bar_counter[ticker]

    # Fabricate a bar
    new_open = current_price
    new_high = current_price * 1.01
    new_low  = current_price * 0.99
    new_close = current_price

    df = price_data[ticker]
    # Replace .append with pd.concat
    new_row_df = pd.DataFrame([{
        "Open": new_open,
        "High": new_high,
        "Low":  new_low,
        "Close": new_close,
        "ATR": np.nan
    }])
    df = pd.concat([df, new_row_df], ignore_index=True)

    # Compute ATR if possible
    if len(df) > 1:
        df['ATR'] = compute_atr(df, window=atr_window)

    # Determine classification direction and risk-to-reward
    direct = classification_direction(current_price, predicted_price, threshold_pct=0.0)
    atr_value = df['ATR'].iloc[-1] if len(df) > 0 else np.nan
    rr_ok = risk_to_reward_ok(current_price, predicted_price, atr_value, rr_multiple=rr_multiple)

    # Retrieve backtest state
    state = position_state[ticker]
    pos = state["position"]
    e_price = state["entry_price"]
    bars_in_trade = state["bars_in_trade"]
    if pos in ("LONG", "SHORT"):
        bars_in_trade += 1

    # Simulate stop-loss exit
    if pos == "LONG":
        if current_price < e_price * (1 - stop_loss_pct / 100):
            logging.info("[BACKTEST] R2R+Class+Stop: LONG Stopped out. Exiting trade.")
            state["position"] = None
            state["entry_price"] = 0.0
            state["bars_in_trade"] = 0
            price_data[ticker] = df
            position_state[ticker] = state
            return "SELL"
    elif pos == "SHORT":
        if current_price > e_price * (1 + stop_loss_pct / 100):
            logging.info("[BACKTEST] R2R+Class+Stop: SHORT Stopped out. Covering trade.")
            state["position"] = None
            state["entry_price"] = 0.0
            state["bars_in_trade"] = 0
            price_data[ticker] = df
            position_state[ticker] = state
            return "COVER"

    # If no open position, check if we should open a new trade
    action = "NONE"
    if state["position"] is None:
        if direct == 1 and rr_ok:
            logging.info(f"[BACKTEST] R2R+Class+Stop: Opening LONG (predicted move qualifies).")
            state["position"] = "LONG"
            state["entry_price"] = current_price
            state["bars_in_trade"] = 0
            action = "BUY"
        elif direct == -1 and rr_ok:
            logging.info(f"[BACKTEST] R2R+Class+Stop: Opening SHORT (predicted move qualifies).")
            state["position"] = "SHORT"
            state["entry_price"] = current_price
            state["bars_in_trade"] = 0
            action = "SHORT"
    else:
        action = "NONE"

    state["bars_in_trade"] = bars_in_trade
    price_data[ticker] = df
    position_state[ticker] = state
    return action
