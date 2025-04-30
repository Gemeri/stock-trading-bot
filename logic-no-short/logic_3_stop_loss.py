"""
logic_3_stop_loss.py

Implements a Stop-Loss strategy (and optionally a trailing stop).

Key ideas:
- A fixed stop: e.g., 3% below your entry if long, or 3% above if short.
- A trailing stop: once the position has some unrealized gains,
  we can adjust the stop to lock in a portion of those gains.

Example usage:
stop_loss_pct = 3.0
if currently long and current_price < entry_price * (1 - stop_loss_pct/100):
    # exit
if currently short and current_price > entry_price * (1 + stop_loss_pct/100):
    # exit
"""

import logging

# Global variable to simulate the entry price in backtesting.
# In an actual backtesting environment, the entry price should be tracked per position.
BACKTEST_ENTRY_PRICE = None

def run_logic(current_price, predicted_price, ticker):
    """
    Live-trading version of the Stop-Loss strategy.

    Uses a fixed stop-loss percentage (and optional trailing stop logic) to exit positions
    if the market moves unfavorably. If no stop loss is triggered, the function then
    uses the predicted price relative to the current price to decide whether to open or
    flip a position.
    """
    from forest import api, buy_shares, sell_shares, short_shares, close_short

    # Decide threshold to open a trade (like a small 0.5% advantage).
    open_threshold_pct = 0.5
    # The stop-loss percentage for closing a trade if it goes against us.
    stop_loss_pct = 3.0

    # Optionally, you can define a trailing stop fraction for realized gains:
    # e.g. trailing_stop_frac = 0.5  # lock in 50% of the gains if they appear
    # For this minimal example, we'll just keep a fixed stop.

    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
        entry_price = float(pos.avg_entry_price)
        if position_qty > 0:
            current_position = "long"
        elif position_qty < 0:
            current_position = "short"
        else:
            current_position = "none"
    except Exception:
        position_qty = 0
        entry_price = 0.0
        current_position = "none"

    # ============= STOP-LOSS CHECK =============
    # If we have a position, see if we got stopped out:
    if current_position == "long":
        # If current price is X% below entry => exit
        if current_price < entry_price * (1 - stop_loss_pct / 100.0):
            logging.info(f"[{ticker}] StopLoss: Price triggered stop for LONG. Selling.")
            sell_shares(ticker, position_qty, current_price, predicted_price)
            return
        else:
            # Optional: check for trailing-stop logic if desired.
            pass

    elif current_position == "short":
        # If current price is X% above entry => exit
        if current_price > entry_price * (1 + stop_loss_pct / 100.0):
            logging.info(f"[{ticker}] StopLoss: Price triggered stop for SHORT. Covering.")
            close_short(ticker, abs(position_qty), current_price)
            return
        else:
            # Optional trailing stop logic for short could be added here.
            pass

    # ============= DECIDE DIRECTION BASED ON PREDICTED PRICE =============
    diff_pct = (predicted_price - current_price) / (current_price if current_price else 1e-6) * 100.0

    if diff_pct > open_threshold_pct:
        # We want to be or remain LONG
        if current_position == "long":
            logging.info(f"[{ticker}] StopLoss: Already long; no change.")
        elif current_position == "short":
            logging.info(f"[{ticker}] StopLoss: Predicted up, but currently short; flipping to long.")
            close_short(ticker, abs(position_qty), current_price)
            account = api.get_account()
            cash = float(account.cash)
            shares_to_buy = int(cash // current_price)
            if shares_to_buy > 0:
                buy_shares(ticker, shares_to_buy, current_price, predicted_price)
        else:
            # No position currently.
            account = api.get_account()
            cash = float(account.cash)
            shares_to_buy = int(cash // current_price)
            if shares_to_buy > 0:
                logging.info(f"[{ticker}] StopLoss: Opening new LONG position.")
                buy_shares(ticker, shares_to_buy, current_price, predicted_price)
            else:
                logging.info(f"[{ticker}] StopLoss: Not enough cash to open long.")

    elif diff_pct < -open_threshold_pct:
        # We want to be or remain SHORT
        if current_position == "short":
            logging.info(f"[{ticker}] StopLoss: Already short; no change.")
        elif current_position == "long":
            logging.info(f"[{ticker}] StopLoss: Predicted down, but currently long; flipping to short.")
            sell_shares(ticker, position_qty, current_price, predicted_price)
            account = api.get_account()
            cash = float(account.cash)
            shares_to_short = int(cash // current_price)
            if shares_to_short > 0:
                short_shares(ticker, shares_to_short, current_price, predicted_price)
        else:
            # No position currently.
            account = api.get_account()
            cash = float(account.cash)
            shares_to_short = int(cash // current_price)
            if shares_to_short > 0:
                logging.info(f"[{ticker}] StopLoss: Opening new SHORT position.")
                short_shares(ticker, shares_to_short, current_price, predicted_price)
            else:
                logging.info(f"[{ticker}] StopLoss: Not enough funds to open short.")
    else:
        # If predicted price is not sufficiently different, hold the current state.
        logging.info(f"[{ticker}] StopLoss: Model is not strongly directional. Holding current state.")

def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    """
    Backtest version of the Stop-Loss strategy.

    Parameters:
      - current_price: The current market price.
      - predicted_price: The model's predicted price.
      - position_qty: The current position quantity (positive for long, negative for short, 0 for none).
                     (A positive number indicates a long position; a negative number indicates a short position.)

    Returns:
      - action: A string representing the trade action:
          • 'BUY'   - Open a new long position.
          • 'SELL'  - Exit an existing long position.
          • 'SHORT' - Open a new short position.
          • 'COVER' - Exit an existing short position.
          • 'NONE'  - Hold the current position / no action.

    Trading Logic:
      1. **Stop-Loss Check:**  
         - If currently long and the current price falls below the entry price by more than stop_loss_pct (3%),
           return "SELL" to exit the long position.
         - If currently short and the current price rises above the entry price by more than stop_loss_pct,
           return "COVER" to exit the short position.

      2. **Directional Signal:**  
         If no stop-loss condition is met, compute the percentage difference between the predicted and current prices.
         - If the predicted price is more than open_threshold_pct (0.5%) above the current price:
             • If already long, return "NONE".
             • If short, return "COVER" (to exit the short, allowing a flip to long).
             • If no position, return "BUY".
         - If the predicted price is more than open_threshold_pct below the current price:
             • If already short, return "NONE".
             • If long, return "SELL" (to exit the long, allowing a flip to short).
             • If no position, return "SHORT".
         - Otherwise, return "NONE" (no action).
    
    Note:
      For backtesting, the stop-loss logic requires an entry price.
      We assume that if a position exists (position_qty != 0), the entry price is stored in the global variable BACKTEST_ENTRY_PRICE.
      If BACKTEST_ENTRY_PRICE is not set when a position is detected, it will be initialized to the current_price.
      In a complete backtesting system, the entry price should be tracked per position.
    """
    global BACKTEST_ENTRY_PRICE

    open_threshold_pct = 0.5
    stop_loss_pct = 3.0

    # Determine current position.
    if position_qty > 0:
        current_position = "long"
    elif position_qty < 0:
        current_position = "short"
    else:
        current_position = "none"

    # For backtesting, if a position is open, ensure we have an entry price.
    if current_position != "none":
        if BACKTEST_ENTRY_PRICE is None:
            BACKTEST_ENTRY_PRICE = current_price  # Assume trade opened at current_price if not set
        entry_price = BACKTEST_ENTRY_PRICE
    else:
        entry_price = None

    # ============= STOP-LOSS CHECK =============
    if current_position == "long":
        if current_price < entry_price * (1 - stop_loss_pct / 100.0):
            return "SELL"
    elif current_position == "short":
        if current_price > entry_price * (1 + stop_loss_pct / 100.0):
            return "COVER"

    # ============= DECIDE DIRECTION BASED ON PREDICTED PRICE =============
    diff_pct = (predicted_price - current_price) / (current_price if current_price else 1e-6) * 100.0

    if diff_pct > open_threshold_pct:
        if current_position == "long":
            return "NONE"
        elif current_position == "short":
            return "COVER"
        else:
            return "BUY"
    elif diff_pct < -open_threshold_pct:
        if current_position == "short":
            return "NONE"
        elif current_position == "long":
            return "SELL"
        else:
            return "SHORT"
    else:
        return "NONE"
