"""
logic_1_no_trade_zone.py

Implements a "No-Trade Zone" or Threshold-based trading strategy for both live trading and backtesting.

Live Trading (run_logic):
  - If (predicted_price - current_price)/current_price > threshold_pct/100, go/hold long.
  - If (current_price - predicted_price)/current_price > threshold_pct/100, go/hold short.
  - Otherwise, do nothing (stay in cash or keep the existing position).

Backtesting (run_backtest):
  - Uses the same trading logic, but based on provided parameters:
      • current_price: Current market price.
      • predicted_price: Predicted future price.
      • position_qty: Current position quantity (positive for long, negative for short, 0 for none).
  - Returns an action (as a string): BUY, SELL, SHORT, COVER, or NONE.
"""

import logging

def run_logic(current_price, predicted_price, ticker):
    """
    No-Trade Zone strategy for live trading:
      - threshold_pct determines how big the gap must be before opening or holding a position.
      - If the difference is within that threshold (in either direction), no new trades are executed.
    """
    # Import from the main trading module. These references allow us to execute trades.
    from forest import api, buy_shares, sell_shares, short_shares, close_short

    threshold_pct = 0.5  # 0.5% threshold
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

    # Calculate the price difference as a percentage.
    diff_pct = (predicted_price - current_price) / (current_price if current_price else 1e-6) * 100

    if diff_pct > threshold_pct:
        # We want to be (or remain) LONG.
        if current_position == "long":
            logging.info(f"[{ticker}] No-Trade-Zone: Already long, no action.")
        elif current_position == "short":
            logging.info(f"[{ticker}] No-Trade-Zone: Predicted up. Closing short, then going long.")
            close_short(ticker, abs(position_qty), current_price)
            # Re-check available cash to buy shares.
            account = api.get_account()
            cash = float(account.cash)
            shares_to_buy = int(cash // current_price)
            if shares_to_buy > 0:
                buy_shares(ticker, shares_to_buy, current_price, predicted_price)
        else:
            # No existing position: attempt to open a long position.
            account = api.get_account()
            cash = float(account.cash)
            shares_to_buy = int(cash // current_price)
            if shares_to_buy > 0:
                logging.info(f"[{ticker}] No-Trade-Zone: Opening new LONG.")
                buy_shares(ticker, shares_to_buy, current_price, predicted_price)
            else:
                logging.info(f"[{ticker}] No-Trade-Zone: Not enough cash to open long.")
    elif diff_pct < -threshold_pct:
        # We want to be (or remain) SHORT.
        if current_position == "short":
            logging.info(f"[{ticker}] No-Trade-Zone: Already short, no action.")
        elif current_position == "long":
            logging.info(f"[{ticker}] No-Trade-Zone: Predicted down. Selling long, then going short.")
            sell_shares(ticker, position_qty, current_price, predicted_price)
            account = api.get_account()
            cash = float(account.cash)
            shares_to_short = int(cash // current_price)
            if shares_to_short > 0:
                short_shares(ticker, shares_to_short, current_price, predicted_price)
            else:
                logging.info(f"[{ticker}] No-Trade-Zone: Insufficient funds to open short.")
        else:
            # No existing position: attempt to open a short position.
            account = api.get_account()
            cash = float(account.cash)
            shares_to_short = int(cash // current_price)
            if shares_to_short > 0:
                logging.info(f"[{ticker}] No-Trade-Zone: Opening new SHORT.")
                short_shares(ticker, shares_to_short, current_price, predicted_price)
            else:
                logging.info(f"[{ticker}] No-Trade-Zone: Insufficient funds to open short.")
    else:
        # Price difference is within the threshold—hold the current position.
        logging.info(f"[{ticker}] No-Trade-Zone: Price difference is within +/- {threshold_pct}%. Holding current position (no change).")

def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    """
    Backtest version of the No-Trade Zone strategy.

    Parameters:
      - current_price: Current market price of the stock.
      - predicted_price: Predicted future price.
      - position_qty: Current position quantity (positive for long, negative for short, 0 for none).

    Returns:
      - action: One of the following strings:
          • 'BUY'   - Open a new long position.
          • 'SELL'  - Close an existing long position (in preparation for a short position).
          • 'SHORT' - Open a new short position.
          • 'COVER' - Close an existing short position (in preparation for a long position).
          • 'NONE'  - No action is taken.
          
    Note:
      In scenarios where a position reversal is required (e.g., from short to long or vice versa),
      the function returns the primary action needed to exit the current position.
    """
    threshold_pct = 0.5  # 0.5% threshold

    # Determine the current position from the position_qty.
    if position_qty > 0:
        current_position = "long"
    elif position_qty < 0:
        current_position = "short"
    else:
        current_position = "none"

    # Calculate the percentage difference.
    diff_pct = (predicted_price - current_price) / (current_price if current_price else 1e-6) * 100

    if diff_pct > threshold_pct:
        if current_position == "long":
            # Already long—no trade action required.
            return "NONE"
        elif current_position == "short":
            # If in a short position and the price is predicted to rise,
            # the primary action is to COVER (exit the short) before going long.
            return "COVER"
        else:
            # With no position, the signal indicates opening a long position.
            return "BUY"
    elif diff_pct < -threshold_pct:
        if current_position == "short":
            # Already short—no trade action required.
            return "NONE"
        elif current_position == "long":
            # If in a long position and the price is predicted to fall,
            # the primary action is to SELL (exit the long) before going short.
            return "SELL"
        else:
            # With no position, the signal indicates opening a short position.
            return "SHORT"
    else:
        # The price difference is within the no-trade threshold—hold the position.
        return "NONE"
