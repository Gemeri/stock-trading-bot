"""
logic_15_forecast_driven.py

This module implements a forecast-driven trading strategy:
  - If the predicted price is higher than the current price:
      * Close any open short positions.
      * If no long position is open, use all available cash to buy the maximum number of shares.
      * If a long position is already open, do nothing.
  - If the predicted price is lower than the current price:
      * Sell all long positions if any exist.
      * If no short position is open, short-sell as many shares as possible using available cash.
      * If a short position is already open, do nothing.
      
The run_logic function is called by the main trading script with the parameters:
    current_price, predicted_price, and ticker.
    
The run_backtest function is intended for backtesting the strategy.
It takes:
    current_price, predicted_price, and position_qty,
and returns one of the following actions:
    BUY, SELL, SHORT, COVER, NONE.
"""

import logging

def run_logic(current_price, predicted_price, ticker):
    """
    Executes the forecast-driven trading strategy.
    
    Parameters:
      current_price (float): The current market price.
      predicted_price (float): The forecasted price.
      ticker (str): The symbol of the asset.
      
    Trading Logic:
      - When predicted_price > current_price:
            * If holding a short position, close it.
            * If not already long (i.e. no long position open), buy maximum shares with available cash.
            * Otherwise, if already long, no new buy action is taken.
      - When predicted_price < current_price:
            * If holding a long position, sell all shares.
            * If not already short (i.e. no short position open), short-sell maximum shares based on available cash.
            * Otherwise, if already short, no new short action is taken.
    """
    # Import trading API functions from the forest module.
    from forest import api, buy_shares, sell_shares, short_shares, close_short

    logger = logging.getLogger(__name__)

    # Retrieve account details and available cash.
    try:
        account = api.get_account()
        cash = float(account.cash)
    except Exception as e:
        logger.error(f"[{ticker}] Error fetching account details: {e}")
        return

    # Retrieve current position for the ticker.
    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception:
        position_qty = 0.0  # Assume no position if none exists.

    logger.info(f"[{ticker}] Current Price: {current_price}, Predicted Price: {predicted_price}, "
                f"Position: {position_qty}, Cash: {cash}")

    # CASE 1: Predicted Price is Higher than the Current Price → Bullish signal
    if predicted_price > current_price:
        # If there is an open short position, close it.
        if position_qty < 0:
            qty_to_close = abs(position_qty)
            logger.info(f"[{ticker}] Predicted price is higher than current. Closing short position of {qty_to_close} shares.")
            close_short(ticker, qty_to_close, current_price, predicted_price)
            position_qty = 0  # Reset position after closing.
            # Refresh account details after closing the short position.
            try:
                account = api.get_account()
                cash = float(account.cash)
            except Exception as e:
                logger.error(f"[{ticker}] Error refreshing account details: {e}")
                return

        # If no long position exists, buy maximum shares possible with the available cash.
        if position_qty == 0:
            max_shares = int(cash // current_price)
            if max_shares > 0:
                logger.info(f"[{ticker}] Buying {max_shares} shares at {current_price}.")
                buy_shares(ticker, max_shares, current_price, predicted_price)
            else:
                logger.info(f"[{ticker}] Insufficient cash to purchase shares.")
        else:
            logger.info(f"[{ticker}] Already in a long position; no additional BUY action taken.")

    # CASE 2: Predicted Price is Lower than the Current Price → Bearish signal
    elif predicted_price < current_price:
        # If there is an open long position, sell all shares.
        if position_qty > 0:
            logger.info(f"[{ticker}] Predicted price is lower than current. Selling {position_qty} shares from long position.")
            sell_shares(ticker, position_qty, current_price, predicted_price)
            position_qty = 0
            # Refresh account details after selling the long position.
            try:
                account = api.get_account()
                cash = float(account.cash)
            except Exception as e:
                logger.error(f"[{ticker}] Error refreshing account details: {e}")
                return

        # If no short position exists, short-sell the maximum number of shares possible using available cash.
        if position_qty == 0:
            max_shares = int(cash // current_price)
            if max_shares > 0:
                logger.info(f"[{ticker}] Short-selling {max_shares} shares at {current_price}.")
                short_shares(ticker, max_shares, current_price, predicted_price)
            else:
                logger.info(f"[{ticker}] Insufficient cash to initiate a short sale.")
        else:
            logger.info(f"[{ticker}] Already in a short position; no additional SHORT action taken.")

    # CASE 3: Predicted Price is Equal to the Current Price → No action
    else:
        logger.info(f"[{ticker}] Predicted price equals current price; no trade action taken.")


def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    """
    Simulates the forecast-driven trading strategy for backtesting purposes.
    
    Parameters:
      current_price (float): The current market price.
      predicted_price (float): The forecasted price.
      position_qty (float): The current position quantity.
                             A negative value indicates a short position,
                             while a positive value indicates a long position.
    
    Returns:
      str: An action string indicating the trading decision:
           - "BUY": When predicted_price > current_price, no short position exists, and no long position is already held.
           - "COVER": When predicted_price > current_price but currently holding a short position.
           - "SELL": When predicted_price < current_price and currently holding a long position.
           - "SHORT": When predicted_price < current_price, no long position exists, and no short position is already held.
           - "NONE": When no trade action should be taken (e.g. predicted_price equals current_price, 
                     already in the desired position, or insufficient conditions to trade).
    
    Trading Logic:
      - When predicted_price > current_price:
            * If in a short position (position_qty < 0), the action is to COVER.
            * If not in any position (position_qty == 0), the action is to BUY.
            * If already in a long position (position_qty > 0), no new BUY is initiated (returns "NONE").
      - When predicted_price < current_price:
            * If in a long position (position_qty > 0), the action is to SELL.
            * If not in any position (position_qty == 0), the action is to SHORT.
            * If already in a short position (position_qty < 0), no new SHORT is initiated (returns "NONE").
      - When predicted_price equals current_price:
            * No action is taken.
    """
    # Bullish scenario
    if predicted_price > current_price:
        if position_qty < 0:
            return "COVER"
        elif position_qty == 0:
            return "BUY"
        else:  # position_qty > 0, already long
            return "NONE"
    
    # Bearish scenario
    elif predicted_price < current_price:
        if position_qty > 0:
            return "SELL"
        elif position_qty == 0:
            return "SHORT"
        else:  # position_qty < 0, already short
            return "NONE"
    
    # No change scenario
    else:
        return "NONE"
