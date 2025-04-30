"""
logic_15_forecast_driven.py

This module implements a forecast-driven trading strategy:
  - If the predicted price is higher than the current price:
      * If no long position is open, use all available cash to buy the maximum number of shares.
      * If a long position is already open, do nothing.
  - If the predicted price is lower than the current price:
      * Sell all long positions if any exist.
      * Otherwise, do nothing.

The run_logic function is called by the main trading script with the parameters:
    current_price, predicted_price, and ticker.

The run_backtest function is intended for backtesting the strategy.
It takes:
    current_price, predicted_price, and position_qty,
and returns one of the following actions:
    BUY, SELL, NONE.
"""

import logging

def run_logic(current_price, predicted_price, ticker):
    """
    Executes the forecast-driven trading strategy without any shorting.
    
    Parameters:
      current_price (float): The current market price.
      predicted_price (float): The forecasted price.
      ticker (str): The symbol of the asset.
      
    Trading Logic:
      - When predicted_price > current_price:
            * If not already long (i.e. no long position open), buy maximum shares with available cash.
            * Otherwise, do nothing.
      - When predicted_price < current_price:
            * If holding a long position, sell all shares.
            * Otherwise, do nothing.
      - When predicted_price == current_price:
            * No action.
    """
    from forest import api, buy_shares, sell_shares

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

    # Bullish signal: Predicted > Current
    if predicted_price > current_price:
        if position_qty == 0:
            max_shares = int(cash // current_price)
            if max_shares > 0:
                logger.info(f"[{ticker}] Buying {max_shares} shares at {current_price}.")
                buy_shares(ticker, max_shares, current_price, predicted_price)
            else:
                logger.info(f"[{ticker}] Insufficient cash to purchase shares.")
        else:
            logger.info(f"[{ticker}] Already in a long position; no BUY action taken.")

    # Bearish signal: Predicted < Current
    elif predicted_price < current_price:
        if position_qty > 0:
            logger.info(f"[{ticker}] Selling {position_qty} shares at {current_price}.")
            sell_shares(ticker, position_qty, current_price, predicted_price)
        else:
            logger.info(f"[{ticker}] No long position to sell; no action taken.")

    # No change scenario
    else:
        logger.info(f"[{ticker}] Predicted price equals current price; no trade action taken.")


def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    """
    Simulates the forecast-driven trading strategy for backtesting (no shorting).
    
    Parameters:
      current_price (float): The current market price.
      predicted_price (float): The forecasted price.
      position_qty (float): The current position quantity (long > 0, flat = 0).
    
    Returns:
      str: An action string indicating the trading decision:
           - "BUY": When predicted_price > current_price and no long position exists.
           - "SELL": When predicted_price < current_price and a long position exists.
           - "NONE": Otherwise.
    """
    # Bullish scenario
    if predicted_price > current_price:
        if position_qty == 0:
            return "BUY"
        else:
            return "NONE"
    
    # Bearish scenario
    elif predicted_price < current_price:
        if position_qty > 0:
            return "SELL"
        else:
            return "NONE"
    
    # No change scenario
    else:
        return "NONE"
