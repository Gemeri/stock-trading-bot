import logging

def run_logic(current_price, predicted_price, ticker):
    from forest import api, buy_shares, sell_shares

    logger = logging.getLogger(__name__)

    # Retrieve account details and available cash
    try:
        account = api.get_account()
        cash = float(account.cash)
    except Exception as e:
        logger.error(f"[{ticker}] Error fetching account details: {e}")
        return

    # Retrieve current position for the ticker
    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception:
        position_qty = 0.0  # Assume no position if none exists

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


def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles, ticker):
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