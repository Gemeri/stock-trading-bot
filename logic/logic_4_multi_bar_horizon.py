"""
logic_4_multi_bar_horizon.py

Implements a "Multi-Bar Horizon" strategy, which reduces constant flipping
by only placing trades once every N bars. The model might predict each bar's
closing price, but we only act on that prediction periodically (e.g. once a day
or once every certain number of bars).

Note: If the script is reloaded or the process restarts, the bar counter resets.
A more robust approach might store bar counts in a database or file.
"""

import logging

# Module-level counter to simulate "bar index"
bar_counter = 0

def run_logic(current_price, predicted_price, ticker):
    """
    Live-trading version of the Multi-Bar Horizon strategy.

    Trades are only placed every N bars (here, every 6 bars). On a trade bar,
    the strategy compares the predicted price against the current price using a threshold.
      - A bullish signal (predicted price > current price by > 0.5%) means:
            • If already long: do nothing.
            • If short: cover the short and then go long.
            • If flat: enter a long position.
      - A bearish signal (predicted price < current price by > 0.5%) means:
            • If already short: do nothing.
            • If long: sell to exit the long and then go short.
            • If flat: enter a short position.
    """
    from forest import api, buy_shares, sell_shares, short_shares, close_short

    # Trade only every N bars.
    bars_between_trades = 6
    global bar_counter

    # Increment the bar counter.
    bar_counter += 1

    # If this is not a trade bar, skip trading.
    if bar_counter % bars_between_trades != 0:
        logging.info(f"[{ticker}] Multi-Bar: bar_counter={bar_counter}, not a trade bar. Skipping trades.")
        return

    logging.info(f"[{ticker}] Multi-Bar: It's a trade bar (bar_counter={bar_counter}). Checking position...")

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

    # Define threshold for a strong directional signal.
    threshold_pct = 0.5  # 0.5% difference
    diff_pct = (predicted_price - current_price) / (current_price if current_price else 1e-6) * 100

    if diff_pct > threshold_pct:
        # Bullish signal: want to be or remain long.
        if current_position == "long":
            logging.info(f"[{ticker}] Multi-Bar: Already long, no action.")
        elif current_position == "short":
            logging.info(f"[{ticker}] Multi-Bar: Predicted up, but currently short. Cover and go long.")
            close_short(ticker, abs(position_qty), current_price)
            account = api.get_account()
            cash = float(account.cash)
            shares_to_buy = int(cash // current_price)
            if shares_to_buy > 0:
                buy_shares(ticker, shares_to_buy, current_price, predicted_price)
        else:  # "none"
            logging.info(f"[{ticker}] Multi-Bar: No position, predicted up => go long.")
            account = api.get_account()
            cash = float(account.cash)
            shares_to_buy = int(cash // current_price)
            if shares_to_buy > 0:
                buy_shares(ticker, shares_to_buy, current_price, predicted_price)

    elif diff_pct < -threshold_pct:
        # Bearish signal: want to be or remain short.
        if current_position == "short":
            logging.info(f"[{ticker}] Multi-Bar: Already short, no action.")
        elif current_position == "long":
            logging.info(f"[{ticker}] Multi-Bar: Predicted down, but currently long. Sell and go short.")
            sell_shares(ticker, position_qty, current_price, predicted_price)
            account = api.get_account()
            cash = float(account.cash)
            shares_to_short = int(cash // current_price)
            if shares_to_short > 0:
                short_shares(ticker, shares_to_short, current_price, predicted_price)
        else:  # "none"
            logging.info(f"[{ticker}] Multi-Bar: No position, predicted down => go short.")
            account = api.get_account()
            cash = float(account.cash)
            shares_to_short = int(cash // current_price)
            if shares_to_short > 0:
                short_shares(ticker, shares_to_short, current_price, predicted_price)

    else:
        # If the predicted move is not strong enough, hold the current position.
        logging.info(f"[{ticker}] Multi-Bar: Predicted move not strong enough. Holding current or staying flat.")

def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    """
    Backtest version of the Multi-Bar Horizon strategy.

    Parameters:
        current_price (float): The current market price.
        predicted_price (float): The predicted future price.
        position_qty (float): The current position quantity (positive for long, negative for short, zero for none).

    Returns:
        action (str): One of the following:
            - 'BUY'   : Open a new long position.
            - 'SELL'  : Exit an existing long position.
            - 'SHORT' : Open a new short position.
            - 'COVER' : Exit an existing short position.
            - 'NONE'  : Hold the current position / take no action.

    Trading logic:
        - The strategy acts only on designated trade bars, defined by trading every N bars.
          If the current bar is not a trade bar, the function returns "NONE".
        - On a trade bar, the function computes the percentage difference between the predicted and current prices.
          A bullish signal is detected if the predicted price is more than 0.5% above the current price,
          and a bearish signal if it is more than 0.5% below.
            • Bullish:
                  - If already long: return "NONE".
                  - If short: return "COVER" (to exit the short, thereby preparing to go long).
                  - If flat: return "BUY".
            • Bearish:
                  - If already short: return "NONE".
                  - If long: return "SELL" (to exit the long, thereby preparing to go short).
                  - If flat: return "SHORT".
            • Otherwise: return "NONE".
    """
    global bar_counter
    bars_between_trades = 6  # same as in live logic

    # Increment the bar counter for backtesting.
    bar_counter += 1

    # If this is not a trade bar, do not trade.
    if bar_counter % bars_between_trades != 0:
        return "NONE"

    # Determine current position from the position_qty.
    if position_qty > 0:
        current_position = "long"
    elif position_qty < 0:
        current_position = "short"
    else:
        current_position = "none"

    # Define the threshold for a strong directional signal.
    threshold_pct = 0.5
    diff_pct = (predicted_price - current_price) / (current_price if current_price else 1e-6) * 100

    if diff_pct > threshold_pct:
        # Bullish signal.
        if current_position == "long":
            return "NONE"
        elif current_position == "short":
            return "COVER"
        else:
            return "BUY"
    elif diff_pct < -threshold_pct:
        # Bearish signal.
        if current_position == "short":
            return "NONE"
        elif current_position == "long":
            return "SELL"
        else:
            return "SHORT"
    else:
        return "NONE"
