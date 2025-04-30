"""
logic_6_risk_to_reward.py

Implements a "Risk-to-Reward" or "Confidence" filter strategy:
- We measure the expected move (predicted_close - current_close).
- We compare that to an ATR (or some other volatility measure).
- Only open (or hold) a position if the expected move is at least
  'required_multiple' times the volatility.

If the criterion is not met, we skip or close the position because the reward
isn't big enough to justify the risk.
"""
import os
import pandas as pd
import logging
from dotenv import load_dotenv

load_dotenv()

# Mapping from environment timeframe to CSV filename timeframe.
TIMEFRAME_MAP = {
    "4Hour": "H4",
    "2Hour": "H2",
    "1Hour": "H1",
    "30Min": "M30",
    "15Min": "M15"
}

def convert_timeframe(timeframe):
    """
    Convert the timeframe from the environment variable (e.g., '4Hour')
    to the CSV filename timeframe (e.g., 'H4').
    """
    return TIMEFRAME_MAP.get(timeframe, timeframe)

def run_logic(current_price, predicted_price, ticker):
    """
    The main logic function called by the router in 'main.py'.

    For demonstration:
    - We assume there's an ATR column in your CSV or you have another way of retrieving it.
    - We'll try to fetch the last known 'atr' from the data or from a placeholder value.
    """
    from forest import api, buy_shares, sell_shares, short_shares, close_short

    # Get the latest ATR value for the ticker (example function)
    atr_value = get_latest_atr_for(ticker)

    required_multiple = 1.0  # e.g., require expected move >= 1×ATR
    should_open = should_trade(predicted_price, current_price, atr_value, required_multiple)

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

    # Trading logic based on risk-to-reward
    if should_open:
        if predicted_price > current_price:
            # Favor a long position
            if current_position == "long":
                logging.info(f"[{ticker}] Risk-Reward: Already long, no change.")
            elif current_position == "short":
                logging.info(f"[{ticker}] Risk-Reward: Flipping from short to long.")
                close_short(ticker, abs(position_qty), current_price)
                account = api.get_account()
                cash = float(account.cash)
                shares_to_buy = int(cash // current_price)
                if shares_to_buy > 0:
                    buy_shares(ticker, shares_to_buy, current_price, predicted_price)
                else:
                    logging.info(f"[{ticker}] Risk-Reward: Not enough cash to flip to long.")
            else:
                logging.info(f"[{ticker}] Risk-Reward: Opening new long position.")
                account = api.get_account()
                cash = float(account.cash)
                shares_to_buy = int(cash // current_price)
                if shares_to_buy > 0:
                    buy_shares(ticker, shares_to_buy, current_price, predicted_price)
                else:
                    logging.info(f"[{ticker}] Risk-Reward: Not enough cash to go long.")

        else:
            # predicted_price < current_price => favor a short position
            if current_position == "short":
                logging.info(f"[{ticker}] Risk-Reward: Already short, no change.")
            elif current_position == "long":
                logging.info(f"[{ticker}] Risk-Reward: Flipping from long to short.")
                sell_shares(ticker, position_qty, current_price, predicted_price)
                account = api.get_account()
                cash = float(account.cash)
                shares_to_short = int(cash // current_price)
                if shares_to_short > 0:
                    short_shares(ticker, shares_to_short, current_price, predicted_price)
                else:
                    logging.info(f"[{ticker}] Risk-Reward: Not enough funds to flip to short.")
            else:
                logging.info(f"[{ticker}] Risk-Reward: Opening new short position.")
                account = api.get_account()
                cash = float(account.cash)
                shares_to_short = int(cash // current_price)
                if shares_to_short > 0:
                    short_shares(ticker, shares_to_short, current_price, predicted_price)
                else:
                    logging.info(f"[{ticker}] Risk-Reward: Not enough funds to go short.")

    else:
        # Not enough expected move => skip or close any open position
        logging.info(f"[{ticker}] Risk-Reward: Not enough expected move to justify risk. Doing nothing or closing position.")
        if current_position == "long":
            logging.info(f"[{ticker}] Risk-Reward: Closing long due to insufficient reward.")
            sell_shares(ticker, position_qty, current_price, predicted_price)
        elif current_position == "short":
            logging.info(f"[{ticker}] Risk-Reward: Covering short due to insufficient reward.")
            close_short(ticker, abs(position_qty), current_price)


def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    """
    Backtest function that applies the same risk-to-reward trading logic as run_logic.

    Parameters:
        current_price (float): The current price.
        predicted_price (float): The predicted price.
        position_qty (float): Current position quantity (positive for long, negative for short).

    Returns:
        str: One of the following actions: BUY, SELL, SHORT, COVER, NONE.
    """
    # For backtesting we use a constant (or historical) ATR value.
    atr_value = get_latest_atr()  # This returns a placeholder ATR value.
    required_multiple = 1.0
    should_open = should_trade(predicted_price, current_price, atr_value, required_multiple)

    # Determine current position from position_qty.
    if position_qty > 0:
        current_position = "long"
    elif position_qty < 0:
        current_position = "short"
    else:
        current_position = "none"

    # Initialize default action.
    action = "NONE"

    if should_open:
        if predicted_price > current_price:
            # Favor a long position.
            if current_position == "long":
                action = "NONE"
            elif current_position == "short":
                # In real trading, this would involve covering shorts then buying long.
                # For backtesting, we return BUY to indicate the desired new long position.
                action = "BUY"
            else:
                action = "BUY"
        else:
            # predicted_price < current_price => favor a short position.
            if current_position == "short":
                action = "NONE"
            elif current_position == "long":
                # In real trading, this would involve selling longs then shorting.
                # For backtesting, we return SHORT to indicate the desired new short position.
                action = "SHORT"
            else:
                action = "SHORT"
    else:
        # Expected move is insufficient. If in a position, exit it.
        if current_position == "long":
            action = "SELL"
        elif current_position == "short":
            action = "COVER"
        else:
            action = "NONE"

    return action


def should_trade(pred_close, curr_close, atr_val, required_multiple=1.0):
    """
    Return True if the expected move is at least 'required_multiple' times the ATR.

    Parameters:
        pred_close (float): The predicted closing price.
        curr_close (float): The current closing price.
        atr_val (float): The current ATR value.
        required_multiple (float): The required multiple of the ATR.

    Returns:
        bool: True if the expected move is sufficient; False otherwise.
    """
    expected_move = abs(pred_close - curr_close)
    return expected_move >= (required_multiple * atr_val)


def get_latest_atr_for(_ticker=None):
    """
    Retrieves the latest ATR value for the ticker and timeframe specified
    in the environment variables TICKERS and BAR_TIMEFRAME.
    
    The CSV file is expected to be named "<TICKERS>_<CSV_TIMEFRAME>.csv" and 
    to contain an "atr" column.

    Parameters:
        _ticker: Ignored. Included for compatibility with previous signatures.
    
    Returns:
        float: The latest ATR value.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        KeyError: If the CSV file does not contain an "atr" column.
    """
    ticker = os.getenv("TICKERS")
    timeframe = os.getenv("BAR_TIMEFRAME")
    csv_timeframe = convert_timeframe(timeframe)
    file_name = f"{ticker}_{csv_timeframe}.csv"

    if not os.path.isfile(file_name):
        raise FileNotFoundError(
            f"CSV file '{file_name}' not found for ticker '{ticker}' and timeframe '{timeframe}'."
        )

    df = pd.read_csv(file_name)
    
    if 'atr' not in df.columns:
        raise KeyError(f"CSV file '{file_name}' does not contain an 'atr' column.")
    
    # Assume that the CSV rows are time-ordered; the last row is the latest.
    atr_value = df['atr'].iloc[-1]
    return float(atr_value)


def get_latest_atr():
    """
    Retrieves the latest ATR value for backtesting using the ticker and timeframe
    specified in the environment variables TICKERS and BAR_TIMEFRAME.
    
    The CSV file is expected to be named "<TICKERS>_<CSV_TIMEFRAME>.csv" and 
    to contain an "atr" column.

    Returns:
        float: The latest ATR value.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        KeyError: If the CSV file does not contain an 'atr' column.
    """
    ticker = os.getenv("TICKERS")
    timeframe = os.getenv("BAR_TIMEFRAME")
    csv_timeframe = convert_timeframe(timeframe)
    file_name = f"{ticker}_{csv_timeframe}.csv"

    if not os.path.isfile(file_name):
        raise FileNotFoundError(
            f"CSV file '{file_name}' not found for ticker '{ticker}' and timeframe '{timeframe}'."
        )

    df = pd.read_csv(file_name)
    
    if 'atr' not in df.columns:
        raise KeyError(f"CSV file '{file_name}' does not contain an 'atr' column.")
    
    atr_value = df['atr'].iloc[-1]
    return float(atr_value)
