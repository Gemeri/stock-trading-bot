"""
logic_8_supertrend.py

Implements an Adaptive SuperTrend using K-means on ATR to determine a volatility cluster.

**Overview**:
- We keep an in-memory DataFrame (or dictionary) of recent bars for the ticker.
- On each run_logic(...) call (or run_backtest(...) for backtesting), we append the new bar's data,
  recalc ATR & rolling K-means assigned ATR, then compute the current SuperTrend direction.
- If the direction flips from bullish to bearish or vice versa, we flip the trade.

This file now contains two externally callable functions:
  - run_logic(...) for live trading (with a ticker identifier)
  - run_backtest(...) for backtesting (with a position_qty instead of a ticker)

The backtest function returns one of the following actions:
    BUY, SELL, SHORT, COVER, NONE
"""

import logging
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Global variables for live trading
price_data = {}   # {ticker: pd.DataFrame with columns ['Open','High','Low','Close','ATR','Assigned_ATR','Dir']}
last_dir = {}     # {ticker: integer (1 for bullish, -1 for bearish, 0 if unknown)}

# Global variables for backtesting (single instrument)
price_data_bt = pd.DataFrame([], columns=['Open','High','Low','Close','ATR','Assigned_ATR','Dir'])
last_dir_bt = 0


def run_logic(current_price, predicted_price, ticker):
    """
    Called from the main script on each new bar to decide trades.
    
    Steps:
      1) Generate a fake bar (for demonstration) and append it to price_data[ticker].
      2) Compute ATR on recent bars.
      3) Use a rolling K-means approach to assign an "Assigned_ATR".
      4) Compute the adaptive SuperTrend.
      5) If SuperTrend direction flips, execute a trade flip via flip_trade().
    
    This function uses live-trading order functions from the 'forest' module.
    """
    from forest import api, buy_shares, sell_shares, short_shares, close_short

    # Generate a fake bar for demonstration (replace with real bar data as needed)
    bar_high = current_price * 1.01  # example: 1% above current price
    bar_low  = current_price * 0.99  # example: 1% below current price
    bar_open = current_price
    bar_close = current_price  # here, current_price is used for the close as well

    # Ensure we have a DataFrame for this ticker
    if ticker not in price_data:
        columns = ['Open','High','Low','Close','ATR','Assigned_ATR','Dir']
        price_data[ticker] = pd.DataFrame([], columns=columns)
        last_dir[ticker] = 0

    df = price_data[ticker]

    # Append the new bar (with placeholders for ATR etc.) using pd.concat
    new_row = pd.DataFrame([{
        'Open': bar_open,
        'High': bar_high,
        'Low': bar_low,
        'Close': bar_close,
        'ATR': np.nan,
        'Assigned_ATR': np.nan,
        'Dir': 0
    }])
    df = pd.concat([df, new_row], ignore_index=True)

    # Recompute ATR over the DataFrame
    df = compute_atr(df, atr_len=10)

    # Compute rolling K-Means assigned ATR
    df['Assigned_ATR'] = rolling_kmeans_assign_centroid(df['ATR'], window_size=30, n_clusters=3)

    # Compute adaptive SuperTrend (using factor=3.0 as a typical multiplier)
    df = compute_supertrend(df, factor=3.0)

    # The new bar's direction is in df['Dir'].iloc[-1]
    current_dir = df['Dir'].iloc[-1]  # +1 (bullish) or -1 (bearish)
    prev_dir = last_dir[ticker]

    if prev_dir == 0:
        # first bar or no previous info: simply set the direction without trading
        last_dir[ticker] = current_dir
    else:
        # If there is a direction flip, execute the trading flip logic
        if prev_dir != current_dir:
            logging.info(f"[{ticker}] SuperTrend direction flip from {prev_dir} to {current_dir}")
            flip_trade(ticker, current_dir, current_price, predicted_price)
            last_dir[ticker] = current_dir
        else:
            # No flip: do nothing
            logging.info(f"[{ticker}] SuperTrend direction unchanged: {current_dir}")

    # Update the stored DataFrame for this ticker
    price_data[ticker] = df


def flip_trade(ticker, new_dir, current_price, predicted_price):
    """
    Executes trade flips based on the new SuperTrend direction.
    
    For new_dir = +1: the strategy wants to be long.
      - If already long, no action.
      - If short, first cover the short then buy.
      - If no position, open a new long position.
      
    For new_dir = -1: the strategy wants to be short.
      - If already short, no action.
      - If long, sell the long position then enter a short position.
      - If no position, open a new short position.
    """
    from forest import api, buy_shares, sell_shares, short_shares, close_short

    # Determine current position
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

    if new_dir == 1:
        # Want to be long
        if current_position == "long":
            logging.info(f"[{ticker}] Already long, no action.")
        elif current_position == "short":
            logging.info(f"[{ticker}] Flipping from short to long.")
            close_short(ticker, abs(position_qty), current_price)
            account = api.get_account()
            cash = float(account.cash)
            shares_to_buy = int(cash // current_price)
            if shares_to_buy > 0:
                buy_shares(ticker, shares_to_buy, current_price, predicted_price)
        else:
            # No position
            logging.info(f"[{ticker}] Opening new long position.")
            account = api.get_account()
            cash = float(account.cash)
            shares_to_buy = int(cash // current_price)
            if shares_to_buy > 0:
                buy_shares(ticker, shares_to_buy, current_price, predicted_price)
    else:
        # new_dir == -1 => Want to be short
        if current_position == "short":
            logging.info(f"[{ticker}] Already short, no action.")
        elif current_position == "long":
            logging.info(f"[{ticker}] Flipping from long to short.")
            sell_shares(ticker, position_qty, current_price, predicted_price)
            account = api.get_account()
            cash = float(account.cash)
            shares_to_short = int(cash // current_price)
            if shares_to_short > 0:
                short_shares(ticker, shares_to_short, current_price, predicted_price)
        else:
            # No position
            logging.info(f"[{ticker}] Opening new short position.")
            account = api.get_account()
            cash = float(account.cash)
            shares_to_short = int(cash // current_price)
            if shares_to_short > 0:
                short_shares(ticker, shares_to_short, current_price, predicted_price)


def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    """
    Backtest function that uses the same SuperTrend logic as run_logic but is meant for
    simulation/backtesting. It accepts:
      - current_price: the current market price.
      - predicted_price: the model’s predicted price.
      - position_qty: a number representing the current position (positive for long, negative for short).
    
    Returns one of the following actions as a string:
        BUY, SELL, SHORT, COVER, NONE

    The trading decision is based on whether the SuperTrend direction (computed from a rolling window)
    has flipped relative to the previous bar.
    """
    global price_data_bt, last_dir_bt

    # Generate a fake bar (for demonstration)
    bar_high = current_price * 1.01
    bar_low  = current_price * 0.99
    bar_open = current_price
    bar_close = current_price

    # Create the new row to be appended
    new_row = pd.DataFrame([{
        'Open': bar_open,
        'High': bar_high,
        'Low': bar_low,
        'Close': bar_close,
        'ATR': np.nan,
        'Assigned_ATR': np.nan,
        'Dir': 0
    }])
    # Use pd.concat instead of .append
    price_data_bt = pd.concat([price_data_bt, new_row], ignore_index=True)

    # Recompute ATR
    price_data_bt = compute_atr(price_data_bt, atr_len=10)

    # Rolling K-Means for Assigned_ATR
    price_data_bt['Assigned_ATR'] = rolling_kmeans_assign_centroid(
        price_data_bt['ATR'], window_size=30, n_clusters=3
    )

    # Compute adaptive SuperTrend (using factor=3.0)
    price_data_bt = compute_supertrend(price_data_bt, factor=3.0)

    # Get the SuperTrend direction for the new bar
    current_dir = price_data_bt['Dir'].iloc[-1]  # +1 (bullish) or -1 (bearish)
    action = "NONE"

    if last_dir_bt == 0:
        # On the first bar, record the direction and take no action.
        last_dir_bt = current_dir
        action = "NONE"
    else:
        if last_dir_bt != current_dir:
            # A flip in SuperTrend direction has occurred: decide on the appropriate action.
            action = flip_trade_backtest(current_dir, position_qty)
            last_dir_bt = current_dir
        else:
            action = "NONE"

    return action


def flip_trade_backtest(new_dir, position_qty):
    """
    Backtest helper function that determines the action to take based on the desired new direction
    and the current position (position_qty). The mapping is as follows:

      For new_dir == 1 (bullish, i.e. want to be long):
        - If already long (position_qty > 0): return "NONE"
        - If short (position_qty < 0): return "COVER"
        - If no position (position_qty == 0): return "BUY"

      For new_dir == -1 (bearish, i.e. want to be short):
        - If already short (position_qty < 0): return "NONE"
        - If long (position_qty > 0): return "SELL"
        - If no position (position_qty == 0): return "SHORT"
    """
    if new_dir == 1:
        if position_qty > 0:
            return "NONE"
        elif position_qty < 0:
            return "COVER"
        else:
            return "BUY"
    elif new_dir == -1:
        if position_qty < 0:
            return "NONE"
        elif position_qty > 0:
            return "SELL"
        else:
            return "SHORT"
    return "NONE"


# ------------------------------------------------------------------------
# Helper functions for ATR, rolling K-Means, and computing SuperTrend
# ------------------------------------------------------------------------

def compute_atr(df, atr_len=10):
    """
    Compute the Average True Range (ATR) in a minimal form for demonstration.
    The ATR is stored in df['ATR'].
    """
    # If less than 2 rows, ATR cannot be computed.
    if len(df) < 2:
        df['ATR'] = np.nan
        return df

    # True Range (TR) is the maximum of:
    #   (High - Low),
    #   abs(High - previous Close),
    #   abs(Low - previous Close)
    highs = df['High']
    lows = df['Low']
    closes = df['Close'].shift(1)

    df['H-L'] = highs - lows
    df['H-PC'] = (highs - closes).abs()
    df['L-PC'] = (lows - closes).abs()

    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(atr_len).mean()

    # Cleanup temporary columns
    df.drop(['H-L', 'H-PC', 'L-PC', 'TR'], axis=1, inplace=True, errors='ignore')
    return df


def rolling_kmeans_assign_centroid(atr_values, window_size=30, n_clusters=3):
    """
    For demonstration, this function uses a rolling window to perform K-Means clustering
    on the ATR values. For each bar (after window_size bars), the current ATR is assigned
    the centroid of the cluster it belongs to.
    
    If there isn’t enough data (or clusters), the raw ATR value is used.
    Returns a numpy array with the same length as atr_values.
    """
    assigned_centroids = np.full_like(atr_values, np.nan, dtype=float)

    for i in range(len(atr_values)):
        if pd.isna(atr_values.iloc[i]):
            continue
        if i < window_size:
            # Not enough data for K-Means; use the ATR value as-is.
            assigned_centroids[i] = atr_values.iloc[i]
            continue

        # Use data from [i-window_size, i) for K-Means
        slice_data = atr_values.iloc[i-window_size : i].dropna().values.reshape(-1, 1)
        if len(slice_data) < n_clusters:
            # Not enough data for the specified number of clusters
            assigned_centroids[i] = atr_values.iloc[i]
            continue

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(slice_data)

        current_atr = atr_values.iloc[i]
        cluster_label = kmeans.predict(np.array([[current_atr]]))[0]
        assigned_centroids[i] = kmeans.cluster_centers_[cluster_label][0]

    return assigned_centroids


def compute_supertrend(df, factor=3.0):
    """
    Compute a minimal version of the SuperTrend indicator using df['Assigned_ATR'].
    
    The SuperTrend value is stored in df['ST'] and the direction in df['Dir']:
      +1 => bullish (long)
      -1 => bearish (short)
    
    This is a simplified version based on typical SuperTrend calculations.
    """
    if 'Assigned_ATR' not in df.columns:
        df['Dir'] = 0
        return df

    # Calculate the basic upper and lower bands
    hl2 = (df['High'] + df['Low']) / 2.0
    # Use .ffill() instead of fillna(method='ffill') to avoid future deprecation
    assigned_atr = df['Assigned_ATR'].ffill().fillna(0.0)
    df['basic_upperband'] = hl2 + factor * assigned_atr
    df['basic_lowerband'] = hl2 - factor * assigned_atr

    # Initialize arrays for SuperTrend and its direction
    st = np.zeros(len(df))
    direction = np.zeros(len(df), dtype=int)
    st_upper = np.zeros(len(df))
    st_lower = np.zeros(len(df))

    if len(df) == 0:
        return df

    # Initialize the first bar
    st[0] = hl2.iloc[0]
    direction[0] = 1  # Assume a bullish start
    st_upper[0] = df['basic_upperband'].iloc[0]
    st_lower[0] = df['basic_lowerband'].iloc[0]

    for i in range(1, len(df)):
        # Calculate the running upper band
        bu = df['basic_upperband'].iloc[i]
        prev_upper = st_upper[i-1]
        st_upper[i] = bu if bu < prev_upper else prev_upper

        # Calculate the running lower band
        bl = df['basic_lowerband'].iloc[i]
        prev_lower = st_lower[i-1]
        st_lower[i] = bl if bl > prev_lower else prev_lower

        prev_st = st[i-1]
        close_i = df['Close'].iloc[i]

        # Determine whether the previous SuperTrend was using the upper or lower band
        using_upper = (prev_st == st_upper[i-1])
        if using_upper:
            # Previously bearish: if the close remains below the upper band, stay bearish;
            # otherwise, flip to bullish.
            if close_i <= st_upper[i]:
                st[i] = st_upper[i]
                direction[i] = -1
            else:
                st[i] = st_lower[i]
                direction[i] = 1
        else:
            # Previously bullish: if the close remains above the lower band, stay bullish;
            # otherwise, flip to bearish.
            if close_i >= st_lower[i]:
                st[i] = st_lower[i]
                direction[i] = 1
            else:
                st[i] = st_upper[i]
                direction[i] = -1

    df['ST'] = st
    df['Dir'] = direction

    # Cleanup temporary columns
    df.drop(['basic_upperband', 'basic_lowerband'], axis=1, inplace=True, errors='ignore')
    return df
