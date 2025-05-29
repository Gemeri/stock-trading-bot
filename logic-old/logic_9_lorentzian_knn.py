"""
logic_9_lorentzian_knn.py

Implements a Lorentzian KNN–style classification, supporting:
 1) An incremental bar-by-bar approach for live scheduled Alpaca trading, called via run_logic(...).
 2) An offline backtest approach via run_backtest(...).

HOW IT WORKS:
- On each new bar (for the live approach):
   - The new bar’s data is appended to a module-level DataFrame (price_data[ticker]).
   - Basic features such as a simple RSI and a kernel-regression slope (KR_Slope) are computed.
   - The historical feature vectors are compared (via an approximate Lorentzian kNN) to yield a classification:
         +1 for bullish, -1 for bearish.
   - Depending on the current position state and the classification, a dynamic exit or an entry is triggered.
- For backtesting:
   - run_backtest(...) is externally called with current_price, predicted_price, and position_qty.
   - It uses the same feature computation and classification logic.
   - Instead of placing orders via an API, it returns one of the following actions:
         "BUY"   - open a long position (if no position exists)
         "SELL"  - exit a long position (or flip from long when bearish)
         "SHORT" - open a short position (if no position exists)
         "COVER" - exit a short position (or flip from short when bullish)
         "NONE"  - no action taken

DEPENDENCIES:
   pip install numpy pandas scikit-learn
"""

import logging
import math
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans  # optional, if you want to do fancier approaches
# from sklearn.ensemble import RandomForestClassifier # if you want an alternative approach

########################################
# Module-level containers for live run & backtest
########################################
price_data = {}     # { ticker: pd.DataFrame with columns [Open, High, Low, Close, RSI, KR_Slope, Label] }
position_state = {} # { ticker: dict with { 'position': 'LONG'/'SHORT'/None, 'bars_in_trade': int } }

########################################
# 1) HELPER FUNCTIONS (Feature calc, KNN)
########################################
def compute_rsi(series, window=14):
    """A simple RSI. For real usage, you might use a dedicated TA library."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)

    ema_up = up.ewm(com=window - 1, adjust=False).mean()
    ema_down = down.ewm(com=window - 1, adjust=False).mean()
    rs = ema_up / ema_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def lorentzian_distance(pointA, pointB):
    """
    Computes a Lorentzian distance between two feature vectors.
    Summation of log(1 + |a - b|) for each pair of components.
    """
    dist = 0.0
    for a, b in zip(pointA, pointB):
        dist += math.log(1 + abs(a - b))
    return dist

def approximate_knn(train_X, train_y, query_x, k=5, skip=2):
    """
    Approximates k-nearest-neighbor classification using Lorentzian distance.
      - train_X: list of feature vectors.
      - train_y: corresponding labels (e.g. +1 for bullish, -1 for bearish).
      - query_x: the feature vector for the current bar.
      - skip: sample every `skip` points to speed up the process.
    Returns:
      +1 if the sum of the k nearest labels is positive,
      -1 if the sum is negative,
       0 otherwise.
    """
    distances = []
    for i in range(0, len(train_X), skip):
        dist = lorentzian_distance(train_X[i], query_x)
        distances.append((dist, train_y[i]))
    distances.sort(key=lambda x: x[0])
    k_neighbors = distances[:k]
    label_sum = sum([n[1] for n in k_neighbors])
    if label_sum > 0:
        return 1
    elif label_sum < 0:
        return -1
    else:
        return 0

def rational_quadratic_kernel(x, xi, bandwidth=10.0):
    """
    Rational quadratic kernel:
       k(x, xi) = 1 / (1 + (|x - xi|^2) / (2 * alpha * bandwidth^2))
    where alpha=1.
    """
    alpha = 1.0
    dist_sq = (x - xi) ** 2
    return 1.0 / (1.0 + dist_sq / (2.0 * alpha * (bandwidth ** 2)))

def kernel_regression_slope(price_array, kernel_window=20, bandwidth=10):
    """
    A minimal Nadaraya–Watson kernel regression to smooth prices and compute a slope.
    Returns:
       - smoothed: the smoothed price array.
       - slope: the first difference of the smoothed prices.
    """
    n = len(price_array)
    smoothed = np.zeros(n)
    slope = np.zeros(n)
    for i in range(n):
        start = max(0, i - kernel_window + 1)
        x_vals = np.arange(start, i + 1)
        y_vals = price_array[start:i + 1]
        weights = []
        for x_val in x_vals:
            w = rational_quadratic_kernel(i, x_val, bandwidth)
            weights.append(w)
        weights = np.array(weights)
        s = np.sum(weights)
        if s == 0:
            smoothed[i] = price_array[i]
        else:
            smoothed[i] = np.sum(weights * y_vals) / s
        if i > 0:
            slope[i] = smoothed[i] - smoothed[i - 1]
        else:
            slope[i] = 0
    return smoothed, slope

###############################################
# 2) The "LIVE" RUN_LOGIC function for Alpaca trading
###############################################
def run_logic(current_price, predicted_price, ticker):
    """
    Called each new bar by your main trading script.
      - Appends the new bar's data (fabricated OHLC) to the ticker's DataFrame.
      - Computes RSI and kernel regression slope.
      - Uses an approximate Lorentzian kNN to classify the bar as bullish (+1) or bearish (-1).
      - Applies dynamic exit and/or fixed hold logic before entering a new position.
    Trades are executed via API functions (buy_shares, sell_shares, short_shares, close_short).
    """
    # Import trading functions and API from your main script
    from forest import api, buy_shares, sell_shares, short_shares, close_short

    # Basic configuration parameters
    neighbors_count = 5
    skip = 2
    hold_bars = 4
    use_dynamic_exit = True
    kernel_window = 20
    kernel_bandwidth = 10

    # Prepare the DataFrame and state for this ticker
    if ticker not in price_data:
        cols = ['Open', 'High', 'Low', 'Close', 'RSI', 'KR_Slope', 'Label']
        price_data[ticker] = pd.DataFrame([], columns=cols)
        position_state[ticker] = {
            'position': None,  # "LONG" or "SHORT" or None
            'bars_in_trade': 0
        }

    df = price_data[ticker]
    state = position_state[ticker]

    # For demonstration, fabricate OHLC values (in real usage, use real bar data)
    new_open = current_price
    new_high = current_price * 1.01
    new_low = current_price * 0.99
    new_close = current_price

    # Append a new row to the DataFrame (replacing df.append with pd.concat)
    new_row_df = pd.DataFrame([{
        'Open': new_open,
        'High': new_high,
        'Low': new_low,
        'Close': new_close,
        'RSI': np.nan,
        'KR_Slope': np.nan,
        'Label': 0
    }])
    df = pd.concat([df, new_row_df], ignore_index=True)

    # 1) Update RSI
    df['RSI'] = compute_rsi(df['Close'])

    # 2) Compute kernel regression slope
    prices = df['Close'].values
    _, slope_array = kernel_regression_slope(prices, kernel_window=kernel_window, bandwidth=kernel_bandwidth)
    df['KR_Slope'] = slope_array

    # 3) Define a label for each row: +1 if KR_Slope > 0, -1 otherwise.
    df['Label'] = df['KR_Slope'].apply(lambda x: 1 if x > 0 else -1)

    # 4) Build training arrays from historical data (using RSI as the sole feature here)
    X = [[rsi] for rsi in df['RSI'].values]
    Y = df['Label'].tolist()

    idx = len(df) - 1
    if idx < 1:
        logging.info(f"[{ticker}] logic_9_lorentzian_knn: Not enough bars yet.")
        price_data[ticker] = df
        return

    # Use all past bars (except the current one) for kNN
    query_x = X[idx]
    X_train = X[:idx]
    Y_train = Y[:idx]
    classification = approximate_knn(X_train, Y_train, query_x, k=neighbors_count, skip=skip)
    # classification: +1 = bullish, -1 = bearish

    # 5) Handle exit logic (if already in a position)
    if state['position'] == "LONG":
        state['bars_in_trade'] += 1
        if use_dynamic_exit:
            current_slope = df['KR_Slope'].iloc[-1]
            if current_slope < 0:
                logging.info(f"[{ticker}] L-KNN: Exiting LONG due to slope flip.")
                try:
                    pos = api.get_position(ticker)
                    qty = float(pos.qty)
                    if qty > 0:
                        sell_shares(ticker, qty, current_price, predicted_price)
                except:
                    pass
                state['position'] = None
                state['bars_in_trade'] = 0
        else:
            if state['bars_in_trade'] >= hold_bars:
                logging.info(f"[{ticker}] L-KNN: Exiting LONG due to hold period.")
                try:
                    pos = api.get_position(ticker)
                    qty = float(pos.qty)
                    if qty > 0:
                        sell_shares(ticker, qty, current_price, predicted_price)
                except:
                    pass
                state['position'] = None
                state['bars_in_trade'] = 0

    elif state['position'] == "SHORT":
        state['bars_in_trade'] += 1
        if use_dynamic_exit:
            current_slope = df['KR_Slope'].iloc[-1]
            if current_slope > 0:
                logging.info(f"[{ticker}] L-KNN: Covering SHORT due to slope flip.")
                try:
                    pos = api.get_position(ticker)
                    qty = float(pos.qty)
                    if qty < 0:
                        close_short(ticker, abs(qty), current_price)
                except:
                    pass
                state['position'] = None
                state['bars_in_trade'] = 0
        else:
            if state['bars_in_trade'] >= hold_bars:
                logging.info(f"[{ticker}] L-KNN: Exiting SHORT after hold period.")
                try:
                    pos = api.get_position(ticker)
                    qty = float(pos.qty)
                    if qty < 0:
                        close_short(ticker, abs(qty), current_price)
                except:
                    pass
                state['position'] = None
                state['bars_in_trade'] = 0

    # 6) Handle entry logic
    #    If classification > 0 => bullish; if classification < 0 => bearish.
    if classification > 0:
        if state['position'] == "LONG":
            logging.info(f"[{ticker}] L-KNN: Already LONG, no action required.")
        elif state['position'] == "SHORT":
            logging.info(f"[{ticker}] L-KNN: Bullish signal while SHORT => flipping position.")
            try:
                pos = api.get_position(ticker)
                qty = float(pos.qty)
                if qty < 0:
                    close_short(ticker, abs(qty), current_price)
            except:
                pass
            try:
                account = api.get_account()
                cash = float(account.cash)
                shares = int(cash // current_price)
                if shares > 0:
                    buy_shares(ticker, shares, current_price, predicted_price)
                    state['position'] = "LONG"
                    state['bars_in_trade'] = 0
            except:
                pass
        else:
            logging.info(f"[{ticker}] L-KNN: Bullish signal with no open position => entering LONG.")
            try:
                account = api.get_account()
                cash = float(account.cash)
                shares = int(cash // current_price)
                if shares > 0:
                    buy_shares(ticker, shares, current_price, predicted_price)
                    state['position'] = "LONG"
                    state['bars_in_trade'] = 0
            except:
                pass

    elif classification < 0:
        if state['position'] == "SHORT":
            logging.info(f"[{ticker}] L-KNN: Already SHORT, no action required.")
        elif state['position'] == "LONG":
            logging.info(f"[{ticker}] L-KNN: Bearish signal while LONG => flipping position.")
            try:
                pos = api.get_position(ticker)
                qty = float(pos.qty)
                if qty > 0:
                    sell_shares(ticker, qty, current_price, predicted_price)
            except:
                pass
            try:
                account = api.get_account()
                cash = float(account.cash)
                shares_to_short = int(cash // current_price)
                if shares_to_short > 0:
                    short_shares(ticker, shares_to_short, current_price, predicted_price)
                    state['position'] = "SHORT"
                    state['bars_in_trade'] = 0
            except:
                pass
        else:
            logging.info(f"[{ticker}] L-KNN: Bearish signal with no open position => entering SHORT.")
            try:
                account = api.get_account()
                cash = float(account.cash)
                shares_to_short = int(cash // current_price)
                if shares_to_short > 0:
                    short_shares(ticker, shares_to_short, current_price, predicted_price)
                    state['position'] = "SHORT"
                    state['bars_in_trade'] = 0
            except:
                pass
    else:
        logging.info(f"[{ticker}] L-KNN: Neutral signal => no new position action.")

    # Save the updated DataFrame
    price_data[ticker] = df

###############################################
# 3) The BACKTEST function (for offline evaluation)
###############################################
def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    """
    Backtest version of the logic function. Called externally with:
      - current_price: the current bar's price.
      - predicted_price: a predicted price (used for logging or evaluation).
      - position_qty: the current position quantity (positive for LONG, negative for SHORT,
                      0 for no open position).

    The function uses the same feature engineering and Lorentzian kNN classification as run_logic.
    It then applies the same exit and entry rules to decide on an action.
    Returns one of the following action strings:
        "BUY"   - Open a long position (if no position exists).
        "SELL"  - Exit a long position or flip from long to short.
        "SHORT" - Open a short position (if no position exists).
        "COVER" - Exit a short position or flip from short to long.
        "NONE"  - No action.
    """
    # Configuration parameters (should mirror run_logic)
    neighbors_count = 5
    skip = 2
    hold_bars = 4
    use_dynamic_exit = True
    kernel_window = 20
    kernel_bandwidth = 10

    # Use a fixed ticker name for backtest data storage.
    ticker = "BACKTEST"

    # Initialize data storage and state if not present.
    if ticker not in price_data:
        cols = ['Open', 'High', 'Low', 'Close', 'RSI', 'KR_Slope', 'Label']
        price_data[ticker] = pd.DataFrame([], columns=cols)
        # Initialize state based on position_qty:
        if position_qty > 0:
            init_position = "LONG"
        elif position_qty < 0:
            init_position = "SHORT"
        else:
            init_position = None
        position_state[ticker] = {
            'position': init_position,
            'bars_in_trade': 0
        }

    df = price_data[ticker]
    state = position_state[ticker]

    # Fabricate OHLC values for the current bar
    new_open = current_price
    new_high = current_price * 1.01
    new_low = current_price * 0.99
    new_close = current_price

    # Append new bar data (replacing df.append with pd.concat)
    new_row_df = pd.DataFrame([{
        'Open': new_open,
        'High': new_high,
        'Low': new_low,
        'Close': new_close,
        'RSI': np.nan,
        'KR_Slope': np.nan,
        'Label': 0
    }])
    df = pd.concat([df, new_row_df], ignore_index=True)

    # Update RSI and kernel regression slope
    df['RSI'] = compute_rsi(df['Close'])
    prices = df['Close'].values
    _, slope_array = kernel_regression_slope(prices, kernel_window=kernel_window, bandwidth=kernel_bandwidth)
    df['KR_Slope'] = slope_array

    # Define labels based on KR_Slope (+1 if positive, -1 if not)
    df['Label'] = df['KR_Slope'].apply(lambda x: 1 if x > 0 else -1)

    # Build training arrays (using RSI as the sole feature)
    X = [[rsi] for rsi in df['RSI'].values]
    Y = df['Label'].tolist()

    idx = len(df) - 1
    if idx < 1:
        price_data[ticker] = df
        return "NONE"

    # Classify the current bar using approximate kNN (without peeking)
    query_x = X[idx]
    X_train = X[:idx]
    Y_train = Y[:idx]
    classification = approximate_knn(X_train, Y_train, query_x, k=neighbors_count, skip=skip)

    action = "NONE"  # default action

    # First, if already in a position, update bars_in_trade and check for an exit signal.
    if state['position'] == "LONG":
        state['bars_in_trade'] += 1
        if use_dynamic_exit:
            current_slope = df['KR_Slope'].iloc[-1]
            if current_slope < 0:
                action = "SELL"  # exit LONG position due to slope flip
                state['position'] = None
                state['bars_in_trade'] = 0
                price_data[ticker] = df
                return action
        else:
            if state['bars_in_trade'] >= hold_bars:
                action = "SELL"
                state['position'] = None
                state['bars_in_trade'] = 0
                price_data[ticker] = df
                return action

    elif state['position'] == "SHORT":
        state['bars_in_trade'] += 1
        if use_dynamic_exit:
            current_slope = df['KR_Slope'].iloc[-1]
            if current_slope > 0:
                action = "COVER"  # exit SHORT position due to slope flip
                state['position'] = None
                state['bars_in_trade'] = 0
                price_data[ticker] = df
                return action
        else:
            if state['bars_in_trade'] >= hold_bars:
                action = "COVER"
                state['position'] = None
                state['bars_in_trade'] = 0
                price_data[ticker] = df
                return action

    # Now, if no exit has been triggered, process the entry logic.
    if classification > 0:
        if state['position'] == "LONG":
            action = "NONE"  # already long
        elif state['position'] == "SHORT":
            # If currently SHORT but signal is bullish, we need to cover the short.
            action = "COVER"
            state['position'] = None
            state['bars_in_trade'] = 0
        else:
            # No current position and bullish signal: open long.
            action = "BUY"
            state['position'] = "LONG"
            state['bars_in_trade'] = 0

    elif classification < 0:
        if state['position'] == "SHORT":
            action = "NONE"  # already short
        elif state['position'] == "LONG":
            # If currently LONG but signal is bearish, sell the long.
            action = "SELL"
            state['position'] = None
            state['bars_in_trade'] = 0
        else:
            # No current position and bearish signal: open short.
            action = "SHORT"
            state['position'] = "SHORT"
            state['bars_in_trade'] = 0
    else:
        action = "NONE"

    # Save updated data and state, then return the decided action.
    price_data[ticker] = df
    return action
