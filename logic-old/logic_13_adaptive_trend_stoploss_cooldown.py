# logic_13_adaptive_trend_stoploss_cooldown.py
"""
logic_13_adaptive_trend_stoploss_cooldown.py

Combines:
 8) Adaptive Trend Filter (SuperTrend with K-Means ATR)
 3) Stop Loss / Trailing Stop
 7) Cooldown / Time-in-Trade Filter

Supports both:
 A) run_logic(...) => Incremental approach for live Alpaca trading
 B) run_backtest(...) => Offline backtesting on historical CSV data

Dependencies:
 - pandas, numpy, scikit-learn

Ensure your main script has the following functions (for live trading):
 - buy_shares(ticker, quantity, current_price, predicted_price)
 - sell_shares(ticker, quantity, current_price, predicted_price)
 - short_shares(ticker, quantity, current_price, predicted_price)
 - close_short(ticker, quantity, current_price)
"""

import logging
import math
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

#############################
# Module-level Containers
#############################
price_data = {}       # {ticker: DataFrame with columns ['Open','High','Low','Close','ATR','Assigned_ATR','SuperTrend','ST_Direction']}
position_state = {}   # {ticker: {'position': 'LONG'/'SHORT'/None, 'entry_price': float, 'best_price_for_stop': float, 'bars_in_trade': int, 'last_trade_bar': int}}
bar_counter = {}      # {ticker: int}

#############################
# 1) Helper Functions
#############################

def compute_atr(df, window=14):
    """
    Compute the Average True Range (ATR).
    """
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=1).mean()
    return atr

def rolling_kmeans_assign_centroid(atr_values, window_size=30, n_clusters=3):
    """
    Perform rolling K-Means clustering on ATR values and assign the nearest centroid.
    """
    assigned_centroids = np.full(len(atr_values), np.nan)

    for i in range(len(atr_values)):
        if pd.isna(atr_values.iloc[i]):
            continue
        if i < window_size:
            # Not enough data for K-Means; assign current ATR directly
            assigned_centroids[i] = atr_values.iloc[i]
            continue

        # Slice the window for K-Means
        window_data = atr_values.iloc[i - window_size:i].dropna().values.reshape(-1, 1)
        if len(window_data) < n_clusters:
            assigned_centroids[i] = atr_values.iloc[i]
            continue

        # Fit K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(window_data)
        current_atr = np.array([[atr_values.iloc[i]]])
        cluster_label = kmeans.predict(current_atr)[0]
        assigned_centroids[i] = kmeans.cluster_centers_[cluster_label][0]

    return assigned_centroids

def compute_adaptive_supertrend(df, atr_multiplier=3.0):
    """
    Compute the Adaptive SuperTrend indicator using Assigned_ATR.
    Adds 'SuperTrend' and 'ST_Direction' columns to df.
    """
    if 'Assigned_ATR' not in df.columns:
        df['Assigned_ATR'] = df['ATR']  # Fallback if Assigned_ATR not present

    hl2 = (df['High'] + df['Low']) / 2.0
    upper_band = hl2 + (atr_multiplier * df['Assigned_ATR'])
    lower_band = hl2 - (atr_multiplier * df['Assigned_ATR'])

    df['SuperTrend'] = np.nan
    df['ST_Direction'] = 0  # +1 for bullish, -1 for bearish

    for i in range(len(df)):
        if i == 0:
            df.at[i, 'SuperTrend'] = lower_band.iloc[i]
            df.at[i, 'ST_Direction'] = 1  # Assume bullish to start
            continue

        # Previous SuperTrend
        prev_supertrend = df.at[i - 1, 'SuperTrend']
        prev_direction = df.at[i - 1, 'ST_Direction']

        if prev_direction == 1:  # was bullish
            if df.at[i, 'Close'] > prev_supertrend:
                df.at[i, 'SuperTrend'] = min(upper_band.iloc[i], prev_supertrend)
                df.at[i, 'ST_Direction'] = 1
            else:
                df.at[i, 'SuperTrend'] = lower_band.iloc[i]
                df.at[i, 'ST_Direction'] = -1
        else:  # was bearish
            if df.at[i, 'Close'] < prev_supertrend:
                df.at[i, 'SuperTrend'] = max(lower_band.iloc[i], prev_supertrend)
                df.at[i, 'ST_Direction'] = -1
            else:
                df.at[i, 'SuperTrend'] = upper_band.iloc[i]
                df.at[i, 'ST_Direction'] = 1

    return df

def classify_direction(current_price, predicted_price, threshold_pct=0.0):
    """
    Classification based on predicted price movement.
    +1 for bullish, -1 for bearish, 0 for flat.
    """
    if current_price <= 0:
        return 0
    diff_pct = (predicted_price - current_price) / current_price * 100
    if diff_pct > threshold_pct:
        return 1
    elif diff_pct < -threshold_pct:
        return -1
    else:
        return 0

def risk_to_reward_ok(current_price, predicted_price, atr_value, rr_multiple=1.0):
    """
    Check if the absolute predicted move meets the risk-to-reward threshold.
    """
    if pd.isna(atr_value) or atr_value <= 0:
        return False
    expected_move = abs(predicted_price - current_price)
    return expected_move >= (rr_multiple * atr_value)

#############################
# 2) Live / Incremental run_logic(...)
#############################

def run_logic(current_price, predicted_price, ticker):
    """
    Incremental trading logic for live Alpaca trading.

    Steps:
      1. Append new bar data to price_data[ticker].
      2. Compute ATR and Assigned_ATR.
      3. Compute Adaptive SuperTrend.
      4. Classification and Risk-to-Reward filter.
      5. Manage positions: enter, exit, trailing stop.
      6. Enforce cooldown period.
    """
    from forest import api, buy_shares, sell_shares, short_shares, close_short

    # Strategy Parameters
    atr_period = 14
    kmeans_window = 30
    n_clusters = 3
    atr_multiplier = 3.0
    rr_multiple = 1.0
    stop_loss_pct = 3.0
    cooldown_bars = 3
    threshold_pct = 0.0  # see classification

    # Initialize data structures if not present
    if ticker not in price_data:
        columns = ['Open', 'High', 'Low', 'Close', 'ATR', 'Assigned_ATR', 'SuperTrend', 'ST_Direction']
        price_data[ticker] = pd.DataFrame(columns=columns)
        position_state[ticker] = {
            'position': None,
            'entry_price': 0.0,
            'best_price_for_stop': 0.0,
            'bars_in_trade': 0,
            'last_trade_bar': -cooldown_bars - 1  # so we can trade immediately
        }
        bar_counter[ticker] = -1

    # Increment bar counter
    bar_counter[ticker] += 1
    current_bar = bar_counter[ticker]

    # Fabricate new bar data (in real usage, replace with actual OHLC)
    new_bar_df = pd.DataFrame([{
        'Open': current_price,
        'High': current_price * 1.01,  # e.g. 1% higher
        'Low': current_price * 0.99,   # e.g. 1% lower
        'Close': current_price,
        'ATR': np.nan,
        'Assigned_ATR': np.nan,
        'SuperTrend': np.nan,
        'ST_Direction': 0
    }])
    df = price_data[ticker]
    df = pd.concat([df, new_bar_df], ignore_index=True)

    # Compute ATR and assigned ATR
    df['ATR'] = compute_atr(df, window=atr_period)
    df['Assigned_ATR'] = rolling_kmeans_assign_centroid(df['ATR'], window_size=kmeans_window, n_clusters=n_clusters)

    # Compute Adaptive SuperTrend
    df = compute_adaptive_supertrend(df, atr_multiplier=atr_multiplier)

    # Update the DataFrame
    price_data[ticker] = df

    # Retrieve the current SuperTrend direction
    current_dir = df['ST_Direction'].iloc[-1]  # +1 or -1
    classification = classify_direction(current_price, predicted_price, threshold_pct=threshold_pct)
    atr_value = df['ATR'].iloc[-1]
    rr_ok = risk_to_reward_ok(current_price, predicted_price, atr_value, rr_multiple=rr_multiple)

    # Retrieve position state
    pos_state = position_state[ticker]
    current_position = pos_state['position']
    entry_price = pos_state['entry_price']
    best_price_for_stop = pos_state['best_price_for_stop']
    bars_in_trade = pos_state['bars_in_trade']
    last_trade_bar = pos_state['last_trade_bar']

    # Increment bars_in_trade if in position
    if current_position in ("LONG", "SHORT"):
        bars_in_trade += 1

    # -- Trailing Stop Logic --
    if current_position == "LONG":
        # Update trailing stop
        if current_price > best_price_for_stop:
            best_price_for_stop = current_price
        stop_price = best_price_for_stop * (1 - stop_loss_pct / 100)
        if current_price < stop_price:
            logging.info(f"[{ticker}] Trailing Stop triggered for LONG at {current_price}")
            try:
                pos = api.get_position(ticker)
                qty = float(pos.qty)
                if qty > 0:
                    sell_shares(ticker, qty, current_price, predicted_price)
            except Exception as e:
                logging.error(f"[{ticker}] Error selling shares: {e}")
            # Reset
            pos_state['position'] = None
            pos_state['entry_price'] = 0.0
            pos_state['best_price_for_stop'] = 0.0
            pos_state['bars_in_trade'] = 0
            pos_state['last_trade_bar'] = current_bar

    elif current_position == "SHORT":
        # Update trailing stop
        if current_price < best_price_for_stop:
            best_price_for_stop = current_price
        stop_price = best_price_for_stop * (1 + stop_loss_pct / 100)
        if current_price > stop_price:
            logging.info(f"[{ticker}] Trailing Stop triggered for SHORT at {current_price}")
            try:
                pos = api.get_position(ticker)
                qty = float(pos.qty)
                if qty < 0:
                    close_short(ticker, abs(qty), current_price)
            except Exception as e:
                logging.error(f"[{ticker}] Error closing short: {e}")
            # Reset
            pos_state['position'] = None
            pos_state['entry_price'] = 0.0
            pos_state['best_price_for_stop'] = 0.0
            pos_state['bars_in_trade'] = 0
            pos_state['last_trade_bar'] = current_bar

    # -- Cooldown Logic --
    can_trade = (current_bar - pos_state['last_trade_bar']) >= cooldown_bars

    # -- Entry Logic --
    if pos_state['position'] is None and can_trade and rr_ok and classification != 0:
        if classification == 1 and current_dir == 1:
            # Enter LONG
            logging.info(f"[{ticker}] Entering LONG at {current_price}")
            try:
                account = api.get_account()
                cash = float(account.cash)
                shares_to_buy = int(cash // current_price)
                if shares_to_buy > 0:
                    buy_shares(ticker, shares_to_buy, current_price, predicted_price)
                    pos_state['position'] = "LONG"
                    pos_state['entry_price'] = current_price
                    pos_state['best_price_for_stop'] = current_price
                    pos_state['bars_in_trade'] = 0
                    pos_state['last_trade_bar'] = current_bar
                else:
                    logging.info(f"[{ticker}] Not enough cash to enter LONG.")
            except Exception as e:
                logging.error(f"[{ticker}] Error buying shares: {e}")

        elif classification == -1 and current_dir == -1:
            # Enter SHORT
            logging.info(f"[{ticker}] Entering SHORT at {current_price}")
            try:
                account = api.get_account()
                cash = float(account.cash)
                shares_to_short = int(cash // current_price)
                if shares_to_short > 0:
                    short_shares(ticker, shares_to_short, current_price, predicted_price)
                    pos_state['position'] = "SHORT"
                    pos_state['entry_price'] = current_price
                    pos_state['best_price_for_stop'] = current_price
                    pos_state['bars_in_trade'] = 0
                    pos_state['last_trade_bar'] = current_bar
                else:
                    logging.info(f"[{ticker}] Not enough funds to enter SHORT.")
            except Exception as e:
                logging.error(f"[{ticker}] Error shorting shares: {e}")

    # Update position state
    pos_state['bars_in_trade'] = bars_in_trade
    pos_state['best_price_for_stop'] = best_price_for_stop
    price_data[ticker] = df
    position_state[ticker] = pos_state


#############################
# 3) Offline / Backtest run_backtest(...)
#############################

backtest_price_data = pd.DataFrame(columns=['Open','High','Low','Close','ATR','Assigned_ATR','SuperTrend','ST_Direction'])
backtest_bar_counter = -1
backtest_position_state = {
    'position': None,             # "LONG", "SHORT", or None
    'entry_price': 0.0,
    'best_price_for_stop': 0.0,
    'bars_in_trade': 0,
    'last_trade_bar': -9999       # so we can trade immediately
}

def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    """
    Offline backtesting logic that parallels run_logic.
    Called externally with:
        current_price   : float
        predicted_price : float
        position_qty    : int (negative => short, positive => long, 0 => no position)

    Returns one of: "BUY", "SELL", "SHORT", "COVER", "NONE"

    This replicates the same trading decision flow as run_logic,
    but instead of placing real trades, it returns the decided action.
    """
    global backtest_price_data
    global backtest_bar_counter
    global backtest_position_state

    # Strategy Parameters (same as in run_logic)
    atr_period = 14
    kmeans_window = 30
    n_clusters = 3
    atr_multiplier = 3.0
    rr_multiple = 1.0
    stop_loss_pct = 3.0
    cooldown_bars = 3
    threshold_pct = 0.0

    action = "NONE"

    # Interpret position_qty for the backtest position
    if position_qty > 0:
        backtest_position_state['position'] = "LONG"
    elif position_qty < 0:
        backtest_position_state['position'] = "SHORT"
    else:
        backtest_position_state['position'] = None

    current_position = backtest_position_state['position']
    entry_price = backtest_position_state['entry_price']
    best_price_for_stop = backtest_position_state['best_price_for_stop']
    bars_in_trade = backtest_position_state['bars_in_trade']
    last_trade_bar = backtest_position_state['last_trade_bar']

    # Increment bar
    backtest_bar_counter += 1
    current_bar = backtest_bar_counter

    # Fabricate a new bar for the backtest DataFrame
    new_bar_df = pd.DataFrame([{
        'Open': current_price,
        'High': current_price * 1.01,
        'Low': current_price * 0.99,
        'Close': current_price,
        'ATR': np.nan,
        'Assigned_ATR': np.nan,
        'SuperTrend': np.nan,
        'ST_Direction': 0
    }])
    df = backtest_price_data
    df = pd.concat([df, new_bar_df], ignore_index=True)

    # Compute ATR & K-Means-based Assigned_ATR
    df['ATR'] = compute_atr(df, window=atr_period)
    df['Assigned_ATR'] = rolling_kmeans_assign_centroid(df['ATR'], window_size=kmeans_window, n_clusters=n_clusters)

    # Compute SuperTrend
    df = compute_adaptive_supertrend(df, atr_multiplier=atr_multiplier)

    # Update the backtest DataFrame reference
    backtest_price_data = df

    # Now do the same classification & decision logic
    current_dir = df['ST_Direction'].iloc[-1]  # +1 or -1
    classification = classify_direction(current_price, predicted_price, threshold_pct=threshold_pct)
    atr_value = df['ATR'].iloc[-1]
    rr_ok = risk_to_reward_ok(current_price, predicted_price, atr_value, rr_multiple=rr_multiple)

    # Track bars in trade if in a position
    if current_position in ("LONG", "SHORT"):
        bars_in_trade += 1

    # -- Trailing Stop Check --
    if current_position == "LONG":
        if current_price > best_price_for_stop:
            best_price_for_stop = current_price
        stop_price = best_price_for_stop * (1 - stop_loss_pct / 100)
        if current_price < stop_price:
            # Simulate SELL
            action = "SELL"
            # Reset
            backtest_position_state['position'] = None
            backtest_position_state['entry_price'] = 0.0
            backtest_position_state['best_price_for_stop'] = 0.0
            backtest_position_state['bars_in_trade'] = 0
            backtest_position_state['last_trade_bar'] = current_bar

    elif current_position == "SHORT":
        if current_price < best_price_for_stop:
            best_price_for_stop = current_price
        stop_price = best_price_for_stop * (1 + stop_loss_pct / 100)
        if current_price > stop_price:
            # Simulate COVER
            action = "COVER"
            # Reset
            backtest_position_state['position'] = None
            backtest_position_state['entry_price'] = 0.0
            backtest_position_state['best_price_for_stop'] = 0.0
            backtest_position_state['bars_in_trade'] = 0
            backtest_position_state['last_trade_bar'] = current_bar

    # Refresh position after potential trailing stop exit
    current_position = backtest_position_state['position']
    last_trade_bar = backtest_position_state['last_trade_bar']

    # -- Cooldown Check --
    can_trade = (current_bar - last_trade_bar) >= cooldown_bars

    # -- Entry Logic (only if flat) --
    if current_position is None and can_trade and rr_ok and classification != 0:
        if classification == 1 and current_dir == 1:
            # Simulate BUY
            action = "BUY"
            backtest_position_state['position'] = "LONG"
            backtest_position_state['entry_price'] = current_price
            backtest_position_state['best_price_for_stop'] = current_price
            backtest_position_state['bars_in_trade'] = 0
            backtest_position_state['last_trade_bar'] = current_bar
        elif classification == -1 and current_dir == -1:
            # Simulate SHORT
            action = "SHORT"
            backtest_position_state['position'] = "SHORT"
            backtest_position_state['entry_price'] = current_price
            backtest_position_state['best_price_for_stop'] = current_price
            backtest_position_state['bars_in_trade'] = 0
            backtest_position_state['last_trade_bar'] = current_bar

    # Update the backtest position state
    backtest_position_state['bars_in_trade'] = bars_in_trade
    backtest_position_state['best_price_for_stop'] = best_price_for_stop

    return action
