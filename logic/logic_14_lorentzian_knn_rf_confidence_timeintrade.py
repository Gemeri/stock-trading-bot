#!/usr/bin/env python
# coding: utf-8

"""
logic_14_lorentzian_knn_rf_confidence_timeintrade.py

Combines:
 9) Lorentzian KNN or Random Forest Classification
 1 or 6) Confidence Check
 7) Time-in-Trade / Cooldown or 4) Multi-Bar Hold

Supports both:
 A) run_logic(...) => Incremental approach for live Alpaca trading
 B) run_backtest(...) => Offline backtesting on historical CSV data

Dependencies:
 - pandas, numpy, scikit-learn, ta (for technical indicators)

Ensure your main script has the following functions (for live trading):
 - buy_shares(ticker, quantity, current_price, predicted_price)
 - sell_shares(ticker, quantity, current_price, predicted_price)
 - short_shares(ticker, quantity, current_price, predicted_price)
 - close_short(ticker, quantity, current_price)

Below is the full script with an added run_backtest(...) function for offline usage,
returning one of the actions: BUY, SELL, SHORT, COVER, NONE.
"""

import logging
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator

# ------------------------------
# Module-level Containers
# ------------------------------
price_data = {}       # {ticker: DataFrame with features and indicators}
position_state = {}   # {ticker: {'position': 'LONG'/'SHORT'/None, 'entry_price': float, 'best_price_for_stop': float, 'bars_in_trade': int, 'last_trade_bar': int}}
model_store = {}      # {ticker: trained ML model (RandomForest or KNN)}
bar_counter = {}      # {ticker: int}

# For backtest tracking (if you want to preserve state between calls)
# This assumes a single backtest pass, single "symbol" scenario.
# If you wish to handle multiple symbols, expand this dictionary approach.
backtest_state = {
    'position': None,             # "LONG"/"SHORT"/None
    'entry_price': 0.0,
    'best_price_for_stop': 0.0,
    'bars_in_trade': 0,
    'last_trade_bar': -9999       # so we can allow trading immediately
}
backtest_bar_counter = 0

# ------------------------------
# 1) Helper Functions
# ------------------------------

def build_features(df):
    """
    Build technical indicators as features for the classifier.
    """
    # Example technical indicators
    rsi = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi.rsi()

    sma = SMAIndicator(close=df['Close'], window=20)
    df['SMA20'] = sma.sma_indicator()

    ema = EMAIndicator(close=df['Close'], window=20)
    df['EMA20'] = ema.ema_indicator()

    # Moving averages difference
    df['SMA_EMA_Diff'] = df['SMA20'] - df['EMA20']

    # Fill NaNs
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)

    # Select features
    feature_cols = ['Close', 'RSI', 'SMA20', 'EMA20', 'SMA_EMA_Diff']
    return df[feature_cols]

def build_labels(df, horizon=4):
    """
    Define labels: +1 if future_close > current_close, else -1
    """
    df['Future_Close'] = df['Close'].shift(-horizon)
    df['Label'] = np.where(df['Future_Close'] > df['Close'], 1, -1)
    df['Label'].fillna(0, inplace=True)  # Assign 0 to the last 'horizon' bars
    return df

def train_model(X_train, y_train, model_type='rf'):
    """
    Train a classifier (Random Forest or KNN) with scaling.
    Returns a Pipeline with scaler and classifier.
    """
    if model_type == 'rf':
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError("Unsupported model type. Choose 'rf' or 'knn'.")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])

    pipeline.fit(X_train, y_train)
    return pipeline

def predict_with_confidence(model, X):
    """
    Predict class and return confidence scores.
    Returns (prediction, confidence)
    """
    try:
        proba = model.predict_proba(X)
        prediction = model.predict(X)[0]
        confidence = max(proba[0])
        return prediction, confidence
    except NotFittedError:
        logging.error("Model is not fitted yet.")
        return 0, 0.0

def get_features_for_live(ticker, current_price):
    """
    Retrieve and compute features for the current bar in live trading.
    """
    if ticker not in price_data:
        # Initialize with minimal data; in real scenarios, maintain a rolling window
        columns = ['Close', 'RSI', 'SMA20', 'EMA20', 'SMA_EMA_Diff']
        price_data[ticker] = pd.DataFrame(columns=columns)

    df = price_data[ticker]

    # For live trading, you should append the new price and compute indicators
    # Here, we assume the necessary historical data is already present
    # and we're only updating with the current price

    new_row = pd.DataFrame([{'Close': current_price}])
    df = pd.concat([df, new_row], ignore_index=True)

    # Recompute indicators if we have enough data
    if len(df) >= 20:
        df = build_features(df)

    price_data[ticker] = df
    return df.iloc[-1].values.reshape(1, -1)  # Return as 2D array for prediction

# ------------------------------
# 2) run_logic(...) for Live Trading
# ------------------------------

def run_logic(current_price, predicted_price, ticker, model_type='rf'):
    """
    Incremental trading logic for live Alpaca trading.

    Steps:
      1. Append new bar data to price_data[ticker].
      2. Compute technical indicators.
      3. Use the trained model to classify direction with confidence.
      4. Apply confidence filter.
      5. Manage positions: enter, exit based on classification and confidence.
      6. Enforce cooldown period.
    """
    from forest import api, buy_shares, sell_shares, short_shares, close_short

    # Strategy Parameters
    confidence_threshold = 0.6
    hold_bars = 3
    forecast_horizon = 4  # Not directly used in live trading
    model_choice = model_type  # 'rf' or 'knn'

    # Initialize data structures
    if ticker not in model_store:
        # For live trading, models should be pre-trained and loaded
        # Here, we assume models are trained during backtesting and saved externally
        logging.info(f"[{ticker}] Model not found. Please ensure the model is trained and loaded.")
        return

    if ticker not in price_data:
        columns = ['Close', 'RSI', 'SMA20', 'EMA20', 'SMA_EMA_Diff']
        price_data[ticker] = pd.DataFrame(columns=columns)
        position_state[ticker] = {
            'position': None,
            'entry_price': 0.0,
            'best_price_for_stop': 0.0,
            'bars_in_trade': 0,
            'last_trade_bar': -hold_bars - 1  # Initialize to allow immediate trading
        }
        bar_counter[ticker] = -1

    # Increment bar counter
    bar_counter[ticker] += 1
    current_bar = bar_counter[ticker]

    # Update price data
    df = price_data[ticker]
    new_row = pd.DataFrame([{'Close': current_price}])
    df = pd.concat([df, new_row], ignore_index=True)

    # Compute technical indicators
    if len(df) >= 20:
        df = build_features(df)
    else:
        # Not enough data to compute indicators
        price_data[ticker] = df
        return

    price_data[ticker] = df

    # Prepare features for prediction
    features = df.iloc[-1][['Close', 'RSI', 'SMA20', 'EMA20', 'SMA_EMA_Diff']].values.reshape(1, -1)

    # Predict direction with confidence
    model = model_store[ticker]
    prediction, confidence = predict_with_confidence(model, features)

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

    # Manage Stop Loss / Trailing Stop
    if current_position == "LONG":
        # Update trailing stop
        if current_price > best_price_for_stop:
            best_price_for_stop = current_price
        stop_price = best_price_for_stop * (1 - 0.03)  # 3% trailing stop
        if current_price < stop_price:
            logging.info(f"[{ticker}] KNN/RF+Confidence+TimeInTrade: Trailing Stop triggered for LONG at {current_price}")
            try:
                pos = api.get_position(ticker)
                qty = float(pos.qty)
                if qty > 0:
                    sell_shares(ticker, qty, current_price, predicted_price)
            except Exception as e:
                logging.error(f"[{ticker}] Error selling shares: {e}")
            # Reset position state
            pos_state['position'] = None
            pos_state['entry_price'] = 0.0
            pos_state['best_price_for_stop'] = 0.0
            pos_state['bars_in_trade'] = 0
            pos_state['last_trade_bar'] = current_bar

    elif current_position == "SHORT":
        # Update trailing stop
        if current_price < best_price_for_stop:
            best_price_for_stop = current_price
        stop_price = best_price_for_stop * (1 + 0.03)  # 3% trailing stop
        if current_price > stop_price:
            logging.info(f"[{ticker}] KNN/RF+Confidence+TimeInTrade: Trailing Stop triggered for SHORT at {current_price}")
            try:
                pos = api.get_position(ticker)
                qty = float(pos.qty)
                if qty < 0:
                    close_short(ticker, abs(qty), current_price)
            except Exception as e:
                logging.error(f"[{ticker}] Error closing short: {e}")
            # Reset position state
            pos_state['position'] = None
            pos_state['entry_price'] = 0.0
            pos_state['best_price_for_stop'] = 0.0
            pos_state['bars_in_trade'] = 0
            pos_state['last_trade_bar'] = current_bar

    # Enforce Cooldown
    can_trade = (current_bar - pos_state['last_trade_bar']) >= hold_bars

    # Entry Conditions
    if pos_state['position'] is None and can_trade and confidence >= confidence_threshold and prediction != 0:
        if prediction == 1:
            # Enter LONG
            logging.info(f"[{ticker}] KNN/RF+Confidence+TimeInTrade: Entering LONG at {current_price} with confidence {confidence:.2f}")
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
        elif prediction == -1:
            # Enter SHORT
            logging.info(f"[{ticker}] KNN/RF+Confidence+TimeInTrade: Entering SHORT at {current_price} with confidence {confidence:.2f}")
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
    position_state[ticker] = pos_state


# ------------------------------
# 3) run_backtest(...) for Offline Backtesting
# ------------------------------
def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    """
    Offline backtesting logic, mirroring run_logic's approach, but returning an action string.

    :param current_price: float - The current bar's price
    :param predicted_price: float - The model's predicted price (used here to determine direction)
    :param position_qty: float - Current position quantity; >0 = LONG, <0 = SHORT, 0 = No position

    Returns one of:
    "BUY",   # to open a new long position
    "SELL",  # to close an existing long position
    "SHORT", # to open a new short position
    "COVER", # to close an existing short position
    "NONE"   # no action
    """

    # We'll mimic the logic with:
    #   - A trailing stop approach
    #   - A hold_bars (cooldown) approach
    #   - A confidence threshold (we'll assume 1.0 here, since we only have predicted_price vs current_price)

    global backtest_state
    global backtest_bar_counter

    # Strategy Parameters similar to run_logic
    confidence_threshold = 0.6
    hold_bars = 3

    # We do not have an explicit model here, so let's infer direction from predicted_price vs current_price
    # If predicted_price > current_price => prediction = 1 (up)
    # If predicted_price < current_price => prediction = -1 (down)
    # If they're about the same, treat as 0
    if abs(predicted_price - current_price) < 1e-9:
        prediction = 0
    else:
        prediction = 1 if predicted_price > current_price else -1

    # Assume confidence = 1.0 for this offline scenario
    confidence = 1.0

    # Increment backtest bar counter
    backtest_bar_counter += 1
    current_bar = backtest_bar_counter

    # Interpret the position from position_qty
    # For the trailing-stop logic, we keep an internal record in backtest_state
    # that remembers 'entry_price', 'best_price_for_stop', 'bars_in_trade', etc.
    if position_qty > 0:
        current_position = "LONG"
    elif position_qty < 0:
        current_position = "SHORT"
    else:
        current_position = None

    # Sync our backtest_state's notion of what the position is
    # (If you prefer, you can rely solely on position_qty, but we'll store
    #  more info for trailing stop.)
    bt_pos = backtest_state['position']
    if bt_pos != current_position:
        # If an external script modifies position_qty outside this function,
        # re-initialize tracking accordingly.
        backtest_state['position'] = current_position
        backtest_state['entry_price'] = current_price
        backtest_state['best_price_for_stop'] = current_price
        backtest_state['bars_in_trade'] = 0
        # If we just entered a position externally, treat last_trade_bar as now
        if current_position is not None:
            backtest_state['last_trade_bar'] = current_bar - hold_bars

    # We can now rename local references for clarity
    entry_price = backtest_state['entry_price']
    best_price_for_stop = backtest_state['best_price_for_stop']
    bars_in_trade = backtest_state['bars_in_trade']
    last_trade_bar = backtest_state['last_trade_bar']

    # If currently in a position, increment bars_in_trade
    if current_position in ("LONG", "SHORT"):
        bars_in_trade += 1

    # Action defaults to NONE (no action taken)
    action = "NONE"

    # Manage trailing stops
    if current_position == "LONG":
        # Update trailing stop
        if current_price > best_price_for_stop:
            best_price_for_stop = current_price
        stop_price = best_price_for_stop * (1 - 0.03)  # 3% trailing stop
        if current_price < stop_price:
            logging.info(f"[Backtest] Trailing stop triggered for LONG at {current_price}")
            action = "SELL"
            # Once we SELL, reset internal state
            backtest_state['position'] = None
            backtest_state['entry_price'] = 0.0
            backtest_state['best_price_for_stop'] = 0.0
            backtest_state['bars_in_trade'] = 0
            backtest_state['last_trade_bar'] = current_bar

    elif current_position == "SHORT":
        # Update trailing stop
        if current_price < best_price_for_stop:
            best_price_for_stop = current_price
        stop_price = best_price_for_stop * (1 + 0.03)  # 3% trailing stop
        if current_price > stop_price:
            logging.info(f"[Backtest] Trailing stop triggered for SHORT at {current_price}")
            action = "COVER"
            # Once we COVER, reset internal state
            backtest_state['position'] = None
            backtest_state['entry_price'] = 0.0
            backtest_state['best_price_for_stop'] = 0.0
            backtest_state['bars_in_trade'] = 0
            backtest_state['last_trade_bar'] = current_bar

    # Enforce cooldown
    can_trade = (current_bar - backtest_state['last_trade_bar']) >= hold_bars

    # If we currently have no position and can trade, check if we should open one
    if backtest_state['position'] is None and can_trade and confidence >= confidence_threshold and prediction != 0:
        if prediction == 1:
            # We enter LONG
            logging.info(f"[Backtest] Entering LONG at {current_price} (prediction up)")
            action = "BUY"
            # Update internal state
            backtest_state['position'] = "LONG"
            backtest_state['entry_price'] = current_price
            backtest_state['best_price_for_stop'] = current_price
            backtest_state['bars_in_trade'] = 0
            backtest_state['last_trade_bar'] = current_bar
        elif prediction == -1:
            # We enter SHORT
            logging.info(f"[Backtest] Entering SHORT at {current_price} (prediction down)")
            action = "SHORT"
            # Update internal state
            backtest_state['position'] = "SHORT"
            backtest_state['entry_price'] = current_price
            backtest_state['best_price_for_stop'] = current_price
            backtest_state['bars_in_trade'] = 0
            backtest_state['last_trade_bar'] = current_bar

    # Update local references back to global state
    backtest_state['bars_in_trade'] = bars_in_trade
    backtest_state['best_price_for_stop'] = best_price_for_stop

    return action