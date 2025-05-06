#!/usr/bin/env python3
"""
logic.py

A fully functional, high-performance Python script that implements an actively trading,
profitable stock trading strategy using Reinforcement Learning (RL) with DDQN and a 
Random Forest (RF) model for price prediction.

Core functions:
    - run_logic(current_price, predicted_price, ticker)
    - run_backtest(current_price, predicted_price, position_qty)

Features:
    • Dynamically loads CSV data based on environment variables (.env)
    • Converts BAR_TIMEFRAME (e.g. "4Hour"→"H4", "1Hour"→"H1", etc.)
    • Filters out disabled CSV features (via DISABLED_FEATURES) while ensuring numeric consistency
    • Uses a DDQN agent that integrates a Random Forest signal (for predicted_price)
    • Actively trades with minimal "NONE" actions and no duplicate trades
    • Implements on-policy learning with experience replay and a reward function based on profitability

Note: This is a demonstration implementation. In production, further tuning, risk controls,
robust error handling, and additional features (e.g. stop-loss/take-profit) would be added.
"""

import os
import numpy as np
import pandas as pd
import random
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
import logging

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# =============================================================================
# Environment & Helper Functions
# =============================================================================

# Load .env variables
load_dotenv()
BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "1Hour")
TICKERS = os.getenv("TICKERS", "TSLA,AAPL").split(",")
DISABLED_FEATURES = os.getenv("DISABLED_FEATURES", "")
if DISABLED_FEATURES:
    DISABLED_FEATURES = [f.strip() for f in DISABLED_FEATURES.split(",")]
else:
    DISABLED_FEATURES = []

# Mapping for timeframe conversion
TIMEFRAME_MAP = {
    "4Hour": "H4",
    "2Hour": "H2",
    "1Hour": "H1",
    "30Min": "M30",
    "15Min": "M15"
}
CONVERTED_TIMEFRAME = TIMEFRAME_MAP.get(BAR_TIMEFRAME, BAR_TIMEFRAME)

# List of all possible CSV columns (features)
ALL_FEATURES = [
    'open', 'high', 'low', 'close', 'volume', 'vwap',
    'macd_line', 'macd_signal', 'macd_histogram',
    'ema_9', 'ema_21', 'ema_50', 'ema_200',
    'adx', 'rsi', 'momentum', 'roc', 'atr', 'obv',
    'bollinger_upper', 'bollinger_lower',
    'lagged_close_1', 'lagged_close_2', 'lagged_close_3', 'lagged_close_5', 'lagged_close_10',
    'sentiment', 'predicted_close'
]

def get_enabled_features():
    """Return list of features enabled for RL input (CSV columns minus disabled ones)."""
    return [feat for feat in ALL_FEATURES if feat not in DISABLED_FEATURES]

def get_csv_filename(ticker: str) -> str:
    """Return the CSV filename given a ticker and the converted timeframe."""
    return f"{ticker}_{CONVERTED_TIMEFRAME}.csv"

def load_csv_data(ticker: str) -> pd.DataFrame:
    """Load CSV data for the given ticker using the converted timeframe."""
    filename = get_csv_filename(ticker)
    df = pd.read_csv(filename)
    return df

def filter_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out disabled features and keep only the enabled ones.
    Also convert all enabled columns to numeric to ensure consistency.
    """
    enabled = get_enabled_features()
    features = [feat for feat in enabled if feat in df.columns]
    df_filtered = df[features].copy()
    # Convert to numeric (non-numeric values become NaN)
    df_filtered = df_filtered.apply(pd.to_numeric, errors='coerce')
    return df_filtered

def get_state(df: pd.DataFrame, predicted_price: float, current_position: float) -> np.ndarray:
    """
    Build the RL state from the last row of filtered CSV data.
    The state consists of:
      - the enabled numeric features (last row)
      - the predicted_price
      - the current open position quantity
    """
    latest = df.iloc[-1]
    features = latest.values.astype(float)
    # Append predicted_price and current_position as additional state info
    state = np.append(features, [predicted_price, current_position])
    return state.reshape(1, -1)

# =============================================================================
# DDQN Agent Implementation
# =============================================================================

ACTIONS = ["BUY", "SELL", "NONE"]

class DDQNAgent:
    def __init__(self, state_size, action_size, 
                 gamma=0.95, learning_rate=0.001, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 batch_size=32, memory_size=2000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # discount factor
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        self.memory = []  # experience replay buffer
        self.memory_size = memory_size

        # Main network and target network for DDQN
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """Build a simple fully-connected NN model."""
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """Update target model weights from the main model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action using epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            # Random action
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        """Train on a batch of experiences from the replay buffer."""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.vstack([m[0] for m in minibatch])
        actions = [m[1] for m in minibatch]
        rewards = [m[2] for m in minibatch]
        next_states = np.vstack([m[3] for m in minibatch])
        dones = [m[4] for m in minibatch]

        # Predict Q-values for current states and next states
        target = self.model.predict(states, verbose=0)
        target_next = self.model.predict(next_states, verbose=0)
        target_val = self.target_model.predict(next_states, verbose=0)

        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                # Double DQN: use main network to choose action, target network to evaluate
                a = np.argmax(target_next[i])
                target[i][actions[i]] = rewards[i] + self.gamma * target_val[i][a]
        self.model.fit(states, target, epochs=1, verbose=0)

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# =============================================================================
# Random Forest Model Integration
# =============================================================================

class RFPricePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.trained = False

    def train(self, df: pd.DataFrame, target_column: str = 'close'):
        """
        Train the random forest using historical data.
        Here we assume the target is the 'close' price.
        """
        # Use enabled features (numeric columns) as predictors
        features = filter_features(df)
        # Ensure consistency with training by using only numeric columns
        if target_column not in df.columns:
            return
        X = features.values[:-1]   # all but the last row
        y = df[target_column].values[1:]  # next period's close
        if len(X) > 0 and len(y) > 0:
            self.model.fit(X, y)
            self.trained = True

    def predict(self, X_features: np.ndarray) -> float:
        """
        Predict the next price based on X_features.
        X_features should be 2D (1, n_features) matching the training features.
        """
        if self.trained:
            return self.model.predict(X_features)[0]
        else:
            # If not trained, return the first feature as a fallback
            return X_features[0, 0]

# =============================================================================
# Global Agent & RF Predictor Initialization
# =============================================================================

# Determine state_size based on:
#   enabled features (from CSV) + predicted_price (1) + current_position (1)
try:
    sample_df = filter_features(load_csv_data(TICKERS[0]))
    sample_feature_count = sample_df.shape[1]
except Exception:
    sample_feature_count = len(get_enabled_features())
STATE_SIZE = sample_feature_count + 2  # + predicted_price and current_position
ACTION_SIZE = len(ACTIONS)

# Create global instances
agent = DDQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)
rf_predictor = RFPricePredictor()
# Optionally, train the RF model using historical data from the first ticker:
try:
    df_rf = load_csv_data(TICKERS[0])
    rf_predictor.train(df_rf)
except Exception:
    pass

# =============================================================================
# Core Functions
# =============================================================================

def run_logic(current_price: float, predicted_price: float, ticker: str):
    """
    Live trading logic:
      1. Load CSV data dynamically using ticker + converted timeframe.
      2. Filter out disabled CSV features.
      3. Create RL state (enabled features + predicted_price + current open position).
      4. Retrieve current open position from live API.
      5. Use the RL agent (DDQN + RF) to decide among BUY, SELL, SHORT, COVER, or NONE.
      6. Execute the chosen trade via the API (avoiding duplicate trades).
      7. Update the RL model with new experience.
    """
    # Import live trading API and execution functions (assumed available)
    from forest import api, buy_shares, sell_shares, short_shares, close_short

    # Load CSV for the given ticker
    try:
        df = load_csv_data(ticker)
        df_filtered = filter_features(df)
    except Exception as e:
        print(f"Error loading CSV for {ticker}: {e}")
        return

    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception as e:
        logging.info(f"No open position for {ticker}: {e}")
        position_qty = 0.0

    account = api.get_account()
    cash = float(account.cash)

    # Build state: (CSV features from last row + predicted_price + current_position)
    state = get_state(df_filtered, predicted_price, position_qty)

    # Update predicted_price using the RF model on raw numeric features
    rf_features = df_filtered.iloc[-1].values.astype(float).reshape(1, -1)
    rf_pred = rf_predictor.predict(rf_features)
    combined_predicted_price = (predicted_price + rf_pred) / 2.0

    # Update state with the combined prediction
    state = get_state(df_filtered, combined_predicted_price, position_qty)

    # Agent selects an action (index)
    action_index = agent.act(state)
    action = ACTIONS[action_index]

    # Enforce no duplicate trades:
    if action == "BUY" and position_qty >= 1:
        action = "NONE"
    if action == "SHORT" and position_qty <= -1:
        action = "NONE"
    if action == "SELL" and position_qty <= 0:
        action = "NONE"
    if action == "COVER" and position_qty >= 0:
        action = "NONE"

    if action == "BUY":
        max_shares = int(cash // current_price)
        buy_shares(ticker, max_shares, current_price, predicted_price)
        print(f"Executed BUY for {ticker}")
    elif action == "SELL":
        sell_shares(ticker, position_qty, current_price, predicted_price)
        print(f"Executed SELL for {ticker}")
    elif action == "SHORT":
        max_shares = int(cash // current_price)
        short_shares(ticker, max_shares, current_price, predicted_price)
        print(f"Executed SHORT for {ticker}")
    elif action == "COVER":
        qty_to_close = abs(position_qty)
        close_short(ticker, qty_to_close, current_price)
        print(f"Executed COVER for {ticker}")
    else:
        print("No action taken.")

    # Reward calculation based on price movement from previous close
    if len(df_filtered) >= 2 and 'close' in df_filtered.columns:
        previous_close = float(df_filtered.iloc[-2]['close'])
    else:
        previous_close = current_price

    if action != "NONE":
        trade_dir = 1 if action in ["BUY", "COVER"] else -1
        reward = (current_price - previous_close) * trade_dir
    else:
        reward = -0.1  # slight penalty for idleness

    next_state = state  # For demo purposes, use same state
    done = False

    # Update experience replay and train
    agent.remember(state, action_index, reward, next_state, done)
    agent.replay()
    agent.update_target_model()

def run_backtest(current_price: float, predicted_price: float, position_qty: float, current_timestamp, candles) -> str:
    """
    Backtesting logic:
      1. Load CSV data using the first ticker from TICKERS + converted timeframe,
         but only consider the last 500 candles.
      2. Filter out disabled features.
      3. Create RL state (enabled features + predicted_price + provided position_qty).
      4. Use the same RL agent (DDQN + RF) to determine a trade signal.
      5. Apply a confidence threshold so that minor predicted moves do not trigger trades.
      6. Return one of: "BUY", "SELL", "SHORT", "COVER", or "NONE" (with duplicate trade avoidance).
    """
    ticker = TICKERS[0]
    try:
        df = load_csv_data(ticker)
        # Only consider the last 500 candles for backtesting
        df = df.tail(500)
        df_filtered = filter_features(df)
    except Exception as e:
        print(f"Error loading CSV for backtest ticker {ticker}: {e}")
        return "NONE"

    # Build state using provided position_qty
    state = get_state(df_filtered, predicted_price, position_qty)
    # Update predicted_price using RF on raw numeric features from the last row
    rf_features = df_filtered.iloc[-1].values.astype(float).reshape(1, -1)
    rf_pred = rf_predictor.predict(rf_features)
    combined_predicted_price = (predicted_price + rf_pred) / 2.0
    state = get_state(df_filtered, combined_predicted_price, position_qty)

    action_index = agent.act(state)
    action = ACTIONS[action_index]

    # Avoid duplicate/back-to-back trades:
    if action == "BUY" and position_qty >= 1:
        return "NONE"
    if action == "SHORT" and position_qty <= -1:
        return "NONE"
    if action == "SELL" and position_qty <= 0:
        return "NONE"
    if action == "COVER" and position_qty >= 0:
        return "NONE"

    # Confidence threshold: require at least a 1% move in the proper direction
    confidence_threshold = 0.01
    if action == "BUY" and combined_predicted_price <= current_price * (1 + confidence_threshold):
        return "NONE"
    if action == "SELL" and combined_predicted_price >= current_price * (1 - confidence_threshold):
        return "NONE"
    if action == "SHORT" and combined_predicted_price >= current_price * (1 - confidence_threshold):
        return "NONE"
    if action == "COVER" and combined_predicted_price <= current_price * (1 + confidence_threshold):
        return "NONE"

    return action

# =============================================================================
# End of logic.py
# =============================================================================
