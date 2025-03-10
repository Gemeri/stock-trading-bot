#!/usr/bin/env python3
"""
logic.py

A fully functional, high-performance trading logic module using
Reinforcement Learning (with TensorFlow) and a Random Forest model
for price prediction. Implements live trading (run_logic) and backtesting
(run_backtest) using dynamically loaded CSV candle data and environment settings.
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor
from dotenv import load_dotenv

# ------------------------------------------------------------------------------
# Environment and Global Settings
# ------------------------------------------------------------------------------

# Load environment variables from .env
load_dotenv()

# Read environment variables
BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "1Hour")
TICKERS = os.getenv("TICKERS", "TSLA,AAPL")
DISABLED_FEATURES = os.getenv("DISABLED_FEATURES", "")

# Process disabled features list (comma separated)
disabled_features_list = [f.strip() for f in DISABLED_FEATURES.split(',')] if DISABLED_FEATURES else []

# Mapping from BAR_TIMEFRAME to CSV suffix
timeframe_mapping = {
    "4Hour": "H4",
    "2Hour": "H2",
    "1Hour": "H1",
    "30Min": "M30",
    "15Min": "M15"
}
converted_timeframe = timeframe_mapping.get(BAR_TIMEFRAME, "H1")

# Use the first ticker in TICKERS for backtesting purposes
first_ticker = TICKERS.split(",")[0].strip()

# Action indices: 0: BUY, 1: SELL, 2: SHORT, 3: COVER
ACTION_MAP = {
    0: "BUY",
    1: "SELL",
    2: "SHORT",
    3: "COVER"
}

# ------------------------------------------------------------------------------
# RL Agent Implementation using TensorFlow (Policy Gradient with Epsilon-Greedy)
# ------------------------------------------------------------------------------

class RLAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, epsilon=0.3):
        self.state_dim = state_dim
        self.action_dim = action_dim  # 4 actions: BUY, SELL, SHORT, COVER
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # probability for random exploration
        self.model = self.build_model()
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.memory = []  # Experience replay buffer

    def build_model(self):
        model = Sequential([
            InputLayer(input_shape=(self.state_dim,)),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_dim, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='categorical_crossentropy')
        return model

    def choose_action(self, state):
        # Epsilon-greedy: sometimes choose a random action
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_dim)
            return action, None
        policy = self.model.predict(np.array([state]), verbose=0)[0]
        action = np.random.choice(self.action_dim, p=policy)
        return action, policy

    def store_transition(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def update_model(self):
        # Perform a policy gradient update using stored transitions.
        if not self.memory:
            return
        states = np.array([trans[0] for trans in self.memory])
        actions = np.array([trans[1] for trans in self.memory])
        rewards = np.array([trans[2] for trans in self.memory])
        # Compute discounted rewards
        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        cumulative = 0.0
        for t in reversed(range(len(rewards))):
            cumulative = rewards[t] + self.gamma * cumulative
            discounted_rewards[t] = cumulative
        # Normalize rewards
        discounted_rewards -= np.mean(discounted_rewards)
        if np.std(discounted_rewards) > 0:
            discounted_rewards /= np.std(discounted_rewards)
        # One-hot encode actions
        actions_onehot = np.zeros((len(actions), self.action_dim))
        actions_onehot[np.arange(len(actions)), actions] = 1
        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(
                labels=actions_onehot, logits=tf.math.log(logits + 1e-10)
            )
            loss = tf.reduce_mean(neg_log_prob * discounted_rewards)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        # Clear memory after update
        self.memory = []

# Global RL agent instance (initialized when state_dim is known)
global_agent = None

# ------------------------------------------------------------------------------
# Random Forest Model for Price Prediction Integration
# ------------------------------------------------------------------------------

rf_model = RandomForestRegressor(n_estimators=10)
rf_trained = False

def update_random_forest(df, target_column='close'):
    """
    Retrain the Random Forest model using the CSV data.
    """
    global rf_model, rf_trained
    if df.empty or target_column not in df.columns:
        return
    # Use all columns (except target and timestamp) as features
    X = df.drop(columns=[target_column, 'timestamp'], errors='ignore').values
    y = df[target_column].values
    rf_model.fit(X, y)
    rf_trained = True

def get_rf_prediction(features):
    """
    Get a Random Forest prediction for the given features.
    """
    global rf_model, rf_trained
    if not rf_trained:
        return None
    prediction = rf_model.predict([features])
    return prediction[0]

# ------------------------------------------------------------------------------
# CSV Data Loading and Preprocessing
# ------------------------------------------------------------------------------

def load_csv_data(ticker, timeframe_suffix):
    """
    Load CSV data for a given ticker and timeframe suffix.
    Filename format: {ticker}_{timeframe_suffix}.csv
    """
    filename = f"{ticker}_{timeframe_suffix}.csv"
    try:
        df = pd.read_csv(filename)
        return df
    except Exception as e:
        print(f"Error loading CSV file {filename}: {e}")
        return pd.DataFrame()

def preprocess_data(df, disabled_features):
    """
    Remove disabled features from the DataFrame.
    """
    for feature in disabled_features:
        if feature in df.columns:
            df = df.drop(columns=[feature])
    return df

def extract_features(df, predicted_price):
    """
    Extract features from the latest CSV row (ignoring timestamp) and append the predicted_price.
    """
    features = df.drop(columns=['timestamp'], errors='ignore').iloc[-1].values.astype(float).tolist()
    features.append(float(predicted_price))
    return np.array(features)

def compute_state(df, predicted_price, current_position):
    """
    Compute the RL state vector from the latest CSV data row,
    the external predicted_price, and the current open position.
    """
    features = extract_features(df, predicted_price)
    # Append current position as part of the state
    state = np.append(features, current_position)
    return state

def compute_state_from_row(row, predicted_price, current_position):
    """
    Compute the RL state vector from a given row of CSV data,
    the predicted_price, and the current position.
    """
    features = row.drop(labels=['timestamp'], errors='ignore').values.astype(float).tolist()
    features.append(float(predicted_price))
    state = np.array(features + [current_position])
    return state

# ------------------------------------------------------------------------------
# Reward Function (with Harsher Punishment for Bad Trades)
# ------------------------------------------------------------------------------

def calculate_reward(action, current_price, predicted_price):
    """
    Calculate reward for a given action.
    A base reward is computed and if the trade is loss-making, a doubled penalty is applied.
    """
    if action == "BUY":
        base_reward = predicted_price - current_price
        reward = base_reward if base_reward >= 0 else base_reward * 2.0
    elif action == "SELL":
        base_reward = current_price - predicted_price
        reward = base_reward if base_reward >= 0 else base_reward * 2.0
    elif action == "SHORT":
        base_reward = current_price - predicted_price
        reward = base_reward if base_reward >= 0 else base_reward * 2.0
    elif action == "COVER":
        base_reward = predicted_price - current_price
        reward = base_reward if base_reward >= 0 else base_reward * 2.0
    else:  # "NONE"
        reward = -0.05  # A heavy penalty for doing nothing when a trade is expected
    return reward

# ------------------------------------------------------------------------------
# Online RL Update (applies to both live trading and backtesting)
# ------------------------------------------------------------------------------

def online_rl_update(state, action, reward, next_state):
    """
    Store the transition and update the RL model.
    """
    global global_agent
    if global_agent:
        global_agent.store_transition(state, action, reward, next_state)
        global_agent.update_model()

# ------------------------------------------------------------------------------
# Trading Logic Functions
# ------------------------------------------------------------------------------

def run_logic(current_price, predicted_price, ticker):
    """
    Live trading logic.
    
    Steps:
      1. Load CSV data dynamically using {ticker}_{converted_timeframe}.csv.
      2. Filter out disabled features.
      3. Update the Random Forest model on the CSV data.
      4. Retrieve current open position from the live API.
      5. Compute the RL state (features + predicted_price + current position).
      6. Use the RL agent to choose an action.
      7. Prevent duplicate or redundant trades.
      8. Execute the trade via external API functions.
      9. Compute reward and update the RL model online.
    """
    from forest import api, buy_shares, sell_shares, short_shares, close_short

    # 1. Load CSV data
    df = load_csv_data(ticker, converted_timeframe)
    if df.empty:
        print("CSV data not available.")
        return

    # 2. Preprocess CSV data
    df = preprocess_data(df, disabled_features_list)

    # 3. Update the Random Forest model
    update_random_forest(df)

    # 4. Retrieve current open position via live API
    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception as e:
        print(f"Error retrieving position: {e}")
        position_qty = 0.0

    # 5. Compute state: (latest features + predicted_price + current position)
    state = compute_state(df, predicted_price, position_qty)

    # 6. Initialize the global RL agent if needed
    global global_agent
    if global_agent is None:
        state_dim = len(state)
        action_dim = len(ACTION_MAP)  # 4 actions
        global_agent = RLAgent(state_dim, action_dim)

    # 7. Choose an action using the RL agent
    action_idx, _ = global_agent.choose_action(state)
    action = ACTION_MAP[action_idx]

    # 8. Prevent duplicate/redundant trades
    if action == "BUY" and position_qty >= 1:
        action = "NONE"
    elif action == "SELL" and position_qty <= 0:
        action = "NONE"
    elif action == "SHORT" and position_qty <= -1:
        action = "NONE"
    elif action == "COVER" and position_qty >= 0:
        action = "NONE"

    # 9. Execute the trade via API calls
    executed_action = "NONE"
    if action == "BUY":
        buy_shares(ticker, current_price)
        executed_action = "BUY"
    elif action == "SELL":
        sell_shares(ticker, current_price)
        executed_action = "SELL"
    elif action == "SHORT":
        short_shares(ticker, current_price)
        executed_action = "SHORT"
    elif action == "COVER":
        close_short(ticker, current_price)
        executed_action = "COVER"

    # 10. Compute reward and update RL model
    reward = calculate_reward(executed_action, current_price, predicted_price)
    next_state = state  # In live trading, next state update is assumed similar; ideally, it comes from new data.
    online_rl_update(state, action_idx, reward, next_state)

    print(f"Live trade executed: {executed_action}, Reward: {reward}")


def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    """
    Backtesting logic.
    
    Simulates trading over the last 500 candles.
    For each candle, the RL agent decides on an action; duplicate trades are prevented,
    positions are updated, and the agent is trained on every timestep.
    Returns the final action taken on the last candle.
    """
    # 1. Load CSV data
    df = load_csv_data(first_ticker, converted_timeframe)
    if df.empty:
        print("CSV data not available for backtesting.")
        return "NONE"

    # 2. Preprocess CSV data and restrict to last 500 candles
    df = preprocess_data(df, disabled_features_list)
    if len(df) > 500:
        df = df.iloc[-500:]

    # 3. Update the Random Forest model
    update_random_forest(df)

    # 4. Initialize the RL agent if needed using the state dimension from the first row
    global global_agent
    first_row = df.iloc[0]
    init_state = compute_state_from_row(first_row, predicted_price, position_qty)
    if global_agent is None:
        state_dim = len(init_state)
        action_dim = len(ACTION_MAP)
        global_agent = RLAgent(state_dim, action_dim, epsilon=0.3)

    # 5. Initialize simulated position (flat=0, long=1, short=-1)
    position = position_qty
    last_action = "NONE"

    # 6. Iterate over each candle to simulate trading
    for idx, row in df.iterrows():
        # Use the row's close price as current price if available, else fallback
        curr_price = row.get('close', current_price)
        # For RF prediction, drop both 'timestamp' and 'close' to match training features
        features_for_rf = row.drop(labels=['timestamp', 'close'], errors='ignore').values.astype(float)
        rf_pred = get_rf_prediction(features_for_rf)
        pred_price = rf_pred if rf_pred is not None else predicted_price

        # Compute state for this candle
        state = compute_state_from_row(row, pred_price, position)
        action_idx, _ = global_agent.choose_action(state)
        action = ACTION_MAP[action_idx]

        # Prevent duplicate trades
        if action == "BUY" and position >= 1:
            action = "NONE"
        elif action == "SELL" and position <= 0:
            action = "NONE"
        elif action == "SHORT" and position <= -1:
            action = "NONE"
        elif action == "COVER" and position >= 0:
            action = "NONE"

        # Simulate trade execution and update position
        if action == "BUY" and position == 0:
            position = 1
        elif action == "SELL" and position == 1:
            position = 0
        elif action == "SHORT" and position == 0:
            position = -1
        elif action == "COVER" and position == -1:
            position = 0

        # Compute reward with harsh penalty for loss-making actions
        reward = calculate_reward(action, curr_price, pred_price)
        next_state = state  # For simulation, we reuse the current state as the next state

        # Store transition and update the RL model
        global_agent.store_transition(state, action_idx, reward, next_state)
        global_agent.update_model()
        last_action = action

    print(f"Backtest final position: {position}")
    print(f"Backtest final action: {last_action}")
    return last_action
