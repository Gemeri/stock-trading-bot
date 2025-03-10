# logic.py
import os
import pandas as pd
import numpy as np
import random
from collections import deque
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# --- Configuration and Utility Functions ---

# Mapping from .env BAR_TIMEFRAME values to CSV filename suffixes.
TIMEFRAME_MAPPING = {
    "4Hour": "H4",
    "2Hour": "H2",
    "1Hour": "H1",
    "30Min": "M30",
    "15Min": "M15"
}

# Full list of CSV columns (features) available.
VALID_FEATURES = [
    'timeframe', 'open', 'high', 'low', 'close', 'volume', 'vwap',
    'price_change', 'high_low_range', 'log_volume', 'sentiment',
    'price_return', 'candle_rise', 'body_size', 'wick_to_body', 'macd_line',
    'rsi', 'momentum', 'roc', 'atr', 'hist_vol', 'obv', 'volume_change',
    'stoch_k', 'bollinger_upper', 'bollinger_lower',
    'lagged_close_1', 'lagged_close_2', 'lagged_close_3',
    'lagged_close_5', 'lagged_close_10'
]

def convert_bar_timeframe(bar_timeframe):
    """Convert BAR_TIMEFRAME (e.g. '4Hour') to CSV filename suffix (e.g. 'H4')."""
    return TIMEFRAME_MAPPING.get(bar_timeframe, "H1")

def load_csv_data(ticker, bar_timeframe, disabled_features):
    """
    Load the CSV file for a given ticker and timeframe.
    Disabled features (comma‐separated string) are removed from the dataframe.
    """
    converted_tf = convert_bar_timeframe(bar_timeframe)
    filename = f"{ticker}_{converted_tf}.csv"
    df = pd.read_csv(filename)
    # Remove any disabled columns.
    disabled_list = [feat.strip() for feat in disabled_features.split(",")] if disabled_features else []
    enabled_columns = [col for col in df.columns if col not in disabled_list]
    df = df[enabled_columns]
    return df

# --- Global Trade Tracking ---
# This dictionary holds the trade entry price for each ticker.
trade_entries = {}

# --- Reinforcement Learning (DQN) Agent Definition ---

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size    = state_size
        self.action_size   = action_size
        self.memory        = deque(maxlen=5000)  # Increased memory for better experience replay.
        self.gamma         = 0.95              # Discount rate
        self.epsilon       = 1.0               # Exploration rate
        self.epsilon_min   = 0.01
        self.epsilon_decay = 0.99              # Slightly slower decay for more exploration early on.
        self.learning_rate = 0.001
        self.model         = self._build_model()
    
    def _build_model(self):
        # Build a simple fully connected neural network for DQN.
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        # Use updated parameter "learning_rate" instead of deprecated "lr"
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        # Epsilon-greedy action selection.
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        # Train on a random sample from memory.
        minibatch = random.sample(self.memory, min(batch_size, len(self.memory)))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# --- Global Agent and RF Model Instances ---
global_agent = None
global_rf_model = None

def initialize_agent_and_rf(df):
    """
    Initialize (or reuse) the global RL agent and Random Forest model.
    The RF model is trained on historical CSV data (features to predict 'close').
    """
    global global_agent, global_rf_model
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    state_data = df.iloc[-1][numeric_cols].values
    # Append two additional values: the refined predicted price and current position.
    state_size = len(state_data) + 2  
    action_size = 5  # Actions: BUY, SELL, SHORT, COVER, NONE
    if global_agent is None:
        global_agent = DQNAgent(state_size, action_size)
    if global_rf_model is None:
        if 'close' in df.columns:
            X = df.select_dtypes(include=[np.number]).drop(columns=['close'], errors='ignore')
            y = df['close']
            global_rf_model = RandomForestRegressor(n_estimators=100)
            global_rf_model.fit(X, y)
    return global_agent, global_rf_model

def build_state(df, predicted_price, current_position):
    """
    Construct the RL state vector.
    Start with numeric features from the last CSV row and append:
      - refined predicted_price (to be updated)
      - current open position.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_features = df.iloc[-1][numeric_cols].values.tolist()
    state = np.array(numeric_features + [predicted_price, current_position])
    return state.reshape(1, -1)

# Map action indices to action strings.
ACTION_MAPPING = {
    0: "BUY",
    1: "SELL",
    2: "SHORT",
    3: "COVER",
    4: "NONE"
}

# --- Reward Calculation Utility ---
def calculate_reward(ticker, action, current_price, position_qty):
    """
    Compute reward based on trade outcomes.
    For BUY/SHORT actions (opening a position), record entry price.
    For SELL/COVER actions (closing a position), compute return from entry.
    A small negative reward is applied for "NONE" or duplicate actions.
    """
    reward = 0
    if action == "BUY":
        # Open a long position if no active trade.
        if position_qty == 0:
            trade_entries[ticker] = current_price
            reward = 0
        else:
            reward = -0.01  # Minor penalty if already in a trade.
    elif action == "SELL":
        # Close a long position.
        entry = trade_entries.get(ticker)
        if position_qty > 0 and entry is not None:
            reward = (current_price - entry) / entry
            trade_entries[ticker] = None
        else:
            reward = -0.01
    elif action == "SHORT":
        # Open a short position.
        if position_qty == 0:
            trade_entries[ticker] = current_price
            reward = 0
        else:
            reward = -0.01
    elif action == "COVER":
        # Close a short position.
        entry = trade_entries.get(ticker)
        if position_qty < 0 and entry is not None:
            reward = (entry - current_price) / entry
            trade_entries[ticker] = None
        else:
            reward = -0.01
    elif action == "NONE":
        reward = -0.01  # Penalize inaction slightly.
    return reward

# --- Core Function: run_logic (Live Trading) ---
def run_logic(current_price, predicted_price, ticker):
    """
    Live trading logic.
    1. Loads the CSV for the ticker (with timeframe conversion).
    2. Filters out disabled features.
    3. Initializes/updates the RL (DQN) agent and RF model.
    4. Retrieves the current open position via the live API.
    5. Builds the state (CSV features + predicted price + current position).
    6. The agent selects an action, and duplicate trades are prevented.
    7. The trade is executed via the live API, and reward is computed based on trade outcomes.
    8. The RL agent is updated via online learning.
    """
    from forest import api, buy_shares, sell_shares, short_shares, close_short

    bar_timeframe     = os.getenv("BAR_TIMEFRAME", "1Hour")
    disabled_features = os.getenv("DISABLED_FEATURES", "")
    df = load_csv_data(ticker, bar_timeframe, disabled_features)
    agent, rf_model = initialize_agent_and_rf(df)
    
    # Get current open position from the live API.
    pos = api.get_position(ticker)
    try:
        position_qty = float(pos.qty)
    except Exception:
        position_qty = 0.0
    
    state = build_state(df, predicted_price, position_qty)
    
    # Use RF model to refine the predicted price.
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        rf_features = df.iloc[-1][numeric_cols].values.reshape(1, -1)
        rf_prediction = rf_model.predict(rf_features)[0]
    except Exception:
        rf_prediction = predicted_price
    combined_predicted_price = (predicted_price + rf_prediction) / 2.0
    state[0, -2] = combined_predicted_price  # Update predicted price in state.
    
    action_index = agent.act(state)
    action = ACTION_MAPPING[action_index]
    
    # Override "NONE" if market conditions favor trading.
    if action == "NONE":
        if combined_predicted_price > current_price and position_qty == 0:
            action = "BUY"
        elif combined_predicted_price < current_price and position_qty > 0:
            action = "SELL"
    
    # Prevent duplicate trades.
    if action == "BUY" and position_qty >= 1:
        action = "NONE"
    if action == "SELL" and position_qty <= 0:
        action = "NONE"
    if action == "SHORT" and position_qty <= -1:
        action = "NONE"
    if action == "COVER" and position_qty >= 0:
        action = "NONE"
    
    # Execute trade via live API.
    if action == "BUY":
        buy_shares(ticker, 1)
    elif action == "SELL":
        sell_shares(ticker, 1)
    elif action == "SHORT":
        short_shares(ticker, 1)
    elif action == "COVER":
        close_short(ticker, 1)
    # "NONE" results in no trade.
    
    # Compute reward based on trade outcome.
    reward = calculate_reward(ticker, action, current_price, position_qty)
    
    # Online RL update.
    next_state = state  # In practice, next_state should reflect new market data.
    done = False
    agent.remember(state, action_index, reward, next_state, done)
    agent.replay(32)
    
    return action

# --- Core Function: run_backtest (Backtesting) ---
def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    """
    Backtesting logic.
    1. Loads the CSV for the first ticker in TICKERS (with timeframe conversion).
    2. Filters out disabled features.
    3. Uses the same RL agent and RF model.
    4. Builds the state (CSV features + predicted price + provided position_qty).
    5. Computes reward in a similar fashion using global trade tracking.
    6. Returns the selected action.
    """
    bar_timeframe     = os.getenv("BAR_TIMEFRAME", "1Hour")
    disabled_features = os.getenv("DISABLED_FEATURES", "")
    tickers_env       = os.getenv("TICKERS", "TSLA")
    first_ticker      = tickers_env.split(",")[0]
    
    df = load_csv_data(first_ticker, bar_timeframe, disabled_features)
    agent, rf_model = initialize_agent_and_rf(df)
    
    state = build_state(df, predicted_price, position_qty)
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        rf_features = df.iloc[-1][numeric_cols].values.reshape(1, -1)
        rf_prediction = rf_model.predict(rf_features)[0]
    except Exception:
        rf_prediction = predicted_price
    combined_predicted_price = (predicted_price + rf_prediction) / 2.0
    state[0, -2] = combined_predicted_price
    
    action_index = agent.act(state)
    action = ACTION_MAPPING[action_index]
    
    if action == "NONE":
        if combined_predicted_price > current_price and position_qty == 0:
            action = "BUY"
        elif combined_predicted_price < current_price and position_qty > 0:
            action = "SELL"
    
    if action == "BUY" and position_qty >= 1:
        action = "NONE"
    if action == "SELL" and position_qty <= 0:
        action = "NONE"
    if action == "SHORT" and position_qty <= -1:
        action = "NONE"
    if action == "COVER" and position_qty >= 0:
        action = "NONE"
    
    # For backtesting, update trade tracking and compute reward similarly (if desired).
    _ = calculate_reward(first_ticker, action, current_price, position_qty)
    
    return action
