import os
import numpy as np
import pandas as pd
import random
from dotenv import load_dotenv
import logging

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

agent = None
STATE_SIZE = None

# =============================================================================
# Environment & Helper Functions
# =============================================================================

# Load .env variables
load_dotenv()
BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "1Hour")
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
    'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions', 'sentiment',
    'price_change', 'high_low_range', 'log_volume', 'macd_line', 'macd_signal', 'macd_histogram',
    'rsi', 'roc', 'atr', 'ema_9', 'ema_21', 'ema_50', 'ema_200', 'adx', 'obv', 'bollinger_upper', 
    'bollinger_lower', 'bollinger_percB', 'returns_1', 'returns_3', 'returns_5', 'std_5', 'std_10',
    'lagged_close_1', 'lagged_close_2', 'lagged_close_3', 'lagged_close_5', 'lagged_close_10',
    'candle_body_ratio', 'wick_dominance', 'gap_vs_prev', 'volume_zscore', 'atr_zscore', 'rsi_zscore',
    'macd_cross', 'macd_hist_flip', 'month', 'hour_sin', 'hour_cos', 'day_of_week_sin', 
    'day_of_week_cos', 'days_since_high', 'days_since_low',
]

def get_enabled_features():
    return [feat for feat in ALL_FEATURES if feat not in DISABLED_FEATURES]

def get_csv_filename(ticker: str) -> str:
    return os.path.join("data", f"{ticker}_{CONVERTED_TIMEFRAME}.csv")

def load_csv_data(ticker: str, until_timestamp=None) -> pd.DataFrame:
    filename = get_csv_filename(ticker)

    # Always try to parse the `timestamp` column if it exists
    try:
        df = pd.read_csv(filename, parse_dates=["timestamp"])
    except ValueError:
        # No timestamp column – load raw
        df = pd.read_csv(filename)

    # Optional leakage-prevention filter
    if until_timestamp is not None and "timestamp" in df.columns:
        cutoff = pd.to_datetime(until_timestamp, utc=True)
        df = df[df["timestamp"] <= cutoff]

    return df.reset_index(drop=True)

def filter_features(df: pd.DataFrame) -> pd.DataFrame:
    enabled = get_enabled_features()
    features = [feat for feat in enabled if feat in df.columns]
    df_filtered = df[features].copy()
    # Convert to numeric (non-numeric values become NaN)
    df_filtered = df_filtered.apply(pd.to_numeric, errors='coerce')
    return df_filtered

def get_state(df: pd.DataFrame, predicted_price: float, current_position: float) -> np.ndarray:
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
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Random action
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
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


def init_models_if_needed(ticker: str, until_timestamp=None):
    global agent, STATE_SIZE

    if agent is not None:
        return  # already initialised

    # Derive feature count using the realistic slice of history
    try:
        df = filter_features(load_csv_data(ticker, until_timestamp))
        feature_count = df.shape[1]
    except Exception:
        feature_count = len(get_enabled_features())

    STATE_SIZE = feature_count + 2  # extra slots for predicted_price & position
    ACTION_SIZE = len(ACTIONS)
    agent = DDQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)

# ---------------------------------------------------------------------------
# Shared constant ­– keep near the top of your module
# ---------------------------------------------------------------------------
THRESHOLD = 0.01        # 1 % difference required between predicted & current

# ---------------------------------------------------------------------------
# Updated run_logic – live trading
# ---------------------------------------------------------------------------
def run_logic(current_price: float, predicted_price: float, ticker: str):
    init_models_if_needed(ticker)

    from forest import api, buy_shares, sell_shares, short_shares, close_short

    # ------------------------------------------------------------------ data
    try:
        df_filtered = filter_features(load_csv_data(ticker))
    except Exception as e:
        logging.error(f"Error loading CSV for {ticker}: {e}")
        return

    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception:
        position_qty = 0.0

    cash = float(api.get_account().cash)

    # ------------------------------------------------------------- RL action
    state = get_state(df_filtered, predicted_price, position_qty)
    action_index = agent.act(state)
    action = ACTIONS[action_index]

    # ---------------- guardrails: duplicate positions & price-edge filter ---
    if action == "BUY"  and position_qty >= 1:
        action = "NONE"
    if action == "SELL" and position_qty <= 0:
        action = "NONE"

    if action == "BUY"  and predicted_price < current_price * (1 + THRESHOLD):
        action = "NONE"
    if action == "SELL" and predicted_price > current_price * (1 - THRESHOLD):
        action = "NONE"

    # ------------------------------------------------------------- execution
    if action == "BUY":
        max_shares = int(cash // current_price)
        buy_shares(ticker, max_shares, current_price, predicted_price)
        logging.info(f"Executed BUY for {ticker}")

    elif action == "SELL":
        sell_shares(ticker, position_qty, current_price, predicted_price)
        logging.info(f"Executed SELL for {ticker}")

    elif action == "SHORT":
        max_shares = int(cash // current_price)
        short_shares(ticker, max_shares, current_price, predicted_price)
        logging.info(f"Executed SHORT for {ticker}")

    elif action == "COVER":
        close_short(ticker, abs(position_qty), current_price)
        logging.info(f"Executed COVER for {ticker}")

    else:
        logging.debug("No action taken.")

    # ------------------------------------------------------------- reward
    if len(df_filtered) >= 2 and "close" in df_filtered.columns:
        prev_close = float(df_filtered.iloc[-2]["close"])
    else:
        prev_close = current_price

    if action != "NONE":
        trade_dir = 1 if action in ["BUY", "COVER"] else -1
        reward = (current_price - prev_close) * trade_dir
    else:
        reward = -0.1       # small penalty for idleness

    # ------------------------------------------------------ RL bookkeeping
    next_state = state          # one-step setup; adjust if you have look-ahead
    done = False

    agent.remember(state, action_index, reward, next_state, done)
    agent.replay()
    agent.update_target_model()

# ---------------------------------------------------------------------------
# Updated run_backtest – offline simulation
# ---------------------------------------------------------------------------
def run_backtest(current_price: float,
                 predicted_price: float,
                 position_qty: float,
                 current_timestamp,
                 candles,          # kept for API compatibility
                 ticker: str) -> str:

    init_models_if_needed(ticker, current_timestamp)

    # --------------------------------------------------------- data slice
    try:
        df_filtered = filter_features(load_csv_data(ticker, current_timestamp).tail(500))
    except Exception as e:
        logging.error(f"Backtest load error for {ticker}: {e}")
        return "NONE"

    if df_filtered.empty:
        logging.warning(f"No data for {ticker} up to {current_timestamp}")
        return "NONE"

    # ------------------------------------------------------------- RL action
    state = get_state(df_filtered, predicted_price, position_qty)
    action_index = agent.act(state)
    action = ACTIONS[action_index]

    # ------------ guardrails: duplicate positions & price-edge filter -------
    if action == "BUY"  and position_qty >= 1:
        action = "NONE"
    if action == "SELL" and position_qty <= 0:
        action = "NONE"

    if action == "BUY"  and predicted_price < current_price * (1 + THRESHOLD):
        action = "NONE"
    if action == "SELL" and predicted_price > current_price * (1 - THRESHOLD):
        action = "NONE"

    # ------------------------------------------------------------- reward
    if len(df_filtered) >= 2 and "close" in df_filtered.columns:
        prev_close = float(df_filtered.iloc[-2]["close"])
    else:
        prev_close = current_price

    if action != "NONE":
        trade_dir = 1 if action == "BUY" else -1
        reward = (current_price - prev_close) * trade_dir
    else:
        reward = -0.1

    # ------------------------------------------------------ RL bookkeeping
    next_state = state
    done = False

    agent.remember(state, action_index, reward, next_state, done)
    agent.replay()
    agent.update_target_model()

    # -------------------------------------- return decision to backtester
    return action