import os
import numpy as np
import pandas as pd
import random
import pickle
import logging
from collections import deque
from dotenv import load_dotenv

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler

# =============================================================================
# Global Config / Paths
# =============================================================================

agent = None
scaler = None
STATE_SIZE = None

MODEL_WEIGHTS_PATH = "models/ddqn_online.weights.h5"
TARGET_WEIGHTS_PATH = "models/ddqn_online_target.weights.h5"
SCALER_PATH = "models/feature_scaler.pkl"

THRESHOLD = 0.01  # edge filter for opening a long
ACTIONS = ["BUY", "SELL", "NONE"]

# For reproducibility
_seed = 1337
random.seed(_seed)
np.random.seed(_seed)
tf.random.set_seed(_seed)
os.environ["PYTHONHASHSEED"] = str(_seed)

# =============================================================================
# Environment & Helper Functions
# =============================================================================

# Load .env variables
load_dotenv()
BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "1Hour")
BACKTEST_TICKER = os.getenv("TICKERS", "TSLA,AAPL").split(",")
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
    # Exclude predicted_close from the feature matrix to avoid duplication with appended predicted_price
    enabled = [feat for feat in ALL_FEATURES if feat not in DISABLED_FEATURES and feat != "predicted_close"]
    return enabled

def get_csv_filename(ticker: str) -> str:
    return f"data/{ticker}_{CONVERTED_TIMEFRAME}.csv"

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
    cols = [feat for feat in get_enabled_features() if feat in df.columns]
    out = df[cols].apply(pd.to_numeric, errors='coerce').astype(np.float32)
    # clean NaNs/infs before scaling
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.fillna(method='ffill', inplace=True)
    out.fillna(0.0, inplace=True)
    return out

def _save_scaler():
    if scaler is None:
        return
    os.makedirs("models", exist_ok=True)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

def _load_scaler():
    global scaler
    if os.path.isfile(SCALER_PATH) and os.path.getsize(SCALER_PATH) > 0:
        try:
            with open(SCALER_PATH, "rb") as f:
                scaler = pickle.load(f)
        except Exception as e:
            logging.warning(f"Could not load scaler: {e}")
            scaler = None

def transform_features(df_features: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
    """
    Apply StandardScaler. Fit only on historical (offline) data, persist for live/backtest.
    """
    global scaler
    if df_features.empty:
        return df_features

    if fit or scaler is None:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df_features.values)
        _save_scaler()
    else:
        # in case columns mismatch (shouldn't), fall back to identity
        try:
            scaled = scaler.transform(df_features.values)
        except Exception as e:
            logging.warning(f"Scaler transform failed ({e}); using unscaled features.")
            scaled = df_features.values

    return pd.DataFrame(scaled, columns=df_features.columns, index=df_features.index).astype(np.float32)

def get_state(df_scaled: pd.DataFrame, predicted_price: float, current_position: float) -> np.ndarray:
    """
    Build state from the LAST ROW of scaled features + [predicted_price, current_position]
    """
    latest = df_scaled.iloc[-1].values.astype(np.float32)
    state = np.append(latest, np.array([predicted_price, current_position], dtype=np.float32))
    return state.reshape(1, -1)

# =============================================================================
# DDQN Agent Implementation
# =============================================================================

class DDQNAgent:
    def __init__(self, state_size, action_size,
                 gamma=0.99, learning_rate=1e-3,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 batch_size=64, memory_size=100_000,
                 target_update_freq=500):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # discount factor
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        self.memory = deque(maxlen=memory_size)

        # Main network and target network for DDQN
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        self.step = 0
        self.target_update_freq = target_update_freq

    def _build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        # Huber loss tends to be stabler than MSE; clip gradients a bit
        model.compile(loss=tf.keras.losses.Huber(), optimizer=Adam(learning_rate=self.learning_rate, clipnorm=1.0))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((
            state.astype(np.float32),
            int(action),
            np.float32(reward),
            next_state.astype(np.float32),
            bool(done)
        ))

    def act(self, state, mask=None):
        """
        Epsilon-greedy with optional boolean mask of allowed actions.
        mask: list/np.array of bools, len == action_size.
        """
        if mask is not None:
            mask = np.array(mask, dtype=bool)
            if mask.shape[0] != self.action_size:
                raise ValueError("Mask length does not match action_size")

        if np.random.rand() <= self.epsilon:
            if mask is None:
                return random.randrange(self.action_size)
            allowed = np.where(mask)[0]
            if allowed.size == 0:
                return self.action_size - 1  # fallback to last action (NONE)
            return int(np.random.choice(allowed))

        q_values = self.model.predict(state.astype(np.float32), verbose=0)[0]
        if mask is not None:
            q_values = np.where(mask, q_values, -1e9)
        return int(np.argmax(q_values))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states      = np.vstack([m[0] for m in minibatch]).astype(np.float32)
        actions     = np.array([m[1] for m in minibatch], dtype=np.int32)
        rewards     = np.array([m[2] for m in minibatch], dtype=np.float32)
        next_states = np.vstack([m[3] for m in minibatch]).astype(np.float32)
        dones       = np.array([m[4] for m in minibatch], dtype=bool)

        target      = self.model.predict(states, verbose=0)
        next_online = self.model.predict(next_states, verbose=0)
        next_target = self.target_model.predict(next_states, verbose=0)

        best_next_actions = np.argmax(next_online, axis=1)
        target_q = rewards + (~dones) * self.gamma * next_target[np.arange(self.batch_size), best_next_actions]
        target[np.arange(self.batch_size), actions] = target_q

        self.model.fit(states, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.step += 1
        if self.step % self.target_update_freq == 0:
            self.update_target_model()

# =============================================================================
# Model Init / Persistence
# =============================================================================

def init_models_if_needed(ticker: str, until_timestamp=None):
    global agent, STATE_SIZE
    if agent is not None:
        return  # already initialised

    # Derive feature count using realistic slice of history
    try:
        df = filter_features(load_csv_data(ticker, until_timestamp))
        feature_count = df.shape[1]
    except Exception:
        feature_count = len(get_enabled_features())

    STATE_SIZE = feature_count + 2  # extra slots for predicted_price & position
    ACTION_SIZE = len(ACTIONS)
    agent = DDQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)

def _model_files_exist() -> bool:
    return (os.path.isfile(MODEL_WEIGHTS_PATH) and os.path.getsize(MODEL_WEIGHTS_PATH) > 0 and
            os.path.isfile(TARGET_WEIGHTS_PATH) and os.path.getsize(TARGET_WEIGHTS_PATH) > 0)

def _save_agent():
    os.makedirs("models", exist_ok=True)
    agent.model.save_weights(MODEL_WEIGHTS_PATH)
    agent.target_model.save_weights(TARGET_WEIGHTS_PATH)

def _load_agent():
    # Need to build the models before loading weights
    agent.model.build((None, STATE_SIZE))
    agent.target_model.build((None, STATE_SIZE))
    agent.model.load_weights(MODEL_WEIGHTS_PATH)
    agent.target_model.load_weights(TARGET_WEIGHTS_PATH)
    agent.update_target_model()

_model_loaded = False

def _ensure_model(ticker: str, until_timestamp=None):
    """
    Make sure a usable model exists and is loaded into `agent`.
    Also ensure scaler is loaded (or fitted offline if first run).
    """
    global _model_loaded
    if _model_loaded:
        return

    _load_scaler()

    if _model_files_exist():
        _load_agent()
    else:
        _offline_pretrain(ticker, until_timestamp)
        _save_agent()

    _model_loaded = True

# =============================================================================
# Offline Pretrain (uses next_state correctly + persisted scaler)
# =============================================================================

def _offline_pretrain(ticker: str, until_timestamp=None, epochs: int = 3):
    """
    Train the DDQN purely on historical rows (no live execution).
    If `until_timestamp` is given, we only use data ≤ that point.
    """
    init_models_if_needed(ticker, until_timestamp)

    df_raw       = load_csv_data(ticker, until_timestamp)
    df_features  = filter_features(df_raw)
    if df_features.empty:
        logging.warning(f"No data available for offline pre-train on {ticker}.")
        return

    # Fit scaler once on full history and scale features
    df_scaled = transform_features(df_features, fit=True)

    closes = df_raw["close"].astype(np.float32).values if "close" in df_raw.columns else None
    if closes is None:
        logging.warning("No 'close' column present; cannot compute rewards properly.")
        return

    for ep in range(epochs):
        logging.info(f"Offline epoch {ep+1}/{epochs}")
        position_qty = 0.0

        # iterate until len-2 so i+1 exists for next_state/reward
        for i in range(0, len(df_scaled) - 1):
            cur_close       = float(closes[i])
            next_close      = float(closes[i + 1])
            predicted_price = float(df_raw.iloc[i].get("predicted_close", cur_close))

            # state at time i (use scaled features at row i)
            state = get_state(df_scaled.iloc[:i+1], predicted_price, position_qty)

            # mask: disallow BUY if already long; disallow SELL if flat
            mask = [
                position_qty <= 0,   # BUY allowed only if not already long
                position_qty > 0,    # SELL allowed only if long
                True                 # NONE always allowed
            ]

            # edge filter for opening long
            if predicted_price < cur_close * (1.0 + THRESHOLD):
                mask[0] = False  # disallow BUY

            act_idx = agent.act(state, mask=mask)
            action  = ACTIONS[act_idx]

            # naive position book-keeping (long/flat only)
            if action == "BUY":
                position_qty = 1.0
            elif action == "SELL":
                position_qty = 0.0

            # next_state at time i+1 (post-action position)
            next_predicted = float(df_raw.iloc[i+1].get("predicted_close", next_close))
            next_state = get_state(df_scaled.iloc[:i+2], next_predicted, position_qty)

            # reward from next bar movement; reward SELL if price falls
            if action == "BUY":
                reward = next_close - cur_close
            elif action == "SELL":
                reward = cur_close - next_close
            else:
                reward = -0.05  # mild time penalty

            done = (i == len(df_scaled) - 2)
            agent.remember(state, act_idx, reward, next_state, done)
            agent.replay()

        agent.update_target_model()

    # keep some exploration but not crazy high
    agent.epsilon = max(0.10, agent.epsilon_min * 2)

# =============================================================================
# Live & Backtest Logic (with proper pending transition handling)
# =============================================================================

# Pending transition for LIVE: finalized on the next call when next price/next state is known
_last_live = {"state": None, "action_idx": None, "price": None, "pos": 0.0}

# Pending transitions for BACKTEST, per ticker
_bt_last = {}  # ticker -> {"state":..., "action_idx":..., "price":..., "pos":...}

def _action_mask(position_qty: float, current_price: float, predicted_price: float):
    """
    Mask illegal/undesired actions given position & threshold edge filter.
    Long-only: BUY to open, SELL to close, NONE always allowed.
    """
    allow_buy  = position_qty <= 0
    allow_sell = position_qty > 0
    allow_none = True

    # Edge filter for BUY: require predicted >= current * (1+THRESHOLD)
    if predicted_price < current_price * (1.0 + THRESHOLD):
        allow_buy = False

    return [allow_buy, allow_sell, allow_none]

# ---------------------------------------------------------------------------
# run_logic – live trading
# ---------------------------------------------------------------------------
def run_logic(current_price: float, predicted_price: float, ticker: str):
    init_models_if_needed(ticker)
    _ensure_model(ticker)  # warm-start hook (loads model and scaler if needed)

    from forest import api, buy_shares, sell_shares  # position sizing is handled inside these

    # ------------------------------ data
    try:
        df_raw = load_csv_data(ticker)
        df_features = filter_features(df_raw)
        df_scaled = transform_features(df_features, fit=False)
        if df_scaled.empty:
            logging.error(f"No features available for {ticker}.")
            return
    except Exception as e:
        logging.error(f"Error loading CSV for {ticker}: {e}")
        return

    try:
        position_qty = float(api.get_position(ticker).qty)
    except Exception:
        position_qty = 0.0

    # Build current state from latest features
    state = get_state(df_scaled, float(predicted_price), float(position_qty))

    # ------------------------------ finalize previous transition (if any)
    if _last_live["state"] is not None and df_scaled.shape[0] >= 1:
        next_state = state
        cur_close = float(df_raw.iloc[-1]["close"]) if "close" in df_raw.columns else float(current_price)
        prev_close = float(_last_live["price"])
        ai = int(_last_live["action_idx"])
        a  = ACTIONS[ai]
        if a == "BUY":
            reward = cur_close - prev_close
        elif a == "SELL":
            reward = prev_close - cur_close
        else:
            reward = -0.05
        agent.remember(_last_live["state"], ai, reward, next_state, False)
        agent.replay()

    # ------------------------------ choose action with mask
    mask = _action_mask(position_qty, current_price, predicted_price)
    action_index = agent.act(state, mask=mask)
    action       = ACTIONS[action_index]

    # ------------------------------ execute
    try:
        cash = float(api.get_account().cash)
    except Exception:
        cash = 0.0

    if action == "BUY":
        # Position sizing is handled by your broker helpers; keep signature as you had
        buy_shares(ticker, int(cash // max(current_price, 1e-9)), current_price, predicted_price)
    elif action == "SELL":
        if position_qty > 0:
            sell_shares(ticker, position_qty, current_price, predicted_price)
    # NONE → do nothing

    # ------------------------------ stash for next step's reward
    last_close = float(df_raw.iloc[-1]["close"]) if "close" in df_raw.columns else float(current_price)
    _last_live.update({
        "state": state,
        "action_idx": action_index,
        "price": last_close,
        "pos": float(position_qty)
    })

# ---------------------------------------------------------------------------
# run_backtest – offline simulation (no leakage; pending finalized next call)
# ---------------------------------------------------------------------------
def run_backtest(current_price: float,
                 predicted_price: float,
                 position_qty: float,
                 current_timestamp,
                 candles,           # kept for API compatibility
                 ticker: str) -> str:

    init_models_if_needed(ticker, current_timestamp)
    _ensure_model(ticker, current_timestamp)   # warm-start hook

    # ------------------------------ data slice (≤ current_timestamp)
    try:
        df_raw = load_csv_data(ticker, current_timestamp)
        df_features = filter_features(df_raw)
        df_scaled = transform_features(df_features, fit=False)
        if df_scaled.empty:
            logging.warning(f"No data for {ticker} up to {current_timestamp}")
            return "NONE"
    except Exception as e:
        logging.error(f"Backtest load error for {ticker}: {e}")
        return "NONE"

    # Build current state
    state = get_state(df_scaled, float(predicted_price), float(position_qty))

    # ------------------------------ finalize previous pending for this ticker
    last = _bt_last.get(ticker)
    if last is not None:
        next_state = state
        cur_close = float(df_raw.iloc[-1]["close"]) if "close" in df_raw.columns else float(current_price)
        prev_close = float(last["price"])
        ai = int(last["action_idx"])
        a  = ACTIONS[ai]
        if a == "BUY":
            reward = cur_close - prev_close
        elif a == "SELL":
            reward = prev_close - cur_close
        else:
            reward = -0.05
        agent.remember(last["state"], ai, reward, next_state, False)
        agent.replay()
        # no need to update_target_model each step; handled by agent

    # ------------------------------ choose action with mask (long-only)
    mask = _action_mask(position_qty, current_price, predicted_price)
    action_index = agent.act(state, mask=mask)
    action       = ACTIONS[action_index]

    # ------------------------------ stash for next step
    last_close = float(df_raw.iloc[-1]["close"]) if "close" in df_raw.columns else float(current_price)
    _bt_last[ticker] = {
        "state": state,
        "action_idx": action_index,
        "price": last_close,
        "pos": float(position_qty)
    }

    return action
