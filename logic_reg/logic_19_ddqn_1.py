# logic.py
import os
import numpy as np
import pandas as pd
from collections import deque
import random

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

from sklearn.ensemble import RandomForestRegressor

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# -----------------------------------------------------------------------
# Mock or Real Trading API
# -----------------------------------------------------------------------
class MockAPI:
    def get_position(self, ticker):
        class Pos:
            def __init__(self):
                self.qty = 0
        return Pos()

    def buy_shares(self, ticker, quantity=1):
        print(f"[MOCK] BUY {quantity} shares of {ticker}")

    def sell_shares(self, ticker, quantity=1):
        print(f"[MOCK] SELL {quantity} shares of {ticker}")

    def short_shares(self, ticker, quantity=1):
        print(f"[MOCK] SHORT {quantity} shares of {ticker}")

    def close_short(self, ticker, quantity=1):
        print(f"[MOCK] COVER {quantity} shares of {ticker}")

api = MockAPI()
def buy_shares(ticker, quantity=1):
    return api.buy_shares(ticker, quantity)
def sell_shares(ticker, quantity=1):
    return api.sell_shares(ticker, quantity)
def short_shares(ticker, quantity=1):
    return api.short_shares(ticker, quantity)
def close_short(ticker, quantity=1):
    return api.close_short(ticker, quantity)

# -----------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------
DATAFRAMES = {}
RF_MODELS = {}
DDQN_AGENTS = {}
TRAINED_FLAGS = {}  # Tracks if we've already done offline training

# -----------------------------------------------------------------------
# Actions: NO "NONE" in the agent's action space
# -----------------------------------------------------------------------
ACTIONS = ["BUY", "SELL"]

# -----------------------------------------------------------------------
# Convert timeframe
# -----------------------------------------------------------------------
def convert_timeframe(tf_str):
    if "Hour" in tf_str:
        return "H" + tf_str.replace("Hour","")
    elif "Min" in tf_str:
        return "M" + tf_str.replace("Min","")
    return tf_str

# -----------------------------------------------------------------------
# Load CSV data
# -----------------------------------------------------------------------
def load_csv_data(ticker, bar_suffix):
    global DATAFRAMES
    if (ticker, bar_suffix) in DATAFRAMES:
        return DATAFRAMES[(ticker, bar_suffix)]

    filename = f"{ticker}_{bar_suffix}.csv"
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"[WARNING] Could not load {filename}: {e}")
        df = pd.DataFrame()

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.sort_values(by='timestamp', inplace=True)

    df.reset_index(drop=True, inplace=True)
    DATAFRAMES[(ticker, bar_suffix)] = df
    return df

# -----------------------------------------------------------------------
# Filter out disabled features from the CSV
# -----------------------------------------------------------------------
def filter_features(df):
    disabled_feats = os.getenv("DISABLED_FEATURES", "")
    disabled_feats = [d.strip() for d in disabled_feats.split(",") if d.strip()]

    possible_feats = [
        'open', 'high', 'low', 'close', 'volume', 'vwap', 'sentiment',
        'macd_line', 'macd_signal', 'macd_histogram',
        'rsi', 'momentum', 'roc', 'atr', 'obv',
        'bollinger_upper', 'bollinger_lower',
        'ema_9', 'ema_21', 'ema_50', 'ema_200', 'adx',
        'lagged_close_1', 'lagged_close_2', 'lagged_close_3',
        'lagged_close_5', 'lagged_close_10',
        'candle_body_ratio', "predicted_close"
        'wick_dominance',
        'gap_vs_prev',
        'volume_zscore',
        'atr_zscore',
        'rsi_zscore',
        'adx_trend',
        'macd_cross',
        'macd_hist_flip',
        'day_of_week',
        'days_since_high',
        'days_since_low'
    ]
    present = [c for c in possible_feats if c in df.columns]
    enabled_feats = [c for c in present if c not in disabled_feats]

    keep_cols = set(enabled_feats)
    if 'close' in df.columns:
        keep_cols.add('close')
    if 'timestamp' in df.columns:
        keep_cols.add('timestamp')

    filtered_df = df[list(keep_cols.intersection(df.columns))]
    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df

# -----------------------------------------------------------------------
# Random Forest for next close
# -----------------------------------------------------------------------
def get_or_train_rf_model(ticker, bar_suffix, df_filtered):
    global RF_MODELS
    rf_key = (ticker, bar_suffix)
    if rf_key in RF_MODELS:
        return RF_MODELS[rf_key]

    df = df_filtered.copy()
    if 'close' not in df.columns or len(df) < 10:
        RF_MODELS[rf_key] = RandomForestRegressor()
        return RF_MODELS[rf_key]

    df['target_next_close'] = df['close'].shift(-1)
    df.dropna(inplace=True)

    ignore_cols = ['timestamp','close','target_next_close']
    X_cols = [c for c in df.columns if c not in ignore_cols]
    if len(X_cols) == 0 or len(df) < 5:
        RF_MODELS[rf_key] = RandomForestRegressor()
        return RF_MODELS[rf_key]

    X = df[X_cols].values
    y = df['target_next_close'].values

    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    rf.fit(X, y)
    RF_MODELS[rf_key] = rf
    return rf

# -----------------------------------------------------------------------
# Build smaller Q-network
# -----------------------------------------------------------------------
def build_q_network(input_dim, output_dim):
    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu', input_shape=(input_dim,)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(output_dim, activation='linear'))
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

# -----------------------------------------------------------------------
# DDQNAgent (No "NONE" action in the space)
# -----------------------------------------------------------------------
class DDQNAgent:
    def __init__(self, state_size, action_size,
                 gamma=0.99, batch_size=16, buffer_size=5000,
                 epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.99,
                 update_target_freq=50):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.update_target_freq = update_target_freq

        self.online_net = build_q_network(state_size, action_size)
        self.target_net = build_q_network(state_size, action_size)
        self.target_net.set_weights(self.online_net.get_weights())

        self.memory = deque(maxlen=self.buffer_size)
        self.train_steps = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Epsilon-greedy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        qvals = self.online_net.predict(np.array([state]), verbose=0)
        return np.argmax(qvals[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, targets = [], []
        for (state, action, reward, next_state, done) in minibatch:
            state = np.array(state)
            next_state = np.array(next_state)
            current_q = self.online_net.predict(np.array([state]), verbose=0)[0]
            target_q = np.copy(current_q)
            if done:
                target_q[action] = reward
            else:
                next_action = np.argmax(self.online_net.predict(np.array([next_state]), verbose=0)[0])
                next_q = self.target_net.predict(np.array([next_state]), verbose=0)[0]
                target_q[action] = reward + self.gamma * next_q[next_action]

            states.append(state)
            targets.append(target_q)

        states = np.array(states)
        targets = np.array(targets)
        self.online_net.fit(states, targets, epochs=1, verbose=0)
        self.train_steps += 1

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target net every so often
        if self.train_steps % self.update_target_freq == 0:
            self.target_net.set_weights(self.online_net.get_weights())

# -----------------------------------------------------------------------
# Return or create new agent
# -----------------------------------------------------------------------
def get_or_create_ddqn_agent(ticker, bar_suffix, state_size, action_size=2):
    global DDQN_AGENTS
    key = (ticker, bar_suffix)
    if key not in DDQN_AGENTS:
        agent = DDQNAgent(state_size, action_size)
        DDQN_AGENTS[key] = agent
    return DDQN_AGENTS[key]

# -----------------------------------------------------------------------
# Reward function (no penalty for "NONE" since RL won't pick "NONE")
# -----------------------------------------------------------------------
def compute_reward(action_str, old_pos, new_pos, price_diff, stop_triggered=False):
    reward = 0.0
    # PnL from old_pos
    if old_pos == 1:
        reward += price_diff
    elif old_pos == -1:
        reward -= price_diff

    # Stop-loss penalty
    if stop_triggered:
        reward -= 2.0

    # If flipping from +1 to -1 or -1 to +1
    if (old_pos == 1 and new_pos == -1) or (old_pos == -1 and new_pos == 1):
        reward -= 0.3

    # We do not penalize "NONE" because the agent cannot select it.
    return reward

# -----------------------------------------------------------------------
# Offline training function (only once per ticker/timeframe)
# -----------------------------------------------------------------------
def offline_train_ddqn(ticker, bar_suffix, df_filtered, episodes=1):
    global TRAINED_FLAGS
    # Check if already trained
    if (ticker, bar_suffix) in TRAINED_FLAGS:
        print(f"[INFO] Already trained {ticker}_{bar_suffix}, skipping offline training.")
        return

    df_filtered = df_filtered.iloc[-500:].copy()  # keep last 500 rows
    if len(df_filtered) < 2:
        print("[INFO] Not enough data to train.")
        return

    print(f"[INFO] Starting offline training for {ticker}_{bar_suffix} with {episodes} episodes, {len(df_filtered)} rows.")
    rf_model = get_or_train_rf_model(ticker, bar_suffix, df_filtered)

    ignore_cols = ['timestamp','close','target_next_close']
    all_cols = [c for c in df_filtered.columns if c not in ignore_cols]
    # 4 actions: BUY, SELL, SHORT, COVER
    state_size = len(all_cols) + 2 + 1
    agent = get_or_create_ddqn_agent(ticker, bar_suffix, state_size, action_size=2)

    for ep in range(episodes):
        print(f"[INFO] Starting Episode {ep+1}/{episodes}")
        old_pos = 0
        for i in range(len(df_filtered)-1):
            # Print step + reward
            if i % 1 == 0:
                # We'll compute reward after the action.
                pass

            row = df_filtered.iloc[i]
            next_row = df_filtered.iloc[i+1]

            # RF pred
            X_cols = [c for c in all_cols]
            rf_pred = 0.0
            if hasattr(rf_model, 'predict'):
                rf_pred = rf_model.predict(row[X_cols].values.reshape(1, -1))[0]

            # Build state
            featvals = []
            for c in X_cols:
                featvals.append(row[c])
            featvals.append(0.0)  # external pred is unknown offline
            featvals.append(rf_pred)
            featvals.append(old_pos)
            state = np.array(featvals, dtype=float)

            action_idx = agent.act(state)
            action_str = ACTIONS[action_idx]

            # Convert old_pos => new_pos
            new_pos = old_pos
            if action_str == "BUY":
                if old_pos <= 0:
                    new_pos = 1
                else:
                    # Overriding to "NONE" for duplicates
                    action_str = "NONE"
            elif action_str == "SELL":
                if old_pos == 1:
                    new_pos = 0
                else:
                    action_str = "NONE"
            elif action_str == "SHORT":
                if old_pos >= 0:
                    new_pos = -1
                else:
                    action_str = "NONE"
            elif action_str == "COVER":
                if old_pos == -1:
                    new_pos = 0
                else:
                    action_str = "NONE"

            # Price difference for reward
            if 'close' in row and 'close' in next_row:
                price_diff = next_row['close'] - row['close']
            else:
                price_diff = 0.0

            # Simple 5% stop-loss
            stop_triggered = False
            if old_pos == 1 and (next_row['close']/row['close']) < 0.95:
                stop_triggered = True
                new_pos = 0
            elif old_pos == -1 and (next_row['close']/row['close']) > 1.05:
                stop_triggered = True
                new_pos = 0

            reward = compute_reward(action_str, old_pos, new_pos, price_diff, stop_triggered)

            if i % 1 == 0:
                print(f"Episode {ep+1}, Step {i}/{len(df_filtered)-1}, Reward={reward:.4f}")

            # Next state
            if i+1 < len(df_filtered):
                rf_pred2 = 0.0
                if hasattr(rf_model, 'predict'):
                    rf_pred2 = rf_model.predict(next_row[X_cols].values.reshape(1, -1))[0]

                next_feats = []
                for c in X_cols:
                    next_feats.append(next_row[c])
                next_feats.append(0.0)
                next_feats.append(rf_pred2)
                next_feats.append(new_pos)
                next_state = np.array(next_feats, dtype=float)
            else:
                next_state = state.copy()

            done = (i == (len(df_filtered)-2))
            agent.remember(state, action_idx, reward, next_state, done)

            old_pos = new_pos
            agent.replay()

    print(f"[INFO] Offline training done for {ticker}_{bar_suffix}.")
    TRAINED_FLAGS[(ticker, bar_suffix)] = True

# -----------------------------------------------------------------------
# run_logic
# -----------------------------------------------------------------------
def run_logic(current_price, predicted_price, ticker):
    timeframe_str = os.getenv("BAR_TIMEFRAME","4Hour")
    bar_suffix = convert_timeframe(timeframe_str)

    df_raw = load_csv_data(ticker, bar_suffix)
    if df_raw.empty:
        print(f"[ERROR] No data for {ticker}_{bar_suffix}.")
        return

    df_filtered = filter_features(df_raw)
    if len(df_filtered) < 2:
        print(f"[ERROR] Not enough data.")
        return

    # Offline train if not done
    offline_train_ddqn(ticker, bar_suffix, df_filtered, episodes=3)

    last_row = df_filtered.iloc[-1]
    rf_model = get_or_train_rf_model(ticker, bar_suffix, df_filtered)
    ignore_cols = ['timestamp','close','target_next_close']
    all_cols = [c for c in df_filtered.columns if c not in ignore_cols]

    if hasattr(rf_model, 'predict'):
        X_rf = last_row[all_cols].values.reshape(1, -1)
        rf_pred = rf_model.predict(X_rf)[0]
    else:
        rf_pred = current_price

    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception as e:
        print(f"No open position for {ticker}: {e}")
        position_qty = 0.0

    account = api.get_account()
    cash = float(account.cash)

    old_pos = 0
    if position_qty > 0:
        old_pos = 1
    elif position_qty < 0:
        old_pos = -1

    featvals = []
    for c in all_cols:
        featvals.append(last_row[c])
    featvals.append(predicted_price)
    featvals.append(rf_pred)
    featvals.append(old_pos)
    state = np.array(featvals, dtype=float)

    agent = get_or_create_ddqn_agent(ticker, bar_suffix, state.shape[0], action_size=2)
    action_idx = agent.act(state)
    action_str = ACTIONS[action_idx]

    new_pos = old_pos
    if action_str == "BUY":
        if old_pos <= 0:
            max_shares = int(cash // current_price)
            print("buy")
            buy_shares(ticker, max_shares, current_price, predicted_price)
            new_pos = 1
        else:
            action_str = "NONE"
    elif action_str == "SELL":
        if old_pos == 1:
            print("sell")
            sell_shares(ticker, position_qty, current_price, predicted_price)
            new_pos = 0
        else:
            action_str = "NONE"
    elif action_str == "SHORT":
        if old_pos >= 0:
            max_shares = int(cash // current_price)
            print("short")
            short_shares(ticker, max_shares, current_price, predicted_price)
            new_pos = -1
        else:
            action_str = "NONE"
    elif action_str == "COVER":
        if old_pos == -1:
            qty_to_close = abs(position_qty)
            print("cover")
            close_short(ticker, qty_to_close, current_price)
            new_pos = 0
        else:
            action_str = "NONE"

    # optional on-policy update
    dummy_next_state = state.copy()
    agent.remember(state, action_idx, 0.0, dummy_next_state, False)
    agent.replay()

# -----------------------------------------------------------------------
# run_backtest
# -----------------------------------------------------------------------
def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    tickers_str = os.getenv("TICKERS", "TSLA,AAPL")
    first_ticker = tickers_str.split(",")[0].strip()
    timeframe_str = os.getenv("BAR_TIMEFRAME","4Hour")
    bar_suffix = convert_timeframe(timeframe_str)

    df_raw = load_csv_data(first_ticker, bar_suffix)
    if df_raw.empty:
        print(f"[ERROR] No data for {first_ticker}_{bar_suffix}.")
        return "NONE"

    df_filtered = filter_features(df_raw)
    if len(df_filtered) < 2:
        return "NONE"

    offline_train_ddqn(first_ticker, bar_suffix, df_filtered, episodes=1)

    last_row = df_filtered.iloc[-1]
    rf_model = get_or_train_rf_model(first_ticker, bar_suffix, df_filtered)
    ignore_cols = ['timestamp','close','target_next_close']
    all_cols = [c for c in df_filtered.columns if c not in ignore_cols]

    if hasattr(rf_model, 'predict'):
        X_rf = last_row[all_cols].values.reshape(1, -1)
        rf_pred = rf_model.predict(X_rf)[0]
    else:
        rf_pred = current_price

    old_pos = 0
    if position_qty > 0:
        old_pos = 1
    elif position_qty < 0:
        old_pos = -1

    featvals = []
    for c in all_cols:
        featvals.append(last_row[c])
    featvals.append(predicted_price)
    featvals.append(rf_pred)
    featvals.append(old_pos)
    state = np.array(featvals, dtype=float)

    agent = get_or_create_ddqn_agent(first_ticker, bar_suffix, state.shape[0], action_size=2)
    action_idx = agent.act(state)
    action_str = ACTIONS[action_idx]

    # override duplicates
    if action_str == "BUY" and old_pos > 0:
        action_str = "NONE"
    elif action_str == "SELL" and old_pos != 1:
        action_str = "NONE"
    elif action_str == "SHORT" and old_pos < 0:
        action_str = "NONE"
    elif action_str == "COVER" and old_pos != -1:
        action_str = "NONE"

    return action_str
