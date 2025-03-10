#!/usr/bin/env python3
"""
logic_2_rl_agent.py

An advanced external trading logic script that uses a reinforcement learning (RL)
agent to make trade decisions based on CSV historical data and environment variables.
This script defines two main functions:
  - run_logic: Executes live trading actions by calling the forest API trade functions.
  - run_backtest: Runs a backtest simulation and returns a suggested action string.

The RL agent is trained offline on historical data extracted from CSV files whose
names depend on the ticker and timeframe defined in the .env file. The agent uses only
the “enabled” features from the CSV data (i.e. those not listed in DISABLED_FEATURES)
plus the externally provided predicted price as input. The agent then outputs one of four
actions: BUY, SELL, SHORT, or COVER. A NONE decision is only used by the live trading logic
to prevent duplicate trades.

Environment variables used (via .env file):
    - BAR_TIMEFRAME: one of ["4Hour", "2Hour", "1Hour", "30Min", "15Min"].
      (Mapped to CSV timeframe codes: H4, H2, H1, M30, M15.)
    - TICKERS: comma separated list (e.g. TSLA,AAPL,...). For backtesting, only the first ticker is used.
    - DISABLED_FEATURES: comma separated list of features to ignore in the CSV data.
    - TOTAL_EPISODES: total episodes we want to eventually train to (default 300).
    - CHUNK_SIZE: how many episodes to train at once in each chunk (default 100).
"""

import os
import sys
import numpy as np
import pandas as pd
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Mapping for BAR_TIMEFRAME conversion from .env to CSV timeframe suffix
TIMEFRAME_MAPPING = {
    "4Hour": "H4",
    "2Hour": "H2",
    "1Hour": "H1",
    "30Min": "M30",
    "15Min": "M15"
}

# Get environment variables
BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "1Hour")
TICKERS = os.getenv("TICKERS", "TSLA")
DISABLED_FEATURES = os.getenv(
    "DISABLED_FEATURES",
    "body_size,candle_rise,high_low_range,hist_vol,log_volume,macd_line,price_change,price_return,roc,rsi,stoch_k,transactions,volume,volume_change,wick_to_body"
)

# New environment variables for partial training
TOTAL_EPISODES = int(os.getenv("TOTAL_EPISODES", 150))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 150))

# Define the full set of available CSV features
AVAILABLE_FEATURES = [
    'open','high','low','close','volume','vwap',
    'price_change','high_low_range','log_volume','sentiment',
    'price_return','candle_rise','body_size','wick_to_body','macd_line',
    'rsi','momentum','roc','atr','hist_vol','obv','volume_change',
    'stoch_k','bollinger_upper','bollinger_lower',
    'lagged_close_1','lagged_close_2','lagged_close_3','lagged_close_5','lagged_close_10'
]

# Compute enabled features by excluding disabled ones.
disabled = set([feat.strip() for feat in DISABLED_FEATURES.split(",") if feat.strip() != ""])
ENABLED_FEATURES = [feat for feat in AVAILABLE_FEATURES if feat not in disabled]

import random
from collections import deque

# Use tensorflow.keras for our neural network (DQN)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam

# We will use gym for our custom environment
try:
    import gym
    from gym import spaces
except ImportError:
    print("The gym library is required. Install it via pip install gym", file=sys.stderr)
    sys.exit(1)


class TradingEnvSimple(gym.Env):
    """
    A multi-step trading environment that allows the agent to BUY, SELL, SHORT, or COVER
    across multiple steps. We track a single unit position:
      - self.position in {0=flat, +1=long, -1=short}
      - self.entry_price where we opened that position.

    The environment steps through historical DataFrame rows, giving the agent:
      - The enabled features for the current row.
      - The next row's close (as a "predicted" price).
      - (Optional) The current position if you want that in state.

    Actions:
       0 = BUY, 1 = SELL, 2 = SHORT, 3 = COVER

    Reward logic (mark-to-market each step):
      - If long, we gain or lose based on price movement from the previous step.
      - If short, we gain or lose inversely.
      - Closing or reversing a position realizes P/L immediately from entry_price to current_price.

    Episode terminates when we reach the end of the CSV data.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, data: pd.DataFrame, enabled_features: list):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.enabled_features = enabled_features

        # We'll include the next close as "predicted price"
        # Optionally, we could also include "position" in the observation:
        self.include_position = False  # Set to True if you want the agent to see self.position in the state

        self.num_features = len(enabled_features) + 1  # +1 for predicted price
        if self.include_position:
            self.num_features += 1  # +1 more for the position

        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_features,),
            dtype=np.float32
        )

        # Only 4 discrete actions are allowed now: BUY, SELL, SHORT, COVER
        self.action_space = spaces.Discrete(4)

        self.current_step = 0
        self.position = 0       # +1=long, -1=short, 0=flat
        self.entry_price = 0.0
        self.last_price = 0.0   # price at the last step to measure mark-to-market

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0

        if len(self.data) < 2:
            # Not enough data
            return np.zeros(self.num_features, dtype=np.float32)

        self.last_price = float(self.data.loc[self.current_step, 'close'])
        return self._get_observation()

    def _get_observation(self):
        if self.current_step >= len(self.data):
            return np.zeros(self.num_features, dtype=np.float32)

        row = self.data.loc[self.current_step]
        features = row[self.enabled_features].values.astype(np.float32)

        # "Predicted price" = next row's close if possible
        if self.current_step < len(self.data) - 1:
            predicted_price = float(self.data.loc[self.current_step + 1, 'close'])
        else:
            predicted_price = float(self.data.loc[self.current_step, 'close'])

        if self.include_position:
            obs = np.concatenate([features, [predicted_price, float(self.position)]])
        else:
            obs = np.concatenate([features, [predicted_price]])

        return obs

    def step(self, action):
        if self.current_step >= len(self.data) - 1:
            # No more steps
            return self._get_observation(), 0.0, True, {}

        current_close = float(self.data.loc[self.current_step, 'close'])
        next_close = float(self.data.loc[self.current_step + 1, 'close'])

        # Mark-to-market P/L from last_price to current_close
        step_reward = 0.0
        if self.position == 1:
            # Long: gain/loss from last_price to current_close
            step_reward = current_close - self.last_price
        elif self.position == -1:
            # Short: gain/loss from last_price down to current_close
            step_reward = self.last_price - current_close

        # Realized P/L from closing or reversing
        immediate_pnl = 0.0

        # 0 = BUY, 1 = SELL, 2 = SHORT, 3 = COVER
        if action == 0:  # BUY
            if self.position == 0:
                # Open new long
                self.position = 1
                self.entry_price = current_close
            elif self.position == -1:
                # Reverse from short to long
                immediate_pnl = (self.entry_price - current_close)
                self.position = 1
                self.entry_price = current_close
            # If already long, do nothing

        elif action == 1:  # SELL
            if self.position == 1:
                # Close long
                immediate_pnl = (current_close - self.entry_price)
                self.position = 0
                self.entry_price = 0.0

        elif action == 2:  # SHORT
            if self.position == 0:
                # Open new short
                self.position = -1
                self.entry_price = current_close
            elif self.position == 1:
                # Reverse from long to short
                immediate_pnl = (current_close - self.entry_price)
                self.position = -1
                self.entry_price = current_close

        elif action == 3:  # COVER
            if self.position == -1:
                # Close short
                immediate_pnl = (self.entry_price - current_close)
                self.position = 0
                self.entry_price = 0.0

        reward = step_reward + immediate_pnl

        self.current_step += 1
        done = (self.current_step >= len(self.data) - 1)

        # If we're done, close any open position for final P/L
        if done and self.position != 0:
            final_close = float(self.data.loc[self.current_step, 'close'])
            if self.position == 1:
                reward += (final_close - self.entry_price)
            elif self.position == -1:
                reward += (self.entry_price - final_close)
            self.position = 0
            self.entry_price = 0.0

        obs = self._get_observation()
        # Update last_price for next step's mark-to-market
        self.last_price = current_close

        return obs, reward, done, {}

    def render(self, mode='human'):
        if self.current_step < len(self.data):
            current_close = float(self.data.loc[self.current_step, 'close'])
            print(f"Step: {self.current_step}, Close Price: {current_close}, Position: {self.position}")
        else:
            print("End of data.")


class RLAgent:
    """
    A Deep Q-Network (DQN) agent using a simple feedforward neural network.
    It now picks from 4 actions: BUY=0, SELL=1, SHORT=2, COVER=3.
    """

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.batch_size = 32

        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(InputLayer(input_shape=(self.state_size,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Epsilon-greedy
        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)[0]
        return np.argmax(q_values)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, targets = [], []

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state[np.newaxis, :], verbose=0)[0])
            current_q = self.model.predict(state[np.newaxis, :], verbose=0)[0]
            current_q[action] = target
            states.append(state)
            targets.append(current_q)

        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


# Global RL agent & a flag if user said no to further training
global_rl_agent = None
user_declined_training = False


def train_rl_agent_from_csv(csv_filename: str, enabled_features: list):
    """
    Train (or partially train) the RL agent using the data in csv_filename.
    Now, we keep prompting after each chunk until:
       - total episodes is reached, or
       - user says 'n'.
    The meta file now saves both episodes completed and the agent's epsilon.
    """
    # Load data
    try:
        data = pd.read_csv(csv_filename)
    except Exception as e:
        raise FileNotFoundError(f"Could not read CSV file {csv_filename}: {e}")

    # Validate columns
    required_columns = set(enabled_features + ['close'])
    if not required_columns.issubset(set(data.columns)):
        missing_cols = required_columns - set(data.columns)
        raise ValueError(f"CSV file {csv_filename} is missing required columns: {missing_cols}")

    env = TradingEnvSimple(data, enabled_features)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Filenames
    base_name = os.path.splitext(os.path.basename(csv_filename))[0]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_filename = os.path.join(script_dir, f"trained_rl_agent_{base_name}.weights.h5")
    meta_filename = os.path.join(script_dir, f"trained_rl_agent_{base_name}.meta.json")

    # Make agent
    agent = RLAgent(state_size, action_size)

    # Load weights if present
    if os.path.exists(model_filename):
        try:
            agent.load(model_filename)
            print(f"Loaded existing model from {model_filename}")
        except Exception as e:
            print(f"Failed to load weights from {model_filename}: {e}")
            print("Proceeding to train a new model from scratch.")

    # Load meta (episodes completed and epsilon)
    if os.path.exists(meta_filename):
        with open(meta_filename, 'r') as f:
            meta_data = json.load(f)
    else:
        meta_data = {"episodes_completed": 0, "epsilon": agent.epsilon}

    episodes_completed = meta_data.get("episodes_completed", 0)
    # Restore epsilon if saved
    agent.epsilon = meta_data.get("epsilon", agent.epsilon)

    global user_declined_training
    while episodes_completed < TOTAL_EPISODES:
        episodes_left = TOTAL_EPISODES - episodes_completed
        print(f"You have {episodes_left} episodes left to reach {TOTAL_EPISODES}.")
        choice = input("Continue training this chunk? (y/n): ").strip().lower()
        if choice != 'y':
            print("User declined further training. Will output NONE in logic calls.")
            user_declined_training = True
            break

        # Train for up to CHUNK_SIZE episodes, or whatever is left
        episodes_to_train = min(CHUNK_SIZE, episodes_left)
        print(f"Training for {episodes_to_train} episodes...")

        for _ in range(episodes_to_train):
            state = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            agent.replay()
            episodes_completed += 1
            print(f"Episode {episodes_completed}/{TOTAL_EPISODES} - Reward: {total_reward:.2f} - Eps: {agent.epsilon:.2f}")

            # Save progress (including current epsilon)
            agent.save(model_filename)
            meta_data["episodes_completed"] = episodes_completed
            meta_data["epsilon"] = agent.epsilon
            with open(meta_filename, 'w') as f:
                json.dump(meta_data, f)

        print(f"Done training chunk (episodes completed: {episodes_completed}).")

    if episodes_completed >= TOTAL_EPISODES:
        print("Reached or exceeded TOTAL_EPISODES, training complete.")

    return agent


def get_csv_filename(ticker: str) -> str:
    """Construct CSV filename from ticker and timeframe."""
    real_timeframe = TIMEFRAME_MAPPING.get(BAR_TIMEFRAME, "H1")
    filename = f"{ticker}_{real_timeframe}.csv"
    return filename


def load_features_from_csv(csv_filename: str, enabled_features: list) -> pd.DataFrame:
    """Load CSV data, ensuring it has the needed columns."""
    data = pd.read_csv(csv_filename)
    required = set(enabled_features + ['close'])
    available = set(data.columns)
    missing = required - available
    if missing:
        raise ValueError(f"CSV file {csv_filename} is missing columns: {missing}")
    return data


def run_logic(current_price, predicted_price, ticker):
    """
    Live trading logic function.
    This function loads the relevant CSV data for the given ticker and timeframe,
    prepares the current state for the RL agent, uses the agent to decide on an action,
    and then calls the corresponding trade function from the forest API.

    The RL agent now only outputs BUY, SELL, SHORT, or COVER.
    The final decision will be overridden to NONE if it would duplicate an existing position.
    """
    # Import forest API trade functions
    from forest import api, buy_shares, sell_shares, short_shares, close_short

    # Get current position for the ticker
    # position_qty > 0 means we have a long (buy) position
    # position_qty < 0 means we have a short position
    position_qty = api.get_position(ticker)

    # Get account info for sizing
    account = api.get_account()
    cash = float(account.cash)
    shares_to_buy = int(cash // current_price)
    if shares_to_buy < 1:
        print("Not enough cash to buy 1 share.")
        return

    # Build CSV filename from ticker and timeframe
    csv_filename = get_csv_filename(ticker)

    # Load CSV data and extract enabled features (this also validates the CSV)
    try:
        data = load_features_from_csv(csv_filename, ENABLED_FEATURES)
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        return

    # If our global RL agent is not yet trained for this data, train it.
    global global_rl_agent
    if global_rl_agent is None:
        print("Training RL agent for live trading decision (this may take a moment)...")
        global_rl_agent = train_rl_agent_from_csv(csv_filename, ENABLED_FEATURES)

    # Prepare the current state for the RL agent (use the last row as proxy for current).
    current_row = data.iloc[-1]
    state_features = current_row[ENABLED_FEATURES].values.astype(np.float32)
    state = np.concatenate([state_features, [float(predicted_price)]])
    if state.shape[0] != global_rl_agent.state_size:
        print("State vector dimension mismatch.")
        return

    # Get action from RL agent (force greedy by setting epsilon=0)
    original_epsilon = global_rl_agent.epsilon
    global_rl_agent.epsilon = 0.0
    action = global_rl_agent.act(state)
    global_rl_agent.epsilon = original_epsilon

    # Map the action from the RL model
    action_map = {
        0: "BUY",
        1: "SELL",
        2: "SHORT",
        3: "COVER"
    }
    chosen_action = action_map.get(action, "BUY")  # default to BUY if something unexpected occurs

    # Prevent duplicate trades by overriding with NONE if necessary.
    if chosen_action == "BUY" and position_qty > 0:
        print(f"Already long ({position_qty} shares). Skipping BUY.")
        chosen_action = "NONE"
    elif chosen_action == "SELL" and position_qty <= 0:
        print("No long position to SELL. Skipping.")
        chosen_action = "NONE"
    elif chosen_action == "SHORT" and position_qty < 0:
        print(f"Already short ({position_qty} shares). Skipping SHORT.")
        chosen_action = "NONE"
    elif chosen_action == "COVER" and position_qty >= 0:
        print("No short position to COVER. Skipping.")
        chosen_action = "NONE"

    # Execute the trade if a valid action remains
    if chosen_action == "BUY":
        print(f"BUY {shares_to_buy} shares of {ticker} at {current_price}.")
        buy_shares(ticker, shares_to_buy, current_price, predicted_price)
    elif chosen_action == "SELL":
        print(f"SELL {position_qty} shares of {ticker}. Closing long.")
        sell_shares(ticker, position_qty, current_price, predicted_price)
    elif chosen_action == "SHORT":
        print(f"SHORT {shares_to_buy} shares of {ticker} at {current_price}.")
        short_shares(ticker, shares_to_buy, current_price, predicted_price)
    elif chosen_action == "COVER":
        print(f"COVER {abs(position_qty)} shares of {ticker}. Closing short.")
        close_short(ticker, abs(position_qty), current_price, predicted_price)
    else:
        print("No action taken.")

    return


def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    """
    Backtesting function.
    Uses historical data from the CSV corresponding to the first ticker in TICKERS and
    the given timeframe to simulate a trade decision.
    Returns one of the following action strings: "BUY", "SELL", "SHORT", "COVER", or "NONE".

    The RL agent only outputs BUY, SELL, SHORT, or COVER.
    If the agent's decision conflicts with the current position (e.g. BUY when already long),
    the final decision is set to NONE.
    """
    # Get the first ticker from TICKERS
    first_ticker = TICKERS.split(",")[0].strip()
    csv_filename = get_csv_filename(first_ticker)

    try:
        data = load_features_from_csv(csv_filename, ENABLED_FEATURES)
    except Exception as e:
        print(f"Error loading CSV data for backtest: {e}")
        return "NONE"

    global global_rl_agent
    if global_rl_agent is None:
        print("Training RL agent for backtest (this may take a moment)...")
        global_rl_agent = train_rl_agent_from_csv(csv_filename, ENABLED_FEATURES)

    current_row = data.iloc[-1]
    state_features = current_row[ENABLED_FEATURES].values.astype(np.float32)
    state = np.concatenate([state_features, [float(predicted_price)]])
    if state.shape[0] != global_rl_agent.state_size:
        print("State vector dimension mismatch in backtest.")
        return "NONE"

    # Force the agent to be greedy
    original_epsilon = global_rl_agent.epsilon
    global_rl_agent.epsilon = 0.0
    action = global_rl_agent.act(state)
    global_rl_agent.epsilon = original_epsilon

    action_map = {
        0: "BUY",
        1: "SELL",
        2: "SHORT",
        3: "COVER"
    }
    chosen_action = action_map.get(action, "BUY")

    # Prevent duplicate trades by overriding with NONE if necessary.
    if chosen_action == "BUY" and position_qty > 0:
        print(f"Already in a long position ({position_qty}). Skipping BUY.")
        chosen_action = "NONE"
    elif chosen_action == "SELL" and position_qty <= 0:
        print("No long position to SELL. Skipping.")
        chosen_action = "NONE"
    elif chosen_action == "SHORT" and position_qty < 0:
        print(f"Already in a short position ({position_qty}). Skipping SHORT.")
        chosen_action = "NONE"
    elif chosen_action == "COVER" and position_qty >= 0:
        print("No short position to COVER. Skipping.")
        chosen_action = "NONE"

    print(f"Backtest decision for {first_ticker}: {chosen_action}")
    return chosen_action
