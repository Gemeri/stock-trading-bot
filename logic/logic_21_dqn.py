# logic.py
"""
A demonstration of a DQN-based Reinforcement Learning approach, using a Random Forest
for price prediction as an input feature. This script shows how one might structure
an actively trading RL system in Python, training on CSV data and filtering out
disabled features. The core methods run_logic() and run_backtest() are provided at
the bottom, as required.

REQUIREMENTS (typical):
    pip install numpy pandas scikit-learn torch  # or tensorflow
    # Possibly others for advanced usage (matplotlib, TA-Lib, etc.)
    
ENVIRONMENT VARIABLES (example):
    BAR_TIMEFRAME=4Hour
    TICKERS=TSLA,AAPL
    DISABLED_FEATURES=body_size,candle_rise,wick_to_body

USAGE:
    - The environment variables control CSV filename suffix (via BAR_TIMEFRAME),
      which tickers to trade, and which features to disable.
    - run_logic(...) is called in live trading, uses real-time position info from 'forest.api'.
    - run_backtest(...) is called in a backtesting engine with the specified position_qty.

Core flow:
    1) Load the CSV for the relevant ticker (or first ticker in backtest).
    2) Filter out disabled features from the CSV columns, leaving only enabled ones.
    3) Train or load a Random Forest model to produce a predicted price (if desired).
    4) Train or load the RL (DQN) model using the CSV data. The agent's state includes
       the features + predicted_price + position quantity, and the action space is:
         [0: BUY, 1: SELL, 2: SHORT, 3: COVER, 4: NONE].
    5) The RL agent chooses an action that (ideally) maximizes profit. We penalize
       "NONE" actions to encourage more frequent trading, but also we do not allow
       redundant trades (BUY if already long, etc.).
    6) In run_logic, actions are dispatched to 'forest' module functions (buy_shares, sell_shares, etc.).
    7) In run_backtest, simply return the chosen action as a string: "BUY", "SELL", "SHORT", "COVER", or "NONE".

Notes on DQN specifics:
    - This code uses a minimal DQN approach (PyTorch). For real usage, you may want more
      advanced techniques (double DQN, dueling networks, PER, etc.). The below code
      trains on the loaded CSV each time for demonstration. In practice, you'd likely
      train offline or keep a persistent agent that updates incrementally.

"""

import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

# For the RL (DQN) framework, we'll use PyTorch in this example.
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging

# Placeholder for the real forest trading API and order functions. 
# The user environment should provide these. In backtest, we won't actually call them.
# Replace with actual imports in real usage:
# from forest import api, buy_shares, sell_shares, short_shares, close_short

###############################################################################
#                         ENVIRONMENT / HELPER FUNCTIONS                      #
###############################################################################

def convert_bar_timeframe(tf_str: str) -> str:
    """
    Convert a timeframe string like "4Hour", "2Hour", "1Hour", "30Min", "15Min"
    into the short suffix used in the CSV filename ("H4", "H2", "H1", "M30", "M15").
    """
    # Basic mapping, can be extended if needed
    mapping = {
        "4Hour": "H4",
        "2Hour": "H2",
        "1Hour": "H1",
        "30Min": "M30",
        "15Min": "M15",
    }
    return mapping.get(tf_str, "H1")  # default to "H1" if not found

def get_disabled_features() -> set:
    """
    Parse DISABLED_FEATURES from environment (comma-separated) 
    into a Python set for easy membership checks.
    Example: DISABLED_FEATURES="body_size,candle_rise" => {'body_size','candle_rise'}
    """
    disabled = os.environ.get("DISABLED_FEATURES", "")
    disabled = disabled.strip()
    if not disabled:
        return set()
    return set([f.strip() for f in disabled.split(",") if f.strip()])

def get_first_ticker() -> str:
    """
    Returns the first ticker from TICKERS environment variable.
    Example: TICKERS="TSLA,AAPL" => "TSLA"
    """
    tickers_str = os.environ.get("TICKERS", "TSLA")
    return tickers_str.split(",")[0].strip()

def get_all_tickers() -> list:
    """
    Returns the full list of tickers from TICKERS environment variable.
    Example: TICKERS="TSLA,AAPL" => ["TSLA", "AAPL"]
    """
    tickers_str = os.environ.get("TICKERS", "TSLA")
    return [t.strip() for t in tickers_str.split(",") if t.strip()]

def load_csv_data(ticker: str, timeframe: str) -> pd.DataFrame:
    """
    Loads the CSV file for the given ticker + timeframe suffix.
    Example filename: "TSLA_H4.csv"
    """
    # e.g. ticker="TSLA", timeframe="H4" => "TSLA_H4.csv"
    filename = f"{ticker}_{timeframe}.csv"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"CSV file not found: {filename}")
    
    df = pd.read_csv(filename, parse_dates=True)
    return df

###############################################################################
#                 RANDOM FOREST FOR PRICE PREDICTION (Example)               #
###############################################################################

def train_random_forest(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    """
    Trains a Random Forest Regressor to predict future price (or next close).
    Returns the trained model.
    """
    rf = RandomForestRegressor(
        n_estimators=50,
        max_depth=5,
        random_state=42
    )
    rf.fit(X, y)
    return rf

def predict_random_forest(rf_model: RandomForestRegressor, X: pd.DataFrame) -> np.ndarray:
    """
    Uses the trained Random Forest Regressor to predict on the given feature set X.
    Returns a numpy array of predictions.
    """
    return rf_model.predict(X)

###############################################################################
#                       DQN (Deep Q-Network) CLASSES                          #
###############################################################################

# Actions: 0=BUY, 1=SELL, 2=SHORT, 3=COVER, 4=NONE
ACTION_LIST = ["BUY", "SELL", "SHORT", "COVER", "NONE"]

class DQNNetwork(nn.Module):
    """
    A simple fully connected neural network for DQN. 
    Adjust layer sizes as needed for your data dimension.
    """
    def __init__(self, state_size: int, action_size: int, hidden_size=64):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    """
    A basic replay buffer for DQN. 
    For on-policy style, you might adapt this to store only current episode data.
    """
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        import random
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = map(
            np.array, zip(*batch)
        )
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    """
    A simple DQN Agent with experience replay. 
    Uses an epsilon-greedy policy for exploration.
    """
    def __init__(
        self, 
        state_size: int, 
        action_size: int = 5, 
        gamma=0.99,
        lr=1e-3,
        batch_size=32
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(10000)
        
        self.epsilon = 1.0  # start of exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def act(self, state):
        """
        Epsilon-greedy action selection.
        state: numpy array, shape=(state_size,)
        """
        if np.random.rand() <= self.epsilon:
            # random action
            return np.random.randint(self.action_size)
        
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(torch.argmax(q_values[0]).item())
    
    def replay(self):
        """
        Train the network on a batch of experiences from memory.
        """
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Get current Q estimates
        q_values = self.policy_net(states_t)
        # Gather the Q values for the chosen actions
        q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Get next Q values from target net
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(1)[0]

        # If done, next_q_values = 0
        target = rewards_t + (1 - dones_t) * self.gamma * next_q_values
        
        loss = F.mse_loss(q_values, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """
        Periodically sync weights from policy_net to target_net.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

###############################################################################
#                      RL TRAINING ENVIRONMENT (SIMULATED)                    #
###############################################################################

class TradingEnvironment:
    """
    A simple environment to simulate trading through time steps in the CSV.
    State includes:
       - The selected features (enabled ones),
       - The predicted_price (from RF, or externally),
       - The current position quantity (discretized or continuous).
    Actions:
       0: BUY
       1: SELL
       2: SHORT
       3: COVER
       4: NONE

    Reward:
       - Change in unrealized or realized PnL,
       - Potential penalty for taking 'NONE' action,
       - Additional risk-based or Sharpe-based measure, if desired.

    This environment is used for training the DQN offline on historical data.
    """
    def __init__(self, df: pd.DataFrame, initial_position=0, penalty_none=0.01):
        """
        df: DataFrame containing columns for features + 'predicted_price' + 'close' (for PnL).
        initial_position: Starting position for the agent.
        penalty_none: Penalty factor for choosing 'NONE' action to encourage activity.
        """
        self.df = df.reset_index(drop=True)
        self.n_steps = len(df)
        logging.info("Steps: " + self.n_steps)
        self.current_step = 0

        # We'll store the 'close' price for PnL calculations
        # (Assuming 'close' is definitely in the dataset.)
        if 'close' not in self.df.columns:
            raise ValueError("DataFrame must contain 'close' column for reward calculation.")

        self.penalty_none = penalty_none
        
        # Identify feature columns (everything except 'close' if not disabled, plus 'predicted_price')
        # We'll assume these were already included in df, including 'predicted_price'.
        feature_cols = list(self.df.columns)
        # remove non-feature columns if they exist
        for col in ['timestamp', 'timeframe', 'close']:
            if col in feature_cols:
                feature_cols.remove(col)
        
        self.feature_cols = feature_cols

        # We'll keep track of position quantity as an integer (long=+n, short=-n).
        self.position = initial_position
        # We'll track an approximate PnL (unrealized) for reward calc
        # For training, let's assume 1 share for simplicity. 
        self.entry_price = None  # price at which position was opened, for PnL calc

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.entry_price = None
        return self._get_state()

    def step(self, action: int):
        """
        action in [0..4]: 0=BUY,1=SELL,2=SHORT,3=COVER,4=NONE
        We'll hold 1 share or -1 share at a time (no multi-lot).
        
        Returns (next_state, reward, done).
        """
        current_close = self.df.loc[self.current_step, 'close']
        
        reward = 0.0
        done = False
        
        # Realistic position transitions:
        #   - If position=0 and action=BUY => position=+1, entry_price=current_close
        #   - If position=+1 and action=SELL => position=0, realized PnL = (current_close - entry_price)
        #   - If position=0 and action=SHORT => position=-1, entry_price=current_close
        #   - If position=-1 and action=COVER => position=0, realized PnL = (entry_price - current_close)
        #   - Action=NONE => do nothing (penalty for inactivity).
        #   - Any "redundant" action does nothing. (like BUY if already +1)
        
        if action == 0:  # BUY
            if self.position == 0:
                # open long
                self.position = 1
                self.entry_price = current_close
            # else: do nothing (redundant)
        
        elif action == 1:  # SELL
            if self.position == 1:
                # close long
                realized_pnl = (current_close - self.entry_price)
                reward += realized_pnl
                self.position = 0
                self.entry_price = None
            # else: do nothing (redundant)
        
        elif action == 2:  # SHORT
            if self.position == 0:
                # open short
                self.position = -1
                self.entry_price = current_close
            # else: do nothing (redundant)
        
        elif action == 3:  # COVER
            if self.position == -1:
                # close short
                realized_pnl = (self.entry_price - current_close)
                reward += realized_pnl
                self.position = 0
                self.entry_price = None
            # else: do nothing
        
        elif action == 4:  # NONE
            # penalize inactivity
            reward -= self.penalty_none
        
        # Optional: add a small unrealized PnL to reward to encourage holding winning trades
        # if self.position == 1:
        #     reward += (current_close - self.entry_price) * 0.01
        # elif self.position == -1:
        #     reward += (self.entry_price - current_close) * 0.01

        # Move to next step
        self.current_step += 1
        if self.current_step >= self.n_steps:
            done = True

        next_state = self._get_state()
        return next_state, reward, done

    def _get_state(self):
        """
        Returns the current feature vector + position as a 1D numpy array.
        """
        if self.current_step >= self.n_steps:
            # If we're beyond the data, return zeros or something consistent
            return np.zeros(len(self.feature_cols) + 1, dtype=np.float32)
        
        row = self.df.loc[self.current_step, self.feature_cols].values
        # Concatenate position as an additional feature
        state = np.append(row, self.position)
        return state.astype(np.float32)

###############################################################################
#                 GLOBAL CACHES (RF MODELS & RL AGENTS)                       #
###############################################################################

# So we don't retrain from scratch on every single call, 
# let's cache the models in a dict keyed by (ticker, timeframe).
_rf_models = {}
_dqn_agents = {}

def get_rf_model(ticker: str, timeframe: str, df_features: pd.DataFrame, df_close: pd.Series):
    """
    Retrieve or train (and cache) the Random Forest model for (ticker, timeframe).
    """
    key = (ticker, timeframe)
    if key not in _rf_models:
        # Train a new model
        rf_model = train_random_forest(df_features, df_close)
        _rf_models[key] = rf_model
    else:
        rf_model = _rf_models[key]
    return rf_model

def get_dqn_agent(ticker: str, timeframe: str, state_size: int) -> DQNAgent:
    """
    Retrieve or create a DQN agent for (ticker, timeframe), sized to the state_size.
    """
    key = (ticker, timeframe)
    if key not in _dqn_agents:
        # Create a new agent
        agent = DQNAgent(state_size=state_size, action_size=len(ACTION_LIST))
        _dqn_agents[key] = agent
    else:
        agent = _dqn_agents[key]
    return agent

###############################################################################
#                        TRAINING ROUTINE FOR THE DQN                         #
###############################################################################

def train_dqn_on_data(
    ticker: str, 
    timeframe: str,
    df: pd.DataFrame,
    n_episodes: int = 5, 
    max_steps_per_episode: int = None
):
    """
    Train the DQN agent on the historical data from the given DataFrame.
    
    n_episodes: how many passes through the data to train.
    max_steps_per_episode: optional cap on how many steps we run within each episode.

    This is a simplistic approach: we treat each pass over the entire dataset
    as one "episode" or multiple episodes. For real usage, you might shuffle
    or do a rolling window, etc.
    """
    # Identify the feature columns for the RL state. 
    # We'll assume 'close' is not in the feature set (and 'timestamp', 'timeframe' are removed).
    # By this point, 'df' should already have 'predicted_price' if we want it.
    # We'll just confirm we have it. If not, we proceed anyway.

    # The environment itself will figure out which columns are used, but let's note them:
    feature_cols = list(df.columns)
    # Remove columns we don't want as RL input
    for col in ['timestamp', 'timeframe', 'close']:
        if col in feature_cols:
            feature_cols.remove(col)
    
    # State size = len(feature_cols) + 1 for position
    state_size = len(feature_cols) + 1

    # Get or create the agent
    agent = get_dqn_agent(ticker, timeframe, state_size)

    # Create environment for training
    env = TradingEnvironment(df=df, initial_position=0, penalty_none=0.02)

    for episode in range(n_episodes):
        state = env.reset()
        done = False
        step_count = 0
        agent.update_target_network()  # Optionally sync target net each episode start

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, float(done))
            agent.replay()

            state = next_state
            step_count += 1
            if max_steps_per_episode is not None and step_count >= max_steps_per_episode:
                break
        
        # End of episode => sync target net
        agent.update_target_network()

###############################################################################
#                             MAIN REQUIRED FUNCTIONS                         #
###############################################################################

def run_logic(current_price, predicted_price, ticker):
    """
    Called in live trading. 
    1) Load CSV for the given ticker + timeframe from .env
    2) Filter out disabled columns
    3) (Optionally) train or retrieve the RF model for 'predicted_price' 
       -- though we also have predicted_price as a parameter.
    4) Train or retrieve the DQN model on the entire CSV
    5) Get current position from api
    6) Prepare the current state, let DQN pick an action
    7) If action is valid (no duplicate trades), execute it
    """

    # forest placeholders
    from forest import api, buy_shares, sell_shares, short_shares, close_short

    # 1) Get timeframe from .env and convert
    bar_timeframe = os.environ.get("BAR_TIMEFRAME", "4Hour")
    tf_suffix = convert_bar_timeframe(bar_timeframe)

    # 2) Load CSV
    df_raw = load_csv_data(ticker, tf_suffix)
    disabled_feats = get_disabled_features()

    # Ensure we have 'close' in the df for reward calculation
    if 'close' not in df_raw.columns:
        raise ValueError("CSV missing 'close' column, required for RL environment.")

    # 3) Filter out disabled features
    # We'll drop them if present
    for col in disabled_feats:
        if col in df_raw.columns:
            df_raw.drop(columns=[col], inplace=True)

    # 4) Optionally add the external predicted_price as a new column 'predicted_price' 
    # For demonstration, we do a simple approach: replicate the predicted_price in a new column 
    # for all rows (the historical rows won't matter much for training).
    df_raw['predicted_price'] = df_raw['close']  # default fallback
    # We won't re-train RF in run_logic for speed, but you could if you want.
    # Just override the last row with the actual predicted_price param
    if len(df_raw) > 0:
        df_raw.loc[df_raw.index[-1], 'predicted_price'] = predicted_price

    # 5) Train the DQN with historical data
    train_dqn_on_data(ticker, tf_suffix, df_raw, n_episodes=2)

    # 6) Obtain current position
    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception as e:
        print(f"No open position for {ticker}: {e}")
        position_qty = 0.0

    account = api.get_account()
    cash = float(account.cash)


    # 7) Build the current state for the DQN to pick an action
    # We'll replicate what the environment does. We take the last row's feature data + position
    # We must match the same columns the environment uses:
    feature_cols = list(df_raw.columns)
    for col in ['timestamp', 'timeframe', 'close']:
        if col in feature_cols:
            feature_cols.remove(col)

    # In case we are at the last row
    last_row = df_raw.iloc[-1][feature_cols].values
    # Append position_qty (though the training environment uses a single share approach. 
    # We'll just feed the current position as is, but you might want to clip or scale.)
    state = np.append(last_row, position_qty).astype(np.float32)

    # 8) DQN chooses an action
    state_size = len(feature_cols) + 1
    agent = get_dqn_agent(ticker, tf_suffix, state_size)
    action_idx = agent.act(state)
    action_str = ACTION_LIST[action_idx]

    # 9) Check for duplicate trades or invalid actions
    # If we already have a long position, skip BUY
    # If we already have a long position but action is SELL => valid => close it
    # If we already have a short position, skip SHORT
    # If no position, skip SELL or COVER
    final_action = "NONE"

    if action_str == "BUY":
        if position_qty >= 1:
            final_action = "NONE"  # redundant
        else:
            final_action = "BUY"
    elif action_str == "SELL":
        if position_qty <= 0:
            final_action = "NONE"  # no long to sell
        else:
            final_action = "SELL"
    elif action_str == "SHORT":
        if position_qty <= -1:
            final_action = "NONE"  # redundant
        else:
            final_action = "SHORT"
    elif action_str == "COVER":
        if position_qty >= 0:
            final_action = "NONE"  # no short to cover
        else:
            final_action = "COVER"
    else:
        final_action = "NONE"

    # 10) Execute the final_action
    if final_action == "BUY":
        max_shares = int(cash // current_price)
        print("buy")
        buy_shares(ticker, max_shares, current_price, predicted_price)
    elif final_action == "SELL":
        print("sell")
        sell_shares(ticker, position_qty, current_price, predicted_price)
    elif final_action == "SHORT":
        max_shares = int(cash // current_price)
        print("short")
        short_shares(ticker, max_shares, current_price, predicted_price)
    elif final_action == "COVER":
        qty_to_close = abs(position_qty)
        print("cover")
        close_short(ticker, qty_to_close, current_price)
    else:
        pass  # "NONE"

def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    """
    Called in a backtesting environment with the current price, predicted price,
    and the existing position quantity.
    1) Load CSV for the FIRST ticker from .env + timeframe
    2) Filter out disabled columns
    3) Train or retrieve the RL + RF model (consistent with run_logic)
    4) Return one of "BUY", "SELL", "SHORT", "COVER", or "NONE"
    """

    # 1) Get timeframe and the first ticker
    bar_timeframe = os.environ.get("BAR_TIMEFRAME", "4Hour")
    tf_suffix = convert_bar_timeframe(bar_timeframe)

    first_ticker = get_first_ticker()

    df_raw = load_csv_data(first_ticker, tf_suffix)

    # 2) Remove disabled features
    disabled_feats = get_disabled_features()
    for col in disabled_feats:
        if col in df_raw.columns:
            df_raw.drop(columns=[col], inplace=True)

    # 3) Add predicted_price column
    df_raw['predicted_price'] = df_raw['close']  # fallback
    if len(df_raw) > 0:
        df_raw.loc[df_raw.index[-1], 'predicted_price'] = predicted_price

    # Train the DQN
    train_dqn_on_data(first_ticker, tf_suffix, df_raw, n_episodes=2)

    # Build the current state
    feature_cols = list(df_raw.columns)
    for col in ['timestamp', 'timeframe', 'close']:
        if col in feature_cols:
            feature_cols.remove(col)

    last_row = df_raw.iloc[-1][feature_cols].values
    state = np.append(last_row, position_qty).astype(np.float32)

    state_size = len(feature_cols) + 1
    agent = get_dqn_agent(first_ticker, tf_suffix, state_size)
    action_idx = agent.act(state)
    action_str = ACTION_LIST[action_idx]

    # Check for duplicate or invalid actions
    final_action = "NONE"

    if action_str == "BUY":
        if position_qty >= 1:
            final_action = "NONE"
        else:
            final_action = "BUY"
    elif action_str == "SELL":
        if position_qty <= 0:
            final_action = "NONE"
        else:
            final_action = "SELL"
    elif action_str == "SHORT":
        if position_qty <= -1:
            final_action = "NONE"
        else:
            final_action = "SHORT"
    elif action_str == "COVER":
        if position_qty >= 0:
            final_action = "NONE"
        else:
            final_action = "COVER"
    else:
        final_action = "NONE"

    return final_action
