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

###############################################################################
#                         ENVIRONMENT / HELPER FUNCTIONS                      #
###############################################################################

def convert_bar_timeframe(tf_str: str) -> str:
    mapping = {
        "4Hour": "H4",
        "2Hour": "H2",
        "1Hour": "H1",
        "30Min": "M30",
        "15Min": "M15",
    }
    return mapping.get(tf_str, "H1")  # default to "H1" if not found

def get_disabled_features() -> set:
    disabled = os.environ.get("DISABLED_FEATURES", "")
    disabled = disabled.strip()
    if not disabled:
        return set()
    return set(f.strip() for f in disabled.split(",") if f.strip())

def get_first_ticker() -> str:
    tickers_str = os.environ.get("TICKERS", "TSLA")
    return tickers_str.split(",")[0].strip()

def get_all_tickers() -> list:
    tickers_str = os.environ.get("TICKERS", "TSLA")
    return [t.strip() for t in tickers_str.split(",") if t.strip()]

def load_csv_data(ticker: str, timeframe: str) -> pd.DataFrame:
    filename = f"{ticker}_{timeframe}.csv"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"CSV file not found: {filename}")
    df = pd.read_csv(filename, parse_dates=True)
    return df

###############################################################################
#                 RANDOM FOREST FOR PRICE PREDICTION (Example)               #
###############################################################################

def train_random_forest(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X, y)
    return rf

def predict_random_forest(rf_model: RandomForestRegressor, X: pd.DataFrame) -> np.ndarray:
    return rf_model.predict(X)

###############################################################################
#                       DQN (Deep Q-Network) CLASSES                          #
###############################################################################

# Actions: 0=BUY, 1=SELL, 2=NONE
ACTION_LIST = ["BUY", "SELL", "NONE"]

class DQNNetwork(nn.Module):
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
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_size: int, action_size: int = 3, gamma=0.99, lr=1e-3, batch_size=32):
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
        
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(torch.argmax(q_values[0]).item())
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(1)[0]
        target = rewards_t + (1 - dones_t) * self.gamma * next_q_values
        
        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """
        Sync weights from policy_net to target_net.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

###############################################################################
#                      RL TRAINING ENVIRONMENT (SIMULATED)                    #
###############################################################################

class TradingEnvironment:
    def __init__(self, df: pd.DataFrame, initial_position=0, penalty_none=0.01):
        self.df = df.reset_index(drop=True)
        self.n_steps = len(df)
        self.current_step = 0

        if 'close' not in self.df.columns:
            raise ValueError("DataFrame must contain 'close' column for reward calculation.")

        self.penalty_none = penalty_none
        feature_cols = list(self.df.columns)
        for col in ['timestamp', 'timeframe', 'close']:
            if col in feature_cols:
                feature_cols.remove(col)
        self.feature_cols = feature_cols

        self.position = initial_position
        self.entry_price = None

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.entry_price = None
        return self._get_state()

    def step(self, action: int):
        current_close = self.df.loc[self.current_step, 'close']
        reward = 0.0
        done = False

        if action == 0:  # BUY
            if self.position == 0:
                self.position = 1
                self.entry_price = current_close
        elif action == 1:  # SELL
            if self.position == 1:
                reward += (current_close - self.entry_price)
                self.position = 0
                self.entry_price = None
        else:
            # NONE or invalid action
            reward -= self.penalty_none

        self.current_step += 1
        if self.current_step >= self.n_steps:
            done = True

        next_state = self._get_state()
        return next_state, reward, done

    def _get_state(self):
        if self.current_step >= self.n_steps:
            return np.zeros(len(self.feature_cols) + 1, dtype=np.float32)
        row = self.df.loc[self.current_step, self.feature_cols].values
        state = np.append(row, self.position)
        return state.astype(np.float32)

###############################################################################
#                 GLOBAL CACHES (RF MODELS & RL AGENTS)                       #
###############################################################################

_rf_models = {}
_dqn_agents = {}

def get_rf_model(ticker: str, timeframe: str, df_features: pd.DataFrame, df_close: pd.Series):
    key = (ticker, timeframe)
    if key not in _rf_models:
        rf_model = train_random_forest(df_features, df_close)
        _rf_models[key] = rf_model
    return _rf_models[key]

def get_dqn_agent(ticker: str, timeframe: str, state_size: int) -> DQNAgent:
    key = (ticker, timeframe)
    if key not in _dqn_agents:
        agent = DQNAgent(state_size=state_size, action_size=len(ACTION_LIST))
        _dqn_agents[key] = agent
    return _dqn_agents[key]

###############################################################################
#                        TRAINING ROUTINE FOR THE DQN                         #
###############################################################################

def train_dqn_on_data(ticker: str, timeframe: str, df: pd.DataFrame, n_episodes: int = 5, max_steps_per_episode: int = None):
    feature_cols = list(df.columns)
    for col in ['timestamp', 'timeframe', 'close']:
        if col in feature_cols:
            feature_cols.remove(col)
    state_size = len(feature_cols) + 1

    agent = get_dqn_agent(ticker, timeframe, state_size)
    env = TradingEnvironment(df=df, initial_position=0, penalty_none=0.02)

    for episode in range(n_episodes):
        state = env.reset()
        done = False
        step_count = 0
        agent.update_target_network()
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, float(done))
            agent.replay()
            state = next_state
            step_count += 1
            if max_steps_per_episode is not None and step_count >= max_steps_per_episode:
                break
        agent.update_target_network()

###############################################################################
#                             MAIN REQUIRED FUNCTIONS                         #
###############################################################################

def run_logic(current_price, predicted_price, ticker):
    from forest import api, buy_shares, sell_shares

    bar_timeframe = os.environ.get("BAR_TIMEFRAME", "4Hour")
    tf_suffix = convert_bar_timeframe(bar_timeframe)

    df_raw = load_csv_data(ticker, tf_suffix)
    disabled_feats = get_disabled_features()
    if 'close' not in df_raw.columns:
        raise ValueError("CSV missing 'close' column, required for RL environment.")
    for col in disabled_feats:
        if col in df_raw.columns:
            df_raw.drop(columns=[col], inplace=True)

    df_raw['predicted_price'] = df_raw['close']
    if len(df_raw) > 0:
        df_raw.loc[df_raw.index[-1], 'predicted_price'] = predicted_price

    train_dqn_on_data(ticker, tf_suffix, df_raw, n_episodes=2)

    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception:
        position_qty = 0.0
    account = api.get_account()
    cash = float(account.cash)

    feature_cols = list(df_raw.columns)
    for col in ['timestamp', 'timeframe', 'close']:
        if col in feature_cols:
            feature_cols.remove(col)
    last_row = df_raw.iloc[-1][feature_cols].values
    state = np.append(last_row, position_qty).astype(np.float32)

    agent = get_dqn_agent(ticker, tf_suffix, len(feature_cols) + 1)
    action_idx = agent.act(state)
    action_str = ACTION_LIST[action_idx]

    final_action = "NONE"
    if action_str == "BUY":
        if position_qty < 1:
            final_action = "BUY"
    elif action_str == "SELL":
        if position_qty > 0:
            final_action = "SELL"

    if final_action == "BUY":
        max_shares = int(cash // current_price)
        buy_shares(ticker, max_shares, current_price, predicted_price)
    elif final_action == "SELL":
        sell_shares(ticker, position_qty, current_price, predicted_price)
    else:
        pass  # NONE

def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    bar_timeframe = os.environ.get("BAR_TIMEFRAME", "4Hour")
    tf_suffix = convert_bar_timeframe(bar_timeframe)
    first_ticker = get_first_ticker()

    df_raw = load_csv_data(first_ticker, tf_suffix)
    disabled_feats = get_disabled_features()
    for col in disabled_feats:
        if col in df_raw.columns:
            df_raw.drop(columns=[col], inplace=True)

    df_raw['predicted_price'] = df_raw['close']
    if len(df_raw) > 0:
        df_raw.loc[df_raw.index[-1], 'predicted_price'] = predicted_price

    train_dqn_on_data(first_ticker, tf_suffix, df_raw, n_episodes=2)

    feature_cols = list(df_raw.columns)
    for col in ['timestamp', 'timeframe', 'close']:
        if col in feature_cols:
            feature_cols.remove(col)
    last_row = df_raw.iloc[-1][feature_cols].values
    state = np.append(last_row, position_qty).astype(np.float32)

    agent = get_dqn_agent(first_ticker, tf_suffix, len(feature_cols) + 1)
    action_idx = agent.act(state)
    action_str = ACTION_LIST[action_idx]

    final_action = "NONE"
    if action_str == "BUY":
        if position_qty < 1:
            final_action = "BUY"
    elif action_str == "SELL":
        if position_qty > 0:
            final_action = "SELL"

    return final_action
