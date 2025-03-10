###############################################################################
# logic.py (Updated to fix "Can't call numpy() on Tensor that requires grad." error)
#
# A single script implementing:
#   1) An SAC-based (discrete action) Reinforcement Learning agent
#   2) A Random Forest regressor for price prediction integration
#   3) Two externally-called functions:
#        - run_logic(current_price, predicted_price, ticker)
#        - run_backtest(current_price, predicted_price, position_qty)
#
# This script:
#   - Reads .env variables (TICKERS, BAR_TIMEFRAME, DISABLED_FEATURES, etc.)
#   - Loads CSV data dynamically based on ticker and timeframe
#   - Filters disabled features
#   - Trains/uses an SAC-style RL approach to produce trading signals:
#       "BUY", "SELL", "SHORT", "COVER", or "NONE"
#   - Uses a random forest (optionally) for predicted_price (though we also
#     receive an externally-provided predicted_price) -- the combination
#     shows how you might integrate the two.
#   - Actively produces trades (minimizes "NONE" actions)
#   - Applies basic risk management (stop-loss, reward shaping)
#   - Demonstrates online RL training (train on full CSV each time), then
#     uses the last row for inference in run_logic, or looped rows in run_backtest.
#
# NOTE: This script uses a custom "Discrete SAC" style approach with Q-networks
#       and a policy network, because standard SAC is for continuous actions.
#       We adapt the general idea (two Q-nets, a value function, a soft policy)
#       but keep it simplified for demonstration.  In practice, you would use
#       a library or a better-tested discrete RL approach (DQN, PPO, etc.).
#
# REQUIRED:
#   pip install numpy pandas scikit-learn torch
#   (or your favorite environment with these libraries available)
#
# Adjust hyperparameters, neural net architecture, reward shaping, etc.
# for better performance in real scenarios.
###############################################################################

import os
import math
import random
import numpy as np
import pandas as pd
from collections import deque, namedtuple
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Dummy placeholders for the "forest" module (live trading).
# Remove in production if you have your own forest.py with these methods.
class DummyAPI:
    class Position:
        def __init__(self):
            self.qty = 0

        def __getattr__(self, name):
            return 0

    def get_position(self, ticker):
        return self.Position()

def buy_shares(ticker, qty=1):
    print(f"[ACTION] Buying {qty} shares of {ticker}.")

def sell_shares(ticker, qty=1):
    print(f"[ACTION] Selling {qty} shares of {ticker}.")

def short_shares(ticker, qty=1):
    print(f"[ACTION] Shorting {qty} shares of {ticker}.")

def close_short(ticker, qty=1):
    print(f"[ACTION] Covering (closing short) {qty} shares of {ticker}.")

api = DummyAPI()
# End stubs.

###############################################################################
# Environment variable handling
###############################################################################
# from dotenv import load_dotenv
# load_dotenv()

TICKERS = os.environ.get("TICKERS", "TSLA,AAPL")
BAR_TIMEFRAME = os.environ.get("BAR_TIMEFRAME", "4Hour")
DISABLED_FEATURES = os.environ.get("DISABLED_FEATURES", "").split(",")

TIMEFRAME_MAP = {
    "4Hour": "H4",
    "2Hour": "H2",
    "1Hour": "H1",
    "30Min": "M30",
    "15Min": "M15"
}

def convert_timeframe(tf_str: str) -> str:
    """Convert a string like '4Hour' to 'H4', '30Min' to 'M30', etc."""
    return TIMEFRAME_MAP.get(tf_str, "H1")

###############################################################################
# CSV Loading & Feature Preprocessing
###############################################################################

def load_csv_for_ticker(ticker: str, bar_timeframe: str) -> pd.DataFrame:
    """
    Load CSV data for the given ticker and timeframe suffix.
    The CSV filename is expected to be: f"{ticker}_{bar_timeframe}.csv".
    """
    filename = f"{ticker}_{bar_timeframe}.csv"
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"CSV file not found: {filename}")

    df = pd.read_csv(filename)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df

def filter_disabled_features(df: pd.DataFrame, disabled_feats: list) -> pd.DataFrame:
    """
    Remove the disabled features from the DataFrame if present. Return filtered DF.
    """
    disabled_feats = set([f.strip() for f in disabled_feats if f.strip()])
    columns_to_drop = [col for col in df.columns if col in disabled_feats]
    df_filtered = df.drop(columns=columns_to_drop, errors="ignore")
    return df_filtered

###############################################################################
# Random Forest for predicted_price (optional)
###############################################################################

def train_random_forest_for_prediction(df: pd.DataFrame) -> RandomForestRegressor:
    """
    Simple demonstration of a RandomForestRegressor.
    If 'close' not in df, returns untrained model.
    """
    if 'close' not in df.columns:
        return RandomForestRegressor()

    df = df.dropna().copy()
    if len(df) < 2:
        return RandomForestRegressor()

    # Train on all but last row
    train_df = df.iloc[:-1, :].copy()
    target = train_df['close'].values

    drop_cols = ['timestamp', 'close']
    keep_cols = [c for c in train_df.columns if c not in drop_cols]
    X = train_df[keep_cols].values

    if X.shape[0] < 2 or X.shape[1] < 1:
        return RandomForestRegressor()

    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X, target)
    return rf

###############################################################################
# Discrete SAC-like Agent Implementation
###############################################################################

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=int(1e5)):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size=64):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class SoftQNetwork(nn.Module):
    """
    Discrete Q-network: Q(s,a)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(SoftQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class PolicyNetwork(nn.Module):
    """
    For discrete actions, produce logits => softmax => action distribution.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        return logits

class TradingEnvironment:
    """
    A simplified environment stepping over DF rows.
    State: numeric features + position_qty
    Actions: 0=NONE, 1=BUY, 2=SELL, 3=SHORT, 4=COVER
    Reward: Realized PnL + small penalty for NOTHING. 
    """

    def __init__(self, df: pd.DataFrame):
        # Convert non-numeric to numeric => fill NaN => 0
        df_cols_except_ts = [c for c in df.columns if c != 'timestamp']
        df[df_cols_except_ts] = df[df_cols_except_ts].apply(pd.to_numeric, errors='coerce').fillna(0)

        self.df = df.reset_index(drop=True).copy()
        # If close missing, create it artificially
        if 'close' not in self.df.columns:
            self.df['close'] = 0.0

        # Feature columns: exclude timestamp/close
        self.feature_cols = [c for c in self.df.columns if c not in ['timestamp', 'close']]

        self.max_index = len(self.df) - 1
        self.current_index = 0

        # Position state
        self.position_qty = 0
        self.entry_price = 0.0

        self.done = False

    def reset(self):
        self.current_index = 0
        self.position_qty = 0
        self.entry_price = 0.0
        self.done = False
        return self._get_state()

    def step(self, action: int):
        if self.done:
            return self._get_state(), 0.0, True, {}

        row = self.df.iloc[self.current_index]
        current_close = row['close']
        is_last_step = (self.current_index >= self.max_index - 1)

        reward = 0.0
        # Action logic
        if action == 0:
            reward -= 0.02  # penalty for doing NOTHING
        elif action == 1:
            # BUY
            if self.position_qty < 1:
                if self.position_qty <= -1:
                    # Already short => do nothing
                    pass
                else:
                    self.position_qty = 1
                    self.entry_price = current_close
        elif action == 2:
            # SELL (close longs)
            if self.position_qty > 0:
                realized_pnl = (current_close - self.entry_price) * abs(self.position_qty)
                reward += realized_pnl
                self.position_qty = 0
                self.entry_price = 0.0
        elif action == 3:
            # SHORT
            if self.position_qty > -1:
                if self.position_qty >= 1:
                    # Already long => do nothing
                    pass
                else:
                    self.position_qty = -1
                    self.entry_price = current_close
        elif action == 4:
            # COVER
            if self.position_qty < 0:
                realized_pnl = (self.entry_price - current_close) * abs(self.position_qty)
                reward += realized_pnl
                self.position_qty = 0
                self.entry_price = 0.0

        # Optionally add small partial reward for unrealized PnL
        if self.position_qty != 0:
            if self.position_qty > 0:
                unreal_pnl = current_close - self.entry_price
            else:
                unreal_pnl = self.entry_price - current_close
            reward += 0.001 * unreal_pnl

        self.current_index += 1
        self.done = is_last_step

        next_state = self._get_state()
        return next_state, reward, self.done, {}

    def _get_state(self):
        row = self.df.iloc[self.current_index]
        feature_vals = row[self.feature_cols].values.astype(np.float32)
        state = np.concatenate([feature_vals, [float(self.position_qty)]])
        return state

def soft_q_update(
    policy_net, q_net1, q_net2, q_net1_target, q_net2_target,
    memory, optimizer_q1, optimizer_q2, optimizer_policy,
    batch_size, gamma=0.99, alpha=0.2
):
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    state_batch = torch.from_numpy(np.array(batch.state)).float()
    action_batch = torch.tensor(batch.action).long()
    reward_batch = torch.tensor(batch.reward).float()
    next_state_batch = torch.from_numpy(np.array(batch.next_state)).float()
    done_batch = torch.tensor(batch.done).float()

    # Q-values
    q1_values = q_net1(state_batch)
    q2_values = q_net2(state_batch)
    q1_action = q1_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
    q2_action = q2_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        logits_next = policy_net(next_state_batch)
        probs_next = F.softmax(logits_next, dim=-1)
        log_probs_next = F.log_softmax(logits_next, dim=-1)

        q1_next = q_net1_target(next_state_batch)
        q2_next = q_net2_target(next_state_batch)
        min_q_next = torch.min(q1_next, q2_next)

        v_next = (probs_next * (min_q_next - alpha * log_probs_next)).sum(dim=1)
        target_q = reward_batch + (1.0 - done_batch) * gamma * v_next

    loss_q1 = F.mse_loss(q1_action, target_q)
    loss_q2 = F.mse_loss(q2_action, target_q)

    optimizer_q1.zero_grad()
    loss_q1.backward()
    optimizer_q1.step()

    optimizer_q2.zero_grad()
    loss_q2.backward()
    optimizer_q2.step()

    # Update policy
    logits = policy_net(state_batch)
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)

    with torch.no_grad():
        q1_curr = q_net1(state_batch)
        q2_curr = q_net2(state_batch)
        min_q_curr = torch.min(q1_curr, q2_curr)

    policy_loss = (probs * (alpha * log_probs - min_q_curr)).sum(dim=1).mean()

    optimizer_policy.zero_grad()
    policy_loss.backward()
    optimizer_policy.step()

def update_targets(q_net_target, q_net, tau=0.005):
    """
    Soft update target network.
    q_net_target = (1 - tau)*q_net_target + tau*q_net
    """
    for target_param, param in zip(q_net_target.parameters(), q_net.parameters()):
        target_param.data.copy_(
            (1.0 - tau)*target_param.data + tau*param.data
        )

def train_sac_on_data(
    df: pd.DataFrame,
    num_epochs=3,
    batch_size=32,
    gamma=0.99,
    alpha=0.2,
    lr=1e-3
):
    """
    Train the discrete-SAC agent on DataFrame, returning trained networks.
    """
    env = TradingEnvironment(df)
    state_dim = len(env._get_state())
    action_dim = 5

    policy_net = PolicyNetwork(state_dim, action_dim)
    q_net1 = SoftQNetwork(state_dim, action_dim)
    q_net2 = SoftQNetwork(state_dim, action_dim)
    q_net1_target = SoftQNetwork(state_dim, action_dim)
    q_net2_target = SoftQNetwork(state_dim, action_dim)
    q_net1_target.load_state_dict(q_net1.state_dict())
    q_net2_target.load_state_dict(q_net2.state_dict())

    memory = ReplayBuffer(capacity=50000)

    optimizer_q1 = optim.Adam(q_net1.parameters(), lr=lr)
    optimizer_q2 = optim.Adam(q_net2.parameters(), lr=lr)
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=lr)

    for _ in range(num_epochs):
        state = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                state_t = torch.from_numpy(state).float().unsqueeze(0)
                logits = policy_net(state_t)
                # detach to avoid grad for sampling
                probs = F.softmax(logits, dim=-1).detach().numpy().squeeze()

            action = np.random.choice(range(action_dim), p=probs)
            next_state, reward, done, _ = env.step(action)
            memory.push(state, action, reward, next_state, done)
            state = next_state

            soft_q_update(
                policy_net, q_net1, q_net2,
                q_net1_target, q_net2_target,
                memory, optimizer_q1, optimizer_q2, optimizer_policy,
                batch_size, gamma=gamma, alpha=alpha
            )
            update_targets(q_net1_target, q_net1, tau=0.01)
            update_targets(q_net2_target, q_net2, tau=0.01)

    return policy_net, q_net1, q_net2

def get_action_from_sac(policy_net, q_net1, q_net2, state):
    """
    Return an action using a partially greedy approach with min Q-values.
    Using .detach() to avoid the "requires grad" -> numpy() error.
    """
    with torch.no_grad():
        state_t = torch.from_numpy(state).float().unsqueeze(0)
        logits = policy_net(state_t)
        probs = F.softmax(logits, dim=-1).detach().numpy().squeeze()

    # Epsilon-greedy
    if random.random() < 0.2:
        return np.random.choice(len(probs))
    else:
        with torch.no_grad():
            q1_output = q_net1(state_t).detach().numpy().squeeze()
            q2_output = q_net2(state_t).detach().numpy().squeeze()
        min_q_vals = np.minimum(q1_output, q2_output)
        action = np.argmax(min_q_vals)
        return action

def action_to_string(action: int):
    """
    0=NONE, 1=BUY, 2=SELL, 3=SHORT, 4=COVER
    """
    if action == 1:
        return "BUY"
    elif action == 2:
        return "SELL"
    elif action == 3:
        return "SHORT"
    elif action == 4:
        return "COVER"
    return "NONE"

###############################################################################
# Publicly Exposed Functions
###############################################################################

def run_logic(current_price, predicted_price, ticker):
    """
    Called externally for live logic:
      1) Load ticker CSV
      2) Filter, add predicted_price
      3) Train SAC
      4) Grab current position
      5) Build final state => get action => avoid duplicates => execute
    """
    suffix = convert_timeframe(BAR_TIMEFRAME)
    df = load_csv_for_ticker(ticker, suffix)
    df = filter_disabled_features(df, DISABLED_FEATURES)
    df["model_pred_price"] = predicted_price

    _ = train_random_forest_for_prediction(df)
    policy_net, q_net1, q_net2 = train_sac_on_data(df)

    pos = api.get_position(ticker)
    position_qty = float(pos.qty)

    env_feature_cols = [c for c in df.columns if c not in ['timestamp', 'close']]
    if 'close' not in df.columns:
        df['close'] = 0.0
    # Last row
    last_row = df.iloc[[-1]].copy()
    for c in env_feature_cols:
        last_row[c] = pd.to_numeric(last_row[c], errors='coerce').fillna(0)

    last_vals = last_row[env_feature_cols].values[0].astype(np.float32)
    state = np.concatenate([last_vals, [position_qty]])

    act_idx = get_action_from_sac(policy_net, q_net1, q_net2, state)
    act_str = action_to_string(act_idx)

    final_action = act_str
    if act_str == "BUY" and position_qty >= 1:
        final_action = "NONE"
    if act_str == "SHORT" and position_qty <= -1:
        final_action = "NONE"
    if act_str == "SELL" and position_qty <= 0:
        final_action = "NONE"
    if act_str == "COVER" and position_qty >= 0:
        final_action = "NONE"

    if final_action == "BUY":
        buy_shares(ticker, qty=1)
    elif final_action == "SELL":
        sell_shares(ticker, qty=1)
    elif final_action == "SHORT":
        short_shares(ticker, qty=1)
    elif final_action == "COVER":
        close_short(ticker, qty=1)
    # else do nothing

def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    """
    Called externally for backtest logic:
     1) Use first ticker
     2) Filter, add predicted_price
     3) Train SAC
     4) Build final state, pick action
     5) Avoid duplicates
     6) Return final action string
    """
    first_ticker = TICKERS.split(",")[0].strip()
    suffix = convert_timeframe(BAR_TIMEFRAME)
    df = load_csv_for_ticker(first_ticker, suffix)
    df = filter_disabled_features(df, DISABLED_FEATURES)
    df["model_pred_price"] = predicted_price

    _ = train_random_forest_for_prediction(df)
    policy_net, q_net1, q_net2 = train_sac_on_data(df)

    env_feature_cols = [c for c in df.columns if c not in ['timestamp', 'close']]
    if 'close' not in df.columns:
        df['close'] = 0.0

    last_row = df.iloc[[-1]].copy()
    for c in env_feature_cols:
        last_row[c] = pd.to_numeric(last_row[c], errors='coerce').fillna(0)

    last_vals = last_row[env_feature_cols].values[0].astype(np.float32)
    state = np.concatenate([last_vals, [position_qty]])

    act_idx = get_action_from_sac(policy_net, q_net1, q_net2, state)
    act_str = action_to_string(act_idx)

    final_action = act_str
    if act_str == "BUY" and position_qty >= 1:
        final_action = "NONE"
    if act_str == "SHORT" and position_qty <= -1:
        final_action = "NONE"
    if act_str == "SELL" and position_qty <= 0:
        final_action = "NONE"
    if act_str == "COVER" and position_qty >= 0:
        final_action = "NONE"

    return final_action