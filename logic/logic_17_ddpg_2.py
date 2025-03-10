"""
logic.py

A single Python script demonstrating a stock-trading strategy using:
  - DDPG Reinforcement Learning (RL) for action decisions
  - Random Forest (RF) for price predictions
  - Integration of CSV feature data (with optional columns disabled)
  - An actively trading approach that minimizes "NONE" actions
  - The same logic for both live trading (run_logic) and backtesting (run_backtest)

This version fixes:
  - IndexError in train_agent_on_data (by bounding our loop to the minimum length of df & RF preds)
  - RandomForest "X does not have valid feature names" warning (by consistently predicting from a DataFrame slice)

Install Requirements:
  - pandas, numpy, scikit-learn, torch, etc.
Ensure environment variables in .env or OS:
  - BAR_TIMEFRAME, TICKERS, DISABLED_FEATURES
"""

import os
import numpy as np
import pandas as pd
from collections import deque
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.ensemble import RandomForestRegressor

# Attempt to import the user-provided stubs for live trading.
try:
    from forest import api, buy_shares, sell_shares, short_shares, close_short
except ImportError:
    # Mock environment for demonstration. In production, remove this block.
    class MockAPI:
        def get_position(self, ticker):
            class Pos:
                qty = 0
            return Pos()
    api = MockAPI()
    def buy_shares(ticker, quantity=1):
        print(f"Mock BUY: {ticker}, qty={quantity}")
    def sell_shares(ticker, quantity=1):
        print(f"Mock SELL: {ticker}, qty={quantity}")
    def short_shares(ticker, quantity=1):
        print(f"Mock SHORT: {ticker}, qty={quantity}")
    def close_short(ticker, quantity=1):
        print(f"Mock COVER: {ticker}, qty={quantity}")

###############################################################################
#                             Helper Functions                                #
###############################################################################

def convert_timeframe(bar_timeframe: str) -> str:
    """
    Convert the BAR_TIMEFRAME to the appropriate filename suffix.
    Example conversions:
       "4Hour" -> "H4"
       "2Hour" -> "H2"
       "1Hour" -> "H1"
       "30Min" -> "M30"
       "15Min" -> "M15"
    """
    tf_map = {
        "4Hour": "H4",
        "2Hour": "H2",
        "1Hour": "H1",
        "30Min": "M30",
        "15Min": "M15"
    }
    return tf_map.get(bar_timeframe, bar_timeframe)

def load_and_filter_csv(ticker: str, suffix: str, disabled_features: list) -> pd.DataFrame:
    """
    Loads CSV for the given ticker + timeframe suffix, filters out disabled features,
    and returns the cleaned DataFrame.
    """
    filename = f"{ticker}_{suffix}.csv"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"CSV file not found: {filename}")
    
    df = pd.read_csv(filename)
    
    # Drop disabled features if they exist in the dataframe
    for feat in disabled_features:
        if feat in df.columns:
            df.drop(columns=[feat], inplace=True)
    
    return df

def get_disabled_features() -> list:
    """
    Reads DISABLED_FEATURES from .env or environment variables.
    Format example: DISABLED_FEATURES=body_size,candle_rise,...
    """
    raw = os.getenv("DISABLED_FEATURES", "")
    if not raw.strip():
        return []
    return [col.strip() for col in raw.split(",")]

def build_random_forest_and_predict(df: pd.DataFrame, target_col="close", n_estimators=50) -> np.ndarray:
    """
    Train a RandomForestRegressor on the CSV data to predict the next bar's target_col.
    Return an array of predicted values (aligned with df rows after shift).
    
    Steps:
      1. Shift target_col up by -1 to create 'target' for next-step prediction.
      2. Fit a RandomForest on all numeric features except 'timestamp','timeframe','target'.
      3. Return an array of predictions (same shape as the final df).
    """
    df_copy = df.copy()
    if target_col not in df_copy.columns:
        raise ValueError(f"Target column {target_col} not in dataframe.")
    
    # SHIFT the target_col so each row tries to predict the next bar's value
    df_copy["target"] = df_copy[target_col].shift(-1)
    df_copy.dropna(inplace=True)  # remove last row (because shift(-1) => NaN)

    drop_cols = ["timestamp", "timeframe", "target"]
    feature_cols = [c for c in df_copy.columns
                    if c not in drop_cols and df_copy[c].dtype != 'object']
    
    # X: DataFrame slice with consistent columns
    X = df_copy[feature_cols]
    y = df_copy["target"].values
    
    # Fit a RandomForest (demo)
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    rf.fit(X, y)
    
    # Predict on the same partial DataFrame for alignment
    preds = rf.predict(X)  # same length as X
    # We'll return a result of the same overall length as original df.
    # So let's build an array of length == original df, with np.nan for last row.
    full_preds = np.full(len(df), np.nan, dtype=np.float64)
    # The indexes in df_copy after dropna correspond to [0..len(df_copy)-1] of some portion of df
    # Typically they're the original [0..len(df)-2].
    # We'll align them carefully:
    valid_indices = df_copy.index  # these are the original row indices in df
    full_preds[valid_indices] = preds
    
    return full_preds

###############################################################################
#                   DDPG-Like RL Implementation (Discrete Hack)               #
###############################################################################

class ReplayBuffer:
    """Simple Experience Replay Buffer."""
    def __init__(self, capacity=int(1e5)):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state, dtype=np.float32),
                np.array(action, dtype=np.float32),
                np.array(reward, dtype=np.float32),
                np.array(next_state, dtype=np.float32),
                np.array(done, dtype=np.float32))
    
    def __len__(self):
        return len(self.buffer)

class ActorNetwork(nn.Module):
    """
    Actor that outputs a continuous value in [-1, 1].
    We'll discretize that into 5 bins => BUY, SELL, SHORT, COVER, NONE
    """
    def __init__(self, state_dim, hidden_dim=64):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.out(x))  # in [-1, 1]
        return x

class CriticNetwork(nn.Module):
    """
    Critic that takes (state, action) => Q-value
    """
    def __init__(self, state_dim, action_dim=1, hidden_dim=64):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

class DDPGAgent:
    """
    Minimal DDPG-like agent. We discretize continuous actor outputs into 5 actions:
        0 -> BUY
        1 -> SELL
        2 -> SHORT
        3 -> COVER
        4 -> NONE
    """
    def __init__(self, state_dim, gamma=0.99, tau=0.001, lr_actor=1e-3, lr_critic=1e-3):
        self.gamma = gamma
        self.tau = tau
        
        self.actor = ActorNetwork(state_dim)
        self.actor_target = ActorNetwork(state_dim)
        
        self.critic = CriticNetwork(state_dim)
        self.critic_target = CriticNetwork(state_dim)
        
        # Initialize target networks
        self.soft_update(self.actor_target, self.actor, 1.0)
        self.soft_update(self.critic_target, self.critic, 1.0)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.replay_buffer = ReplayBuffer()
        
        # Epsilon for exploration
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
    
    def select_continuous_action(self, state):
        """
        Returns a continuous action in [-1, 1].
        Adds noise if in exploration mode.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t).item()
        
        # Add exploration noise
        if np.random.rand() < self.epsilon:
            action += np.random.normal(0, 0.1)
        
        action = np.clip(action, -1.0, 1.0)
        return action
    
    def continuous_to_discrete(self, action_val: float) -> int:
        """
        Map continuous action in [-1,1] to discrete:
          [-1.0, -0.6) => SHORT (2)
          [-0.6, -0.2) => COVER (3)
          [-0.2,  0.2) => NONE  (4)
          [ 0.2,  0.6) => SELL  (1)
          [ 0.6,  1.0] => BUY   (0)
        """
        if action_val < -0.6:
            return 2  # SHORT
        elif action_val < -0.2:
            return 3  # COVER
        elif action_val < 0.2:
            return 4  # NONE
        elif action_val < 0.6:
            return 1  # SELL
        else:
            return 0  # BUY
    
    def discrete_to_str(self, action_idx: int) -> str:
        action_map = {
            0: "BUY",
            1: "SELL",
            2: "SHORT",
            3: "COVER",
            4: "NONE"
        }
        return action_map[action_idx]
    
    def push_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def soft_update(self, target_net, source_net, tau=None):
        if tau is None:
            tau = self.tau
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(
                tau * source_param.data + (1.0 - tau) * target_param.data
            )
    
    def train_step(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return
        
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        
        state_t = torch.FloatTensor(state)
        action_t = torch.FloatTensor(action).unsqueeze(1)
        reward_t = torch.FloatTensor(reward).unsqueeze(1)
        next_state_t = torch.FloatTensor(next_state)
        done_t = torch.FloatTensor(done).unsqueeze(1)
        
        # Critic update
        with torch.no_grad():
            next_action = self.actor_target(next_state_t)
            next_q = self.critic_target(next_state_t, next_action)
            target_q = reward_t + self.gamma * (1.0 - done_t) * next_q
        
        current_q = self.critic(state_t, action_t)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update (maximize Q => minimize -Q)
        pred_action = self.actor(state_t)
        actor_loss = -self.critic(state_t, pred_action).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_action(self, state, training=True):
        if not training:
            # Temporarily zero out exploration
            old_eps = self.epsilon
            self.epsilon = 0.0
            action_val = self.select_continuous_action(state)
            self.epsilon = old_eps
        else:
            action_val = self.select_continuous_action(state)
        
        discrete_idx = self.continuous_to_discrete(action_val)
        return discrete_idx

###############################################################################
#                           Training Utility                                  #
###############################################################################

GLOBAL_RL_AGENT = None
GLOBAL_FEATURE_COLS = []

def train_agent_on_data(df: pd.DataFrame, random_forest_preds=None, episodes=1):
    """
    Trains/updates the RL agent on the historical DataFrame rows. 
    The environment:
      - state = [enabled numeric features, predicted_price, position_qty]
      - action = discrete from {BUY, SELL, SHORT, COVER, NONE}
      - reward = PnL change + penalty_for_none
    """
    global GLOBAL_RL_AGENT
    global GLOBAL_FEATURE_COLS
    
    # Identify numeric feature columns (excluding 'timestamp', 'timeframe')
    drop_cols = ["timestamp", "timeframe"]
    feature_cols = [c for c in df.columns if c not in drop_cols and df[c].dtype != 'object']
    
    if GLOBAL_RL_AGENT is None:
        # Initialize agent and store feature columns
        GLOBAL_FEATURE_COLS = feature_cols
        state_dim = len(feature_cols) + 2  # +1 for predicted_price, +1 for position_qty
        GLOBAL_RL_AGENT = DDPGAgent(state_dim=state_dim)
    else:
        # Ensure we're using the same feature columns as originally established
        # (In case new ones appear or vanish, handle carefully or keep consistent.)
        GLOBAL_FEATURE_COLS = list(set(GLOBAL_FEATURE_COLS).intersection(set(feature_cols)))
    
    # If no random_forest_preds is provided, fallback to using df["close"]
    if random_forest_preds is None:
        random_forest_preds = df["close"].values
    
    # Because of shift, random_forest_preds might have NaN at the end => let's fill it with something
    # or we can clip to the same size as df
    # We'll just convert any NaN to the last known value or 0.
    random_forest_preds = np.nan_to_num(random_forest_preds, nan=df["close"].iloc[-1] if len(df) > 0 else 0)
    
    # We'll only iterate up to the minimum length of df vs. random_forest_preds
    length = min(len(df), len(random_forest_preds))
    if length < 2:
        # Not enough data to train
        return
    
    for _ in range(episodes):
        position_qty = 0.0
        prev_close = None
        for i in range(length - 1):
            row = df.iloc[i]
            next_row = df.iloc[i+1]
            
            # Build state
            numeric_vals = row[GLOBAL_FEATURE_COLS].values.astype(float)
            pred_price = random_forest_preds[i]
            state = np.concatenate([numeric_vals, [pred_price, position_qty]])
            
            action_idx = GLOBAL_RL_AGENT.get_action(state, training=True)
            
            # Next state
            next_numeric_vals = next_row[GLOBAL_FEATURE_COLS].values.astype(float)
            next_pred_price = random_forest_preds[i+1]
            next_state = np.concatenate([next_numeric_vals, [next_pred_price, position_qty]])
            
            # Reward: naive PnL change + penalty for NONE
            current_close = row.get("close", 0.0)
            next_close = next_row.get("close", current_close)
            
            pnl_change = position_qty * (next_close - current_close) if (prev_close is not None) else 0.0
            penalty_none = -0.01 if (action_idx == 4) else 0.0  # 4 => NONE
            reward = pnl_change + penalty_none
            
            # Simulate environment step updating position
            # 0 -> BUY
            # 1 -> SELL
            # 2 -> SHORT
            # 3 -> COVER
            # 4 -> NONE
            if action_idx == 0:  # BUY
                if position_qty < 1:
                    position_qty = 1
            elif action_idx == 1:  # SELL
                if position_qty >= 1:
                    position_qty = 0
            elif action_idx == 2:  # SHORT
                if position_qty > -1:
                    position_qty = -1
            elif action_idx == 3:  # COVER
                if position_qty <= -1:
                    position_qty = 0
            # else NONE => no change
            
            done = 1.0 if (i == (length - 2)) else 0.0
            
            # For Critic update, store the continuous action from the agent
            # We'll get it from the last call:
            continuous_action_val = GLOBAL_RL_AGENT.select_continuous_action(state)
            
            GLOBAL_RL_AGENT.push_experience(state, continuous_action_val, reward, next_state, done)
            GLOBAL_RL_AGENT.train_step(batch_size=32)
            
            prev_close = current_close

def decide_action_with_trained_agent(df_row: pd.Series,
                                     position_qty: float,
                                     predicted_price: float) -> str:
    """
    Once the agent is trained, pick a final action string for the current row.
    Enforce "no redundant trades":
      - If action=BUY but already long, => NONE
      - If action=SELL but no long, => NONE
      - If action=SHORT but already short, => NONE
      - If action=COVER but no short, => NONE
    """
    global GLOBAL_RL_AGENT
    global GLOBAL_FEATURE_COLS
    
    if GLOBAL_RL_AGENT is None:
        # If no agent, do NOTHING
        return "NONE"
    
    # Build state
    numeric_vals = df_row[GLOBAL_FEATURE_COLS].values.astype(float)
    state = np.concatenate([numeric_vals, [predicted_price, position_qty]])
    
    action_idx = GLOBAL_RL_AGENT.get_action(state, training=False)
    action_str = GLOBAL_RL_AGENT.discrete_to_str(action_idx)
    
    # No duplicate trades
    if action_str == "BUY" and position_qty >= 1:
        return "NONE"
    if action_str == "SELL" and position_qty < 1:
        return "NONE"
    if action_str == "SHORT" and position_qty <= -1:
        return "NONE"
    if action_str == "COVER" and position_qty > -1:
        return "NONE"
    
    return action_str

###############################################################################
#                          Required Functions                                 #
###############################################################################

def run_logic(current_price, predicted_price, ticker):
    """
    - Load CSV with bar_timeframe from env
    - Filter out DISABLED_FEATURES
    - Train/update RL model with optional RF predictions
    - Retrieve current position from api
    - Decide among buy_shares, sell_shares, short_shares, close_short, or none
    - Execute the action
    """
    bar_timeframe = os.getenv("BAR_TIMEFRAME", "4Hour")
    suffix = convert_timeframe(bar_timeframe)
    
    disabled_feats = get_disabled_features()
    df = load_and_filter_csv(ticker, suffix, disabled_feats)
    
    # Build or update RandomForest predictions
    try:
        rf_preds = build_random_forest_and_predict(df, target_col="close")
    except:
        rf_preds = None
    
    # Train RL
    train_agent_on_data(df, random_forest_preds=rf_preds, episodes=1)
    
    # Get current position from live API
    pos = api.get_position(ticker)
    position_qty = float(pos.qty)
    
    # Use the last row of df as the "current" state
    if len(df) == 0:
        return  # No data => do nothing
    
    df_row = df.iloc[-1]
    action_str = decide_action_with_trained_agent(df_row, position_qty, predicted_price)
    
    # Execute it if it doesn't violate "no duplicate trades"
    if action_str == "BUY" and position_qty < 1:
        buy_shares(ticker, quantity=1)
    elif action_str == "SELL" and position_qty >= 1:
        sell_shares(ticker, quantity=1)
    elif action_str == "SHORT" and position_qty > -1:
        short_shares(ticker, quantity=1)
    elif action_str == "COVER" and position_qty <= -1:
        close_short(ticker, quantity=1)
    # else NONE => do nothing

def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    """
    - Load CSV for the *first* ticker in TICKERS from env
    - Filter out DISABLED_FEATURES
    - Train or update RL model (with same logic)
    - Return an action among BUY, SELL, SHORT, COVER, NONE
    - Enforce no duplicate trades via position_qty
    """
    tickers_env = os.getenv("TICKERS", "TSLA,AAPL")
    first_ticker = tickers_env.split(",")[0].strip()
    
    bar_timeframe = os.getenv("BAR_TIMEFRAME", "4Hour")
    suffix = convert_timeframe(bar_timeframe)
    
    disabled_feats = get_disabled_features()
    df = load_and_filter_csv(first_ticker, suffix, disabled_feats)
    
    try:
        rf_preds = build_random_forest_and_predict(df, target_col="close")
    except:
        rf_preds = None
    
    train_agent_on_data(df, random_forest_preds=rf_preds, episodes=1)
    
    if len(df) == 0:
        return "NONE"
    
    df_row = df.iloc[-1]
    action_str = decide_action_with_trained_agent(df_row, position_qty, predicted_price)
    
    # No duplicate trades
    if action_str == "BUY" and position_qty >= 1:
        return "NONE"
    if action_str == "SELL" and position_qty < 1:
        return "NONE"
    if action_str == "SHORT" and position_qty <= -1:
        return "NONE"
    if action_str == "COVER" and position_qty > -1:
        return "NONE"
    
    return action_str
