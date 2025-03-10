import os
import random
import numpy as np
import pandas as pd
from collections import deque
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.optim as optim
import copy

# =============================================================================
# LOAD ENVIRONMENT VARIABLES AND SET GLOBALS
# =============================================================================

load_dotenv()  # load .env variables

# Mapping for the timeframe names:
TIMEFRAME_MAP = {
    "4Hour": "H4",
    "2Hour": "H2",
    "1Hour": "H1",
    "30Min": "M30",
    "15Min": "M15"
}

# Full list of available features in CSVs:
AVAILABLE_FEATURES = [
    'open','high','low','close','volume','vwap',
    'price_change','high_low_range','log_volume','sentiment',
    'price_return','candle_rise','body_size','wick_to_body','macd_line',
    'rsi','momentum','roc','atr','hist_vol','obv','volume_change',
    'stoch_k','bollinger_upper','bollinger_lower',
    'lagged_close_1','lagged_close_2','lagged_close_3','lagged_close_5','lagged_close_10'
]

# Get disabled features (if any) from the environment (default shown)
disabled_features_str = os.getenv("DISABLED_FEATURES", 
    "body_size,candle_rise,high_low_range,hist_vol,log_volume,macd_line,price_change,price_return,roc,rsi,stoch_k,transactions,volume,volume_change,wick_to_body")
DISABLED_FEATURES = [feat.strip() for feat in disabled_features_str.split(",") if feat.strip()]

# Dictionary to store trained models for reuse (keyed by CSV filename)
trained_models = {}

# =============================================================================
# REPLAY BUFFER
# =============================================================================

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# =============================================================================
# ACTOR AND CRITIC NETWORKS (Using PyTorch)
# =============================================================================

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim=1, hidden_dim=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.tanh = nn.Tanh()  # to force output to [-1, 1]
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim=1, hidden_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# =============================================================================
# TRADING ENVIRONMENT (For Training on Historical CSV Data)
# =============================================================================

class TradingEnv:
    """
    A trading environment that steps through the CSV candle data.
    The state consists of (enabled features, current position, predicted price).
    Position is defined as: 1 for long, -1 for short, 0 for flat.
    Reward is computed as the change in price times the current position.
    """
    def __init__(self, data, enabled_features):
        self.data = data.reset_index(drop=True)
        self.enabled_features = enabled_features
        self.n = len(self.data)
        self.index = 0
        self.position = 0  # 1: long, -1: short, 0: flat
        self.last_price = self.data.loc[0, 'close']
        
    def reset(self):
        self.index = 0
        self.position = 0
        self.last_price = self.data.loc[0, 'close']
        return self._get_state()
    
    def _get_state(self):
        row = self.data.loc[self.index]
        features = row[self.enabled_features].values.astype(np.float32)
        # For training, simulate a predicted price as the next candle's close (or same if at end)
        if self.index < self.n - 1:
            predicted_price = float(self.data.loc[self.index+1, 'close'])
        else:
            predicted_price = float(row['close'])
        state = np.concatenate([features, np.array([self.position], dtype=np.float32),
                                np.array([predicted_price], dtype=np.float32)])
        return state
    
    def step(self, action):
        # Use a threshold to decide which discrete trade to make.
        THRESHOLD = 0.3
        current_price = float(self.data.loc[self.index, 'close'])
        # Reward: change in price * current position (i.e. profit or loss on held position)
        reward = (current_price - self.last_price) * (self.position)
        self.last_price = current_price
        
        # --- Decision Logic Based on Action and Current Position ---
        if self.position == 0:
            if action > THRESHOLD:
                # Open a long position (BUY)
                self.position = 1
            elif action < -THRESHOLD:
                # Open a short position (SHORT)
                self.position = -1
            # Else: remain flat (NONE)
        elif self.position == 1:
            if action < -THRESHOLD:
                # Close long position (SELL)
                self.position = 0
        elif self.position == -1:
            if action > THRESHOLD:
                # Close short position (COVER)
                self.position = 0
        # -----------------------------------------------------------------
        self.index += 1
        done = self.index >= self.n - 1
        next_state = self._get_state() if not done else np.zeros_like(self._get_state())
        return next_state, reward, done, {}

# =============================================================================
# DDPG TRAINING FUNCTION WITH EPSILON-GREEDY EXPLORATION
# =============================================================================

def train_ddpg(env, state_dim, num_episodes=50, batch_size=64, gamma=0.99, tau=0.005,
               actor_lr=1e-4, critic_lr=1e-3, buffer_capacity=100000,
               epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, noise_std=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = Actor(state_dim).to(device)
    critic = Critic(state_dim).to(device)
    target_actor = copy.deepcopy(actor)
    target_critic = copy.deepcopy(critic)
    
    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)
    
    replay_buffer = ReplayBuffer(buffer_capacity)
    
    epsilon = epsilon_start
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            # Epsilon-greedy exploration: choose a random action with probability epsilon
            if random.random() < epsilon:
                action = np.random.uniform(-1, 1)
            else:
                with torch.no_grad():
                    action = actor(state_tensor).cpu().data.numpy().flatten()[0]
                    # Add Gaussian exploration noise
                    action = action + np.random.normal(0, noise_std)
            action = np.clip(action, -1, 1)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            
            if len(replay_buffer) > batch_size:
                s, a, r, s_next, d = replay_buffer.sample(batch_size)
                s = torch.FloatTensor(s).to(device)
                a = torch.FloatTensor(a).unsqueeze(1).to(device)
                r = torch.FloatTensor(r).unsqueeze(1).to(device)
                s_next = torch.FloatTensor(s_next).to(device)
                d = torch.FloatTensor(d).unsqueeze(1).to(device)
                
                with torch.no_grad():
                    next_action = target_actor(s_next)
                    target_q = target_critic(s_next, next_action)
                    y = r + gamma * (1 - d) * target_q
                
                current_q = critic(s, a)
                critic_loss = nn.MSELoss()(current_q, y)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                
                # Actor loss: maximize Q(s, actor(s))
                actor_loss = -critic(s, actor(s)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                
                # Soft update target networks:
                for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.3f}")
    return actor

# =============================================================================
# NORMALIZATION FUNCTIONS
# =============================================================================

def normalize_state(state, feature_means, feature_stds, num_features):
    """
    Normalize the first num_features of the state vector.
    Then, also normalize the predicted price using the same statistics as for 'close'.
    (Assumes that the 'close' feature is among the enabled features.)
    """
    norm_features = (state[:num_features] - feature_means) / (feature_stds + 1e-8)
    # Position is the next value (not normalized)
    position = state[num_features]
    # Normalize predicted price using 'close' stats (if available)
    if 'close' in AVAILABLE_FEATURES and 'close' not in DISABLED_FEATURES:
        try:
            close_index = [i for i, feat in enumerate(AVAILABLE_FEATURES) if feat == 'close' and feat not in DISABLED_FEATURES][0]
            close_mean = feature_means[close_index]
            close_std = feature_stds[close_index]
        except IndexError:
            close_mean, close_std = 0, 1
    else:
        close_mean, close_std = 0, 1
    predicted_price = state[num_features+1]
    norm_predicted_price = (predicted_price - close_mean) / (close_std + 1e-8)
    norm_state = np.concatenate([norm_features, np.array([position, norm_predicted_price], dtype=np.float32)])
    return norm_state

# =============================================================================
# CLASS TO HOLD THE TRAINED MODEL AND NORMALIZATION PARAMETERS
# =============================================================================

class TrainedDDPGModel:
    def __init__(self, actor, feature_means, feature_stds, enabled_features):
        self.actor = actor
        self.feature_means = feature_means
        self.feature_stds = feature_stds
        self.enabled_features = enabled_features
        self.state_dim = len(enabled_features) + 2  # (features, position, predicted price)

# =============================================================================
# DATA LOADING AND MODEL OBTAINING FUNCTIONS
# =============================================================================

def load_data(csv_filename):
    """
    Load the CSV file and determine which features are enabled.
    """
    df = pd.read_csv(csv_filename)
    enabled_features = [feat for feat in AVAILABLE_FEATURES if feat in df.columns and feat not in DISABLED_FEATURES]
    df.fillna(0, inplace=True)
    return df, enabled_features

def get_trained_model(csv_filename):
    """
    If a trained model exists for the CSV filename, return it.
    Otherwise, load the data, compute normalization parameters, train the DDPG agent,
    and store the trained model for later use.
    """
    global trained_models
    if csv_filename in trained_models:
        return trained_models[csv_filename]
    df, enabled_features = load_data(csv_filename)
    feature_data = df[enabled_features].values.astype(np.float32)
    feature_means = np.mean(feature_data, axis=0)
    feature_stds = np.std(feature_data, axis=0)
    
    env = TradingEnv(df, enabled_features)
    state_dim = len(enabled_features) + 2  # (features + position + predicted price)
    
    print(f"Training DDPG model on {csv_filename} ...")
    actor = train_ddpg(env, state_dim)
    
    model = TrainedDDPGModel(actor, feature_means, feature_stds, enabled_features)
    trained_models[csv_filename] = model
    return model

# =============================================================================
# DECISION MAKING: FROM MODEL OUTPUT TO TRADE ACTION
# =============================================================================

def get_action_from_model(model, current_features, position, predicted_price):
    """
    Build the state from raw feature values (for the enabled features), the current position
    and the externally supplied predicted price. Then normalize and pass through the actor.
    """
    state = np.concatenate([current_features, np.array([position, predicted_price], dtype=np.float32)])
    norm_state = normalize_state(state, model.feature_means, model.feature_stds, len(model.enabled_features))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_tensor = torch.FloatTensor(norm_state).unsqueeze(0).to(device)
    with torch.no_grad():
        action_value = model.actor(state_tensor).cpu().data.numpy().flatten()[0]
    return action_value

def map_action_to_decision(action_value, position, threshold=0.3):
    """
    Given the continuous action value (in [-1, 1]) and the current position,
    return one of the discrete decisions: BUY, SELL, SHORT, COVER or NONE.
    
    Logic:
      - If flat (position == 0):
          action > threshold  --> BUY
          action < -threshold --> SHORT
          Otherwise, force a random decision between BUY and SHORT to encourage trading.
      - If long (position > 0):
          action < -threshold --> SELL
      - If short (position < 0):
          action > threshold --> COVER
      - Otherwise, do nothing.
    """
    decision = "NONE"
    if position == 0:
        if action_value > threshold:
            decision = "BUY"
        elif action_value < -threshold:
            decision = "SHORT"
        else:
            decision = random.choice(["BUY", "SHORT"])
    elif position > 0:
        if action_value < -threshold:
            decision = "SELL"
    elif position < 0:
        if action_value > threshold:
            decision = "COVER"
    return decision

# =============================================================================
# CORE FUNCTIONS: run_logic and run_backtest
# =============================================================================

def run_logic(current_price, predicted_price, ticker):
    """
    In live trading, use the RL model to decide the trade.
    - The CSV filename is constructed as: (ticker)_(realTimeframe).csv.
    - The BAR_TIMEFRAME is taken from the .env and mapped (e.g. "1Hour" -> "H1").
    - The latest candle (row) is used for its features.
    - The current position is obtained via the API.
    - Then, based on the continuous output from the actor, a discrete decision is made.
    - Finally, if the decision is applicable (e.g. BUY only if flat), the corresponding trade is executed.
    """
    from forest import api, buy_shares, sell_shares, short_shares, close_short

    bar_timeframe = os.getenv("BAR_TIMEFRAME", "1Hour")
    real_timeframe = TIMEFRAME_MAP.get(bar_timeframe, "H1")
    csv_filename = f"{ticker}_{real_timeframe}.csv"
    
    model = get_trained_model(csv_filename)
    
    df, _ = load_data(csv_filename)
    latest_row = df.iloc[-1]
    current_features = latest_row[model.enabled_features].values.astype(np.float32)
    
    pos = api.get_position(ticker)
    try:
        position_qty = float(pos.qty)
    except Exception:
        position_qty = 0.0
    if position_qty > 0:
        position = 1
    elif position_qty < 0:
        position = -1
    else:
        position = 0
    
    action_value = get_action_from_model(model, current_features, position, predicted_price)
    decision = map_action_to_decision(action_value, position)
    
    if decision == "BUY" and position == 0:
        buy_shares(ticker)
    elif decision == "SELL" and position > 0:
        sell_shares(ticker)
    elif decision == "SHORT" and position == 0:
        short_shares(ticker)
    elif decision == "COVER" and position < 0:
        close_short(ticker)

def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    """
    For backtesting the RL strategy:
    - The CSV filename is built from the first ticker listed in TICKERS (from .env)
      and the (mapped) BAR_TIMEFRAME.
    - The latest candle’s features are used.
    - The current position is determined from the given position_qty.
    - The same RL model is then used to decide and the decision is returned as a string.
    """
    tickers = os.getenv("TICKERS", "")
    first_ticker = tickers.split(",")[0].strip() if tickers else ""
    
    bar_timeframe = os.getenv("BAR_TIMEFRAME", "1Hour")
    real_timeframe = TIMEFRAME_MAP.get(bar_timeframe, "H1")
    csv_filename = f"{first_ticker}_{real_timeframe}.csv"
    
    model = get_trained_model(csv_filename)
    
    df, _ = load_data(csv_filename)
    latest_row = df.iloc[-1]
    current_features = latest_row[model.enabled_features].values.astype(np.float32)
    
    if position_qty > 0:
        position = 1
    elif position_qty < 0:
        position = -1
    else:
        position = 0
        
    action_value = get_action_from_model(model, current_features, position, predicted_price)
    decision = map_action_to_decision(action_value, position)
    return decision
