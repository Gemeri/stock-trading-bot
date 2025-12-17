import os
import logging
import pandas as pd
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from dotenv import load_dotenv
import logic.tools as tools

# =============================================================================
# Load .env variables and set up configuration
# =============================================================================
load_dotenv()
def get_csv_filename(ticker):
    return tools.get_csv_filename(ticker)


# =============================================================================
# Technical Indicator Functions
# =============================================================================
def compute_indicators(df):
    # Even if the CSV contained these columns (and they might be disabled),
    # we compute these indicators as needed.
    df['ema_short'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=26, adjust=False).mean()
    df['ema_diff'] = df['ema_short'] - df['ema_long']
    df['macd_line'] = df['ema_diff']
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    
    # ATR calculation (period = 14)
    period = 14
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['true_range'].rolling(window=period).mean()
    
    # RSI calculation (period = 14)
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Remove temporary columns
    df.drop(columns=['prev_close', 'tr1', 'tr2', 'tr3', 'true_range'], inplace=True)
    return df

# =============================================================================
# Custom Trading Environment
# =============================================================================
class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.current_step = 0
        self.done = False
        
        # Portfolio state variables
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.shares_held = 0    # positive for long
        self.position = 0       # 0: none, 1: long
        
        # Actions: 0: BUY, 1: SELL, 2: NONE
        self.action_space = spaces.Discrete(3)
        
        # Observation: features from the dataframe (excluding non-feature columns)
        excluded_columns = ['timestamp'] if 'timestamp' in self.df.columns else []
        self.feature_columns = [col for col in self.df.columns if col not in excluded_columns]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=(len(self.feature_columns) + 3,), dtype=np.float32)
        self.portfolio_history = []
    
    def _get_observation(self):
        idx = min(self.current_step, len(self.df) - 1)
        row = self.df.iloc[idx][self.feature_columns].values.astype(np.float32)
        extra = np.array([self.balance, self.shares_held, self.position], dtype=np.float32)
        return np.concatenate([row, extra])
    
    def _take_action(self, action):
        current_price = self.df.iloc[min(self.current_step, len(self.df)-1)]['close']
        if action == 0:  # BUY
            if self.position == 1:
                return
            max_shares = int(self.balance // current_price)
            if max_shares > 0:
                cost = max_shares * current_price
                self.balance -= cost
                self.shares_held += max_shares
                self.position = 1
        elif action == 1:  # SELL
            if self.position == 1:
                revenue = self.shares_held * current_price
                self.balance += revenue
                self.shares_held = 0
                self.position = 0
        elif action == 2:  # NONE
            pass
    
    def step(self, action):
        if self.done:
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0, self.done, {}
        self._take_action(action)
        self.current_step += 1
        if self.current_step >= len(self.df):
            self.done = True
            current_price = self.df.iloc[-1]['close']
        else:
            current_price = self.df.iloc[self.current_step]['close']
        portfolio_value = self.balance
        if self.position == 1:
            portfolio_value += self.shares_held * current_price
        self.portfolio_history.append(portfolio_value)
        reward = portfolio_value
        obs = self._get_observation() if not self.done else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {'portfolio_value': portfolio_value}
        return obs, reward, self.done, info
    
    def reset(self):
        self.current_step = 0
        self.done = False
        self.balance = self.initial_balance
        self.shares_held = 0
        self.position = 0
        self.portfolio_history = []
        return self._get_observation()
    
    def render(self, mode='human', close=False):
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Shares: {self.shares_held}, Position: {self.position}")

# =============================================================================
# RL Competition Logic (unchanged functionality)
# =============================================================================
def simulate_episode(model, env):
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
    final_value = info['portfolio_value']
    return final_value, env.portfolio_history

def run_rl_competition(df):
    env = TradingEnv(df)
    model1 = DQN("MlpPolicy", env, verbose=0)
    model2 = DQN("MlpPolicy", env, verbose=0)
    
    # Initial training phase
    model1.learn(total_timesteps=5000)
    model2.learn(total_timesteps=5000)
    
    num_rounds = 5
    best_model = None
    for round in range(1, num_rounds + 1):
        final_value1, _ = simulate_episode(model1, env)
        final_value2, _ = simulate_episode(model2, env)
        if final_value1 >= final_value2:
            best_model = model1
            loser = model2
        else:
            best_model = model2
            loser = model1
        loser.learn(total_timesteps=1000, reset_num_timesteps=False)
    
    # Run through the full dataset using NONE action to get final observation
    obs = env.reset()
    while True:
        obs, reward, done, info = env.step(2)
        if done:
            break
    next_action, _ = best_model.predict(obs, deterministic=True)
    return next_action

# =============================================================================
# CSV Loading and Preprocessing
# =============================================================================
def load_and_preprocess_csv(csv_path, predicted_price):
    df = pd.read_csv(csv_path)
    # Compute technical indicators (which will (re)calculate needed features such as rsi)
    df = compute_indicators(df)
    # Add predicted_price as a new feature column
    df['predicted_price'] = predicted_price
    return df

def preprocess_candles_for_backtest(candles, current_timestamp, predicted_price):
    if 'timestamp' in candles.columns:
        df = candles[candles['timestamp'] <= current_timestamp].copy()
    else:
        df = candles.copy()
    df = compute_indicators(df)
    df['predicted_price'] = predicted_price
    return df

# =============================================================================
# Main External Trade Logic Functions
# =============================================================================
def run_logic(current_price, predicted_price, ticker):
    from forest import api, buy_shares, sell_shares, short_shares, close_short

    # Load and preprocess the CSV data
    df = load_and_preprocess_csv(get_csv_filename(ticker), predicted_price)
    # Run the RL competition logic and get the predicted action
    action = run_rl_competition(df)
    
    # Get live position (if none exists, assume 0)
    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception as e:
        logging.info(f"No open position for {ticker}: {e}")
        position_qty = 0.0

    account = api.get_account()
    cash = float(account.cash)
    
    # Execute trade based on RL action (preventing duplicate trades)
    if action == 0 and position_qty <= 0:
        max_shares = int(cash // current_price)
        logging.info("buy")
        buy_shares(ticker, max_shares, current_price, predicted_price)
    elif action == 1 and position_qty > 0:
        logging.info("sell")
        sell_shares(ticker, position_qty, current_price, predicted_price)
    # Action 2 (NONE) means HOLD – do nothing.

def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles, ticker):
    df = preprocess_candles_for_backtest(candles, current_timestamp, predicted_price)
    action = run_rl_competition(df)
    
    if action == 0 and position_qty <= 0:
        logging.info("buy")
        return "BUY"
    elif action == 1 and position_qty > 0:
        logging.info("sell")
        return "SELL"
    else:
        return "NONE"