import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from stable_baselines3 import A2C
from sklearn.ensemble import RandomForestRegressor
from forest import api, buy_shares, sell_shares, short_shares, close_short

load_dotenv()

BAR_TIMEFRAME = os.getenv('BAR_TIMEFRAME')
TICKERS = os.getenv('TICKERS').split(',')
DISABLED_FEATURES = os.getenv('DISABLED_FEATURES').split(',')

# Convert timeframe mapping
timeframe_conversion = {"4Hour": "H4", "2Hour": "H2", "1Hour": "H1", "30Min": "M30", "15Min": "M15"}
timeframe_suffix = timeframe_conversion.get(BAR_TIMEFRAME, "H1")

# Helper functions
def get_csv_path(ticker):
    return f"{ticker}_{timeframe_suffix}.csv"

def load_data(ticker):
    df = pd.read_csv(get_csv_path(ticker), parse_dates=['timestamp'])
    # Exclude disabled features, timestamp, and close for RF training purposes
    features = [col for col in df.columns if col not in DISABLED_FEATURES + ['timestamp', 'close']]
    return df, features

# Initialize RF model for price prediction
def train_rf_model(df, features):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    X = df[features]
    y = df['close']
    rf.fit(X, y)
    return rf

# Prepare RL environment
def create_rl_env(df, features):
    from gym import spaces
    import gym

    class TradingEnv(gym.Env):
        def __init__(self, data):
            super().__init__()
            self.data = data
            self.current_step = 0
            self.action_space = spaces.Discrete(5)  # BUY, SELL, SHORT, COVER, NONE
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(data.shape[1],), dtype=np.float32)

        def reset(self):
            self.current_step = 0
            return self.data.iloc[self.current_step].values

        def step(self, action):
            reward = 0
            done = self.current_step >= len(self.data) - 2
            current_price = self.data.iloc[self.current_step]['close']
            next_price = self.data.iloc[self.current_step + 1]['close']

            # Reward based on price difference
            if action == 0:  # BUY
                reward = next_price - current_price
            elif action == 1:  # SELL
                reward = current_price - next_price
            elif action == 2:  # SHORT
                reward = current_price - next_price
            elif action == 3:  # COVER
                reward = next_price - current_price
            elif action == 4:  # NONE
                reward = -0.01  # slight penalty

            self.current_step += 1
            next_state = self.data.iloc[self.current_step].values
            return next_state, reward, done, {}

    # It is assumed that the passed DataFrame already includes the 'close' column.
    return TradingEnv(df[features])

# Train RL Model
def train_rl_model(env):
    model = A2C('MlpPolicy', env, gamma=0.99, learning_rate=0.001, verbose=0)
    model.learn(total_timesteps=len(env.data))
    return model

# Core functions
def run_logic(current_price, predicted_price, ticker):
    df, features = load_data(ticker)
    # Train RF using features that exclude 'close'
    rf_model = train_rf_model(df, features)

    # Compute the predicted price column
    df['predicted_price'] = rf_model.predict(df[features])
    # Build the feature list for the RL environment.
    # We add back 'close' because the environment's reward function uses it.
    features_rl = features + ['predicted_price', 'close']

    env = create_rl_env(df, features_rl)
    rl_model = train_rl_model(env)

    current_state = df[features_rl].iloc[-1].values
    action, _ = rl_model.predict(current_state)

    pos = api.get_position(ticker)
    position_qty = float(pos.qty)

    if action == 0 and position_qty <= 0:
        buy_shares(ticker)
    elif action == 1 and position_qty > 0:
        sell_shares(ticker)
    elif action == 2 and position_qty >= 0:
        short_shares(ticker)
    elif action == 3 and position_qty < 0:
        close_short(ticker)

def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    ticker = TICKERS[0]

    if isinstance(candles, pd.DataFrame):
        candles = len(candles)
    else:
        candles = int(candles)

    df, features = load_data(ticker)

    if candles < len(df):
        train_df = df.iloc[:-candles]
        test_df = df.iloc[-candles:]
    else:
        train_df = df.copy()
        test_df = df.copy()

    # Train RF model on train_df using the RF features (excluding 'close')
    rf_model = train_rf_model(train_df, features)
    
    # Update both train and test dataframes with the predicted price column
    train_df = train_df.copy()
    train_df['predicted_price'] = rf_model.predict(train_df[features])
    
    test_df = test_df.copy()
    test_df['predicted_price'] = rf_model.predict(test_df[features])
    
    # Build the RL feature list (include 'close')
    features_rl = features + ['predicted_price', 'close']

    env = create_rl_env(train_df, features_rl)
    rl_model = train_rl_model(env)

    current_row = test_df[test_df['timestamp'] == current_timestamp]

    if current_row.empty:
        return "NONE"

    current_state = current_row[features_rl].values[0]
    action, _ = rl_model.predict(current_state)

    if action == 0 and position_qty <= 0:
        return "BUY"
    elif action == 1 and position_qty > 0:
        return "SELL"
    elif action == 2 and position_qty >= 0:
        return "SHORT"
    elif action == 3 and position_qty < 0:
        return "COVER"
    else:
        return "NONE"
