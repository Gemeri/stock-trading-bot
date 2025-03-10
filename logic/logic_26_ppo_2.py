import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.ensemble import RandomForestRegressor
from dotenv import load_dotenv

load_dotenv()

BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME")
TICKERS = os.getenv("TICKERS").split(',')
DISABLED_FEATURES = os.getenv("DISABLED_FEATURES", "").split(',')

TIMEFRAME_SUFFIX = {
    "4Hour": "H4", "2Hour": "H2", "1Hour": "H1",
    "30Min": "M30", "15Min": "M15"
}[BAR_TIMEFRAME]

ACTIONS = ["BUY", "SELL", "SHORT", "COVER", "NONE"]


# Helper Functions
def load_data(ticker):
    filename = f"{ticker}_{TIMEFRAME_SUFFIX}.csv"
    df = pd.read_csv(filename)
    df = df.drop(columns=[feat for feat in DISABLED_FEATURES if feat in df.columns], errors='ignore')
    return df.select_dtypes(include=[np.number])


def get_features(df, predicted_price, rf_columns):
    df = df.copy()
    df['predicted_price'] = predicted_price
    features_df = df[rf_columns]
    return features_df.iloc[-1:].values


# RL Environment
from gym import spaces
import gym

class StockTradingEnv(gym.Env):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(df.shape[1],), dtype=np.float32)
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self.df.iloc[self.current_step].values

    def step(self, action):
        reward = self._calculate_reward(action)
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        obs = self.df.iloc[self.current_step].values if not done else self.df.iloc[-1].values
        return obs, reward, done, {}

    def _calculate_reward(self, action):
        price_change = self.df['close'].iloc[self.current_step] - self.df['open'].iloc[self.current_step]
        reward = price_change if ACTIONS[action] in ["BUY", "COVER"] else -price_change
        if ACTIONS[action] == "NONE":
            reward -= abs(price_change) * 0.1
        return reward


# Training and loading models
def train_rl_model(df):
    env = DummyVecEnv([lambda: StockTradingEnv(df)])
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=5000)
    return model


def load_rf_model(df):
    X = df.drop(columns=['close'], errors='ignore')
    y = df['close']
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    return model


# Main Functions
def run_logic(current_price, predicted_price, ticker):
    from forest import api, buy_shares, sell_shares, short_shares, close_short

    df = load_data(ticker)
    rf_model = load_rf_model(df)
    rf_columns = df.drop(columns=['close'], errors='ignore').columns

    features = get_features(df, predicted_price, rf_columns)
    predicted_close = rf_model.predict(features)[0]

    rl_model = train_rl_model(df)
    obs = np.append(features.flatten(), predicted_close)

    action, _ = rl_model.predict(obs)

    pos = api.get_position(ticker)
    position_qty = float(pos.qty)

    if ACTIONS[action] == "BUY" and position_qty <= 0:
        buy_shares(ticker)
    elif ACTIONS[action] == "SELL" and position_qty > 0:
        sell_shares(ticker)
    elif ACTIONS[action] == "SHORT" and position_qty >= 0:
        short_shares(ticker)
    elif ACTIONS[action] == "COVER" and position_qty < 0:
        close_short(ticker)


def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    ticker = TICKERS[0]
    df = load_data(ticker)

    candles = candles.shape[0] if isinstance(candles, pd.DataFrame) else int(candles)

    backtest_df = df.tail(candles)
    train_df = df.head(len(df) - candles)

    rf_model = load_rf_model(train_df)
    rf_columns = train_df.drop(columns=['close'], errors='ignore').columns
    rl_model = train_rl_model(train_df)

    features = get_features(backtest_df, predicted_price, rf_columns)
    predicted_close = rf_model.predict(features)[0]

    obs = np.append(features.flatten(), predicted_close)
    action, _ = rl_model.predict(obs)

    if ACTIONS[action] == "BUY" and position_qty <= 0:
        return "BUY"
    elif ACTIONS[action] == "SELL" and position_qty > 0:
        return "SELL"
    elif ACTIONS[action] == "SHORT" and position_qty >= 0:
        return "SHORT"
    elif ACTIONS[action] == "COVER" and position_qty < 0:
        return "COVER"
    return "NONE"
