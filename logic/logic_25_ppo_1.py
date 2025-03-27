import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from stable_baselines3 import PPO
from sklearn.ensemble import RandomForestRegressor
import gym
from gym import spaces
from forest import api, buy_shares, sell_shares, short_shares, close_short
import logging

# Load environment variables
load_dotenv()

BAR_TIMEFRAME = os.getenv('BAR_TIMEFRAME')
TICKERS = os.getenv('TICKERS').split(',')

# Define the enabled features (do not include timestamp for training purposes)
ENABLED_FEATURES = [
    "open", "high", "low", "close", "vwap", "momentum", "atr", "obv",
    "bollinger_upper", "bollinger_lower", "lagged_close_1", "lagged_close_2",
    "lagged_close_3", "lagged_close_5", "lagged_close_10", "sentiment"
]

# Map the timeframe from the .env value to the file suffix
timeframe_conversion = {"4Hour": "H4", "2Hour": "H2", "1Hour": "H1", "30Min": "M30", "15Min": "M15"}
timeframe_suffix = timeframe_conversion.get(BAR_TIMEFRAME, "H1")


def get_csv_path(ticker):
    """Return the CSV filename based on ticker and timeframe."""
    return f"{ticker}_{timeframe_suffix}.csv"


def load_data(ticker):
    """
    Load CSV data for a given ticker.
    Assumes CSV includes columns: timestamp, open, high, low, close, vwap, momentum, atr,
    obv, bollinger_upper, bollinger_lower, lagged_close_1, lagged_close_2, lagged_close_3,
    lagged_close_5, lagged_close_10, sentiment.
    """
    df = pd.read_csv(get_csv_path(ticker), parse_dates=['timestamp'])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    # Drop rows with missing data
    df.dropna(inplace=True)
    # Convert numeric columns to float32
    for col in ENABLED_FEATURES:
        df[col] = df[col].astype(np.float32)
    return df


def train_rf_model(df):
    """
    Train a RandomForestRegressor on the historical data to predict the close price.
    Uses all ENABLED_FEATURES except 'close' as predictors.
    Returns the trained model and the list of features used for prediction.
    """
    rf_features = [f for f in ENABLED_FEATURES if f != "close"]
    X = df[rf_features]
    y = df["close"]
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    return rf, rf_features


class TradingEnv(gym.Env):
    """
    Custom Trading Environment for Reinforcement Learning.
    Observation: [enabled features] + predicted_price.
    Action Space:
      0: BUY
      1: SELL
      2: SHORT
      3: COVER
      4: HOLD (None)
    Reward: Based on realized profit/loss with a small transaction cost.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, df, feature_columns):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.current_step = 0
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.feature_columns),), dtype=np.float32
        )
        # For simulation of trading performance (not used in RL decision)
        self.initial_balance = 100000.0
        self.balance = self.initial_balance
        self.position = 0    # +1 for long, -1 for short, 0 for flat
        self.last_trade_price = 0.0
        self.transaction_cost = 0.001  # 0.1% cost per trade

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.last_trade_price = 0.0
        return self._get_observation()

    def _get_observation(self):
        obs = self.df.iloc[self.current_step][self.feature_columns].values.astype(np.float32)
        return obs

    def step(self, action):
        """
        Execute one time step within the environment.
        The reward is computed based on price change and the trade action.
        """
        current_close = self.df.iloc[self.current_step]["close"]
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        next_close = self.df.iloc[self.current_step]["close"]
        reward = 0.0

        # Execute action if it avoids duplicate trades
        if action == 0:  # BUY
            if self.position <= 0:
                if self.position < 0:  # Cover short if needed
                    reward += (self.last_trade_price - current_close) * abs(self.position)
                    self.position = 0
                self.last_trade_price = current_close
                self.position = 1
        elif action == 1:  # SELL
            if self.position > 0:
                reward += (current_close - self.last_trade_price) * self.position
                self.position = 0
        elif action == 2:  # SHORT
            if self.position >= 0:
                if self.position > 0:  # Sell long if needed
                    reward += (current_close - self.last_trade_price) * self.position
                    self.position = 0
                self.last_trade_price = current_close
                self.position = -1
        elif action == 3:  # COVER
            if self.position < 0:
                reward += (self.last_trade_price - current_close) * abs(self.position)
                self.position = 0
        elif action == 4:  # HOLD / NONE
            reward = 0.0

        # Apply a transaction cost if a trade occurred
        if action in [0, 1, 2, 3]:
            reward -= self.transaction_cost * current_close

        # Additional reward for the position held over the period
        if self.position != 0:
            if self.position > 0:
                reward += (next_close - current_close)
            elif self.position < 0:
                reward += (current_close - next_close)

        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, done, {}

    def render(self, mode="human", close=False):
        profit = self.balance + (self.position * self.df.iloc[self.current_step]["close"]) - self.initial_balance
        print(f"Step: {self.current_step}, Balance: {self.balance}, Position: {self.position}, Profit: {profit}")


def create_rl_env(df, rf_model, rf_features):
    """
    Computes the predicted_price column using the Random Forest model,
    then creates the TradingEnv using the ENABLED_FEATURES and the predicted price.
    """
    df = df.copy()
    # Predict the close price from the RF model and add it as a column
    X_rf = df[rf_features]
    df["predicted_price"] = rf_model.predict(X_rf).astype(np.float32)
    # RL features: all ENABLED_FEATURES plus predicted_price (note: 'close' is included for reward calculation)
    feature_columns = ENABLED_FEATURES + ["predicted_price"]
    return TradingEnv(df, feature_columns)


def train_rl_model(env):
    """
    Train the RL model using PPO.
    Given our relatively small dataset (2810 rows), we train for a number of timesteps
    to help the agent adequately explore.
    """
    model = PPO("MlpPolicy", env, verbose=0, gamma=0.99, learning_rate=0.0003)
    total_timesteps = len(env.df) * 20  # 20 epochs over the dataset
    model.learn(total_timesteps=total_timesteps)
    return model


def run_logic(current_price, predicted_price, ticker):
    logging.info("Test number 2")
    """
    Live trading logic:
      1. Load CSV for the provided ticker (ticker_timeframe.csv).
      2. Compute predicted_price via a Random Forest model.
      3. Create and train the RL (PPO) model.
      4. Retrieve the current open position from the live API.
      5. Execute a trade (buy, sell, short, or cover) if appropriate.
    """
    
    # Load historical data
    df = load_data(ticker)
    # Train Random Forest model to predict close price
    rf_model, rf_features = train_rf_model(df)
    # Build the RL environment and include predicted_price
    env = create_rl_env(df, rf_model, rf_features)
    # Train the RL agent
    rl_model = train_rl_model(env)
    # Build the state from the last row using the updated DataFrame (which now includes "predicted_price")
    feature_columns = ENABLED_FEATURES + ["predicted_price"]
    current_state = env.df.iloc[-1][feature_columns].values.astype(np.float32)
    action, _ = rl_model.predict(current_state)
    
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
    elif action == 2 and position_qty >= 0:
        max_shares = int(cash // current_price)
        logging.info("short")
        short_shares(ticker, max_shares, current_price, predicted_price)
    elif action == 3 and position_qty < 0:
        qty_to_close = abs(position_qty)
        logging.info("cover")
        close_short(ticker, qty_to_close, current_price)
    # Action 4 is HOLD (do nothing)


def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    """
    Backtesting logic:
      1. Load CSV for the first ticker in TICKERS.
      2. Use only the last 500 candles (or all if less available) for backtesting.
      3. Compute predicted_price and build the RL environment.
      4. Train the RL (PPO) agent.
      5. Using current_timestamp, find the current state.
      6. Return one of: "BUY", "SELL", "SHORT", "COVER", or "NONE".
    """
    ticker = TICKERS[0]
    df = load_data(ticker)
    # Use the last 500 candles for backtesting
    if len(df) > 500:
        df_backtest = df.iloc[-500:].reset_index(drop=True)
    else:
        df_backtest = df.copy().reset_index(drop=True)
    
    # Train RF model on the backtest dataset
    rf_model, rf_features = train_rf_model(df_backtest)
    # Create the RL environment for backtesting
    env = create_rl_env(df_backtest, rf_model, rf_features)
    # Train the RL agent
    rl_model = train_rl_model(env)
    
    # Locate the row corresponding to the current timestamp using the updated DataFrame from the environment
    current_row = env.df[env.df["timestamp"] == pd.to_datetime(current_timestamp)]
    if current_row.empty:
        return "NONE"
    feature_columns = ENABLED_FEATURES + ["predicted_price"]
    current_state = current_row.iloc[0][feature_columns].values.astype(np.float32)
    action, _ = rl_model.predict(current_state)
    
    # Determine signal based on RL action and current position
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