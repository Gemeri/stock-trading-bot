#!/usr/bin/env python3
"""
Algorithmic Trading System integrating:
- Genetic Algorithm (GA) for strategy optimization (now supports long & short)
- XGBoost for signal filtering
- PPO (Stable-Baselines3) for Reinforcement Learning (now supports long & short)
- Proper train/test split: Train on first 80%, backtest on last 20%.

Ensure you have:
    pandas, numpy, matplotlib, xgboost, scikit-learn, gym, stable-baselines3
"""

import os
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from collections import deque

# XGBoost for predictions and sklearn for splitting and accuracy
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Reinforcement Learning libraries
import gym
from gym import spaces
from stable_baselines3 import PPO

# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# -----------------------------
# Step 1: Data Loading and Preprocessing
# -----------------------------
def load_and_preprocess_data(csv_file):
    """
    Load CSV file and preprocess data:
    - Convert 'timestamp' to datetime and sort chronologically.
    - Compute additional technical indicators: EMA (short & long), MACD, RSI.
    - Calculate ATR-based stop-loss and take-profit levels.
    
    Expected CSV columns (at minimum):
        timestamp, open, high, low, close, vwap, momentum, atr, obv,
        bollinger_upper, bollinger_lower, lagged_close_1, lagged_close_2,
        lagged_close_3, lagged_close_5, lagged_close_10, sentiment
    """
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        logging.error(f"Error loading CSV file: {e}")
        raise

    # Convert timestamp to datetime and sort
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Exponential Moving Averages
    df['ema_short'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=50, adjust=False).mean()

    # MACD: Difference between 12 and 26 period EMAs
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']

    # RSI calculation (14-period)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR: Recompute if not provided or if it contains nulls
    if 'atr' not in df.columns or df['atr'].isnull().all():
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()

    # ATR-based stop-loss / take-profit (demo approach)
    df['stop_loss'] = df['close'] - df['atr']
    df['take_profit'] = df['close'] + df['atr']

    # Fill any remaining NaNs
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    logging.info("Data loaded and preprocessed successfully.")
    return df

# -----------------------------
# Step 2: Genetic Algorithm (GA) for Strategy Optimization
# -----------------------------
class TradingStrategy:
    """
    Container for trading strategy parameters.
    """
    def __init__(self, momentum_weight, sentiment_weight, atr_weight, vwap_weight,
                 ema_diff_weight, risk_reward, entry_threshold):
        self.momentum_weight = momentum_weight
        self.sentiment_weight = sentiment_weight
        self.atr_weight = atr_weight
        self.vwap_weight = vwap_weight
        self.ema_diff_weight = ema_diff_weight
        self.risk_reward = risk_reward
        self.entry_threshold = entry_threshold

    def to_dict(self):
        return {
            'momentum_weight': self.momentum_weight,
            'sentiment_weight': self.sentiment_weight,
            'atr_weight': self.atr_weight,
            'vwap_weight': self.vwap_weight,
            'ema_diff_weight': self.ema_diff_weight,
            'risk_reward': self.risk_reward,
            'entry_threshold': self.entry_threshold
        }

def initialize_population(pop_size):
    """
    Generate an initial population of random trading strategies.
    """
    population = []
    for _ in range(pop_size):
        strategy = TradingStrategy(
            momentum_weight = random.uniform(0, 1),
            sentiment_weight = random.uniform(0, 1),
            atr_weight = random.uniform(0, 1),
            vwap_weight = random.uniform(0, 1),
            ema_diff_weight = random.uniform(0, 1),
            risk_reward = random.uniform(1, 3),
            entry_threshold = random.uniform(0.1, 1)
        )
        population.append(strategy)
    return population

def backtest_strategy(strategy, data):
    """
    A backtest that supports both long and short positions:
    - If signal > threshold and position == 0, open long.
    - If signal < -threshold and position == 0, open short.
    - If position == 1 (long) and signal < -threshold, close long.
    - If position == -1 (short) and signal > threshold, close short.

    Returns a fitness score (cumulative return + Sharpe ratio).
    """
    initial_capital = 10000
    capital = initial_capital
    position = 0  # 1: long, -1: short, 0: flat
    entry_price = 0
    returns = []

    data = data.copy()
    data['ema_diff'] = data['ema_short'] - data['ema_long']

    # Normalize indicators (demo approach)
    data['norm_momentum'] = data['momentum'] / (data['momentum'].abs().rolling(14).mean() + 1e-10)
    data['norm_sentiment'] = data['sentiment']
    data['norm_atr'] = data['atr'] / (data['close'] + 1e-10)
    data['norm_vwap'] = data['vwap'] / (data['close'] + 1e-10)
    data['norm_ema_diff'] = data['ema_diff'] / (data['close'] + 1e-10)

    for i in range(1, len(data)):
        signal = (
            strategy.momentum_weight * data.loc[data.index[i], 'norm_momentum'] +
            strategy.sentiment_weight * data.loc[data.index[i], 'norm_sentiment'] +
            strategy.atr_weight * data.loc[data.index[i], 'norm_atr'] +
            strategy.vwap_weight * data.loc[data.index[i], 'norm_vwap'] +
            strategy.ema_diff_weight * data.loc[data.index[i], 'norm_ema_diff']
        )

        # --- Long Entry ---
        if position == 0 and signal > strategy.entry_threshold:
            position = 1
            entry_price = data.loc[data.index[i], 'open']

        # --- Short Entry ---
        elif position == 0 and signal < -strategy.entry_threshold:
            position = -1
            entry_price = data.loc[data.index[i], 'open']

        # --- Close Long ---
        elif position == 1 and signal < -strategy.entry_threshold:
            exit_price = data.loc[data.index[i], 'open']
            trade_return = (exit_price - entry_price) / entry_price  # long exit
            returns.append(trade_return)
            capital *= (1 + trade_return * strategy.risk_reward)
            position = 0

        # --- Close Short ---
        elif position == -1 and signal > strategy.entry_threshold:
            exit_price = data.loc[data.index[i], 'open']
            # short exit: if we shorted at entry_price, profit is (entry_price - exit_price)/entry_price
            trade_return = (entry_price - exit_price) / entry_price
            returns.append(trade_return)
            capital *= (1 + trade_return * strategy.risk_reward)
            position = 0

    # Close any open position at end-of-data
    if position == 1:
        # close the long at final close
        exit_price = data.iloc[-1]['close']
        trade_return = (exit_price - entry_price) / entry_price
        returns.append(trade_return)
        capital *= (1 + trade_return * strategy.risk_reward)

    elif position == -1:
        # close the short at final close
        exit_price = data.iloc[-1]['close']
        trade_return = (entry_price - exit_price) / entry_price
        returns.append(trade_return)
        capital *= (1 + trade_return * strategy.risk_reward)

    cumulative_return = (capital - initial_capital) / initial_capital
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) if returns else 0
    fitness = cumulative_return + sharpe_ratio
    return fitness

def tournament_selection(population, fitnesses, tournament_size=3):
    """
    Select the best strategy from a random subset (tournament) of the population.
    """
    selected = random.sample(list(zip(population, fitnesses)), tournament_size)
    selected = sorted(selected, key=lambda x: x[1], reverse=True)
    return selected[0][0]

def uniform_crossover(parent1, parent2):
    """
    Create a child strategy by randomly choosing each parameter from one of the two parents.
    """
    child_params = {}
    for param in parent1.to_dict().keys():
        child_params[param] = random.choice([parent1.to_dict()[param], parent2.to_dict()[param]])
    return TradingStrategy(**child_params)

def adaptive_mutation(strategy, mutation_rate=0.1):
    """
    Mutate strategy parameters with a small random noise.
    For 'risk_reward', ensure a minimum value of 1.
    """
    params = strategy.to_dict()
    mutated_params = {}
    for key, value in params.items():
        if random.random() < mutation_rate:
            noise = random.uniform(-0.1, 0.1)
            if key == 'risk_reward':
                mutated_params[key] = max(1, value + noise)
            else:
                mutated_params[key] = max(0, value + noise)
        else:
            mutated_params[key] = value
    return TradingStrategy(**mutated_params)

def run_genetic_algorithm(data, pop_size=20, generations=10):
    """
    Runs the Genetic Algorithm over multiple generations using only the TRAINING data.
    Returns the best evolved trading strategy.
    """
    population = initialize_population(pop_size)
    best_strategy = None
    best_fitness = -np.inf

    for gen in range(generations):
        logging.info(f"GA Generation {gen+1}")
        fitnesses = [backtest_strategy(strategy, data) for strategy in population]

        for strat, fit in zip(population, fitnesses):
            if fit > best_fitness:
                best_fitness = fit
                best_strategy = strat

        # Evolve
        new_population = []
        for _ in range(pop_size):
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child = uniform_crossover(parent1, parent2)
            child = adaptive_mutation(child)
            new_population.append(child)
        population = new_population

    logging.info(f"Best GA Strategy: {best_strategy.to_dict()}, Fitness: {best_fitness}")
    return best_strategy

# -----------------------------
# Step 3: XGBoost Model for Signal Filtering
# -----------------------------
def train_xgb_model(train_data):
    """
    Train an XGBoost model on the TRAINING set only.
    Target: Binary classification (1 if price up in next 5 candles, else 0).
    """
    df = train_data.copy()

    # Create target: 1 if price up in next 5 candles, else 0
    df['future_close'] = df['close'].shift(-5)
    df.dropna(inplace=True)
    df['target'] = (df['future_close'] > df['close']).astype(int)

    feature_cols = [
        'momentum', 'atr', 'vwap', 'ema_short', 'ema_long',
        'macd', 'rsi', 'bollinger_upper', 'bollinger_lower',
        'lagged_close_1', 'lagged_close_2', 'lagged_close_3',
        'lagged_close_5', 'lagged_close_10', 'sentiment'
    ]

    X = df[feature_cols]
    y = df['target']

    # We do an internal train_test_split for in-sample model evaluation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    logging.info(f"XGBoost model trained (in-sample). Validation Accuracy: {acc:.2f}")

    return model

def xgb_predict(model, features):
    """
    Use the trained XGBoost model to predict the probability of a price increase.
    """
    return model.predict_proba(features)[:, 1]

# -----------------------------
# Step 4: Reinforcement Learning Environment and Agent
# -----------------------------
class TradingEnv(gym.Env):
    """
    Custom Trading Environment for RL (using only TRAINING data).
    Observations: [close, ema_short, ema_long, macd, rsi, atr]
    Actions: 0 = Hold, 1 = Buy/Cover, 2 = Sell/Short

    position:
      0 = flat
      1 = long
     -1 = short
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data, initial_capital=10000):
        super(TradingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.current_step = 0
        self.capital = initial_capital
        self.position = 0    # 0=flat, 1=long, -1=short
        self.entry_price = 0
        self.trade_log = []

        # Observation space: 6 features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        # Action space: 0=Hold, 1=Buy/Cover, 2=Sell/Short
        self.action_space = spaces.Discrete(3)

    def _next_observation(self):
        obs = np.array([
            self.data.loc[self.current_step, 'close'],
            self.data.loc[self.current_step, 'ema_short'],
            self.data.loc[self.current_step, 'ema_long'],
            self.data.loc[self.current_step, 'macd'],
            self.data.loc[self.current_step, 'rsi'],
            self.data.loc[self.current_step, 'atr']
        ], dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0)
        return obs

    def step(self, action):
        done = False
        reward = 0
        price = self.data.loc[self.current_step, 'open']

        # Action interpretation:
        # 0 = hold
        # 1 = buy/cover
        # 2 = sell/short

        if action == 1:
            # If flat, open long
            if self.position == 0:
                self.position = 1
                self.entry_price = price
                self.trade_log.append((self.data.loc[self.current_step, 'timestamp'], 'BUY', price))
            # If short, cover short
            elif self.position == -1:
                exit_price = price
                trade_return = (self.entry_price - exit_price) / self.entry_price  # short exit
                reward = trade_return
                self.capital *= (1 + trade_return)
                self.trade_log.append((self.data.loc[self.current_step, 'timestamp'], 'COVER', price, trade_return))
                self.position = 0

        elif action == 2:
            # If flat, open short
            if self.position == 0:
                self.position = -1
                self.entry_price = price
                self.trade_log.append((self.data.loc[self.current_step, 'timestamp'], 'SHORT', price))
            # If long, close long
            elif self.position == 1:
                exit_price = price
                trade_return = (exit_price - self.entry_price) / self.entry_price  # long exit
                reward = trade_return
                self.capital *= (1 + trade_return)
                self.trade_log.append((self.data.loc[self.current_step, 'timestamp'], 'SELL', price, trade_return))
                self.position = 0

        # action == 0 => hold, do nothing

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True
            # Close any open position automatically at the end
            if self.position == 1:
                # close the long
                exit_price = self.data.loc[self.current_step, 'close']
                trade_return = (exit_price - self.entry_price) / self.entry_price
                self.capital *= (1 + trade_return)
                self.trade_log.append((self.data.loc[self.current_step, 'timestamp'], 'SELL-END', exit_price, trade_return))
                self.position = 0

            elif self.position == -1:
                # cover the short
                exit_price = self.data.loc[self.current_step, 'close']
                trade_return = (self.entry_price - exit_price) / self.entry_price
                self.capital *= (1 + trade_return)
                self.trade_log.append((self.data.loc[self.current_step, 'timestamp'], 'COVER-END', exit_price, trade_return))
                self.position = 0

        obs = self._next_observation() if not done else np.zeros(self.observation_space.shape)
        return obs, reward, done, {}

    def reset(self):
        self.current_step = 0
        self.capital = self.initial_capital
        self.position = 0
        self.entry_price = 0
        self.trade_log = []
        return self._next_observation()

    def render(self, mode='human', close=False):
        logging.info(f"Step: {self.current_step}, Capital: {self.capital}, Position: {self.position}")

def train_rl_agent(train_data, timesteps=10000):
    """
    Train an RL agent (PPO) on the TRAINING set only, supporting long & short.
    """
    env = TradingEnv(train_data)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=timesteps)
    logging.info("RL agent trained successfully.")
    return model, env

def run_logic(current_price, predicted_price, ticker):
    """
    Live trading function.
    Trains GA, XGBoost, and RL on the ENTIRE CSV (after loading and processing based on .env settings)
    and then computes a trading signal from the latest candle to decide which trade action to execute.
    Trade orders are executed via the imported functions from forest:
        api, buy_shares, sell_shares, short_shares, close_short.
    """
    import os
    import logging
    import pandas as pd
    import numpy as np
    from dotenv import load_dotenv
    from forest import api, buy_shares, sell_shares, short_shares, close_short

    load_dotenv()  # load .env variables

    # Retrieve .env variables
    bar_timeframe = os.getenv("BAR_TIMEFRAME", "1Hour")
    tickers_env = os.getenv("TICKERS", "TSLA")
    disabled_features = os.getenv("DISABLED_FEATURES", "")
    news_mode = os.getenv("NEWS_MODE", "on").lower()  # "on" or "off"

    # Convert BAR_TIMEFRAME to CSV suffix
    timeframe_map = {
        "4Hour": "H4",
        "2Hour": "H2",
        "1Hour": "H1",
        "30Min": "M30",
        "15Min": "M15"
    }
    timeframe_suffix = timeframe_map.get(bar_timeframe, "H1")

    # Use the first ticker from TICKERS to determine CSV filename
    first_ticker = tickers_env.split(",")[0].strip()
    csv_filename = f"{first_ticker}_{timeframe_suffix}.csv"

    # Load and preprocess the full CSV data
    df = load_and_preprocess_data(csv_filename)

    # Filter out disabled features BUT ensure essential columns are kept.
    essential_cols = [
        "ema_short", "ema_long", "macd", "rsi", "atr", "vwap",
        "momentum", "bollinger_upper", "bollinger_lower",
        "lagged_close_1", "lagged_close_2", "lagged_close_3",
        "lagged_close_5", "lagged_close_10"
    ]
    disabled_list = [feat.strip() for feat in disabled_features.split(",") if feat.strip()]
    for col in disabled_list:
        if col.lower() == "sentiment":
            if news_mode == "off" and col in df.columns:
                df.drop(columns=[col], inplace=True)
        elif col.lower() not in [e.lower() for e in essential_cols]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    # Train models on the entire dataset (live training)
    best_strategy = run_genetic_algorithm(df, pop_size=30, generations=15)
    xgb_model = train_xgb_model(df)
    rl_model, rl_env = train_rl_agent(df, timesteps=20000)

    # Compute normalized indicators for the latest candle (using the last row of df)
    latest = df.iloc[-1].copy()
    latest["ema_diff"] = latest["ema_short"] - latest["ema_long"]

    # Normalize indicators (use rolling calculations from the full df if available)
    if "momentum" in df.columns:
        roll_mean = df["momentum"].abs().rolling(14).mean()
        norm_momentum = df["momentum"].iloc[-1] / (roll_mean.iloc[-1] + 1e-10)
    else:
        norm_momentum = 0
    norm_sentiment = latest["sentiment"] if ("sentiment" in latest and news_mode != "off") else 0
    norm_atr = latest["atr"] / (latest["close"] + 1e-10) if "atr" in latest else 0
    norm_vwap = latest["vwap"] / (latest["close"] + 1e-10) if "vwap" in latest else 0
    norm_ema_diff = latest["ema_diff"] / (latest["close"] + 1e-10)

    # Compute GA signal using best_strategy parameters
    ga_signal = (
        best_strategy.momentum_weight * norm_momentum +
        best_strategy.sentiment_weight * norm_sentiment +
        best_strategy.atr_weight * norm_atr +
        best_strategy.vwap_weight * norm_vwap +
        best_strategy.ema_diff_weight * norm_ema_diff
    )

    # Prepare features for XGBoost prediction from the latest candle.
    # Explicitly convert to float to ensure proper dtypes.
    feature_cols = [
        "momentum", "atr", "vwap", "ema_short", "ema_long",
        "macd", "rsi", "bollinger_upper", "bollinger_lower",
        "lagged_close_1", "lagged_close_2", "lagged_close_3",
        "lagged_close_5", "lagged_close_10"
    ]
    if news_mode != "off":
        feature_cols.append("sentiment")
    xgb_features = latest[feature_cols].to_frame().T.astype(float)
    prob_up = xgb_model.predict_proba(xgb_features)[:, 1][0]

    # Determine trade action based on GA signal and XGB probability.
    # Conditions mirror those used in backtesting:
    #   - If flat (position_qty == 0) and ga_signal > entry_threshold and prob_up > 0.6: BUY.
    #   - If long (position_qty > 0) and (ga_signal < -entry_threshold or prob_up < 0.4): SELL.
    #   - If flat (position_qty == 0) and ga_signal < -entry_threshold and prob_up < 0.4: SHORT.
    #   - If short (position_qty < 0) and (ga_signal > entry_threshold or prob_up > 0.6): COVER.
    action = 4  # Default is HOLD (no trade)

    # Get live position from the trading API
    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception as e:
        logging.info(f"No open position for {ticker}: {e}")
        position_qty = 0.0

    account = api.get_account()
    cash = float(account.cash)

    if position_qty == 0:
        if ga_signal > best_strategy.entry_threshold and prob_up > 0.6:
            action = 0  # BUY
        elif ga_signal < -best_strategy.entry_threshold and prob_up < 0.4:
            action = 2  # SHORT
    elif position_qty > 0:
        if ga_signal < -best_strategy.entry_threshold or prob_up < 0.4:
            action = 1  # SELL
    elif position_qty < 0:
        if ga_signal > best_strategy.entry_threshold or prob_up > 0.6:
            action = 3  # COVER

    # Execute trade based on the determined action, preventing duplicate trades.
    if action == 0 and position_qty <= 0:
        max_shares = int(cash // current_price)
        logging.info("Executing BUY order")
        buy_shares(ticker, max_shares, current_price, predicted_price)
    elif action == 1 and position_qty > 0:
        logging.info("Executing SELL order")
        sell_shares(ticker, position_qty, current_price, predicted_price)
    elif action == 2 and position_qty >= 0:
        max_shares = int(cash // current_price)
        logging.info("Executing SHORT order")
        short_shares(ticker, max_shares, current_price, predicted_price)
    elif action == 3 and position_qty < 0:
        qty_to_close = abs(position_qty)
        logging.info("Executing COVER order")
        close_short(ticker, qty_to_close, current_price)
    else:
        logging.info("No trade action taken")


def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    """
    Backtest trading decision for a given candle.
    Trains GA, XGBoost, and RL on the training data from the CSV up to the provided current_timestamp,
    then computes the signal on the current backtest candle (from the candles dataframe) and
    returns a trade decision as a string ("BUY", "SELL", "SHORT", "COVER", or "NONE").
    """
    import os
    import logging
    import pandas as pd
    import numpy as np
    from dotenv import load_dotenv
    from forest import api, buy_shares, sell_shares, short_shares, close_short

    load_dotenv()

    # Retrieve .env variables
    bar_timeframe = os.getenv("BAR_TIMEFRAME", "1Hour")
    tickers_env = os.getenv("TICKERS", "TSLA")
    disabled_features = os.getenv("DISABLED_FEATURES", "")
    news_mode = os.getenv("NEWS_MODE", "on").lower()

    # Convert BAR_TIMEFRAME to CSV suffix
    timeframe_map = {
        "4Hour": "H4",
        "2Hour": "H2",
        "1Hour": "H1",
        "30Min": "M30",
        "15Min": "M15"
    }
    timeframe_suffix = timeframe_map.get(bar_timeframe, "H1")

    first_ticker = tickers_env.split(",")[0].strip()
    csv_filename = f"{first_ticker}_{timeframe_suffix}.csv"

    # Load full CSV training data and preprocess
    df_full = load_and_preprocess_data(csv_filename)

    # Filter out disabled features BUT ensure essential columns are kept.
    essential_cols = [
        "ema_short", "ema_long", "macd", "rsi", "atr", "vwap",
        "momentum", "bollinger_upper", "bollinger_lower",
        "lagged_close_1", "lagged_close_2", "lagged_close_3",
        "lagged_close_5", "lagged_close_10"
    ]
    disabled_list = [feat.strip() for feat in disabled_features.split(",") if feat.strip()]
    for col in disabled_list:
        if col.lower() == "sentiment":
            if news_mode == "off" and col in df_full.columns:
                df_full.drop(columns=[col], inplace=True)
        elif col.lower() not in [e.lower() for e in essential_cols]:
            if col in df_full.columns:
                df_full.drop(columns=[col], inplace=True)

    # Ensure current_timestamp is a datetime object
    if not isinstance(current_timestamp, pd.Timestamp):
        current_timestamp = pd.to_datetime(current_timestamp)

    # Use all training data up to the current candle timestamp
    train_data = df_full[df_full["timestamp"] <= current_timestamp].copy()
    if train_data.empty:
        logging.info("No training data available up to the current timestamp.")
        return "NONE"

    # Train models on the training data
    best_strategy = run_genetic_algorithm(train_data, pop_size=30, generations=15)
    xgb_model = train_xgb_model(train_data)
    rl_model, rl_env = train_rl_agent(train_data, timesteps=20000)

    # Identify the current backtest candle from the provided candles dataframe.
    if "timestamp" in candles.columns:
        candle_row = candles[candles["timestamp"] == current_timestamp]
        if candle_row.empty:
            candle_row = candles.iloc[[-1]]
        else:
            candle_row = candle_row.iloc[[0]]
    else:
        candle_row = candles.iloc[[-1]]
    # Create an explicit copy to avoid SettingWithCopyWarning
    current_candle = candle_row.squeeze().copy()

    # If essential computed columns are missing, populate them using training data.
    if "ema_short" not in current_candle or "ema_long" not in current_candle:
        current_candle["ema_short"] = train_data["close"].ewm(span=10, adjust=False).mean().iloc[-1]
        current_candle["ema_long"] = train_data["close"].ewm(span=50, adjust=False).mean().iloc[-1]
    current_candle["ema_diff"] = current_candle["ema_short"] - current_candle["ema_long"]

    if "macd" not in current_candle:
        current_candle["macd"] = (train_data["ema12"].iloc[-1] - train_data["ema26"].iloc[-1]
                                  if "ema12" in train_data.columns and "ema26" in train_data.columns
                                  else 0)
    if "rsi" not in current_candle:
        current_candle["rsi"] = train_data["rsi"].iloc[-1] if "rsi" in train_data.columns else 0

    if "momentum" in train_data.columns:
        roll_mean = train_data["momentum"].abs().rolling(14).mean()
        norm_momentum = train_data["momentum"].iloc[-1] / (roll_mean.iloc[-1] + 1e-10)
    else:
        norm_momentum = 0
    norm_sentiment = current_candle["sentiment"] if ("sentiment" in current_candle and news_mode != "off") else 0
    norm_atr = current_candle["atr"] / (current_candle["close"] + 1e-10) if "atr" in current_candle else 0
    norm_vwap = current_candle["vwap"] / (current_candle["close"] + 1e-10) if "vwap" in current_candle else 0
    norm_ema_diff = current_candle["ema_diff"] / (current_candle["close"] + 1e-10)

    ga_signal = (
        best_strategy.momentum_weight * norm_momentum +
        best_strategy.sentiment_weight * norm_sentiment +
        best_strategy.atr_weight * norm_atr +
        best_strategy.vwap_weight * norm_vwap +
        best_strategy.ema_diff_weight * norm_ema_diff
    )

    # Prepare feature set for XGBoost prediction from the current candle.
    # Convert features to float to satisfy XGBoost requirements.
    feature_cols = [
        "momentum", "atr", "vwap", "ema_short", "ema_long",
        "macd", "rsi", "bollinger_upper", "bollinger_lower",
        "lagged_close_1", "lagged_close_2", "lagged_close_3",
        "lagged_close_5", "lagged_close_10"
    ]
    if news_mode != "off":
        feature_cols.append("sentiment")
    xgb_features = current_candle[feature_cols].to_frame().T.astype(float)
    prob_up = xgb_model.predict_proba(xgb_features)[:, 1][0]

    # Determine trade action based on computed signal and probability.
    action = 4  # Default HOLD
    if position_qty == 0:
        if ga_signal > best_strategy.entry_threshold and prob_up > 0.6:
            action = 0  # BUY
        elif ga_signal < -best_strategy.entry_threshold and prob_up < 0.4:
            action = 2  # SHORT
    elif position_qty > 0:
        if ga_signal < -best_strategy.entry_threshold or prob_up < 0.4:
            action = 1  # SELL
    elif position_qty < 0:
        if ga_signal > best_strategy.entry_threshold or prob_up > 0.6:
            action = 3  # COVER

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
