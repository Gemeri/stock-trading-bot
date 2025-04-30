#!/usr/bin/env python3
"""
Algorithmic Trading System integrating:
- Genetic Algorithm (GA) for strategy optimization
- XGBoost for signal filtering
- PPO (Stable-Baselines3) for Reinforcement Learning
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
from forest import api, buy_shares, sell_shares, short_shares, close_short
from dotenv import load_dotenv
import os
import pandas as pd
load_dotenv()

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
    A simple backtest:
    - Computes a weighted signal from various normalized indicators.
    - Enters a long position if the signal exceeds a threshold.
    - Exits when the signal drops below negative threshold.

    Returns a fitness score based on cumulative return and Sharpe ratio.
    """
    initial_capital = 10000
    capital = initial_capital
    position = 0  # 1: long, 0: no position
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

        # Enter long if signal > threshold
        if position == 0 and signal > strategy.entry_threshold:
            position = 1
            entry_price = data.loc[data.index[i], 'open']
        # Exit if signal < -threshold
        elif position == 1 and signal < -strategy.entry_threshold:
            exit_price = data.loc[data.index[i], 'open']
            trade_return = (exit_price - entry_price) / entry_price
            returns.append(trade_return)
            capital *= (1 + trade_return * strategy.risk_reward)
            position = 0

    # Close any open position at end-of-data
    if position == 1:
        exit_price = data.iloc[-1]['close']
        trade_return = (exit_price - entry_price) / entry_price
        returns.append(trade_return)
        capital *= (1 + trade_return * strategy.risk_reward)

    cumulative_return = (capital - initial_capital) / initial_capital
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) if len(returns) > 0 else 0
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
    Target: Binary classification (1 if price up in next 5 periods, else 0).
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
    Use the trained XGBoost model to predict probability of price increase.
    """
    return model.predict_proba(features)[:, 1]

# -----------------------------
# Step 4: Reinforcement Learning Environment and Agent
# -----------------------------
class TradingEnv(gym.Env):
    """
    Custom Trading Environment for RL (using only TRAINING data).
    Observations: [close, ema_short, ema_long, macd, rsi, atr]
    Actions: 0 = Hold, 1 = Buy, 2 = Sell
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data, initial_capital=10000):
        super(TradingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.current_step = 0
        self.capital = initial_capital
        self.position = 0
        self.entry_price = 0
        self.trade_log = []

        # Observation space: 6 features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        # Action space: discrete(3)
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

        # Execute action
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = price
            self.trade_log.append((self.data.loc[self.current_step, 'timestamp'], 'BUY', price))
        elif action == 2 and self.position == 1:  # Sell
            self.position = 0
            exit_price = price
            trade_return = (exit_price - self.entry_price) / self.entry_price
            reward = trade_return
            self.capital *= (1 + trade_return)
            self.trade_log.append((self.data.loc[self.current_step, 'timestamp'], 'SELL', price, trade_return))

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True

        obs = self._next_observation() if not done else np.zeros(self.observation_space.shape)
        return obs, reward, done, {}

    def reset(self):
        self.current_step = 0
        self.capital = self.initial_capital
        self.position = 0
        self.trade_log = []
        return self._next_observation()

    def render(self, mode='human', close=False):
        logging.info(f"Step: {self.current_step}, Capital: {self.capital}, Position: {self.position}")

def train_rl_agent(train_data, timesteps=10000):
    """
    Train an RL agent (PPO) on the TRAINING set only.
    """
    env = TradingEnv(train_data)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=timesteps)
    logging.info("RL agent trained successfully.")
    return model, env

def run_logic(current_price, predicted_price, ticker):
    """
    Live trading logic:
      - Loads the CSV dynamically using .env settings (first ticker and timeframe suffix).
      - Filters out disabled features and sets sentiment to 0 if NEWS_MODE is off.
      - If a required column like 'rsi' was disabled, it is recomputed.
      - Trains GA, XGBoost, and RL models on the entire CSV.
      - Computes normalized indicators on the latest (current) candle.
      - Determines a combined GA+ML decision.
      - Retrieves the current position via the API and then calls buy_shares or sell_shares.
    """
    # Import required trading functions from the main module

    # Get .env variables
    bar_timeframe = os.getenv("BAR_TIMEFRAME", "1Hour")
    tickers_env = os.getenv("TICKERS", "TSLA")
    disabled_features = os.getenv("DISABLED_FEATURES", "")
    news_mode = os.getenv("NEWS_MODE", "on").lower()

    # Map BAR_TIMEFRAME to CSV filename suffix
    timeframe_map = {
        "4Hour": "H4",
        "2Hour": "H2",
        "1Hour": "H1",
        "30Min": "M30",
        "15Min": "M15"
    }
    suffix = timeframe_map.get(bar_timeframe, "H1")

    # Use the first ticker from TICKERS to load the CSV
    first_ticker = tickers_env.split(",")[0].strip()
    csv_file = f"{first_ticker}_{suffix}.csv"

    # Load and preprocess CSV data (the original function computes indicators if needed)
    df = load_and_preprocess_data(csv_file)

    # Filter out any disabled features
    if disabled_features:
        disabled_list = [feat.strip() for feat in disabled_features.split(",") if feat.strip()]
        for feat in disabled_list:
            if feat in df.columns:
                df.drop(columns=[feat], inplace=True)

    # Recompute 'rsi' if it was disabled but is needed
    if 'rsi' not in df.columns:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))

    # If NEWS_MODE is off, set sentiment to 0 regardless of CSV content
    if news_mode == "off":
        df['sentiment'] = 0.0

    # Train models on the entire CSV data
    best_strategy = run_genetic_algorithm(df, pop_size=30, generations=15)
    xgb_model = train_xgb_model(df)
    rl_model, _ = train_rl_agent(df, timesteps=20000)

    # Use the latest candle as the current trading data and override its price with current_price
    current_row = df.iloc[-1].copy()
    current_row['open'] = current_price
    current_row['close'] = current_price  # Use current_price for both open and close for decision purposes

    # Compute additional indicator: ema_diff
    current_row['ema_diff'] = current_row['ema_short'] - current_row['ema_long']

    # Compute normalized indicators (using entire series for rolling calculation where needed)
    norm_momentum = (current_row['momentum'] / 
                     (df['momentum'].abs().rolling(14).mean().iloc[-1] + 1e-10)
                     if 'momentum' in current_row else 0)
    norm_sentiment = current_row['sentiment']  # Already 0 if NEWS_MODE is off
    norm_atr = (current_row['atr'] / (current_price + 1e-10)
                if 'atr' in current_row else 0)
    norm_vwap = (current_row['vwap'] / (current_price + 1e-10)
                 if 'vwap' in current_row else 0)
    norm_ema_diff = (current_row['ema_diff'] / (current_price + 1e-10)
                     if current_price != 0 else 0)

    # Calculate GA weighted signal
    ga_signal = (best_strategy.momentum_weight * norm_momentum +
                 best_strategy.sentiment_weight * norm_sentiment +
                 best_strategy.atr_weight * norm_atr +
                 best_strategy.vwap_weight * norm_vwap +
                 best_strategy.ema_diff_weight * norm_ema_diff)

    # Prepare feature vector for XGBoost prediction
    feature_cols = ['momentum', 'atr', 'vwap', 'ema_short', 'ema_long', 'macd', 'rsi',
                    'bollinger_upper', 'bollinger_lower',
                    'lagged_close_1', 'lagged_close_2', 'lagged_close_3',
                    'lagged_close_5', 'lagged_close_10', 'sentiment']
    feature_data = {}
    for col in feature_cols:
        feature_data[col] = current_row[col] if col in current_row else 0.0
    features_df = pd.DataFrame([feature_data])
    prob_up = xgb_model.predict_proba(features_df)[:, 1][0]

    # Retrieve current position from API
    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception as e:
        logging.info(f"No open position for {ticker}: {e}")
        position_qty = 0.0

    # Determine action:
    # - If no open position and the GA signal is high and XGB probability > 0.6 → BUY (action 0)
    # - If in position and GA signal is low (or XGB probability < 0.4) → SELL (action 1)
    # - Otherwise, HOLD (action 4)
    if position_qty <= 0 and ga_signal > best_strategy.entry_threshold and prob_up > 0.6:
        action = 0  # BUY
    elif position_qty > 0 and (ga_signal < -best_strategy.entry_threshold or prob_up < 0.4):
        action = 1  # SELL
    else:
        action = 4  # HOLD

    # Get account cash and execute trade based on action
    account = api.get_account()
    cash = float(account.cash)
    if action == 0 and position_qty <= 0:
        max_shares = int(cash // current_price)
        logging.info("buy")
        buy_shares(ticker, max_shares, current_price, predicted_price)
    elif action == 1 and position_qty > 0:
        logging.info("sell")
        sell_shares(ticker, position_qty, current_price, predicted_price)
    else:
        logging.info("hold")


def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    """
    Backtesting logic:
      - Loads the CSV using .env settings (first ticker and timeframe suffix).
      - Filters out disabled features and sets sentiment to 0 if NEWS_MODE is off.
      - If a required column like 'rsi' was disabled, it is recomputed.
      - From the CSV, selects training data up to current_timestamp.
      - Trains GA, XGBoost, and RL models on this training set.
      - Uses the candle corresponding to current_timestamp (or the last one if not found)
        and overrides its price with current_price.
      - Computes normalized indicators and the GA weighted signal.
      - Prepares the feature vector for XGBoost.
      - Returns a trade decision string: "BUY" if conditions are met and no position exists,
        "SELL" if conditions are met and a long position exists, otherwise "NONE".
    """

    # Get .env variables
    bar_timeframe = os.getenv("BAR_TIMEFRAME", "1Hour")
    tickers_env = os.getenv("TICKERS", "TSLA")
    disabled_features = os.getenv("DISABLED_FEATURES", "")
    news_mode = os.getenv("NEWS_MODE", "on").lower()

    # Map BAR_TIMEFRAME to CSV filename suffix
    timeframe_map = {
        "4Hour": "H4",
        "2Hour": "2Hour",
        "1Hour": "1Hour",
        "30Min": "M30",
        "15Min": "15Min"
    }
    suffix = timeframe_map.get(bar_timeframe, "H4")

    # Use the first ticker from TICKERS to load the CSV
    first_ticker = tickers_env.split(",")[0].strip()
    csv_file = f"{first_ticker}_{suffix}.csv"

    # Load and preprocess CSV data
    df = load_and_preprocess_data(csv_file)

    # Filter out any disabled features
    if disabled_features:
        disabled_list = [feat.strip() for feat in disabled_features.split(",") if feat.strip()]
        for feat in disabled_list:
            if feat in df.columns:
                df.drop(columns=[feat], inplace=True)

    # Recompute 'rsi' if it was disabled but is needed
    if 'rsi' not in df.columns:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))

    # If NEWS_MODE is off, set sentiment to 0
    if news_mode == "off":
        df['sentiment'] = 0.0

    # Convert current_timestamp to a datetime object and filter training data up to that time
    current_ts = pd.to_datetime(current_timestamp)
    training_data = df[df['timestamp'] <= current_ts].copy()
    if training_data.empty:
        logging.info("No training data available up to current_timestamp")
        return "NONE"

    # Train models on training_data
    best_strategy = run_genetic_algorithm(training_data, pop_size=30, generations=15)
    xgb_model = train_xgb_model(training_data)
    rl_model, _ = train_rl_agent(training_data, timesteps=20000)

    # Retrieve the candle corresponding to current_timestamp; if not found, use the last available candle
    candle_df = training_data[training_data['timestamp'] == current_ts]
    if candle_df.empty:
        candle = training_data.iloc[-1].copy()
    else:
        candle = candle_df.iloc[0].copy()

    # Override the candle's price with current_price
    candle['open'] = current_price
    candle['close'] = current_price

    # Compute additional indicator: ema_diff
    candle['ema_diff'] = candle['ema_short'] - candle['ema_long']

    # Compute normalized indicators using training_data for rolling calculation
    norm_momentum = (candle['momentum'] / 
                     (training_data['momentum'].abs().rolling(14).mean().iloc[-1] + 1e-10)
                     if 'momentum' in candle else 0)
    norm_sentiment = candle['sentiment']
    norm_atr = (candle['atr'] / (current_price + 1e-10)
                if 'atr' in candle else 0)
    norm_vwap = (candle['vwap'] / (current_price + 1e-10)
                 if 'vwap' in candle else 0)
    norm_ema_diff = (candle['ema_diff'] / (current_price + 1e-10)
                     if current_price != 0 else 0)

    # Calculate GA weighted signal
    ga_signal = (best_strategy.momentum_weight * norm_momentum +
                 best_strategy.sentiment_weight * norm_sentiment +
                 best_strategy.atr_weight * norm_atr +
                 best_strategy.vwap_weight * norm_vwap +
                 best_strategy.ema_diff_weight * norm_ema_diff)

    # Prepare feature vector for XGBoost prediction
    feature_cols = ['momentum', 'atr', 'vwap', 'ema_short', 'ema_long', 'macd', 'rsi',
                    'bollinger_upper', 'bollinger_lower',
                    'lagged_close_1', 'lagged_close_2', 'lagged_close_3',
                    'lagged_close_5', 'lagged_close_10', 'sentiment']
    feature_data = {}
    for col in feature_cols:
        feature_data[col] = candle[col] if col in candle else 0.0
    features_df = pd.DataFrame([feature_data])
    prob_up = xgb_model.predict_proba(features_df)[:, 1][0]

    # Determine action based on the trained models and current indicators:
    # - If no open position and conditions for long are met → BUY (action 0)
    # - If in position and conditions for exit are met → SELL (action 1)
    # - Otherwise, do nothing.
    if position_qty <= 0 and ga_signal > best_strategy.entry_threshold and prob_up > 0.6:
        action = 0  # BUY
    elif position_qty > 0 and (ga_signal < -best_strategy.entry_threshold or prob_up < 0.4):
        action = 1  # SELL
    else:
        action = 4  # HOLD

    # Return a string corresponding to the decided action
    if action == 0 and position_qty <= 0:
        return "BUY"
    elif action == 1 and position_qty > 0:
        return "SELL"
    else:
        return "NONE"
