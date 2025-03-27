#!/usr/bin/env python3
"""
FULL UPDATED SCRIPT with extra data inspection, robust normalization/clipping,
reward scaling, additional safeguards in the trading environment (including capping
the number of shares when price is near zero), and PPO gradient clipping/adjusted learning rate.
Logs are sent to both the console and a log file.
"""

import os
import random
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler

import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# --------------------
# Warning Filters & Numpy error behavior
# --------------------
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in cast")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")
# Optionally ignore all RuntimeWarnings:
# warnings.simplefilter("ignore", RuntimeWarning)
np.seterr(over='ignore', invalid='ignore')

# --------------------
# Logging Configuration: both console and file
# --------------------
log_filename = "console_output.txt"
logging.basicConfig(
    level=logging.INFO,  # Use DEBUG for extra details
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, mode='a')
    ]
)

# --------------------
# Data Inspection Function
# --------------------
def inspect_data(df, columns, prefix=""):
    for col in columns:
        try:
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            std_val = df[col].std()
            num_nan = df[col].isna().sum()
            logging.info(f"{prefix} {col}: min={min_val}, max={max_val}, mean={mean_val}, std={std_val}, nans={num_nan}")
        except Exception as e:
            logging.error(f"Error inspecting {col}: {e}")

# --------------------
# Utility Functions
# --------------------
def convert_timeframe(bar_timeframe):
    timeframe_map = {
        "4Hour": "H4",
        "2Hour": "H2",
        "1Hour": "H1",
        "30Min": "M30",
        "15Min": "M15"
    }
    return timeframe_map.get(bar_timeframe, "H1")

def get_enabled_features(df, disabled_features, news_mode):
    # If "sentiment" is disabled (or news_mode is off), drop it.
    for col in disabled_features:
        if col.lower() == "sentiment":
            if not news_mode and col in df.columns:
                df.drop(columns=[col], inplace=True)
        elif col.lower() != "rsi" and col in df.columns:
            df.drop(columns=[col], inplace=True)
    return df

# --------------------
# Data Loading & Preprocessing
# --------------------
def load_and_preprocess_data(csv_file, disabled_features=[], news_mode=True):
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        logging.error(f"Error loading CSV file: {e}")
        raise

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Compute technical indicators (example set)
    df['ema_short'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']

    # RSI calculation
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR calculation if not already present
    if 'atr' not in df.columns or df['atr'].isnull().all():
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()

    df['stop_loss'] = df['close'] - df['atr']
    df['take_profit'] = df['close'] + df['atr']

    df = get_enabled_features(df, disabled_features, news_mode)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    # Add predicted_price using regression (see below)
    df = add_predicted_price(df, [col for col in df.columns if col != 'timestamp'])
    return df

def add_predicted_price(df, ml_features):
    ml_model_choice = os.getenv('ML_MODEL', 'forest').lower()
    df = df.copy()
    df['target_close'] = df['close'].shift(-1)
    df_model = df.dropna(subset=['target_close'])
    X = df_model[ml_features]
    y = df_model['target_close']
    if ml_model_choice == 'xgboost':
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    else:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    predicted_prices = model.predict(df[ml_features])
    df['predicted_price'] = predicted_prices
    df.drop(columns=['target_close'], inplace=True)
    return df

# --------------------
# Genetic Algorithm for Strategy Optimization
# --------------------
class TradingStrategy:
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
    population = []
    for _ in range(pop_size):
        strat = TradingStrategy(
            momentum_weight=random.uniform(0, 1),
            sentiment_weight=random.uniform(0, 1),
            atr_weight=random.uniform(0, 1),
            vwap_weight=random.uniform(0, 1),
            ema_diff_weight=random.uniform(0, 1),
            risk_reward=random.uniform(1, 3),
            entry_threshold=random.uniform(0.1, 1)
        )
        population.append(strat)
    return population

def backtest_strategy(strategy, data):
    initial_capital = 10000
    capital = initial_capital
    position = 0  # positive for long, negative for short
    entry_price = 0
    returns = []
    data = data.copy()
    data['ema_diff'] = data['ema_short'] - data['ema_long']
    data['norm_momentum'] = data['momentum'] / (data['momentum'].abs().rolling(14).mean() + 1e-10)
    data['norm_sentiment'] = data['sentiment'] if 'sentiment' in data.columns else 0
    data['norm_atr'] = data['atr'] / (data['close'] + 1e-10)
    data['norm_vwap'] = data['vwap'] / (data['close'] + 1e-10)
    data['norm_ema_diff'] = data['ema_diff'] / (data['close'] + 1e-10)

    for i in range(1, len(data)):
        signal = (strategy.momentum_weight * data.loc[data.index[i], 'norm_momentum'] +
                  strategy.sentiment_weight * data.loc[data.index[i], 'norm_sentiment'] +
                  strategy.atr_weight * data.loc[data.index[i], 'norm_atr'] +
                  strategy.vwap_weight * data.loc[data.index[i], 'norm_vwap'] +
                  strategy.ema_diff_weight * data.loc[data.index[i], 'norm_ema_diff'])
        if position == 0 and signal > strategy.entry_threshold:
            position = 1
            entry_price = data.loc[data.index[i], 'open']
        elif position == 0 and signal < -strategy.entry_threshold:
            position = -1
            entry_price = data.loc[data.index[i], 'open']
        elif position == 1 and signal < -strategy.entry_threshold:
            exit_price = data.loc[data.index[i], 'open']
            trade_return = (exit_price - entry_price) / entry_price
            returns.append(trade_return)
            capital *= (1 + trade_return * strategy.risk_reward)
            position = 0
        elif position == -1 and signal > strategy.entry_threshold:
            exit_price = data.loc[data.index[i], 'open']
            trade_return = (entry_price - exit_price) / entry_price
            returns.append(trade_return)
            capital *= (1 + trade_return * strategy.risk_reward)
            position = 0

    # Close any open position at the end
    if position == 1:
        exit_price = data.iloc[-1]['close']
        trade_return = (exit_price - entry_price) / entry_price
        returns.append(trade_return)
        capital *= (1 + trade_return * strategy.risk_reward)
    elif position == -1:
        exit_price = data.iloc[-1]['close']
        trade_return = (entry_price - exit_price) / entry_price
        returns.append(trade_return)
        capital *= (1 + trade_return * strategy.risk_reward)

    cumulative_return = (capital - initial_capital) / initial_capital
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) if returns else 0
    fitness = cumulative_return + sharpe_ratio
    return fitness

def tournament_selection(population, fitnesses, tournament_size=3):
    selected = random.sample(list(zip(population, fitnesses)), tournament_size)
    selected = sorted(selected, key=lambda x: x[1], reverse=True)
    return selected[0][0]

def uniform_crossover(parent1, parent2):
    child_params = {}
    for param in parent1.to_dict().keys():
        child_params[param] = random.choice([parent1.to_dict()[param], parent2.to_dict()[param]])
    return TradingStrategy(**child_params)

def adaptive_mutation(strategy, mutation_rate=0.1):
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
    population = initialize_population(pop_size)
    best_strategy = None
    best_fitness = -np.inf
    for gen in range(generations):
        logging.info(f"GA Generation {gen+1}")
        fitnesses = [backtest_strategy(strategy, data) for strategy in population]
        for strat, fit in zip(population, fitnesses):
            logging.debug(f"Strategy {strat.to_dict()} fitness: {fit}")
            if fit > best_fitness:
                best_fitness = fit
                best_strategy = strat
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

# --------------------
# XGBoost Model for Signal Filtering
# --------------------
def train_xgb_model(train_data):
    df = train_data.copy()
    df['future_close'] = df['close'].shift(-5)
    df.dropna(inplace=True)
    df['target'] = (df['future_close'] > df['close']).astype(int)
    feature_cols = [
        'momentum', 'atr', 'vwap', 'ema_short', 'ema_long',
        'macd', 'rsi', 'bollinger_upper', 'bollinger_lower',
        'lagged_close_1', 'lagged_close_2', 'lagged_close_3',
        'lagged_close_5', 'lagged_close_10'
    ]
    if 'sentiment' in df.columns:
        feature_cols.append('sentiment')
    X = df[feature_cols]
    y = df['target']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    logging.info(f"XGBoost model trained. Validation Accuracy: {acc:.2f}")
    return model

def xgb_predict(model, features):
    return model.predict_proba(features)[:, 1]

# --------------------
# Trading Environment with Extra Safeguards and Logging
# --------------------
class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, feature_list, initial_capital=10000):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.feature_list = feature_list
        self.max_t = len(self.df) - 1
        self.initial_capital = initial_capital
        self.reset()
        obs_dim = len(self.feature_list) + 1  # features plus current position
        self.observation_space = spaces.Box(low=-1, high=1, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)  # 0: BUY, 1: SELL, 2: SHORT, 3: COVER, 4: HOLD

    def reset(self):
        self.t = 0
        self.cash = self.initial_capital
        self.position = 0
        self.current_price = self.df.loc[self.t, 'close']
        state = self._get_state()
        logging.debug(f"Env reset. State: {state}")
        return state

    def _get_state(self):
        state_features = self.df.loc[self.t, self.feature_list].values.astype(np.float32)
        state_features = np.nan_to_num(state_features, nan=0.0)
        state = np.append(state_features, float(self.position))
        return state

    def step(self, action):
        done = False
        prev_portfolio = self.compute_portfolio_value()
        # Use the open price for trading at this step
        price = self.df.loc[self.t, 'open']
        # Safeguard: if price is not finite or is near zero, set to 1e-5 and cap trade size.
        if not np.isfinite(price) or price < 1e-5:
            logging.warning(f"Price is not finite or near zero at step {self.t}. Adjusting price from {price} to 1e-5")
            price = 1e-5
        max_shares = 1000  # Cap the number of shares to avoid explosion

        # Also check cash validity
        if not np.isfinite(self.cash):
            logging.error(f"Cash is not finite at step {self.t} (cash={self.cash}). Resetting cash to initial capital.")
            self.cash = self.initial_capital

        # Process actions:
        # 0: BUY, 1: SELL, 2: SHORT, 3: COVER, 4: HOLD
        if action == 0:  # BUY
            if self.position <= 0:
                shares = int(self.cash / price)
                shares = min(shares, max_shares)
                if shares > 0:
                    self.cash -= shares * price
                    self.position += shares
        elif action == 1:  # SELL
            if self.position > 0:
                self.cash += self.position * price
                self.position = 0
        elif action == 2:  # SHORT
            if self.position >= 0:
                if self.position > 0:
                    self.cash += self.position * price
                    self.position = 0
                shares = int(self.cash / price)
                shares = min(shares, max_shares)
                if shares > 0:
                    self.cash += shares * price
                    self.position -= shares
        elif action == 3:  # COVER
            if self.position < 0:
                self.cash += (-self.position) * price
                self.position = 0
        elif action == 4:  # HOLD
            pass

        self.t += 1
        if self.t >= self.max_t:
            done = True

        new_portfolio = self.compute_portfolio_value()
        raw_reward = new_portfolio - prev_portfolio
        reward = raw_reward / 1e6  # scale reward
        if action == 4:
            reward -= 0.001  # slight penalty for holding
        logging.debug(f"Step {self.t}: raw_reward={raw_reward}, scaled_reward={reward}")
        obs = self._get_state() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, done, {}

    def compute_portfolio_value(self):
        if self.t < len(self.df):
            current_price = self.df.loc[self.t, 'close']
        else:
            current_price = self.current_price
        return self.cash + self.position * current_price

    def render(self, mode='human', close=False):
        logging.info(f"Step: {self.t}, Cash: {self.cash:.2f}, Position: {self.position}, Portfolio: {self.compute_portfolio_value():.2f}")

# --------------------
# RL Competition Training Function with Lower LR and Gradient Clipping
# --------------------
def train_rl_competition(df, feature_list):
    scaler = RobustScaler()
    df_copy = df.copy()
    transformed = scaler.fit_transform(df_copy[feature_list]).astype(np.float32)
    transformed = np.clip(transformed, -5, 5)
    transformed = np.nan_to_num(transformed, nan=0.0)
    df_copy.loc[:, feature_list] = transformed.astype(np.float32)

    # Create two environments for two competing RL agents.
    env1 = TradingEnv(df_copy, feature_list)
    env2 = TradingEnv(df_copy, feature_list)

    # Use PPO with a lower learning rate and gradient clipping.
    # Pass max_grad_norm directly as a parameter.
    model1 = PPO("MlpPolicy", env1, verbose=1, learning_rate=1e-4, max_grad_norm=0.5)
    model2 = PPO("MlpPolicy", env2, verbose=1, learning_rate=1e-4, max_grad_norm=0.5)

    rounds = 0
    max_round = 10
    portfolio1 = portfolio2 = 0

    while rounds < max_round:
        rounds += 1
        logging.info(f"RL Competition Round {rounds}")
        model1.learn(total_timesteps=len(df_copy)-1, reset_num_timesteps=False)
        portfolio1 = env1.compute_portfolio_value()
        model2.learn(total_timesteps=len(df_copy)-1, reset_num_timesteps=False)
        portfolio2 = env2.compute_portfolio_value()
        logging.info(f"Round {rounds}: Model1 portfolio: ${portfolio1:.2f}, Model2 portfolio: ${portfolio2:.2f}")
        if portfolio1 >= 20000 or portfolio2 >= 20000:
            logging.info("Early stopping: A portfolio reached $20,000 or above.")
            break
        if rounds == 10 and max(portfolio1, portfolio2) < 12000:
            max_round = 21
            logging.info("No model reached $12,000 by round 10; extending training to round 21.")

    final_model = model1 if portfolio1 >= portfolio2 else model2
    logging.info(f"Final RL Model chosen with portfolio value: ${max(portfolio1, portfolio2):.2f}")
    return final_model

# --------------------
# Core Functions: run_logic and run_backtest
# --------------------
def run_logic(current_price, predicted_price, ticker):
    from dotenv import load_dotenv
    load_dotenv()
    bar_timeframe = os.getenv("BAR_TIMEFRAME", "1Hour")
    tickers_env = os.getenv("TICKERS", "TSLA")
    disabled_features_env = os.getenv("DISABLED_FEATURES", "")
    news_mode = os.getenv("NEWS_MODE", "on").lower() == "on"

    disabled_features = [feat.strip() for feat in disabled_features_env.split(",") if feat.strip()]
    timeframe_suffix = convert_timeframe(bar_timeframe)
    csv_filename = f"{ticker}_{timeframe_suffix}.csv"

    df = load_and_preprocess_data(csv_filename, disabled_features, news_mode)
    inspect_data(df, df.columns, prefix="Post-Preprocessing - ")

    best_strategy = run_genetic_algorithm(df, pop_size=30, generations=15)
    xgb_model = train_xgb_model(df)

    feature_list = [col for col in df.columns if col not in ['timestamp', 'target_close']]
    if 'predicted_price' not in feature_list:
        feature_list.append('predicted_price')

    scaler = RobustScaler()
    df_copy = df.copy()
    transformed = scaler.fit_transform(df_copy[feature_list]).astype(np.float32)
    transformed = np.clip(transformed, -5, 5)
    transformed = np.nan_to_num(transformed, nan=0.0)
    df_copy.loc[:, feature_list] = transformed.astype(np.float32)
    inspect_data(df_copy, feature_list, prefix="After Robust Scaling - ")

    final_rl_model = train_rl_competition(df, feature_list)

    full_env = TradingEnv(df_copy, feature_list)
    obs = full_env.reset()
    done = False
    while not done:
        action, _ = final_rl_model.predict(obs)
        obs, reward, done, _ = full_env.step(action)
    final_action, _ = final_rl_model.predict(obs)
    logging.info(f"Final trade action based on RL: {final_action}")
    return final_action

def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    from dotenv import load_dotenv
    load_dotenv()
    bar_timeframe = os.getenv("BAR_TIMEFRAME", "1Hour")
    tickers_env = os.getenv("TICKERS", "TSLA")
    disabled_features_env = os.getenv("DISABLED_FEATURES", "")
    news_mode = os.getenv("NEWS_MODE", "on").lower() == "on"

    disabled_features = [feat.strip() for feat in disabled_features_env.split(",") if feat.strip()]
    timeframe_suffix = convert_timeframe(bar_timeframe)
    first_ticker = tickers_env.split(",")[0].strip()
    csv_filename = f"{first_ticker}_{timeframe_suffix}.csv"

    df_full = load_and_preprocess_data(csv_filename, disabled_features, news_mode)
    inspect_data(df_full, df_full.columns, prefix="Backtest - Raw Preprocessed Data")

    if not isinstance(current_timestamp, pd.Timestamp):
        current_timestamp = pd.to_datetime(current_timestamp)
    train_data = df_full[df_full['timestamp'] <= current_timestamp].copy()
    if train_data.empty:
        logging.info("No training data available up to the current timestamp.")
        return "NONE"

    best_strategy = run_genetic_algorithm(train_data, pop_size=30, generations=15)
    xgb_model = train_xgb_model(train_data)

    feature_list = [col for col in train_data.columns if col not in ['timestamp', 'target_close']]
    if 'predicted_price' not in feature_list:
        feature_list.append('predicted_price')
    inspect_data(train_data, feature_list, prefix="Backtest Selected Features - ")

    scaler = RobustScaler()
    train_copy = train_data.copy()
    transformed = scaler.fit_transform(train_copy[feature_list]).astype(np.float32)
    transformed = np.clip(transformed, -5, 5)
    transformed = np.nan_to_num(transformed, nan=0.0)
    train_copy.loc[:, feature_list] = transformed.astype(np.float32)

    final_rl_model = train_rl_competition(train_data, feature_list)

    env = TradingEnv(train_copy, feature_list)
    obs = env.reset()
    done = False
    while not done:
        action, _ = final_rl_model.predict(obs)
        obs, reward, done, _ = env.step(action)
    final_action, _ = final_rl_model.predict(obs)

    decision = "NONE"
    if final_action == 0 and position_qty <= 0:
        decision = "BUY"
    elif final_action == 1 and position_qty > 0:
        decision = "SELL"
    elif final_action == 2 and position_qty >= 0:
        decision = "SHORT"
    elif final_action == 3 and position_qty < 0:
        decision = "COVER"
    else:
        decision = "NONE"

    logging.info(f"Backtest trade decision: {decision}")
    return decision