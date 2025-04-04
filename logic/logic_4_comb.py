#!/usr/bin/env python3
"""
Combined Trading System:
- Two agents each run a complete GA + XGBoost + RL system (as in Script 1).
- The agents then compete using a round-based RL competition (as in Script 2).
- In live trading (run_logic) the entire CSV (built from ticker and timeframe) is used,
  whereas in backtesting (run_backtest) only candles up to the current timestamp are used.
- The losing RL model in the competition is updated via online learning.
- The system dynamically filters out disabled CSV features (via DISABLED_FEATURES)
  and uses sentiment only when NEWS_MODE is on.
- Idle (“HOLD”/“NONE”) actions are penalized to encourage active trading.
  
Required libraries: pandas, numpy, matplotlib, xgboost, scikit‑learn, gym, stable‑baselines3, python‑dotenv, forest (trading API)
"""

import os
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from collections import deque
from dotenv import load_dotenv

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import gym
from gym import spaces
from stable_baselines3 import PPO

# -----------------------------
# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# -----------------------------
# Load Environment Variables & Timeframe Mapping
load_dotenv()

BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "1Hour")
TICKERS = os.getenv("TICKERS", "TSLA")
DISABLED_FEATURES = os.getenv("DISABLED_FEATURES", "")
NEWS_MODE = os.getenv("NEWS_MODE", "on").lower()  # "on" or "off"

TIMEFRAME_MAP = {
    "4Hour": "H4",
    "2Hour": "H2",
    "1Hour": "H1",
    "30Min": "M30",
    "15Min": "M15"
}
TIMEFRAME_SUFFIX = TIMEFRAME_MAP.get(BAR_TIMEFRAME, "H1")

# -----------------------------
# Data Loading and Preprocessing

def load_and_preprocess_data(csv_file):
    """
    Loads CSV data and computes technical indicators:
      - Converts 'timestamp' to datetime and sorts chronologically.
      - Computes EMAs, MACD, RSI, ATR, stop_loss and take_profit.
      - Filters out any features specified in DISABLED_FEATURES (except RSI, which is always computed).
      - Removes 'sentiment' if NEWS_MODE is off.
    """
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        logging.error(f"Error loading CSV file {csv_file}: {e}")
        raise

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Compute EMAs and MACD
    df['ema_short'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']

    # Compute RSI (14 period)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # Compute ATR if not provided
    if 'atr' not in df.columns or df['atr'].isnull().all():
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()

    # Demo stop_loss and take_profit (ATR based)
    df['stop_loss'] = df['close'] - df['atr']
    df['take_profit'] = df['close'] + df['atr']

    # Fill missing values
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    # Filter out disabled features (except RSI)
    disabled_list = [feat.strip() for feat in DISABLED_FEATURES.split(",") if feat.strip()]
    for col in disabled_list:
        if col.lower() == "sentiment" and NEWS_MODE == "off":
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
        elif col.lower() != "rsi" and col in df.columns:
            df.drop(columns=[col], inplace=True)

    if NEWS_MODE != "on" and 'sentiment' in df.columns:
        df.drop(columns=['sentiment'], inplace=True)

    logging.info("Data loaded and preprocessed successfully.")
    return df

def add_predicted_price(df, ml_features):
    """
    Adds a 'predicted_price' column using a regression model (XGBoost or RandomForest) trained on historical data.
    The target is the next close price.
    """
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

def load_and_prepare_data(csv_file):
    """
    Loads and processes CSV data:
      - Preprocesses the data.
      - Defines a base feature list (including computed indicators).
      - Adds the predicted_price feature.
      - Scales all features using StandardScaler.
    Returns the processed DataFrame and the list of features.
    """
    df = load_and_preprocess_data(csv_file)

    base_features = [
        'open', 'high', 'low', 'close', 'vwap', 'momentum', 'atr', 'obv',
        'bollinger_upper', 'bollinger_lower', 'lagged_close_1', 'lagged_close_2',
        'lagged_close_3', 'lagged_close_5', 'lagged_close_10'
    ]
    if NEWS_MODE == "on" and 'sentiment' in df.columns:
        base_features.append('sentiment')

    computed_features = ['ema_short', 'ema_long', 'macd', 'rsi']
    ml_features = base_features + computed_features

    df = add_predicted_price(df, ml_features)
    features = ml_features + ['predicted_price']

    scaler = StandardScaler()
    scaler.fit(df[features])
    df.loc[:, features] = scaler.transform(df[features])
    return df, features

# -----------------------------
# Genetic Algorithm (GA) for Strategy Optimization

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
        strategy = TradingStrategy(
            momentum_weight=random.uniform(0, 1),
            sentiment_weight=random.uniform(0, 1),
            atr_weight=random.uniform(0, 1),
            vwap_weight=random.uniform(0, 1),
            ema_diff_weight=random.uniform(0, 1),
            risk_reward=random.uniform(1, 3),
            entry_threshold=random.uniform(0.1, 1)
        )
        population.append(strategy)
    return population

def backtest_strategy(strategy, data):
    initial_capital = 10000
    capital = initial_capital
    position = 0  # 1 for long, -1 for short, 0 for flat
    entry_price = 0
    returns = []

    data = data.copy()
    data['ema_diff'] = data['ema_short'] - data['ema_long']

    # Normalize indicators (demo approach)
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

        # Long entry
        if position == 0 and signal > strategy.entry_threshold:
            position = 1
            entry_price = data.loc[data.index[i], 'open']
        # Short entry
        elif position == 0 and signal < -strategy.entry_threshold:
            position = -1
            entry_price = data.loc[data.index[i], 'open']
        # Close long
        elif position == 1 and signal < -strategy.entry_threshold:
            exit_price = data.loc[data.index[i], 'open']
            trade_return = (exit_price - entry_price) / entry_price
            returns.append(trade_return)
            capital *= (1 + trade_return * strategy.risk_reward)
            position = 0
        # Close short
        elif position == -1 and signal > strategy.entry_threshold:
            exit_price = data.loc[data.index[i], 'open']
            trade_return = (entry_price - exit_price) / entry_price
            returns.append(trade_return)
            capital *= (1 + trade_return * strategy.risk_reward)
            position = 0

    # Close any open position at end-of-data
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

def run_genetic_algorithm(data, pop_size=30, generations=15):
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
# XGBoost Model for Signal Filtering (Classification)

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
    if NEWS_MODE == "on" and 'sentiment' in df.columns:
        feature_cols.append("sentiment")
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

# -----------------------------
# Combined Trading Environment for RL Competition

class TradingEnv(gym.Env):
    """
    Trading Environment for RL competition with 5 actions:
      0: BUY, 1: SELL, 2: SHORT, 3: COVER, 4: HOLD
    The state consists of the preprocessed feature vector from the current candle plus:
      - ga_signal: computed using the agent’s best GA strategy
      - xgb_prob: probability from the agent’s XGBoost classifier
      - current position (as a scalar)
    A slight penalty is applied for HOLD to encourage active trading.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, features, best_strategy, xgb_model, initial_capital=10000):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.features = features
        self.best_strategy = best_strategy
        self.xgb_model = xgb_model
        self.initial_capital = initial_capital
        self.current_step = 0
        self.capital = initial_capital
        self.position = 0  # positive for long, negative for short
        self.entry_price = 0
        self.max_steps = len(self.df) - 1

        # Observation: [features...] + [ga_signal, xgb_prob, position]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.features) + 3,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)  # BUY, SELL, SHORT, COVER, HOLD

    def _get_current_row(self):
        return self.df.iloc[self.current_step]

    def _compute_signals(self, row):
        # Compute a simple EMA difference for normalization (if available)
        ema_diff = row['ema_short'] - row['ema_long'] if 'ema_short' in row and 'ema_long' in row else 0
        norm_close = row['close'] if row['close'] != 0 else 1e-10
        norm_momentum = row['momentum'] / (abs(row['momentum']) + 1e-10) if 'momentum' in row else 0
        norm_sentiment = row['sentiment'] if 'sentiment' in row else 0
        norm_atr = row['atr'] / norm_close if 'atr' in row else 0
        norm_vwap = row['vwap'] / norm_close if 'vwap' in row else 0
        norm_ema_diff = ema_diff / norm_close

        ga_signal = (self.best_strategy.momentum_weight * norm_momentum +
                     self.best_strategy.sentiment_weight * norm_sentiment +
                     self.best_strategy.atr_weight * norm_atr +
                     self.best_strategy.vwap_weight * norm_vwap +
                     self.best_strategy.ema_diff_weight * norm_ema_diff)
        # Prepare features for XGBoost prediction (using the same columns as in training)
        feature_cols = [
            'momentum', 'atr', 'vwap', 'ema_short', 'ema_long',
            'macd', 'rsi', 'bollinger_upper', 'bollinger_lower',
            'lagged_close_1', 'lagged_close_2', 'lagged_close_3',
            'lagged_close_5', 'lagged_close_10'
        ]
        if NEWS_MODE == "on" and 'sentiment' in self.df.columns:
            feature_cols.append('sentiment')
        # Explicitly cast the features to float to avoid dtype issues
        xgb_features = row[feature_cols].to_frame().T.astype(float)
        xgb_prob = self.xgb_model.predict_proba(xgb_features)[:, 1][0]
        return ga_signal, xgb_prob


    def _get_state(self):
        row = self._get_current_row()
        state_features = row[self.features].values.astype(np.float32)
        ga_signal, xgb_prob = self._compute_signals(row)
        state = np.concatenate([state_features, np.array([ga_signal, xgb_prob, float(self.position)])])
        return state

    def step(self, action):
        done = False
        row = self._get_current_row()
        price = row['open']  # Use the open price for trades
        reward = 0

        if action == 0:  # BUY
            if self.position <= 0:
                shares = self.capital // price
                if shares > 0:
                    self.capital -= shares * price
                    self.position += shares
                    self.entry_price = price
        elif action == 1:  # SELL
            if self.position > 0:
                self.capital += self.position * price
                profit = (price - self.entry_price) / self.entry_price
                reward = profit * self.position
                self.position = 0
        elif action == 2:  # SHORT
            if self.position >= 0:
                if self.position > 0:
                    self.capital += self.position * price
                    self.position = 0
                shares = self.capital // price
                if shares > 0:
                    self.capital += shares * price  # receive cash from short sale
                    self.position -= shares
                    self.entry_price = price
        elif action == 3:  # COVER
            if self.position < 0:
                self.capital -= abs(self.position) * price
                profit = (self.entry_price - price) / self.entry_price
                reward = profit * abs(self.position)
                self.position = 0
        elif action == 4:  # HOLD (penalty to discourage idleness)
            reward = -0.01

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
            final_price = self.df.iloc[self.max_steps]['close']
            if self.position > 0:
                self.capital += self.position * final_price
                profit = (final_price - self.entry_price) / self.entry_price
                reward += profit * self.position
                self.position = 0
            elif self.position < 0:
                self.capital -= abs(self.position) * final_price
                profit = (self.entry_price - final_price) / self.entry_price
                reward += profit * abs(self.position)
                self.position = 0

        next_state = self._get_state() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return next_state, reward, done, {}

    def reset(self):
        self.current_step = 0
        self.capital = self.initial_capital
        self.position = 0
        self.entry_price = 0
        return self._get_state()

    def compute_portfolio_value(self):
        current_price = self.df.iloc[self.current_step]['close'] if self.current_step < len(self.df) else self.df.iloc[-1]['close']
        return self.capital + self.position * current_price

    def render(self, mode='human'):
        logging.info(f"Step: {self.current_step}, Capital: {self.capital}, Position: {self.position}")

# -----------------------------
# RL Competition Training (2 Agents)

def train_rl_competition(df, features, best_strategy1, xgb_model1, best_strategy2, xgb_model2, initial_capital=10000):
    """
    Creates two RL agents (each using its own GA and XGBoost models embedded in the environment)
    and trains them in rounds. In each round, each agent learns on the same data.
    The agent with the lower portfolio value resets to its previous parameters.
    Training stops early if any model reaches a portfolio value of 20,000 or if, by round 10,
    neither model has a portfolio value above 12,000 (in which case training extends to 21 rounds).
    Returns the final (winning) RL model.
    """
    env1 = TradingEnv(df, features, best_strategy1, xgb_model1, initial_capital=initial_capital)
    env2 = TradingEnv(df, features, best_strategy2, xgb_model2, initial_capital=initial_capital)

    model1 = PPO("MlpPolicy", env1, verbose=0)
    model2 = PPO("MlpPolicy", env2, verbose=0)

    rounds = 0
    max_round = 10
    portfolio_value1 = 0
    portfolio_value2 = 0

    while rounds < max_round:
        rounds += 1
        params1 = model1.get_parameters()
        params2 = model2.get_parameters()

        model1.learn(total_timesteps=len(df)-1, reset_num_timesteps=False)
        portfolio_value1 = env1.compute_portfolio_value()

        model2.learn(total_timesteps=len(df)-1, reset_num_timesteps=False)
        portfolio_value2 = env2.compute_portfolio_value()

        logging.info(f"Round {rounds}: Model1 portfolio: ${portfolio_value1:.2f}, Model2 portfolio: ${portfolio_value2:.2f}")

        if portfolio_value1 >= 20000 or portfolio_value2 >= 20000:
            logging.info("Early stop: Portfolio reached 20000 or above.")
            break

        if rounds == 10 and max(portfolio_value1, portfolio_value2) < 12000:
            max_round = 21
            logging.info("Extending training to 21 rounds as portfolio values are below 12000.")

        # Reset the losing model's parameters to simulate online learning improvement
        if portfolio_value1 < portfolio_value2:
            model1.set_parameters(params1)
        else:
            model2.set_parameters(params2)

    final_model = model1 if portfolio_value1 >= portfolio_value2 else model2
    return final_model

# -----------------------------
# Core Functions: run_logic and run_backtest

def run_logic(current_price, predicted_price, ticker):
    """
    Live Trading Logic:
      1. Dynamically loads CSV data based on ticker and BAR_TIMEFRAME.
      2. Prepares the data (filters disabled features, adds predicted_price, scales features).
      3. For each of two agents, runs GA and trains an XGBoost classifier on the entire dataset.
      4. Trains two RL agents (via competition rounds) using the combined environment that includes GA and XGBoost signals.
      5. Uses the winning RL model to decide the next action based on the latest candle.
      6. Retrieves current live position via the trading API and executes trade orders accordingly.
    """
    from forest import api, buy_shares, sell_shares, short_shares, close_short

    csv_filename = f"{ticker}_{TIMEFRAME_SUFFIX}.csv"
    df, features = load_and_prepare_data(csv_filename)

    # Train GA and XGBoost for both agents on the entire dataset
    best_strategy1 = run_genetic_algorithm(df, pop_size=30, generations=15)
    xgb_model1 = train_xgb_model(df)

    best_strategy2 = run_genetic_algorithm(df, pop_size=30, generations=15)
    xgb_model2 = train_xgb_model(df)

    # Train RL agents via competition
    final_model = train_rl_competition(df, features, best_strategy1, xgb_model1, best_strategy2, xgb_model2)

    # Obtain the final observation by simulating the environment until the last candle
    env = TradingEnv(df, features, best_strategy1, xgb_model1)  # Using one agent’s parameters for prediction
    obs = env.reset()
    while True:
        action, _ = final_model.predict(obs)
        obs, reward, done, _ = env.step(action)
        if done:
            break
    next_action, _ = final_model.predict(obs)

    # Retrieve current live position via API
    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception as e:
        logging.info(f"No open position for {ticker}: {e}")
        position_qty = 0.0

    account = api.get_account()
    cash = float(account.cash)

    # Map RL action to trade orders (ensuring no duplicate trades)
    if next_action == 0 and position_qty <= 0:  # BUY
        max_shares = int(cash // current_price)
        logging.info("Executing BUY order")
        buy_shares(ticker, max_shares, current_price, predicted_price)
    elif next_action == 1 and position_qty > 0:  # SELL
        logging.info("Executing SELL order")
        sell_shares(ticker, position_qty, current_price, predicted_price)
    elif next_action == 2 and position_qty >= 0:  # SHORT
        max_shares = int(cash // current_price)
        logging.info("Executing SHORT order")
        short_shares(ticker, max_shares, current_price, predicted_price)
    elif next_action == 3 and position_qty < 0:  # COVER
        qty_to_close = abs(position_qty)
        logging.info("Executing COVER order")
        close_short(ticker, qty_to_close, current_price)
    else:
        logging.info("No trade action taken")

def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    """
    Backtesting Logic:
      1. Loads CSV data for backtesting using the first ticker in TICKERS and BAR_TIMEFRAME.
      2. Prepares data (filters disabled features, adds predicted_price, scales features).
      3. Uses only the training data up to the provided current_timestamp.
      4. For each of two agents, runs GA and trains XGBoost on the training data.
      5. Trains two RL agents (via competition) on the training data.
      6. Simulates the environment and returns a trade decision string ("BUY", "SELL", "SHORT", "COVER", or "NONE"),
         ensuring duplicate trades are prevented using the provided position_qty.
    """
    first_ticker = TICKERS.split(",")[0].strip()
    csv_filename = f"{first_ticker}_{TIMEFRAME_SUFFIX}.csv"
    df_full, features = load_and_prepare_data(csv_filename)

    if not isinstance(current_timestamp, pd.Timestamp):
        current_timestamp = pd.to_datetime(current_timestamp)
    train_data = df_full[df_full['timestamp'] <= current_timestamp].copy()
    if train_data.empty:
        logging.info("No training data available up to the current timestamp.")
        return "NONE"

    # Train GA and XGBoost for both agents on the training data
    best_strategy1 = run_genetic_algorithm(train_data, pop_size=30, generations=15)
    xgb_model1 = train_xgb_model(train_data)

    best_strategy2 = run_genetic_algorithm(train_data, pop_size=30, generations=15)
    xgb_model2 = train_xgb_model(train_data)

    # Train RL agents via competition on training data
    final_model = train_rl_competition(train_data, features, best_strategy1, xgb_model1, best_strategy2, xgb_model2)

    # Identify current candle from the candles DataFrame
    if "timestamp" in candles.columns:
        candle_row = candles[candles["timestamp"] == current_timestamp]
        if candle_row.empty:
            candle_row = candles.iloc[[-1]]
        else:
            candle_row = candle_row.iloc[[0]]
    else:
        candle_row = candles.iloc[[-1]]
    current_candle = candle_row.squeeze()

    # Create a temporary environment for backtest prediction
    env = TradingEnv(train_data, features, best_strategy1, xgb_model1)
    obs = env.reset()
    while True:
        action, _ = final_model.predict(obs)
        obs, reward, done, _ = env.step(action)
        if done:
            break
    final_action, _ = final_model.predict(obs)

    # Map action to decision string, taking into account the current position_qty
    if final_action == 0 and position_qty <= 0:
        return "BUY"
    elif final_action == 1 and position_qty > 0:
        return "SELL"
    elif final_action == 2 and position_qty >= 0:
        return "SHORT"
    elif final_action == 3 and position_qty < 0:
        return "COVER"
    else:
        return "NONE"
