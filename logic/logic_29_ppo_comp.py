import os
import logging
import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  # (used only if needed for debugging)

from dotenv import load_dotenv
load_dotenv()

# ---------------- Utility Functions ---------------- #

def calculate_rsi(series, period=14):
    """
    Calculate the Relative Strength Index (RSI) for a given series.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_predicted_price(df, ml_features):
    """
    Uses either an XGBoost regressor or a RandomForest regressor (based on ML_MODEL)
    to predict the next close price. The function trains on historical data using
    ml_features as predictors and adds a 'predicted_price' column to the dataframe.
    """
    ml_model_choice = os.getenv('ML_MODEL', 'forest').lower()
    df = df.copy()
    # Create target as the next close price (shift -1)
    df['target_close'] = df['close'].shift(-1)
    df_model = df.dropna(subset=['target_close'])
    X = df_model[ml_features]
    y = df_model['target_close']
    if ml_model_choice == 'xgboost':
        import xgboost as xgb
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    else:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    # Predict on the entire dataset using the provided ml_features
    predicted_prices = model.predict(df[ml_features])
    df['predicted_price'] = predicted_prices
    df.drop(columns=['target_close'], inplace=True)
    return df

def load_and_prepare_data():
    """
    Loads the CSV dynamically based on .env variables. It converts the BAR_TIMEFRAME
    to the correct filename suffix, uses the first ticker from TICKERS, and filters out
    any disabled features as specified by DISABLED_FEATURES (with special handling for
    sentiment based on NEWS_MODE). It computes additional indicators and also uses an ML
    model (xgboost or random forest) to predict price, which is added as a feature.
    Finally, the selected features are scaled.
    """
    # Get environment variables
    bar_timeframe = os.getenv('BAR_TIMEFRAME')            # e.g. "4Hour"
    tickers = os.getenv('TICKERS')                        # e.g. "TSLA,AAPL"
    disabled_features_env = os.getenv('DISABLED_FEATURES', '')
    news_mode = os.getenv('NEWS_MODE', 'false').lower() == 'true'
    
    disabled_features = [feat.strip() for feat in disabled_features_env.split(',') if feat.strip()]
    
    # Convert timeframe to filename suffix
    timeframe_map = {"4Hour": "H4", "2Hour": "H2", "1Hour": "H1", "30Min": "M30", "15Min": "M15"}
    timeframe_suffix = timeframe_map.get(bar_timeframe, bar_timeframe)
    
    # Use the first ticker from the comma-separated list
    ticker_first = tickers.split(',')[0].strip() if tickers else "DEFAULT"
    csv_file = f"{ticker_first}_{timeframe_suffix}.csv"
    
    # Load CSV file
    df = pd.read_csv(csv_file)
    
    # Remove disabled features (if any), except for 'rsi' which is always computed
    for col in disabled_features:
        if col != 'rsi' and col in df.columns:
            df.drop(columns=[col], inplace=True)
    
    # Remove sentiment if NEWS_MODE is off
    if not news_mode and 'sentiment' in df.columns:
        df.drop(columns=['sentiment'], inplace=True)
    
    # Compute additional indicators
    df['short_ema'] = df['close'].ewm(span=12, adjust=False).mean()
    df['long_ema'] = df['close'].ewm(span=26, adjust=False).mean()
    df['ema_diff'] = df['short_ema'] - df['long_ema']
    df['macd_line'] = df['ema_diff']
    df['signal_line'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['rsi'] = calculate_rsi(df['close'], 14)
    
    # Drop rows with NaN values resulting from indicator calculations
    df.dropna(inplace=True)
    
    # Build the base feature list from the CSV (if enabled)
    base_features = [
        'open', 'high', 'low', 'close', 'vwap', 'momentum', 'atr', 'obv',
        'bollinger_upper', 'bollinger_lower', 'lagged_close_1', 'lagged_close_2',
        'lagged_close_3', 'lagged_close_5', 'lagged_close_10'
    ]
    if news_mode:
        base_features.append('sentiment')
    base_features = [f for f in base_features if f not in disabled_features]
    
    # Computed features
    computed_features = ['short_ema', 'long_ema', 'ema_diff', 'macd_line', 'signal_line', 'rsi']
    
    # Features used for ML price prediction (do not include predicted_price)
    ml_features = base_features + computed_features
    
    # Add predicted_price using the selected ML model
    df = add_predicted_price(df, ml_features)
    
    # Final feature list for RL training includes predicted_price
    features = ml_features + ['predicted_price']
    
    # Scale the features using StandardScaler
    scaler = StandardScaler()
    scaler.fit(df[features])
    df.loc[:, features] = scaler.transform(df[features])
    
    return df, features

# ---------------- Trading Environment and RL Competition ---------------- #

class TradingEnv(gym.Env):
    """
    A trading environment that simulates trades based on a DataFrame of historical data.
    It uses a set of features (which now includes predicted_price) and computes a portfolio
    value based on executed actions.
    """
    def __init__(self, df, features):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.features = features
        self.max_t = len(df) - 1  # Last index for stepping
        self.action_space = spaces.Discrete(5)  # BUY, SELL, SHORT, COVER, NONE
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.features) + 1,), dtype=np.float32
        )
        self.reset()

    def reset(self):
        self.t = 0
        self.cash = 10000  # Initial cash balance
        self.position = 0
        self.p_t = self.df['close'].iloc[self.t]
        return self.get_state()

    def get_state(self):
        state = self.df.iloc[self.t][self.features].values
        state = np.append(state, float(self.position))
        return state.astype(np.float32)

    def step(self, action):
        self.execute_action(action)
        if self.t + 1 < self.max_t:
            p_next = self.df['close'].iloc[self.t + 1]
            reward = self.position * (p_next - self.p_t)
        else:
            reward = 0
        self.t += 1
        if self.t >= self.max_t:
            done = True
        else:
            done = False
            self.p_t = self.df['close'].iloc[self.t]
        state = self.get_state()
        return state, reward, done, {}

    def execute_action(self, action):
        price = self.p_t
        if action == 0:  # BUY
            if self.position <= 0 and self.cash > 0:
                shares = int(self.cash / price)
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
                if self.cash > 0:
                    shares = int(self.cash / price)
                    self.cash += shares * price
                    self.position -= shares
        elif action == 3:  # COVER
            if self.position < 0:
                self.cash += self.position * price
                self.position = 0
        elif action == 4:  # HOLD/NONE
            pass

    def compute_portfolio_value(self):
        return self.cash + self.position * self.p_t

def train_rl_model(df, features):
    """
    Runs RL competition rounds on the provided dataframe and returns the final trained model.
    Two PPO models are trained in rounds with the following additional logic:
      - If in any round a model reaches a portfolio value of 20,000 or more, the training stops early.
      - If by round 10 neither model reaches at least 12,000 in portfolio value,
        the training is extended up to a maximum of 21 rounds.
      - At round 21 (if reached), the best model from that round is chosen.
    """
    env1 = TradingEnv(df, features)
    env2 = TradingEnv(df, features)
    
    model1 = PPO("MlpPolicy", env1, verbose=0)
    model2 = PPO("MlpPolicy", env2, verbose=0)
    
    rounds = 0
    max_round = 10
    portfolio_value1 = None
    portfolio_value2 = None
    
    while rounds < max_round:
        rounds += 1
        params1 = model1.get_parameters()
        params2 = model2.get_parameters()
        
        model1.learn(total_timesteps=len(df) - 1, reset_num_timesteps=False)
        portfolio_value1 = env1.compute_portfolio_value()
        
        model2.learn(total_timesteps=len(df) - 1, reset_num_timesteps=False)
        portfolio_value2 = env2.compute_portfolio_value()
        
        if portfolio_value1 > portfolio_value2:
            winner = "Model1"
            model2.set_parameters(params2)
        else:
            winner = "Model2"
            model1.set_parameters(params1)
        
        logging.info(f"Round {rounds}: Model1 portfolio: ${portfolio_value1:.2f}, Model2 portfolio: ${portfolio_value2:.2f}, Winner: {winner}")
        
        # Early stop if any model reaches a portfolio value of 20,000 or more.
        if portfolio_value1 >= 20000 or portfolio_value2 >= 20000:
            logging.info("Early stop: Portfolio reached 20000 or above.")
            break
        
        # If reached round 10 and best portfolio is below 12,000, extend training up to round 21.
        if rounds == 10:
            if max(portfolio_value1, portfolio_value2) < 12000:
                max_round = 21
                logging.info("No model reached 12000 by round 10; extending training to round 21.")
    
    final_model = model1 if portfolio_value1 > portfolio_value2 else model2
    return final_model

# ---------------- Main Functions: run_logic and run_backtest ---------------- #

def run_logic(current_price, predicted_price, ticker):
    """
    Live trading logic:
      - Loads and processes the entire CSV (based on .env variables) including
        the new predicted_price feature.
      - Trains the RL model on the full dataset using the modified training loop.
      - Uses the trained model to decide the next action.
      - Imports trading API functions from 'forest' and executes the corresponding trade.
    """
    from forest import api, buy_shares, sell_shares, short_shares, close_short
    
    df, features = load_and_prepare_data()
    final_model = train_rl_model(df, features)
    
    full_env = TradingEnv(df, features)
    obs = full_env.reset()
    while True:
        action, _ = final_model.predict(obs)
        obs, reward, done, _ = full_env.step(action)
        if done:
            break
    next_action, _ = final_model.predict(obs)
    
    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception as e:
        logging.info(f"No open position for {ticker}: {e}")
        position_qty = 0.0
    
    account = api.get_account()
    cash = float(account.cash)
    
    if next_action == 0 and position_qty <= 0:
        max_shares = int(cash // current_price)
        logging.info("Executing BUY")
        buy_shares(ticker, max_shares, current_price, predicted_price)
    elif next_action == 1 and position_qty > 0:
        logging.info("Executing SELL")
        sell_shares(ticker, position_qty, current_price, predicted_price)
    elif next_action == 2 and position_qty >= 0:
        max_shares = int(cash // current_price)
        logging.info("Executing SHORT")
        short_shares(ticker, max_shares, current_price, predicted_price)
    elif next_action == 3 and position_qty < 0:
        qty_to_close = abs(position_qty)
        logging.info("Executing COVER")
        close_short(ticker, qty_to_close, current_price)
    else:
        logging.info("No trade executed (HOLD)")

def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    """
    Backtest logic:
      - Processes the provided candles DataFrame (assumed to be sorted from oldest to newest),
        computes the new predicted_price feature, and scales all features.
      - Trains the RL model using all candle data up to the current timestamp using the modified training loop.
      - Uses the trained model to decide an action.
      - Returns the trade decision string ("BUY", "SELL", "SHORT", "COVER", or "NONE").
    """
    # Get environment variables for feature processing
    disabled_features_env = os.getenv('DISABLED_FEATURES', '')
    news_mode = os.getenv('NEWS_MODE', 'false').lower() == 'true'
    disabled_features = [feat.strip() for feat in disabled_features_env.split(',') if feat.strip()]
    
    if not news_mode and 'sentiment' in candles.columns:
        candles = candles.drop(columns=['sentiment'])
    
    for col in disabled_features:
        if col != 'rsi' and col in candles.columns:
            candles = candles.drop(columns=[col])
    
    candles['short_ema'] = candles['close'].ewm(span=12, adjust=False).mean()
    candles['long_ema'] = candles['close'].ewm(span=26, adjust=False).mean()
    candles['ema_diff'] = candles['short_ema'] - candles['long_ema']
    candles['macd_line'] = candles['ema_diff']
    candles['signal_line'] = candles['macd_line'].ewm(span=9, adjust=False).mean()
    candles['rsi'] = calculate_rsi(candles['close'], 14)
    
    candles.dropna(inplace=True)
    
    base_features = [
        'open', 'high', 'low', 'close', 'vwap', 'momentum', 'atr', 'obv',
        'bollinger_upper', 'bollinger_lower', 'lagged_close_1', 'lagged_close_2',
        'lagged_close_3', 'lagged_close_5', 'lagged_close_10'
    ]
    if news_mode:
        base_features.append('sentiment')
    base_features = [f for f in base_features if f not in disabled_features]
    
    computed_features = ['short_ema', 'long_ema', 'ema_diff', 'macd_line', 'signal_line', 'rsi']
    ml_features = base_features + computed_features
    
    candles = add_predicted_price(candles, ml_features)
    
    features = ml_features + ['predicted_price']
    
    scaler = StandardScaler()
    scaler.fit(candles[features])
    candles.loc[:, features] = scaler.transform(candles[features])
    
    df = candles.copy()
    final_model = train_rl_model(df, features)
    
    env = TradingEnv(df, features)
    obs = env.reset()
    while True:
        action, _ = final_model.predict(obs)
        obs, reward, done, _ = env.step(action)
        if done:
            break
    final_action, _ = final_model.predict(obs)
    
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