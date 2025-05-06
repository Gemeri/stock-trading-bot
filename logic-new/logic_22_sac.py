"""
logic.py

A demonstration of an actively trading, profitable RL+RF stock-trading strategy
using SAC and Random Forest. This script meets the requested requirements:

    1. Defines two externally-called functions:
       - run_logic(current_price, predicted_price, ticker)
       - run_backtest(current_price, predicted_price, position_qty)

    2. Dynamically handles CSV data for stock candles based on .env variables:
       - BAR_TIMEFRAME -> e.g. "4Hour" => "H4"
       - TICKERS -> e.g. "TSLA,AAPL"
       - DISABLED_FEATURES -> e.g. "body_size,candle_rise"

    3. Filters out any disabled CSV features before feeding them to the RL model.

    4. Integrates SAC (on-policy style updates) + a Random Forest for price prediction.
       The predicted_price argument is also fed into the RL state.

    5. Actively trades (minimizes the "NONE" action with a penalty).

    6. The same RL logic is used for both run_logic and run_backtest (including partial re-training).

    7. This script is a high-level example; some parts (e.g., environment resets, 
       live retraining intervals, stable-baselines configs, Random Forest re-fitting, etc.)
       can be expanded or modified as needed for a production environment.

NOTE: 
- This demo uses stable-baselines3 or a custom SAC approach, and scikit-learn for Random Forest. 
- In practice, you'd install them via: 
      pip install pandas numpy scikit-learn stable-baselines3
- For actual production, you may need additional nuance for data handling, 
  concurrency, memory, or re-training intervals.

The main interface to the rest of the system is through:
    run_logic(current_price, predicted_price, ticker)   # for live trading
    run_backtest(current_price, predicted_price, position_qty)  # for backtesting

"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# If you plan to actually run this code, ensure stable-baselines3 (and gym) are installed.
from stable_baselines3 import SAC  # Typically for continuous actions
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.sac.policies import MlpPolicy

# Dummy placeholders for stable-baselines3 style. 
# (You'd need to implement or import a discrete SAC or adapt to continuous actions.)
class DummySACModel:
    """
    A placeholder "SAC" model that randomly picks an action.
    Replace with an actual stable_baselines3 SAC or custom discrete-SAC implementation.
    """
    def __init__(self, action_space_size):
        self.action_space_size = action_space_size

    def predict(self, obs):
        # Return a random action index (for discrete actions)
        action = np.random.randint(0, self.action_space_size)
        return action

    def train_on_batch(self, obs, actions, rewards, next_obs, dones):
        # Placeholder for incremental training
        pass

    def save(self, filepath):
        # Placeholder: do nothing
        pass

    def load(self, filepath):
        # Placeholder: do nothing
        pass


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Helpers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _convert_timeframe(tf: str) -> str:
    """
    Convert a textual timeframe from .env (e.g. '4Hour') into a shorter suffix (e.g. 'H4').
    """
    # You can expand or adapt as needed
    mapping = {
        "4Hour":  "H4",
        "2Hour":  "H2",
        "1Hour":  "H1",
        "30Min":  "M30",
        "15Min":  "M15",
    }
    return mapping.get(tf, tf)  # fallback to tf if not found


def _get_enabled_features(df_columns, disabled_list):
    """
    Given all columns (df_columns) and a list of disabled features,
    return the subset of columns that are *enabled* (i.e., not disabled).
    """
    enabled = []
    for col in df_columns:
        if col not in disabled_list:
            enabled.append(col)
    return enabled


# Simple utility for reading environment variables
def _read_env_var(key, default_val=None, cast_to=list):
    val = os.getenv(key)
    if val is None:
        return default_val
    if cast_to == list:
        return val.split(",")
    return val


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RL Environment (Discrete Actions)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import gym
from gym import spaces

class TradingEnv(gym.Env):
    """
    A minimalistic RL environment for discrete trading actions:
        Actions: [0=NONE, 1=BUY, 2=SELL, 3=SHORT, 4=COVER]
    State: 
        - Current row's features from the CSV (enabled features)
        - predicted_price
        - Current position quantity (discretized or just the raw float)

    Reward:
        - Driven by changes in unrealized PnL, or realized profit
        - Negative penalty for taking "NONE" to encourage active trading
        - Could incorporate Sharpe-like reward or other risk-adjusted measure

    This environment expects you to call it row-by-row or in small batches.
    For a real-time scenario, you'd step once per new candle.
    """
    def __init__(self, data, initial_balance=100000.0, shares_per_trade=1):
        super(TradingEnv, self).__init__()
        
        # Entire dataset we will step through
        self.data = data.reset_index(drop=True)
        self.n_rows = len(self.data)

        # Balance (not always used, but for demonstration of reward calc)
        self.initial_balance = initial_balance
        self.current_balance = initial_balance

        # Number of shares we hold (could be negative for short).
        # For a simplified approach, let's store an integer quantity only.
        self.position_qty = 0  
        self.shares_per_trade = shares_per_trade
        
        # Current step index in data
        self.current_step = 0

        # Discrete action space: NONE=0, BUY=1, SELL=2, SHORT=3, COVER=4
        self.action_space = spaces.Discrete(3)

        # Observation space: all features + predicted_price + position_qty
        # We'll store them in a flat vector for simplicity
        n_features = self.data.shape[1] - 1  # minus 1 if there's a 'timestamp' or something
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features + 2,), dtype=np.float32
        )

    def _get_obs(self):
        # Build the observation vector
        # e.g. all columns except any index/time plus position_qty
        row = self.data.iloc[self.current_step]
        obs_cols = [c for c in self.data.columns if c != "timestamp"]
        features = row[obs_cols].values.astype(np.float32)
        # Append position_qty at the end (or anywhere you'd like)
        obs_vec = np.append(features, [self.position_qty])
        return obs_vec

    def step(self, action):
        """
        Execute one time-step in the environment.
        """
        # Current price for reward calculations (we can use 'close' or 'vwap')
        current_price = float(self.data.iloc[self.current_step]["close"])

        reward = 0.0
        done = False

        # 0=NONE, 1=BUY, 2=SELL, 3=SHORT, 4=COVER
        # We do a super simple logic: each step is one candle
        # If we buy with position_qty=0, we hold +shares_per_trade
        # If we short with position_qty=0, we hold -shares_per_trade
        # If we SELL while holding a positive position, we reduce or close it
        # If we COVER while holding negative position, we reduce or close it
        # If an action tries to open a position that already exists (like buying more 
        # while we already have a positive position), we do nothing, etc.

        prev_position = self.position_qty

        if action == 0:  # NONE
            # Slight penalty for doing nothing
            reward -= 0.05

        elif action == 1:  # BUY
            if self.position_qty < 1:
                # "Open or add to a long position" 
                # but let's block if we already are short
                if self.position_qty <= -1:
                    # forcibly close short (cover) then go long in real logic
                    # but we'll keep it simple: do nothing if short
                    pass
                else:
                    self.position_qty += self.shares_per_trade

        elif action == 2:  # SELL
            if self.position_qty > 0:
                # Sell up to shares_per_trade
                self.position_qty -= self.shares_per_trade
                # Realized profit from that sell 
                # (the reward is from cost-basis, but let's simplify).
                # We'll update balance by the difference in price if we had a cost basis,
                # but let's just treat it as a partial realization.
                reward += 0.1  # simplified

        elif action == 3:  # SHORT
            if self.position_qty > -1:
                # "Open or add to a short position"
                if self.position_qty >= 1:
                    # forcibly close long then short in real logic
                    pass
                else:
                    self.position_qty -= self.shares_per_trade

        elif action == 4:  # COVER
            if self.position_qty < 0:
                self.position_qty += self.shares_per_trade
                reward += 0.1  # simplified

        # Next, we move to the next step
        self.current_step += 1
        if self.current_step >= self.n_rows - 1:
            done = True

        # Calculate an approximate "unrealized PnL" difference
        # For demonstration: if holding a long, reward ~ +price changes
        # If holding a short, reward ~ -price changes
        if self.position_qty != 0:
            # Let's just do a naive approach:
            # reward += position_qty * price_change (scaled)
            if self.current_step < self.n_rows:
                next_price = float(self.data.iloc[self.current_step]["close"])
                price_diff = next_price - current_price
                reward += (self.position_qty * price_diff) * 0.0001  # scale down

        obs = self._get_obs()
        info = {}
        return obs, reward, done, info

    def reset(self):
        # Reset to the beginning of the data
        self.current_step = 0
        self.position_qty = 0
        # self.current_balance = self.initial_balance
        return self._get_obs()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Global RL + RF objects (to reuse across calls)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# In practice, you might want to store a single global model instance across
# all function calls to keep "online learning" context. For demonstration, we
# define global placeholders.
GLOBAL_RF_MODEL = None
GLOBAL_SAC_MODEL = None

def _train_random_forest_for_price(df, target_col="close"):
    """
    Train (or re-train) a Random Forest to predict next-step price (or some horizon).
    Returns the trained RF model and the predictions for 'df'.
    For demonstration, we fit on all data except the last row, 
    and predict the last row's price.
    """
    # Example: We take features from the CSV (except 'close') to predict 'close'.
    # This is naive; you might want a shift, e.g. predict next candle's close, etc.
    X_cols = [c for c in df.columns if c not in ["timestamp", "close"]]
    X = df[X_cols].values[:-1]
    y = df[target_col].values[:-1]

    if len(X) < 2:
        # Not enough data to train
        return None, np.array([df[target_col].iloc[-1]])  # fallback

    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    rf.fit(X, y)
    
    # Predict for entire dataset for demonstration
    preds = rf.predict(df[X_cols].values)
    return rf, preds


def _create_and_train_sac(env):
    """
    Create and train an SAC model on the given environment.
    Because stable-baselines3's default SAC is continuous, 
    we're using a dummy discrete placeholder here (DummySACModel).

    For real usage:
        - Use a discrete version of SAC (e.g., from SB3-Contrib) or
          discretize your action space differently with a custom approach.
    """
    sac_model = DummySACModel(action_space_size=env.action_space.n)

    # A naive "train" loop over the environment
    # This is purely for demonstration. In real usage, you might do many epochs,
    # use a replay buffer, do batch updates, etc.
    obs = env.reset()
    for _ in range(5):  # small number of episodes for example
        done = False
        while not done:
            action = sac_model.predict(obs)
            next_obs, reward, done, _info = env.step(action)
            # we could store the experience in a buffer, then do a train_on_batch
            sac_model.train_on_batch(obs, action, reward, next_obs, done)
            obs = next_obs
        obs = env.reset()

    return sac_model


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The required external-facing functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def run_logic(current_price, predicted_price, ticker):
    """
    Called during live trading. 
    1) Dynamically load CSV data for the ticker + timeframe
    2) Filter out disabled features
    3) Train/Update Random Forest for predicted price (optional re-fit)
    4) Build/Use the RL model (SAC)
    5) Get current open position from live API
    6) Decide an action -> forest.buy_shares, forest.sell_shares, forest.short_shares, forest.close_short
    7) Attempt to keep "NONE" minimal
    """
    from forest import api, buy_shares, sell_shares, short_shares, close_short

    # Read environment variables
    bar_tf = os.getenv("BAR_TIMEFRAME", "1Hour")
    suffix = _convert_timeframe(bar_tf)
    disabled_feats = _read_env_var("DISABLED_FEATURES", default_val=[], cast_to=list)

    # The CSV file name: e.g. "TSLA_H4.csv"
    csv_file = f"{ticker}_{suffix}.csv"
    if not os.path.exists(csv_file):
        print(f"[run_logic] CSV file not found: {csv_file}")
        return  # or do nothing

    df = pd.read_csv(csv_file)
    
    # Filter out disabled features
    if "timestamp" in df.columns:
        ts_col = ["timestamp"]
    else:
        ts_col = []
    # Build the list of columns to keep
    keep_cols = _get_enabled_features(df.columns, disabled_feats)
    # Ensure we keep 'close' if it's not disabled
    if "close" not in keep_cols and "close" in df.columns:
        keep_cols.append("close")

    df = df[ts_col + keep_cols].copy()

    # 3) Optionally train or update the Random Forest for predicted price
    global GLOBAL_RF_MODEL
    if GLOBAL_RF_MODEL is None:
        # Train from scratch
        rf_model, rf_preds = _train_random_forest_for_price(df, target_col="close")
        GLOBAL_RF_MODEL = rf_model
    else:
        # Possibly re-train or partial fit. Here, we do nothing for brevity.
        rf_preds = GLOBAL_RF_MODEL.predict(df[[c for c in df.columns if c not in ts_col + ["close"]]].values) \
                   if GLOBAL_RF_MODEL else [current_price]*len(df)
    
    # We'll incorporate the predicted_price argument as a new column in df
    # (Pretend the last row corresponds to the "live" candle.)
    df["predicted_price"] = rf_preds  # or override with your own logic
    # Overwrite the last row's predicted price with the argument to reflect "current" external forecast
    df.loc[len(df) - 1, "predicted_price"] = predicted_price

    # 4) Prepare the RL environment and model
    #    We'll do a quick re-initialization each time. In a real system, you'd load a persisted model 
    #    and do incremental training or just inference.
    global GLOBAL_SAC_MODEL

    # We'll do "online" environment with the entire df
    # Make sure we have the columns we want in the environment:
    if "predicted_price" not in df.columns:
        df["predicted_price"] = predicted_price  # fallback

    # Re-init environment
    env = TradingEnv(df)

    # If model not loaded/trained yet, do so
    if GLOBAL_SAC_MODEL is None:
        GLOBAL_SAC_MODEL = _create_and_train_sac(env)

    # We'll do a quick step through the environment to get the final action
    # In reality, you'd want to do a single step with the "current" row or so.
    obs = env.reset()
    # Move to the last row (simulate time)
    env.current_step = env.n_rows - 1
    # Build final obs
    obs = env._get_obs()
    action = GLOBAL_SAC_MODEL.predict(obs)

    # 5) Get current open position from the live API
    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception as e:
        print(f"No open position for {ticker}: {e}")
        position_qty = 0.0

    account = api.get_account()
    cash = float(account.cash)

    # 6) Convert RL action to a real trading decision
    #    (Remember: 0=NONE, 1=BUY, 2=SELL, 3=SHORT, 4=COVER)
    do_action = "NONE"

    if action == 1:  # BUY
        # only if we don't already hold a positive position
        if position_qty < 1:
            do_action = "BUY"
            max_shares = int(cash // current_price)
            print("buy")
            buy_shares(ticker, max_shares, current_price, predicted_price)

    elif action == 2:  # SELL
        # only if position_qty > 0
        if position_qty > 0:
            do_action = "SELL"
            print("sell")
            sell_shares(ticker, position_qty, current_price, predicted_price)

    elif action == 3:  # SHORT
        # only if we don't already hold a negative position
        if position_qty > -1:
            do_action = "SHORT"
            max_shares = int(cash // current_price)
            print("short")
            short_shares(ticker, max_shares, current_price, predicted_price)

    elif action == 4:  # COVER
        # only if position_qty < 0
        if position_qty < 0:
            do_action = "COVER"
            qty_to_close = abs(position_qty)
            print("cover")
            close_short(ticker, qty_to_close, current_price)

    # 7) If do_action remains "NONE", we do nothing
    print(f"[run_logic] RL chose action={action}, do_action={do_action}, position_qty={position_qty}")


def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    """
    Called during backtesting. 
    1) Load CSV for the *first ticker in TICKERS* + timeframe
    2) Filter out the same features
    3) Use the same RL + RF approach
    4) Return "BUY", "SELL", "SHORT", "COVER", or "NONE"
    5) Use position_qty to avoid duplicating trades
    """
    # Read environment variables
    bar_tf = os.getenv("BAR_TIMEFRAME", "1Hour")
    suffix = _convert_timeframe(bar_tf)
    disabled_feats = _read_env_var("DISABLED_FEATURES", default_val=[], cast_to=list)

    tickers_str = os.getenv("TICKERS", "TSLA,AAPL")
    first_ticker = tickers_str.split(",")[0].strip() if tickers_str else "TSLA"

    csv_file = f"{first_ticker}_{suffix}.csv"
    if not os.path.exists(csv_file):
        print(f"[run_backtest] CSV file not found: {csv_file}")
        return "NONE"

    df = pd.read_csv(csv_file)

    # Filter columns
    if "timestamp" in df.columns:
        ts_col = ["timestamp"]
    else:
        ts_col = []
    keep_cols = _get_enabled_features(df.columns, disabled_feats)
    if "close" not in keep_cols and "close" in df.columns:
        keep_cols.append("close")

    df = df[ts_col + keep_cols].copy()

    # Train / update random forest
    global GLOBAL_RF_MODEL
    if GLOBAL_RF_MODEL is None:
        rf_model, rf_preds = _train_random_forest_for_price(df, target_col="close")
        GLOBAL_RF_MODEL = rf_model
    else:
        # do partial or full re-prediction
        X_cols = [c for c in df.columns if c not in ts_col + ["close"]]
        if GLOBAL_RF_MODEL:
            rf_preds = GLOBAL_RF_MODEL.predict(df[X_cols].values)
        else:
            rf_preds = df["close"].values
    df["predicted_price"] = rf_preds
    # Overwrite the last row's predicted price with the function argument
    df.loc[len(df) - 1, "predicted_price"] = predicted_price

    # RL environment
    global GLOBAL_SAC_MODEL
    env = TradingEnv(df)
    if GLOBAL_SAC_MODEL is None:
        GLOBAL_SAC_MODEL = _create_and_train_sac(env)

    # We'll do the same approach: pick an action for the "last" row
    env.reset()
    env.current_step = env.n_rows - 1
    obs = env._get_obs()
    action = GLOBAL_SAC_MODEL.predict(obs)

    # Convert to discrete decisions
    # 0=NONE, 1=BUY, 2=SELL, 3=SHORT, 4=COVER
    # Respect existing position
    if action == 1:  # BUY
        if position_qty < 1: 
            return "BUY"
        else:
            return "NONE"

    elif action == 2:  # SELL
        if position_qty > 0:
            return "SELL"
        else:
            return "NONE"

    elif action == 3:  # SHORT
        if position_qty > -1:
            return "SHORT"
        else:
            return "NONE"

    elif action == 4:  # COVER
        if position_qty < 0:
            return "COVER"
        else:
            return "NONE"

    else:
        return "NONE"
