# logic.py
# --------------------------------------------------------------------------------
# Enhanced Implementation aiming to reduce drawdowns and maximize profit using:
#   - A2C with a custom reward function and more training timesteps.
#   - Random Forest for predicted price input.
#   - Simple stop-loss and take-profit logic inside the environment.
#   - Slightly larger training data to help the RL agent converge.
#
# Adjusted to avoid "ValueError: Not enough data to run the environment..."
# by allowing a single row of data in the environment (though this can
# degrade training if it sees too few rows).
#
# --------------------------------------------------------------------------------

import os
import gym
import math
import numpy as np
import pandas as pd

from typing import List, Tuple, Any
from gym import spaces
from datetime import datetime

try:
    from stable_baselines3 import A2C
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    raise ImportError("Please install stable-baselines3 (pip install stable-baselines3).")

try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError:
    raise ImportError("Please install scikit-learn (pip install scikit-learn).")

# Hypothetical trading module with required functions:
#   api         -> get_position(ticker) returning an object with .qty
#   buy_shares  -> place a buy order
#   sell_shares -> place a sell order
#   short_shares-> open a short
#   close_short -> close an existing short
# You must provide or mock these in your environment.
from forest import api, buy_shares, sell_shares, short_shares, close_short

# --------------------------------------------------------------------------------
# 1) Environment Variables & Utility Functions
# --------------------------------------------------------------------------------

def convert_bar_timeframe(bar_timeframe: str) -> str:
    """
    Convert e.g. "4Hour" -> "H4", "2Hour" -> "H2", "1Hour" -> "H1",
    "30Min" -> "M30", "15Min" -> "M15", etc.
    """
    tf = bar_timeframe.lower().strip()
    if "hour" in tf:
        number = ''.join(filter(str.isdigit, tf))
        return f"H{number}"
    elif "min" in tf:
        number = ''.join(filter(str.isdigit, tf))
        return f"M{number}"
    else:
        return "H1"


def get_env_var(name: str, default_val: str = "") -> str:
    """
    Safely get environment variable, returning default_val if missing.
    """
    return os.environ.get(name, default_val)


def parse_tickers(tickers_str: str):
    """
    Parse a comma-separated string of tickers into a list.
    Example: "TSLA,AAPL" -> ["TSLA", "AAPL"]
    """
    if not tickers_str:
        return []
    return [t.strip().upper() for t in tickers_str.split(",") if t.strip()]


def load_csv_data(ticker: str, bar_timeframe_suffix: str) -> pd.DataFrame:
    """
    Loads the CSV data for the given ticker and timeframe suffix, e.g.:
      f"{ticker}_{bar_timeframe_suffix}.csv"
    """
    filename = f"{ticker}_{bar_timeframe_suffix}.csv"
    df = pd.read_csv(filename)
    return df


def filter_disabled_features(df: pd.DataFrame, disabled_feats: list) -> pd.DataFrame:
    """
    Removes any columns that appear in disabled_feats.
    Also removes 'timeframe', 'timestamp', or any non-numeric columns if needed.
    Then replaces Inf with NaN, drops rows with NaN, resets index.
    """
    df = df.copy()

    # Drop disabled columns
    for feat in disabled_feats:
        if feat in df.columns:
            df = df.drop(columns=[feat])

    # Remove 'timeframe','timestamp' unless specifically included
    for col in ['timeframe', 'timestamp']:
        if col in df.columns and col not in disabled_feats:
            df = df.drop(columns=[col], errors='ignore')

    # Remove any remaining non-numeric columns
    numeric_cols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
    df = df[numeric_cols].copy()

    # Replace Inf/-Inf with NaN, then drop
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df


# --------------------------------------------------------------------------------
# 2) TradingEnv with Enhanced Reward and Stop-Loss / Take-Profit
# --------------------------------------------------------------------------------

class TradingEnv(gym.Env):
    """
    Custom Gym environment for stock trading with A2C + advanced reward:
      - Observations: (filtered_features + predicted_price + position_qty)
      - Actions: 0=NONE,1=BUY,2=SELL,3=SHORT,4=COVER
      - Stop-loss / take-profit logic
      - More thorough reward shaping
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_position: float = 0.0,
        predicted_price: float = None,
        random_forest_model=None,
        stop_loss_pct: float = 0.05,    # 5% stop-loss
        take_profit_pct: float = 0.10,  # 10% take-profit
        capital: float = 10000.0,       # Starting capital for drawdown calculations
        penalty_no_action: float = 0.05 # Penalty for "NONE" to encourage trades
    ):
        super().__init__()

        self.data = data.reset_index(drop=True)
        self.n_steps = len(self.data)
        # Fix: allow a single row of data to avoid the error
        # but note it might degrade training or cause weird behavior if there's no data.
        if self.n_steps < 1:
            raise ValueError("Not enough data to run the environment after filtering (need >=1 row).")

        # Basic position and capital trackers
        self.position_qty = initial_position
        self.current_step = 0
        self.starting_capital = capital
        self.current_capital = capital  # track capital changes for drawdown
        self.max_capital = capital      # track peak for drawdown calculations

        # Market modeling
        self.external_predicted_price = predicted_price
        self.rf_model = random_forest_model

        # Stop-loss / take-profit
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # Penalty for doing nothing
        self.penalty_no_action = penalty_no_action

        # We'll store the last trade price so we can measure PnL from the entry
        self.entry_price = None

        # Setup observation space
        self.numeric_cols = list(self.data.columns)
        obs_dim = len(self.numeric_cols) + 2  # +2 => predicted_price + position_qty
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Action space: 0=NONE,1=BUY,2=SELL,3=SHORT,4=COVER
        self.action_space = spaces.Discrete(3)

        # Internal for seeding
        self.np_random = None

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _get_current_price(self) -> float:
        if self.current_step >= self.n_steps:
            return 0.0
        row = self.data.iloc[self.current_step]
        return float(row.get('close', 0.0))

    def _get_predicted_price(self, row_vals: np.ndarray) -> float:
        if self.external_predicted_price is not None:
            return float(self.external_predicted_price)
        elif self.rf_model is not None:
            pred = self.rf_model.predict(row_vals.reshape(1, -1))[0]
            return float(pred)
        else:
            return float(self._get_current_price())

    def _get_observation(self) -> np.ndarray:
        if self.current_step >= self.n_steps:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        row = self.data.iloc[self.current_step].astype(float)
        row_vals = row[self.numeric_cols].values
        pred_price = self._get_predicted_price(row_vals)

        obs = np.concatenate([row_vals, [pred_price, self.position_qty]])
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        return obs

    def _apply_stop_loss_take_profit(self, current_close: float) -> bool:
        """
        Force exit if price hits stop-loss or take-profit thresholds relative to entry_price.
        Returns True if forcibly exited.
        """
        if self.entry_price is None or self.position_qty == 0:
            return False

        position_side = 1 if self.position_qty > 0 else -1
        price_change = (current_close - self.entry_price) * position_side
        pct_change = price_change / abs(self.entry_price) if self.entry_price != 0 else 0

        if pct_change <= -self.stop_loss_pct:
            # Stop-loss triggered
            self.position_qty = 0
            self.entry_price = None
            return True
        elif pct_change >= self.take_profit_pct:
            # Take-profit triggered
            self.position_qty = 0
            self.entry_price = None
            return True

        return False

    def step(self, action: int):
        current_close = self._get_current_price()
        done = False
        reward = 0.0

        # if we have a position, add immediate PnL from price movement
        # We'll approximate incremental movement from last_price to current_close
        if hasattr(self, 'last_price') and self.last_price is not None and self.position_qty != 0:
            price_diff = (current_close - self.last_price)
            if self.position_qty > 0:
                reward += price_diff * self.position_qty
            elif self.position_qty < 0:
                reward += -price_diff * abs(self.position_qty)

        new_position = self.position_qty
        realized_pnl = 0.0
        closed_position = False

        # 0=NONE,1=BUY,2=SELL,3=SHORT,4=COVER
        if action == 1:  # BUY
            if new_position < 0:
                # we cover the short and go long
                realized_pnl = (self.entry_price - current_close) * abs(new_position)
                new_position = 1
                closed_position = True
            elif new_position == 0:
                new_position = 1
                self.entry_price = current_close
        elif action == 2:  # SELL
            if new_position > 0:
                realized_pnl = (current_close - self.entry_price) * new_position
                new_position = 0
                closed_position = True
        elif action == 3:  # SHORT
            if new_position > 0:
                realized_pnl = (current_close - self.entry_price) * new_position
                closed_position = True
            if new_position >= 0:
                new_position = -1
                self.entry_price = current_close
        elif action == 4:  # COVER
            if new_position < 0:
                realized_pnl = (self.entry_price - current_close) * abs(new_position)
                new_position = 0
                closed_position = True
        else:
            reward -= self.penalty_no_action

        reward += realized_pnl
        if closed_position:
            self.current_capital += realized_pnl
            if new_position == 0:
                self.entry_price = None
            else:
                self.entry_price = current_close
        elif new_position != self.position_qty:
            # just opened
            self.entry_price = current_close

        self.position_qty = new_position

        # check stop-loss / take-profit
        triggered = self._apply_stop_loss_take_profit(current_close)
        if triggered:
            # penalty or reward for forced exit
            reward -= 5.0

        # track drawdown
        if self.current_capital > self.max_capital:
            self.max_capital = self.current_capital

        drawdown = 1.0 - (self.current_capital / self.max_capital) if self.max_capital > 0 else 0
        if drawdown > 0.1:
            reward -= drawdown * 10

        self.last_price = current_close

        self.current_step += 1
        if self.current_step >= self.n_steps:
            done = True

        if done:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._get_observation()

        return obs, reward, done, {}

    def reset(self):
        self.current_step = 0
        self.position_qty = 0
        self.entry_price = None
        self.last_price = None
        self.current_capital = self.starting_capital
        self.max_capital = self.starting_capital
        return self._get_observation()


# --------------------------------------------------------------------------------
# 3) Global (Cached) Model Store
# --------------------------------------------------------------------------------

MODEL_REGISTRY = {}

def get_models_for(
    ticker: str,
    timeframe_suffix: str,
    df: pd.DataFrame,
    disabled_feats: list
) -> Tuple[A2C, RandomForestRegressor, pd.DataFrame, list]:
    """
    Return (A2C_model, RF_model, filtered_df, X_cols) from a global registry.
    If not present, train them.
    """
    key = (ticker, timeframe_suffix)
    if key in MODEL_REGISTRY:
        return MODEL_REGISTRY[key]

    filtered_df = filter_disabled_features(df.copy(), disabled_feats)
    if len(filtered_df) < 1:
        raise ValueError("No valid data rows after filtering/cleaning. Can't train or trade.")

    if 'close' not in filtered_df.columns:
        raise ValueError("No 'close' column in data after filtering. Can't train RF on 'close'.")

    # Build target as next close
    shift_df = filtered_df.copy()
    shift_df['target_close'] = shift_df['close'].shift(-1)
    shift_df = shift_df.dropna().copy()
    X_cols = [c for c in shift_df.columns if c != 'target_close']
    X = shift_df[X_cols].copy()
    y = shift_df['target_close'].copy()

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)

    if len(X) < 5:
        raise ValueError("Not enough rows to train RF after final cleaning.")

    rf_model = RandomForestRegressor(n_estimators=30, random_state=42)
    rf_model.fit(X, y.loc[X.index])

    # Build environment
    env = TradingEnv(
        data=filtered_df,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        capital=10000.0,
        penalty_no_action=0.05
    )
    vec_env = DummyVecEnv([lambda: env])

    model = A2C(
        "MlpPolicy",
        vec_env,
        verbose=0,
        learning_rate=1e-3,
        gamma=0.99,
        n_steps=5,
        seed=42
    )

    model.learn(total_timesteps=5000)

    MODEL_REGISTRY[key] = (model, rf_model, filtered_df, X_cols)
    return MODEL_REGISTRY[key]


# --------------------------------------------------------------------------------
# 4) Required Functions: run_logic & run_backtest
# --------------------------------------------------------------------------------

def run_logic(current_price: float, predicted_price: float, ticker: str):
    """
    Live logic. Single-row environment for the final bar + predicted_price.
    """
    bar_timeframe = get_env_var("BAR_TIMEFRAME", "1Hour")
    timeframe_suffix = convert_bar_timeframe(bar_timeframe)
    disabled_str = get_env_var("DISABLED_FEATURES", "")
    disabled_feats = [f.strip() for f in disabled_str.split(",") if f.strip()]

    df = load_csv_data(ticker, timeframe_suffix)
    model, rf_model, filtered_df, feature_cols = get_models_for(
        ticker, timeframe_suffix, df, disabled_feats
    )

    # Get live position (if none exists, assume 0)
    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception as e:
        print(f"No open position for {ticker}: {e}")
        position_qty = 0.0

    account = api.get_account()
    cash = float(account.cash)

    if len(filtered_df) == 0:
        return

    last_row = filtered_df.iloc[[-1]].copy()
    live_env = TradingEnv(
        data=last_row,
        initial_position=position_qty,
        predicted_price=predicted_price,
        random_forest_model=rf_model,
        stop_loss_pct=0.05,
        take_profit_pct=0.10
    )

    obs = live_env.reset()
    action_array, _states = model.predict(obs, deterministic=False)
    action = int(action_array)

    action_map = {
        0: "NONE",
        1: "BUY",
        2: "SELL",
        3: "SHORT",
        4: "COVER",
    }
    chosen_action = action_map[action]

    def execute_action(a):
        if a == "BUY":
            if position_qty < 1:
                max_shares = int(cash // current_price)
                print("buy")
                buy_shares(ticker, max_shares, current_price, predicted_price)
        elif a == "SELL":
            if position_qty > 0:
                print("sell")
                sell_shares(ticker, position_qty, current_price, predicted_price)
        elif a == "SHORT":
            if position_qty > -1:
                max_shares = int(cash // current_price)
                print("short")
                short_shares(ticker, max_shares, current_price, predicted_price)
        elif a == "COVER":
            if position_qty < 0:
                qty_to_close = abs(position_qty)
                print("cover")
                close_short(ticker, qty_to_close, current_price)

    # prevent duplicates
    if chosen_action == "BUY" and position_qty >= 1:
        return
    if chosen_action == "SELL" and position_qty <= 0:
        return
    if chosen_action == "SHORT" and position_qty <= -1:
        return
    if chosen_action == "COVER" and position_qty >= 0:
        return

    execute_action(chosen_action)


def run_backtest(current_price: float, predicted_price: float, position_qty: float, current_timestamp, candles):
    """
    Backtest logic. Single-row environment for last bar of first TICKER in .env.
    """
    bar_timeframe = get_env_var("BAR_TIMEFRAME", "1Hour")
    timeframe_suffix = convert_bar_timeframe(bar_timeframe)

    tickers_list = parse_tickers(get_env_var("TICKERS", "TSLA"))
    ticker = tickers_list[0] if tickers_list else "TSLA"

    disabled_str = get_env_var("DISABLED_FEATURES", "")
    disabled_feats = [f.strip() for f in disabled_str.split(",") if f.strip()]

    df = load_csv_data(ticker, timeframe_suffix)
    model, rf_model, filtered_df, feature_cols = get_models_for(
        ticker, timeframe_suffix, df, disabled_feats
    )

    if len(filtered_df) == 0:
        return "NONE"

    last_row = filtered_df.iloc[[-1]].copy()
    backtest_env = TradingEnv(
        data=last_row,
        initial_position=position_qty,
        predicted_price=predicted_price,
        random_forest_model=rf_model,
        stop_loss_pct=0.05,
        take_profit_pct=0.10
    )

    obs = backtest_env.reset()
    action_array, _states = model.predict(obs, deterministic=False)
    action = int(action_array)

    action_map = {
        0: "NONE",
        1: "BUY",
        2: "SELL",
    }
    chosen_action = action_map[action]

    # prevent duplicates
    if chosen_action == "BUY" and position_qty >= 1:
        return "NONE"
    if chosen_action == "SELL" and position_qty <= 0:
        return "NONE"
    if chosen_action == "SHORT" and position_qty <= -1:
        return "NONE"
    if chosen_action == "COVER" and position_qty >= 0:
        return "NONE"

    return chosen_action
