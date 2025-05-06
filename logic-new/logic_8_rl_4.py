"""
trade_logic_rl.py

An RL-driven trading strategy using Stable-Baselines3 (PPO). 

Features:
  - Loads historical data per TICKERS/.env and BAR_TIMEFRAME.
  - Defines a Gym Env with actions {0: HOLD, 1: BUY, 2: SELL}.
  - Trains a PPO agent on the entire available history (up to current point).
  - run_logic: retrains on full CSV, then applies agent to the latest observation.
  - run_backtest: for each candle, retrains on history up to that timestamp, then applies agent.

Requirements:
  pip install stable-baselines3 gym pandas numpy python-dotenv
"""

import os
import logging
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from forest import api, buy_shares, sell_shares

# Load settings
load_dotenv()
BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "1Hour")
TICKERS = os.getenv("TICKERS", "TSLA").split(",")
TIMEFRAME_MAP = {
    "4Hour": "H4",
    "2Hour": "H2",
    "1Hour": "H1",
    "30Min": "M30",
    "15Min": "M15"
}
CONVERTED_TIMEFRAME = TIMEFRAME_MAP.get(BAR_TIMEFRAME, BAR_TIMEFRAME)

def get_csv_filename(ticker: str) -> str:
    return f"{ticker}_{CONVERTED_TIMEFRAME}.csv"

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop any rows with NaN or infinite values, reset the index.
    """
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df_clean

class TradingEnv(gym.Env):
    """Gym environment for trading one asset with RL."""
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df.reset_index(drop=True)
        # observation: all features except timestamp
        self.obs_cols = [c for c in df.columns if c != "timestamp"]
        self.action_space = spaces.Discrete(3)  # 0=HOLD,1=BUY,2=SELL
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.obs_cols),), dtype=np.float32
        )
        self._reset_internal()

    def _reset_internal(self):
        self.step_idx = 0
        self.position = 0.0  # shares held
        self.cash = 1_000_000.0  # starting capital
        self.net_worth = self.cash

    def reset(self):
        self._reset_internal()
        return self._get_obs()

    def _get_obs(self):
        row = self.df.loc[self.step_idx, self.obs_cols].values.astype(np.float32)
        return row

    def step(self, action: int):
        price = float(self.df.loc[self.step_idx, "close"])
        # execute action
        if action == 1 and self.position == 0:
            # buy max
            self.position = self.cash // price
            self.cash -= self.position * price
        elif action == 2 and self.position > 0:
            # sell all
            self.cash += self.position * price
            self.position = 0
        # advance
        self.step_idx += 1
        done = self.step_idx >= len(self.df) - 1
        # reward = change in net worth
        next_price = float(self.df.loc[self.step_idx, "close"])
        self.net_worth = self.cash + self.position * next_price
        reward = self.net_worth - 1_000_000.0
        return self._get_obs(), reward, done, {}

    def render(self, mode="human"):
        print(f"Step: {self.step_idx}, Cash: {self.cash:.2f}, "
              f"Position: {self.position}, Net worth: {self.net_worth:.2f}")


def train_agent(df: pd.DataFrame) -> PPO:
    """
    Train a PPO agent on the full, cleaned DataFrame history.
    Automatically reduces n_steps to len(df)-1 to avoid
    requesting more rollout steps than available.
    """
    df_clean = clean_df(df)
    if len(df_clean) < 2:
        raise ValueError("Not enough data to train RL agent – need at least 2 rows after cleaning.")
    
    # Create environment (unchanged)
    env = TradingEnv(df_clean)
    
    # Choose n_steps ≤ len(df_clean)-1
    n_steps = min(2048, len(df_clean) - 1)
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        n_steps=n_steps,
    )
    total_timesteps = len(df_clean) * 5  # you can tune this multiplier
    model.learn(total_timesteps=total_timesteps)
    return model


def run_logic(current_price: float, predicted_price: float, ticker: str):
    """
    Live trading decision:
      - Retrains agent on full CSV history.
      - Applies agent to latest candle (with predicted_close replaced by predicted_price).
    """
    logger = logging.getLogger(__name__)
    # Load data
    fname = get_csv_filename(ticker)
    df = pd.read_csv(fname, parse_dates=["timestamp"])
    # replace last predicted_close with live prediction
    df.at[df.index[-1], "predicted_close"] = predicted_price
    # train
    agent = train_agent(df)
    # prepare obs
    last_obs = df.loc[df.index[-1], df.columns != "timestamp"].values.astype(np.float32)
    action, _ = agent.predict(last_obs, deterministic=True)
    # execute
    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception:
        position_qty = 0.0
    cash = float(api.get_account().cash)
    logger.info(f"[{ticker}] Action from RL agent: {action} "
                f"(0=HOLD,1=BUY,2=SELL); Position: {position_qty}, Cash: {cash}")
    if action == 1 and position_qty == 0:
        shares = int(cash // current_price)
        if shares > 0:
            buy_shares(ticker, shares, current_price, predicted_price)
    elif action == 2 and position_qty > 0:
        sell_shares(ticker, position_qty, current_price, predicted_price)
    # HOLD does nothing


def run_backtest(
    current_price: float,
    predicted_price: float,
    position_qty: float,
    current_timestamp,
    candles: pd.DataFrame
) -> str:
    """
    Backtest decision at a given timestamp:
      - Load the full CSV.
      - Filter history up to current_timestamp (inclusive).
      - Clean it, abort early if too short.
      - Train the agent, then predict action on the last row.
      - Return "BUY", "SELL", or "NONE".
    """
    # Load full history
    ticker = TICKERS[0]
    fname = get_csv_filename(ticker)
    df_full = pd.read_csv(fname, parse_dates=["timestamp"])
    
    # Filter up to now
    hist = df_full[df_full.timestamp <= pd.to_datetime(current_timestamp)].copy()
    hist = clean_df(hist)
    
    # If not enough history, do nothing
    if len(hist) < 2:
        return "NONE"
    
    # Inject the live prediction
    hist.at[hist.index[-1], "predicted_close"] = predicted_price
    
    # Train agent
    agent = train_agent(hist)
    
    # Build observation vector (drop timestamp)
    obs = hist.iloc[-1].drop("timestamp").values.astype(np.float32)
    action, _ = agent.predict(obs, deterministic=True)
    
    # Map to backtest signal
    if action == 1 and position_qty == 0:
        return "BUY"
    elif action == 2 and position_qty > 0:
        return "SELL"
    else:
        return "NONE"