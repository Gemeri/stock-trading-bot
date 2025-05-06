"""
rl_logic_best.py  ·  RL‑powered trade‑logic module (stateless API)
Author : ChatGPT  ·  Date  : 2025‑05‑06  ·  Python 3.10+

Requirements
------------
Gymnasium ≥ 0.29 · Stable‑Baselines3 ≥ 2.2.1 · Pandas ≥ 2.2 · NumPy ≥ 1.26
"""

from __future__ import annotations

import os
import logging
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

# ────────────────────────────────#
# Global constants & ENV config
# ────────────────────────────────#
BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "1Hour")
TICKERS       = os.getenv("TICKERS", "TSLA").split(",")        # spec: single ticker
TIMEFRAME_MAP = {"4Hour": "H4", "2Hour": "H2", "1Hour": "H1",
                 "30Min": "M30", "15Min": "M15"}
CONVERTED_TIMEFRAME = TIMEFRAME_MAP.get(BAR_TIMEFRAME, BAR_TIMEFRAME)

INITIAL_CASH          = 100_000.0
COMMISSION_PER_SHARE  = 0.01          # round‑turn commission
RANDOM_SEED           = 42
np.random.seed(RANDOM_SEED)

_model_cache: dict[str, PPO] = {}

FEATURE_COLUMNS: list[str] = []

LOG_DIR = "./logs/rl_logic_best"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s  [%(levelname)s]  %(message)s"))
logger.addHandler(handler)


# ────────────────────────────────#
# Utility helpers
# ────────────────────────────────#
def get_csv_filename(ticker: str) -> str:
    """Construct the historical CSV name."""
    return f"{ticker}_{CONVERTED_TIMEFRAME}.csv"


def load_full_csv(ticker: str) -> pd.DataFrame:
    """Read the entire CSV (chronologically ascending)."""
    path = get_csv_filename(ticker)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing historical file: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


# ────────────────────────────────#
# Reward‑shaping wrapper (extra credit)
# ────────────────────────────────#
class DrawdownRewardWrapper(gym.Wrapper):
    """
    Wraps TradingEnv to:
      1) Amplify negative step‐rewards by `loss_scale`
      2) Subtract a drawdown penalty proportional to current drawdown.
    """
    def __init__(self, env: gym.Env, penalty_scale: float = 0.1, loss_scale: float = 2.0):
        super().__init__(env)
        self.penalty_scale = penalty_scale
        self.loss_scale = loss_scale

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 1) If we lost money this step, make it sting harder
        if reward < 0:
            reward *= self.loss_scale

        # 2) Always subtract a drawdown penalty
        #    (info["drawdown"] comes from TradingEnv.step)
        dd = info.get("drawdown", 0.0)
        reward -= dd * self.penalty_scale

        return obs, reward, terminated, truncated, info


# ────────────────────────────────#
# Custom trading environment
# ────────────────────────────────#
class TradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, window_size: int = 1):
        super().__init__()
        # ─── CLEAN OUT NaNs/Infs ───
        clean = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        self.df           = clean.copy()
        self.window_size  = window_size
        self.current_step = window_size

        # Discrete 3-action space
        self.action_space = spaces.Discrete(3)

        # Observation = all columns except timestamp
        obs_example = self.df.iloc[self.current_step].drop("timestamp")
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(obs_example),), dtype=np.float32
        )

        # Portfolio state
        self.cash                = INITIAL_CASH
        self.shares_held         = 0
        self.portfolio_value     = INITIAL_CASH
        self.max_portfolio_value = INITIAL_CASH
        self.current_drawdown    = 0.0

    # ──────────────────────────#
    # Private helpers
    # ──────────────────────────#
    def _get_observation(self) -> np.ndarray:
        row = self.df.iloc[self.current_step].drop("timestamp")
        return row.astype(np.float32).values

    def _trade(self, action: int, price: float) -> None:
        if action == 1:                   # BUY‑max
            max_qty = int(self.cash // price)
            if max_qty > 0:
                cost = max_qty * price + max_qty * COMMISSION_PER_SHARE
                self.cash        -= cost
                self.shares_held += max_qty

        elif action == 2:                 # SELL‑all
            if self.shares_held > 0:
                revenue = self.shares_held * price - self.shares_held * COMMISSION_PER_SHARE
                self.cash        += revenue
                self.shares_held  = 0
        # action 0 ⇒ pass

    # ──────────────────────────#
    # Gym API
    # ──────────────────────────#
    def step(self, action: int):
        assert self.action_space.contains(action)

        price      = float(self.df.iloc[self.current_step]["close"])
        prev_value = self.portfolio_value

        self._trade(action, price)

        # Update stats
        self.portfolio_value     = self.cash + self.shares_held * price
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
        self.current_drawdown    = (
            self.max_portfolio_value - self.portfolio_value
        ) / self.max_portfolio_value

        reward = self.portfolio_value - prev_value  # Δ equity

        # Advance
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated  = False

        info: Dict[str, Any] = {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "shares_held": self.shares_held,
            "drawdown": self.current_drawdown,
        }
        return self._get_observation(), reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_step        = self.window_size
        self.cash                = INITIAL_CASH
        self.shares_held         = 0
        self.portfolio_value     = INITIAL_CASH
        self.max_portfolio_value = INITIAL_CASH
        self.current_drawdown    = 0.0
        return self._get_observation(), {}


# ────────────────────────────────#
# Training utilities
# ────────────────────────────────#
class EarlyStopDrawdown(BaseCallback):
    """Interrupt learning if drawdown > 20 %."""
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.threshold = 0.20

    def _on_step(self) -> bool:
        dd = self.locals["infos"][0].get("drawdown", 0)
        if dd > self.threshold:
            logger.warning(f"Early stop ‑ drawdown {dd:.2%}")
            return False
        return True


def train_agent(candles: pd.DataFrame, save_path: str) -> None:
    """
    Train PPO once on the provided `candles` DataFrame.
    Assumes `candles` already contains only historical rows (< current_timestamp).
    """
    global FEATURE_COLUMNS

    # 1) Clean NaNs/Infs from exactly the rows you passed in
    train_df = (
        candles
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )

    if train_df.empty:
        raise ValueError("Training set empty after dropping NaNs—check the slice you passed in.")

    # 2) Record feature order (drop timestamp)
    FEATURE_COLUMNS = train_df.columns.drop("timestamp").tolist()

    # 3) Build the env with our loss‐ and drawdown‐shaping wrapper
    env = DummyVecEnv([lambda: DrawdownRewardWrapper(
        TradingEnv(train_df),
        penalty_scale=0.1,
        loss_scale=2.0
    )])

    # 4) Configure SB3 logger (no stdout duplicate)
    sb3_logger = configure(LOG_DIR, ["tensorboard", "csv"])

    # 5) Instantiate PPO with tuned hyper-parameters
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=4096,            # larger rollout batch
        batch_size=256,
        learning_rate=1e-4,      # slower updates
        gamma=0.995,             # longer-term discounting
        gae_lambda=0.90,         # bias-variance trade-off
        seed=RANDOM_SEED,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )
    model.set_logger(sb3_logger)

    logger.info(f"Training PPO on {len(train_df)} clean rows …")
    model.learn(total_timesteps=len(train_df) * 3)

    model.save(save_path)
    logger.info(f"Model saved → {save_path}")



# ──────────────────────────────────────────────────────────
def _load_or_train(current_timestamp: str | None = None) -> PPO:
    """
    - If `current_timestamp` is provided:
        • Load the full CSV
        • Filter to rows < current_timestamp
        • Pass that slice directly to train_agent()
        • Load & cache the fresh model
    - Otherwise:
        • Load from cache or disk
    """
    ticker     = TICKERS[0]
    model_file = f"ppo_{ticker}_{CONVERTED_TIMEFRAME}.zip"

    # Online‐retraining mode
    if current_timestamp is not None:
        full_df = load_full_csv(ticker)
        cutoff  = pd.to_datetime(current_timestamp)
        slice_df = full_df[full_df["timestamp"] < cutoff]

        # Train on that slice (no double-filtering)
        train_agent(slice_df, model_file)

        model = PPO.load(model_file)
        _model_cache[ticker] = model
        return model

    # Static load-only mode
    if ticker in _model_cache:
        return _model_cache[ticker]
    if os.path.exists(model_file):
        model = PPO.load(model_file)
        _model_cache[ticker] = model
        return model

    raise ValueError(
        "Model not found on disk. "
        "First call backtest with a timestamp so we can train it."
    )



# ────────────────────────────────#
# Live‑feature helper
# ────────────────────────────────#
def _latest_feature_vector(ticker: str) -> Tuple[np.ndarray, pd.Series]:
    df  = load_full_csv(ticker)
    row = df.iloc[-1].copy()
    return row.drop("timestamp").astype(np.float32).values, row


# ────────────────────────────────#
#  ███  PUBLIC API  ███
# ────────────────────────────────#
def run_logic(
    current_price: float,
    predicted_price: float,
    ticker: str
) -> None:
    """
    Live trading call → side-effects via forest.buy_shares / sell_shares.
    """
    global FEATURE_COLUMNS

    # 1) Load the model (train if needed)
    model = _load_or_train()

    # 2) If FEATURE_COLUMNS wasn't set (loading a pre-trained model), infer it
    if not FEATURE_COLUMNS:
        df = load_full_csv(ticker)
        FEATURE_COLUMNS = df.columns.drop("timestamp").tolist()

    # 3) Get latest features & override
    df  = load_full_csv(ticker)
    row = df.iloc[-1].copy()
    row["close"]           = current_price
    row["predicted_close"] = predicted_price

    # 4) Slice same features
    obs = row[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    # 5) Predict action
    action, _ = model.predict(obs, deterministic=True)

    # 6) Execute via your forest API
    import importlib
    forest = importlib.import_module("forest")
    try:
        cash     = float(forest.api.get_account().cash)
        held_qty = float(forest.api.get_position(ticker).qty or 0.0)
    except Exception as e:
        logger.error(f"[{ticker}] API error: {e}")
        return

    if action == 1 and held_qty == 0:
        qty = int(cash // current_price)
        if qty > 0:
            forest.buy_shares(ticker, qty, current_price, predicted_price)
    elif action == 2 and held_qty > 0:
        forest.sell_shares(ticker, held_qty, current_price, predicted_price)
    # else NONE → do nothing


# ────────────────────────────────#
#  run_backtest
# ────────────────────────────────#
def run_backtest(
    current_price: float,
    predicted_price: float,
    position_qty: float,
    current_timestamp: str,
    candles: pd.DataFrame
) -> str:
    """
    For each test candle:
      1) Retrain on the *full* CSV history < current_timestamp.
      2) Reload that model.
      3) Predict the action for this candle.
      4) Return "BUY"/"SELL"/"NONE".
    """
    global FEATURE_COLUMNS

    # 1) Retrain (if possible) and load a fresh model
    model = _load_or_train(current_timestamp)

    # 2) Ensure FEATURE_COLUMNS is set
    if not FEATURE_COLUMNS:
        df = load_full_csv(TICKERS[0])
        FEATURE_COLUMNS = df.columns.drop("timestamp").tolist()

    # 3) Locate this exact candle in the sliding window
    row = candles.loc[candles["timestamp"] == current_timestamp].iloc[0].copy()
    #    Override the live prices:
    row["close"]           = current_price
    row["predicted_close"] = predicted_price

    # 4) Build the observation vector in the original training order
    obs = row[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    # 5) Ask the policy
    action, _ = model.predict(obs, deterministic=True)

    # 6) Map numeric → string
    if action == 1 and position_qty == 0:
        return "BUY"
    if action == 2 and position_qty > 0:
        return "SELL"
    return "NONE"


# ────────────────────────────────#
# Minimal reproducible example
# ────────────────────────────────#
if __name__ == "__main__":
    """
    $ python rl_logic_best.py
    Trains the agent once using a 90 / 10 chronological split and runs an
    illustrative looped back‑test (no PnL printout, just a sanity smoke test).
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=TICKERS[0],
                        help="Symbol (default picks env var)")
    args = parser.parse_args()

    ticker = args.ticker
    df = load_full_csv(ticker)

    split = int(len(df) * 0.90)
    train_df = df.iloc[:split]
    test_df  = df.iloc[split:]

    train_agent(test_df, f"ppo_{ticker}_{CONVERTED_TIMEFRAME}.zip")

    position_qty = 0
    for _, r in test_df.iterrows():
        action = run_backtest(
            current_price   = float(r["close"]),
            predicted_price = float(r["predicted_close"]),
            position_qty    = position_qty,
            current_timestamp = str(r["timestamp"]),
            candles         = test_df,
        )
        if action == "BUY":
            position_qty = int(INITIAL_CASH // r["close"])
        elif action == "SELL":
            position_qty = 0

    logger.info("Demo walk‑forward loop completed.")
