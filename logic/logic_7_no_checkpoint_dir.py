"""
RL trading script for TSLA 4-hour candles using PPO (Stable-Baselines3).

Key points
----------
* Trains from scratch every call (run_logic / run_backtest)
* Uses ONLY the requested feature columns (excludes “predicted_close”)
* Three discrete actions: HOLD(0), BUY(1), SELL(2)
* Buys max affordable shares; sells entire position
* Reward:
    - If action ≠ HOLD  →  (current_close − previous_close) × trade_dir
    - If action == HOLD → –0.1   (idle penalty)
* n_steps = 1024, total_timesteps = 42_000  (≈ 30 passes over a 1-year window)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
load_dotenv()
DATA_DIR = Path("data")            # TSLA_H4.csv lives here
CSV_FILE = DATA_DIR / "TSLA_H4.csv"

INIT_CASH = 1_000.0
N_STEPS = 1024                     # rollout length
TOTAL_TIMESTEPS = 42_000           # ≈ 30 full passes over 1-yr window
FEATURE_COLS: List[str] = [
    "timestamp", "open", "high", "low", "close", "volume", "vwap",
    "transactions", "sentiment", "price_change", "high_low_range",
    "log_volume", "macd_line", "macd_signal", "macd_histogram", "rsi",
    "momentum", "roc", "atr", "ema_9", "ema_21", "ema_50", "ema_200",
    "adx", "obv", "bollinger_upper", "bollinger_lower", "lagged_close_1",
    "lagged_close_2", "lagged_close_3", "lagged_close_5", "lagged_close_10",
    "candle_body_ratio", "wick_dominance", "gap_vs_prev", "volume_zscore",
    "atr_zscore", "rsi_zscore", "adx_trend", "macd_cross", "macd_hist_flip",
    "day_of_week", "days_since_high", "days_since_low",
]
# NB: timestamp will be dropped before feeding the network

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

BUY, SELL, HOLD = 1, 2, 0
ACTION_MAP = {HOLD: "NONE", BUY: "BUY", SELL: "SELL"}

# --------------------------------------------------------------------------- #
# Data utilities
# --------------------------------------------------------------------------- #

def load_tsla_data() -> pd.DataFrame:
    """Load and clean TSLA 4-hour candle data."""
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"CSV not found at {CSV_FILE}")
    df = pd.read_csv(CSV_FILE, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Keep ONLY requested cols (drop predicted_close if present)
    keep_cols = [c for c in FEATURE_COLS if c in df.columns]
    df = df[keep_cols].copy()

    # Forward/back-fill missing
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.ffill().bfill()
    df.dropna(inplace=True)

    return df


# --------------------------------------------------------------------------- #
# Custom Gym environment
# --------------------------------------------------------------------------- #

class TSLAEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.n_rows = len(df)

        self.feature_cols = [c for c in df.columns if c != "timestamp"]
        self.obs_dim = len(self.feature_cols) + 2  # features + cash + position_qty

        self.action_space = gym.spaces.Discrete(3)  # 0-HOLD, 1-BUY, 2-SELL
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        self.reset()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _row(self, idx: int) -> pd.Series:
        return self.df.iloc[idx]

    def _current_price(self) -> float:
        return float(self._row(self.idx)["close"])

    def _get_obs(self) -> np.ndarray:
        features = self._row(self.idx)[self.feature_cols].astype(np.float32).values
        return np.concatenate([features, [self.cash, self.position_qty]], axis=0)

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.idx = 0
        self.cash = INIT_CASH
        self.position_qty = 0
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert action in (HOLD, BUY, SELL)
        price = self._current_price()

        # -------- trading logic -------- #
        if action == BUY and self.position_qty == 0:
            max_shares = int(self.cash // price)
            if max_shares == 0:  # can't afford even one
                action = HOLD
            else:
                self.position_qty = max_shares
                self.cash -= max_shares * price

        elif action == SELL and self.position_qty > 0:
            self.cash += self.position_qty * price
            self.position_qty = 0

        else:
            action = HOLD  # invalid trade converts to HOLD

        # -------- reward logic -------- #
        if self.idx >= 1:
            prev_close = float(self._row(self.idx - 1)["close"])
        else:
            prev_close = price

        if action != HOLD:
            trade_dir = 1 if action == BUY else -1
            reward = (price - prev_close) * trade_dir
        else:
            reward = -0.1  # idle penalty

        # -------- advance time -------- #
        self.idx += 1
        done = self.idx >= self.n_rows

        obs = self._get_obs() if not done else np.zeros(self.obs_dim, dtype=np.float32)
        return obs, reward, done, False, {}

# --------------------------------------------------------------------------- #
# PPO training helper
# --------------------------------------------------------------------------- #

def train_ppo(df: pd.DataFrame) -> PPO:
    env = DummyVecEnv([lambda: TSLAEnv(df)])
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=N_STEPS,
        batch_size=128,
        n_epochs=10,
        learning_rate=2.5e-4,
        gamma=0.97,
        gae_lambda=0.92,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        verbose=0,
        device="cpu"
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    return model


# --------------------------------------------------------------------------- #
# Strategy entry-points
# --------------------------------------------------------------------------- #

def _make_observation(row: pd.Series, cash: float, qty: float, feature_cols: List[str]) -> np.ndarray:
    features = row[feature_cols].astype(np.float32).values
    return np.concatenate([features, [cash, qty]], axis=0)

# --------------------------------------------------------------------------- #
# LIVE / PAPER-TRADING
# --------------------------------------------------------------------------- #
def run_logic(current_price: float, predicted_price: float, ticker: str):
    """
    Called in live-trading context.

    * Trains PPO on the **entire CSV** every call.
    * Uses the most recent candle (with current_price patched in) to choose action.
    * Executes trade via `forest` SDK.
    """
    import logging
    from forest import api, buy_shares, sell_shares  # brokerage SDK

    logger = logging.getLogger(__name__)

    # Load / train
    df = load_tsla_data()
    df.loc[df.index[-1], "close"] = current_price  # patch latest close
    agent = train_ppo(df)

    # Account snapshot
    try:
        account = api.get_account()
        cash = float(account.cash)
    except Exception as e:
        logger.error(f"[{ticker}] Error fetching account details: {e}")
        return

    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception:
        position_qty = 0.0

    obs = _make_observation(df.iloc[-1], cash, position_qty, [c for c in df.columns if c != "timestamp"])

    action, _ = agent.predict(obs, deterministic=True)
    action = int(action)
    action_name = ACTION_MAP[action]

    logger.info(f"[{ticker}] PPO action={action_name} | Px={current_price:.2f} | Qty={position_qty} | Cash={cash:.2f}")

    if action == BUY and position_qty == 0:
        qty = int(cash // current_price)
        if qty:
            logger.info(f"[{ticker}] Buying {qty} shares at {current_price}")
            buy_shares(ticker, qty, current_price, predicted_price)
    elif action == SELL and position_qty > 0:
        logger.info(f"[{ticker}] Selling {position_qty} shares at {current_price}")
        sell_shares(ticker, position_qty, current_price, predicted_price)
    else:
        logger.info(f"[{ticker}] No trade executed (HOLD).")


# --------------------------------------------------------------------------- #
# BACKTEST
# --------------------------------------------------------------------------- #
def run_backtest(
    current_price: float,
    predicted_price: float,
    position_qty: float,
    current_timestamp: Union[str, pd.Timestamp],
    candles,
) -> str:
    """
    Backtest entry-point.

    Requirements:
    * Re-train **from scratch** on data **≤ current_timestamp**.
    * NO future data leakage.
    * Decide BUY / SELL / NONE based on PPO output + current position.
    """
    if isinstance(current_timestamp, str):
        current_timestamp = pd.Timestamp(current_timestamp)

    df_full = load_tsla_data()
    df_train = df_full[df_full["timestamp"] <= current_timestamp].copy()
    if len(df_train) < 100:  # too little history
        return "NONE"

    # Ensure the last row's close equals current_price being tested
    df_train.loc[df_train.index[-1], "close"] = current_price

    agent = train_ppo(df_train)

    obs = _make_observation(
        df_train.iloc[-1],
        cash=0.0 if position_qty > 0 else INIT_CASH,
        qty=position_qty,
        feature_cols=[c for c in df_train.columns if c != "timestamp"],
    )

    action, _ = agent.predict(obs, deterministic=True)
    action = int(action)

    if action == BUY and position_qty == 0:
        return "BUY"
    if action == SELL and position_qty > 0:
        return "SELL"
    return "NONE"
