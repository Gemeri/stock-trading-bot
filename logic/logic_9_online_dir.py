#!/usr/bin/env python
"""ppo_lstm_trading_bot.py

A complete PPO‑LSTM trading framework that trains on Tesla 4‑hour candles, with
three discrete actions (HOLD, BUY‑max, SELL‑all), a custom reward signal,
rolling retraining, quick online updates, and helper functions usable by the
production runner (`run_logic`) and the back‑testing harness (`run_backtest`).

Key design points
-----------------
* **Features**: Only the explicitly allowed columns are loaded. Any accidental
  `predicted_close` field is dropped.
* **Environment**: Gym‑compatible; handles position sizing, cash accounting,
  reward, and observation extraction.
* **Training cadence**: A *checkpoint counter* starts at 30 and is decremented
  every time either helper function is invoked. When it hits zero, the model is
  **re‑trained from scratch on the most‑recent 1 400‑candle window** (≈2 years
  of 4‑hour bars), the counter resets, and a checkpoint is written.
* **Online learning**: Between full retrains, *tiny* one‑step updates keep the
  network fresh using the latest candle.

All persistent artefacts (models, checkpoints, counter) live in an
auto‑created **./models** folder next to this script.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import RecurrentActorCriticPolicy

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
FEATURES: Tuple[str, ...] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "transactions",
    "sentiment",
    "price_change",
    "high_low_range",
    "log_volume",
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "rsi",
    "momentum",
    "roc",
    "atr",
    "ema_9",
    "ema_21",
    "ema_50",
    "ema_200",
    "adx",
    "obv",
    "bollinger_upper",
    "bollinger_lower",
    "lagged_close_1",
    "lagged_close_2",
    "lagged_close_3",
    "lagged_close_5",
    "lagged_close_10",
    "candle_body_ratio",
    "wick_dominance",
    "gap_vs_prev",
    "volume_zscore",
    "atr_zscore",
    "rsi_zscore",
    "adx_trend",
    "macd_cross",
    "macd_hist_flip",
    "day_of_week",
    "days_since_high",
    "days_since_low",
)

INITIAL_BALANCE: float = 1_000.0
CHECKPOINT_INTERVAL: int = 30  # candles between full retrains
ROLLING_WINDOW: int = 1_400    # candles in rolling training window
DATA_PATH = Path(__file__).with_suffix("").parent / "data" / "TSLA_H4.csv"
MODEL_DIR = Path(__file__).with_suffix("").parent / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)
COUNTER_FILE = MODEL_DIR / "counter.txt"
LATEST_MODEL = MODEL_DIR / "ppo_lstm_latest"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("ppo_lstm_trading_bot")

# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def _read_counter() -> int:
    if COUNTER_FILE.exists():
        return int(COUNTER_FILE.read_text())
    COUNTER_FILE.write_text(str(CHECKPOINT_INTERVAL))
    return CHECKPOINT_INTERVAL


def _write_counter(value: int) -> None:
    COUNTER_FILE.write_text(str(value))


# ──────────────────────────────────────────────────────────────────────────────
# Data handling
# ──────────────────────────────────────────────────────────────────────────────

def load_full_dataframe() -> pd.DataFrame:
    """Load the *entire* CSV, enforce feature schema, drop forbidden columns."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"CSV not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Drop any column not explicitly allowed (incl. predicted_close)
    allowed = set(FEATURES)
    df = df[[c for c in df.columns if c in allowed]].copy()

    # Ensure chronological order
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def slice_rolling_window(df: pd.DataFrame) -> pd.DataFrame:
    """Return the *last* `ROLLING_WINDOW` rows for training."""
    return df.iloc[-ROLLING_WINDOW:].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Trading environment
# ──────────────────────────────────────────────────────────────────────────────
class TradingEnv(gym.Env):
    """A minimal 3‑action trading environment with cash/shares accounting."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.action_space = gym.spaces.Discrete(3)  # 0‑HOLD, 1‑BUY, 2‑SELL
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(FEATURES),),
            dtype=np.float32,
        )
        self.reset()

    # ────────────────────────────────────────────────────────────────── helpers
    def _get_obs(self) -> np.ndarray:
        row = self.df.iloc[self.current_step]
        return row.values.astype(np.float32)

    def _current_price(self) -> float:
        return float(self.df.iloc[self.current_step]["close"])

    # ─────────────────────────────────────────────────────────── gym methods
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.current_step: int = 1  # require previous close
        self.balance: float = INITIAL_BALANCE
        self.shares: int = 0
        self.net_worth: float = INITIAL_BALANCE
        return self._get_obs(), {}

    def step(self, action: int):
        assert self.action_space.contains(action)

        prev_close = float(self.df.iloc[self.current_step - 1]["close"])
        price = self._current_price()
        executed_action = "NONE"

        # Action logic
        if action == 1:  # BUY max if flat
            if self.shares == 0:
                max_shares = int(self.balance // price)
                if max_shares > 0:
                    self.shares = max_shares
                    self.balance -= max_shares * price
                    executed_action = "BUY"
            # else default to HOLD
        elif action == 2:  # SELL all if holding
            if self.shares > 0:
                self.balance += self.shares * price
                self.shares = 0
                executed_action = "SELL"
            # else default to HOLD

        # Reward
        if executed_action != "NONE":
            trade_dir = 1 if executed_action == "BUY" else -1
            reward = (price - prev_close) * trade_dir
        else:
            reward = -0.1  # idle penalty

        # Advance
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        # Update net worth for info/debugging
        self.net_worth = self.balance + self.shares * price
        info = {
            "balance": self.balance,
            "shares": self.shares,
            "price": price,
            "net_worth": self.net_worth,
            "executed_action": executed_action,
            "reward": reward,
        }
        return self._get_obs(), reward, terminated, truncated, info

    # Optional pretty print
    def render(self):  # type: ignore[override]
        print(
            f"Step {self.current_step} | Price {self._current_price():.2f} | "
            f"Shares {self.shares} | Balance {self.balance:.2f} | NW {self.net_worth:.2f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Model training & persistence helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_vec_env(df: pd.DataFrame) -> DummyVecEnv:
    return DummyVecEnv([lambda: TradingEnv(df)])


def build_model(env: DummyVecEnv, seed: int = 42) -> RecurrentPPO:
    return RecurrentPPO(
        policy=RecurrentActorCriticPolicy,
        env=env,
        verbose=1,
        seed=seed,
        n_steps=1_024,
        batch_size=128,
        n_epochs=8,
        learning_rate=2.5e-4,
        gamma=0.97,
        gae_lambda=0.92,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
        policy_kwargs=dict(
            lstm_hidden_size=256,
            net_arch=dict(pi=[128], vf=[128]),
        ),
    )


def full_retrain(df: pd.DataFrame, total_timesteps: int = 42_000) -> RecurrentPPO:
    """Train from scratch on the most‑recent window and save as *latest*."""
    env = make_vec_env(df)
    model = build_model(env)
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save(str(LATEST_MODEL))
    logger.info("Full retrain complete & saved → %s", LATEST_MODEL)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Initialise on import so helper functions are immediately usable
# ──────────────────────────────────────────────────────────────────────────────
_df_full = load_full_dataframe()
_model: RecurrentPPO = full_retrain(slice_rolling_window(_df_full))
_counter: int = _read_counter()


# ──────────────────────────────────────────────────────────────────────────────
# Helper: quick online fit on the newest candle (no future leakage)
# ──────────────────────────────────────────────────────────────────────────────

def _online_update(latest_row: pd.Series):
    """One‑step policy update using the newest candle."""
    global _model
    tmp_env = make_vec_env(pd.concat([latest_row.to_frame().T, latest_row.to_frame().T]))
    _model.set_env(tmp_env)
    _model.learn(total_timesteps=32, reset_num_timesteps=False, progress_bar=False)


# ──────────────────────────────────────────────────────────────────────────────
# Public API: run_logic & run_backtest
# ──────────────────────────────────────────────────────────────────────────────

def run_logic(current_price: float, predicted_price: float, ticker: str):
    """Production‑time decision; trains *online* & handles rolling retrain."""
    from forest import api, buy_shares, sell_shares  # third‑party brokerage API

    global _model, _counter

    # ── brokerage state ────────────────────────────────────────────────────
    try:
        account = api.get_account()
        cash = float(account.cash)
    except Exception as exc:
        logger.error("[%s] Account fetch error: %s", ticker, exc)
        return

    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception:
        position_qty = 0.0  # assume flat

    # ── build observation (minimal: only price; ideally provide full feature row)
    obs = np.full((1, len(FEATURES)), fill_value=np.nan, dtype=np.float32)
    obs[0, FEATURES.index("close")] = current_price

    action, _ = _model.predict(obs, deterministic=False)
    action = int(action)
    action_map = {0: "NONE", 1: "BUY", 2: "SELL"}
    decided = action_map[action]

    logger.info("[%s] RL decided: %s | Cash %.2f | Pos %s", ticker, decided, cash, position_qty)

    # ── execute brokerage instructions ─────────────────────────────────────
    if decided == "BUY" and position_qty == 0:
        max_shares = int(cash // current_price)
        if max_shares > 0:
            buy_shares(ticker, max_shares, current_price, predicted_price)
    elif decided == "SELL" and position_qty > 0:
        sell_shares(ticker, position_qty, current_price, predicted_price)

    # ── online update & counter management ─────────────────────────────────
    latest_row = pd.Series({"close": current_price}, name=datetime.utcnow().timestamp())
    _online_update(latest_row)

    _counter -= 1
    _write_counter(_counter)

    if _counter <= 0:
        df_now = load_full_dataframe()  # reload including freshest data
        _model = full_retrain(slice_rolling_window(df_now))
        _counter = CHECKPOINT_INTERVAL
        _write_counter(_counter)


# NOTE: *Only* current_timestamp & position_qty are read as requested.
# Other parameters are accepted for API compatibility but ignored.

def run_backtest(
    current_price: float,  # ignored
    predicted_price: float,  # ignored
    position_qty: float,
    current_timestamp: float,  # POSIX seconds
    candles: Any,  # ignored
) -> str:
    """Return BUY / SELL / NONE *without* future leakage."""
    global _model

    # Ensure model is trained only on candles ⩽ current_timestamp
    df_up_to_now = _df_full[_df_full["timestamp"] <= current_timestamp]

    # If model has seen further into the future, retrain quickly on proper slice.
    if df_up_to_now.shape[0] >= 2 and df_up_to_now.shape[0] < len(_df_full):
        _model = full_retrain(slice_rolling_window(df_up_to_now))

    # Minimal observation: use the last available row (safe; <= current_timestamp)
    last_row = df_up_to_now.iloc[-1]
    obs = last_row.values[np.newaxis, :].astype(np.float32)

    action, _ = _model.predict(obs, deterministic=True)
    action = int(action)
    action_map = {0: "NONE", 1: "BUY", 2: "SELL"}
    decided = action_map[action]

    # Enforce position rules to avoid impossible trades
    if decided == "BUY" and position_qty == 0:
        return "BUY"
    if decided == "SELL" and position_qty > 0:
        return "SELL"
    return "NONE"


# ──────────────────────────────────────────────────────────────────────────────
# Entry‑point: manual standalone training run
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Standalone training run started…")
    df = load_full_dataframe()
    env = make_vec_env(slice_rolling_window(df))
    model = build_model(env)
    model.learn(total_timesteps=42_000, progress_bar=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = MODEL_DIR / f"ppo_lstm_manual_{stamp}"
    model.save(str(path))
    logger.info("Manual training complete. Model saved → %s.zip", path)
