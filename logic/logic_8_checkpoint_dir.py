"""
================================================================================
PPO-LSTM TRADING AGENT  –  TSLA 4-hour candles
--------------------------------------------------------------------------------
• Uses sb3-contrib RecurrentPPO with an LSTM policy exactly as configured below.
• Custom Gym environment implements the 3-action logic (BUY, SELL, HOLD/NONE),
  portfolio bookkeeping (cash, shares) and the **reward rule** you supplied.
• Loads **only** the whitelisted feature columns from `data/TSLA_H4.csv`
  (drops `predicted_close` if it is present).

Training cadence
----------------
* A fresh model is fit every **30 candles**.
* The live/check-pointed model is stored in `ppo_lstm_tsla.zip`.
* A simple text file `retrain_counter.txt` keeps the remaining-candles counter.

Public API  (called externally)
-------------------------------
1. `run_logic(current_price, predicted_price, ticker)`
      • For live trading.  Re-trains on **all** available candles whenever
        the counter hits 0, then decrements the counter and places trades
        with the broker helper in `forest`.

2. `run_backtest(current_timestamp, position_qty)`
      • Used by your back-tester, once per candle.
      • Ensures the model **never** sees data newer than `current_timestamp`.
      • Same 30-candle retrain cadence, counter shared with run_logic.
      • Returns the string **"BUY" | "SELL" | "NONE"**.

Both functions respect the “max-buy / sell-all / default-to-HOLD” rules.

Dependencies
------------
pip install pandas numpy gym sb3-contrib stable-baselines3==2.3.0
================================================================================
"""

import os
import logging
import pathlib
from datetime import datetime

import gym
import numpy as np
import pandas as pd
from gym import spaces
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

# -----------------------------------------------------------------------------#
# Configuration constants
# -----------------------------------------------------------------------------#
DATA_PATH = pathlib.Path("data/TSLA_H4.csv")
CHECKPOINT_PATH = pathlib.Path("ppo_lstm_tsla.zip")
COUNTER_PATH = pathlib.Path("retrain_counter.txt")
RETRAIN_EVERY = 30           # candles
STARTING_CASH = 1_000.0      # USD for the environment

FEATURE_COLS = [
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

NUMERICAL_FEATURES = [c for c in FEATURE_COLS if c != "timestamp"]

ACTION_HOLD, ACTION_BUY, ACTION_SELL = 0, 1, 2
ACTION_MAP = {ACTION_HOLD: "NONE", ACTION_BUY: "BUY", ACTION_SELL: "SELL"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PPO-LSTM-TSLA")


# -----------------------------------------------------------------------------#
# Utility helpers
# -----------------------------------------------------------------------------#
def _load_counter() -> int:
    if COUNTER_PATH.exists():
        try:
            return int(COUNTER_PATH.read_text().strip())
        except ValueError:
            pass
    return RETRAIN_EVERY


def _save_counter(value: int) -> None:
    COUNTER_PATH.write_text(str(value))


def _load_csv(until_ts: datetime | None = None) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # Parse and filter timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if until_ts is not None:
        df = df[df["timestamp"] <= until_ts]
    # Drop any rogue columns
    cols_to_use = [c for c in FEATURE_COLS if c in df.columns]
    df = df[cols_to_use]
    # Ensure NO predicted_close
    df = df[[c for c in df.columns if c != "predicted_close"]]
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# -----------------------------------------------------------------------------#
# Custom Gym environment
# -----------------------------------------------------------------------------#
class TslaTradingEnv(gym.Env):
    """
    4-hour-candle TSLA trading environment
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        assert {"close", "timestamp"}.issubset(df.columns)
        self.df = df.copy().reset_index(drop=True)
        self.n_features = len(NUMERICAL_FEATURES) + 2  # cash & position_qty

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self.current_step = 1  # start at 1 so we always have a previous row
        self.cash = STARTING_CASH
        self.position_qty = 0.0
        self.previous_close = float(self.df.iloc[0]["close"])

    # --------------------------------------------------------------------- #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 1
        self.cash = STARTING_CASH
        self.position_qty = 0.0
        self.previous_close = float(self.df.iloc[0]["close"])
        return self._get_obs(), {}

    # --------------------------------------------------------------------- #
    def step(self, action: int):
        row = self.df.iloc[self.current_step]
        current_price = float(row["close"])

        # Default values
        reward = -0.1
        trade_dir = 0

        # Execute action subject to inventory rules
        if action == ACTION_BUY:
            if self.position_qty == 0:
                max_shares = int(self.cash // current_price)
                if max_shares > 0:
                    self.position_qty = max_shares
                    self.cash -= max_shares * current_price
                    trade_dir = 1
                else:
                    action = ACTION_HOLD  # insufficient cash
            else:
                action = ACTION_HOLD  # already long
        elif action == ACTION_SELL:
            if self.position_qty > 0:
                self.cash += self.position_qty * current_price
                self.position_qty = 0
                trade_dir = -1
            else:
                action = ACTION_HOLD  # nothing to sell

        # Reward logic
        if action != ACTION_HOLD:
            reward = (current_price - self.previous_close) * trade_dir
        # otherwise reward already = -0.1

        # Prepare for next step
        self.previous_close = current_price
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        info = {}

        return self._get_obs(), reward, done, False, info

    # --------------------------------------------------------------------- #
    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        features = row[NUMERICAL_FEATURES].astype(np.float32).to_numpy()
        obs = np.concatenate(
            [features, np.array([self.cash], dtype=np.float32), np.array([self.position_qty], dtype=np.float32)]
        )
        return obs


# -----------------------------------------------------------------------------#
# Model (re-)training
# -----------------------------------------------------------------------------#
def _train_model(df: pd.DataFrame) -> RecurrentPPO:
    env = TslaTradingEnv(df)
    model = RecurrentPPO(
        policy=RecurrentActorCriticPolicy,
        env=env,
        verbose=1,
        seed=42,
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
    model.learn(total_timesteps=42_000, progress_bar=True)
    model.save(CHECKPOINT_PATH)
    _save_counter(RETRAIN_EVERY)
    logger.info("Model trained & checkpointed.")
    return model


def _load_model() -> RecurrentPPO | None:
    if CHECKPOINT_PATH.exists():
        return RecurrentPPO.load(CHECKPOINT_PATH, print_system_info=False)
    return None


def _maybe_retrain(train_df: pd.DataFrame, force_retrain: bool = False) -> RecurrentPPO:
    counter = _load_counter()
    model = _load_model()

    if counter <= 0 or model is None or force_retrain:
        logger.info("Retraining model (counter reset).")
        model = _train_model(train_df)
        counter = RETRAIN_EVERY
    else:
        counter -= 1
        _save_counter(counter)
        logger.info("Using existing model – %d candles until next retrain.", counter)
    return model


# -----------------------------------------------------------------------------#
# === PUBLIC FUNCTIONS ========================================================#
# -----------------------------------------------------------------------------#
def run_logic(current_price: float, predicted_price: float, ticker: str):
    """
    Live-trading entry-point.
    • Re-trains on the *full* CSV every 30 calls.
    • Executes trades through `forest.buy_shares` / `forest.sell_shares`.
    """
    from forest import api, buy_shares, sell_shares  # type: ignore

    # Ensure up-to-date model (full data)
    full_df = _load_csv()
    model = _maybe_retrain(full_df)

    # Build observation from the *last* row in the CSV
    last_row = full_df.iloc[-1]
    obs_features = last_row[NUMERICAL_FEATURES].astype(np.float32).to_numpy()
    # Fetch portfolio state
    account = api.get_account()
    cash = float(account.cash)
    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception:
        position_qty = 0.0

    obs = np.concatenate(
        [obs_features, np.array([cash], dtype=np.float32), np.array([position_qty], dtype=np.float32)]
    )

    action, _ = model.predict(obs, deterministic=True)
    action = int(action)

    logger.info(
        "[%s] Live — model action: %s | Price: %.2f | Cash: %.2f | Pos: %.0f | PredProb: %.2f",
        ticker,
        ACTION_MAP[action],
        current_price,
        cash,
        position_qty,
        predicted_price,
    )

    if action == ACTION_BUY and position_qty == 0:
        max_shares = int(cash // current_price)
        if max_shares > 0:
            buy_shares(ticker, max_shares, current_price, predicted_price)
    elif action == ACTION_SELL and position_qty > 0:
        sell_shares(ticker, position_qty, current_price, predicted_price)
    else:
        logger.info("[%s] No trade executed.", ticker)


def run_backtest(current_timestamp: str, position_qty: float) -> str:
    """
    Back-tester entry-point.  
    • `current_timestamp` is the MOST RECENT candle available to the agent.  
    • The model is *never* trained on data newer than that timestamp.  
    • Returns "BUY", "SELL" or "NONE".
    """
    ts = pd.to_datetime(current_timestamp, utc=True)
    df_upto_now = _load_csv(until_ts=ts)

    if len(df_upto_now) < 50:  # safeguard: need enough data to learn
        return "NONE"

    model = _maybe_retrain(df_upto_now)

    last_row = df_upto_now.iloc[-1]
    obs_features = last_row[NUMERICAL_FEATURES].astype(np.float32).to_numpy()
    cash_dummy = STARTING_CASH if position_qty == 0 else 0.0  # placeholder cash
    obs = np.concatenate(
        [obs_features, np.array([cash_dummy], dtype=np.float32), np.array([position_qty], dtype=np.float32)]
    )

    action, _ = model.predict(obs, deterministic=True)
    action = int(action)

    # Enforce inventory rules
    if action == ACTION_BUY and position_qty > 0:
        action = ACTION_HOLD
    if action == ACTION_SELL and position_qty == 0:
        action = ACTION_HOLD

    return ACTION_MAP[action]


# -----------------------------------------------------------------------------#
# Main-guard (optional quick smoke-test)
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    df_full = _load_csv()
    _train_model(df_full)
    print("Initial model trained.  Counter reset to", _load_counter())
