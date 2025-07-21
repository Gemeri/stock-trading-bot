"""
ppo_lstm_tsla_trader.py
-------------------------------------------------
Recurrent-PPO (LSTM) agent for 4-hour TSLA candles
• Uses ONLY the requested feature set (ignores `predicted_close`)
• Starts each episode with \$1 000 cash, no shares
• BUY / SELL / HOLD logic exactly as specified
• Reward = directional price move vs. previous close, –0.1 for idleness
• Functions `run_logic` and `run_backtest` appear at the bottom
-------------------------------------------------
"""

from __future__ import annotations
import logging
import os
from typing import List, Tuple, Dict, Any

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import RecurrentPPO
from stable_baselines3.common.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS & HELPERS
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join("data", "TSLA_H4.csv")
INITIAL_BALANCE = 1_000.0
FEATURES: List[str] = [
    "timestamp", "open", "high", "low", "close", "volume", "vwap", "transactions",
    "sentiment", "price_change", "high_low_range", "log_volume",
    "macd_line", "macd_signal", "macd_histogram",
    "rsi", "momentum", "roc", "atr",
    "ema_9", "ema_21", "ema_50", "ema_200",
    "adx", "obv",
    "bollinger_upper", "bollinger_lower",
    "lagged_close_1", "lagged_close_2", "lagged_close_3",
    "lagged_close_5", "lagged_close_10",
    "candle_body_ratio", "wick_dominance", "gap_vs_prev",
    "volume_zscore", "atr_zscore", "rsi_zscore",
    "adx_trend", "macd_cross", "macd_hist_flip",
    "day_of_week", "days_since_high", "days_since_low",
]
# columns actually seen by the agent (timestamp dropped; two extra slots appended
# for scaled cash & position qty):
OBS_FEATURES: List[str] = [col for col in FEATURES if col != "timestamp"]

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s — %(levelname)s — %(message)s",
                    level=logging.INFO)


def _load_data() -> pd.DataFrame:
    """Load the CSV, enforce the feature list, drop `predicted_close` if present."""
    df = pd.read_csv(DATA_PATH)
    if "predicted_close" in df.columns:
        df = df.drop(columns=["predicted_close"])
    # keep only the requested columns (plus timestamp for integrity)
    df = df[[c for c in df.columns if c in FEATURES]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True, ignore_index=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM TRADING ENVIRONMENT
# ──────────────────────────────────────────────────────────────────────────────
class TradingEnv(gym.Env):
    """Minimal 3-action trading environment for TSLA 4-hour candles."""

    metadata = {"render.modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = INITIAL_BALANCE,
    ) -> None:
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance

        obs_dim = len(OBS_FEATURES) + 2  # market features + cash_scaled + pos_scaled
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)  # 0 = HOLD, 1 = BUY, 2 = SELL

        # internal state
        self._idx: int
        self._cash: float
        self._shares: int
        self._previous_close: float

    # --------------------------------------------------------------------- reset
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._idx = 0
        self._cash = self.initial_balance
        self._shares = 0
        self._previous_close = float(self.df.iloc[self._idx]["close"])
        observation = self._get_observation()
        return observation, {}

    # ---------------------------------------------------------------------- step
    def step(self, action: int):
        done = False
        truncated = False
        info: Dict[str, Any] = {}

        current_row = self.df.iloc[self._idx]
        current_price = float(current_row["close"])

        # ── enforce “BUY max, SELL all, otherwise HOLD” rules
        executed_action = action
        if action == 1 and self._shares > 0:          # attempted BUY while long
            executed_action = 0                       # HOLD instead
        if action == 2 and self._shares == 0:         # attempted SELL while flat
            executed_action = 0                       # HOLD instead

        # ── reward logic
        if executed_action != 0:
            trade_dir = 1 if executed_action == 1 else -1
            reward = (current_price - self._previous_close) * trade_dir
        else:
            reward = -0.1  # slight penalty for idleness

        # ── portfolio update (after reward is calculated)
        if executed_action == 1:                      # BUY
            max_shares = int(self._cash // current_price)
            if max_shares > 0:
                self._shares += max_shares
                self._cash -= max_shares * current_price
        elif executed_action == 2:                    # SELL
            if self._shares > 0:
                self._cash += self._shares * current_price
                self._shares = 0

        # advance time
        self._previous_close = current_price
        self._idx += 1
        if self._idx >= len(self.df) - 1:
            done = True

        observation = self._get_observation()
        return observation, reward, done, truncated, info

    # ---------------------------------------------------------- observation helper
    def _get_observation(self) -> np.ndarray:
        row = self.df.iloc[self._idx]
        market_feats = row[OBS_FEATURES].astype(np.float32).values
        cash_scaled = np.float32(self._cash / 10_000.0)   # simple rescale
        pos_scaled = np.float32(self._shares / 1_000.0)
        obs = np.concatenate([market_feats, [cash_scaled, pos_scaled]]).astype(np.float32)
        return obs


# ──────────────────────────────────────────────────────────────────────────────
# MODEL TRAIN / PREDICT HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _train_agent(
    env: gym.Env,
    total_timesteps: int = 42_000,
    seed: int = 42,
) -> RecurrentPPO:
    """Train a fresh Recurrent-PPO (LSTM) agent on `env`."""
    model = RecurrentPPO(
        policy=RecurrentActorCriticPolicy,
        env=DummyVecEnv([lambda: env]),
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
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    return model


def _make_observation_vector(
    row: pd.Series, cash: float, position_qty: float
) -> np.ndarray:
    """Build the same observation vector the env would supply."""
    feats = row[OBS_FEATURES].astype(np.float32).values
    cash_scaled = np.float32(cash / 10_000.0)
    pos_scaled = np.float32(position_qty / 1_000.0)
    return np.concatenate([feats, [cash_scaled, pos_scaled]]).reshape(1, -1)


def _action_to_str(action: int) -> str:
    return ("NONE", "BUY", "SELL")[action]


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY-POINT: OPTIONAL TRAIN-AND-SAVE (for standalone runs)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _df = _load_data()
    _env = TradingEnv(_df)
    _ = _train_agent(_env)  # model discarded; this block is illustrative


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC API FUNCTIONS (used by external orchestration scripts)
# ──────────────────────────────────────────────────────────────────────────────
def run_logic(current_price: float, predicted_price: float, ticker: str):
    """
    • Trains on the *entire* CSV each call.
    • Uses the trained agent to decide BUY / SELL / HOLD.
    • Executes trades through `forest.api`.
    NOTE: `predicted_price` is kept in the signature for compatibility but is
    **NOT** used by the RL agent.
    """
    # forest broker-API imports (assumed provided in runtime environment)
    from forest import api, buy_shares, sell_shares

    # broker state
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

    # ── environment & training (uses full dataset)
    df = _load_data()
    env = TradingEnv(df)
    model = _train_agent(env)

    # latest observation (most recent candle)
    last_row = df.iloc[-1]
    obs = _make_observation_vector(last_row, cash, position_qty)
    action, _ = model.predict(obs, deterministic=True)
    action = int(action)

    logger.info(
        f"[{ticker}] RL decided action={_action_to_str(action)} | "
        f"Price={current_price} | Pos={position_qty} | Cash={cash}"
    )

    # ── execution respecting account constraints
    if action == 1:  # BUY
        if position_qty == 0:
            max_shares = int(cash // current_price)
            if max_shares > 0:
                logger.info(f"[{ticker}] Buying {max_shares} shares at {current_price}.")
                buy_shares(ticker, max_shares, current_price, predicted_price)
            else:
                logger.info(f"[{ticker}] Insufficient cash to buy.")
        else:
            logger.info(f"[{ticker}] Already long; BUY skipped.")
    elif action == 2:  # SELL
        if position_qty > 0:
            logger.info(f"[{ticker}] Selling {position_qty} shares at {current_price}.")
            sell_shares(ticker, position_qty, current_price, predicted_price)
        else:
            logger.info(f"[{ticker}] No shares to sell; SELL skipped.")
    else:  # HOLD
        logger.info(f"[{ticker}] HOLD action taken.")


def run_backtest(
    current_price: float,
    predicted_price: float,
    position_qty: float,
    current_timestamp: str,
    candles: Any,
) -> str:
    """
    Proper backtest routine:
    • Filters CSV to *strictly earlier* than `current_timestamp`
      (prevents future data leakage).
    • Trains agent from scratch on that subset.
    • Builds observation for `current_timestamp` row.
    • Returns "BUY" / "SELL" / "NONE" per spec.
    Only `current_timestamp` and `position_qty` are actually used; other
    parameters are preserved for signature compatibility.
    """
    df_full = _load_data()
    ts = pd.to_datetime(current_timestamp)

    df_train = df_full[df_full["timestamp"] < ts].copy()
    df_pred_row = df_full[df_full["timestamp"] == ts]
    if df_pred_row.empty or df_train.empty:
        raise ValueError("Current timestamp not found or insufficient training data.")

    # env & training on historical subset
    env = TradingEnv(df_train)
    model = _train_agent(env)

    # observation for prediction timestamp
    obs = _make_observation_vector(df_pred_row.iloc[0], INITIAL_BALANCE, position_qty)
    action, _ = model.predict(obs, deterministic=True)
    action = int(action)

    # map to BUY / SELL / NONE, with position checks
    if action == 1 and position_qty == 0:
        return "BUY"
    if action == 2 and position_qty > 0:
        return "SELL"
    return "NONE"
