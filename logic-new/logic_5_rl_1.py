# logic_rl_trade.py
"""
Reinforcement‑Learning‑driven trading logic — **bug‑fixed**.

This revision addresses the runtime `ValueError: … logits … contain NaN` by ensuring that every
observation fed to the PPO agent is finite.  All NaNs/Infs in the feature set are forward/backward
filled (then any residual rows are dropped) at load time.  An additional assert in the environment
makes debugging easier if corrupted data ever sneaks through.
"""
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional heavy deps – script will gracefully degrade if RL stack missing
# ---------------------------------------------------------------------------
try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    _RL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Environment‑wide constants & helpers
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()
BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "1Hour")
TICKERS = os.getenv("TICKERS", "TSLA").split(",")
TIMEFRAME_MAP = {"4Hour": "H4", "2Hour": "H2", "1Hour": "H1", "30Min": "M30", "15Min": "M15"}
CONVERTED_TIMEFRAME = TIMEFRAME_MAP.get(BAR_TIMEFRAME, BAR_TIMEFRAME)

MODEL_DIR = Path(os.getenv("MODEL_DIR", ".models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BUY, SELL, NONE = "BUY", "SELL", "NONE"

_logger = logging.getLogger(__name__)

# ----- feature schema (order matters for NN inputs) -----
FEATURE_COLS: List[str] = [
    "open","high","low","close","volume","vwap","macd_line","macd_signal","macd_histogram",
    "ema_9","ema_21","ema_50","ema_200","adx","rsi","momentum","roc","atr","obv",
    "bollinger_upper","bollinger_lower","lagged_close_1","lagged_close_2","lagged_close_3",
    "lagged_close_5","lagged_close_10","sentiment","predicted_close",
]

def get_csv_filename(ticker: str) -> str:
    return f"{ticker}_{CONVERTED_TIMEFRAME}.csv"

# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def _load_data(ticker: str) -> pd.DataFrame:
    csv_path = Path(get_csv_filename(ticker))
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file {csv_path} not found. Ensure data is downloaded.")

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Replace infinities then forward/back fill *all* features to avoid NaNs in obs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[FEATURE_COLS] = df[FEATURE_COLS].ffill().bfill()
    df.dropna(subset=FEATURE_COLS, inplace=True)

    return df

# ---------------------------------------------------------------------------
# RL Environment
# ---------------------------------------------------------------------------
if _RL_AVAILABLE:

    class RLTradingEnv(gym.Env):
        """Single‑asset trading environment (long‑only)."""

        metadata = {"render.modes": []}

        def __init__(self, df: pd.DataFrame, initial_cash: float = 10_000.0):
            super().__init__()
            self.df = df.reset_index(drop=True)
            self.initial_cash = initial_cash
            self.action_space = gym.spaces.Discrete(3)  # 0‑hold, 1‑buy, 2‑sell
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(len(FEATURE_COLS),), dtype=np.float32
            )
            self.reset()

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------
        def _get_obs(self):
            obs = self.df.loc[self._step_idx, FEATURE_COLS].astype(np.float32).values
            assert np.isfinite(obs).all(), "Observation contains non‑finite values!"
            return obs

        def _current_price(self) -> float:
            return float(self.df.loc[self._step_idx, "close"])

        # ------------------------------------------------------------------
        # Gym API
        # ------------------------------------------------------------------
        def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
            super().reset(seed=seed)
            self._step_idx = 0
            self.cash = self.initial_cash
            self.position = 0  # shares held
            return self._get_obs(), {}

        def step(self, action: int):
            price = self._current_price()
            reward = 0.0
            done = False

            # Execute trade
            if action == 1 and self.position == 0:  # BUY
                self.position = int(self.cash // price)
                self.cash -= self.position * price
            elif action == 2 and self.position > 0:  # SELL
                self.cash += self.position * price
                reward = self.cash - self.initial_cash  # realised P/L
                self.position = 0

            # Advance timeline
            self._step_idx += 1
            if self._step_idx >= len(self.df) - 1:
                done = True
                # Liquidate residual
                if self.position > 0:
                    self.cash += self.position * self._current_price()
                    self.position = 0
                reward = self.cash - self.initial_cash

            obs = self._get_obs() if not done else np.zeros(len(FEATURE_COLS), dtype=np.float32)
            return obs, reward, done, False, {}
else:

    class RLTradingEnv:  # type: ignore
        pass

# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _train_agent(df: pd.DataFrame, timesteps: int = 25_000, *, model_path: Path | None = None):
    if not _RL_AVAILABLE:
        return None
    env = DummyVecEnv([lambda: RLTradingEnv(df)])
    agent = PPO("MlpPolicy", env, verbose=0)
    agent.learn(total_timesteps=min(timesteps, len(df) * 50), progress_bar=False)
    if model_path:
        agent.save(model_path)
    return agent


def _get_or_train_agent(ticker: str, df: pd.DataFrame, model_suffix: str):
    if not _RL_AVAILABLE:
        return None
    model_file = MODEL_DIR / f"{ticker}_{CONVERTED_TIMEFRAME}_{model_suffix}.zip"
    if model_file.exists():
        try:
            return PPO.load(model_file)
        except Exception:
            _logger.warning("Corrupt model file %s – retraining…", model_file)
    _logger.info("Training PPO on %s candles (suffix=%s)…", len(df), model_suffix)
    return _train_agent(df, model_path=model_file)

# ---------------------------------------------------------------------------
# Policy helper
# ---------------------------------------------------------------------------

def _rl_decision(agent, state: np.ndarray) -> str:
    if agent is None or not _RL_AVAILABLE:
        return NONE
    action, _ = agent.predict(state, deterministic=True)
    return {0: NONE, 1: BUY, 2: SELL}[int(action)]

# ---------------------------------------------------------------------------
# Public entry‑points
# ---------------------------------------------------------------------------

def run_logic(current_price: float, predicted_price: float, ticker: str):
    """Production trading hook."""
    try:
        from forest import api, buy_shares, sell_shares  # type: ignore
    except ImportError as exc:
        raise ImportError("forest brokerage SDK missing") from exc

    df = _load_data(ticker)
    agent = _get_or_train_agent(ticker, df, model_suffix="full")

    latest_row = df.iloc[-1].copy()
    latest_row.update({"close": current_price, "predicted_close": predicted_price})
    obs = latest_row[FEATURE_COLS].astype(np.float32).values

    decision = _rl_decision(agent, obs)

    # broker context
    account = api.get_account()
    cash = float(account.cash)
    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception:
        position_qty = 0.0

    _logger.info("[%s] RL decision=%s | Px=%.2f | Qty=%s | Cash=%.2f", ticker, decision, current_price, position_qty, cash)

    if decision == BUY and position_qty == 0:
        qty = int(cash // current_price)
        if qty:
            buy_shares(ticker, qty, current_price, predicted_price)
    elif decision == SELL and position_qty > 0:
        sell_shares(ticker, position_qty, current_price, predicted_price)


def run_backtest(current_price: float, predicted_price: float, position_qty: float,
                 current_timestamp: Union[pd.Timestamp, str], candles) -> str:
    """Walk‑forward back‑test."""
    if isinstance(current_timestamp, str):
        current_timestamp = pd.Timestamp(current_timestamp)

    ticker = TICKERS[0]
    df_full = _load_data(ticker)
    df_train = df_full[df_full["timestamp"] <= current_timestamp].copy()
    if df_train.empty:
        return NONE

    # overwrite latest predicted_close so obs reflects ML forecast accuracy
    df_train.loc[df_train.index[-1], "predicted_close"] = predicted_price

    agent = _get_or_train_agent(ticker, df_train, model_suffix=current_timestamp.strftime("%Y%m%dT%H%M"))

    obs = df_train.iloc[-1][FEATURE_COLS].astype(np.float32).values
    decision = _rl_decision(agent, obs)

    if decision == BUY and position_qty == 0:
        return BUY
    if decision == SELL and position_qty > 0:
        return SELL
    return NONE
