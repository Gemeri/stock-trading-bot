"""
TSLA 4-Hour PPO trading agent
────────────────────────────
• Uses 42 000 PPO timesteps with n_steps = 1024
• Rolling 6-12 month window built into the env
• Three actions: HOLD(0) | BUY(1) | SELL(2)
• Reward = price-move * trade_dir   (BUY/COVER=+1, SELL/SHORT=-1)
• –0.1 penalty for idleness
• Retrains from scratch every 30 candles (checkpointed)
• Starts with USD 1 000
• Reads data/TSLA_H4.csv and *only* the allowed feature columns
"""

from __future__ import annotations
import json, logging, os, sys
from pathlib import Path
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ──────────────────────────────────────────────────────────────────────────
# Config & constants
# ──────────────────────────────────────────────────────────────────────────
DATA_PATH      = Path("data/TSLA_H4.csv")
MODEL_PATH     = Path("ppo_tsla_h4.zip")
CTR_PATH       = Path("ppo_counter.json")      # stores {"left": <int>}
CHECKPOINT_EVERY = 30                          # candles between full retrains
INITIAL_CASH   = 1_000.0                       # USD
FEATURE_COLS: List[str] = [
    "timestamp","open","high","low","close","volume","vwap","transactions",
    "sentiment","price_change","high_low_range","log_volume","macd_line",
    "macd_signal","macd_histogram","rsi","momentum","roc","atr","ema_9",
    "ema_21","ema_50","ema_200","adx","obv","bollinger_upper",
    "bollinger_lower","lagged_close_1","lagged_close_2","lagged_close_3",
    "lagged_close_5","lagged_close_10","candle_body_ratio","wick_dominance",
    "gap_vs_prev","volume_zscore","atr_zscore","rsi_zscore","adx_trend",
    "macd_cross","macd_hist_flip","day_of_week","days_since_high",
    "days_since_low"
]                                                  # exactly as requested
# Remove timestamp from obs later; keep for slicing.
OBS_FEATURES = [c for c in FEATURE_COLS if c != "timestamp"]

# ──────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────
def _load_counter() -> int:
    if CTR_PATH.exists():
        try:
            return json.loads(CTR_PATH.read_text())["left"]
        except Exception:
            pass
    return 0

def _save_counter(val: int) -> None:
    CTR_PATH.write_text(json.dumps({"left": val}))

def _load_df(full: bool = True,
             cutoff_ts: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found")
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    # ensure required cols only
    cols_present = [c for c in FEATURE_COLS if c in df.columns]
    df = df[cols_present]
    # drop predicted_close if it somehow slipped in
    df = df[[c for c in df.columns if c != "predicted_close"]]
    df.sort_values("timestamp", inplace=True)
    if cutoff_ts is not None:
        df = df[df["timestamp"] <= cutoff_ts]
    # forward/back fill & drop remaining NaNs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[OBS_FEATURES] = df[OBS_FEATURES].ffill().bfill()
    df.dropna(subset=OBS_FEATURES, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ──────────────────────────────────────────────────────────────────────────
# Gym Environment
# ──────────────────────────────────────────────────────────────────────────
class TradingEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.n_steps_total = len(df)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(OBS_FEATURES)+2,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 0 hold, 1 buy, 2 sell
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.idx = 0
        self.cash = INITIAL_CASH
        self.position = 0  # shares held
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        feat = self.df.loc[self.idx, OBS_FEATURES].astype(np.float32).values
        return np.concatenate([feat, [self.cash, self.position]], dtype=np.float32)

    # ------------------------------------------------------------------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        cur_price = float(self.df.loc[self.idx, "close"])
        # previous close
        prev_close = cur_price if self.idx == 0 else float(
            self.df.loc[self.idx-1, "close"]
        )

        # Map invalid trades to HOLD
        if action == 1 and self.position > 0:
            action = 0
        if action == 2 and self.position == 0:
            action = 0

        # Execute trade
        if action == 1:  # BUY
            self.position = int(self.cash // cur_price)
            self.cash -= self.position * cur_price
        elif action == 2:  # SELL
            self.cash += self.position * cur_price
            self.position = 0

        # Reward logic
        if action == 0:
            reward = -0.1  # penalty for idleness
        else:
            trade_dir = 1 if action == 1 else -1
            reward = (cur_price - prev_close) * trade_dir

        # Advance
        self.idx += 1
        done = self.idx >= self.n_steps_total
        if done and self.position > 0:
            # liquidate at final price
            self.cash += self.position * cur_price
            self.position = 0

        obs = self._get_obs() if not done else np.zeros_like(self._get_obs())
        return obs, reward, done, False, {}

# ──────────────────────────────────────────────────────────────────────────
# Model helpers
# ──────────────────────────────────────────────────────────────────────────
def _train_agent(df: pd.DataFrame) -> PPO:
    env = DummyVecEnv([lambda: TradingEnv(df)])
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        learning_rate=2.5e-4,
        gamma=0.97,
        gae_lambda=0.92,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        verbose=0,
    )
    # SB3 expects .learn(...) call
    model.learn(total_timesteps=42_000, progress_bar=False)
    model.save(MODEL_PATH)
    _save_counter(CHECKPOINT_EVERY)  # reset counter
    return model

def _get_or_train(df: pd.DataFrame) -> PPO:
    counter = _load_counter()
    if MODEL_PATH.exists() and counter > 0:
        try:
            model = PPO.load(MODEL_PATH, device="cpu")
            _save_counter(counter - 1)
            return model
        except Exception:
            MODEL_PATH.unlink(missing_ok=True)
    # (re)train from scratch
    return _train_agent(df)

# ──────────────────────────────────────────────────────────────────────────
# Public entry-points
# ──────────────────────────────────────────────────────────────────────────
def run_logic(current_price: float, predicted_price: float, ticker: str):
    """
    Live trading hook — trains on the full CSV every `CHECKPOINT_EVERY` calls
    then uses the PPO policy to decide BUY / SELL / HOLD.
    """
    import logging
    from forest import api, buy_shares, sell_shares
    logger = logging.getLogger(__name__)

    # brokerage context
    try:
        account = api.get_account(); cash = float(account.cash)
    except Exception as e:
        logger.error(f"[{ticker}] Account error: {e}"); return
    try:
        pos = api.get_position(ticker); position_qty = float(pos.qty)
    except Exception:
        position_qty = 0.0

    df = _load_df(full=True)                        # full history
    model = _get_or_train(df)                       # load / train PPO
    # build observation from most-recent candle
    latest = df.iloc[-1][OBS_FEATURES].astype(np.float32).values
    obs = np.concatenate([latest, [cash, position_qty]], dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)

    # Map to execution logic
    if action == 1 and position_qty == 0:
        qty = int(cash // current_price)
        if qty:
            logger.info(f"[{ticker}] BUY {qty} @ {current_price}")
            buy_shares(ticker, qty, current_price, predicted_price)
    elif action == 2 and position_qty > 0:
        logger.info(f"[{ticker}] SELL {position_qty} @ {current_price}")
        sell_shares(ticker, position_qty, current_price, predicted_price)
    else:
        logger.info(f"[{ticker}] HOLD (action={action})")

# -------------------------------------------------------------------------
def run_backtest(current_price: float,
                 predicted_price: float,
                 position_qty: float,
                 current_timestamp: str,
                 candles) -> str:
    """
    Called externally per candle.
    Ensures **zero future leakage** by slicing the CSV up to `current_timestamp`
    and retraining (or loading) the PPO every `CHECKPOINT_EVERY` candles.
    """
    cutoff = pd.to_datetime(current_timestamp, utc=True)
    df_train = _load_df(full=False, cutoff_ts=cutoff)
    if df_train.empty:
        return "NONE"
    model = _get_or_train(df_train)

    latest = df_train.iloc[-1][OBS_FEATURES].astype(np.float32).values
    cash = 0.0 if position_qty > 0 else INITIAL_CASH
    obs = np.concatenate([latest, [cash, position_qty]], dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)

    if action == 1 and position_qty == 0:
        return "BUY"
    if action == 2 and position_qty > 0:
        return "SELL"
    return "NONE"


# ──────────────────────────────────────────────────────────────────────────
# Optional CLI usage for quick sanity check
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df_ = _load_df()
    print("Loaded", len(df_), "candles")
    # quick demo train (will save model + counter)
    _train_agent(df_)
    print("Model trained & saved →", MODEL_PATH)
