# ───────────────────────────────────── Imports ──────────────────────────────────
from __future__ import annotations

import logging
import os
from datetime import datetime

import gym
import numpy as np
import pandas as pd
from gym import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

# ─────────────────────────────── Configuration ─────────────────────────────────
INITIAL_BAL = 1_000.0
HOLD_PENALTY = -1e-3               # small inaction penalty
TOTAL_TIMESTEPS = 10_000           # used by both helpers – adjust to taste

BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "1Hour")
TIMEFRAME_MAP = {
    "4Hour": "H4", "2Hour": "H2", "1Hour": "H1",
    "30Min": "M30", "15Min": "M15"
}
CONVERTED_TIMEFRAME = TIMEFRAME_MAP.get(BAR_TIMEFRAME, BAR_TIMEFRAME)

FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions', 'sentiment',
    'greed_index', 'news_count', 'news_volume_z', 'price_change', 'high_low_range', 
    'macd_line', 'macd_signal', 'macd_histogram', 'rsi', 'roc', 'atr', 'ema_9', 'ema_21', 
    'ema_50', 'ema_200', 'adx', 'obv', 'bollinger_percB', 'returns_1', 
    'returns_3', 'returns_5', 'std_5', 'std_10', 'lagged_close_1', 'lagged_close_2', 
    'lagged_close_3', 'lagged_close_5', 'lagged_close_10', 'candle_body_ratio', 
    'wick_dominance', 'gap_vs_prev', 'volume_zscore', 'atr_zscore', 'rsi_zscore',
    'month', 'hour_sin', 'hour_cos', 'day_of_week_sin',  'day_of_week_cos', 
    'days_since_high', 'days_since_low', "d_sentiment"
]
# Extra hybrid feature injected at inference time (0-filled during training)
PRED_FEATURE = "predicted_price"

# ──────────────────────────────── Environment ──────────────────────────────────
class StockTradingEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, df: pd.DataFrame, initial_balance: float = INITIAL_BAL):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.initial_balance = initial_balance

        # ── NEW: build feature list *including* predicted_price ────────────────
        self.feature_cols = [c for c in FEATURE_COLS if c != "timestamp"]
        # ensure the column actually exists in self.df
        if PRED_FEATURE not in self.df.columns:
            self.df[PRED_FEATURE] = 0.0
        self.feature_cols.append(PRED_FEATURE)
        # ─────────────────────────────────────────────────────────────────────────

        self.state_dim = len(self.feature_cols) + 2  # +[shares, cash]
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.state_dim,),
                                            dtype=np.float32)
        self._reset_internal_state()

    # ───────────────────────────── gym API ──────────────────────────────
    def reset(self):
        self._reset_internal_state()
        return self._get_obs()

    def step(self, action):
        # Convert continuous to discrete dir
        a = float(action[0])
        if a >= 0.33:
            discrete = 1      # BUY
        elif a <= -0.33:
            discrete = -1     # SELL
        else:
            discrete = 0      # HOLD

        prev_close = self.df.loc[self.idx - 1, "close"] if self.idx > 0 else \
            self.df.loc[self.idx, "close"]
        price = self.df.loc[self.idx, "close"]

        reward = HOLD_PENALTY  # default HOLD penalty
        if discrete != 0:
            if discrete == 1 and self.shares == 0:
                # BUY max possible
                max_shares = int(self.cash // price)
                if max_shares > 0:
                    self.shares = max_shares
                    self.cash -= max_shares * price
            elif discrete == -1 and self.shares > 0:
                # SELL all
                self.cash += self.shares * price
                self.shares = 0

            # reward based on correct direction
            price_move = (price - prev_close) / prev_close
            reward = discrete * price_move  # positive if direction matches move

        self.net_worth = self.cash + self.shares * price
        self.idx += 1
        done = self.idx >= len(self.df) - 1

        return self._get_obs(), reward, done, {}

    # ────────────────────────────── Helpers ─────────────────────────────
    def _reset_internal_state(self):
        self.idx = 1                       # need prev_close for reward
        self.cash = self.initial_balance
        self.shares = 0
        self.net_worth = self.cash

    def _get_obs(self):
        row = self.df.iloc[self.idx]
        features = row[self.feature_cols].values.astype(np.float32)
        return np.concatenate([features,
                               [self.shares],
                               [self.cash]], dtype=np.float32)

# ─────────────────────────── tqdm Progress Hook ────────────────────────────────
class ProgressBarCallback(BaseCallback):
    """Tqdm wrapper for SB3 learning."""

    def __init__(self, total_timesteps: int):
        super().__init__(verbose=0)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps,
                         desc="SAC training", unit="step")

    def _on_step(self):
        self.pbar.update(1)
        return True

    def _on_training_end(self):
        self.pbar.close()

# ──────────────────────────────── Utilities ────────────────────────────────────
def _load_df(ticker, cutoff: datetime | None = None) -> pd.DataFrame:
    df_raw = pd.read_csv(os.path.join("data", f"{ticker}_{CONVERTED_TIMEFRAME}.csv"), parse_dates=["timestamp"])
    df = df_raw[[c for c in FEATURE_COLS if c in df_raw.columns]].copy()

    # ── NEW: pull in predicted_close if present ─────────────────────────────
    if "predicted_close" in df_raw.columns:
        df[PRED_FEATURE] = df_raw["predicted_close"].values
    else:
        df[PRED_FEATURE] = 0.0
    # ─────────────────────────────────────────────────────────────────────────────

    # Sort & truncate to avoid leakage
    df.sort_values("timestamp", inplace=True)
    if cutoff is not None:
        df = df[df["timestamp"] <= cutoff].copy()

    # Clean up
    df.fillna(method="ffill", inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def _train_sac(df: pd.DataFrame, timesteps: int = TOTAL_TIMESTEPS) -> SAC:
    """Train SAC from scratch and return the fitted model."""
    env = StockTradingEnv(df)
    model = SAC("MlpPolicy", env, verbose=0, batch_size=256,
                learning_rate=3e-4, tau=0.02, gamma=0.995,
                train_freq=1, gradient_steps=1,
                ent_coef="auto")

    model.learn(total_timesteps=timesteps,
                callback=ProgressBarCallback(timesteps))
    return model, env

def _infer_action(model: SAC, env: StockTradingEnv,
                  state_vec: np.ndarray) -> int:
    """
    Predict *discrete* action ∈ {-1, 0, 1} from raw state_vec.
    """
    # stable-baselines3 `predict` expects 2-D obs
    cont_action, _ = model.predict(state_vec[None, :], deterministic=True)
    a = float(cont_action[0])
    if a >= 0.33:
        return 1
    if a <= -0.33:
        return -1
    return 0

def run_logic(current_price: float, predicted_price: float, ticker: str):
    from forest import api, buy_shares, sell_shares   # noqa
    logger = logging.getLogger(__name__)

    # 1) Load history (with possible historic predictions if present)
    df = _load_df(ticker)
    # Override the *very last* row with the live forecast
    df.iloc[-1, df.columns.get_loc(PRED_FEATURE)] = predicted_price

    # 2) Train SAC from scratch
    model, env = _train_sac(df)

    # 3) Build the latest state (obs includes predicted_price slot)
    state = env._get_obs()
    # NOTE: actual account shares & cash are fetched next
    state[-2] = 0                   # placeholder for live shares
    state[-1] = INITIAL_BAL         # placeholder for live cash
    state[env.feature_cols.index(PRED_FEATURE)] = predicted_price

    # 4) Decide & execute
    discrete = _infer_action(model, env, state)
    action_str = {1: "BUY", -1: "SELL", 0: "NONE"}[discrete]

    try:
        account = api.get_account()
        cash = float(account.cash)
    except Exception as e:
        logger.error(f"[{ticker}] Unable to fetch account: {e}")
        return
    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception:
        position_qty = 0.0

    logger.info(f"[{ticker}] RL-action={action_str}, Price={current_price}, "
                f"Pred={predicted_price:.3f}, Qty={position_qty}, Cash={cash}")

    if discrete == 1:
        if position_qty == 0:
            max_shares = int(cash // current_price)
            if max_shares > 0:
                buy_shares(ticker, max_shares, current_price, predicted_price)
                logger.info(f"[{ticker}] → BUY {max_shares}")
            else:
                logger.info(f"[{ticker}] No cash to buy.")
        else:
            logger.info(f"[{ticker}] Already long; skipped BUY.")
    elif discrete == -1:
        if position_qty > 0:
            sell_shares(ticker, position_qty, current_price, predicted_price)
            logger.info(f"[{ticker}] → SELL {position_qty}")
        else:
            logger.info(f"[{ticker}] Nothing to sell; skipped SELL.")
    else:
        logger.info(f"[{ticker}] HOLD action taken.")


def run_backtest(current_price: float,
                 predicted_price: float,
                 position_qty: int,
                 current_timestamp: datetime,
                 candles,
                 ticker):
    # 1) Load history up to timestamp (with historic preds if any)
    df = _load_df(ticker, cutoff=current_timestamp)
    if len(df) < 50:
        return "NONE"

    # 2) Inject the current live forecast
    df.iloc[-1, df.columns.get_loc(PRED_FEATURE)] = predicted_price

    # 3) Re-train SAC
    model, env = _train_sac(df, timesteps=max(2_000, len(df) * 5))

    # 4) Build state & decide
    state = env._get_obs()
    state[-2] = position_qty    # live shares
    state[-1] = INITIAL_BAL     # placeholder cash
    state[env.feature_cols.index(PRED_FEATURE)] = predicted_price

    discrete = _infer_action(model, env, state)
    if discrete == 1 and position_qty == 0:
        return "BUY"
    if discrete == -1 and position_qty > 0:
        return "SELL"
    return "NONE"