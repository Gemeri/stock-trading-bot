from __future__ import annotations

import os
import warnings
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

from sklearn.preprocessing import StandardScaler

import torch
from sb3_contrib import QRDQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.utils import set_random_seed


# ------------------------------
# Configuration & Logging
# ------------------------------

DATA_PATH = os.path.join("data", "TSLA_H4.csv")

REQUIRED_FEATURES: List[str] = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions', 'sentiment',
    'price_change', 'high_low_range', 'log_volume', 'macd_line', 'macd_signal', 'macd_histogram',
    'rsi', 'roc', 'atr', 'ema_9', 'ema_21', 'ema_50', 'ema_200', 'adx', 'obv', 'bollinger_upper',
    'bollinger_lower', 'bollinger_percB', 'returns_1', 'returns_3', 'returns_5', 'std_5', 'std_10',
    'lagged_close_1', 'lagged_close_2', 'lagged_close_3', 'lagged_close_5', 'lagged_close_10',
    'candle_body_ratio', 'wick_dominance', 'gap_vs_prev', 'volume_zscore', 'atr_zscore', 'rsi_zscore',
    'macd_cross', 'macd_hist_flip', 'month', 'hour_sin', 'hour_cos', 'day_of_week_sin',
    'day_of_week_cos', 'days_since_high', 'days_since_low',
]
# Note: If 'predicted_close' exists in CSV, it will be appended to the feature set.

DEFAULT_SEED = 1337
DEFAULT_TIMESTEPS = 30_000  # Tune as desired. Remember: backtest retrains per call.

INITIAL_BALANCE = 1000.0
INACTION_PENALTY = 1e-3  # small penalty for NONE
EPS = 1e-12  # numerical safety

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("QRDQN_TSLA4H")

warnings.filterwarnings("ignore", category=FutureWarning)


# ------------------------------
# Data Utilities
# ------------------------------

def _ensure_and_prepare_df(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Parse timestamp and sort
    if 'timestamp' not in df.columns:
        raise ValueError("CSV must contain 'timestamp' column.")
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Check required features presence
    missing = [c for c in REQUIRED_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # If vwap is missing/NaN, fallback to close
    if 'vwap' in df.columns:
        df['vwap'] = df['vwap'].fillna(df['close'])

    # If 'predicted_close' exists, keep it; otherwise, we do NOT create it (per user's rule).
    # The model will simply not have it as part of inputs if absent.
    return df


def _feature_columns(df: pd.DataFrame) -> List[str]:
    cols = REQUIRED_FEATURES.copy()
    if 'predicted_close' in df.columns:
        cols.append('predicted_close')
    return cols


def _encode_timestamp_inplace(df: pd.DataFrame) -> None:
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["timestamp"] = df["timestamp"].astype("int64") // 10**9
    df["timestamp"] = df["timestamp"].astype("float64")



@dataclass
class EnvConfig:
    feature_cols: List[str]
    initial_balance: float = INITIAL_BALANCE
    inaction_penalty: float = INACTION_PENALTY


# ------------------------------
# Custom Trading Environment
# ------------------------------

class TSLA4HDirectionalEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, config: EnvConfig, seed: int = DEFAULT_SEED):
        super().__init__()
        self.df = df.copy()
        self.config = config
        self.seed(seed)

        # Timestamp encoding
        _encode_timestamp_inplace(self.df)

        # Drop any rows with NaNs across features used
        self.feature_cols = config.feature_cols
        self.df = self.df.dropna(subset=self.feature_cols + ['close']).reset_index(drop=True)

        # At least 2 rows needed for prior-close reward
        if len(self.df) < 2:
            raise ValueError("Not enough rows after cleaning to build environment (need >= 2).")

        # Fit scaler on feature columns
        self.scaler = StandardScaler()
        self.scaler.fit(self.df[self.feature_cols].astype(np.float64))

        # Spaces
        n_features = len(self.feature_cols)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # NONE, BUY, SELL

        # Internal state
        self.idx: int = 1  # start from 1 so prior-close exists at idx-1
        self.balance: float = float(self.config.initial_balance)
        self.shares: int = 0

    def seed(self, seed: Optional[int] = None):
        if seed is None:
            seed = DEFAULT_SEED
        set_random_seed(seed)
        np.random.seed(seed)
        self._seed = seed

    def _get_obs(self) -> np.ndarray:
        row = self.df.loc[self.idx, self.feature_cols].astype(np.float64).values.reshape(1, -1)
        obs = self.scaler.transform(row).astype(np.float32).squeeze(0)
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.seed(seed)
        self.idx = 1
        self.balance = float(self.config.initial_balance)
        self.shares = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        done = False
        info = {}

        # Current and previous closes for directional reward
        cur_close = float(self.df.loc[self.idx, 'close'])
        prev_close = float(self.df.loc[self.idx - 1, 'close'])
        pct_move = (cur_close - prev_close) / (prev_close + EPS)

        # Action remapping for invalid trades
        original_action = int(action)
        if original_action == 1 and self.shares > 0:
            # Attempted BUY while already long -> default to NONE
            action = 0
        elif original_action == 2 and self.shares == 0:
            # Attempted SELL while flat -> default to NONE
            action = 0

        # Execute trade rules (balance/position bookkeeping only; reward isn't PnL-based)
        if action == 1:
            # BUY max shares
            max_shares = int(self.balance // cur_close)
            if max_shares > 0:
                cost = max_shares * cur_close
                self.balance -= cost
                self.shares += max_shares
        elif action == 2:
            # SELL all
            if self.shares > 0:
                proceeds = self.shares * cur_close
                self.balance += proceeds
                self.shares = 0

        # Reward logic
        if action == 0:
            reward = -self.config.inaction_penalty
        else:
            trade_dir = 1.0 if action == 1 else -1.0  # BUY=+1, SELL=-1
            reward = float(trade_dir * pct_move)

        # Advance time
        self.idx += 1
        if self.idx >= len(self.df) - 1:
            done = True

        obs = self._get_obs()
        return obs, reward, done, False, info


# ------------------------------
# Model Training & Inference
# ------------------------------

def _make_env(df: pd.DataFrame, feature_cols: List[str]) -> Monitor:
    env = TSLA4HDirectionalEnv(df=df, config=EnvConfig(feature_cols=feature_cols))
    return Monitor(env)


def _train_qrdqn(env: gym.Env, timesteps: int = DEFAULT_TIMESTEPS, seed: int = DEFAULT_SEED) -> QRDQN:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_kwargs = dict(net_arch=[256, 256])

    model = QRDQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=256,
        gamma=0.99,
        target_update_interval=1_000,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.10,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=policy_kwargs,
        tensorboard_log=None,
        seed=seed,
        device=device,
        verbose=0,
    )
    model.learn(total_timesteps=int(timesteps), callback=ProgressBarCallback())
    return model


def _prepare_observation_for_timestamp(
    df: pd.DataFrame,
    env_monitor: Monitor,
    feature_cols: List[str],
    ts: pd.Timestamp,
    predicted_price: Optional[float] = None,
) -> np.ndarray:
    # Unwrap env & get scaler/feature order used during training
    env: TSLA4HDirectionalEnv = env_monitor.env
    scaler = env.scaler
    env_feature_order = list(env.feature_cols)  # authoritative order for scaler

    # Work on a datetime-sorted copy for locating the row
    df_sorted = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_sorted["timestamp"]):
        df_sorted["timestamp"] = pd.to_datetime(df_sorted["timestamp"], utc=True, errors="coerce")
    df_sorted = df_sorted.sort_values("timestamp").reset_index(drop=True)

    # Find the last index where timestamp <= ts
    ts_cutoff = pd.to_datetime(ts, utc=True)
    idx_series = df_sorted["timestamp"] <= ts_cutoff
    if not idx_series.any():
        raise ValueError("current_timestamp is earlier than the earliest row in the CSV.")
    row_idx = int(np.where(idx_series.values)[0][-1])

    # Start from the raw feature row
    raw = df_sorted.loc[row_idx, feature_cols].copy()

    # Overwrite predicted_close with provided predicted_price if present
    if "predicted_close" in feature_cols and predicted_price is not None:
        raw["predicted_close"] = float(predicted_price)

    # Ensure timestamp is UNIX seconds float
    ts_raw = df_sorted.loc[row_idx, "timestamp"]
    ts_unix = (pd.to_datetime(ts_raw, utc=True).value // 10**9)
    raw["timestamp"] = float(ts_unix)

    # Build single-row DataFrame in exactly the env's feature order
    # (reorder from caller-provided feature_cols if needed)
    tmp = pd.DataFrame([raw], columns=feature_cols)

    # Coerce numerics safely; non-numeric -> NaN (filled next)
    tmp = tmp.apply(pd.to_numeric, errors="coerce")

    # Reorder columns to match scaler training order
    # (env_feature_order is the order used when fitting the scaler)
    tmp = tmp.reindex(columns=env_feature_order)

    # Fill NaNs with env medians over the same feature set
    med = env.df[env_feature_order].median(numeric_only=True)
    tmp = tmp.fillna(med)

    # Transform with scaler -> np.float32 vector
    obs = scaler.transform(tmp.values.astype(np.float64)).astype(np.float32).squeeze(0)
    return obs



def _override_lastrow_for_live(
    df: pd.DataFrame, feature_cols: List[str], current_price: Optional[float], predicted_price: Optional[float]
) -> Tuple[pd.DataFrame, int]:
    """
    For run_logic (live): use the last available row as the current state.
    Optionally override 'close' with current_price and 'predicted_close' with predicted_price (if present).
    Returns a copy of df and index of last row.
    """
    dfx = df.copy()
    last_idx = len(dfx) - 1
    if current_price is not None:
        dfx.loc[last_idx, 'close'] = float(current_price)
    if 'predicted_close' in feature_cols and (predicted_price is not None):
        dfx.loc[last_idx, 'predicted_close'] = float(predicted_price)
    return dfx, last_idx


def _predict_action(model: QRDQN, obs: np.ndarray) -> int:
    action, _ = model.predict(obs, deterministic=True)
    return int(action)


def _action_to_str(action: int) -> str:
    return "NONE" if action == 0 else ("BUY" if action == 1 else "SELL")


# ------------------------------
# Public API functions (as requested)
# ------------------------------

def run_logic(current_price: float, predicted_price: float, ticker: str) -> None:
    import logging
    from forest import api, buy_shares, sell_shares

    log = logging.getLogger(__name__)

    # Load full CSV
    df = _ensure_and_prepare_df(DATA_PATH)

    # Feature columns (add 'predicted_close' only if it exists)
    feat_cols = _feature_columns(df)

    # Train on entire CSV
    env = _make_env(df, feat_cols)
    model = _train_qrdqn(env, timesteps=DEFAULT_TIMESTEPS, seed=DEFAULT_SEED)

    # Prepare live observation from last row, with overrides
    dfx, last_idx = _override_lastrow_for_live(df, feat_cols, current_price, predicted_price)

    # Build single observation using the SAME scaler used by the env
    # For consistency, mimic the env's timestamp encoding on the copy
    # We reuse helper by passing the last timestamp
    ts_last = dfx['timestamp'].iloc[last_idx]
    if not pd.api.types.is_datetime64_any_dtype(dfx['timestamp']):
        # If internal df has already been converted, re-read original CSV to get datetime
        dforig = pd.read_csv(DATA_PATH)
        dforig['timestamp'] = pd.to_datetime(dforig['timestamp'], utc=True, errors='coerce')
        ts_last = dforig['timestamp'].iloc[last_idx]
    obs = _prepare_observation_for_timestamp(dfx, env, feat_cols, ts=pd.to_datetime(ts_last, utc=True),
                                             predicted_price=predicted_price)

    # Decide action
    action_int = _predict_action(model, obs)
    action_str = _action_to_str(action_int)

    # Retrieve broker status
    try:
        account = api.get_account()
        cash = float(account.cash)
    except Exception as e:
        log.error(f"[{ticker}] Error fetching account details: {e}")
        return

    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception:
        position_qty = 0.0

    log.info(f"[{ticker}] Current Price: {current_price}, Predicted Price: {predicted_price}, "
             f"Model Action: {action_str}, Position: {position_qty}, Cash: {cash}")

    # Enforce action constraints at execution time too
    if action_str == "BUY":
        if position_qty == 0:
            max_shares = int(cash // float(current_price))
            if max_shares > 0:
                log.info(f"[{ticker}] Buying {max_shares} shares at {current_price}.")
                buy_shares(ticker, max_shares, current_price, predicted_price)
            else:
                log.info(f"[{ticker}] Insufficient cash to purchase shares.")
        else:
            log.info(f"[{ticker}] Already in a long position; defaulting to NONE.")
    elif action_str == "SELL":
        if position_qty > 0:
            log.info(f"[{ticker}] Selling {position_qty} shares at {current_price}.")
            sell_shares(ticker, position_qty, current_price, predicted_price)
        else:
            log.info(f"[{ticker}] No long position to sell; defaulting to NONE.")
    else:
        log.info(f"[{ticker}] Model action NONE; no trade.")


def run_backtest(
    current_price: float,
    predicted_price: float,
    position_qty: float,
    current_timestamp,
    candles,
    ticker
) -> str:
    # Load full CSV
    df_full = _ensure_and_prepare_df(DATA_PATH)

    # Filter to <= current_timestamp (no future leakage)
    if not pd.api.types.is_datetime64_any_dtype(df_full['timestamp']):
        df_full['timestamp'] = pd.to_datetime(df_full['timestamp'], utc=True, errors='coerce')
    ts_cutoff = pd.to_datetime(current_timestamp, utc=True)

    df_train = df_full[df_full['timestamp'] <= ts_cutoff].copy()
    if len(df_train) < 200:
        # Need enough steps to learn something; if too small, we degrade gracefully.
        # You can relax/adjust this threshold as needed.
        logger.warning("Training window is small; results may be noisy.")

    # Determine features (only add 'predicted_close' if present)
    feat_cols = _feature_columns(df_train)

    # Train env (up to cutoff only)
    env = _make_env(df_train, feat_cols)
    model = _train_qrdqn(env, timesteps=DEFAULT_TIMESTEPS, seed=DEFAULT_SEED)

    # Build observation at `current_timestamp`
    obs = _prepare_observation_for_timestamp(
        df=df_full,               # use full to locate the exact row at ts
        env_monitor=env,
        feature_cols=feat_cols,
        ts=ts_cutoff,
        predicted_price=predicted_price if ('predicted_close' in feat_cols) else None
    )

    # Get proposed action from policy
    action_int = _predict_action(model, obs)

    # Enforce constraints with provided position_qty
    if action_int == 1 and position_qty > 0:
        action_int = 0  # default to NONE if already long
    elif action_int == 2 and position_qty <= 0:
        action_int = 0  # default to NONE if flat

    return _action_to_str(action_int)


# ------------------------------
# Optional: quick sanity test
# ------------------------------

if __name__ == "__main__":
    # Minimal smoke-test (does not place trades)
    try:
        df0 = _ensure_and_prepare_df(DATA_PATH)
        feats = _feature_columns(df0)
        env0 = _make_env(df0, feats)
        model0 = _train_qrdqn(env0, timesteps=5_000, seed=DEFAULT_SEED)

        # Build obs from last timestamp for dry run
        last_ts = df0['timestamp'].iloc[-1]
        if not pd.api.types.is_datetime64_any_dtype(df0['timestamp']):
            df0['timestamp'] = pd.to_datetime(df0['timestamp'], utc=True, errors='coerce')
            last_ts = df0['timestamp'].iloc[-1]

        obs0 = _prepare_observation_for_timestamp(
            df=df0, env_monitor=env0, feature_cols=feats,
            ts=pd.to_datetime(last_ts, utc=True),
            predicted_price=None  # or set a float if 'predicted_close' exists
        )
        act = _predict_action(model0, obs0)
        logger.info(f"Dry-run action at last row: {_action_to_str(act)}")
    except Exception as e:
        logger.exception(f"Sanity test failed: {e}")
