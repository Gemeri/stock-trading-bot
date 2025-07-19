"""
PPO Trading Script – TSLA 4H Candles (Rolling Retrain, Production-Ready Reward)
--------------------------------------------------------------------------
Key requirements implemented:
- Ticker: TSLA (single-asset, long-only: BUY max, SELL all, HOLD)
- Data: 4‑hour candles from CSV: data/TSLA_H4.csv
- Features: EXACT list provided (timestamp + 44 numeric cols). Ignore/drop predicted_close if present.
- Initial training cash: $1_000 (environment internal). Live cash from broker used only at inference.
- PPO hyperparams: tuned for ~1‑yr rolling window (~1,400 candles) – total_timesteps=42_000, n_steps=1_024.
- Reward: risk‑aware, production style. Uses *percent portfolio change* minus *transaction costs* and *drawdown penalty*.
- Auto rolling train window util (6–12 mo) for backtest (no leakage: train <= current_timestamp).
- Retrain from scratch EACH call to run_logic() or run_backtest() (per user instruction).
- Invalid actions auto‑convert to HOLD (BUY when already long; SELL when flat).
- Observations: scaled features (z‑score fit on training slice) + portfolio allocation fractions (cash%, equity%).

At bottom: run_logic() and run_backtest() with specified signatures and broker integration stub (forest SDK).

Notes:
- This script assumes that the CSV contains at least the required feature columns. Extra columns are ignored.
- The CSV may contain market gaps (weekends/holidays) – this is fine; environment steps through rows.
- Production recommendation: persist trained model to disk & reuse intra‑day; user explicitly requested retrain each call, so we comply.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd

# Stable-Baselines3 / Gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure as sb3_logger_configure
import gymnasium as gym
from gymnasium import spaces

# Scaling
from sklearn.preprocessing import StandardScaler

# Progress bar (optional nice UX during training)
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Config / Constants
# ---------------------------------------------------------------------------
DATA_PATH = Path("data/TSLA_H4.csv")
INITIAL_CASH_TRAIN = 1_000.0  # env cash for training episodes
DEFAULT_LOOKBACK_CANDLES = 1_400  # ~1y of 4H TSLA data in your dataset

# EXACT feature schema requested (timestamp + 44 numeric features) – predicted_close excluded
FEATURE_COLS: List[str] = [
    "timestamp","open","high","low","close","volume","vwap","transactions","sentiment",
    "price_change","high_low_range","log_volume","macd_line","macd_signal","macd_histogram","rsi",
    "momentum","roc","atr","ema_9","ema_21","ema_50","ema_200","adx","obv","bollinger_upper",
    "bollinger_lower","lagged_close_1","lagged_close_2","lagged_close_3","lagged_close_5","lagged_close_10",
    "candle_body_ratio","wick_dominance","gap_vs_prev","volume_zscore","atr_zscore","rsi_zscore","adx_trend",
    "macd_cross","macd_hist_flip","day_of_week","days_since_high","days_since_low"
]

# Action mapping
ACT_HOLD, ACT_BUY, ACT_SELL = 0, 1, 2
ACTION_TO_STR = {ACT_HOLD: "NONE", ACT_BUY: "BUY", ACT_SELL: "SELL"}
STR_TO_ACTION = {v: k for k, v in ACTION_TO_STR.items()}

# PPO Hyperparameters (tailored)
PPO_TOTAL_TIMESTEPS = 42_000
PPO_N_STEPS = 1_024
PPO_BATCH_SIZE = 128
PPO_N_EPOCHS = 10
PPO_GAMMA = 0.97
PPO_GAE_LAMBDA = 0.92
PPO_CLIP_RANGE = 0.2
PPO_ENT_COEF = 0.005
PPO_VF_COEF = 0.5
PPO_MAX_GRAD_NORM = 0.5
PPO_LR = 2.5e-4
PPO_NET_ARCH = [dict(pi=[256, 256], vf=[256, 256])]

# Transaction cost assumptions (bps = basis points)
FEE_RATE = 0.0005   # 5 bps = 0.05% per notional traded – adjust as needed
SLIPPAGE_RATE = 0.0005  # additional simulated 5 bps market impact

# Drawdown penalty scaling (light touch)
DRAWDOWN_PENALTY = 0.1  # multiply by fractional dd delta when capital falls below peak

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
_logger = logging.getLogger(__name__)
if not _logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Data Loading & Preprocessing
# ---------------------------------------------------------------------------

def load_price_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and clean TSLA 4H data.

    - Enforces feature whitelist.
    - Drops *predicted_close* if present.
    - Forward/back fill remaining NaNs, drop any still missing.
    - Sorts by timestamp.
    - Ensures numeric dtype for feature columns.
    """
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path, parse_dates=["timestamp"], infer_datetime_format=True)

    # Drop predicted_close if present
    if "predicted_close" in df.columns:
        df = df.drop(columns=["predicted_close"])

    # Keep only requested columns if present; ignore extras
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns in CSV: {missing}")
    df = df[FEATURE_COLS].copy()

    # Sort
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Fill / clean
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Coerce numeric for all but timestamp
    for col in df.columns:
        if col == "timestamp":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ---------------------------------------------------------------------------
# Feature Scaling Utilities
# ---------------------------------------------------------------------------

class FeatureScaler:
    """Simple wrapper around StandardScaler for feature matrix.
    Stores fit params and applies transform/ inverse transform.
    """
    def __init__(self):
        self.scaler: Optional[StandardScaler] = None
        self.cols: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame, feature_cols: List[str]) -> None:
        self.cols = feature_cols
        self.scaler = StandardScaler()
        self.scaler.fit(df[feature_cols].values)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        assert self.scaler is not None and self.cols is not None
        return self.scaler.transform(df[self.cols].values).astype(np.float32)

    def transform_row(self, row: pd.Series) -> np.ndarray:
        assert self.scaler is not None and self.cols is not None
        arr = row[self.cols].values.reshape(1, -1)
        return self.scaler.transform(arr).astype(np.float32).ravel()


# ---------------------------------------------------------------------------
# Reward Helper
# ---------------------------------------------------------------------------

def compute_trade_cost(notional: float, fee_rate: float = FEE_RATE, slip_rate: float = SLIPPAGE_RATE) -> float:
    return notional * (fee_rate + slip_rate)


# ---------------------------------------------------------------------------
# Gymnasium Environment – Rolling Single Episode Through Historical Data
# ---------------------------------------------------------------------------

class TSLATradingEnv(gym.Env):
    """A 1D single‑asset long‑only trading env for PPO.

    Observation = scaled feature vector + [cash_frac, equity_frac]. Values are ~N(0,1) for features, [0,1] for fracs.
    Action space: 0=HOLD, 1=BUY(max), 2=SELL(all). Invalid BUY/SELL auto‑converted to HOLD.

    Reward (per step):
        step_ret = (val_t1 - val_t0) / val_t0   # percent change in portfolio value
        trade_cost deducted when trades occur
        drawdown penalty when new equity curve low vs rolling peak
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        scaler: FeatureScaler,
        initial_cash: float = INITIAL_CASH_TRAIN,
        fee_rate: float = FEE_RATE,
        slip_rate: float = SLIPPAGE_RATE,
        drawdown_penalty: float = DRAWDOWN_PENALTY,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.scaler = scaler
        self.initial_cash = float(initial_cash)
        self.fee_rate = fee_rate
        self.slip_rate = slip_rate
        self.drawdown_penalty = drawdown_penalty

        # Precompute scaled feature matrix
        self.feature_cols = [c for c in self.df.columns if c != "timestamp"]
        self.features_scaled = self.scaler.transform(self.df)

        # Spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.feature_cols) + 2,), dtype=np.float32)

        # Internal state placeholders
        self._i: int = 0
        self.cash: float = self.initial_cash
        self.qty: int = 0
        self._portfolio_value_peak: float = self.initial_cash

    # ---------------------------------------
    # Internal helpers
    # ---------------------------------------
    def _price(self, idx: int) -> float:
        return float(self.df.loc[idx, "close"])  # using close as execution proxy

    def _portfolio_value(self, price: float | None = None) -> float:
        if price is None:
            price = self._price(self._i)
        return self.cash + self.qty * price

    def _obs(self) -> np.ndarray:
        feat = self.features_scaled[self._i]  # scaled features
        # Portfolio fractions (avoid div by 0)
        px = self._price(self._i)
        val = max(self._portfolio_value(px), 1e-8)
        cash_frac = self.cash / val
        eq_frac = (self.qty * px) / val
        return np.concatenate([feat, np.array([cash_frac, eq_frac], dtype=np.float32)], axis=0).astype(np.float32)

    # ---------------------------------------
    # Gym API
    # ---------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):  # Gymnasium signature
        super().reset(seed=seed)
        self._i = 0
        self.cash = self.initial_cash
        self.qty = 0
        self._portfolio_value_peak = self.initial_cash
        obs = self._obs()
        info = {}
        return obs, info

    def step(self, action: int):  # Gymnasium signature returns (obs, reward, terminated, truncated, info)
        assert 0 <= action <= 2
        px = self._price(self._i)
        pre_val = self._portfolio_value(px)

        # Adjust action if invalid
        if action == ACT_BUY and self.qty > 0:
            action = ACT_HOLD
        elif action == ACT_SELL and self.qty == 0:
            action = ACT_HOLD

        # Execute trade
        trade_cost = 0.0
        if action == ACT_BUY:  # buy max whole shares
            qty_buy = int(self.cash // px)
            if qty_buy > 0:
                notional = qty_buy * px
                trade_cost = compute_trade_cost(notional, self.fee_rate, self.slip_rate)
                self.cash -= notional + trade_cost
                self.qty += qty_buy
        elif action == ACT_SELL:  # sell all
            if self.qty > 0:
                notional = self.qty * px
                trade_cost = compute_trade_cost(notional, self.fee_rate, self.slip_rate)
                self.cash += notional - trade_cost
                self.qty = 0

        # Advance index
        self._i += 1
        terminated = self._i >= len(self.df) - 1  # last step yields terminal next obs
        truncated = False

        # Compute reward based on next price (or current if terminal)
        next_px = px if terminated else self._price(self._i)
        post_val = self._portfolio_value(next_px)

        # Percent portfolio return
        step_ret = (post_val - pre_val) / max(pre_val, 1e-8)

        # Drawdown penalty
        self._portfolio_value_peak = max(self._portfolio_value_peak, post_val)
        dd = (self._portfolio_value_peak - post_val) / max(self._portfolio_value_peak, 1e-8)  # fractional drawdown 0..1
        reward = step_ret - self.drawdown_penalty * dd

        # Observations
        obs = np.zeros(self.observation_space.shape, dtype=np.float32) if terminated else self._obs()
        info: Dict[str, Any] = {
            "price": px,
            "next_price": next_px,
            "portfolio_value": post_val,
            "drawdown": dd,
            "raw_step_return": step_ret,
            "reward": reward,
            "trade_cost": trade_cost,
            "action": ACTION_TO_STR[action],
        }
        return obs, float(reward), bool(terminated), bool(truncated), info


# ---------------------------------------------------------------------------
# Progress Bar Callback (optional)
# ---------------------------------------------------------------------------

class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="PPO Training", unit="step", leave=True)

    def _on_rollout_end(self):
        if self.pbar is not None:
            self.pbar.n = self.num_timesteps
            self.pbar.refresh()

    def _on_step(self) -> bool:  # called after each call to env.step()
        return True

    def _on_training_end(self):
        if self.pbar is not None:
            self.pbar.close()


# ---------------------------------------------------------------------------
# Training Utilities
# ---------------------------------------------------------------------------

def _fit_scaler(df: pd.DataFrame) -> FeatureScaler:
    scaler = FeatureScaler()
    feature_cols = [c for c in df.columns if c != "timestamp"]
    scaler.fit(df, feature_cols)
    return scaler


def _make_env(df: pd.DataFrame, scaler: FeatureScaler, initial_cash: float = INITIAL_CASH_TRAIN) -> TSLATradingEnv:
    return TSLATradingEnv(df, scaler, initial_cash=initial_cash)


def _train_ppo_on_df(df: pd.DataFrame, *, total_timesteps: int = PPO_TOTAL_TIMESTEPS, seed: int = 0) -> Tuple[PPO, FeatureScaler]:
    """Train PPO from scratch on provided DataFrame slice. Returns model + scaler."""
    set_random_seed(seed)
    scaler = _fit_scaler(df)
    env = DummyVecEnv([lambda: _make_env(df, scaler)])  # vec wrapper expected by SB3

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=PPO_LR,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
        n_epochs=PPO_N_EPOCHS,
        gamma=PPO_GAMMA,
        gae_lambda=PPO_GAE_LAMBDA,
        clip_range=PPO_CLIP_RANGE,
        ent_coef=PPO_ENT_COEF,
        vf_coef=PPO_VF_COEF,
        max_grad_norm=PPO_MAX_GRAD_NORM,
        policy_kwargs=dict(net_arch=PPO_NET_ARCH),
        verbose=0,
    )

    callback = ProgressBarCallback(total_timesteps=total_timesteps)
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    return model, scaler


# ---------------------------------------------------------------------------
# Rolling Window Helper (Backtest)
# ---------------------------------------------------------------------------

def slice_train_window(df: pd.DataFrame, end_ts: pd.Timestamp, lookback: int = DEFAULT_LOOKBACK_CANDLES) -> pd.DataFrame:
    """Return a slice of *df* ending at *end_ts* (inclusive) with up to *lookback* most recent rows.
    Ensures no future data leakage.
    """
    df = df[df["timestamp"] <= end_ts].copy()
    if df.empty:
        raise ValueError("No training data available before the requested timestamp.")
    if len(df) > lookback:
        df = df.iloc[-lookback:].copy()
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Observation Builder for Inference (live/backtest eval outside env)
# ---------------------------------------------------------------------------

def build_live_obs(latest_row: pd.Series, scaler: FeatureScaler, cash: float, position_qty: float) -> np.ndarray:
    """Return scaled feature row + [cash_frac, eq_frac] consistent with training env."""
    feat = scaler.transform_row(latest_row)  # scaled numeric features
    px = float(latest_row["close"])  # we assume latest_row has up-to-date price injected by caller
    port_val = cash + position_qty * px
    if port_val <= 0:
        port_val = 1e-8
    cash_frac = cash / port_val
    eq_frac = (position_qty * px) / port_val
    obs = np.concatenate([feat, np.array([cash_frac, eq_frac], dtype=np.float32)], axis=0).astype(np.float32)
    return obs


# ---------------------------------------------------------------------------
# Public Entry Points – Production Integration
# ---------------------------------------------------------------------------

def run_logic(current_price: float, predicted_price: float, ticker: str):
    """Live trading hook.

    Per user instruction:
    - Retrain PPO *from scratch* on the *entire* CSV each call.
    - Use live broker cash/position to form observation; env trained with $1k but model generalizes via fractional inputs.
    - Ignore predicted_price for RL features (still pass to broker logs/orders).
    """
    import logging
    from forest import api, buy_shares, sell_shares  # type: ignore

    logger = logging.getLogger(__name__)

    # Load full history
    df_full = load_price_data(DATA_PATH)

    # Train PPO (full history) – scratch each call as requested
    logger.info("[run_logic] Training PPO on full dataset (%s rows)…", len(df_full))
    model, scaler = _train_ppo_on_df(df_full, total_timesteps=PPO_TOTAL_TIMESTEPS)

    # Pull live account info
    try:
        account = api.get_account(); cash = float(account.cash)
    except Exception as e:  # pragma: no cover
        logger.error(f"[{ticker}] Error fetching account: {e}")
        return
    try:
        pos = api.get_position(ticker); position_qty = float(pos.qty)
    except Exception:  # no position
        position_qty = 0.0

    # Build latest feature row using last historical row, but update close to live current_price
    latest_row = df_full.iloc[-1].copy()
    latest_row.loc["close"] = current_price  # inject live price
    # (Optional: could also update open/high/low/volume intrabar; leaving historical values is acceptable)

    obs = build_live_obs(latest_row, scaler, cash=cash, position_qty=position_qty)
    action, _ = model.predict(obs, deterministic=True)
    action = int(action)

    logger.info(f"[{ticker}] PPO action={ACTION_TO_STR[action]} | Px={current_price:.2f} | Qty={position_qty} | Cash={cash:.2f}")

    # Enforce action semantics (invalid -> HOLD)
    if action == ACT_BUY and position_qty == 0:
        qty = int(cash // current_price)
        if qty > 0:
            logger.info(f"[{ticker}] BUY {qty} @ {current_price}")
            buy_shares(ticker, qty, current_price, predicted_price)
        else:
            logger.info(f"[{ticker}] Insufficient cash; HOLD.")
    elif action == ACT_SELL and position_qty > 0:
        logger.info(f"[{ticker}] SELL {position_qty} @ {current_price}")
        sell_shares(ticker, position_qty, current_price, predicted_price)
    else:
        logger.info(f"[{ticker}] HOLD (no valid trade).")



def run_backtest(current_price: float, predicted_price: float, position_qty: float, current_timestamp, candles) -> str:
    """Backtest hook.

    Arguments:
        current_price: Ignored for training; used only for logging if desired.
        predicted_price: Ignored for RL features (logged only).
        position_qty: Position size immediately *before* taking an action at *current_timestamp*.
        current_timestamp: Timestamp of the *latest* candle allowed in training (inclusive). No future leakage.
        candles: Unused placeholder (upstream API compatibility).

    Behavior:
        - Load full CSV.
        - Slice rows up to and including *current_timestamp*.
        - If empty or only 1 row, return HOLD.
        - Train PPO from scratch on that slice (or last DEFAULT_LOOKBACK_CANDLES rows if larger).
        - Build obs from FINAL row in the slice using provided position_qty and cash assumption:
            * If already long (position_qty>0), assume 0 cash (capital fully allocated) for obs.
            * Else assume INITIAL_CASH_TRAIN cash (so model sees capital to deploy).
        - Predict action; enforce invalid -> HOLD; return "BUY"/"SELL"/"NONE".
    """
    import logging
    logger = logging.getLogger(__name__)

    # Ensure timestamp type
    if isinstance(current_timestamp, str):
        current_timestamp = pd.Timestamp(current_timestamp, tz=None)

    # Load & slice
    df_full = load_price_data(DATA_PATH)
    try:
        df_train = slice_train_window(df_full, end_ts=current_timestamp, lookback=DEFAULT_LOOKBACK_CANDLES)
    except ValueError:
        return "NONE"

    if len(df_train) < 10:  # too little history to train
        return "NONE"

    logger.info("[run_backtest] Training PPO on %s rows up to %s", len(df_train), current_timestamp)
    model, scaler = _train_ppo_on_df(df_train, total_timesteps=PPO_TOTAL_TIMESTEPS)

    # Build obs from last training row (the candle at current_timestamp or last prior)
    latest_row = df_train.iloc[-1].copy()
    # Use training close; DO NOT inject forward prices (would leak)

    # Cash assumption: if already long, assume fully invested (cash=0); else assume INITIAL_CASH_TRAIN available.
    cash = 0.0 if position_qty > 0 else INITIAL_CASH_TRAIN

    obs = build_live_obs(latest_row, scaler, cash=cash, position_qty=position_qty)
    action, _ = model.predict(obs, deterministic=True)
    action = int(action)

    # Enforce semantic correction
    if action == ACT_BUY and position_qty == 0:
        return "BUY"
    if action == ACT_SELL and position_qty > 0:
        return "SELL"
    return "NONE"


# ---------------------------------------------------------------------------
# __main__ quick smoke test (optional)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal self-test: load, train, predict last obs using env cash=1k, flat position.
    df_all = load_price_data(DATA_PATH)
    model, scaler = _train_ppo_on_df(df_all, total_timesteps=5_000)  # short smoke
    last = df_all.iloc[-1]
    obs = build_live_obs(last, scaler, cash=INITIAL_CASH_TRAIN, position_qty=0)
    act, _ = model.predict(obs, deterministic=True)
    print("Smoke test action:", ACTION_TO_STR[int(act)])
