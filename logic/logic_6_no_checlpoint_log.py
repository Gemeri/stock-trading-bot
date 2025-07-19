"""
TSLA 4-Hour PPO Trading Bot
---------------------------------------------
• Long-only, all-in BUY / all-out SELL / HOLD
• $1 000 starting equity
• 4-hour candles, rolling 1 400-candle (≈2 y) window
• Reward  = ln(V_{t+1}/V_t)  (step log-return)
• Trains from scratch every call (no leakage)
• Uses only the explicit feature list (ignores “predicted_close” if present)
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# ---------------------------------------------------------------------------#
#  CONFIGURATION
# ---------------------------------------------------------------------------#
CSV_PATH        = Path("data/TSLA_H4.csv")
START_CASH      = 1_000.0
ROLLING_WINDOW  = 1_400                  # candles for back-test retrain
TOTAL_TIMESTEPS = 42_000                 # ≈ 30× full passes over 1-year window
N_STEPS         = 1_024
BATCH_SIZE      = 128

FEATURE_COLS: List[str] = [
    "timestamp",          # kept for slicing; dropped from obs vector later
    "open", "high", "low", "close",
    "volume", "vwap", "transactions", "sentiment",
    "price_change", "high_low_range", "log_volume",
    "macd_line", "macd_signal", "macd_histogram",
    "rsi", "momentum", "roc", "atr",
    "ema_9", "ema_21", "ema_50", "ema_200",
    "adx", "obv", "bollinger_upper", "bollinger_lower",
    "lagged_close_1", "lagged_close_2", "lagged_close_3",
    "lagged_close_5", "lagged_close_10",
    "candle_body_ratio", "wick_dominance", "gap_vs_prev",
    "volume_zscore", "atr_zscore", "rsi_zscore",
    "adx_trend", "macd_cross", "macd_hist_flip",
    "day_of_week", "days_since_high", "days_since_low",
]

OBS_FEATURES = [c for c in FEATURE_COLS if c != "timestamp"]  # final NN inputs

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------#
#  DATA LOADING UTIL
# ---------------------------------------------------------------------------#
def load_tsla_csv() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    # keep only required columns & drop “predicted_close” if present
    unwanted = [c for c in df.columns if c not in FEATURE_COLS]
    df.drop(columns=unwanted, inplace=True, errors="ignore")
    # forward/back fill then drop any still-missing rows
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[OBS_FEATURES] = df[OBS_FEATURES].ffill().bfill()
    df.dropna(subset=OBS_FEATURES, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------#
#  GYM ENVIRONMENT
# ---------------------------------------------------------------------------#
class TSLAEnv(gym.Env):
    """4-hour long-only environment with log-return reward."""

    action_space = spaces.Discrete(3)  # 0=HOLD, 1=BUY, 2=SELL

    def __init__(self, data: pd.DataFrame):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.n_steps = len(data)
        self.obs_dim = len(OBS_FEATURES) + 2  # + cash, position_qty
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.reset(seed=None)

    # --------------------------------------------------------- helpers
    def _obs(self) -> np.ndarray:
        feats = self.data.loc[self._idx, OBS_FEATURES].values.astype(np.float32)
        return np.concatenate([feats, [self.cash, self.position_qty]]).astype(np.float32)

    # --------------------------------------------------------- Gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._idx = 0
        self.cash = START_CASH
        self.position_qty = 0
        return self._obs(), {}

    def step(self, action: int):
        price_now = float(self.data.loc[self._idx, "close"])
        value_prev = self.cash + self.position_qty * price_now

        # --- all-in BUY / all-out SELL ------------------------------------
        if action == 1 and self.position_qty == 0:          # BUY
            self.position_qty = int(self.cash // price_now)
            self.cash -= self.position_qty * price_now
        elif action == 2 and self.position_qty > 0:         # SELL
            self.cash += self.position_qty * price_now
            self.position_qty = 0
        # else it’s effectively HOLD

        # --- advance time -------------------------------------------------
        self._idx += 1
        done = self._idx >= self.n_steps

        price_next = price_now if done else float(self.data.loc[self._idx, "close"])
        value_next = self.cash + self.position_qty * price_next

        # --- reward: step log-return -------------------------------------
        reward = float(np.log(value_next / value_prev))  # already small (~±0.05)

        obs = self._obs() if not done else np.zeros(self.obs_dim, np.float32)
        info = {"portfolio_value": value_next}
        return obs, reward, done, False, info


# ---------------------------------------------------------------------------#
#  PPO TRAINING WRAPPER
# ---------------------------------------------------------------------------#
def train_ppo(df: pd.DataFrame) -> PPO:
    env = TSLAEnv(df)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
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
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    return model


# ---------------------------------------------------------------------------#
#  BROKERAGE LOGIC  (LIVE)
# ---------------------------------------------------------------------------#
def run_logic(current_price: float, predicted_price: float, ticker: str):
    """
    Live mode:
    • trains PPO on the *entire* CSV
    • decides BUY / SELL / HOLD using the latest row + account state
    """
    import logging
    from forest import api, buy_shares, sell_shares  # brokerage SDK

    logger = logging.getLogger(__name__)
    df = load_tsla_csv()                             # full history
    model = train_ppo(df)                            # train from scratch

    # --- brokerage state ---------------------------------------------------
    try:
        account = api.get_account()
        cash = float(account.cash)
    except Exception as e:
        logger.error(f"[{ticker}] Account fetch error: {e}")
        return

    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception:
        position_qty = 0.0

    # --- latest observation -----------------------------------------------
    last_row = df.iloc[-1]
    obs_vec = np.concatenate(
        [last_row[OBS_FEATURES].values.astype(np.float32),
         [cash, position_qty]]
    )
    action, _ = model.predict(obs_vec, deterministic=True)

    logger.info(
        f"[{ticker}] PPO action={action} | Px={current_price:.2f} "
        f"| Qty={position_qty} | Cash={cash:.2f}"
    )

    # --- translate action --------------------------------------------------
    if action == 1 and position_qty == 0:                           # BUY
        qty = int(cash // current_price)
        if qty:
            logger.info(f"[{ticker}] BUY {qty} @ {current_price}")
            buy_shares(ticker, qty, current_price, predicted_price)
    elif action == 2 and position_qty > 0:                          # SELL
        logger.info(f"[{ticker}] SELL {position_qty} @ {current_price}")
        sell_shares(ticker, position_qty, current_price, predicted_price)
    else:
        logger.info(f"[{ticker}] HOLD / no-op.")


# ---------------------------------------------------------------------------#
#  BACK-TEST LOGIC  (one candle ahead)
# ---------------------------------------------------------------------------#
def run_backtest(
    current_price: float,
    predicted_price: float,
    position_qty: float,
    current_timestamp,                # str or pd-timestamp
    candles,                          # unused – provided by caller
) -> str:
    """
    Back-test mode:
    • uses ONLY data ≤ current_timestamp  (no future leakage)
    • keeps last 1 400 candles if more available
    • retrains PPO each call
    • returns BUY / SELL / NONE
    """
    if not isinstance(current_timestamp, pd.Timestamp):
        current_timestamp = pd.Timestamp(current_timestamp)

    df_full = load_tsla_csv()
    df_train = df_full[df_full["timestamp"] <= current_timestamp].copy()
    if df_train.empty:
        return "NONE"

    # enforce rolling window (max 1 400 candles)
    df_train = df_train.iloc[-ROLLING_WINDOW:]

    model = train_ppo(df_train)

    # observation built from *current_timestamp* row
    row = df_train.iloc[-1]
    cash_now = 0.0 if position_qty else START_CASH
    obs_vec = np.concatenate(
        [row[OBS_FEATURES].values.astype(np.float32),
         [cash_now, position_qty]]
    )
    action, _ = model.predict(obs_vec, deterministic=True)

    if action == 1 and position_qty == 0:
        return "BUY"
    if action == 2 and position_qty > 0:
        return "SELL"
    return "NONE"
