from __future__ import annotations

import json
import logging
import math
import os
import shutil
import tempfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple
import torch.nn as nn

import gymnasium as gym
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# --------------------------------------------------------------------------- #
#                                CONFIGURATION                                #
# --------------------------------------------------------------------------- #
CONFIG: Dict[str, object] = {
    "FRAME_STACK": 3,
    "FEATURE_NAMES": [
        "timestamp",
        "open", "high", "low", "close", "volume", "vwap", "sentiment",
        "macd_line", "macd_signal", "macd_histogram",
        "rsi", "momentum", "roc", "atr", "obv",
        "bollinger_upper", "bollinger_lower",
        "ema_9", "ema_21", "ema_50", "ema_200", "adx",
        "lagged_close_1", "lagged_close_2", "lagged_close_3",
        "lagged_close_5", "lagged_close_10",
        "candle_body_ratio", "predicted_close",
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
    ],
    "BUFFER_SIZE": 50_000,
    "TRAIN_TIMESTEPS": 12_000,
    "PPO_PARAMS": {
        "policy": "MlpPolicy",
        "policy_kwargs": dict(net_arch=[256, 128], activation_fn=nn.Tanh),
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 512,
        "gamma": 0.999,
        "verbose": 0,
    },
    "SHARPE_TARGET": 1.5,
    "COMMISSION": 0.0005,   # 5 bp
    "DRAWDOWN_PENALTY": 0.001,
    "LOG_LEVEL": logging.INFO,
}
REQUIRED_FEATS: List[str] = [f for f in CONFIG["FEATURE_NAMES"] if f != "timestamp"]

# --------------------------------------------------------------------------- #
#                               Utility helpers                               #
# --------------------------------------------------------------------------- #
load_dotenv()
BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "1Hour")
TICKERS_ENV = os.getenv("TICKERS", "TSLA").split(",")

TIMEFRAME_MAP = {"4Hour": "H4", "2Hour": "H2", "1Hour": "H1",
                 "30Min": "M30", "15Min": "M15"}
CONVERTED = TIMEFRAME_MAP.get(BAR_TIMEFRAME, BAR_TIMEFRAME)


def csv_path(ticker: str) -> str:
    return f"{ticker}_{CONVERTED}.csv"


def atomic_save(src: Path, dst: Path) -> None:
    tmp = dst.with_suffix(".tmp")
    shutil.move(src, tmp)
    tmp.replace(dst)


# --------------------------------------------------------------------------- #
#                          Prioritised Replay Buffer                          #
# --------------------------------------------------------------------------- #
class PrioritisedReplayBuffer:

    def __init__(self, size: int, alpha: float = 0.6):
        self.size = size
        self.alpha = alpha
        self._storage: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self._priorities: Deque[float] = deque(maxlen=size)
        self._pos = 0

    def _max_priority(self) -> float:
        return max(self._priorities, default=1.0)

    def add(self, transition: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> None:
        priority = self._max_priority()
        if len(self._storage) < self.size:
            self._storage.append(transition)
            self._priorities.append(priority)
        else:
            self._storage[self._pos] = transition
            self._priorities[self._pos] = priority
            self._pos = (self._pos + 1) % self.size

    def sample_env(self, batch_size: int = 2048) -> gym.Env:
        probs = np.array(self._priorities) ** self.alpha
        probs /= probs.sum()

        obs_dim = self._storage[0][0].shape[0]

        class ReplayEnv(gym.Env):
            metadata = {"render.modes": []}

            def __init__(self):
                super().__init__()
                self.observation_space = spaces.Box(-np.inf, np.inf,
                                                    shape=(obs_dim,),
                                                    dtype=np.float32)
                self.action_space = spaces.Discrete(3)
                self._steps = 0

            def reset(self, seed: Optional[int] = None, **kwargs):
                self._steps = 0
                idx = np.random.choice(len(probs), p=probs)
                s, a, r, s2, done = self._storage[idx]
                return s, {}

            def step(self, action):
                idx = np.random.choice(len(probs), p=probs)
                s, a, r, s2, done = self._storage[idx]
                self._steps += 1
                trunc = self._steps >= batch_size
                return s2, r, done, trunc, {}

        return ReplayEnv()

    def __len__(self) -> int:  # noqa: D401
        return len(self._storage)


# --------------------------------------------------------------------------- #
#                            Trading Gym Environment                          #
# --------------------------------------------------------------------------- #
class RLTradingEnv(gym.Env):

    metadata = {"render.modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        frame_stack: int,
        commission: float,
        dd_penalty: float,
        normaliser: "RunningNorm",
    ):
        super().__init__()
        self.df = df.copy().reset_index(drop=True)
        self.frame_stack = frame_stack
        self.cursor = frame_stack - 1  # start after enough history
        self.position_qty = 0
        self.entry_price = 0.0
        self.commission = commission
        self.dd_penalty = dd_penalty
        self.running_max = -np.inf
        self.normaliser = normaliser

        self.feature_cols = [
            c for c in df.columns if c != "timestamp"
        ]  # drop timestamp

        obs_dim = len(self.feature_cols) * frame_stack
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # BUY = 0, SELL = 1, NONE = 2

    # --------------------------------------------------------------------- #
    #                           Gym primitives                              #
    # --------------------------------------------------------------------- #
    def _get_obs(self) -> np.ndarray:
        frames_normed: List[np.ndarray] = []
        for i in range(self.frame_stack):
            raw = (
                self.df
                .iloc[self.cursor - i][self.feature_cols]
                .values
                .astype(np.float32)
            )
            normed = self.normaliser.transform(raw)
            frames_normed.append(normed)
        # maintain chronological order, then flatten
        return np.concatenate(frames_normed[::-1])

    def reset(self, seed: Optional[int] = None, **kwargs):
        super().reset(seed=seed)
        self.cursor = self.frame_stack - 1
        self.position_qty = 0
        self.entry_price = 0.0
        self.running_max = -np.inf
        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        current_row = self.df.iloc[self.cursor]
        price = current_row["close"]

        reward = 0.0
        # BUY
        if action == 0 and self.position_qty == 0:
            self.position_qty = 1
            self.entry_price = price
            reward -= self.commission
        # SELL
        elif action == 1 and self.position_qty > 0:
            pnl = price - self.entry_price
            reward += pnl - self.commission
            self.position_qty = 0
            self.entry_price = 0.0
        # NONE – no direct cost

        unreal_pnl = 0.0                            # ← default value
        if self.position_qty > 0:
            unreal_pnl = price - self.entry_price
            reward += unreal_pnl

        # Drawdown penalty
        portfolio_value = (
            unreal_pnl if self.position_qty > 0 else 0.0
        )
        self.running_max = max(self.running_max, portfolio_value)
        drawdown = max(0.0, self.running_max - portfolio_value)
        reward -= self.dd_penalty * drawdown

        # advance time
        self.cursor += 1
        done = self.cursor >= len(self.df) - 1
        trunc = False

        obs = self._get_obs() if not done else self.observation_space.sample()
        info = {}
        return obs, reward, done, trunc, info


# --------------------------------------------------------------------------- #
#                            Running mean/var norm                            #
# --------------------------------------------------------------------------- #
class RunningNorm:

    def __init__(self, eps: float = 1e-8):
        self.mean = None
        self.var = None
        self.count = 0
        self.eps = eps

    def update(self, x: np.ndarray) -> None:
        x = x.astype(np.float32)
        if self.mean is None:
            self.mean = np.zeros_like(x)
            self.var = np.ones_like(x)

        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += delta * delta2

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        if self.mean is None:
            return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        base_dim = self.mean.shape[0]
        if x.ndim == 1 and x.shape[0] != base_dim:
            if x.shape[0] % base_dim != 0:
                raise ValueError(f"Cannot normalise vector of length {x.shape[0]}")
            repeats = x.shape[0] // base_dim
            mean = np.tile(self.mean, repeats)
            var = np.tile(self.var, repeats)
        else:
            mean, var = self.mean, self.var
        normalized = (x - mean) / (np.sqrt(var / max(1, self.count)) + self.eps)
        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------- #
#                                SB3 Callback                                 #
# --------------------------------------------------------------------------- #
class ReplayCallback(BaseCallback):
    def __init__(self, buffer: PrioritisedReplayBuffer, verbose: int = 0):
        super().__init__(verbose)
        self.buffer = buffer

    def _on_step(self) -> bool:  # noqa: D401
        replay_env = self.training_env
        # no direct access – skip
        return True


# --------------------------------------------------------------------------- #
#                                Trader class                                 #
# --------------------------------------------------------------------------- #
class Trader:
    _instances: Dict[str, "Trader"] = {}

    def __new__(cls, ticker: str):
        if ticker not in cls._instances:
            cls._instances[ticker] = super().__new__(cls)
        return cls._instances[ticker]

    # --------------------------------------------------------------------- #
    def __init__(self, ticker: str):
        if hasattr(self, "_inited"):
            return
        self._inited = True

        self.ticker = ticker
        self.model_path = Path(f"{ticker}_rl_agent.zip")
        self.memory_dir = Path("AgentMemory") / ticker
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.logfile = self.memory_dir / "trades.log"

        self.buffer = PrioritisedReplayBuffer(CONFIG["BUFFER_SIZE"])
        self.normaliser = RunningNorm()

        logging.basicConfig(
            level=CONFIG["LOG_LEVEL"],
            format=f"[{ticker}] %(asctime)s ‑ %(levelname)s ‑ %(message)s",
        )
        self.logger = logging.getLogger(ticker)

        self.model: Optional[PPO] = None
        self._load_or_init_model()

        # In‑memory df cache to avoid repeated disk reads
        self.historical_df: Optional[pd.DataFrame] = None
        self.last_loaded_ts: Optional[pd.Timestamp] = None

    # --------------------------------------------------------------------- #
    #                           Model management                             #
    # --------------------------------------------------------------------- #
    def _load_or_init_model(self) -> None:
        expected_obs_shape = (len(REQUIRED_FEATS) * CONFIG["FRAME_STACK"],)
        if self.model_path.exists():
            try:
                self.model = PPO.load(self.model_path, device="cpu")
                if self.model.observation_space.shape != expected_obs_shape:
                    self.logger.warning("Feature dimension changed – rebuilding PPO.")
                    self.model = None
                else:
                    self.logger.info("Loaded existing agent.")
            except Exception as e:  # noqa: BLE001
                self.logger.error(f"Could not load agent – {e}; re-initialising.")
                self.model = None

        if self.model is None:
            dummy_df = pd.DataFrame(
                [[0.0] * len(CONFIG["FEATURE_NAMES"])],
                columns=CONFIG["FEATURE_NAMES"],
            )
            dummy_df = self._prepare_dataframe(dummy_df)
            env = RLTradingEnv(
                dummy_df,
                CONFIG["FRAME_STACK"],
                CONFIG["COMMISSION"],
                CONFIG["DRAWDOWN_PENALTY"],
                self.normaliser,
            )
            self.model = PPO(env=env, **CONFIG["PPO_PARAMS"])
            self.logger.info("Initialised new PPO agent.")

    # --------------------------------------------------------------------- #
    def _load_data_until(self, ts: pd.Timestamp | str) -> pd.DataFrame:
        # 1) parse ts
        if isinstance(ts, str):
            try:
                ts = pd.to_datetime(ts)
            except Exception:
                self.logger.error(f"Could not parse timestamp {ts!r}")
                return pd.DataFrame(columns=CONFIG["FEATURE_NAMES"])

        # 2) reload only when needed
        need_reload = (
            self.historical_df is None
            or self.last_loaded_ts is None          # ← added guard
            or ts > self.last_loaded_ts
        )
        if need_reload:
            try:
                df = (
                    pd.read_csv(
                        csv_path(self.ticker),
                        parse_dates=["timestamp"],
                        infer_datetime_format=True,
                    )
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )
            except FileNotFoundError:
                self.logger.error("CSV not found.")
                return pd.DataFrame(columns=CONFIG["FEATURE_NAMES"])
            self.historical_df = df
            self.last_loaded_ts = df["timestamp"].iloc[-1]

        # 3) return slice ≤ ts
        return self.historical_df[self.historical_df["timestamp"] <= ts].copy()


    # --------------------------------------------------------------------- #
    def _incremental_train(self, upto_ts: pd.Timestamp) -> None:
        df = self._load_data_until(upto_ts)
        if df.empty or len(df) <= CONFIG["FRAME_STACK"]:
            return

        df = (
            df.fillna(method="ffill")
            .fillna(method="bfill")
            .fillna(0.0)
            .assign(predicted_close=df["close"])  # ensure column exists
        )
        df = self._prepare_dataframe(df)

        for row in df.drop(columns=["timestamp"]).values.astype(np.float32):
            self.normaliser.update(row)

        env = RLTradingEnv(
            df,
            CONFIG["FRAME_STACK"],
            CONFIG["COMMISSION"],
            CONFIG["DRAWDOWN_PENALTY"],
            self.normaliser,
        )
        # --------- FIX: rebuild model if obs-space size changed -------------
        if self.model.observation_space.shape != env.observation_space.shape:
            self.logger.warning("Observation space mismatch – re-creating PPO.")
            self.model = PPO(env=env, **CONFIG["PPO_PARAMS"])
        else:
            self.model.set_env(env)

        self.model.learn(CONFIG["TRAIN_TIMESTEPS"], progress_bar=False)

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td) / "agent.zip"
            self.model.save(tmp_path)
            atomic_save(tmp_path, self.model_path)

    @staticmethod
    def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in REQUIRED_FEATS:
            if col not in df.columns:
                df[col] = 0.0
        extras = [c for c in df.columns if c not in REQUIRED_FEATS and c != "timestamp"]
        df = df.drop(columns=extras, errors="ignore")
        return df[["timestamp"] + REQUIRED_FEATS]

    @staticmethod
    def _prepare_row(row: pd.Series) -> pd.Series:
        data = {"timestamp": row.get("timestamp", pd.Timestamp.utcnow())}
        for col in REQUIRED_FEATS:
            data[col] = row.get(col, 0.0)
        return pd.Series(data)

    # --------------------------------------------------------------------- #
    def act(self, row: pd.Series) -> str:
        """Return BUY / SELL / NONE given latest candle row."""
        obs = self._row_to_obs(row)
        action_idx, _ = self.model.predict(obs, deterministic=True)
        action_str = {0: "BUY", 1: "SELL", 2: "NONE"}[int(action_idx)]
        return action_str

    def _row_to_obs(self, row: pd.Series) -> np.ndarray:
        row = self._prepare_row(row)
        values = row.loc[REQUIRED_FEATS].astype(np.float32).values
        stacked = np.tile(values, CONFIG["FRAME_STACK"])
        return self.normaliser.transform(stacked)

    # --------------------------------------------------------------------- #
    def log_trade(self, row: pd.Series, action: str, reward: float = 0.0) -> None:
        # 1) Timestamp to ISO string
        ts = row.get("timestamp", None)
        if isinstance(ts, str):
            try:
                ts = pd.to_datetime(ts)
            except Exception:
                pass
        ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)

        # 2) Convert features to native Python types
        features: list = []
        for col in row.index:
            if col == "timestamp":
                continue
            val = row[col]
            # unwrap numpy scalar
            if isinstance(val, (np.generic,)):
                val = val.item()
            # ensure Python float for integer/float
            if isinstance(val, (int, float)):
                val = float(val)
            features.append(val)

        # 3) Build entry and write
        entry = {
            "timestamp": ts_str,
            "features": features,
            "action": action,
            "reward": float(reward),
        }
        try:
            with self.logfile.open("a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            self.logger.error(f"Trade log failed: {exc}")

    # --------------------------------------------------------------------- #
    #                         Public high‑level API                          #
    # --------------------------------------------------------------------- #
    def decide_live(
        self, current_price: float, predicted_price: float
    ) -> Tuple[str, pd.Series]:
        try:
            df = pd.read_csv(
                csv_path(self.ticker),
                parse_dates=["timestamp"],
                infer_datetime_format=True,
            )
        except FileNotFoundError:
            self.logger.error("CSV not found for live decide.")
            return "NONE", pd.Series(dtype=float)

        row = df.iloc[-1].copy()
        row["predicted_close"] = predicted_price
        row = self._prepare_row(row)
        action = self.act(row)
        self.log_trade(row, action)
        return action, row

    # decide_backtest remains unchanged except for adding _prepare_row call
    def decide_backtest(
        self,
        current_price: float,
        predicted_price: float,
        current_timestamp: pd.Timestamp,
        candles: pd.DataFrame,
    ) -> str:
        self._incremental_train(current_timestamp)
        row = candles.iloc[-1].copy()
        row["predicted_close"] = predicted_price
        row = self._prepare_row(row)
        action = self.act(row)
        self.log_trade(row, action)
        return action

    # --------------------------------------------------------------------- #
    def offline_train_full(self) -> None:
        try:
            df = pd.read_csv(
                csv_path(self.ticker),
                parse_dates=["timestamp"],
                infer_datetime_format=True,
            )
        except FileNotFoundError:
            self.logger.error("CSV not found for offline training.")
            return
        self.logger.info(f"Starting offline training on {len(df)} rows.")
        self._incremental_train(df["timestamp"].iloc[-1])
        self.logger.info("Offline training complete.")


# --------------------------------------------------------------------------- #
#                      run_logic / run_backtest interface                     #
# --------------------------------------------------------------------------- #
def run_logic(
    current_price: float,
    predicted_price: float,
    ticker: str,
) -> None:
    from forest import api, buy_shares, sell_shares  # type: ignore

    trader = Trader(ticker)
    action, _ = trader.decide_live(current_price, predicted_price)

    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception:
        position_qty = 0.0

    account_cash = 0.0
    try:
        account_cash = float(api.get_account().cash)
    except Exception:  # noqa: BLE001
        pass

    if action == "BUY" and position_qty == 0:
        max_shares = int(account_cash // current_price)
        if max_shares > 0:
            buy_shares(ticker, max_shares, current_price, predicted_price)
    elif action == "SELL" and position_qty > 0:
        sell_shares(ticker, position_qty, current_price, predicted_price)
    # NONE → pass


def run_backtest(
    current_price: float,
    predicted_price: float,
    position_qty: float,
    current_timestamp: pd.Timestamp,
    candles: pd.DataFrame,
) -> str:
    ticker = TICKERS_ENV[0]  # always single ticker per spec
    trader = Trader(ticker)
    action = trader.decide_backtest(
        current_price,
        predicted_price,
        current_timestamp,
        candles,
    )

    # Constrain by current position
    if action == "BUY" and position_qty > 0:
        return "NONE"
    if action == "SELL" and position_qty == 0:
        return "NONE"
    return action