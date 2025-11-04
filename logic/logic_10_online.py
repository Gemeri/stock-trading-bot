import os
import pickle
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import gym
import sys
from gym import spaces
from dotenv import load_dotenv
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.policies import ActorCriticPolicy

# Memory system
import faiss
from sentence_transformers import SentenceTransformer

# Your trading API
from forest import api, buy_shares, sell_shares

# ---------------------------------
# Configuration & Utilities
# ---------------------------------
load_dotenv()
BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "1Hour")

TIMEFRAME_MAP = {
    "4Hour": "H4",
    "2Hour": "H2",
    "1Hour": "H1",
    "30Min": "M30",
    "15Min": "M15"
}
CONVERTED_TIMEFRAME = TIMEFRAME_MAP.get(BAR_TIMEFRAME, BAR_TIMEFRAME)

FEATURES = [
    'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions', 'sentiment',
    'greed_index', 'news_count', 'news_volume_z', 'price_change', 'high_low_range', 
    'macd_line', 'macd_signal', 'macd_histogram', 'rsi', 'roc', 'atr', 'ema_9', 'ema_21', 
    'ema_50', 'ema_200', 'adx', 'obv', 'bollinger_lower', 'bollinger_percB', 'returns_1', 
    'returns_3', 'returns_5', 'std_5', 'std_10', 'lagged_close_1', 'lagged_close_2', 
    'lagged_close_3', 'lagged_close_5', 'lagged_close_10', 'candle_body_ratio', 
    'wick_dominance', 'gap_vs_prev', 'volume_zscore', 'atr_zscore', 'rsi_zscore',
    'month', 'hour_sin', 'hour_cos', 'day_of_week_sin',  'day_of_week_cos', 
    'days_since_high', 'days_since_low', "d_sentiment"
]

sys.setrecursionlimit(10000)

def get_csv_filename(ticker: str) -> str:
    return os.path.join("data", f"{ticker}_{CONVERTED_TIMEFRAME}.csv")

def preprocess_df(df: pd.DataFrame, FEATURES: list) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    df[FEATURES] = (
        df[FEATURES]
        .fillna(method="ffill")
        .fillna(method="bfill")
        .fillna(0)
        .replace([np.inf, -np.inf], 0)
    )
    return df
# ---------------------------------
# Custom Gym Environment
# ---------------------------------

EXCLUDE_ON_SAVE: list[str] = ["env"]

class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, df: pd.DataFrame, initial_cash: float = 1e6):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_cash = initial_cash

        # Actions: 0 = HOLD, 1 = BUY (all-in), 2 = SELL (all-out)
        self.action_space = spaces.Discrete(3)

        # Observations: your full feature vector
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(FEATURES),), dtype=np.float32
        )

        self._reset_internal()

    def _reset_internal(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares = 0
        self.done = False
        self.entry_step = None
        self.entry_price = None

    def reset(self):
        self._reset_internal()
        return self._get_obs()

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        return row[FEATURES].values.astype(np.float32)

    def step(self, action: int):
        row = self.df.iloc[self.current_step]
        price = float(row["open"])  # or "close", choose your fill price

        reward = 0.0
        info = {}
        # BUY
        if action == 1 and self.shares == 0:
            self.shares = int(self.cash // price)
            self.cash -= self.shares * price
            self.entry_step = self.current_step
            self.entry_price = price
        # SELL
        elif action == 2 and self.shares > 0:
            proceeds = self.shares * price
            pnl = proceeds - (self.shares * self.entry_price)
            self.cash += proceeds
            reward = pnl
            # record trade metadata in info
            info["trade"] = {
                "entry_step": self.entry_step,
                "exit_step": self.current_step,
                "entry_price": self.entry_price,
                "exit_price": price,
                "pnl": pnl
            }
            self.shares = 0
            self.entry_step = None
            self.entry_price = None
        # HOLD or invalid BUY/SELL => no change

        # advance
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True

        obs = self._get_obs()
        return obs, reward, self.done, info

    def render(self, mode="human"):
        # Optional: integrate dashboard or print statements
        pass


# ---------------------------------
# Memory System (FAISS + SBERT)
# ---------------------------------
class MemorySystem:
    INDEX_FILE = "memory.index"
    RECORDS_FILE = "memory.pkl"

    def __init__(self, embedding_dim: int = 384):
        self.dimension = embedding_dim
        self.index = faiss.IndexFlatL2(self.dimension)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.records = []  # list of dicts: ticker, entry/exit times, pnl, actions, summary

        # Try load existing
        if os.path.exists(self.INDEX_FILE):
            self.index = faiss.read_index(self.INDEX_FILE)
        if os.path.exists(self.RECORDS_FILE):
            with open(self.RECORDS_FILE, "rb") as f:
                self.records = pickle.load(f)

    def store_trade(self, ticker, metadata: dict, summary: str):
        emb = self.embedder.encode(summary)
        self.index.add(np.array([emb], dtype=np.float32))
        self.records.append({
            "ticker": ticker,
            **metadata,
            "summary": summary
        })

    def save(self):
        faiss.write_index(self.index, self.INDEX_FILE)
        with open(self.RECORDS_FILE, "wb") as f:
            pickle.dump(self.records, f)


# ---------------------------------
# Core Logic Functions
# ---------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def train_or_load_agent(env: gym.Env, model_path: str) -> tuple[PPO, VecNormalize]:
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    norm_stats_path = model_path.replace(".zip", "_vecnormalize.pkl")

    if os.path.exists(model_path) and os.path.exists(norm_stats_path):
        custom_objects = {"policy_class": ActorCriticPolicy}
        model = PPO.load(model_path, env=vec_env, custom_objects=custom_objects)
        vec_env = VecNormalize.load(norm_stats_path, vec_env)
        model.set_env(vec_env)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log="./ppo_tb/"
        )
        model.learn(total_timesteps=200_000)
        model.save(model_path, exclude=EXCLUDE_ON_SAVE)
        vec_env.save(norm_stats_path)

    return model, vec_env


def run_backtest(current_price: float,
                 predicted_price: float,
                 position_qty: float,
                 current_timestamp: pd.Timestamp,
                 candles: pd.DataFrame,
                 ticker) -> str:
    # load & preprocess full history
    full = pd.read_csv(get_csv_filename(ticker), parse_dates=["timestamp"])
    full = preprocess_df(full, FEATURES)

    # slice to avoid future data
    hist = full[full["timestamp"] <= current_timestamp].reset_index(drop=True)
    env = TradingEnv(hist)

    # train/load + online tune
    model_path = f"{ticker}_ppo.zip"
    norm_stats_path = model_path.replace(".zip", "_vecnormalize.pkl")
    model, vec_env = train_or_load_agent(env, model_path)

    model.learn(total_timesteps=10_000)
    # again exclude the env when saving
    model.save(model_path, exclude=EXCLUDE_ON_SAVE)
    vec_env.save(norm_stats_path)

    # prepare & normalize obs from `candles`
    last_obs = candles.iloc[-1][FEATURES].values.astype(np.float32)
    obs = np.expand_dims(last_obs, 0)
    obs = vec_env.normalize_obs(obs)

    action, _ = model.predict(obs, deterministic=True)
    action = int(action[0])

    if action == 1 and position_qty == 0:
        return "BUY"
    elif action == 2 and position_qty > 0:
        return "SELL"
    else:
        return "NONE"
