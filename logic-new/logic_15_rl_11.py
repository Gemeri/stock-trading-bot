"""
rl_trade_logic.py

An end-to-end RL-based trading logic module:
  - run_logic: live trading – trains on full history, picks action BUY/SELL/NONE.
  - run_backtest: backtesting – online learning up to current_timestamp, then action.
  - MemorySystem: vector DB of past trades for context retrieval.
  - Persistence of model and memory between runs.
  - Logging & monitoring of trade performance.
"""

import os
import pickle
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import gym
from gym import spaces
from dotenv import load_dotenv
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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
TICKERS = os.getenv("TICKERS", "TSLA").split(",")

TIMEFRAME_MAP = {
    "4Hour": "H4",
    "2Hour": "H2",
    "1Hour": "H1",
    "30Min": "M30",
    "15Min": "M15"
}
CONVERTED_TIMEFRAME = TIMEFRAME_MAP.get(BAR_TIMEFRAME, BAR_TIMEFRAME)

FEATURES = [
    "open","high","low","close","volume","vwap",
    "macd_line","macd_signal","macd_histogram",
    "ema_9","ema_21","ema_50","ema_200",
    "adx","rsi","momentum","roc","atr","obv",
    "bollinger_upper","bollinger_lower",
    "lagged_close_1","lagged_close_2","lagged_close_3",
    "lagged_close_5","lagged_close_10",
    "sentiment","predicted_close"
]

def get_csv_filename(ticker: str) -> str:
    """Constructs the CSV filename for a given ticker & timeframe."""
    return f"{ticker}_{CONVERTED_TIMEFRAME}.csv"

def preprocess_df(df: pd.DataFrame, FEATURES: list) -> pd.DataFrame:
    """
    Sorts by timestamp and fills or clamps any NaN/Inf in FEATURES.
    """
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
class TradingEnv(gym.Env):
    """A simple trading environment for RL, no shorts, discrete actions."""
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
    """Stores embeddings of past trades for similarity-based recall."""
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
        """Add a new trade to the index & records."""
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
    """
    Wrap env in DummyVecEnv+VecNormalize(obs+reward), then
    load an existing PPO or train a new one, saving both model
    and normalization stats.
    """
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    norm_stats_path = model_path.replace(".zip", "_vecnormalize.pkl")

    if os.path.exists(model_path) and os.path.exists(norm_stats_path):
        model = PPO.load(model_path, env=vec_env)
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
        model.save(model_path)
        vec_env.save(norm_stats_path)

    return model, vec_env

# ---------------------------------
# run_logic — live trading
# ---------------------------------
def run_logic(current_price: float, predicted_price: float, ticker: str):
    """
    Live trading:
      1. Load & preprocess full CSV history.
      2. Build env, train/load PPO + VecNormalize.
      3. Normalize last obs, predict action.
      4. Execute buy/sell via forest.api.
    """
    load_dotenv()

    df = pd.read_csv(get_csv_filename(ticker), parse_dates=["timestamp"])
    df = preprocess_df(df, FEATURES)

    env = TradingEnv(df)
    model_path = f"{ticker}_ppo.zip"
    model, vec_env = train_or_load_agent(env, model_path)

    last_obs = df.iloc[-1][FEATURES].values.astype(np.float32)
    obs = np.expand_dims(last_obs, 0)
    obs = vec_env.normalize_obs(obs)

    action, _ = model.predict(obs, deterministic=True)
    action = int(action[0])

    if action == 1:   # BUY
        cash = float(api.get_account().cash)
        qty = int(cash // current_price)
        if qty > 0:
            buy_shares(ticker, qty, current_price, predicted_price)
    elif action == 2: # SELL
        pos = api.get_position(ticker)
        qty = float(pos.qty) if pos else 0
        if qty > 0:
            sell_shares(ticker, qty, current_price, predicted_price)
    # else NONE

def run_backtest(current_price: float,
                 predicted_price: float,
                 position_qty: float,
                 current_timestamp: pd.Timestamp,
                 candles: pd.DataFrame) -> str:
    """
    Backtesting step:
      - Preprocess full CSV up to current_timestamp.
      - Train/load PPO+VecNormalize, then fine-tune 10k steps.
      - Normalize last candle obs, predict, return BUY/SELL/NONE.
    """
    full = pd.read_csv(get_csv_filename(TICKERS[0]), parse_dates=["timestamp"])
    full = preprocess_df(full, FEATURES)

    hist = full[full["timestamp"] <= current_timestamp].reset_index(drop=True)
    env = TradingEnv(hist)

    model_path = f"{TICKERS[0]}_ppo.zip"
    model, vec_env = train_or_load_agent(env, model_path)
    model.learn(total_timesteps=10_000)
    model.save(model_path)
    vec_env.save(model_path.replace(".zip", "_vecnormalize.pkl"))

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