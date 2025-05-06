import os
import logging
import joblib
import json
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# Load environment variables
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

# Paths for persistence
MODEL_DIR = "models"
MEMORY_DIR = "memory"
LOG_DIR = "logs"
TRADES_LOG = "trades/trade_log.csv"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(TRADES_LOG), exist_ok=True)

_backtest_model    = None
_backtest_env_df   = None
_backtest_memory   = None

# Configure logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'trading_agent.log'),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Utility to get CSV filename
def get_csv_filename(ticker: str) -> str:
    return f"{ticker}_{CONVERTED_TIMEFRAME}.csv"

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, memory_summary: np.ndarray = None):
        super().__init__()

        # 1) keep only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        # 2) drop any rows with NaNs (e.g. initial EMAs, RSI, etc)
        numeric_df = numeric_df.dropna().reset_index(drop=True)

        self.df = numeric_df
        self.current_step = 0
        self.position = 0
        self.cash = 1e6
        self.total_shares_bought = 0
        self.total_shares_sold = 0
        self.memory_summary = memory_summary

        # build observation space
        n_features = self.df.shape[1]
        mem_dim    = memory_summary.shape[0] if memory_summary is not None else 0
        obs_dim    = n_features + mem_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)

    def reset(self):
        self.current_step        = 0
        self.position            = 0
        self.cash                = 1e6
        self.total_shares_bought = 0
        self.total_shares_sold   = 0
        return self._get_observation()

    def step(self, action):
        # always fetch from numeric-only df
        current_price = float(self.df.iloc[self.current_step]['close'])
        done = False

        # BUY
        if action == 1 and self.position == 0:
            shares = int(self.cash // current_price)
            self.position += shares
            self.cash     -= shares * current_price
            self.total_shares_bought += shares
            logger.info(f"Bought {shares} @ {current_price}")

        # SELL
        elif action == 2 and self.position > 0:
            self.cash  += self.position * current_price
            self.total_shares_sold += self.position
            logger.info(f"Sold {self.position} @ {current_price}")
            self.position = 0

        # reward = change in portfolio value
        portfolio_value = self.cash + self.position * current_price
        reward = portfolio_value - 1e6

        self.current_step += 1
        if self.current_step >= len(self.df):
            done = True

        obs = self._get_observation() if not done else \
              np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {'portfolio_value': portfolio_value}
        return obs, reward, done, info

    def _get_observation(self):
        # numeric row + optional memory vector
        obs = self.df.iloc[self.current_step].values.astype(np.float32)
        if self.memory_summary is not None:
            obs = np.concatenate([obs, self.memory_summary])
        return obs


class MemorySystem:
    def __init__(self):
        # Keep an SBERT embedder around so we know the dimension
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.dim = self.embedder.get_sentence_embedding_dimension()

        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name='trade_memory',
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )

    def has_trades(self) -> bool:
        return self.collection.count() > 0

    def get_last_trade_summary(self) -> str:
        results = self.collection.get()
        # assume at least one document if has_trades() is True
        return results["documents"][-1]

    def store_trade(self, metadata: dict, sequence: list, summary: str):
        # drop None values so chroma only sees str/int/float/bool
        clean_meta = {k: v for k, v in metadata.items() if v is not None}
        self.collection.add(
            metadatas=[clean_meta],
            documents=[summary],
            embeddings=None,
            ids=[str(metadata['entry_time'])]
        )

    def query_summary(self, summary: str, n_results: int = 1) -> np.ndarray:
        # query the collection
        res = self.collection.query(query_texts=[summary], n_results=n_results)
        embs = res.get("embeddings")
        # if no embeddings returned or first is None, fall back to zero-vector
        if not embs or embs[0] is None:
            return np.zeros(self.dim, dtype=np.float32)
        return np.array(embs[0], dtype=np.float32)


# Load full CSV up to a given timestamp
def load_data_until(ticker: str, timestamp: pd.Timestamp) -> pd.DataFrame:
    fname = get_csv_filename(ticker)
    df = pd.read_csv(fname, parse_dates=['timestamp'])
    return df[df['timestamp'] <= timestamp].reset_index(drop=True)

# Train RL agent on historical data
def train_agent(historical_df: pd.DataFrame, memory: MemorySystem = None) -> PPO:
    memory_summary = None
    if memory:
        # aggregate memory summary vector
        # placeholder: zero vector
        memory_summary = np.zeros(384, dtype=np.float32)
    env = TradingEnv(historical_df, memory_summary)
    model = PPO('MlpPolicy', env, verbose=0)
    model.learn(total_timesteps=100_000)
    return model

# Save and load utilities
def save_model(model: PPO, ticker: str):
    path = os.path.join(MODEL_DIR, f"rl_agent_{ticker}.zip")
    model.save(path)


def load_model(ticker: str) -> PPO:
    path = os.path.join(MODEL_DIR, f"rl_agent_{ticker}.zip")
    return PPO.load(path)


def run_logic(current_price: float, predicted_price: float, ticker: str):
    """
    Live trading: identical online‐learning + memory pipeline to run_backtest,
    always feeding a fixed‐length memory_summary vector.
    """
    global _backtest_model, _backtest_env_df, _backtest_memory
    from forest import api, buy_shares, sell_shares

    # 1) Fetch cash & position
    try:
        cash = float(api.get_account().cash)
    except Exception as e:
        logger.error(f"[{ticker}] account error: {e}", exc_info=True)
        return
    try:
        pos_qty = float(api.get_position(ticker).qty)
    except Exception:
        pos_qty = 0.0

    # 2) Load and clean full CSV
    fname        = get_csv_filename(ticker)
    full_df      = pd.read_csv(fname, parse_dates=['timestamp'])
    numeric_full = (
        full_df
        .select_dtypes(include=[np.number])
        .dropna()
        .reset_index(drop=True)
    )
    if numeric_full.empty:
        logger.warning(f"[{ticker}] no numeric data; skipping.")
        return

    # 3) Warm-up or append the newest candle
    latest = numeric_full.iloc[[-1]].astype(np.float32)
    if _backtest_model is None:
        # First call: instantiate memory & env_df, warm-up
        _backtest_memory = MemorySystem()
        mem_zero         = np.zeros(_backtest_memory.dim, dtype=np.float32)
        _backtest_env_df = numeric_full.copy()
        env              = TradingEnv(_backtest_env_df, memory_summary=mem_zero)
        _backtest_model  = PPO("MlpPolicy", env, verbose=0)
        _backtest_model.learn(total_timesteps=50_000)
    else:
        # Append only the features we track
        common = [c for c in _backtest_env_df.columns if c in latest.columns]
        row_df = latest[common]
        _backtest_env_df = pd.concat([_backtest_env_df, row_df], ignore_index=True)

    # 4) Build a fixed-length memory vector
    if _backtest_memory.has_trades():
        last_sum = _backtest_memory.get_last_trade_summary()
        mem_vec   = _backtest_memory.query_summary(last_sum, n_results=1)
    else:
        mem_vec   = np.zeros(_backtest_memory.dim, dtype=np.float32)

    # 5) One-step online learning
    env = TradingEnv(_backtest_env_df, memory_summary=mem_vec)
    _backtest_model.set_env(env)
    _backtest_model.learn(total_timesteps=1, reset_num_timesteps=False)

    # 6) Predict action
    obs, = [env._get_observation()]
    raw_action, _ = _backtest_model.predict(obs, deterministic=True)
    action = int(raw_action.item() if hasattr(raw_action, "item") else raw_action)
    decision = {0: "NONE", 1: "BUY", 2: "SELL"}[action]
    logger.info(f"[{ticker}] RL decision = {decision}")

    # 7) Store trade in memory
    if decision != "NONE":
        meta = {
            "entry_time": str(full_df['timestamp'].iat[-1]),
            "position_qty": pos_qty
        }
        summ = f"{decision} {pos_qty} @ {current_price}"
        _backtest_memory.store_trade(meta, [], summ)

    # 8) Execute via broker API
    if decision == "BUY" and pos_qty == 0:
        qty = int(cash // current_price)
        if qty > 0:
            logger.info(f"[{ticker}] EXECUTE BUY {qty} @ {current_price}")
            buy_shares(ticker, qty, current_price, predicted_price)
    elif decision == "SELL" and pos_qty > 0:
        logger.info(f"[{ticker}] EXECUTE SELL {pos_qty} @ {current_price}")
        sell_shares(ticker, pos_qty, current_price, predicted_price)
    else:
        logger.info(f"[{ticker}] EXECUTE NONE")


def run_backtest(current_price: float,
                 predicted_price: float,
                 position_qty: float,
                 current_timestamp: str,
                 candles: pd.DataFrame) -> str:
    """
    Online-learning backtest with memory: appends only matching columns,
    always supplies a fixed-length memory_summary, and keeps observation
    space constant so set_env never fails.
    """
    global _backtest_model, _backtest_env_df, _backtest_memory

    # 1) Extract latest numeric row
    cand_num = (
        candles
        .select_dtypes(include=[np.number])
        .dropna(axis=1, how='any')
    )
    latest = cand_num.iloc[[-1]].astype(np.float32)

    # 2) Initial warm-up
    if _backtest_model is None:
        full = pd.read_csv(get_csv_filename(TICKERS[0]), parse_dates=["timestamp"])
        hist = (
            full
            .select_dtypes(include=[np.number])
            .dropna()
            .reset_index(drop=True)
        )
        _backtest_memory = MemorySystem()
        mem_zero         = np.zeros(_backtest_memory.dim, dtype=np.float32)
        _backtest_env_df = hist.copy()
        env              = TradingEnv(_backtest_env_df, memory_summary=mem_zero)
        _backtest_model  = PPO("MlpPolicy", env, verbose=0)
        _backtest_model.learn(total_timesteps=50_000)
    else:
        # 3) Append only the tracked feature columns
        common = [c for c in _backtest_env_df.columns if c in latest.columns]
        row_df = latest[common]
        _backtest_env_df = pd.concat([_backtest_env_df, row_df], ignore_index=True)

    # 4) Build fixed-length memory vector
    if _backtest_memory.has_trades():
        last_sum = _backtest_memory.get_last_trade_summary()
        mem_vec   = _backtest_memory.query_summary(last_sum, n_results=1)
    else:
        mem_vec   = np.zeros(_backtest_memory.dim, dtype=np.float32)

    # 5) One-step online update
    env = TradingEnv(_backtest_env_df, memory_summary=mem_vec)
    _backtest_model.set_env(env)
    _backtest_model.learn(total_timesteps=1, reset_num_timesteps=False)

    # 6) Predict
    obs, = [env._get_observation()]
    raw_action, _ = _backtest_model.predict(obs, deterministic=True)
    action        = int(raw_action.item() if hasattr(raw_action, "item") else raw_action)
    result        = {0: "NONE", 1: "BUY", 2: "SELL"}[action]

    # 7) Store trade if any
    if result != "NONE":
        meta    = {
            "entry_time": current_timestamp,
            "position_qty": position_qty
        }
        summ    = f"{result} {position_qty} @ {current_price}"
        _backtest_memory.store_trade(meta, [], summ)

    return result
