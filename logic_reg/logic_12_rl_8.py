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


# Memory System using ChromaDB and SBERT embedders
class MemorySystem:
    def __init__(self, dim=384):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name='trade_memory',
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )

    def store_trade(self, metadata: dict, sequence: list, summary: str):
        self.collection.add(
            metadatas=[metadata],
            documents=[summary],
            embeddings=None,
            ids=[str(metadata['entry_time'])]
        )

    def query(self, summary: str, n_results=5):
        results = self.collection.query(
            query_texts=[summary], n_results=n_results
        )
        return results['metadatas'][0]

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

# Main run_logic for live trading

def run_logic(current_price: float, predicted_price: float, ticker: str):
    """
    Executes the RL-driven strategy: predicts action with pre-trained agent.
    """
    # Load or train model
    try:
        model = load_model(ticker)
    except Exception:
        # fallback: train on full CSV
        df_full = pd.read_csv(get_csv_filename(ticker), parse_dates=['timestamp'])
        model = train_agent(df_full)
        save_model(model, ticker)

    # Load latest data
    df = pd.read_csv(get_csv_filename(ticker), parse_dates=['timestamp'])
    # keep only numeric features
    numeric_df = df.select_dtypes(include=[np.number]).dropna().reset_index(drop=True)
    if numeric_df.empty:
        logger.error("No numeric data available for ticker %s", ticker)
        return
    # build observation
    last_features = numeric_df.iloc[-1].values.astype(np.float32)
    obs = np.concatenate([last_features, np.zeros(384, dtype=np.float32)])
    # predict action
    action, _ = model.predict(obs)
    logger.info(f"Action {action} predicted by RL model for {ticker}")

    from forest import api, buy_shares, sell_shares
    if action == 1:
        max_shares = int(api.get_account().cash // current_price)
        buy_shares(ticker, max_shares, current_price, predicted_price)
    elif action == 2:
        pos = api.get_position(ticker)
        sell_shares(ticker, float(pos.qty), current_price, predicted_price)
    # action == 0 → hold


def run_backtest(current_price: float,
                 predicted_price: float,
                 position_qty: float,
                 current_timestamp: str,
                 candles: pd.DataFrame) -> str:
    """
    Backtests step-by-step with online learning: trains agent up to current_timestamp.
    """
    # 1) Train up to current_timestamp
    ts = pd.to_datetime(current_timestamp)
    ticker = TICKERS[0]
    hist_df = load_data_until(ticker, ts)
    memory = MemorySystem()
    model = train_agent(hist_df, memory)
    save_model(model, ticker)

    # 2) Prepare obs from the passed-in candles DataFrame
    numeric_candles = (
        candles
        .select_dtypes(include=[np.number])
        .dropna()
        .reset_index(drop=True)
    )
    if numeric_candles.empty:
        raise ValueError("No numeric data in candles DataFrame for backtest.")
    last_features = numeric_candles.iloc[-1].values.astype(np.float32)
    obs = np.concatenate([last_features, np.zeros(384, dtype=np.float32)])

    # 3) Predict & log
    action, _ = model.predict(obs)
    action_map = {0: 'NONE', 1: 'BUY', 2: 'SELL'}

    pv = position_qty * current_price
    log_entry = {
        'timestamp': current_timestamp,
        'action':   action_map[action],
        'position_qty': position_qty,
        'portfolio_value': pv
    }
    pd.DataFrame([log_entry]) \
      .to_csv(
        TRADES_LOG,
        mode='a',
        header=not os.path.exists(TRADES_LOG),
        index=False
      )
    logger.info(f"Backtest step at {current_timestamp}: action={action_map[action]}")
    return action_map[action]
