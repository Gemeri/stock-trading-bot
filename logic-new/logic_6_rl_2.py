"""
rl_trade_logic.py

Reinforcement‑Learning trading logic that adapts dynamically to historical data and
decides among BUY, SELL or NONE at each decision point.

Key design points
-----------------
* Very light‑weight tabular Q‑learning agent (24 discrete states × 3 actions) – no
  external ML dependencies required.
* State definition focuses on the most information‑dense technical signals **and**
  how well *predicted_close* has historically tracked the next‑period *close*:

    state = (predicted_diff_sign,
             rsi_zone,
             macd_sign,
             adx_trend)

  – predicted_diff_sign = 1 if (predicted_close − close) > 0 else 0  
  – rsi_zone ∈ {0:oversold, 1:neutral, 2:overbought}  
  – macd_sign ∈ {0:bearish, 1:bullish}  
  – adx_trend ∈ {0:weak, 1:strong}

* **No look‑ahead bias**: in *run_backtest* the agent is (re‑)trained only on rows
  with *timestamp ≤ current_timestamp*.

* Profitable bias: the reward is the *net* P/L obtained from opening/closing a
  long position (no shorting) minus a tiny transaction cost.

The file exposes the two required public functions:

    run_logic(current_price, predicted_price, ticker)
    run_backtest(current_price, predicted_price, position_qty,
                 current_timestamp, candles)

Both share the same internal helper pipeline; the only difference is that
*run_logic* executes real orders through the host `forest` API while
*run_backtest* merely returns strings ("BUY"/"SELL"/"NONE").

Because the Q‑table is tiny, each training pass takes milliseconds.
"""

import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Environment helpers                                                         #
# --------------------------------------------------------------------------- #

load_dotenv()
BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "1Hour")
TICKERS = os.getenv("TICKERS", "TSLA").split(",")  # guaranteed single ticker
TIMEFRAME_MAP = {"4Hour": "H4", "2Hour": "H2", "1Hour": "H1",
                 "30Min": "M30", "15Min": "M15"}
CONVERTED_TIMEFRAME = TIMEFRAME_MAP.get(BAR_TIMEFRAME, BAR_TIMEFRAME)
TRANSACTION_FEE = 0.0005  # 5 bps round‑turn

# Action constants
HOLD, BUY, SELL = 0, 1, 2
ACTION_TO_STR = {HOLD: "NONE", BUY: "BUY", SELL: "SELL"}


def get_csv_filename(ticker: str) -> str:
    return f"{ticker}_{CONVERTED_TIMEFRAME}.csv"


def load_price_data(ticker: str) -> pd.DataFrame:
    """Load CSV as DataFrame with datetime index (assumes file exists)."""
    fname = get_csv_filename(ticker)
    df = pd.read_csv(fname)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Tabular Q‑learning agent                                                    #
# --------------------------------------------------------------------------- #

class QLearningAgent:
    """Simple ε‑greedy Q‑learning for a very small discrete state‑space."""

    def __init__(self,
                 alpha: float = 0.2,
                 gamma: float = 0.9,
                 epsilon: float = 0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Nested dict: Q[state][action] = value
        self.Q = defaultdict(lambda: np.zeros(3, dtype=float))

    # ---------------------- state discretisation --------------------------- #

    @staticmethod
    def _rsi_zone(rsi: float) -> int:
        if rsi < 30:
            return 0
        if rsi > 70:
            return 2
        return 1

    @staticmethod
    def _macd_sign(macd_hist: float) -> int:
        return 1 if macd_hist >= 0 else 0

    @staticmethod
    def _adx_trend(adx: float) -> int:
        return 1 if adx >= 25 else 0

    def state_from_row(self, row: pd.Series) -> Tuple[int, int, int, int]:
        pred_sign = 1 if (row["predicted_close"] - row["close"]) >= 0 else 0
        return (
            pred_sign,
            self._rsi_zone(row["rsi"]),
            self._macd_sign(row["macd_histogram"]),
            self._adx_trend(row["adx"]),
        )

    # ----------------------------- RL API ---------------------------------- #

    def choose_action(self, state):
        """ε‑greedy selection."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(3)
        return int(np.argmax(self.Q[state]))

    def learn(self, s, a, r, s_next, done):
        """Q‑update."""
        q_sa = self.Q[s][a]
        q_next_max = 0.0 if done else np.max(self.Q[s_next])
        self.Q[s][a] += self.alpha * (r + self.gamma * q_next_max - q_sa)


# --------------------------------------------------------------------------- #
# Trading environment (long / flat)                                          #
# --------------------------------------------------------------------------- #

class TradingEnv:
    """Very light environment that simulates long/flat trading without shorting."""

    def __init__(self, df: pd.DataFrame, agent: QLearningAgent):
        self.df = df.reset_index(drop=True)
        self.agent = agent
        self.reset()

    # ----------------------------- helpers --------------------------------- #

    def reset(self):
        self.step_ptr = 0
        self.position = 0  # 0 = flat, 1 = long
        self.entry_price = 0.0
        first_state = self.agent.state_from_row(self.df.iloc[self.step_ptr])
        return first_state

    def _price(self, idx: int) -> float:
        return float(self.df.loc[idx, "close"])

    def step(self, action: int):
        """Advance one timestep applying *action* and returning reward."""
        done = False
        reward = 0.0

        price_t = self._price(self.step_ptr)
        price_tp1 = self._price(self.step_ptr + 1)  # safe because we stop early

        # Execute portfolio logic
        if action == BUY and self.position == 0:
            self.position = 1
            self.entry_price = price_t
            reward -= price_t * TRANSACTION_FEE
        elif action == SELL and self.position == 1:
            # realise P/L
            pnl = (price_t - self.entry_price)
            reward += pnl - price_t * TRANSACTION_FEE
            self.position = 0
            self.entry_price = 0.0
        # HOLD or invalid actions just keep pos.

        # Carrying position → mark‑to‑market reward
        if self.position == 1:
            unrealised = price_tp1 - price_t
            reward += unrealised

        # Advance pointer
        self.step_ptr += 1
        if self.step_ptr >= len(self.df) - 1:
            done = True

        next_state = self.agent.state_from_row(self.df.iloc[self.step_ptr])
        return next_state, reward, done

# --------------------------------------------------------------------------- #
# Core training / inference pipeline                                         #
# --------------------------------------------------------------------------- #

def _train_agent(hist_df: pd.DataFrame,
                 episodes: int = 50,
                 logger: logging.Logger = None) -> QLearningAgent:
    agent = QLearningAgent()
    env = TradingEnv(hist_df, agent)

    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state

    if logger:
        logger.debug("Training complete. Q‑table size: %d", len(agent.Q))
    return agent


def _decide_action(agent: QLearningAgent, latest_row: pd.Series) -> int:
    state = agent.state_from_row(latest_row)
    # Greedy (exploit) at inference
    return int(np.argmax(agent.Q[state]))


def _prepare_training_set(full_df: pd.DataFrame,
                          cutoff_ts: pd.Timestamp | None) -> pd.DataFrame:
    """Return df up to *cutoff_ts* (inclusive). None = full history."""
    if cutoff_ts is None:
        return full_df.copy()
    mask = full_df["timestamp"] <= cutoff_ts
    return full_df.loc[mask].copy()


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #

def run_logic(current_price: float,
              predicted_price: float,
              ticker: str):
    """Live‑trading entry point. Trains on the full CSV then sends orders."""
    from forest import api, buy_shares, sell_shares  # delayed import for testing

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # --------------------------------------------------------------------- #
    # 1. Train agent on *all available* historical data                     #
    # --------------------------------------------------------------------- #
    df_full = load_price_data(ticker)
    agent = _train_agent(df_full)

    # --------------------------------------------------------------------- #
    # 2. Choose an action for the latest row                                #
    # --------------------------------------------------------------------- #
    latest_row = df_full.iloc[-1].copy()
    latest_row["close"] = current_price
    latest_row["predicted_close"] = predicted_price

    action = _decide_action(agent, latest_row)
    logger.info("[%s] RL decided action: %s", ticker, ACTION_TO_STR[action])

    # --------------------------------------------------------------------- #
    # 3. Execute order via forest API                                       #
    # --------------------------------------------------------------------- #
    try:
        account = api.get_account()
        cash = float(account.cash)
    except Exception as e:
        logger.error("[%s] Cannot fetch account: %s", ticker, e)
        return

    # fetch current position
    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception:
        position_qty = 0.0

    if action == BUY and position_qty == 0:
        max_shares = int(cash // current_price)
        if max_shares > 0:
            logger.info("[%s] Buying %d shares @ %.2f", ticker, max_shares, current_price)
            buy_shares(ticker, max_shares, current_price, predicted_price)
    elif action == SELL and position_qty > 0:
        logger.info("[%s] Selling %d shares @ %.2f", ticker, position_qty, current_price)
        sell_shares(ticker, position_qty, current_price, predicted_price)
    else:
        logger.info("[%s] No trade taken", ticker)


def run_backtest(current_price: float,
                 predicted_price: float,
                 position_qty: float,
                 current_timestamp: str | pd.Timestamp,
                 candles: pd.DataFrame) -> str:
    """Backtest entry point called once per candle.

    Parameters
    ----------
    current_price : float
        Price of *current* candle (index i)
    predicted_price : float
        Forecast for candle i+1
    position_qty : float
        Existing position size (>0 if long)
    current_timestamp : str | pd.Timestamp
        Timestamp of current candle
    candles : pd.DataFrame
        All candles under test (does NOT include *future* data)
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)

    if isinstance(current_timestamp, str):
        current_timestamp = pd.to_datetime(current_timestamp, utc=True)

    # --------------------------------------------------------------------- #
    # 1. Load complete CSV & cut off at current_timestamp                   #
    # --------------------------------------------------------------------- #
    ticker = TICKERS[0]
    df_full = load_price_data(ticker)
    train_df = _prepare_training_set(df_full, current_timestamp)

    # Edge case: need at least two rows to train / simulate
    if len(train_df) < 10:
        return "NONE"

    # --------------------------------------------------------------------- #
    # 2. (Re)train agent and get action                                     #
    # --------------------------------------------------------------------- #
    agent = _train_agent(train_df)

    # Form synthetic last row using the live values passed‑in
    latest_row = train_df.iloc[-1].copy()
    latest_row["close"] = current_price
    latest_row["predicted_close"] = predicted_price
    action = _decide_action(agent, latest_row)

    # --------------------------------------------------------------------- #
    # 3. Enforce position rules (no shorting)                               #
    # --------------------------------------------------------------------- #
    if action == BUY and position_qty == 0:
        return "BUY"
    if action == SELL and position_qty > 0:
        return "SELL"
    return "NONE"
