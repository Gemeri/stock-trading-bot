"""
Reinforcement‑Learning Trading Logic  ✅ syntax‑checked
────────────────────────────────────────────────────────────────────────────
Exports
    • run_logic(...)
    • run_backtest(...)
────────────────────────────────────────────────────────────────────────────
Key improvements
    ✔ Offline pre‑training (executed only once)
    ✔ Position‑aware state vector
    ✔ Reward shaping on next‑bar returns
    ✔ Double‑DQN with soft target update
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv

# ╭─────────────────────────── ENV / CONSTANTS ───────────────────────────╮
load_dotenv()

BAR_TIMEFRAME: str = os.getenv("BAR_TIMEFRAME", "1Hour")
TICKERS: List[str] = os.getenv("TICKERS", "TSLA").split(",")  # spec: exactly one

TIMEFRAME_MAP = {
    "4Hour": "H4",
    "2Hour": "H2",
    "1Hour": "H1",
    "30Min": "M30",
    "15Min": "M15",
}
CONVERTED_TF: str = TIMEFRAME_MAP.get(BAR_TIMEFRAME, BAR_TIMEFRAME)

MODEL_DIR = Path(__file__).with_suffix("") / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

ACTION_NONE, ACTION_BUY, ACTION_SELL = 0, 1, 2
ACTIONS = {ACTION_NONE: "NONE", ACTION_BUY: "BUY", ACTION_SELL: "SELL"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(7)
np.random.seed(7)
torch.manual_seed(7)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ╭────────────────────────────── HELPERS ────────────────────────────────╮
def get_csv_filename(ticker: str) -> str:
    return f"{ticker}_{CONVERTED_TF}.csv"


def _load_full_history(ticker: str) -> pd.DataFrame:
    csv_path = Path(get_csv_filename(ticker))
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV {csv_path} not found")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _state_from_row(
    row: pd.Series,
    current_price: float,
    predicted_price: float,
    position_qty: float,
) -> np.ndarray:
    """Convert a candle row → numeric state with context features."""
    if isinstance(row, pd.DataFrame):
        row = row.squeeze()

    row = row.copy()
    if "predicted_close" in row.index:
        row["predicted_close"] = predicted_price

    row["signal_gap"] = predicted_price - current_price
    row["position_qty"] = position_qty

    numeric = pd.to_numeric(row, errors="coerce").fillna(0.0)
    return numeric.to_numpy(dtype=np.float32)


# ╭──────────────────────── REPLAY & NETWORK ────────────────────────────╮
class Replay:
    def __init__(self, capacity: int = 50_000):
        self.capacity = capacity
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []

    def push(self, transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        idx = random.sample(range(len(self.buffer)), batch_size)
        return [self.buffer[i] for i in idx]

    def __len__(self):
        return len(self.buffer)


class Net(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        return self.layers(x)


class Agent:
    """Double‑DQN agent with soft target update."""

    def __init__(
        self,
        state_dim: int,
        pth: Path,
        bufpth: Path,
        gamma: float = 0.995,
        lr: float = 3e-4,
        batch_size: int = 128,
        eps_hi: float = 0.25,
        eps_lo: float = 0.02,
        eps_decay: float = 0.9995,
    ):
        self.policy = Net(state_dim).to(DEVICE)
        self.target = Net(state_dim).to(DEVICE)
        self.target.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.gamma = gamma
        self.batch_size = batch_size

        self.eps = eps_hi
        self.eps_lo = eps_lo
        self.eps_decay = eps_decay

        self.pth = pth
        self.bufpth = bufpth
        self.replay = Replay()

        self._load()

    # ─── persistence ────────────────────────────────────────────────
    def _load(self) -> None:
        if self.pth.exists():
            self.policy.load_state_dict(
                torch.load(self.pth, map_location=DEVICE)
            )
            self.target.load_state_dict(self.policy.state_dict())
            logger.info(f"Loaded model from {self.pth}")
        if self.bufpth.exists():
            self.replay = pickle.loads(self.bufpth.read_bytes())
            logger.info(
                f"Loaded replay buffer with {len(self.replay)} transitions"
            )

    def _save(self) -> None:
        torch.save(self.policy.state_dict(), self.pth)
        self.bufpth.write_bytes(pickle.dumps(self.replay))

    # ─── interaction ────────────────────────────────────────────────
    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.eps:
            return random.randint(0, 2)
        with torch.no_grad():
            s = (
                torch.tensor(state, dtype=torch.float32, device=DEVICE)
                .unsqueeze(0)
            )
            q = self.policy(s)
            return int(torch.argmax(q).item())

    def store(self, transition) -> None:
        self.replay.push(transition)

    def learn(self) -> None:
        if len(self.replay) < max(self.batch_size, 2_000):
            return

        batch = self.replay.sample(self.batch_size)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.tensor(np.array(state), dtype=torch.float32, device=DEVICE)
        next_state = torch.tensor(
            np.array(next_state), dtype=torch.float32, device=DEVICE
        )
        action = (
            torch.tensor(action, dtype=torch.long, device=DEVICE)
            .unsqueeze(1)
        )
        reward = (
            torch.tensor(reward, dtype=torch.float32, device=DEVICE)
            .unsqueeze(1)
        )
        done = (
            torch.tensor(done, dtype=torch.float32, device=DEVICE)
            .unsqueeze(1)
        )

        q_values = self.policy(state).gather(1, action)

        next_actions = torch.argmax(self.policy(next_state), 1, keepdim=True)
        next_q = self.target(next_state).gather(1, next_actions)

        target = reward + self.gamma * next_q * (1 - done)

        loss = nn.SmoothL1Loss()(q_values, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # soft‑update target network
        tau = 0.005
        for tp, pp in zip(self.target.parameters(), self.policy.parameters()):
            tp.data.mul_(1 - tau).add_(tau * pp.data)

        # ε‑decay
        if self.eps > self.eps_lo:
            self.eps *= self.eps_decay


# ╭──────────────────────── OFFLINE PRE‑TRAINING ─────────────────────────╮
def _offline_pretrain(agent: Agent, df: pd.DataFrame) -> None:
    if agent.pth.exists():
        return  # already trained

    logger.info("⏳  Running one‑off offline pre‑training …")
    for idx in range(len(df) - 1):
        cur, nxt = df.iloc[idx], df.iloc[idx + 1]
        state = _state_from_row(
            cur, cur.close, cur.predicted_close, position_qty=0.0
        )
        next_state = _state_from_row(
            nxt, nxt.close, nxt.predicted_close, position_qty=0.0
        )
        pct_ret = (nxt.close - cur.close) / cur.close

        if pct_ret > 0:
            act, rew = ACTION_BUY, pct_ret
        elif pct_ret < 0:
            act, rew = ACTION_SELL, -pct_ret
        else:
            act, rew = ACTION_NONE, 0.0

        agent.store((state, act, rew, next_state, False))
        agent.learn()

    agent._save()
    logger.info("✅  Offline pre‑training complete")


def _get_agent(ticker: str, state_dim: int) -> Agent:
    base = MODEL_DIR / f"{ticker}_{CONVERTED_TF}"
    agent = Agent(
        state_dim,
        pth=base.with_suffix(".pth"),
        bufpth=base.with_suffix(".replay"),
    )
    if not base.with_suffix(".pth").exists():
        _offline_pretrain(agent, _load_full_history(ticker))
    return agent


# ╭──────────────────────────── LIVE LOGIC ──────────────────────────────╮
def run_logic(current_price: float, predicted_price: float, ticker: str) -> None:
    """
    Called once per new live candle.
    Executes trading action via forest API.
    """
    from forest import api, buy_shares, sell_shares  # your framework

    df = _load_full_history(ticker)
    last_row = df.iloc[-1]

    # account / position
    account = api.get_account()
    cash = float(account.cash)
    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
        entry_price = float(pos.avg_entry_price)
    except Exception:
        position_qty, entry_price = 0.0, 0.0

    state = _state_from_row(
        last_row, current_price, predicted_price, position_qty
    )
    agent = _get_agent(ticker, state.size)
    action = agent.act(state, training=True)

    reward = 0.0
    done = False
    next_state = state  # placeholder; could store next live state if available

    if action == ACTION_BUY and position_qty == 0:
        shares = int(cash // current_price)
        if shares > 0:
            buy_shares(ticker, shares, current_price, predicted_price)
    elif action == ACTION_SELL and position_qty > 0:
        sell_shares(ticker, position_qty, current_price, predicted_price)
        reward = (current_price - entry_price) / entry_price
        done = True
    else:
        action = ACTION_NONE

    agent.store((state, action, reward, next_state, done))
    agent.learn()
    if random.random() < 0.002:
        agent._save()

    logger.info(
        f"[{ticker}] LIVE {ACTIONS[action]} | price={current_price:.2f} "
        f"cash={cash:.0f} pos={position_qty}"
    )


# ╭─────────────────────────── BACK‑TEST LOGIC ──────────────────────────╮
def run_backtest(
    current_price: float,
    predicted_price: float,
    position_qty: float,
    current_timestamp: datetime,
    candles: pd.DataFrame,
) -> str:
    """
    Called per candle during back‑testing.
    Returns "BUY", "SELL", or "NONE".
    """
    ticker = TICKERS[0]  # single‑ticker guarantee

    df_full = _load_full_history(ticker)
    df_hist = df_full[df_full["timestamp"] <= pd.to_datetime(current_timestamp)]
    if len(df_hist) < 2:
        return "NONE"

    prev_row = df_hist.iloc[-2]
    cur_row = df_hist.iloc[-1]

    prev_state = _state_from_row(
        prev_row,
        prev_row.close,
        prev_row.predicted_close,
        position_qty,
    )
    state = _state_from_row(
        cur_row, current_price, predicted_price, position_qty
    )

    agent = _get_agent(ticker, state.size)

    # Reward: direction‑correct next‑bar return
    pct_ret = (current_price - prev_row.close) / prev_row.close
    last_act = (
        ACTION_BUY
        if pct_ret > 0
        else ACTION_SELL
        if pct_ret < 0
        else ACTION_NONE
    )
    reward = abs(pct_ret)

    agent.store((prev_state, last_act, reward, state, False))
    agent.learn()

    chosen = agent.act(state, training=False)
    if random.random() < 0.01:
        agent._save()

    return ACTIONS[chosen]
