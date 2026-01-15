from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim


Direction = Literal["BUY", "SELL"]
ActionStr = Literal["WAIT", "EXECUTE"]


# =========================
# Utilities
# =========================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _select_feature_columns(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    exclude = set(exclude or [])
    # Keep numeric columns only
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]
    if not cols:
        raise ValueError("No numeric feature columns found. Provide numeric columns in df.")
    return cols


def _safe_fillna(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    out[cols] = out[cols].replace([np.inf, -np.inf], np.nan)
    out[cols] = out[cols].fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    return out


def _compute_scaler_stats(train_df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    mean = train_df[feature_cols].mean(axis=0).to_numpy(dtype=np.float32)
    std = train_df[feature_cols].std(axis=0).to_numpy(dtype=np.float32)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def _build_state(
    df_feat: np.ndarray,
    idx: int,
    window: int,
    feat_dim: int,
    mean: np.ndarray,
    std: np.ndarray,
    wait_steps: int,
    horizon: int,
) -> np.ndarray:
    """
    State = [normalized features over last `window` rows flattened] + [wait_fraction]
    - idx is the current candle index (inclusive)
    """
    start = idx - window + 1
    if start < 0:
        # pad by repeating first row
        pad_count = -start
        pad = np.repeat(df_feat[[0]], pad_count, axis=0)
        chunk = np.concatenate([pad, df_feat[0:idx + 1]], axis=0)
    else:
        chunk = df_feat[start:idx + 1]

    # chunk shape: (window, feat_dim)
    if chunk.shape[0] != window:
        # If something went wrong, pad/truncate
        if chunk.shape[0] < window:
            pad = np.repeat(chunk[[0]], window - chunk.shape[0], axis=0)
            chunk = np.concatenate([pad, chunk], axis=0)
        else:
            chunk = chunk[-window:]

    # Normalize
    chunk = (chunk - mean) / std

    wait_fraction = np.float32(wait_steps / max(horizon, 1))
    state = np.concatenate([chunk.reshape(-1).astype(np.float32), np.array([wait_fraction], dtype=np.float32)], axis=0)
    return state


def _best_fill_terminal_reward(
    exec_prices: np.ndarray,
    t0: int,
    te: int,
    horizon: int,
    direction: Direction,
    cost: float,
    wait_steps: int,
    wait_penalty: float,
    time_penalty: float,
) -> float:
    """
    Terminal reward at EXECUTE: negative percent-regret vs best price in [t0, t0+horizon], minus costs and penalties.
    - BUY: best is minimum price in window
    - SELL: best is maximum price in window
    """
    end = min(t0 + horizon, len(exec_prices) - 1)
    window = exec_prices[t0:end + 1]

    exec_price = float(exec_prices[te])

    if direction == "BUY":
        best_price = float(np.min(window))
        regret = exec_price - best_price
    else:
        best_price = float(np.max(window))
        regret = best_price - exec_price

    # Percent regret to keep scaling stable across price levels
    percent_regret = regret / max(best_price, 1e-12)

    reward = -percent_regret
    reward -= cost
    reward -= wait_steps * wait_penalty
    reward -= wait_steps * time_penalty
    return float(reward)


# =========================
# Replay Buffer
# =========================

class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.state_dim = state_dim

        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros((capacity,), dtype=np.int64)
        self.r = np.zeros((capacity,), dtype=np.float32)
        self.ns = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)

        self.pos = 0
        self.size = 0

    def add(self, s: np.ndarray, a: int, r: float, ns: np.ndarray, done: bool) -> None:
        self.s[self.pos] = s
        self.a[self.pos] = a
        self.r[self.pos] = r
        self.ns[self.pos] = ns
        self.done[self.pos] = 1.0 if done else 0.0

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = {
            "s": torch.tensor(self.s[idx], device=self.device),
            "a": torch.tensor(self.a[idx], device=self.device).unsqueeze(1),
            "r": torch.tensor(self.r[idx], device=self.device).unsqueeze(1),
            "ns": torch.tensor(self.ns[idx], device=self.device),
            "done": torch.tensor(self.done[idx], device=self.device).unsqueeze(1),
        }
        return batch


# =========================
# Q-Network
# =========================

class QNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 2, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =========================
# Model container
# =========================

@dataclass
class TimingModel:
    q: QNet
    target_q: QNet
    mean: np.ndarray
    std: np.ndarray
    feature_cols: List[str]
    window: int
    horizon: int
    direction: Direction
    exec_price_col: str
    device: torch.device

    # runtime helper (optional stateful usage)
    wait_steps_runtime: int = 0


# =========================
# FIT / PREDICT
# =========================

def fit(
    df: pd.DataFrame,
    direction: Direction = "BUY",
    *,
    window: int = 64,
    horizon: int = 9,
    exec_price_col: str = "open",  # will use NEXT candle open by default for execution price
    exclude_cols: Optional[List[str]] = None,
    seed: int = 42,
    # training:
    train_frac: float = 0.8,
    total_env_steps: int = 60_000,
    gamma: float = 0.98,
    lr: float = 1e-3,
    batch_size: int = 256,
    replay_capacity: int = 200_000,
    warmup_steps: int = 5_000,
    target_update_every: int = 1_000,
    grad_clip: float = 1.0,
    # exploration:
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_steps: int = 40_000,
    # reward shaping / costs:
    cost: float = 0.0005,
    wait_penalty: float = 0.00005,
    time_penalty: float = 0.0001,
    wait_shaping_reward: float = -0.00005,
) -> TimingModel:
    """
    Train a DQN-style agent for timing within `horizon` candles using "best fill within next horizon" reward.
    Returns a TimingModel you can pass into `predict(...)`.

    df requirements:
    - numeric feature columns (OHLCV/indicators/etc)
    - must contain `exec_price_col` (default: "open") if you want next-open execution price
    """
    set_seed(seed)

    if exec_price_col not in df.columns:
        raise ValueError(f"df must contain exec_price_col='{exec_price_col}'. Columns: {list(df.columns)[:20]}...")

    exclude_cols = exclude_cols or []
    # Often you want to exclude timestamp-like columns from features
    for c in ["timestamp", "date", "datetime"]:
        if c in df.columns and c not in exclude_cols:
            exclude_cols.append(c)

    feature_cols = _select_feature_columns(df, exclude=exclude_cols)
    df = _safe_fillna(df, feature_cols + [exec_price_col])

    # Execution price series: next candle open (decision at t, execute at t+1 open)
    # If you prefer "execute at close", set exec_price_col="close" AND change shift to 0 below.
    exec_prices = df[exec_price_col].shift(-1).to_numpy(dtype=np.float32)
    # last row has no next open; fill with its own open (won't be used for training if we bound indices)
    exec_prices[-1] = df[exec_price_col].iloc[-1].astype(np.float32)

    # Time split for scaling (avoid leakage)
    n = len(df)
    if n < (window + horizon + 10):
        raise ValueError(f"df is too small. Need at least ~window+horizon+10 rows. Got {n}.")

    split = int(n * train_frac)
    train_df = df.iloc[:split]
    mean, std = _compute_scaler_stats(train_df, feature_cols)

    # Build feature matrix
    df_feat = df[feature_cols].to_numpy(dtype=np.float32)
    feat_dim = df_feat.shape[1]

    state_dim = window * feat_dim + 1  # + wait_fraction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q = QNet(state_dim=state_dim, action_dim=2).to(device)
    target_q = QNet(state_dim=state_dim, action_dim=2).to(device)
    target_q.load_state_dict(q.state_dict())
    target_q.eval()

    optimizer = optim.Adam(q.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()  # Huber

    buffer = ReplayBuffer(replay_capacity, state_dim=state_dim, device=device)

    # Valid episode start range: must have window history and horizon future (for reward window)
    # Also exec_prices[t] uses next open; ensure t <= n-2 if you want strictly valid. We bound accordingly.
    min_start = window - 1
    max_start = (n - 2) - horizon  # ensure te and window endpoints are valid indices
    if max_start <= min_start:
        raise ValueError("Not enough data for the chosen window/horizon with next-open execution assumption.")

    def epsilon_by_step(step: int) -> float:
        if step >= eps_decay_steps:
            return eps_end
        frac = step / max(eps_decay_steps, 1)
        return eps_start + (eps_end - eps_start) * frac

    def dqn_update() -> float:
        batch = buffer.sample(batch_size)
        s = batch["s"]
        a = batch["a"]
        r = batch["r"]
        ns = batch["ns"]
        done = batch["done"]

        # Q(s,a)
        q_vals = q(s).gather(1, a)

        with torch.no_grad():
            # Double DQN: action from online net, value from target net
            next_actions = torch.argmax(q(ns), dim=1, keepdim=True)
            next_q_vals = target_q(ns).gather(1, next_actions)
            target = r + gamma * (1.0 - done) * next_q_vals

        loss = loss_fn(q_vals, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(q.parameters(), grad_clip)
        optimizer.step()

        return float(loss.item())

    # ---- training loop (offline episodes sampled from history)
    env_step = 0
    update_step = 0
    last_loss = 0.0

    while env_step < total_env_steps:
        # Sample an episode start (any candle can be a start)
        t0 = np.random.randint(min_start, max_start + 1)
        t = t0
        wait_steps = 0

        # Run up to horizon steps; episode ends on EXECUTE or forced at horizon
        for _ in range(horizon + 1):
            s = _build_state(df_feat, idx=t, window=window, feat_dim=feat_dim,
                             mean=mean, std=std, wait_steps=wait_steps, horizon=horizon)

            # Forced execute at horizon
            forced = (wait_steps >= horizon)

            if forced:
                action = 1  # EXECUTE
            else:
                eps = epsilon_by_step(env_step)
                if np.random.rand() < eps:
                    action = np.random.randint(0, 2)  # 0=WAIT, 1=EXECUTE
                else:
                    with torch.no_grad():
                        qs = q(torch.tensor(s, device=device).unsqueeze(0))
                        action = int(torch.argmax(qs, dim=1).item())

            if action == 0 and not forced:
                # WAIT
                r = float(wait_shaping_reward)
                t_next = t + 1
                wait_next = wait_steps + 1
                done = False

                ns = _build_state(df_feat, idx=t_next, window=window, feat_dim=feat_dim,
                                  mean=mean, std=std, wait_steps=wait_next, horizon=horizon)

                buffer.add(s, action, r, ns, done)

                t = t_next
                wait_steps = wait_next
                env_step += 1

            else:
                # EXECUTE (or forced EXECUTE)
                r = _best_fill_terminal_reward(
                    exec_prices=exec_prices,
                    t0=t0,
                    te=t,
                    horizon=horizon,
                    direction=direction,
                    cost=cost,
                    wait_steps=wait_steps,
                    wait_penalty=wait_penalty,
                    time_penalty=time_penalty,
                )
                done = True

                # Next state doesn't matter for terminal, but store something consistent
                ns = s.copy()
                buffer.add(s, 1, r, ns, done)

                env_step += 1
                break

            # Learn
            if buffer.size >= warmup_steps and (env_step % 4 == 0):
                last_loss = dqn_update()
                update_step += 1

                if target_update_every and (update_step % target_update_every == 0):
                    target_q.load_state_dict(q.state_dict())

            if env_step >= total_env_steps:
                break

        # A few extra updates per episode once warm
        if buffer.size >= warmup_steps:
            for _ in range(1):
                last_loss = dqn_update()
                update_step += 1
                if target_update_every and (update_step % target_update_every == 0):
                    target_q.load_state_dict(q.state_dict())

    model = TimingModel(
        q=q,
        target_q=target_q,
        mean=mean,
        std=std,
        feature_cols=feature_cols,
        window=window,
        horizon=horizon,
        direction=direction,
        exec_price_col=exec_price_col,
        device=device,
    )
    return model


def predict(
    model: TimingModel,
    df: pd.DataFrame,
    *,
    wait_steps: Optional[int] = None,
) -> ActionStr:
    """
    Stateless prediction:
    - df should contain at least model.window rows (more is fine). Uses the latest row as "current candle".
    - wait_steps is how many candles you've already waited since the timing window started (0..horizon).
      If not provided, uses model.wait_steps_runtime (can be used statefully).
    Returns: "WAIT" or "EXECUTE"
    """
    if wait_steps is None:
        wait_steps = model.wait_steps_runtime

    # Ensure feature columns exist
    missing = [c for c in model.feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"df missing feature columns: {missing}")

    df = _safe_fillna(df, model.feature_cols + ([model.exec_price_col] if model.exec_price_col in df.columns else []))

    df_feat = df[model.feature_cols].to_numpy(dtype=np.float32)
    feat_dim = df_feat.shape[1]

    idx = len(df) - 1
    if idx < 0:
        raise ValueError("df is empty.")

    state = _build_state(
        df_feat=df_feat,
        idx=idx,
        window=model.window,
        feat_dim=feat_dim,
        mean=model.mean,
        std=model.std,
        wait_steps=int(wait_steps),
        horizon=model.horizon,
    )

    with torch.no_grad():
        qs = model.q(torch.tensor(state, device=model.device).unsqueeze(0))
        action_int = int(torch.argmax(qs, dim=1).item())

    return "WAIT" if action_int == 0 else "EXECUTE"


# =========================
# Optional: stateful runner (tracks wait_steps for you)
# =========================

class TimingRunner:
    """
    Keeps track of wait_steps across candles.
    You should call reset() when a new BUY/SELL intent starts (or after executing).
    """
    def __init__(self, model: TimingModel):
        self.model = model
        self.wait_steps = 0

    def reset(self) -> None:
        self.wait_steps = 0

    def step(self, df_latest: pd.DataFrame) -> ActionStr:
        action = predict(self.model, df_latest, wait_steps=self.wait_steps)
        if action == "WAIT":
            self.wait_steps = min(self.wait_steps + 1, self.model.horizon)
        else:
            self.wait_steps = 0
        return action