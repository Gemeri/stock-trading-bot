from __future__ import annotations

import math
import random
from collections import deque
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
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]
    if not cols:
        raise ValueError("No numeric feature columns found. Provide numeric columns in df.")
    return cols


def _unique_existing_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for c in cols:
        if c in df.columns and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _safe_fillna(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    cols_u = _unique_existing_cols(out, cols)
    if not cols_u:
        return out

    block = out.loc[:, cols_u]
    block = block.replace([np.inf, -np.inf], np.nan)
    block = block.ffill().bfill().fillna(0.0)

    out.loc[:, cols_u] = block
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
    """
    start = idx - window + 1
    if start < 0:
        pad_count = -start
        pad = np.repeat(df_feat[[0]], pad_count, axis=0)
        chunk = np.concatenate([pad, df_feat[0:idx + 1]], axis=0)
    else:
        chunk = df_feat[start:idx + 1]

    if chunk.shape[0] != window:
        if chunk.shape[0] < window:
            pad = np.repeat(chunk[[0]], window - chunk.shape[0], axis=0)
            chunk = np.concatenate([pad, chunk], axis=0)
        else:
            chunk = chunk[-window:]

    chunk = (chunk - mean) / std

    wait_fraction = np.float32(wait_steps / max(horizon, 1))
    state = np.concatenate(
        [chunk.reshape(-1).astype(np.float32), np.array([wait_fraction], dtype=np.float32)],
        axis=0
    )
    return state


# =========================
# Triple-barrier helpers (same math as your triple_barrier.py)
# =========================

def _tb_build_future_extreme_returns(prices: np.ndarray, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(prices)
    max_ret = np.full(n, np.nan, dtype=float)
    min_ret = np.full(n, np.nan, dtype=float)

    last_real = n - 2  # n-1 is "filled" last exec price
    last_i = last_real - horizon
    if last_i < 0:
        return max_ret, min_ret

    for i in range(last_i + 1):
        entry = float(prices[i])
        fw = prices[i + 1: i + horizon + 1]
        rets = fw / max(entry, 1e-12) - 1.0
        max_ret[i] = float(np.max(rets))
        min_ret[i] = float(np.min(rets))

    return max_ret, min_ret


def _tb_rolling_quantile_past_only(arr: np.ndarray, window: int, q: float) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)

    for i in range(n):
        start = max(0, i - window)
        past = arr[start:i]
        past = past[np.isfinite(past)]
        if past.size < max(30, int(0.25 * min(window, i))):
            continue
        out[i] = float(np.quantile(past, q))
    return out


def _tb_precompute_dynamic_barriers(
    prices: np.ndarray,
    *,
    horizon: int,
    q_window: int,
    q_tp: float,
    q_sl: float,
    min_tp: float,
    max_tp: float,
    min_sl: float,
    max_sl: float,
) -> Tuple[np.ndarray, np.ndarray]:
    max_ret, min_ret = _tb_build_future_extreme_returns(prices, horizon=horizon)

    tp_dyn = _tb_rolling_quantile_past_only(max_ret, window=q_window, q=q_tp)
    sl_dyn_raw = _tb_rolling_quantile_past_only(min_ret, window=q_window, q=q_sl)
    sl_dyn = np.abs(sl_dyn_raw)

    tp_dyn = np.clip(tp_dyn, min_tp, max_tp)
    sl_dyn = np.clip(sl_dyn, min_sl, max_sl)

    return tp_dyn.astype(np.float32), sl_dyn.astype(np.float32)


def _tb_precompute_scores(
    prices: np.ndarray,
    *,
    direction: Direction,
    horizon: int,
    tp_dyn: np.ndarray,
    sl_dyn: np.ndarray,
    no_trigger_sl_frac: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each i, compute:
      label[i] = 1 if TP hit first else 0 (same as triple_barrier.py)
      score[i] = +tp_dyn[i] if TP hit first
                 -sl_dyn[i] if SL hit first
                 -(no_trigger_sl_frac * sl_dyn[i]) if neither hit within horizon
    """
    n = len(prices)
    last_real = n - 2

    label = np.zeros(n, dtype=np.int8)
    score = np.full(n, np.nan, dtype=np.float32)

    for i in range(n):
        if i + horizon > last_real:
            break

        tp = float(tp_dyn[i])
        sl = float(sl_dyn[i])
        if not (np.isfinite(tp) and np.isfinite(sl)):
            continue

        entry = float(prices[i])

        if direction == "BUY":
            upper = entry * (1.0 + tp)
            lower = entry * (1.0 - sl)

            hit = 0  # 1=tp, -1=sl, 0=none
            for j in range(1, horizon + 1):
                fp = float(prices[i + j])
                if fp >= upper:
                    hit = 1
                    break
                if fp <= lower:
                    hit = -1
                    break

            if hit == 1:
                label[i] = 1
                score[i] = np.float32(tp)
            elif hit == -1:
                label[i] = 0
                score[i] = np.float32(-sl)
            else:
                label[i] = 0
                score[i] = np.float32(-(no_trigger_sl_frac * sl))

        else:  # SELL
            upper = entry * (1.0 + sl)  # risk
            lower = entry * (1.0 - tp)  # profit

            hit = 0
            for j in range(1, horizon + 1):
                fp = float(prices[i + j])
                if fp <= lower:
                    hit = 1
                    break
                if fp >= upper:
                    hit = -1
                    break

            if hit == 1:
                label[i] = 1
                score[i] = np.float32(tp)
            elif hit == -1:
                label[i] = 0
                score[i] = np.float32(-sl)
            else:
                label[i] = 0
                score[i] = np.float32(-(no_trigger_sl_frac * sl))

    return label, score


def _best_net_in_window(
    score: np.ndarray,
    *,
    t0: int,
    horizon_wait: int,
    per_step_wait_cost: float,
) -> Tuple[int, float]:
    """
    best_net = max_k score[k] - (k-t0)*per_step_wait_cost over k in [t0..t0+horizon_wait]
    returns (best_t, best_net)
    """
    k_end = t0 + horizon_wait
    cand = score[t0:k_end + 1].astype(np.float32)

    steps = np.arange(0, cand.shape[0], dtype=np.float32)
    net = cand - steps * np.float32(per_step_wait_cost)

    net = np.where(np.isfinite(net), net, -np.inf)

    best_off = int(np.argmax(net))
    best_t = t0 + best_off
    best_net = float(net[best_off])
    return best_t, best_net


# =========================
# Replay Buffer (now supports n-step)
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
        self.n_steps = np.ones((capacity,), dtype=np.int16)

        self.pos = 0
        self.size = 0

    def add(self, s: np.ndarray, a: int, r: float, ns: np.ndarray, done: bool, n_steps: int = 1) -> None:
        self.s[self.pos] = s
        self.a[self.pos] = a
        self.r[self.pos] = r
        self.ns[self.pos] = ns
        self.done[self.pos] = 1.0 if done else 0.0
        self.n_steps[self.pos] = int(max(1, n_steps))

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "s": torch.tensor(self.s[idx], device=self.device),
            "a": torch.tensor(self.a[idx], device=self.device).unsqueeze(1),
            "r": torch.tensor(self.r[idx], device=self.device).unsqueeze(1),
            "ns": torch.tensor(self.ns[idx], device=self.device),
            "done": torch.tensor(self.done[idx], device=self.device).unsqueeze(1),
            "n": torch.tensor(self.n_steps[idx], device=self.device).unsqueeze(1),
        }


# =========================
# Q-Network (DUELING DQN)
# =========================

class QNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 2, hidden: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden, 1)
        self.adv = nn.Linear(hidden, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        v = self.value(h)  # (B,1)
        a = self.adv(h)    # (B,A)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q


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

    tb_horizon: int
    tb_q_window: int
    tb_q_tp: float
    tb_q_sl: float
    tb_min_tp: float
    tb_max_tp: float
    tb_min_sl: float
    tb_max_sl: float

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
    exec_price_col: str = "open",
    exclude_cols: Optional[List[str]] = None,
    seed: int = 42,
    # training:
    train_frac: float = 0.8,
    total_env_steps: int = 80_000,
    gamma: float = 0.98,
    lr: float = 1e-3,
    batch_size: int = 256,
    replay_capacity: int = 200_000,
    warmup_steps: int = 5_000,
    target_update_every: int = 1_000,
    grad_clip: float = 1.0,
    # exploration / action imbalance control:
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_steps: int = 50_000,
    explore_execute_prob: float = 0.35,  # random action chooses EXECUTE w.p. this (else WAIT)
    # n-step:
    n_step: int = 3,
    # costs / shaping:
    cost: float = 0.0005,
    wait_penalty: float = 0.00005,
    time_penalty: float = 0.0001,
    wait_shaping_reward: float = -0.00002,
    miss_best_penalty: float = 0.002,
    # execute reward shaping (symmetric + regret):
    execute_sign_weight: float = 0.50,  # symmetric (+1/-1) component weight
    regret_weight: float = 1.00,        # regret-vs-best component weight
    # triple barrier params:
    tb_horizon: Optional[int] = None,
    tb_q_window: int = 32,
    tb_q_tp: float = 0.70,
    tb_q_sl: float = 0.20,
    tb_min_tp: float = 0.005,
    tb_max_tp: float = 0.08,
    tb_min_sl: float = 0.003,
    tb_max_sl: float = 0.05,
    no_trigger_sl_frac: float = 0.25,
) -> TimingModel:
    """
    Full upgrade:
    1) Symmetric execute reward (+1/-1) component (weighted) + regret-vs-best-in-window
    2) Dueling DQN network
    3) n-step returns (default 3)
    4) Action imbalance control via explore_execute_prob
    """
    set_seed(seed)

    if exec_price_col not in df.columns:
        raise ValueError(f"df must contain exec_price_col='{exec_price_col}'.")

    tb_h = int(horizon if tb_horizon is None else tb_horizon)
    if tb_h < 1:
        raise ValueError("tb_horizon must be >= 1")
    if tb_q_window < 30:
        raise ValueError("tb_q_window should be >= 30")
    if n_step < 1:
        raise ValueError("n_step must be >= 1")
    if not (0.0 <= explore_execute_prob <= 1.0):
        raise ValueError("explore_execute_prob must be in [0,1]")

    exclude_cols = exclude_cols or []
    for c in ["timestamp", "date", "datetime"]:
        if c in df.columns and c not in exclude_cols:
            exclude_cols.append(c)

    feature_cols = _select_feature_columns(df, exclude=exclude_cols)
    df = _safe_fillna(df, feature_cols + [exec_price_col])

    # Execution price series: next candle price (decision at t, execute at t+1)
    exec_prices = df[exec_price_col].shift(-1).to_numpy(dtype=np.float32)
    exec_prices[-1] = np.float32(float(df[exec_price_col].iloc[-1]))

    n = len(df)
    if n < (window + horizon + tb_h + 20):
        raise ValueError(
            "df too small. Need ~window + horizon(wait) + tb_horizon(trade) + 20 rows. "
            f"Got {n}."
        )

    split = int(n * train_frac)
    train_df = df.iloc[:split]
    mean, std = _compute_scaler_stats(train_df, feature_cols)

    df_feat = df[feature_cols].to_numpy(dtype=np.float32)
    feat_dim = df_feat.shape[1]
    state_dim = window * feat_dim + 1

    # Precompute dynamic barriers + per-index TB score
    tp_dyn, sl_dyn = _tb_precompute_dynamic_barriers(
        prices=exec_prices.astype(np.float64),
        horizon=tb_h,
        q_window=tb_q_window,
        q_tp=tb_q_tp,
        q_sl=tb_q_sl,
        min_tp=tb_min_tp,
        max_tp=tb_max_tp,
        min_sl=tb_min_sl,
        max_sl=tb_max_sl,
    )

    tb_label, tb_score = _tb_precompute_scores(
        prices=exec_prices.astype(np.float64),
        direction=direction,
        horizon=tb_h,
        tp_dyn=tp_dyn,
        sl_dyn=sl_dyn,
        no_trigger_sl_frac=no_trigger_sl_frac,
    )
    _ = tb_label  # optional debug use

    per_step_wait_cost = float(wait_penalty + time_penalty)
    denom = float(max(tb_max_tp, tb_max_sl, 1e-6))  # normalize regret to ~[-1,0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = QNet(state_dim=state_dim, action_dim=2).to(device)
    target_q = QNet(state_dim=state_dim, action_dim=2).to(device)
    target_q.load_state_dict(q.state_dict())
    target_q.eval()

    optimizer = optim.Adam(q.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    buffer = ReplayBuffer(replay_capacity, state_dim=state_dim, device=device)

    # Bounds: need window history, and ensure wait horizon + TB horizon fits before n-2
    min_start = window - 1
    max_start = (n - 2) - horizon - tb_h
    if max_start <= min_start:
        raise ValueError("Not enough data for chosen window/horizon/tb_horizon.")

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
        nsteps = batch["n"]  # (B,1) int

        q_vals = q(s).gather(1, a)

        with torch.no_grad():
            # Double DQN
            next_actions = torch.argmax(q(ns), dim=1, keepdim=True)
            next_q_vals = target_q(ns).gather(1, next_actions)

            # gamma ** n
            n_f = nsteps.to(dtype=torch.float32)
            gamma_pow = torch.exp(n_f * math.log(max(gamma, 1e-12)))

            target = r + gamma_pow * (1.0 - done) * next_q_vals

        loss = loss_fn(q_vals, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(q.parameters(), grad_clip)
        optimizer.step()

        return float(loss.item())

    # n-step helper buffer (per episode)
    nbuf: deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque()

    def nstep_emit_if_ready(done_just_happened: bool = False) -> None:
        """
        If enough steps accumulated OR episode ended, emit one or more n-step transitions.
        """
        nonlocal nbuf

        def _emit_one() -> bool:
            if not nbuf:
                return False

            R = 0.0
            used = 0
            ns_last = nbuf[0][3]
            done_last = nbuf[0][4]

            for i, (_, __, r_i, ns_i, d_i) in enumerate(nbuf):
                R += (gamma ** i) * float(r_i)
                used = i + 1
                ns_last = ns_i
                done_last = d_i
                if d_i or used >= n_step:
                    break

            s0, a0, *_ = nbuf[0]
            buffer.add(s0, int(a0), float(R), ns_last, bool(done_last), n_steps=int(used))
            nbuf.popleft()
            return True

        # If we have n_step steps, we can emit immediately
        while len(nbuf) >= n_step:
            _emit_one()

        # If episode ended, flush remaining
        if done_just_happened:
            while nbuf:
                _emit_one()

    env_step = 0
    update_step = 0

    while env_step < total_env_steps:
        # sample a usable episode start (must have at least one finite score in the window)
        for _tries in range(50):
            t0 = int(np.random.randint(min_start, max_start + 1))
            best_t, best_net = _best_net_in_window(
                tb_score, t0=t0, horizon_wait=horizon, per_step_wait_cost=per_step_wait_cost
            )
            if np.isfinite(best_net):
                break
        else:
            break

        t = t0
        wait_steps = 0
        nbuf.clear()

        for _ in range(horizon + 1):
            s = _build_state(
                df_feat, idx=t, window=window, feat_dim=feat_dim,
                mean=mean, std=std, wait_steps=wait_steps, horizon=horizon
            )

            forced = (wait_steps >= horizon)

            if forced:
                action = 1
            else:
                eps = epsilon_by_step(env_step)
                if np.random.rand() < eps:
                    # action imbalance control: pick EXECUTE with explore_execute_prob
                    action = 1 if (np.random.rand() < explore_execute_prob) else 0
                else:
                    with torch.no_grad():
                        qs = q(torch.tensor(s, device=device).unsqueeze(0))
                        action = int(torch.argmax(qs, dim=1).item())

            if action == 0 and not forced:
                # WAIT reward
                r = float(wait_shaping_reward)
                if t == best_t:
                    r -= float(miss_best_penalty)

                t_next = t + 1
                wait_next = wait_steps + 1

                ns = _build_state(
                    df_feat, idx=t_next, window=window, feat_dim=feat_dim,
                    mean=mean, std=std, wait_steps=wait_next, horizon=horizon
                )

                nbuf.append((s, 0, r, ns, False))
                nstep_emit_if_ready(done_just_happened=False)

                t = t_next
                wait_steps = wait_next
                env_step += 1

            else:
                # EXECUTE reward:
                # - symmetric component: +1 if TB score positive else -1
                # - plus regret-vs-best (normalized) to enforce timing
                base = float(tb_score[t]) if np.isfinite(tb_score[t]) else float(-0.5 * tb_max_sl)
                sign = 1.0 if base > 0.0 else -1.0

                net = base - float(wait_steps) * per_step_wait_cost
                regret_raw = net - float(best_net)          # <= 0
                regret_norm = float(regret_raw / denom)     # roughly in [-1,0]

                r = (execute_sign_weight * sign) + (regret_weight * regret_norm)
                r -= float(cost)

                # terminal transition
                ns = s.copy()
                nbuf.append((s, 1, float(r), ns, True))
                nstep_emit_if_ready(done_just_happened=True)

                env_step += 1
                break

            if buffer.size >= warmup_steps and (env_step % 4 == 0):
                _ = dqn_update()
                update_step += 1
                if target_update_every and (update_step % target_update_every == 0):
                    target_q.load_state_dict(q.state_dict())

            if env_step >= total_env_steps:
                break

        if buffer.size >= warmup_steps:
            _ = dqn_update()
            update_step += 1
            if target_update_every and (update_step % target_update_every == 0):
                target_q.load_state_dict(q.state_dict())

    return TimingModel(
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
        tb_horizon=tb_h,
        tb_q_window=tb_q_window,
        tb_q_tp=tb_q_tp,
        tb_q_sl=tb_q_sl,
        tb_min_tp=tb_min_tp,
        tb_max_tp=tb_max_tp,
        tb_min_sl=tb_min_sl,
        tb_max_sl=tb_max_sl,
    )


def predict(
    model: TimingModel,
    df: pd.DataFrame,
    *,
    wait_steps: Optional[int] = None,
) -> ActionStr:
    if wait_steps is None:
        wait_steps = model.wait_steps_runtime

    missing = [c for c in model.feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"df missing feature columns: {missing}")

    cols_to_fill = list(model.feature_cols)
    if model.exec_price_col in df.columns and model.exec_price_col not in cols_to_fill:
        cols_to_fill.append(model.exec_price_col)

    df = _safe_fillna(df, cols_to_fill)

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


class TimingRunner:
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
