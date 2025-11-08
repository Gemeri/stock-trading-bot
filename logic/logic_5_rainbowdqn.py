import os
import math
import random
import logging
from collections import deque, namedtuple
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset
from tqdm.auto import tqdm

# --------------------------------------------------------------------------
# ------------------------------  CONFIG  ----------------------------------
# --------------------------------------------------------------------------

BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "1Hour")
TIMEFRAME_MAP = {
    "4Hour": "H4", "2Hour": "H2", "1Hour": "H1",
    "30Min": "M30", "15Min": "M15"
}
CONVERTED_TIMEFRAME = TIMEFRAME_MAP.get(BAR_TIMEFRAME, BAR_TIMEFRAME)

def get_csv_filename(ticker):
    return os.path.join("data", f"{ticker}_{CONVERTED_TIMEFRAME}.csv")

FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions', 'sentiment',
    'greed_index', 'news_count', 'news_volume_z', 'price_change', 'high_low_range', 
    'macd_line', 'macd_signal', 'macd_histogram', 'rsi', 'roc', 'atr', 'ema_9', 'ema_21', 
    'ema_50', 'ema_200', 'adx', 'obv', 'bollinger_percB', 'returns_1', 
    'returns_3', 'returns_5', 'std_5', 'std_10', 'lagged_close_1', 'lagged_close_2', 
    'lagged_close_3', 'lagged_close_5', 'lagged_close_10', 'candle_body_ratio', 
    'wick_dominance', 'gap_vs_prev', 'volume_zscore', 'atr_zscore', 'rsi_zscore',
    'month', 'hour_sin', 'hour_cos', 'day_of_week_sin',  'day_of_week_cos', 
    'days_since_high', 'days_since_low', "d_sentiment"
]

ACTIONS = ["HOLD", "BUY", "SELL"]
ACTION_HOLD, ACTION_BUY, ACTION_SELL = 0, 1, 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rainbow hyper-parameters
NUM_ATOMS = 51
V_MIN, V_MAX = -1.0, 1.0
GAMMA = 0.99
N_STEPS = 3
BUFFER_CAPACITY = 40_000
BATCH_SIZE = 256
LEARNING_RATE = 2.5e-4
UPDATE_TARGET_EVERY = 1_000
PRIOR_ALPHA = 0.6
PRIOR_BETA_START = 0.4
PRIOR_BETA_FRAMES = 250_000
TRAINING_FRAMES = 8_000
WARMUP_STEPS = 1_000
INACT_PENALTY = 0.001

# --------------------------------------------------------------------------
# --------------------------  DATA HANDLING  -------------------------------
# --------------------------------------------------------------------------
def load_dataframe(ticker, stop_ts: datetime | None = None) -> pd.DataFrame:
    df = pd.read_csv(get_csv_filename(ticker))
    df["close_raw"] = df["close"]
    df = df[[c for c in FEATURE_COLUMNS if c in df.columns] + ["close_raw"]]
    if stop_ts is not None:
        df = df[pd.to_datetime(df["timestamp"]) <= stop_ts]

    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    for col in df.columns:
        if col in ("timestamp", "close_raw"):
            continue
        col_min, col_max = df[col].min(), df[col].max()
        df[col] = (df[col] - col_min) / (col_max - col_min) if col_max != col_min else 0.0
    return df


# --------------------------------------------------------------------------
# --------------------------  ENVIRONMENT  ---------------------------------
# --------------------------------------------------------------------------
Transition = namedtuple("Transition",
                        ("state", "action", "reward", "next_state", "done"))

class TradingEnv:
    def __init__(self, df: pd.DataFrame, predicted_signal: float = 0.5,
                 starting_cash: float = 1_000.0) -> None:
        self.df = df
        self.predicted_signal = predicted_signal
        self.starting_cash = starting_cash
        self.pointer = 1
        self.shares_held = 0
        self.cash = starting_cash
        self.done = False

    # ------------------------
    def reset(self):
        self.pointer = 1
        self.shares_held = 0
        self.cash = self.starting_cash
        self.done = False
        return self._get_state()

        # ------------------------
    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode is done")

        # use un-scaled price if available
        prev_close = self.df.at[self.pointer - 1, "close_raw"] if "close_raw" in self.df.columns \
                    else self.df.at[self.pointer - 1, "close"]
        curr_close = self.df.at[self.pointer, "close_raw"] if "close_raw" in self.df.columns \
                    else self.df.at[self.pointer, "close"]

        # zero-price guard (can happen after scaling on tiny slices)
        if curr_close <= 1e-8:
            curr_close = 1e-8

        # ---------- trading constraints ----------
        executed_action = action
        if action == ACTION_BUY and self.shares_held > 0:
            executed_action = ACTION_HOLD
        if action == ACTION_SELL and self.shares_held == 0:
            executed_action = ACTION_HOLD

        # ---------- execute trade ----------
        if executed_action == ACTION_BUY:
            max_shares = int(self.cash // curr_close)
            self.shares_held += max_shares
            self.cash -= max_shares * curr_close

        elif executed_action == ACTION_SELL:
            self.cash += self.shares_held * curr_close
            self.shares_held = 0

        # ---------- reward ----------
        if executed_action == ACTION_HOLD:
            reward = -INACT_PENALTY
        else:
            direction = +1 if executed_action == ACTION_BUY else -1
            reward = direction * (curr_close - prev_close)

        # ---------- advance ----------
        self.pointer += 1
        if self.pointer >= len(self.df) - 1:
            self.done = True

        next_state = self._get_state()
        return next_state, reward, self.done, {}

    # ------------------------
    def _get_state(self):
        row = self.df.iloc[self.pointer].copy()
        state = row.drop(labels=["timestamp"]).to_numpy(dtype=np.float32)
        state = np.append(state, np.float32(self.predicted_signal))
        return state

    # ------------------------
    @property
    def state_size(self) -> int:
        return len(self.df.columns) - 1 + 1


# --------------------------------------------------------------------------
# --------------------------  RAINBOW CORE  --------------------------------
# --------------------------------------------------------------------------
def projection_distribution(next_distr, rewards, dones):
    dtype = next_distr.dtype
    batch_size = rewards.size(0)
    delta_z = float(V_MAX - V_MIN) / (NUM_ATOMS - 1)

    support = torch.linspace(
        V_MIN, V_MAX, NUM_ATOMS,
        device=DEVICE, dtype=dtype
    )
    dones = dones.float()

    proj_dist = torch.zeros(
        next_distr.size(), device=DEVICE, dtype=dtype
    )

    offset = torch.arange(
        0, batch_size * NUM_ATOMS, NUM_ATOMS,
        device=DEVICE
    )

    for j in range(NUM_ATOMS):
        Tz = rewards + (1 - dones) * (GAMMA ** N_STEPS) * support[j]
        Tz = Tz.clamp(V_MIN, V_MAX)
        b = (Tz - V_MIN) / delta_z
        l, u = b.floor().long(), b.ceil().long()

        m_l = (next_distr[:, j] * (u.float() - b)).to(dtype)
        m_u = (next_distr[:, j] * (b - l.float())).to(dtype)

        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), m_l.view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), m_u.view(-1))

    proj_dist += 1e-6
    proj_dist /= proj_dist.sum(dim=1, keepdim=True)
    return proj_dist



class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer("weight_eps", torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("bias_eps", torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1. / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(0.017)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(0.017)

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_eps.copy_(epsilon_out.ger(epsilon_in))
        self.bias_eps.copy_(epsilon_out)

    @staticmethod
    def _scale_noise(size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_eps
            bias = self.bias_mu   + self.bias_sigma   * self.bias_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class RainbowNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden = 256
        self.fc1 = NoisyLinear(input_dim, hidden)
        self.fc2 = NoisyLinear(hidden, hidden)

        # Dueling streams
        self.value_head = NoisyLinear(hidden, NUM_ATOMS)
        self.adv_head = NoisyLinear(hidden, len(ACTIONS) * NUM_ATOMS)

        self.register_buffer("support", torch.linspace(V_MIN, V_MAX, NUM_ATOMS))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        val = self.value_head(x).view(-1, 1, NUM_ATOMS)
        adv = self.adv_head(x).view(-1, len(ACTIONS), NUM_ATOMS)
        adv = adv - adv.mean(dim=1, keepdim=True)

        q_atoms = val + adv
        probs = F.softmax(q_atoms, dim=2)
        return probs

    def q_values(self, x):
        probs = self(x)
        q = torch.sum(probs * self.support, dim=2)
        return q

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


# --------------------------------------------------------------------------
# ----------------------  PRIORITISED REPLAY  ------------------------------
# --------------------------------------------------------------------------
class PrioritisedReplay:
    def __init__(self, capacity: int, alpha: float):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, transition: Transition):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, device=DEVICE, dtype=torch.float32)

        batch = Transition(*zip(*samples))
        return batch, weights, indices

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio.cpu().detach().item()


# --------------------------------------------------------------------------
# --------------------------  AGENT LOGIC  ---------------------------------
# --------------------------------------------------------------------------
class RainbowAgent:
    def __init__(self, state_size):
        self.policy_net = RainbowNet(state_size).to(DEVICE)
        self.target_net = RainbowNet(state_size).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = PrioritisedReplay(BUFFER_CAPACITY, PRIOR_ALPHA)

        self.support = torch.linspace(V_MIN, V_MAX, NUM_ATOMS).to(DEVICE)
        self.delta_z = float(V_MAX - V_MIN) / (NUM_ATOMS - 1)
        self.frame_idx = 0

    # ----------
    def act(self, state: np.ndarray):
        state = torch.tensor(state, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net.q_values(state)
        action = int(torch.argmax(q_values, dim=1).item())
        return action

    # ----------
    def remember(self, *args):
        self.memory.push(Transition(*args))

    # ----------
    def update(self, beta: float):
        if len(self.memory.buffer) < BATCH_SIZE:
            return

        # ─ sample & unpack ────────────────────────────────────────────────
        batch, weights, indices = self.memory.sample(BATCH_SIZE, beta)

        states = torch.tensor(np.array(batch.state), device=DEVICE)
        actions = torch.tensor(batch.action, device=DEVICE).unsqueeze(1)
        rewards = torch.tensor(batch.reward, device=DEVICE).unsqueeze(1)
        next_states = torch.tensor(np.array(batch.next_state), device=DEVICE)
        dones = torch.tensor(batch.done, device=DEVICE).unsqueeze(1)

        # ─ current value distribution ─────────────────────────────────────
        dist = self.policy_net(states)
        action_mask = actions.unsqueeze(-1).expand(-1, -1, NUM_ATOMS)
        dist = dist.gather(1, action_mask).squeeze(1)

        # ─ target value distribution (double DQN) ─────────────────────────
        next_action = self.policy_net.q_values(next_states).argmax(1)
        next_dist = self.target_net(next_states)[
                          range(BATCH_SIZE), next_action]

        target_dist = projection_distribution(next_dist, rewards, dones)

        # ─ element-wise KL loss (TD error proxy) ──────────────────────────
        td_loss = -torch.sum(target_dist * torch.log(dist + 1e-8), dim=1)
        loss = (td_loss * weights).mean()

        # ─ optimise ───────────────────────────────────────────────────────
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        with torch.no_grad():
            new_prios = td_loss + 1e-5
        self.memory.update_priorities(indices, new_prios)

    # ----------
    def train(self, env: TradingEnv):
        print(f"[debug] len(env.df) = {len(env.df)}")
        state = env.reset()
        for _ in tqdm(range(TRAINING_FRAMES), desc="Training Rainbow-DQN"):
            self.frame_idx += 1

            # Beta annealing for importance sampling
            beta = min(1.0, PRIOR_BETA_START + self.frame_idx * (1.0 - PRIOR_BETA_START) / PRIOR_BETA_FRAMES)

            action = random.randrange(len(ACTIONS)) if self.frame_idx < WARMUP_STEPS else self.act(state)
            next_state, reward, done, _ = env.step(action)

            self.remember(state, action, reward, next_state, done)
            self.update(beta)

            state = next_state
            if done:
                state = env.reset()

            if self.frame_idx % UPDATE_TARGET_EVERY == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.target_net.reset_noise()
                self.policy_net.reset_noise()

        # Final sync
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    # ----------
    def evaluate_action(self, state: np.ndarray) -> int:
        self.policy_net.eval()
        return self.act(state)


# --------------------------------------------------------------------------
# ----------------------  PUBLIC ENTRY-POINTS  -----------------------------
# --------------------------------------------------------------------------
def _prepare_env(full_df: pd.DataFrame, predicted_price: float, stop_ts=None):
    df_slice = full_df if stop_ts is None else full_df[pd.to_datetime(full_df["timestamp"]) <= stop_ts]
    env = TradingEnv(df_slice, predicted_signal=predicted_price)
    return env


# -------------------------  RUN LOGIC  ------------------------------------
def run_logic(current_price: float, predicted_price: float, ticker: str):
    from forest import api, buy_shares, sell_shares

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    try:
        df_full = load_dataframe(ticker)
    except FileNotFoundError:
        logger.error("CSV not found at %s", get_csv_filename(ticker))
        return

    env = _prepare_env(df_full, predicted_price)
    agent = RainbowAgent(env.state_size)
    agent.train(env)

    # Latest state (last row + predicted signal)
    last_state = env._get_state()
    action = agent.evaluate_action(last_state)

    # --- brokerage interaction ------------------------------------------
    try:
        account = api.get_account()
        cash = float(account.cash)
    except Exception as e:
        logger.error("[%s] Account query failed: %s", ticker, e)
        return

    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception:
        position_qty = 0.0

    logger.info("[%s] Agent action: %s", ticker, ACTIONS[action])

    if action == ACTION_BUY:
        if position_qty == 0:
            max_shares = int(cash // current_price)
            if max_shares > 0:
                buy_shares(ticker, max_shares, current_price, predicted_price)
                logger.info("[%s] BUY %d @ %.2f", ticker, max_shares, current_price)
        else:
            logger.info("[%s] Buy signal but already long – HOLD", ticker)

    elif action == ACTION_SELL:
        if position_qty > 0:
            sell_shares(ticker, position_qty, current_price, predicted_price)
            logger.info("[%s] SELL %d @ %.2f", ticker, position_qty, current_price)
        else:
            logger.info("[%s] Sell signal but no shares – HOLD", ticker)
    else:
        logger.info("[%s] HOLD – no trade executed", ticker)


# -------------------------  RUN BACKTEST  ---------------------------------
def run_backtest(current_price: float,
                 predicted_price: float,
                 position_qty: float,
                 current_timestamp: datetime,
                 candles: pd.DataFrame,
                 ticker):
    df_hist = load_dataframe(ticker, stop_ts=current_timestamp)
    if len(df_hist) < 10:
        return "NONE"

    env = _prepare_env(df_hist, predicted_price)
    agent = RainbowAgent(env.state_size)
    agent.train(env)

    latest_state = env._get_state()
    action = agent.evaluate_action(latest_state)

    # Map trading constraints identical to live logic
    if action == ACTION_BUY and position_qty > 0:
        action = ACTION_HOLD
    if action == ACTION_SELL and position_qty == 0:
        action = ACTION_HOLD

    return "BUY" if action == ACTION_BUY else "SELL" if action == ACTION_SELL else "NONE"