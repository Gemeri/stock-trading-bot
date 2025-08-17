import os
import math
import logging
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

# --------------------------- Logging -----------------------------------------

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)

# --------------------------- Constants ---------------------------------------

DATA_PATH = os.path.join("data", "TSLA_H4.csv")

# Strict base feature gate (exactly as provided)
BASE_FEATURES = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions', 'sentiment',
    'price_change', 'high_low_range', 'log_volume', 'macd_line', 'macd_signal', 'macd_histogram',
    'rsi', 'roc', 'atr', 'ema_9', 'ema_21', 'ema_50', 'ema_200', 'adx', 'obv', 'bollinger_upper',
    'bollinger_lower', 'bollinger_percB', 'returns_1', 'returns_3', 'returns_5', 'std_5', 'std_10',
    'lagged_close_1', 'lagged_close_2', 'lagged_close_3', 'lagged_close_5', 'lagged_close_10',
    'candle_body_ratio', 'wick_dominance', 'gap_vs_prev', 'volume_zscore', 'atr_zscore', 'rsi_zscore',
    'macd_cross', 'macd_hist_flip', 'month', 'hour_sin', 'hour_cos', 'day_of_week_sin',
    'day_of_week_cos', 'days_since_high', 'days_since_low',
]
# External prediction channel name (always present in model input)
EXT_PRED_COL = "ext_predicted_close"

# Actions
ACT_NONE = 0  # HOLD
ACT_BUY = 1
ACT_SELL = 2
ACTION_TO_STR = {ACT_NONE: "NONE", ACT_BUY: "BUY", ACT_SELL: "SELL"}

# Determinism
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --------------------------- Utilities ---------------------------------------


def _load_csv_strict(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at: {path}")
    df = pd.read_csv(path)
    # Keep ONLY allowed base features + optional 'predicted_close' if present
    keep_cols = [c for c in BASE_FEATURES if c in df.columns]
    missing_critical = [c for c in ['timestamp', 'close'] if c not in keep_cols]
    if missing_critical:
        raise ValueError(f"Missing critical columns in CSV: {missing_critical}")

    # Preserve order
    df = df[[*keep_cols, *( ['predicted_close'] if 'predicted_close' in df.columns else [] )]]

    # Parse timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    # Drop any duplicate timestamps just in case
    df = df[~df['timestamp'].duplicated()].reset_index(drop=True)

    return df


def _build_feature_matrix(
    df: pd.DataFrame,
    feature_cols: List[str],
    use_pred_from_csv: bool
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Returns:
        X_scaled: (N, D) float32
        closes: (N,) float32
        scaler: fitted StandardScaler on X_raw
    """
    # Select only those features that are actually present
    present = [c for c in feature_cols if c in df.columns]
    if 'close' not in df.columns:
        raise ValueError("CSV must include 'close' column.")
    closes = df['close'].astype(float).values.astype(np.float32)

    # Prepare the external prediction channel
    # If the CSV has 'predicted_close', use it; otherwise, use actual close as neutral fallback
    if use_pred_from_csv and ('predicted_close' in df.columns):
        ext_pred = df['predicted_close'].astype(float).values
    else:
        ext_pred = df['close'].astype(float).values  # neutral fallback

    # Feature matrix
    # We exclude 'timestamp' from the scaler/features; it is not numeric / not for the model.
    numeric_cols = [c for c in present if c != 'timestamp']
    X_raw = df[numeric_cols].astype(float).values
    # Append the external prediction channel as the last column
    X_raw = np.hstack([X_raw, ext_pred.reshape(-1, 1)])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)
    return X_scaled, closes.astype(np.float32), scaler, numeric_cols


def _make_state_vector(
    row: pd.Series,
    numeric_cols: List[str],
    scaler: StandardScaler,
    ext_pred_value: float
) -> np.ndarray:
    """
    Build a single standardized feature vector for inference from a DataFrame row,
    respecting the numeric column order and appending ext_pred_value.
    """
    feat_vals = row[numeric_cols].astype(float).values.reshape(1, -1)
    ext_pred = np.array([[float(ext_pred_value)]], dtype=np.float32)
    X = np.hstack([feat_vals, ext_pred])
    X_scaled = scaler.transform(X).astype(np.float32)
    return X_scaled.squeeze(0)


# --------------------------- Replay Buffer -----------------------------------


class ReplayBuffer:
    def __init__(self, capacity: int = 200_000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, s, a, r, ns, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (s, a, r, ns, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.array, zip(*batch))
        return (
            torch.from_numpy(s).float(),
            torch.from_numpy(a).long().view(-1, 1),
            torch.from_numpy(r).float().view(-1, 1),
            torch.from_numpy(ns).float(),
            torch.from_numpy(d.astype(np.float32)).float().view(-1, 1),
        )

    def __len__(self):
        return len(self.buffer)


# --------------------------- Q Network (Discrete) ----------------------------


class QNetwork(nn.Module):
    def __init__(self, in_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------- Conservative Q-Learner --------------------------


@dataclass
class CQLConfig:
    gamma: float = 0.99
    lr: float = 3e-4
    alpha: float = 0.2  # CQL regularizer weight
    batch_size: int = 64
    target_update_interval: int = 250
    tau: float = 1.0  # hard update when 1.0
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_steps: int = 10_000
    inaction_penalty: float = 1e-3  # small penalty for NONE


class ConservativeQLearner:
    def __init__(self, state_dim: int, n_actions: int, cfg: CQLConfig):
        self.cfg = cfg
        self.q = QNetwork(state_dim, n_actions)
        self.q_target = QNetwork(state_dim, n_actions)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optim = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.n_actions = n_actions
        self.total_updates = 0
        self.device = torch.device("cpu")
        self.q.to(self.device)
        self.q_target.to(self.device)

        self.epsilon = cfg.epsilon_start

    def _decay_epsilon(self):
        # Linear decay
        self.epsilon = max(
            self.cfg.epsilon_final,
            self.epsilon - (self.cfg.epsilon_start - self.cfg.epsilon_final) / max(1, self.cfg.epsilon_decay_steps)
        )

    def act(self, state: np.ndarray, valid_mask: Optional[np.ndarray] = None, greedy: bool = False) -> int:
        """
        Epsilon-greedy. valid_mask is a boolean array (n_actions,) where False means action is invalid.
        Invalid selections are re-mapped to NONE at the caller side as a safety, but we bias selection here too.
        """
        if (not greedy) and (random.random() < self.epsilon):
            # Explore among valid actions if given, else among all
            if valid_mask is not None and valid_mask.any():
                valid_idxs = np.flatnonzero(valid_mask)
                return int(np.random.choice(valid_idxs))
            return int(np.random.randint(0, self.n_actions))

        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_vals = self.q(s).cpu().numpy().squeeze(0)

        if valid_mask is not None and valid_mask.any():
            invalid = ~valid_mask
            q_vals = q_vals.copy()
            q_vals[invalid] = -1e9  # mask invalid
    
        q_vals = q_vals.copy()
        q_vals[ACT_NONE] -= 1e-6

        best_idxs = np.flatnonzero(q_vals == q_vals.max())
        a = int(np.random.choice(best_idxs))

        logger.debug(f"mask={(valid_mask.tolist() if valid_mask is not None else None)}, act={ACTION_TO_STR[a]}")
        return a

    def update(self, buffer: ReplayBuffer):
        if len(buffer) < self.cfg.batch_size:
            return

        s, a, r, ns, done = buffer.sample(self.cfg.batch_size)
        s, a, r, ns, done = s.to(self.device), a.to(self.device), r.to(self.device), ns.to(self.device), done.to(self.device)

        # Current Q(s,a)
        q_pred = self.q(s).gather(1, a)

        # Target: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            q_next = self.q_target(ns)
            max_q_next, _ = torch.max(q_next, dim=1, keepdim=True)
            target = r + (1.0 - done) * self.cfg.gamma * max_q_next

        # Bellman error (MSE)
        bellman_loss = nn.functional.mse_loss(q_pred, target)

        # CQL term: alpha * (logsumexp_a Q(s,a) - Q(s,a_taken))
        q_all = self.q(s)  # (B, A)
        logsumexp = torch.logsumexp(q_all, dim=1, keepdim=True)
        q_taken = q_pred  # already gathered
        cql_term = (logsumexp - q_taken).mean()

        loss = bellman_loss + self.cfg.alpha * cql_term

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.optim.step()

        self.total_updates += 1
        if self.cfg.tau >= 1.0:
            # Hard update periodically
            if self.total_updates % self.cfg.target_update_interval == 0:
                self.q_target.load_state_dict(self.q.state_dict())
        else:
            # Polyak
            with torch.no_grad():
                for p_targ, p in zip(self.q_target.parameters(), self.q.parameters()):
                    p_targ.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)

        # Update epsilon
        self._decay_epsilon()


# --------------------------- Training Loop -----------------------------------


def _train_cql_on_history(
    X: np.ndarray,
    closes: np.ndarray,
    epochs: int = 5,
    cfg: Optional[CQLConfig] = None,
    initial_cash: float = 1000.0,
) -> ConservativeQLearner:
    """
    Online training loop over historical sequence with inventory constraints,
    directional reward, and CQL regularizer.
    """
    if cfg is None:
        cfg = CQLConfig()

    n, feat_dim = X.shape
    if n < 3:
        raise ValueError("Not enough rows to train CQL (need >= 3).")

    agent = ConservativeQLearner(state_dim=feat_dim, n_actions=3, cfg=cfg)
    buffer = ReplayBuffer(capacity=200_000)

    total_steps = (n - 1) * epochs
    pbar = tqdm(total=total_steps, desc="Training (CQL)", leave=False)

    for _ in range(epochs):
        cash = initial_cash
        shares = 0

        # iterate over t, using next step as target for reward
        for t in range(n - 1):
            s = X[t]
            price_t = float(closes[t])
            next_price = float(closes[t + 1])

            # Validity mask (preferably bias exploration away from invalid actions)
            can_buy = (shares == 0) and (price_t > 0.0) and (int(cash // price_t) > 0)
            can_sell = (shares > 0)
            valid_mask = np.array(
                [True, can_buy, can_sell], dtype=bool
            )  # NONE is always valid

            a = agent.act(s, valid_mask=valid_mask, greedy=False)

            # Enforce invalid -> NONE
            executed_a = a
            if a == ACT_BUY and not can_buy:
                executed_a = ACT_NONE
            if a == ACT_SELL and not can_sell:
                executed_a = ACT_NONE

            # Execute inventory changes at current price
            if executed_a == ACT_BUY:
                max_shares = int(cash // price_t)
                if max_shares > 0:
                    cash -= max_shares * price_t
                    shares += max_shares
                else:
                    executed_a = ACT_NONE  # safety fallback

            elif executed_a == ACT_SELL:
                if shares > 0:
                    cash += shares * price_t
                    shares = 0
                else:
                    executed_a = ACT_NONE

            # Directional reward based on *next* price movement
            if executed_a == ACT_NONE:
                reward = -cfg.inaction_penalty
            else:
                trade_dir = 1.0 if executed_a == ACT_BUY else -1.0
                price_move = (next_price - price_t) / max(1e-12, price_t)
                reward = trade_dir * price_move

            ns = X[t + 1]
            done = 1.0 if (t + 1 == n - 1) else 0.0

            buffer.push(s, executed_a, reward, ns, done)
            agent.update(buffer)

            pbar.update(1)

    pbar.close()
    return agent


# --------------------------- Inference Helpers -------------------------------


def _decide_action_for_state(agent, state_vec, current_price, position_qty, cash: Optional[float] = None):
    shares = float(position_qty)
    price = float(current_price)

    # Allow BUY when cash is unknown (backtests) — only enforce position constraint
    can_buy  = (shares == 0.0) and (price > 0.0) and (cash is None or int(float(cash) // price) > 0)
    can_sell = (shares > 0.0)
    valid_mask = np.array([True, can_buy, can_sell], dtype=bool)

    a = agent.act(state_vec, valid_mask=valid_mask, greedy=True)
    if a == ACT_BUY and not can_buy:  a = ACT_NONE
    if a == ACT_SELL and not can_sell: a = ACT_NONE
    return ACTION_TO_STR[int(a)]


# --------------------------- Public API: run_logic ----------------------------


def run_logic(current_price: float, predicted_price: float, ticker: str):
    """
    Production entrypoint for live trading decision.
    - Trains on the entire CSV (as requested)
    - Pulls account / position via forest.api
    - Executes BUY (max shares) / SELL (all) / NONE
    """
    import logging
    from forest import api, buy_shares, sell_shares  # provided by caller's environment

    logger = logging.getLogger(__name__)
    logger.info(f"[{ticker}] Starting CQL training (live).")

    # 1) Load & prepare data (entire CSV)
    df = _load_csv_strict(DATA_PATH)

    # Establish which numeric columns we'll use from the base set
    # (timestamp excluded; only numeric base features)
    # We will always append the external prediction channel at the end
    present = [c for c in BASE_FEATURES if c in df.columns]
    numeric_cols = [c for c in present if c != 'timestamp']

    use_pred_from_csv = 'predicted_close' in df.columns
    X, closes, scaler, numeric_cols_used = _build_feature_matrix(
        df, feature_cols=present, use_pred_from_csv=use_pred_from_csv
    )

    # 2) Train CQL on the full history
    cfg = CQLConfig(
        gamma=0.99,
        lr=3e-4,
        alpha=0.2,
        batch_size=64,
        target_update_interval=250,
        tau=1.0,
        epsilon_start=1.0,
        epsilon_final=0.05,
        epsilon_decay_steps=10_000,
        inaction_penalty=1e-3,
    )
    agent = _train_cql_on_history(X, closes, epochs=5, cfg=cfg, initial_cash=1000.0)

    # 3) Build the latest state vector, using the runtime predicted_price as the external prediction
    latest_row = df.iloc[-1]
    state_vec = _make_state_vector(
        latest_row,
        numeric_cols=numeric_cols_used,
        scaler=scaler,
        ext_pred_value=float(predicted_price),
    )

    # 4) Get account & position
    try:
        account = api.get_account()
        cash = float(account.cash)
    except Exception as e:
        logger.error(f"[{ticker}] Error fetching account details: {e}")
        return

    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception:
        position_qty = 0.0

    logger.info(f"[{ticker}] Current Price: {current_price:.4f}, Predicted Price: {predicted_price:.4f}, "
                f"Position: {position_qty}, Cash: {cash:.2f}")

    # 5) Decide action with constraints
    action = _decide_action_for_state(
        agent, state_vec, current_price=float(current_price), position_qty=position_qty, cash=cash
    )

    # 6) Execute per rules
    if action == "BUY":
        if position_qty == 0:
            max_shares = int(cash // float(current_price))
            if max_shares > 0:
                logger.info(f"[{ticker}] BUY {max_shares} shares @ {current_price}.")
                buy_shares(ticker, max_shares, float(current_price), float(predicted_price))
            else:
                logger.info(f"[{ticker}] Insufficient cash to buy.")
        else:
            logger.info(f"[{ticker}] Already long; defaulting to NONE.")

    elif action == "SELL":
        if position_qty > 0:
            logger.info(f"[{ticker}] SELL {position_qty} shares @ {current_price}.")
            sell_shares(ticker, float(position_qty), float(current_price), float(predicted_price))
        else:
            logger.info(f"[{ticker}] No shares to sell; defaulting to NONE.")

    else:
        logger.info(f"[{ticker}] NONE (hold).")


# --------------------------- Public API: run_backtest -------------------------


def run_backtest(
    current_price: float,
    predicted_price: float,
    position_qty: float,
    current_timestamp,  # must be a datetime-like (pd.Timestamp / datetime)
    candles=None,  # intentionally unused, kept for signature compatibility
    ticker=None,
) -> str:
    """
    Backtest entrypoint (stateless call):
    - Trains CQL *only* on data up to and including current_timestamp
    - Returns the action ('BUY'/'SELL'/'NONE') for the candle AFTER current_timestamp
    - Enforces constraints: BUY->NONE if already long; SELL->NONE if flat
    - Trains from scratch each call; no future leakage beyond current_timestamp
    - Only uses predicted_price, current_timestamp, and position_qty  (as requested).
    """
    if not isinstance(current_timestamp, (pd.Timestamp, )):
        # try coercion
        try:
            current_timestamp = pd.to_datetime(current_timestamp)
        except Exception:
            raise TypeError("current_timestamp must be a datetime-like object (e.g., pandas.Timestamp).")

    df = _load_csv_strict(DATA_PATH)

    # Filter strictly to rows up to current_timestamp
    df_hist = df[df['timestamp'] <= current_timestamp].copy()
    if len(df_hist) < 3:
        # Not enough to train / infer
        return "NONE"

    present = [c for c in BASE_FEATURES if c in df_hist.columns]
    use_pred_from_csv = 'predicted_close' in df_hist.columns

    X, closes, scaler, numeric_cols_used = _build_feature_matrix(
        df_hist, feature_cols=present, use_pred_from_csv=use_pred_from_csv
    )

    # Train only on this historical window
    cfg = CQLConfig(
        gamma=0.99,
        lr=3e-4,
        alpha=0.2,
        batch_size=64,
        target_update_interval=250,
        tau=1.0,
        epsilon_start=1.0,
        epsilon_final=0.05,
        epsilon_decay_steps=10_000,
        inaction_penalty=1e-3,
    )
    agent = _train_cql_on_history(X, closes, epochs=5, cfg=cfg, initial_cash=1000.0)

    # Build state for *the last row* in df_hist (which corresponds to current_timestamp)
    last_row = df_hist.iloc[-1]
    state_vec = _make_state_vector(
        last_row,
        numeric_cols=numeric_cols_used,
        scaler=scaler,
        ext_pred_value=float(predicted_price),
    )

    # in run_backtest(...):
    action_str = _decide_action_for_state(
        agent, state_vec,
        current_price=float(current_price),
        position_qty=float(position_qty),
        cash=float('inf')  # assume we can buy at least 1 share
    )


    # Enforce spec: BUY when already long => NONE; SELL when flat => NONE
    if action_str == "BUY" and position_qty > 0:
        return "NONE"
    if action_str == "SELL" and position_qty <= 0:
        return "NONE"
    return action_str


# --------------------------- (Optional) Quick Self-Test -----------------------

if __name__ == "__main__":
    # Minimal smoke test (does not trade). Safe to leave here; no side effects.
    try:
        df_temp = _load_csv_strict(DATA_PATH)
        present = [c for c in BASE_FEATURES if c in df_temp.columns]
        X, closes, scaler, numeric_cols_used = _build_feature_matrix(
            df_temp, feature_cols=present, use_pred_from_csv=('predicted_close' in df_temp.columns)
        )
        _ = _train_cql_on_history(X, closes, epochs=1, initial_cash=1000.0)
        logger.info("CQL smoke test finished.")
    except Exception as e:
        logger.warning(f"Smoke test skipped or failed: {e}")
