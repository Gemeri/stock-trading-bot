import os
import json
import math
import time
import hashlib
import logging
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

# ML / DL
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import weight_norm
try:
    import config
except ImportError:
    print("Not running from script")

warnings.filterwarnings("ignore", category=FutureWarning)

TCN_CACHE_DIR = "tcn-cache"
os.makedirs(TCN_CACHE_DIR, exist_ok=True)
RETRAIN_EVERY = int(10)

try:
    from tqdm import trange
except Exception:
    def trange(n, **kwargs):
        return range(n)


# --------------------------------------------------------------------------------------
# Config / Environment
# --------------------------------------------------------------------------------------

LOGGER_NAME = __name__
logger = logging.getLogger(LOGGER_NAME)
try:
    BAR_TIMEFRAME = config.BAR_TIMEFRAME
except Exception:
    BAR_TIMEFRAME = "4Hour"
TIMEFRAME_MAP = {
    "4Hour": "H4", "2Hour": "H2", "1Hour": "H1",
    "30Min": "M30", "15Min": "M15"
}
CONVERTED_TIMEFRAME = TIMEFRAME_MAP.get(BAR_TIMEFRAME, BAR_TIMEFRAME)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DOTA_DIR = os.path.join(os.path.dirname(__file__), "data")
def timeframe_subdor(tf_code: str) -> str:
    """Return the directory path for a given timeframe code, creating it if needed."""

    path = os.path.join(DOTA_DIR, tf_code)
    os.makedirs(path, exist_ok=True)
    return path

def timeframe_subdir(tf_code: str) -> str:
    """Return the directory path for a given timeframe code, creating it if needed."""

    path = os.path.join(DOTA_DIR, tf_code)
    os.makedirs(path, exist_ok=True)
    return path

def get_csv_filename(ticker: str) -> str:
    rel_path = os.path.join(timeframe_subdor(CONVERTED_TIMEFRAME), f"{ticker}_{CONVERTED_TIMEFRAME}.csv")
    if os.path.exists(rel_path):
        return rel_path
    # Fallback to absolute
    return os.path.join(timeframe_subdir(CONVERTED_TIMEFRAME), f"{ticker}_{CONVERTED_TIMEFRAME}.csv")




# Strict base feature gate (exactly as provided)
BASE_FEATURES = [
    'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions', 'sentiment',
    'greed_index', 'news_count', 'news_volume_z', 'price_change', 'high_low_range',
    'macd_line', 'macd_signal', 'macd_histogram', 'rsi', 'roc', 'atr', 'ema_9', 'ema_21',
    'ema_50', 'ema_200', 'adx', 'obv', 'bollinger_percB', 'returns_1',
    'returns_3', 'returns_5', 'std_5', 'std_10', 'lagged_close_1', 'lagged_close_2',
    'lagged_close_3', 'lagged_close_5', 'lagged_close_10', 'candle_body_ratio',
    'wick_dominance', 'gap_vs_prev', 'volume_zscore', 'atr_zscore', 'rsi_zscore',
    'month', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
    'days_since_high', 'days_since_low', "d_sentiment"
]

# -------------------- Model / Training defaults (env overridable) ---------------------
LOOKBACK = int(256)
ENSEMBLE_SIZE = 3
KERNEL_SIZE = 5
CHANNELS = [int(x) for x in os.getenv("TCN_CHANNELS", "64,64,64,64,64,64").split(",")]
DROPOUT = 0.05

LR = float(os.getenv("TCN_LR", "3e-4"))
WEIGHT_DECAY = float(os.getenv("TCN_WEIGHT_DECAY", "1e-4"))
BATCH_SIZE = 256
EPOCHS = 80
PATIENCE = 10
VAL_FRACTION = 0.15

# Labeling
ALPHA = 0.25
ATR_PERIOD = 14

# Loss shaping
USE_FOCAL = os.getenv("TCN_USE_FOCAL", "1") == "1"
FOCAL_GAMMA = 2.0
USE_RET_WEIGHTS = os.getenv("TCN_USE_RET_WEIGHTS", "1") == "1"
RET_WEIGHT_POW = 0.5

# Trading thresholds (defaults; will be overridden by learned adaptive thresholds if available)
BUY_THRESHOLD = 0.55
SELL_THRESHOLD = 0.45

# Paths
MODEL_ROOT = os.getenv("TCN_MODEL_DIR", "models")
os.makedirs(MODEL_ROOT, exist_ok=True)

# Torch device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def _safe_parse_ts(ts) -> pd.Timestamp:
    """
    Parse many timestamp formats into a naive pandas Timestamp.
    - ints/floats treated as epoch seconds or ms
    - strings parsed via pandas
    - always returns naive (timezone-stripped) Timestamp
    """
    if isinstance(ts, (int, float)):
        val = int(ts)
        if val > 10_000_000_000:  # ms heuristic
            t = pd.to_datetime(val, unit="ms", utc=True)
        else:
            t = pd.to_datetime(val, unit="s", utc=True)
    else:
        t = pd.to_datetime(ts, utc=True, errors="coerce")

    if pd.isna(t):
        raise ValueError(f"Could not parse timestamp: {ts!r}")

    if getattr(t, "tz", None) is not None:
        t = t.tz_convert(None)
    return t

def _ticker_key(ticker: str) -> str:
    return f"{ticker}_{CONVERTED_TIMEFRAME}"

def _model_dir(ticker: str) -> str:
    d = os.path.join(TCN_CACHE_DIR, _ticker_key(ticker))
    os.makedirs(d, exist_ok=True)
    return d

def _meta_path(ticker: str) -> str:
    return os.path.join(_model_dir(ticker), "meta.json")

def _scaler_path(ticker: str) -> str:
    return os.path.join(_model_dir(ticker), "scaler.npy")

def _model_path(ticker: str, idx: int) -> str:
    return os.path.join(_model_dir(ticker), f"tcn_{idx}.pt")

# NEW: countdown file path
def _countdown_path(ticker: str) -> str:
    return os.path.join(_model_dir(ticker), "countdown.json")

# ===== NEW: countdown helpers =====
def _load_countdown(ticker: str) -> Dict[str, float]:
    p = _countdown_path(ticker)
    if os.path.exists(p):
        try:
            with open(p, "r") as f:
                obj = json.load(f)
                # Backward-compatible defaults
                if "retrain_every" not in obj:
                    obj["retrain_every"] = RETRAIN_EVERY
                if "remaining" not in obj:
                    obj["remaining"] = obj["retrain_every"]
                if "call_count" not in obj:
                    obj["call_count"] = 0
                return obj
        except Exception:
            pass
    return {"retrain_every": RETRAIN_EVERY, "remaining": RETRAIN_EVERY, "call_count": 0}

def _save_countdown(ticker: str, obj: Dict[str, float]) -> None:
    p = _countdown_path(ticker)
    try:
        with open(p, "w") as f:
            json.dump(obj, f, indent=2)
    except Exception as e:
        logger.warning(f"[{_ticker_key(ticker)}] Failed to persist countdown: {e}")


def _hash_features(features: List[str]) -> str:
    s = "|".join(features)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _load_csv(ticker: str) -> pd.DataFrame:
    """
    Load CSV and return a dataframe with a clean, naive 'timestamp' column.
    - Accepts columns named 'timestamp' | 'time' | 'date'
    - Parses as UTC, then drops tz to naive
    - Drops rows where timestamp couldn't be parsed (NaT)
    - Sorts by timestamp ascending
    """
    path = get_csv_filename(ticker)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data CSV not found: {path}")

    df = pd.read_csv(path)

    # Find a timestamp-like column
    ts_col = None
    for c in ("timestamp", "time", "date"):
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        raise ValueError("CSV must include a 'timestamp' (or 'time'/'date') column.")

    s = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    if hasattr(s, "dt") and s.dt.tz is not None:
        s = s.dt.tz_convert(None)

    df = df.assign(timestamp=s).dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def _fill_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill (causal) then repair remaining NaNs/Infs.
    - First ffill
    - For specific variance-carrying cols, fill remaining NaNs by rolling median
    - Replace inf with NaN
    - Final fillna(0)
    """
    df = df.copy()
    df = df.ffill()

    # Gentle repair for variance-carrying cols before final zeros
    gentle_cols = [c for c in df.columns if c in [
        "std_5", "std_10", "atr", "atr_zscore", "volume_zscore", "news_volume_z"
    ]]
    for c in gentle_cols:
        if c in df.columns:
            med = df[c].rolling(50, min_periods=1).median()
            df[c] = df[c].fillna(med)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df

# --------------------------------------------------------------------------------------
# Labeling: ATR, returns, masked multi-horizon targets
# --------------------------------------------------------------------------------------

def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = _true_range(high, low, close)
    # Wilder's ATR
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def _build_masked_labels(df: pd.DataFrame, alpha: float, atr_period: int) -> Dict[str, np.ndarray]:
    """
    Build masked labels for horizons h=1,2,3 using ATR-normalized thresholds.
    y_h: 1 if r_{t+h} > alpha * ATR_t/close_t; 0 if < -alpha*...; -1 otherwise (ignore)
    Also returns sample weights per horizon proportional to |r_{t+h}|.
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    atr = _atr(high, low, close, period=atr_period)
    # Normalize by price to express as return-like magnitude
    atr_norm = (atr / close.replace(0, np.nan)).fillna(0.0)

    y_list = []
    w_list = []
    for h in (1, 2, 3):
        # forward return
        r = (close.shift(-h) / close) - 1.0
        up_th = alpha * atr_norm
        dn_th = -alpha * atr_norm

        y = np.full(len(df), -1.0, dtype=np.float32)  # ignore by default
        y[(r > up_th)] = 1.0
        y[(r < dn_th)] = 0.0

        # sample weights: emphasize decisive returns
        if USE_RET_WEIGHTS:
            w = np.power(np.abs(r).fillna(0.0).values, RET_WEIGHT_POW)
        else:
            w = np.ones(len(df), dtype=np.float32)

        # last h rows have no future target
        y[-h:] = -1.0
        w[-h:] = 0.0

        y_list.append(y)
        w_list.append(w.astype(np.float32))

    return {
        "y1": y_list[0], "y2": y_list[1], "y3": y_list[2],
        "w1": w_list[0], "w2": w_list[1], "w3": w_list[2]
    }

# --------------------------------------------------------------------------------------
# TCN (multi-head)
# --------------------------------------------------------------------------------------

class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                           padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size,
                                           padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.final_relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + res)

class TCN(nn.Module):
    def __init__(self, num_inputs: int, channels: List[int], kernel_size: int, dropout: float):
        super().__init__()
        layers = []
        in_ch = num_inputs
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        # Multi-horizon heads -> 3 logits (h=1,2,3)
        self.head = nn.Linear(in_ch, 3)

    def forward(self, x):
        """
        x: (B, F, L)
        returns logits: (B, 3) for horizons 1,2,3
        """
        y = self.network(x)      # (B, C, L)
        last = y[:, :, -1]       # (B, C)
        logits = self.head(last) # (B, 3)
        return logits

# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------

class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y123: np.ndarray, w123: np.ndarray):
        """
        X: (N, F, L)
        y123: (N, 3) with values in {0,1} or -1 for ignore
        w123: (N, 3) sample weights per horizon
        """
        self.X = X.astype(np.float32)
        self.y123 = y123.astype(np.float32)
        self.w123 = w123.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y123[idx], self.w123[idx]

@dataclass
class TrainArtifacts:
    models: List[TCN]
    scaler: StandardScaler
    features: List[str]
    temperature: float
    buy_threshold: float
    sell_threshold: float

# --------------------------------------------------------------------------------------
# Sequence builder & scaling (train-only scaler)
# --------------------------------------------------------------------------------------

def _drop_near_constant(train_df: pd.DataFrame, features: List[str], eps: float = 1e-8) -> List[str]:
    stds = train_df[features].astype(np.float32).std().fillna(0.0)
    kept = [f for f in features if stds.get(f, 0.0) > eps]
    return kept

def _build_sequences_scaled(
    df_clean: pd.DataFrame,
    features: List[str],
    lookback: int,
    val_frac: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, List[str]]:
    """
    Build sequences X and masked labels y for horizons (1,2,3) with a scaler fit on the train slice only.
    Returns:
        X: (N,F,L)
        y123: (N,3) in {0,1} or -1
        w123: (N,3) sample weights
        train_idx, val_idx: arrays of indices for sequences
        scaler: fitted on train slice only
        features_used: after dropping near-constant features
    """
    # Masked labels & weights on the full cleaned df (before any slicing)
    lbl = _build_masked_labels(df_clean, alpha=ALPHA, atr_period=ATR_PERIOD)
    y1, y2, y3 = lbl["y1"], lbl["y2"], lbl["y3"]
    w1, w2, w3 = lbl["w1"], lbl["w2"], lbl["w3"]

    # Compute total number of windows (needs at least lookback)
    M = len(df_clean)
    if M < lookback + 3:  # allow space for horizons
        return (np.empty((0, len(features), lookback), np.float32),
                np.empty((0,3), np.float32),
                np.empty((0,3), np.float32),
                np.array([], dtype=int),
                np.array([], dtype=int),
                np.array([], dtype=int),
                StandardScaler(), features)

    # Split by windows count (contiguous walk-forward)
    N = M - lookback  # number of windows whose last index is i in [lookback..M-1]
    val_n = max(1, int(round(N * val_frac)))
    train_n = N - val_n
    if train_n <= 0:
        train_n = N - 1
        val_n = 1

    # Rows participating in train windows end at index train_end = lookback-1 .. lookback-1+(train_n-1)
    train_last_row = (lookback - 1) + (train_n - 1)
    # Fit scaler on rows that are used by ANY train window
    fit_end_row = train_last_row  # end row included
    fit_start_row = 0
    fit_df = df_clean.iloc[fit_start_row: fit_end_row + 1]

    # Drop near-constant features based on this fit_df (pre-scaling)
    features_used = _drop_near_constant(fit_df, features)

    scaler = StandardScaler()
    scaler.fit(fit_df[features_used].values.astype(np.float32))

    # Scale entire df_clean with train-fitted scaler
    df_scaled = df_clean.copy()
    df_scaled[features_used] = scaler.transform(df_scaled[features_used].values.astype(np.float32))

    # Build X windows and y/w per window
    X = np.zeros((N, len(features_used), lookback), dtype=np.float32)
    y123 = np.zeros((N, 3), dtype=np.float32)
    w123 = np.zeros((N, 3), dtype=np.float32)

    for i in range(lookback - 1, M - 0):  # i is the last row index of the window
        win_idx = i - (lookback - 1)
        if win_idx >= N:
            break
        X[win_idx] = df_scaled[features_used].values[i - lookback + 1: i + 1].T
        # Labels aligned to window end at i (so use y at i)
        y123[win_idx, 0] = y1[i]
        y123[win_idx, 1] = y2[i]
        y123[win_idx, 2] = y3[i]

        w123[win_idx, 0] = w1[i]
        w123[win_idx, 1] = w2[i]
        w123[win_idx, 2] = w3[i]

    # Train / Val indices (contiguous, no shuffling)
    train_idx = np.arange(0, train_n, dtype=int)
    val_idx = np.arange(train_n, N, dtype=int)
    return X, y123, w123, train_idx, val_idx, np.arange(N, dtype=int), scaler, features_used

# --------------------------------------------------------------------------------------
# Loss, training, calibration, thresholds
# --------------------------------------------------------------------------------------

def _masked_bce_focal_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor, gamma: float = 2.0, use_focal: bool = True) -> torch.Tensor:
    """
    logits: (B,3), targets: (B,3) in {0,1,-1}, weights: (B,3)
    mask = (targets >= 0)
    """
    mask = (targets >= 0.0)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    # BCE with logits per element
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits[mask], targets[mask], reduction='none'
    )

    if use_focal:
        # p_t = sigmoid(logit) if y=1 else (1-sigmoid)
        p = torch.sigmoid(logits[mask])
        t = targets[mask]
        pt = p * t + (1 - p) * (1 - t)
        focal_factor = (1.0 - pt).pow(gamma)
        bce = focal_factor * bce

    # apply per-sample weights (already masked)
    w = weights[mask]
    bce = bce * w

    # average over valid elements
    loss = bce.mean()
    return loss

def _evaluate_logits_to_probs(avg_logits: np.ndarray, temperature: float) -> np.ndarray:
    """
    avg_logits: (N, 3) averaged across ensemble (per horizon)
    Combine 3 horizons by averaging logits, then apply temperature scaling and sigmoid -> p_up.
    """
    # average across horizons in logit space
    logit = avg_logits.mean(axis=1)  # (N,)
    logit = logit / max(1e-6, temperature)
    p = 1.0 / (1.0 + np.exp(-logit))
    return p

def _grid_search_temperature(val_logits_ens: np.ndarray, val_targets: np.ndarray) -> float:
    """
    Simple grid search for temperature scaling minimizing NLL on the valid mask of horizon-1 targets.
    We use only horizon-1 labels (most immediate) for calibration; you can change to combine all.
    val_logits_ens: (N, 3)
    val_targets: (N,) in {0,1,-1}
    """
    mask = val_targets >= 0
    if mask.sum() == 0:
        return 1.0
    logits = val_logits_ens[mask, :].mean(axis=1)  # average horizons
    y = val_targets[mask].astype(np.float32)

    temps = np.linspace(0.5, 3.0, 26)  # 0.5..3.0
    best_t, best_nll = 1.0, float("inf")
    for T in temps:
        z = logits / T
        p = 1.0 / (1.0 + np.exp(-z))
        # clamp for numerical stability
        p = np.clip(p, 1e-6, 1 - 1e-6)
        nll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
        if nll < best_nll:
            best_nll, best_t = nll, T
    return float(best_t)

def _best_thresholds_from_val(p_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, float]:
    """
    Choose adaptive buy/sell thresholds from validation set.
    For BUY (positive class): maximize Youden's J (TPR - FPR).
    For SELL (negative class): do the same on flipped labels (treat y=0 as positive).
    Returns (tau_buy, tau_sell).
    """
    mask = y_val >= 0
    if mask.sum() < 10:
        return BUY_THRESHOLD, SELL_THRESHOLD

    p = p_val[mask]
    y = y_val[mask].astype(np.int32)

    # candidate thresholds
    cands = np.unique(np.round(p, 3))
    if len(cands) < 10:
        cands = np.linspace(0.3, 0.7, 81)

    def youden_j(p, y, thr, pos_label=1):
        if pos_label == 1:
            y_pos = y
        else:
            y_pos = 1 - y
        yhat = (p >= thr).astype(int)
        tp = ((yhat == 1) & (y_pos == 1)).sum()
        fp = ((yhat == 1) & (y_pos == 0)).sum()
        fn = ((yhat == 0) & (y_pos == 1)).sum()
        tn = ((yhat == 0) & (y_pos == 0)).sum()
        tpr = tp / max(1, (tp + fn))
        fpr = fp / max(1, (fp + tn))
        return tpr - fpr

    best_buy, best_j = BUY_THRESHOLD, -1.0
    for t in cands:
        j = youden_j(p, y, t, pos_label=1)
        if j > best_j:
            best_j, best_buy = j, float(t)

    best_sell, best_j2 = SELL_THRESHOLD, -1.0
    for t in cands:
        j = youden_j(p, y, t, pos_label=0)  # negative as "positive"
        if j > best_j2:
            best_j2, best_sell = j, float(t)

    return best_buy, best_sell

# --------------------------------------------------------------------------------------
# Training / Prediction
# --------------------------------------------------------------------------------------

# ===== Replace _train_single_model to include a tqdm epoch bar =====
def _train_single_model(
    X_train: np.ndarray, y_train: np.ndarray, w_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray, w_val: np.ndarray,
    num_features: int, seed: int
) -> Tuple[TCN, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TCN(num_inputs=num_features, channels=CHANNELS, kernel_size=KERNEL_SIZE, dropout=DROPOUT).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(opt, T_max=EPOCHS)
    criterion = _masked_bce_focal_loss

    train_loader = DataLoader(SeqDataset(X_train, y_train, w_train), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(SeqDataset(X_val, y_val, w_val), batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    best_val = float("inf")
    best_state = None
    patience = PATIENCE
    no_improve = 0

    for epoch in trange(EPOCHS, desc=f"TCN train (seed={seed})", ncols=100):
        model.train()
        train_loss = 0.0
        for xb, yb, wb in train_loader:
            xb, yb, wb = xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)            # (B,3)
            loss = criterion(logits, yb, wb, gamma=FOCAL_GAMMA, use_focal=USE_FOCAL)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= max(1, len(train_loader.dataset))

        # Validation (masked BCE w/o focal for stability)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, wb in val_loader:
                xb, yb, wb = xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
                logits = model(xb)
                loss = _masked_bce_focal_loss(logits, yb, wb, gamma=FOCAL_GAMMA, use_focal=False)
                val_loss += loss.item() * xb.size(0)
        val_loss /= max(1, len(val_loader.dataset))

        logger.debug(f"Epoch {epoch+1}/{EPOCHS} - train {train_loss:.5f} - val {val_loss:.5f}")
        scheduler.step()

        if val_loss + 1e-8 < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Collect validation logits for calibration (per-window, avg over heads later)
    model.eval()
    all_logits = []
    with torch.no_grad():
        for xb, yb, wb in val_loader:
            xb = xb.to(DEVICE)
            logits = model(xb).cpu().numpy()  # (B,3)
            all_logits.append(logits)
    if len(all_logits) == 0:
        val_logits = np.zeros((0, 3), dtype=np.float32)
    else:
        val_logits = np.concatenate(all_logits, axis=0)  # (N_val,3)

    return model.to(DEVICE), val_logits


def _prepare_training_matrices(
    df: pd.DataFrame, features: List[str], lookback: int, val_frac: float, up_to_ts: Optional[pd.Timestamp] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, List[str], np.ndarray]:
    """
    Returns:
        X_train, y_train, w_train,
        X_val,   y_val,   w_val,
        scaler, features_used, val_targets_h1 (N_val,) for calibration/thresholds
    """
    if up_to_ts is not None:
        df = df.loc[df["timestamp"] <= up_to_ts].copy()

    if df.empty:
        empty = (np.empty((0, len(features), lookback), np.float32),
                 np.empty((0,3), np.float32),
                 np.empty((0,3), np.float32),
                 np.empty((0, len(features), lookback), np.float32),
                 np.empty((0,3), np.float32),
                 np.empty((0,3), np.float32),
                 StandardScaler(), features, np.empty((0,), np.float32))
        return empty

    # Clean
    df_clean = _fill_and_clean(df)

    # Ensure required features exist
    for f in features:
        if f not in df_clean.columns:
            raise ValueError(f"Missing feature '{f}' in CSV.")

    # Build sequences with train-only scaler & masked labels
    X, y123, w123, train_idx, val_idx, all_idx, scaler, features_used = _build_sequences_scaled(
        df_clean, features, lookback, val_frac
    )

    if X.shape[0] == 0 or len(train_idx) == 0 or len(val_idx) == 0:
        empty = (np.empty((0, len(features_used), lookback), np.float32),
                 np.empty((0,3), np.float32),
                 np.empty((0,3), np.float32),
                 np.empty((0, len(features_used), lookback), np.float32),
                 np.empty((0,3), np.float32),
                 np.empty((0,3), np.float32),
                 scaler, features_used, np.empty((0,), np.float32))
        return empty

    # Split
    X_train, y_train, w_train = X[train_idx], y123[train_idx], w123[train_idx]
    X_val,   y_val,   w_val   = X[val_idx],   y123[val_idx],   w123[val_idx]

    # For calibration/thresholds we use horizon-1 targets on val
    val_targets_h1 = y_val[:, 0].copy()

    return X_train, y_train, w_train, X_val, y_val, w_val, scaler, features_used, val_targets_h1

def _train_ensemble(
    df: pd.DataFrame, features: List[str], lookback: int, seeds: List[int], up_to_ts: Optional[pd.Timestamp] = None
) -> TrainArtifacts:
    Xtr, ytr, wtr, Xv, yv, wv, scaler, features_used, val_targets_h1 = _prepare_training_matrices(
        df, features, lookback, VAL_FRACTION, up_to_ts=up_to_ts
    )
    if Xtr.shape[0] == 0 or Xv.shape[0] == 0:
        raise RuntimeError("Not enough data to train TCN after cleaning and splitting.")

    models: List[TCN] = []
    val_logits_ens = []  # collect val logits per model for temperature scaling
    for i, seed in enumerate(seeds):
        m, val_logits = _train_single_model(
            Xtr, ytr, wtr, Xv, yv, wv, num_features=Xtr.shape[1], seed=seed
        )
        models.append(m)
        val_logits_ens.append(val_logits)

    # Aggregate validation logits across ensemble (average later)
    if len(val_logits_ens) == 0:
        raise RuntimeError("No validation logits available.")
    # Ensure same number of val windows per model
    min_len = min(v.shape[0] for v in val_logits_ens)
    val_logits_ens = np.stack([v[:min_len] for v in val_logits_ens], axis=0)  # (E, N_val, 3)
    val_targets_h1 = val_targets_h1[:min_len]

    # Average logits across ensemble for calibration
    avg_val_logits = val_logits_ens.mean(axis=0)  # (N_val, 3)

    # Temperature scaling (on val)
    temperature = _grid_search_temperature(avg_val_logits, val_targets_h1)

    # Calibrated probabilities on validation
    p_val = _evaluate_logits_to_probs(avg_val_logits, temperature)  # (N_val,)

    # Adaptive thresholds from validation
    tau_buy, tau_sell = _best_thresholds_from_val(p_val, val_targets_h1)

    return TrainArtifacts(
        models=models, scaler=scaler, features=features_used,
        temperature=temperature, buy_threshold=tau_buy, sell_threshold=tau_sell
    )

# --------------------------------------------------------------------------------------
# Prediction helpers
# --------------------------------------------------------------------------------------

def _predict_p_up(models: List[TCN], scaler: StandardScaler, features: List[str], window_df: pd.DataFrame, temperature: float) -> float:
    """
    window_df: last LOOKBACK rows of features (raw, already cleaned)
    Returns calibrated p_up using ensemble-avg of logits across 3 horizons.
    """
    if len(window_df) < LOOKBACK:
        raise RuntimeError("Insufficient window length for prediction.")
    X_latest = window_df[features].values.astype(np.float32)
    X_latest = scaler.transform(X_latest)
    X_latest = X_latest[-LOOKBACK:]  # (L, F)
    X_latest = np.expand_dims(X_latest.T, axis=0)  # (1, F, L)
    xb = torch.from_numpy(X_latest).to(DEVICE)

    with torch.no_grad():
        logits_list = []
        for m in models:
            m.eval()
            logits = m(xb).detach().cpu().numpy()  # (1,3)
            logits_list.append(logits[0])
        avg_logits = np.stack(logits_list, axis=0).mean(axis=0, keepdims=False)  # (3,)
    # combine horizons by logit average and calibrate
    p_up = _evaluate_logits_to_probs(avg_logits.reshape(1, 3), temperature)[0]
    return float(p_up)

# --------------------------------------------------------------------------------------
# Persistence for LIVE (run_logic). Backtests train in-memory.
# --------------------------------------------------------------------------------------

def _csv_mtime(path: str) -> float:
    return os.path.getmtime(path) if os.path.exists(path) else 0.0

# ===== Replace _ensure_live_models with a unified, countdown-aware cache =====
def _ensure_models_cached(ticker: str, up_to_ts: Optional[pd.Timestamp] = None, decrement: bool = True) -> TrainArtifacts:
    """
    Loads cached ensemble+scaler+meta from tcn-cache if available and countdown not expired.
    Retrains if:
      - missing files, or
      - CSV updated, or
      - arch/features changed, or
      - countdown 'remaining' <= 0
    After a successful load (no retrain) and decrement=True, countdown.remaining -= 1.
    After a retrain, countdown.remaining = retrain_every - (1 if decrement else 0).
    If up_to_ts is provided, training strictly uses data <= up_to_ts (walk-forward safe).
    """
    df = _load_csv(ticker)
    feats = BASE_FEATURES.copy()
    if "predicted_close" in df.columns and not df["predicted_close"].isna().any():
        feats.append("predicted_close")

    key = _ticker_key(ticker)
    mdir = _model_dir(ticker)
    meta_path = _meta_path(ticker)
    scaler_path = _scaler_path(ticker)
    csv_path = get_csv_filename(ticker)
    countdown = _load_countdown(ticker)

    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    expected_models = [_model_path(ticker, i) for i in range(ENSEMBLE_SIZE)]
    csv_mtime = _csv_mtime(csv_path)

    # retrain triggers
    need_retrain = False
    reasons = []

    if not all(os.path.exists(p) for p in expected_models):
        need_retrain = True; reasons.append("missing_model")
    if not os.path.exists(scaler_path):
        need_retrain = True; reasons.append("missing_scaler")
    if not meta or meta.get("csv_mtime", 0) < csv_mtime:
        need_retrain = True; reasons.append("csv_updated")

    arch_fingerprint = {
        "features_sha256": _hash_features(feats),
        "lookback": LOOKBACK,
        "channels": CHANNELS,
        "kernel_size": KERNEL_SIZE,
        "dropout": DROPOUT,
        "alpha": ALPHA,
        "atr_period": ATR_PERIOD
    }
    if any(meta.get(k) != v for k, v in arch_fingerprint.items()):
        need_retrain = True; reasons.append("arch_or_features_changed")

    # Countdown trigger (only if everything else looks valid)
    if not need_retrain:
        if countdown.get("remaining", RETRAIN_EVERY) <= 0:
            need_retrain = True; reasons.append("countdown_exhausted")

    # ----- train or load -----
    if need_retrain:
        logger.info(f"[{key}] Training TCN ensemble ({', '.join(reasons)})...")
        seeds = [777 + i * 17 for i in range(ENSEMBLE_SIZE)]
        # Train on full or up_to_ts slice (walk-forward safety)
        artifacts = _train_ensemble(df, feats, LOOKBACK, seeds=seeds, up_to_ts=up_to_ts)

        # Persist scaler
        np.save(scaler_path, {"mean": artifacts.scaler.mean_, "scale": artifacts.scaler.scale_}, allow_pickle=True)
        # Persist models
        for i, m in enumerate(artifacts.models):
            torch.save(m.state_dict(), _model_path(ticker, i))
        # Save meta
        last_ts = df["timestamp"].max() if up_to_ts is None else _safe_parse_ts(up_to_ts)
        meta = {
            "csv_mtime": csv_mtime,
            **arch_fingerprint,
            "features_used": artifacts.features,
            "temperature": artifacts.temperature,
            "buy_threshold": artifacts.buy_threshold,
            "sell_threshold": artifacts.sell_threshold,
            "last_trained_ts": str(last_ts),
            "saved_at": time.time(),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Reset countdown after training
        remaining = int(countdown.get("retrain_every", RETRAIN_EVERY))
        if decrement:
            remaining = max(0, remaining - 1)  # current call consumes one
        countdown.update({
            "retrain_every": int(countdown.get("retrain_every", RETRAIN_EVERY)),
            "remaining": remaining,
            "call_count": int(countdown.get("call_count", 0)) + 1
        })
        _save_countdown(ticker, countdown)
        logger.info(f"[{key}] TCN ensemble trained and cached. Countdown remaining: {remaining}")
        return artifacts

    # Load cached artifacts
    obj = np.load(scaler_path, allow_pickle=True).item()
    scaler = StandardScaler()
    scaler.mean_ = obj["mean"]; scaler.scale_ = obj["scale"]
    scaler.n_features_in_ = scaler.mean_.shape[0]

    features_used = meta.get("features_used", feats)

    models: List[TCN] = []
    for i in range(ENSEMBLE_SIZE):
        m = TCN(num_inputs=len(features_used), channels=CHANNELS, kernel_size=KERNEL_SIZE, dropout=DROPOUT).to(DEVICE)
        m.load_state_dict(torch.load(_model_path(ticker, i), map_location=DEVICE))
        m.eval()
        models.append(m)

    # Decrement countdown on successful load
    if decrement:
        rem = int(countdown.get("remaining", RETRAIN_EVERY))
        rem = max(0, rem - 1)
        countdown.update({
            "remaining": rem,
            "call_count": int(countdown.get("call_count", 0)) + 1
        })
        _save_countdown(ticker, countdown)

    return TrainArtifacts(
        models=models,
        scaler=scaler,
        features=features_used,
        temperature=float(meta.get("temperature", 1.0)),
        buy_threshold=float(meta.get("buy_threshold", BUY_THRESHOLD)),
        sell_threshold=float(meta.get("sell_threshold", SELL_THRESHOLD)),
    )

# --------------------------------------------------------------------------------------
# Unified decision logic (IDENTICAL for live & backtest)
# --------------------------------------------------------------------------------------

def decide_action(p_up: float, position_qty: float, tau_buy: float, tau_sell: float) -> str:
    """
    EXACT SAME strategy used by run_logic and run_backtest.
    - BUY if p_up >= tau_buy and not in position
    - SELL if p_up <= tau_sell and in position
    - else NONE
    """
    if p_up >= tau_buy and position_qty == 0:
        return "BUY"
    if p_up <= tau_sell and position_qty > 0:
        return "SELL"
    return "NONE"

# --------------------------------------------------------------------------------------
# Public API: run_logic (live trading) & run_backtest
# --------------------------------------------------------------------------------------

def _latest_window_df(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    df = _fill_and_clean(df)
    return df.iloc[-LOOKBACK:].copy()

def _window_df_until(df: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    dfc = df.loc[df["timestamp"] <= ts].copy()
    dfc = _fill_and_clean(dfc)
    return dfc.iloc[-LOOKBACK:].copy()

# ===== Replace run_logic to use the countdown cache =====
def run_logic(current_price, predicted_price, ticker):
    """
    Live trading entry point.
    - Uses cached TCN from tcn-cache and retrains only when countdown hits 0 or cache invalid.
    """
    from forest import api, buy_shares, sell_shares

    logger = logging.getLogger(__name__)
    key = _ticker_key(ticker)

    try:
        account = api.get_account()
        cash = float(account.cash)
    except Exception as e:
        logger.error(f"[{key}] Error fetching account details: {e}")
        return

    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception:
        position_qty = 0.0

    try:
        df_raw = _load_csv(ticker)
        artifacts = _ensure_models_cached(ticker, up_to_ts=None, decrement=True)
        feats_used = artifacts.features

        win_df = _latest_window_df(df_raw, feats_used)
        if len(win_df) < LOOKBACK:
            logger.warning(f"[{key}] Not enough data for prediction (need {LOOKBACK}).")
            return

        p_up = _predict_p_up(artifacts.models, artifacts.scaler, feats_used, win_df, artifacts.temperature)
        action = decide_action(p_up, position_qty, artifacts.buy_threshold, artifacts.sell_threshold)

        logger.info(f"[{key}] p_up={p_up:.4f} | current_price={current_price} | pos={position_qty} | cash={cash} | action={action}")

        if action == "BUY":
            max_shares = int(cash // float(current_price))
            if max_shares > 0:
                logger.info(f"[{key}] Buying {max_shares} @ {current_price}")
                buy_shares(ticker, max_shares, current_price, p_up)
            else:
                logger.info(f"[{key}] Insufficient cash to buy.")
        elif action == "SELL":
            if position_qty > 0:
                logger.info(f"[{key}] Selling {position_qty} @ {current_price}")
                sell_shares(ticker, position_qty, current_price, p_up)
            else:
                logger.info(f"[{key}] No position to sell.")
        else:
            logger.info(f"[{key}] No action (hold).")

    except Exception as e:
        logger.exception(f"[{key}] run_logic failure: {e}")


# ===== Replace run_backtest to use the same countdown cache (walk-forward safe) =====
def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles, ticker):
    """
    Backtest entry point.
    - Walk-forward: trains only on data <= current_timestamp when countdown triggers.
    - Between retrains, reuses the cached model (faster).
    - Countdown shared with live (per ticker+tf) as requested.
    """
    logger = logging.getLogger(__name__)
    key = _ticker_key(ticker)

    try:
        df_full = _load_csv(ticker)
        if df_full.empty:
            return "NONE"

        ts = _safe_parse_ts(current_timestamp)

        # Cleaned subset for prediction window
        df_upto = _fill_and_clean(df_full.loc[df_full["timestamp"] <= ts].copy())
        if df_upto.empty or len(df_upto) < LOOKBACK:
            return "NONE"

        # Ensure cached models (may retrain if countdown expired)
        artifacts = _ensure_models_cached(ticker, up_to_ts=ts, decrement=True)

        # Predict on the tail window ending at ts
        win_df = df_upto.tail(LOOKBACK).copy()
        p_up = _predict_p_up(artifacts.models, artifacts.scaler, artifacts.features, win_df, artifacts.temperature)

        action = decide_action(p_up, position_qty, artifacts.buy_threshold, artifacts.sell_threshold)
        logger.info(f"[{key} BT] ts={ts} | p_up={p_up:.4f} | current_price={current_price} | pos={position_qty} | action={action}")
        return action

    except Exception as e:
        logger.exception(f"[{key}] run_backtest failure: {e}")
        return "NONE"
    

def _read_tickerlist(list_path: str = os.path.join(DATA_DIR, "tickerlist.txt")) -> List[str]:
    """
    Parse tickers from /data/tickerlist.txt.
    - Each line is a ticker, optionally followed by a comma and extra text (ignored).
    - Lines like 'selection_model=logic_20' are ignored.
    - Empty lines and comments ('# ...') are ignored.
    - Returns unique tickers in original order, uppercased.
    Example accepted lines:
        TSLA,catboost
        AMD
        BYND
        SOFI
        selection_model=logic_20   (ignored)
    """
    tickers: List[str] = []
    seen = set()

    try:
        with open(list_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                low = line.lower()
                if low.startswith("selection_model=") or "selection_model=" in low:
                    continue

                # Take only the token before the first comma
                token = line.split(",", 1)[0].strip()
                if not token:
                    continue

                t = token.upper()
                if t not in seen:
                    seen.add(t)
                    tickers.append(t)
    except FileNotFoundError:
        logger.warning(f"Ticker list not found: {list_path}")

    return tickers


def _cache_exists_for_ticker(ticker: str) -> bool:
    """
    A 'cache exists' means: all model files, scaler, and meta.json exist for this ticker.
    """
    expected_models = [_model_path(ticker, i) for i in range(ENSEMBLE_SIZE)]
    if not all(os.path.exists(p) for p in expected_models):
        return False
    if not os.path.exists(_scaler_path(ticker)):
        return False
    if not os.path.exists(_meta_path(ticker)):
        return False
    return True


def _countdown_remaining_for_ticker(ticker: str) -> int:
    """
    Read countdown.json and return the remaining counter (defaults to RETRAIN_EVERY if missing).
    """
    cd = _load_countdown(ticker)
    try:
        return int(cd.get("remaining", RETRAIN_EVERY))
    except Exception:
        return RETRAIN_EVERY


def _train_or_skip_ticker(ticker: str) -> str:
    """
    Decide to train/retrain or skip based on:
      - Always TRAIN if cache/model doesn't exist (cold start).
      - SKIP ONLY IF cache exists AND countdown remaining > 0.
      - Otherwise TRAIN/RETRAIN (countdown exhausted, or other triggers inside _ensure_models_cached).
    Uses _ensure_models_cached(..., decrement=False) so batch maintenance doesn't consume countdown.
    Returns a short status string for reporting.
    """
    key = _ticker_key(ticker)

    # Cold start: no cache -> must train
    if not _cache_exists_for_ticker(ticker):
        try:
            logger.info(f"[{key}] No cache found -> training...")
            _ensure_models_cached(ticker, up_to_ts=None, decrement=False)
            logger.info(f"[{key}] Train complete.")
            return "trained (cold start)"
        except Exception as e:
            logger.exception(f"[{key}] Training failed: {e}")
            return "error"

    # Cache exists: consult countdown
    remaining = _countdown_remaining_for_ticker(ticker)
    if remaining > 0:
        logger.info(f"[{key}] Countdown remaining={remaining} -> skip retrain.")
        return f"skipped (remaining={remaining})"

    # Countdown exhausted (or <=0): retrain (also handles csv/arch changes internally)
    try:
        logger.info(f"[{key}] Countdown exhausted -> retraining...")
        _ensure_models_cached(ticker, up_to_ts=None, decrement=False)
        logger.info(f"[{key}] Retrain complete.")
        return "retrained"
    except Exception as e:
        logger.exception(f"[{key}] Retrain failed: {e}")
        return "error"


def maintain_models_from_tickerlist(list_path: str = os.path.join(DATA_DIR, "tickerlist.txt")) -> Dict[str, str]:
    """
    Read tickers from the list and train/retrain as required.
    Returns a dict: {ticker: status}.
    """
    results: Dict[str, str] = {}
    tickers = _read_tickerlist(list_path)

    if not tickers:
        logger.info("No tickers found to process.")
        return results

    for t in tickers:
        # Ensure data CSV exists before attempting anything
        csv_path = get_csv_filename(t)
        if not os.path.exists(csv_path):
            logger.warning(f"[{_ticker_key(t)}] CSV not found -> skipping: {csv_path}")
            results[t] = "skipped (no CSV)"
            continue

        status = _train_or_skip_ticker(t)
        results[t] = status

    # Nice one-line summary in logs
    try:
        summary = ", ".join([f"{k}:{v}" for k, v in results.items()])
        logger.info(f"[maintenance] {summary}")
    except Exception:
        pass

    return results


# --------------------------------------------------------------------------------------
# Run maintenance automatically when executed as a script
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Optional: ensure basic logging config if not set by the host app
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    maintain_models_from_tickerlist()
