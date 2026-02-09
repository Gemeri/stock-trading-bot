# tcn.py
"""
Temporal Convolutional Network (TCN) binary classifier for trading execution timing.

Goal:
- Predict whether to EXECUTE now (label 1) or WAIT (label 0).
- "Execute" can mean BUY or SELL, chosen by the main script; this model only predicts execute-vs-wait.

Data assumptions:
- Time-ordered 4H candles (2–3 candles/day).
- X can be:
    (A) 2D: [n_samples, n_features]  -> this module will build rolling sequences of length LOOKBACK
    (B) 3D: [n_samples, seq_len, n_features] -> treated as already-sequenced

Half-life weighting:
- Optional exponential time-decay weights favoring more recent samples.
- If half_life is None, no time-decay weighting is applied.

Public API:
- fit(x_train, y_train, half_life=None) -> trained model bundle
- predict(model, X) -> np.ndarray of probabilities P(label==1)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Tuple, List

import numpy as np
import pandas as pd

# --- Torch imports kept local to this module ---
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# User-tunable defaults
# =========================
LOOKBACK: int = 32               # rolling window length for 2D inputs
BATCH_SIZE: int = 256
EPOCHS: int = 40
LR: float = 1e-3
WEIGHT_DECAY: float = 1e-4
DROPOUT: float = 0.15
PATIENCE: int = 6                # early stopping patience (epochs)
VAL_FRACTION: float = 0.12       # last % as validation (time-based)
GRAD_CLIP_NORM: float = 1.0

# TCN channel sizes; adjust if you want larger/smaller model
TCN_CHANNELS: Tuple[int, ...] = (64, 64, 64, 64)
KERNEL_SIZE: int = 3


# =========================
# Utilities
# =========================
def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic-ish (may reduce speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _to_numpy_2d_or_3d(x: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    if isinstance(x, pd.DataFrame):
        x = x.copy()
        # coerce numeric, preserve column order
        x = x.apply(pd.to_numeric, errors="coerce")
        x = x.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        # final safety
        x = x.fillna(0.0)
        arr = x.to_numpy(dtype=np.float32)
        return arr
    else:
        arr = np.asarray(x)
        if arr.ndim not in (2, 3):
            raise ValueError(f"X must be 2D or 3D, got shape {arr.shape}")
        arr = arr.astype(np.float32, copy=False)
        # sanitize
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr


def _to_numpy_y(y: Union[pd.Series, np.ndarray, list]) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    # sanitize to 0/1 ints
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = (arr > 0.5).astype(np.int64)
    return arr


def _make_sequences_from_2d(X2: np.ndarray, y: Optional[np.ndarray], lookback: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convert 2D features [T, F] to 3D sequences [N, L, F] using a rolling window.
    The i-th sequence ends at time t = (lookback-1 + i).
    If y is provided (length T), it will be aligned to the sequence endpoints:
        y_seq[i] = y[lookback-1 + i]
    """
    if X2.ndim != 2:
        raise ValueError("_make_sequences_from_2d expects 2D X")

    T, F = X2.shape
    if T <= 0 or F <= 0:
        raise ValueError(f"Invalid X shape: {X2.shape}")

    if lookback < 2:
        raise ValueError("lookback must be >= 2")

    if T < lookback:
        # pad on the left by repeating the first row
        pad = np.repeat(X2[:1], repeats=(lookback - T), axis=0)
        X2p = np.concatenate([pad, X2], axis=0)
        T = X2p.shape[0]
        X2 = X2p

        if y is not None:
            # pad y with zeros on the left to match (conservative)
            ypad = np.zeros((lookback - len(y),), dtype=np.int64) if len(y) < lookback else np.zeros((0,), dtype=np.int64)
            y = np.concatenate([ypad, y], axis=0)

    N = T - lookback + 1
    X3 = np.zeros((N, lookback, F), dtype=np.float32)
    for i in range(N):
        X3[i] = X2[i : i + lookback]

    y_seq = None
    if y is not None:
        if len(y) != T:
            raise ValueError(f"y length {len(y)} does not match X length {T}")
        y_seq = y[lookback - 1 : lookback - 1 + N].astype(np.int64, copy=False)

    return X3, y_seq


def _standardize_fit(X3: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit standardization on training data and return standardized X and (mean, std).
    Standardize per-feature across all timesteps and samples.
    """
    # flatten across samples and time
    flat = X3.reshape(-1, X3.shape[-1]).astype(np.float32, copy=False)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    Xn = (X3 - mean) / std
    return Xn.astype(np.float32, copy=False), mean.astype(np.float32), std.astype(np.float32)


def _standardize_apply(X3: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    Xn = (X3 - mean) / std
    return Xn.astype(np.float32, copy=False)


def _time_decay_weights(n: int, half_life: int) -> np.ndarray:
    """
    Most recent sample has weight 1.0; older samples decay by half every 'half_life' samples.
    w[i] = 0.5^((n-1-i)/half_life)
    """
    if half_life <= 0:
        raise ValueError("half_life must be a positive integer")
    idx = np.arange(n, dtype=np.float32)
    age = (n - 1) - idx
    w = np.power(0.5, age / float(half_life)).astype(np.float32)
    return w


# =========================
# Model: TCN
# =========================
class Chomp1d(nn.Module):
    """Remove right-side padding to keep causality."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size <= 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation  # causal padding to the left (implemented via right pad + chomp)
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.final_act = nn.ReLU()

        # init
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.act2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.final_act(out + res)


class TCNClassifier(nn.Module):
    def __init__(
        self,
        num_features: int,
        channels: Tuple[int, ...] = TCN_CHANNELS,
        kernel_size: int = KERNEL_SIZE,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = num_features
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_ch = out_ch

        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(in_ch, 1)
        nn.init.xavier_uniform_(self.head.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, F]
        returns logits: [B]
        """
        # Conv1d expects [B, C, L] where C=features
        x = x.transpose(1, 2)  # [B, F, L]
        z = self.tcn(x)        # [B, C, L]
        last = z[:, :, -1]     # last timestep features [B, C]
        logit = self.head(last).squeeze(-1)  # [B]
        return logit


# =========================
# Dataset / Bundle
# =========================
class _SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None):
        self.X = torch.from_numpy(X)  # float32
        self.y = torch.from_numpy(y.astype(np.float32, copy=False))
        self.w = None if w is None else torch.from_numpy(w.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        if self.w is None:
            return self.X[idx], self.y[idx]
        return self.X[idx], self.y[idx], self.w[idx]


@dataclass
class TCNBundle:
    net: nn.Module
    mean: np.ndarray
    std: np.ndarray
    lookback: int
    feature_names: Optional[List[str]]
    device: str


# =========================
# Public API
# =========================
def fit(
    x_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray, list],
    half_life: Optional[int] = None,
):
    """
    Train a TCN binary classifier.

    Parameters
    ----------
    x_train:
        DataFrame or ndarray.
        - If 2D: shape [T, F], sequences are built internally using LOOKBACK.
        - If 3D: shape [N, L, F], used directly (L can differ from LOOKBACK).
    y_train:
        1D labels aligned to x_train time dimension.
        Must be 0/1 (or coercible to 0/1).
    half_life:
        Optional int. If set, applies exponential time-decay weights favoring newer samples.

    Returns
    -------
    model bundle (TCNBundle)
    """
    _set_seed(42)

    feature_names: Optional[List[str]] = None
    if isinstance(x_train, pd.DataFrame):
        feature_names = list(x_train.columns)

    X = _to_numpy_2d_or_3d(x_train)
    y = _to_numpy_y(y_train)

    # Convert to sequences
    if X.ndim == 2:
        X3, y_seq = _make_sequences_from_2d(X, y, lookback=LOOKBACK)
        lookback = LOOKBACK
        y_use = y_seq
    else:
        # X is [N, L, F]; y should be [N] (aligned)
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"For 3D X, y length must match X[0]. Got y={y.shape[0]}, X={X.shape[0]}")
        X3 = X
        lookback = int(X3.shape[1])
        y_use = y

    if y_use is None:
        raise RuntimeError("y alignment failed")

    n = X3.shape[0]
    if n < 200:
        raise ValueError(f"Not enough training sequences after lookback alignment: {n} samples")

    # Standardize
    X3n, mean, std = _standardize_fit(X3)

    # Time-based split (last VAL_FRACTION is validation)
    n_val = max(64, int(n * VAL_FRACTION))
    n_train = n - n_val
    X_tr, y_tr = X3n[:n_train], y_use[:n_train]
    X_va, y_va = X3n[n_train:], y_use[n_train:]

    # Optional time-decay weights (favor recent within TRAIN ONLY)
    w_tr = None
    if half_life is not None:
        if not isinstance(half_life, (int, np.integer)):
            raise ValueError("half_life must be an int or None")
        w_tr = _time_decay_weights(len(y_tr), int(half_life))

    # Class imbalance handling via pos_weight (bounded)
    pos = float(np.sum(y_tr == 1))
    neg = float(np.sum(y_tr == 0))
    if pos <= 1.0:
        raise ValueError("Training labels contain too few positives (label==1) to train a classifier.")
    raw_pos_weight = neg / pos
    pos_weight = float(np.clip(raw_pos_weight, 1.0, 10.0))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = TCNClassifier(
        num_features=int(X3n.shape[-1]),
        channels=TCN_CHANNELS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
    ).to(device)

    # Loss returns per-sample so we can apply time-decay weights if present
    bce = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device),
    )
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    ds_tr = _SeqDataset(X_tr, y_tr, w=w_tr)
    ds_va = _SeqDataset(X_va, y_va, w=None)

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    best_val = float("inf")
    best_state = None
    bad = 0

    for epoch in range(1, EPOCHS + 1):
        net.train()
        tr_loss_sum = 0.0
        tr_count = 0

        for batch in dl_tr:
            if len(batch) == 2:
                xb, yb = batch
                wb = None
            else:
                xb, yb, wb = batch

            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            logits = net(xb)

            loss_vec = bce(logits, yb)
            if wb is not None:
                wb = wb.to(device)
                loss_vec = loss_vec * wb

            loss = loss_vec.mean()
            loss.backward()

            if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP_NORM)

            opt.step()

            bs = int(xb.shape[0])
            tr_loss_sum += float(loss.item()) * bs
            tr_count += bs

        tr_loss = tr_loss_sum / max(1, tr_count)

        # Validation
        net.eval()
        va_loss_sum = 0.0
        va_count = 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = net(xb)
                loss_vec = bce(logits, yb)
                loss = loss_vec.mean()
                bs = int(xb.shape[0])
                va_loss_sum += float(loss.item()) * bs
                va_count += bs

        va_loss = va_loss_sum / max(1, va_count)

        # Early stopping
        if va_loss + 1e-6 < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

        # (No prints by default; keep module quiet for drop-in use.)

    if best_state is not None:
        net.load_state_dict(best_state)

    bundle = TCNBundle(
        net=net,
        mean=mean,
        std=std,
        lookback=lookback,
        feature_names=feature_names,
        device=device,
    )
    return bundle


def predict(model, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """
    Predict probabilities P(execute_now == 1).

    Input formats:
    - If X is 2D [T, F], rolling sequences of length model.lookback are created.
      Output length will be N = max(1, T - lookback + 1).
      (If T < lookback, left-padding is applied so output length = 1.)
    - If X is 3D [N, L, F], used directly. If L < lookback, left-padding is applied per-sample.

    Returns:
    - np.ndarray of shape [N] with float probabilities in [0, 1].
    """
    if not isinstance(model, TCNBundle):
        raise TypeError("model must be the object returned by fit() (TCNBundle)")

    # If DataFrame, enforce training feature order when available
    if isinstance(X, pd.DataFrame) and model.feature_names is not None:
        missing = [c for c in model.feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"Predict DataFrame missing columns: {missing}")
        X = X[model.feature_names]

    Xn = _to_numpy_2d_or_3d(X)

    # Build / pad sequences
    if Xn.ndim == 2:
        X3, _ = _make_sequences_from_2d(Xn, y=None, lookback=model.lookback)
    else:
        # X3 is [N, L, F]
        X3 = Xn
        N, L, F = X3.shape
        if L < model.lookback:
            # left-pad each sample by repeating first timestep (conservative)
            pad_len = model.lookback - L
            pad = np.repeat(X3[:, :1, :], repeats=pad_len, axis=1)
            X3 = np.concatenate([pad, X3], axis=1).astype(np.float32, copy=False)

    # Standardize with training stats
    X3s = _standardize_apply(X3, model.mean, model.std)

    net = model.net
    device = model.device
    net.eval()

    probs: List[float] = []
    with torch.no_grad():
        xb = torch.from_numpy(X3s).to(device)
        # chunk to avoid GPU/CPU memory spikes
        chunk = 4096
        for i in range(0, xb.shape[0], chunk):
            logits = net(xb[i : i + chunk])
            p = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64, copy=False)
            probs.append(p)

    proba = np.concatenate(probs, axis=0) if probs else np.zeros((0,), dtype=np.float64)
    return np.asarray(proba, dtype=float)


def save(model: TCNBundle, path: Union[str, Path]) -> None:
    data = {
        "state_dict": model.net.state_dict(),
        "mean": model.mean,
        "std": model.std,
        "lookback": model.lookback,
        "feature_names": model.feature_names,
        "channels": TCN_CHANNELS,
        "kernel_size": KERNEL_SIZE,
        "dropout": DROPOUT,
    }
    torch.save(data, path)


def load(path: Union[str, Path]) -> TCNBundle:
    data = torch.load(path, map_location="cpu")
    mean = np.asarray(data["mean"], dtype=np.float32)
    std = np.asarray(data["std"], dtype=np.float32)
    num_features = int(mean.shape[0])
    channels = tuple(data.get("channels", TCN_CHANNELS))
    kernel_size = int(data.get("kernel_size", KERNEL_SIZE))
    dropout = float(data.get("dropout", DROPOUT))

    net = TCNClassifier(
        num_features=num_features,
        channels=channels,
        kernel_size=kernel_size,
        dropout=dropout,
    )
    net.load_state_dict(data["state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(device)

    return TCNBundle(
        net=net,
        mean=mean,
        std=std,
        lookback=int(data["lookback"]),
        feature_names=data.get("feature_names"),
        device=device,
    )
