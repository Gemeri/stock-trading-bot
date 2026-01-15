from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class LSTMModelBundle:
    model: nn.Module
    seq_len: int
    feature_mean: np.ndarray
    feature_std: np.ndarray
    device: str


class _LSTMClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_classes: int = 3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.lstm(x)          # out: (B, T, H)
        last = out[:, -1, :]           # (B, H)
        logits = self.head(last)       # (B, C)
        return logits


def _as_2d_float_array(x: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    if isinstance(x, pd.DataFrame):
        arr = x.to_numpy()
    elif isinstance(x, np.ndarray):
        arr = x
    else:
        raise TypeError("x_train must be a pandas DataFrame or a 2D numpy array")

    if arr.ndim != 2:
        raise ValueError(f"x_train must be 2D, got shape {arr.shape}")

    arr = np.asarray(arr, dtype=np.float32)
    return arr


def _encode_labels(y_train: Union[pd.Series, np.ndarray, list]) -> np.ndarray:
    y_arr = np.asarray(y_train).reshape(-1)
    if len(y_arr) == 0:
        raise ValueError("y_train is empty")

    # Label mapping (-1/0/1 -> 0/1/2). If already 0/1/2, leave as-is.
    label_mapping = {-1: 0, 0: 1, 1: 2}
    unique_vals = set(np.unique(y_arr).tolist())
    if unique_vals.issubset({-1, 0, 1}):
        y_encoded = np.vectorize(label_mapping.get)(y_arr).astype(np.int64)
    else:
        y_encoded = y_arr.astype(np.int64)

    # Basic sanity: expecting classes 0..2
    if np.min(y_encoded) < 0 or np.max(y_encoded) > 2:
        raise ValueError(f"Expected labels in {{0,1,2}} after encoding, got range [{y_encoded.min()},{y_encoded.max()}]")
    return y_encoded


def _half_life_weights(n: int, half_life: Optional[int]) -> Optional[np.ndarray]:
    if half_life is None:
        return None
    hl = max(1, int(half_life))
    ages = np.arange(n - 1, -1, -1, dtype=np.float32)  # oldest largest, newest 0
    w = np.exp(-np.log(2.0) * ages / float(hl)).astype(np.float32)
    return w


def _make_sequences(
    X2: np.ndarray,
    y: Optional[np.ndarray],
    w: Optional[np.ndarray],
    seq_len: int,
):
    """
    If seq_len==1: returns (N,1,F).
    If seq_len>1: builds sliding windows; target is y[t] for the window ending at t.
    """
    n, f = X2.shape
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if seq_len == 1:
        X3 = X2.reshape(n, 1, f)
        return X3, y, w

    if n < seq_len:
        raise ValueError(f"Not enough rows ({n}) to build seq_len={seq_len} windows")

    m = n - seq_len + 1
    X3 = np.empty((m, seq_len, f), dtype=np.float32)
    for i in range(m):
        X3[i] = X2[i : i + seq_len]

    y_out = None if y is None else y[seq_len - 1 :]
    w_out = None if w is None else w[seq_len - 1 :]
    return X3, y_out, w_out


def fit(
    x_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray, list],
    half_life: Optional[int] = None,
):
    seq_len = 1
    hidden_size = 64
    num_layers = 2
    dropout = 0.1
    epochs = 30
    batch_size = 256
    lr = 1e-3
    val_ratio = 0.1
    patience = 5
    random_seed = 42

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    X2 = _as_2d_float_array(x_train)
    y = _encode_labels(y_train)

    n = len(y)
    if len(X2) != n:
        raise ValueError(f"x_train and y_train length mismatch: {len(X2)} vs {n}")

    w = _half_life_weights(n, half_life)

    # Standardize features (fit on full training input)
    mean = X2.mean(axis=0, dtype=np.float64)
    std = X2.std(axis=0, dtype=np.float64)
    std[std == 0.0] = 1.0
    X2s = ((X2 - mean) / std).astype(np.float32)

    # Build sequences
    X3, y_seq, w_seq = _make_sequences(X2s, y, w, seq_len)

    # Split train/val (use the most recent tail as validation, typical for time series)
    m = len(X3)
    val_n = int(round(m * float(val_ratio)))
    val_n = max(1, val_n) if m >= 2 else 0
    train_n = m - val_n

    if val_n == 0:
        X_tr, y_tr, w_tr = X3, y_seq, w_seq
        X_va, y_va, w_va = None, None, None
    else:
        X_tr, y_tr = X3[:train_n], y_seq[:train_n]
        X_va, y_va = X3[train_n:], y_seq[train_n:]
        w_tr = None if w_seq is None else w_seq[:train_n]
        w_va = None if w_seq is None else w_seq[train_n:]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = _LSTMClassifier(
        n_features=X3.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        n_classes=3,
    ).to(device)

    # DataLoaders
    def make_loader(Xa, ya, wa, shuffle: bool):
        Xt = torch.from_numpy(Xa)
        yt = torch.from_numpy(ya.astype(np.int64))
        if wa is None:
            wt = torch.ones(len(ya), dtype=torch.float32)
        else:
            wt = torch.from_numpy(wa.astype(np.float32))
        ds = TensorDataset(Xt, yt, wt)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    train_loader = make_loader(X_tr, y_tr, w_tr, shuffle=True)
    val_loader = None if X_va is None else make_loader(X_va, y_va, w_va, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for _epoch in range(int(epochs)):
        model.train()
        for xb, yb, wb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)

            logits = model(xb)
            # per-sample CE loss, then apply weights
            loss_per = F.cross_entropy(logits, yb, reduction="none")
            loss = (loss_per * wb).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        if val_loader is None:
            continue

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb, wb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                wb = wb.to(device)
                logits = model(xb)
                loss_per = F.cross_entropy(logits, yb, reduction="none")
                val_loss = (loss_per * wb).mean().item()
                val_losses.append(val_loss)

        cur_val = float(np.mean(val_losses)) if val_losses else float("inf")
        if cur_val + 1e-8 < best_val:
            best_val = cur_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    bundle = LSTMModelBundle(
        model=model,
        seq_len=int(seq_len),
        feature_mean=mean.astype(np.float32),
        feature_std=std.astype(np.float32),
        device=device,
    )
    return bundle


@torch.no_grad()
def predict(model: LSTMModelBundle, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """
    Returns probability of class index 1 (same convention as the CatBoost version's [:, 1]).
    If seq_len > 1, predictions align to the end of each window, so output length is:
      - N if seq_len == 1
      - N - seq_len + 1 if seq_len > 1
    """
    X2 = _as_2d_float_array(X)

    # Standardize using training stats
    mean = model.feature_mean
    std = model.feature_std
    X2s = ((X2 - mean) / std).astype(np.float32)

    # Sequence-ify
    X3, _, _ = _make_sequences(X2s, None, None, model.seq_len)

    net = model.model
    net.eval()

    device = model.device
    xb = torch.from_numpy(X3).to(device)

    logits = net(xb)
    probs = torch.softmax(logits, dim=-1)  # (Nseq, 3)
    proba_class_1 = probs[:, 1].detach().cpu().numpy().astype(float)
    return proba_class_1
