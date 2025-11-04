# tcn_small.py
# ---------------------------------------------------------------------
# Temporal CNN (TCN) for short-horizon returns (h=1 by default)
# Channels (example): ['returns_1','returns_3','returns_5','rsi','macd_line','macd_signal',
#                      'macd_histogram','ema_slope_9','ema_slope_21','ema200_dist',
#                      'bollinger_percB','rv_20','volume_zscore','vwap_dev',
#                      'candle_body_ratio','wick_dominance']
#
# Outputs: y_pred (bps), signal [-1,1], pred_std (MC-dropout or EWMA proxy), confidence
# ---------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable, Tuple
import io
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ------------------------
# Shared result container
# ------------------------
@dataclass
class SubmodelResult:
    # Core
    y_pred: pd.Series                   # expected return over `horizon` (bps)
    signal: pd.Series                   # normalized signal in [-1, 1]

    # Optional
    proba_up: Optional[pd.Series] = None
    pred_std: Optional[pd.Series] = None
    confidence: Optional[pd.Series] = None
    costs: Optional[pd.Series] = None
    trade_mask: Optional[pd.Series] = None

    # Live diagnostics
    live_metrics: Optional[Dict[str, float]] = None

    # Explainability / bookkeeping
    feature_importance: Optional[pd.Series] = None
    used_features: Optional[List[str]] = None
    warmup_bars: int = 0
    model_name: str = ""
    params: Optional[Dict[str, Any]] = None
    state: Optional[Any] = None


# ===========================================================
# Small TCN building blocks (causal, dilated, residual)
# ===========================================================
class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        # left-pad only for causality
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.c1 = CausalConv1d(ch, ch, kernel_size, dilation)
        self.c2 = CausalConv1d(ch, ch, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)
        self.norm = nn.LayerNorm(ch)  # time-independent normalization

    def forward(self, x):
        # x: [B, C, T]
        y = self.c1(x)
        y = self.act(y)
        y = self.dropout(y)
        y = self.c2(y)
        y = self.act(y)
        y = self.dropout(y)
        # residual + channel-wise norm (transpose to [B,T,C] for LayerNorm)
        y = y + x
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        return y


class SmallTCN(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int = 32, n_blocks: int = 3, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Conv1d(in_ch, hidden_ch, kernel_size=1)
        blocks = []
        for b in range(n_blocks):
            dilation = 2 ** b
            blocks.append(TCNBlock(hidden_ch, kernel_size, dilation, dropout))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_ch, 1)  # reads last timestep features

    def forward(self, x):
        # x: [B, C, T]
        z = self.in_proj(x)
        z = self.blocks(z)
        last = z[:, :, -1]           # [B, C]
        out = self.head(last)        # [B, 1]
        return out.squeeze(-1)       # [B]


# ===========================================================
# Data utilities
# ===========================================================
class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X  # [N, C, W]
        self.y = y  # [N]

    def __len__(self): 
        return len(self.y)

    def __getitem__(self, i):
        xb = torch.from_numpy(self.X[i]).float()                 # [C, W]
        yb = torch.tensor(self.y[i], dtype=torch.float32)        # scalar -> tensor
        return xb, yb


def _make_windows(
    df_feat: pd.DataFrame,
    target_bps: pd.Series,
    features: List[str],
    window: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
    """
    Build causal windows (use t-window..t-1 to predict t..t+h-1 aggregated).
    Returns (X[C,W], y, index_of_prediction_time).
    """
    Xdf = df_feat[features].copy()
    y = target_bps.reindex(Xdf.index).astype(float)
    # valid range where we have a full window and a target
    idx = Xdf.index
    start = window
    end = len(idx) - (horizon - 1)
    if end <= start:
        return np.zeros((0, len(features), window), dtype=np.float32), np.zeros((0,), dtype=np.float32), idx[:0]
    rows = []
    ys = []
    out_idx = []
    Xvals = Xdf.values.astype(np.float32)
    yvals = y.values.astype(np.float32)
    for t in range(start, end):
        rows.append(Xvals[t - window:t].T)    # [C, W]
        ys.append(yvals[t + horizon - 1])     # predict return ending at (t + h - 1)
        out_idx.append(idx[t])
    X = np.stack(rows, axis=0)
    y_arr = np.asarray(ys, dtype=np.float32)
    out_index = pd.Index(out_idx, name=Xdf.index.name)
    return X, y_arr, out_index


# ===========================================================
# Training helper (builds `state` used by predict_submodel)
# ===========================================================
def fit_tcn_state(
    df_train: pd.DataFrame,
    target_bps: pd.Series,
    *,
    features: List[str],
    horizon: int = 1,
    window: int = 128,
    hidden_ch: int = 32,
    n_blocks: int = 3,
    kernel_size: int = 5,
    dropout: float = 0.1,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    clip_grad: float = 1.0,
    residual_ewm_alpha: float = 0.08,
    model_name: str = "tcn_small",
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Fit a small TCN on rolling windows. Returns a serializable `state`.
    """
    # Align & drop NA
    Xdf = df_train[features].copy()
    mask = Xdf.notna().all(axis=1) & target_bps.notna()
    Xdf = Xdf.loc[mask]
    y = target_bps.loc[mask]

    # Channel-wise standardization (fit on *train*)
    mu = Xdf.mean().astype(float)
    sd = Xdf.std(ddof=0).replace(0, 1.0).astype(float)
    Z = (Xdf - mu) / sd

    # Windows
    X, y_arr, out_idx = _make_windows(Z, y, features, window, horizon)
    if len(X) == 0:
        raise ValueError("Not enough data to form windows; reduce `window` or check inputs.")

    # Torch objects
    ds = WindowDataset(X, y_arr)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    torch.manual_seed(42)
    model = SmallTCN(in_ch=len(features), hidden_ch=hidden_ch, n_blocks=n_blocks, kernel_size=kernel_size, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb = xb.to(device)         # [B,C,W]
            yb = yb.to(device)         # [B]
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()

    # Serialize weights to bytes
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    model_bytes = buf.getvalue()

    # Simple per-channel importance from first conv weights
    with torch.no_grad():
        W = model.in_proj.weight.detach().cpu().numpy()  # [hidden_ch, C, 1]
        ch_importance = np.abs(W).sum(axis=(0, 2))
        ch_importance = ch_importance / (ch_importance.sum() + 1e-12)
        fi = {f: float(w) for f, w in zip(features, ch_importance)}

    state: Dict[str, Any] = {
        "model_name": model_name,
        "feature_order": list(features),
        "mu": mu.to_dict(),
        "sd": sd.to_dict(),
        "window": int(window),
        "horizon": int(horizon),
        "hidden_ch": int(hidden_ch),
        "n_blocks": int(n_blocks),
        "kernel_size": int(kernel_size),
        "dropout": float(dropout),
        "model_bytes": model_bytes,
        "residual_ewm_alpha": float(residual_ewm_alpha),
        "feature_importance": fi,
        "device": device,
    }
    return state


# ---------------------------------------------
# Small utilities
# ---------------------------------------------
def _tanh_signal(y_pred_bps: pd.Series, k_bps: float) -> pd.Series:
    return np.tanh(y_pred_bps / float(k_bps))

def _confidence_from_std(pred_std: pd.Series) -> pd.Series:
    if pred_std.isna().all():
        return pd.Series(index=pred_std.index, dtype=float)
    m = pred_std.median()
    mad = (pred_std - m).abs().median()
    z = (pred_std - m) / (mad + 1e-9)
    conf = 1.0 / (1.0 + z.clip(lower=0))
    return conf.clip(0, 1)


# ===========================================================
# Required submodel API (inference only; no targets read)
# ===========================================================
def predict_submodel(
    df: pd.DataFrame,
    *,
    horizon: int,
    features: List[str],
    as_of: Optional[pd.Timestamp] = None,
    state: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    cost_model: Optional[Callable[[pd.Series], pd.Series]] = None,
) -> SubmodelResult:
    """
    Temporal CNN (TCN). Inference-only.

    Params (optional):
      - clip_k_bps: float (default 60.0) -> tanh scaling for signal
      - mc_passes: int (default 16)      -> MC-dropout forward passes for pred_std
      - min_abs_edge_bps: float|None     -> optional gating (not typical here)
    """
    if state is None:
        raise ValueError(
            "TCN requires a fitted `state`. Fit with fit_tcn_state(...) and pass it here."
        )

    params = params or {}
    clip_k_bps = float(params.get("clip_k_bps", 60.0))
    mc_passes = int(params.get("mc_passes", 16))
    min_abs_edge_bps = params.get("min_abs_edge_bps", None)

    # Slice & checks
    Xfull = df if as_of is None else df.loc[:as_of]
    trained_order = state.get("feature_order", features)
    missing = [f for f in trained_order if f not in Xfull.columns]
    if missing:
        raise KeyError(f"Missing required features for TCN: {missing}")
    Xdf = Xfull[trained_order].copy()

    # Standardize with stored stats
    mu = pd.Series(state["mu"])
    sd = pd.Series(state["sd"]).replace(0, 1.0)
    Z = (Xdf - mu) / sd

    window = int(state["window"])
    h_fit = int(state.get("horizon", 1))
    if horizon != h_fit:
        # You can still use the model, but the target mapping was trained for h_fit
        pass

    # Windows (no target required)
    X, _, out_idx = _make_windows(Z, pd.Series(index=Z.index, dtype=float), trained_order, window, 1)
    if len(X) == 0:
        # No windows -> empty result aligned to index
        empty = pd.Series(index=Z.index, dtype=float)
        return SubmodelResult(
            y_pred=empty, signal=empty, pred_std=empty, confidence=empty,
            feature_importance=pd.Series(state.get("feature_importance", {})),
            used_features=trained_order, warmup_bars=window, model_name=state.get("model_name", "tcn_small"),
            params={"horizon": horizon, "clip_k_bps": clip_k_bps, "mc_passes": mc_passes}, state=state
        )

    # Restore model
    device = state.get("device", "cpu")
    model = SmallTCN(
        in_ch=len(trained_order),
        hidden_ch=int(state["hidden_ch"]),
        n_blocks=int(state["n_blocks"]),
        kernel_size=int(state["kernel_size"]),
        dropout=float(state["dropout"]),
    ).to(device)
    buf = io.BytesIO(state["model_bytes"])
    model.load_state_dict(torch.load(buf, map_location=device))
    model.train()  # keep dropout ON for MC predictions; no BatchNorm present

    # Tensorize inputs
    X_tensor = torch.from_numpy(X).float().to(device)  # [N,C,W]

    # MC-dropout predictions
    with torch.no_grad():
        preds = []
        for _ in range(max(1, mc_passes)):
            y = model(X_tensor)              # [N]
            preds.append(y.detach().cpu().numpy())
        P = np.stack(preds, axis=0)          # [K,N]
    mean_pred = P.mean(axis=0)
    std_pred = P.std(axis=0)

    # Map to Series aligned to out_idx (windows end time)
    y_pred = pd.Series(mean_pred, index=out_idx, name=f"tcn_h{horizon}_bps")
    pred_std = pd.Series(std_pred, index=out_idx, name="tcn_pred_std")

    # Fallback uncertainty if mc_passes==1: EWMA of deviations
    if mc_passes <= 1:
        alpha = float(state.get("residual_ewm_alpha", 0.08))
        pred_std = (y_pred - y_pred.ewm(alpha=alpha, adjust=False).mean()).abs().ewm(alpha=alpha, adjust=False).mean()
        pred_std.name = "tcn_pred_std"

    # Signal
    signal = pd.Series(np.tanh(y_pred.values / clip_k_bps), index=y_pred.index, name="tcn_signal")

    # Confidence
    confidence = _confidence_from_std(pred_std)

    # Optional costs passthrough
    costs = None
    if callable(cost_model):
        try:
            costs = cost_model(signal).reindex(y_pred.index)
        except Exception:
            costs = None

    # Optional gating
    trade_mask = None
    if min_abs_edge_bps is not None:
        trade_mask = (y_pred.abs() >= float(min_abs_edge_bps))
        trade_mask.name = "tcn_trade_ok"

    # Feature importance from training (channel importance)
    fi_dict = state.get("feature_importance", {})
    fi = pd.Series({f: float(fi_dict.get(f, 0.0)) for f in trained_order}, name="input_channel_importance")

    return SubmodelResult(
        y_pred=y_pred,
        signal=signal,
        proba_up=None,
        pred_std=pred_std,
        confidence=confidence,
        costs=costs,
        trade_mask=trade_mask,
        live_metrics=None,
        feature_importance=fi,
        used_features=trained_order,
        warmup_bars=window,
        model_name=str(state.get("model_name", "tcn_small")),
        params={
            "horizon": horizon,
            "clip_k_bps": clip_k_bps,
            "mc_passes": mc_passes,
            "window": window,
        },
        state=state,
    )
