"""
RBF SVR price prediction logic (production-ready, model-only API)

- Trains on ONLY the 'close' series using sliding-window lags
- Uses non-linear SVR with RBF kernel
- No look-ahead:
    For target index t (t >= lookback):
      X[t] = [close[t-lookback], ..., close[t-1]]
      y[t] = close[t]
- Public API:
    fit(X, Y=None) -> ModelBundle
    predict(model_bundle, X) -> np.ndarray of percent growths
      * Output is aligned to input length N.
      * Entries [0 : lookback-1] are np.nan (insufficient history).
      * For index t >= lookback, we predict close[t] and then return:
            100 * (pred_close[t] - close[t-1]) / close[t-1]
        (So +1.0 means +1%, not 0.01.)

Env vars:
    SVR_LOOKBACK (default 30)
    SVR_MIN_TRAIN_SIZE (default 300)
    SVR_EPSILON (default 0.05)
    SVR_C (default 10.0)
    SVR_GAMMA (default "scale")
    SVR_ENABLE_CV (default "false")
    SVR_CV_SPLITS (default 5)
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# --------------------------------------------------------------------------------------
# Hyperparameters (overridable via environment)
# --------------------------------------------------------------------------------------

SVR_LOOKBACK = int(os.getenv("SVR_LOOKBACK", "30"))                 # number of lag closes per sample
SVR_MIN_TRAIN_SIZE = int(os.getenv("SVR_MIN_TRAIN_SIZE", "300"))    # minimum rows required to fit
SVR_EPSILON = float(os.getenv("SVR_EPSILON", "0.05"))
SVR_C = float(os.getenv("SVR_C", "10.0"))
SVR_GAMMA = os.getenv("SVR_GAMMA", "scale")                         # 'scale' or 'auto' or float string

# Cross-validation toggles (kept lightweight)
SVR_ENABLE_CV = os.getenv("SVR_ENABLE_CV", "false").lower() in {"1", "true", "yes"}
SVR_CV_SPLITS = int(os.getenv("SVR_CV_SPLITS", "5"))

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Internal structures
# --------------------------------------------------------------------------------------

@dataclass
class ModelBundle:
    """
    Holds the trained regressor and scalers.
    """
    model: object                 # SVR or NaiveModel
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    lookback: int


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def _extract_close_series(X: Union[pd.DataFrame, pd.Series, np.ndarray, list, tuple]) -> pd.Series:
    """
    Extracts a float Series of 'close' values from various X formats.
    - If DataFrame, prefer column 'close' (case-insensitive). If single column, use it.
    - If Series/array-like, treat as closes.
    Missing/inf values are forward/back-filled to keep windows valid.
    """
    if isinstance(X, pd.DataFrame):
        cols_lower = {c.lower(): c for c in X.columns}
        if "close" in cols_lower:
            s = pd.to_numeric(X[cols_lower["close"]], errors="coerce")
        elif X.shape[1] == 1:
            s = pd.to_numeric(X.iloc[:, 0], errors="coerce")
        else:
            raise ValueError("svr.py expects X to include a 'close' column or be 1D closes.")
    elif isinstance(X, pd.Series):
        s = pd.to_numeric(X, errors="coerce")
    else:
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            s = pd.Series(arr)
        elif arr.ndim == 2 and arr.shape[1] >= 1:
            s = pd.Series(arr[:, 0])
        else:
            raise ValueError("Unsupported X shape for svr.py; provide closes as 1D or first column.")
    s = s.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    if s.isna().any():
        s = s.fillna(0.0)
    return s.reset_index(drop=True)


def _build_supervised_from_close(close: pd.Series, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs (X, y) where each X row is the previous `lookback` closes,
    and y is the next close.

    Example:
        X[i] = [close[i-lookback], ..., close[i-1]]
        y[i] = close[i]
    Returns:
        X: shape (n_samples, lookback)
        y: shape (n_samples,)
    """
    values = close.to_numpy(dtype=float)
    n = len(values)
    if n <= lookback:
        return np.empty((0, lookback)), np.empty((0,))

    X = np.lib.stride_tricks.sliding_window_view(values, window_shape=lookback)
    X = X[:-1]                      # exclude the last window because we need a subsequent y
    y = values[lookback:]           # target is the item after each window
    return X, y


def _is_constant(arr: np.ndarray) -> bool:
    return np.allclose(np.nanstd(arr), 0.0)


def _safe_pct(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """
    Computes 100 * numer / denom with protection against tiny/zero denominators.
    Returns 0 where |denom| is too small.
    """
    denom = np.asarray(denom, dtype=float)
    numer = np.asarray(numer, dtype=float)
    out = np.zeros_like(numer, dtype=float)
    mask = np.abs(denom) > 1e-12
    out[mask] = 100.0 * numer[mask] / denom[mask]
    return out


class NaiveModel:
    """
    Simple mean-constant predictor used for degenerate or insufficient data cases.
    Mimics scikit's .predict interface.
    """
    def __init__(self, const_value: float):
        self.const_value = float(const_value)

    def predict(self, X_) -> np.ndarray:
        return np.full((len(X_),), self.const_value, dtype=float)


def _train_rbf_svr(
    X: np.ndarray,
    y: np.ndarray,
    enable_cv: bool,
) -> Tuple[object, StandardScaler, StandardScaler]:
    """
    Trains an RBF SVR on standardized X and y.
    Returns fitted (model, x_scaler, y_scaler). Model could be SVR or NaiveModel.
    """
    # Insufficient data: mirror original behavior (handled earlier in fit),
    # but keep a defensive fallback here matching "mean-y" behavior.
    if X.size == 0 or y.size == 0:
        x_scaler = StandardScaler(with_mean=False, with_std=False)
        y_scaler = StandardScaler(with_mean=False, with_std=False)
        return NaiveModel(float(y[-1]) if y.size else 0.0), x_scaler, y_scaler

    # If features or targets are constant, fall back to mean(y) predictor (original behavior)
    if _is_constant(X) or _is_constant(y):
        y_mean = float(np.nanmean(y))
        x_scaler = StandardScaler(with_mean=False, with_std=False)
        y_scaler = StandardScaler(with_mean=False, with_std=False)
        return NaiveModel(y_mean), x_scaler, y_scaler

    # Standardize X and y
    x_scaler = StandardScaler()
    Xs = x_scaler.fit_transform(X)

    y_scaler = StandardScaler()
    ys = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    base = SVR(kernel="rbf", C=SVR_C, epsilon=SVR_EPSILON, gamma=SVR_GAMMA)

    if enable_cv:
        # Match original tiny time-series CV heuristic
        tscv = TimeSeriesSplit(n_splits=max(3, min(SVR_CV_SPLITS, len(Xs) // 40 or 3)))

        # Normalize gamma grid to valid floats when provided as strings
        grid_gamma = []
        for g in ([SVR_GAMMA] if SVR_GAMMA in {"scale", "auto"} else [SVR_GAMMA, "scale"]):
            try:
                grid_gamma.append(float(g))
            except (TypeError, ValueError):
                grid_gamma.append(g)

        param_grid = {
            "C": [SVR_C, max(1.0, SVR_C / 2.0), SVR_C * 2.0],
            "epsilon": [SVR_EPSILON, max(0.01, SVR_EPSILON / 2.0)],
            "gamma": grid_gamma,
        }

        gs = GridSearchCV(
            estimator=base,
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            cv=tscv,
            n_jobs=1,
            verbose=0,
            refit=True,
        )
        gs.fit(Xs, ys)
        model = gs.best_estimator_
    else:
        model = base.fit(Xs, ys)

    return model, x_scaler, y_scaler


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------

def fit(X: Union[pd.DataFrame, pd.Series, np.ndarray, list, tuple],
        Y: Optional[Union[pd.Series, np.ndarray, list]] = None) -> ModelBundle:
    """
    Train an RBF SVR on the provided series using ONLY the 'close' values.

    Parameters
    ----------
    X : DataFrame/Series/array-like
        Market data containing 'close' or a 1D array of closes.
    Y : Ignored (kept for signature compatibility)

    Returns
    -------
    ModelBundle
        Includes the trained model (SVR or NaiveModel), fitted scalers, and lookback used.
    """
    close = _extract_close_series(X)

    # Mirror original "insufficient data => skip" semantics by returning a constant predictor
    if len(close) < max(SVR_MIN_TRAIN_SIZE, SVR_LOOKBACK + 1):
        const = float(close.iloc[-1]) if len(close) else 0.0
        model = NaiveModel(const)
        x_scaler = StandardScaler(with_mean=False, with_std=False)
        y_scaler = StandardScaler(with_mean=False, with_std=False)
        return ModelBundle(model=model, x_scaler=x_scaler, y_scaler=y_scaler, lookback=SVR_LOOKBACK)

    # Build supervised dataset and train
    X_lag, y = _build_supervised_from_close(close, SVR_LOOKBACK)
    model, x_scaler, y_scaler = _train_rbf_svr(X_lag, y, enable_cv=SVR_ENABLE_CV)

    return ModelBundle(model=model, x_scaler=x_scaler, y_scaler=y_scaler, lookback=SVR_LOOKBACK)


def predict(model_bundle: ModelBundle,
            X: Union[pd.DataFrame, pd.Series, np.ndarray, list, tuple]) -> np.ndarray:
    """
    Produce one-step-ahead *percent growth* predictions aligned to the input rows.

    For an input series of length N:
      - Returns an array of length N
      - Entries [0 : lookback-1] are np.nan (insufficient history)
      - For t >= lookback, we:
          1) Predict close[t] using window [t-lookback, ..., t-1]
          2) Return growth% = 100 * (pred_close[t] - close[t-1]) / close[t-1]

    Notes
    -----
    - A value of +1.0 means +1% (not 0.01).
    - If the previous close is non-finite or ~0, growth is defined as 0 to avoid blow-ups.
    - Matches original behavior by clamping invalid/negative predicted closes
      back to the previous close before computing %.
    """
    if not isinstance(model_bundle, ModelBundle):
        raise TypeError("predict(model, X) expects model to be a ModelBundle returned by fit().")

    close = _extract_close_series(X)
    n = len(close)
    lookback = int(model_bundle.lookback)

    # Default output with NaNs where we cannot form a window
    out_pct = np.full(n, np.nan, dtype=float)
    if n <= lookback:
        return out_pct

    # Windows correspond to targets at indices [lookback .. n-1]
    X_lag, _ = _build_supervised_from_close(close, lookback)
    if X_lag.size == 0:
        return out_pct

    # Standardize with training scalers (if meaningful)
    try:
        Xs = model_bundle.x_scaler.transform(X_lag)
    except Exception:
        Xs = X_lag

    y_pred_std = np.asarray(model_bundle.model.predict(Xs)).reshape(-1, 1)
    try:
        y_pred = model_bundle.y_scaler.inverse_transform(y_pred_std).ravel()  # predicted closes at [lookback..n-1]
    except Exception:
        y_pred = y_pred_std.ravel()

    # Base close (previous candle) for each prediction is the last element of each window, i.e., close[t-1]
    close_arr = close.to_numpy(dtype=float)
    base_close = close_arr[lookback - 1 : n - 1]  # shape (n - lookback,)

    # Mirror original: clamp invalid or non-positive predictions to previous close
    # (prevents nonsensical negative prices)
    valid_pred = np.isfinite(y_pred) & (y_pred > 0.0)
    y_pred_clamped = np.where(valid_pred, y_pred, base_close)

    # Convert predicted close to percent growth vs previous close
    pct_growth = _safe_pct(y_pred_clamped - base_close, base_close)

    # Place percentages aligned to indices [lookback .. n-1]
    out_pct[lookback:] = pct_growth

    # Replace any remaining non-finite outputs with 0.0%
    bad = ~np.isfinite(out_pct)
    if bad.any():
        out_pct[bad] = 0.0

    return out_pct
