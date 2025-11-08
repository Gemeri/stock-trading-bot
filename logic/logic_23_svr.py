"""
RBF SVR price prediction logic script (production-ready)
- Trains on ONLY the 'close' series using sliding-window lags
- Uses non-linear SVR with RBF kernel
- No look-ahead: backtests train strictly on data <= current_timestamp
- Provides run_logic (live) and run_backtest (simulation) entrypoints
"""

from __future__ import annotations

import os
import math
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# --------------------------------------------------------------------------------------
# Configuration & CSV access (kept here for self-containment and reliability)
# --------------------------------------------------------------------------------------

BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "1Hour")
TIMEFRAME_MAP = {
    "4Hour": "H4", "2Hour": "H2", "1Hour": "H1",
    "30Min": "M30", "15Min": "M15"
}
CONVERTED_TIMEFRAME = TIMEFRAME_MAP.get(BAR_TIMEFRAME, BAR_TIMEFRAME)

def get_csv_filename(ticker: str) -> str:
    return os.path.join("data", f"{ticker}_{CONVERTED_TIMEFRAME}.csv")

# SVR + windowing hyperparameters (overridable via env)
SVR_LOOKBACK = int(os.getenv("SVR_LOOKBACK", "30"))             # number of lag closes per sample
SVR_MIN_TRAIN_SIZE = int(os.getenv("SVR_MIN_TRAIN_SIZE", "300"))# minimum rows required to fit
SVR_EPSILON = float(os.getenv("SVR_EPSILON", "0.05"))
SVR_C = float(os.getenv("SVR_C", "10.0"))
SVR_GAMMA = os.getenv("SVR_GAMMA", "scale")                     # 'scale' or 'auto' or float string

# Cross-validation toggles for live use (kept lightweight)
SVR_ENABLE_CV = os.getenv("SVR_ENABLE_CV", "false").lower() in {"1", "true", "yes"}
# If CV enabled, use a tiny search grid to remain fast enough for live calls
SVR_CV_SPLITS = int(os.getenv("SVR_CV_SPLITS", "5"))
SVR_CV_GRID = {
    "C": [SVR_C, max(1.0, SVR_C / 2.0), SVR_C * 2.0],
    "epsilon": [SVR_EPSILON, max(0.01, SVR_EPSILON / 2.0)],
    "gamma": [SVR_GAMMA] if SVR_GAMMA in {"scale", "auto"} else [SVR_GAMMA, "scale"],
}

# Risk controls
MAX_CASH_UTILIZATION = float(os.getenv("MAX_CASH_UTILIZATION", "1.0"))  # 1.0 means use all available cash


# --------------------------------------------------------------------------------------
# Internal utilities
# --------------------------------------------------------------------------------------

@dataclass
class ModelBundle:
    model: SVR
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    lookback: int
    last_window_key: Tuple[float, ...]  # signature of the last window we fit on (for caching)


# In-memory model cache for live trading (by ticker). Backtests intentionally bypass cache.
_MODEL_CACHE: Dict[str, ModelBundle] = {}


def _read_close_series(ticker: str) -> pd.DataFrame:
    """
    Reads only the columns needed: timestamp, close.
    Ensures timestamp is parsed and sorted ascending.
    """
    csv_path = get_csv_filename(ticker)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found for {ticker}: {csv_path}")

    # Read minimally to keep memory + IO tight
    usecols = ["timestamp", "close"]
    df = pd.read_csv(csv_path, usecols=usecols)

    # Parse timestamp to pandas datetime (naive)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "close"]).copy()

    # Enforce numeric close
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).copy()

    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _filter_upto_timestamp(df: pd.DataFrame, upto_ts) -> pd.DataFrame:
    """
    Filters df to rows where timestamp <= upto_ts (inclusive).
    upto_ts can be str, pd.Timestamp, or compatible.
    """
    if not isinstance(upto_ts, pd.Timestamp):
        upto_ts = pd.to_datetime(upto_ts, errors="coerce")
    if pd.isna(upto_ts):
        # If timestamp can't be parsed, fail safe by returning entire df (or raise)
        raise ValueError("current_timestamp could not be parsed into a valid datetime.")
    return df[df["timestamp"] <= upto_ts].copy()


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
    X = X[:-1]  # exclude the last window because we need a subsequent y
    y = values[lookback:]  # target is the item after each window
    return X, y


def _is_constant(arr: np.ndarray) -> bool:
    return np.allclose(np.nanstd(arr), 0.0)


def _train_rbf_svr(
    X: np.ndarray,
    y: np.ndarray,
    enable_cv: bool,
) -> Tuple[SVR, StandardScaler, StandardScaler]:
    """
    Trains an RBF SVR on standardized X and y.
    Returns fitted (model, x_scaler, y_scaler).
    """
    if X.size == 0 or y.size == 0:
        raise ValueError("Insufficient data to train SVR.")

    # If features are constant (degenerate), fall back to naive predictor
    if _is_constant(X) or _is_constant(y):
        # We emulate a model that always predicts mean(y)
        class NaiveModel:
            def __init__(self, const_value):
                self.const_value = float(const_value)
            def predict(self, X_):
                return np.full((len(X_),), self.const_value, dtype=float)

        y_mean = float(np.nanmean(y))
        x_scaler = StandardScaler(with_mean=False, with_std=False)
        y_scaler = StandardScaler(with_mean=False, with_std=False)
        # Create a dummy SVR-like object
        naive_svr = NaiveModel(y_mean)  # type: ignore
        return naive_svr, x_scaler, y_scaler

    # Standardize X and y
    x_scaler = StandardScaler()
    Xs = x_scaler.fit_transform(X)

    y_scaler = StandardScaler()
    ys = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    # Base model
    base = SVR(kernel="rbf", C=SVR_C, epsilon=SVR_EPSILON, gamma=SVR_GAMMA)

    if enable_cv:
        # Lightweight, time-series-aware CV
        tscv = TimeSeriesSplit(n_splits=max(3, min(SVR_CV_SPLITS, len(Xs) // 40 or 3)))
        param_grid = {}

        # Normalize gamma to valid floats when provided as strings
        grid_gamma = []
        for g in SVR_CV_GRID.get("gamma", ["scale"]):
            try:
                grid_gamma.append(float(g))
            except (TypeError, ValueError):
                grid_gamma.append(g)
        param_grid["gamma"] = grid_gamma

        # C and epsilon grid
        param_grid["C"] = SVR_CV_GRID.get("C", [SVR_C])
        param_grid["epsilon"] = SVR_CV_GRID.get("epsilon", [SVR_EPSILON])

        # Guardrail: don't explode search size
        # (<= 18 combos typical)
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


def _predict_next_close_from_series(
    close_series: pd.Series,
    lookback: int,
    use_cache_key: Optional[Tuple[float, ...]] = None,
    allow_cache: bool = False,
    cache_bucket: Optional[str] = None,
    enable_cv: bool = False,
) -> float:
    """
    Trains on all available history in close_series (respecting pre-filtering for backtests)
    and predicts the NEXT close (i.e., t+1).

    If allow_cache is True and cache_bucket (ticker) provided, attempts to reuse model
    when the last window hasn't changed.
    """
    # Build supervised matrices
    X, y = _build_supervised_from_close(close_series, lookback)
    if X.shape[0] == 0:
        # Not enough data; fallback to last close (naive)
        return float(close_series.iloc[-1])

    # The last window is used to predict the NEXT close
    last_window = X[-1, :].copy()
    last_window_key = tuple(map(float, last_window))

    # Try cache reuse for live logic
    if allow_cache and cache_bucket:
        bundle = _MODEL_CACHE.get(cache_bucket)
        if bundle and bundle.lookback == lookback and bundle.last_window_key == last_window_key:
            # Reuse cached model/scalers
            model, x_scaler, y_scaler = bundle.model, bundle.x_scaler, bundle.y_scaler
        else:
            # Train and update cache
            model, x_scaler, y_scaler = _train_rbf_svr(X, y, enable_cv=enable_cv)
            _MODEL_CACHE[cache_bucket] = ModelBundle(
                model=model,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                lookback=lookback,
                last_window_key=last_window_key,
            )
    else:
        # Backtests or cache disabled: always retrain on the filtered subset
        model, x_scaler, y_scaler = _train_rbf_svr(X, y, enable_cv=enable_cv)

    # Predict the next close
    X_last = last_window.reshape(1, -1)
    try:
        X_last_std = x_scaler.transform(X_last)
    except Exception:
        # In degenerate scaler cases (no-std), transform may be identity
        X_last_std = X_last

    y_pred_std = model.predict(X_last_std).reshape(-1, 1)
    try:
        y_pred = y_scaler.inverse_transform(y_pred_std).ravel()[0]
    except Exception:
        y_pred = float(y_pred_std.ravel()[0])

    # Clamp to non-negative
    if not np.isfinite(y_pred) or y_pred <= 0:
        y_pred = float(close_series.iloc[-1])

    return float(y_pred)


def _safe_int_shares(cash: float, price: float) -> int:
    if not np.isfinite(cash) or not np.isfinite(price) or price <= 0:
        return 0
    max_cash = max(0.0, float(cash) * MAX_CASH_UTILIZATION)
    return int(max_cash // float(price))


# --------------------------------------------------------------------------------------
# Public API: run_logic and run_backtest
# --------------------------------------------------------------------------------------

def run_logic(current_price: float, predicted_price: float, ticker: str):
    """
    LIVE trading entrypoint.
    - Ignores the predicted_price argument (per requirement)
    - Loads full CSV for the ticker
    - Trains RBF SVR on 'close' only (with lag window)
    - Predicts next close and trades based on predicted vs current

    Trading rule:
    - If predicted > current and no position, BUY (max affordable)
    - If predicted < current and have position, SELL (all)
    - Else NONE

    Uses caching to avoid retraining if the last window didn't change.
    """
    from forest import api, buy_shares, sell_shares  # imported here to avoid issues in non-trading contexts

    logger = logging.getLogger(__name__)

    # Account & position
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

    # Load data and predict next close using only 'close'
    try:
        df = _read_close_series(ticker)
        if len(df) < max(SVR_MIN_TRAIN_SIZE, SVR_LOOKBACK + 1):
            logger.warning(f"[{ticker}] Insufficient data (have {len(df)} rows). Skipping trade.")
            return

        predicted = _predict_next_close_from_series(
            close_series=df["close"],
            lookback=SVR_LOOKBACK,
            allow_cache=True,
            cache_bucket=ticker,
            enable_cv=SVR_ENABLE_CV,
        )
    except Exception as e:
        logger.error(f"[{ticker}] Prediction error: {e}")
        return

    logger.info(f"[{ticker}] Current Price: {current_price:.6f}, Predicted Next Close (SVR): {predicted:.6f}, "
                f"Position: {position_qty}, Cash: {cash}")

    # Decision logic
    try:
        if predicted > current_price:
            if position_qty == 0:
                shares = _safe_int_shares(cash, current_price)
                if shares > 0:
                    logger.info(f"[{ticker}] BUY {shares} @ {current_price:.6f} (pred {predicted:.6f}).")
                    buy_shares(ticker, shares, current_price, predicted)
                else:
                    logger.info(f"[{ticker}] Insufficient cash to buy.")
            else:
                logger.info(f"[{ticker}] Already long; no BUY.")
        elif predicted < current_price:
            if position_qty > 0:
                logger.info(f"[{ticker}] SELL {position_qty} @ {current_price:.6f} (pred {predicted:.6f}).")
                sell_shares(ticker, position_qty, current_price, predicted)
            else:
                logger.info(f"[{ticker}] No position; no SELL.")
        else:
            logger.info(f"[{ticker}] Predicted equals current; no action.")
    except Exception as e:
        logger.error(f"[{ticker}] Order error: {e}")


def run_backtest(
    current_price: float,
    predicted_price: float,        # ignored by design
    position_qty: float,
    current_timestamp,             # str | datetime-like; inclusive filter
    candles,                       # unused (kept for signature compatibility)
    ticker: str
) -> str:
    """
    BACKTEST entrypoint.
    - Ignores predicted_price and candles (kept for signature compatibility)
    - Loads CSV and strictly filters to rows with timestamp <= current_timestamp
    - Trains RBF SVR on 'close' (lags), predicts next close at that point in time
    - Returns one of "BUY", "SELL", "NONE"

    Trading rule (same as live):
    - If predicted > current and no position: BUY
    - If predicted < current and have position: SELL
    - Else: NONE
    """
    logger = logging.getLogger(__name__)

    try:
        df_all = _read_close_series(ticker)
        df = _filter_upto_timestamp(df_all, current_timestamp)
        if df.empty:
            logger.debug(f"[{ticker}] No data <= {current_timestamp}.")
            return "NONE"

        if len(df) < max(SVR_MIN_TRAIN_SIZE, SVR_LOOKBACK + 1):
            # Not enough history up to this point to responsibly train
            return "NONE"

        # IMPORTANT: Always retrain on the filtered subset (no cache) to avoid leakage
        predicted = _predict_next_close_from_series(
            close_series=df["close"],
            lookback=SVR_LOOKBACK,
            allow_cache=False,
            cache_bucket=None,
            enable_cv=False,  # keep backtests fast and deterministic by default
        )
        print(predicted)
        # Decision logic mirrors live logic
        if predicted > current_price * 1.01:
            return "BUY" if float(position_qty) == 0.0 else "NONE"
        elif predicted < current_price * 1.01:
            return "SELL" if float(position_qty) > 0.0 else "NONE"
        else:
            return "NONE"

    except Exception as e:
        logger.error(f"[{ticker}] Backtest error at {current_timestamp}: {e}")
        return "NONE"
