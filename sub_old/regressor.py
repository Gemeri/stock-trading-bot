# regressor.py
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional
from tqdm import tqdm
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES: List[str] = [
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

# Multi-horizon steps ahead (in bars)
MULTI_HORIZONS: List[int] = [2, 4, 6, 8, 10, 12]


def _target_col(h: int) -> str:
    # We train in RETURN space (stationary).
    return f"target_h{h}"


def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Percent targets: target_h{h} = 100 * (close_{t+h} - close_t) / close_t
    e.g., +1% => 1.0
    """
    d = df.copy()
    close = d["close"].astype(float)
    for h in MULTI_HORIZONS:
        d[_target_col(h)] = 100.0 * (close.shift(-h) - close) / close
    return d.reset_index(drop=True)


# ---------- Utilities ----------
def _ensure_2d_y(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    return y

def _as_2d_x(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2:
        x = x.reshape(-1, x.shape[-1])
    return x

def _clean_xy(X: np.ndarray, Y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Drop any rows where X or Y contains NaN/Inf. Ensures Y is 2D.
    """
    X = _as_2d_x(np.asarray(X, dtype=float))
    Y = _ensure_2d_y(np.asarray(Y, dtype=float))

    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)

    if X.shape[0] != Y.shape[0]:
        n = min(X.shape[0], Y.shape[0])
        X, Y = X[:n], Y[:n]
        if w is not None:
            w = w[:n]

    x_ok = np.isfinite(X).all(axis=1)
    y_ok = np.isfinite(Y).all(axis=1)
    ok = x_ok & y_ok
    Xc, Yc = X[ok], Y[ok]
    Wc = w[ok] if w is not None else None
    return Xc, Yc, Wc

def _split_train_val(
    X: np.ndarray,
    Y: np.ndarray,
    min_train: int = 64,
    val_frac: float = 0.10,
    max_val: int = 200,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Robust tail split that guarantees non-empty labels for training.
    If dataset is small, skip validation entirely.
    """
    n = len(X)
    w = None if weights is None else np.asarray(weights, dtype=float).reshape(-1)[:n]
    if n <= min_train + 1:
        return X, Y, None, None, w, None

    val_size = int(round(val_frac * n))
    val_size = max(1, min(max_val, val_size))

    if n - val_size < min_train:
        val_size = max(1, n - min_train)

    if n - val_size < 1 or val_size < 1:
        return X, Y, None, None, w, None

    X_tr, X_val = X[:-val_size], X[-val_size:]
    Y_tr, Y_val = Y[:-val_size], Y[-val_size:]
    W_tr = w[:-val_size] if w is not None else None
    W_val = w[-val_size:] if w is not None else None
    return X_tr, Y_tr, X_val, Y_val, W_tr, W_val


# ---------- Fit / Predict ----------
def fit(X_train: np.ndarray, Y_train: np.ndarray, sample_weight: Optional[np.ndarray] = None):
    """
    Fit CatBoost regressor on RETURN targets (not levels).
    """
    logging.info("FIT CatBoostRegressor – training on return targets")

    Xc, Yc, Wc = _clean_xy(X_train, Y_train, sample_weight=sample_weight)
    if Xc.shape[0] == 0 or Yc.size == 0 or Yc.shape[0] == 0:
        raise ValueError("Training data is empty after cleaning (no usable labels/rows).")

    X_tr, Y_tr, X_val, Y_val, W_tr, W_val = _split_train_val(
        Xc, Yc, min_train=64, val_frac=0.10, max_val=200, weights=Wc
    )

    multi_output = (Y_tr.ndim == 2 and Y_tr.shape[1] > 1)
    loss = "MultiRMSE" if multi_output else "RMSE"

    model = CatBoostRegressor(
        iterations=600,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3,
        bagging_temperature=0.5,
        loss_function=loss,
        eval_metric=loss,
        random_seed=42,
        verbose=False,
    )

    train_pool = Pool(X_tr, Y_tr, weight=W_tr)

    if X_val is None:
        model.fit(train_pool)
    else:
        eval_pool = Pool(X_val, Y_val, weight=W_val)
        model.fit(train_pool, eval_set=eval_pool, use_best_model=True, early_stopping_rounds=60)

    return model


def predict(model, X: np.ndarray) -> np.ndarray:
    """
    Predict percent returns per horizon.
    Output units: percent points (1.0 == +1%).
    """
    X = _as_2d_x(X)
    raw = model.predict(X)
    preds_pct = np.asarray(raw, dtype=float)
    if preds_pct.ndim == 1:
        preds_pct = preds_pct.reshape(-1, 1)
    return preds_pct


# ---------- Optional CLI walk-forward (now in RETURN space) ----------
@dataclass
class WalkForwardConfig:
    n_back: int = 500
    window: int = 500

def _walk_forward(df: pd.DataFrame, cfg: WalkForwardConfig, output_path: str) -> pd.DataFrame:
    """
    Walk-forward demo that saves predicted and actual RETURNS (not levels).
    """
    logging.info("Walk-forward (multi-horizon reg on returns; output saved)")
    df = compute_targets(df)
    tgt_cols = [_target_col(h) for h in MULTI_HORIZONS]

    recs = []
    start = max(0, len(df) - cfg.n_back - 1)
    end = len(df) - 2

    close = pd.to_numeric(df["close"], errors="coerce").astype(float).values

    for t in tqdm(range(start, end), desc="Walk-forward (multi-reg)"):
        row_feats = df.loc[[t], FEATURES].values
        tr_slice = df.iloc[max(0, t - cfg.window): t].dropna(subset=FEATURES + tgt_cols)
        if len(tr_slice) < 50:
            rec = {"t": t}
            for h in MULTI_HORIZONS:
                rec[f"pred_h{h}"] = float('nan')
                rec[f"actual_h{h}"] = float('nan')
            recs.append(rec)
            continue

        X_tr = tr_slice[FEATURES].values
        Y_tr = tr_slice[tgt_cols].values  # returns
        model = fit(X_tr, Y_tr)

        yhat_ret = np.asarray(predict(model, row_feats), dtype=float)[0]  # returns
        rec = {"t": t}
        for i, h in enumerate(MULTI_HORIZONS):
            # store returns for both pred and actual
            rec[f"pred_h{h}"] = float(yhat_ret[i]) if i < len(yhat_ret) else float('nan')
            if t + h < len(close):
                actual_ret = (close[t + h] - close[t]) / close[t]
            else:
                actual_ret = np.nan
            rec[f"actual_h{h}"] = float(actual_ret) if np.isfinite(actual_ret) else float('nan')

        recs.append(rec)

    res = pd.DataFrame(recs).set_index("t")
    res.to_csv(output_path)

    # Metrics for h=1 (on returns)
    mask_h1 = res["pred_h1"].notna() & res["actual_h1"].notna()
    if mask_h1.sum() > 0:
        mae = mean_absolute_error(res.loc[mask_h1, "actual_h1"], res.loc[mask_h1, "pred_h1"])
        rmse = mean_squared_error(res.loc[mask_h1, "actual_h1"], res.loc[mask_h1, "pred_h1"], squared=False)
        print(f"Saved to {output_path} | (returns) h=1 MAE={mae:.6f} RMSE={rmse:.6f}")
    else:
        print(f"Saved to {output_path} (no valid h=1 slice)")
    return res
