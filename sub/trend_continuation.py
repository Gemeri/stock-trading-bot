import logging
from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier, Pool
from sub.common import compute_meta_labels, USE_META_LABEL

FEATURES = [
    'ema_9', 'ema_21', 'ema_50', 'ema_200',
    'adx', 'obv',
    'rsi', 'macd_line',
    'open', 'high', 'low', 'close',
    'days_since_high', 'days_since_low'
]

# Fixed CatBoost params (previous best combo), no grid search
CATBOOST_PARAMS = {
    'iterations': 600,
    'depth': 6,
    'learning_rate': 0.06,
    'l2_leaf_reg': 3,
    'bagging_temperature': 0.1,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'random_state': 42,
    'verbose': False,
}

def compute_labels(df, mode='momentum', n_ahead=2, thresh=0.002, quantile=0.4):
    df = df.copy()
    if USE_META_LABEL:
        return compute_meta_labels(df).rename(columns={'meta_label': 'label'})
    df['fwd_ret'] = (df['close'].shift(-n_ahead) - df['close']) / df['close']
    if mode == 'momentum':
        df['label'] = (df['fwd_ret'] > thresh).astype(int)
    elif mode == 'quantile':
        q_hi = df['fwd_ret'].quantile(1 - quantile)
        q_lo = df['fwd_ret'].quantile(quantile)
        df['label'] = np.where(
            df['fwd_ret'] >= q_hi, 1,
            np.where(df['fwd_ret'] <= q_lo, 0, np.nan)
        )
    else:
        raise ValueError("Unknown mode for label: choose 'momentum' or 'quantile'")
    return df

def sharpe(returns):
    arr = np.asarray(returns)
    if arr.size == 0 or arr.std() == 0:
        return np.nan
    return arr.mean() / arr.std() * np.sqrt(252)

def max_drawdown(equity):
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return dd.min()


ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]

def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.values
    return np.asarray(x)

def _select_features(X: ArrayLike) -> ArrayLike:
    """
    Ensure we use exactly the FEATURES list when a DataFrame is provided.
    If X is an ndarray, we assume it's already aligned to FEATURES.
    """
    if isinstance(X, pd.DataFrame):
        missing = [f for f in FEATURES if f not in X.columns]
        if missing:
            raise KeyError(f"Missing required features: {missing}")
        return X[FEATURES]
    return X

def fit(X_train: ArrayLike, y_train: ArrayLike) -> CatBoostClassifier:
    """
    Train a single CatBoostClassifier using the fixed params above.
    Uses the provided FEATURES columns if X is a DataFrame.
    A simple chronological 80/20 split is used for early stopping.
    """
    logging.info("Running FIT (CatBoost) on trend_continuation (no grid search)")

    # Enforce feature usage
    X_feat = _select_features(X_train)

    X_all = _to_numpy(X_feat)
    y_all = _to_numpy(y_train).ravel()

    n = len(X_all)
    if n < 50:
        # Not enough data for a split; train on all without eval_set
        logging.warning("Fewer than 50 rows; training on all data without validation.")
        pool_all = Pool(X_all, y_all)
        model = CatBoostClassifier(**CATBOOST_PARAMS)
        model.fit(pool_all)
        return model

    # Chronological split for early stopping
    split = int(n * 0.8)
    pool_tr = Pool(X_all[:split], y_all[:split])
    pool_val = Pool(X_all[split:], y_all[split:])

    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(
        pool_tr,
        eval_set=pool_val,
        early_stopping_rounds=50,
        use_best_model=True,
    )
    return model

def predict(model: CatBoostClassifier, X: ArrayLike) -> np.ndarray:
    """
    Predict class probabilities (positive class) using the trained model.
    Ensures the FEATURES columns are used when X is a DataFrame.
    """
    logging.info("Running PREDICT (CatBoost) on trend_continuation")
    X_feat = _select_features(X)
    X_np = _to_numpy(X_feat)
    return model.predict_proba(X_np)[:, 1]
