import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Union

from catboost import CatBoostClassifier, Pool
from sub.common import compute_meta_labels, USE_META_LABEL
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

FEATURES = [
    'ema_9', 'ema_21', 'ema_50', 'ema_200',
    'adx', 'obv',
    'rsi', 'macd_line',
    'open', 'high', 'low', 'close',
    'days_since_high', 'days_since_low'
]

PARAM_GRID = {
    'iterations':      [600],
    'depth':           [6],
    'learning_rate':   [0.06],
    'l2_leaf_reg':     [3],
    'bagging_temperature':[0.1]
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
        df['label'] = np.where(df['fwd_ret'] >= q_hi, 1,
                        np.where(df['fwd_ret'] <= q_lo, 0, np.nan))
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

def fit(X_train: ArrayLike, y_train: ArrayLike) -> CatBoostClassifier:
    logging.info("Running FIT (CatBoost) on trend_continuation")

    X_all = _to_numpy(X_train)
    y_all = _to_numpy(y_train).ravel()

    tscv        = TimeSeriesSplit(n_splits=5)
    best_score  = -np.inf
    best_params = None

    for params in tqdm(ParameterGrid(PARAM_GRID), desc="Hyper-param grid", unit="combo"):
        fold_scores = []

        for tr_idx, val_idx in tscv.split(X_all):
            pool_tr  = Pool(X_all[tr_idx],  y_all[tr_idx])
            pool_val = Pool(X_all[val_idx], y_all[val_idx])

            mdl = CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="AUC",
                random_state=42,
                verbose=False,
                **params
            )
            mdl.fit(
                pool_tr,
                eval_set=pool_val,
                early_stopping_rounds=50,
                use_best_model=True,
            )
            prob_val   = mdl.predict_proba(pool_val)[:, 1]
            fold_scores.append(roc_auc_score(y_all[val_idx], prob_val))

        mean_auc = float(np.mean(fold_scores))
        if mean_auc > best_score:
            best_score, best_params = mean_auc, params

    logging.info(f"Best params: {best_params} | CV ROC-AUC: {best_score:.4f}")

    full_pool   = Pool(X_all, y_all)
    best_model  = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        random_state=42,
        verbose=False,
        **best_params
    )
    best_model.fit(full_pool, early_stopping_rounds=50, use_best_model=True)
    return best_model


def predict(model: CatBoostClassifier, X: ArrayLike) -> np.ndarray:
    logging.info("Running PREDICT (CatBoost) on trend_continuation")
    X_np = _to_numpy(X)
    return model.predict_proba(X_np)[:, 1]
