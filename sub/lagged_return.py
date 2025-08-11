import os
import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sub.common import compute_meta_labels, USE_META_LABEL

from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error


FEATURES = [
    'close',
    'lagged_close_1', 'lagged_close_2', 'lagged_close_3', 'lagged_close_5', 'lagged_close_10',
    'returns_1', 'returns_3', 'returns_5'
]

BEST_PARAMS = {
    "n_estimators":     600,
    "learning_rate":    0.03,
    "max_depth":        6,
    "subsample":        0.80,
    "colsample_bytree": 0.80,
    "min_child_weight": 20,
    "reg_alpha":        0.0,
    "reg_lambda":       1.0,
}

def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if USE_META_LABEL:
        return compute_meta_labels(df).rename(columns={'meta_label': 'label'})
    df['close_t1'] = df['close'].shift(-1)
    df['target']  = (df['close_t1'] - df['close']) / df['close']
    return df

def sharpe(returns: np.ndarray) -> float:
    arr = np.asarray(returns)
    return np.nan if arr.std()==0 else arr.mean()/arr.std()*np.sqrt(252)

def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    return ((equity - roll_max)/roll_max).min()


def fit(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    logging.info("RUNNING FIT on lagged_return – static params + early-stop")

    model = XGBRegressor(
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
        **BEST_PARAMS,
    )

    val_size = max(100, int(0.1 * len(X_train)))
    X_tr, X_val = X_train.iloc[:-val_size], X_train.iloc[-val_size:]
    y_tr, y_val = y_train.iloc[:-val_size], y_train.iloc[-val_size:]

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model

def predict(model: XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    logging.info("Ruinning PREDICT on lagged_return")
    ret = model.predict(X)
    tau = getattr(model, "tau", 1.0)
    return 1 / (1 + np.exp(-(ret / tau)))
