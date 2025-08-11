import numpy as np
import pandas as pd
import logging
import argparse

import matplotlib.pyplot as plt
from sub.common import compute_meta_labels, USE_META_LABEL
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    'bollinger_upper', 'bollinger_lower', 'bollinger_percB',
    'atr', 'atr_zscore',
    'candle_body_ratio', 'wick_dominance',
    'rsi_zscore',
    'macd_histogram', 'macd_hist_flip', 'macd_cross',
    'high_low_range',
    'std_5', 'std_10',
    'gap_vs_prev'
]

PARAM_GRID = {
    'n_estimators':     [250],
    'max_depth':        [5, 10, 20],
    'min_samples_leaf': [1],
    'max_features':     [len(FEATURES)]
}


def compute_labels(df):
    df = df.copy()

    if USE_META_LABEL:
        # hand off to meta-label helper exactly as before
        return compute_meta_labels(df).rename(columns={'meta_label': 'label'})

    df['mean_band'] = (df['bollinger_upper'] + df['bollinger_lower']) / 2
    c0, c1, c2 = df['close'], df['close'].shift(-1), df['close'].shift(-2)

    up1 = (c0 > df['mean_band']) & ((c0 - c1) >= df['atr'])
    up2 = (c0 > df['mean_band']) & ((c0 - c2) >= df['atr'])
    dn1 = (c0 < df['mean_band']) & ((c1 - c0) >= df['atr'])
    dn2 = (c0 < df['mean_band']) & ((c2 - c0) >= df['atr'])

    df['label'] = (up1 | up2 | dn1 | dn2).astype(int)

    return df.dropna(subset=FEATURES + ['label']).reset_index(drop=True)

def sharpe(returns):
    if len(returns)==0 or returns.std()==0:
        return np.nan
    return returns.mean()/returns.std()*np.sqrt(252)

def max_drawdown(equity):
    roll_max = equity.cummax()
    return ((equity - roll_max)/roll_max).min()

def fit(X_train: np.ndarray, y_train: np.ndarray):
    logging.info("Running FIT on mean_reversion (TS-CV + calibration)")

    tscv = TimeSeriesSplit(n_splits=5)

    base_rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    grid = GridSearchCV(
        estimator=base_rf,
        param_grid=PARAM_GRID,
        cv=tscv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    ).fit(X_train, y_train)

    best_rf = grid.best_estimator_
    logging.info("Best params ⇒ %s", grid.best_params_)

    calib = CalibratedClassifierCV(
        best_rf, method="isotonic", cv=TimeSeriesSplit(n_splits=3)
    ).fit(X_train, y_train)

    _rf_params = best_rf.get_params()

    def _get_params(deep=True):
        return {k: _rf_params[k] for k in PARAM_GRID}

    calib.get_params = _get_params
    return calib


def predict(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    logging.info("Running PREDICT on mean_reversion")
    return model.predict_proba(X)[:, 1]
