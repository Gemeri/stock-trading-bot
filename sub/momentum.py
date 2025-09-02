import logging

import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sub.common import compute_meta_labels, USE_META_LABEL
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from typing import Any

FEATURES = [
    'rsi', 'roc',
    'macd_line', 'macd_signal', 'macd_histogram',
    'rsi_zscore', 'macd_cross', 'macd_hist_flip',
    'adx',
    'price_change'
]

# Updated grid for HistGradientBoostingClassifier
PARAM_GRID = {
    'max_iter':          [250, 500],
    'max_depth':         [6, 10],
    'learning_rate':     [0.01, 0.1],
    'l2_regularization': [0.0, 1.0],
}


def compute_labels(df):
    df = df.copy()

    if USE_META_LABEL:
        return compute_meta_labels(df).rename(columns={'meta_label': 'label'})

    df['close_t1'] = df['close'].shift(-1)
    df['label'] = (df['close_t1'] > df['close']).astype(int)

    return df.dropna(subset=FEATURES + ['label']).reset_index(drop=True)


def sharpe(returns):
    if len(returns) == 0 or returns.std() == 0:
        return np.nan
    return returns.mean() / returns.std() * np.sqrt(252)


def max_drawdown(equity):
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return drawdown.min()


def fit(X_train: np.ndarray, y_train: np.ndarray) -> CalibratedClassifierCV:
    logging.info("Running FIT on momentum (TS-CV + calibration)")

    splitter = TimeSeriesSplit(n_splits=5)

    base = HistGradientBoostingClassifier(
        loss="log_loss",
        early_stopping=True,
        random_state=42,
    )

    grid = GridSearchCV(
        estimator=base,
        param_grid=PARAM_GRID,
        cv=splitter,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    ).fit(X_train, y_train)

    best_hgb = grid.best_estimator_
    logging.info("Best params ⇒ %s", grid.best_params_)

    iso = CalibratedClassifierCV(
        best_hgb, method="isotonic", cv=TimeSeriesSplit(n_splits=3)
    ).fit(X_train, y_train)

    _best = best_hgb.get_params()

    def _get_params(deep: bool = True):
        # Only expose tuned params downstream (to mirror previous behavior)
        return {k: _best[k] for k in PARAM_GRID}

    iso.get_params = _get_params  # type: ignore[attr-defined]
    return iso


def predict(model: Any, X: np.ndarray) -> np.ndarray:
    logging.info("Running PREDICT on momentum")
    return model.predict_proba(X)[:, 1]
