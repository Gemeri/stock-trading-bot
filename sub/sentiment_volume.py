import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from lightgbm import LGBMClassifier
from sub.common import compute_meta_labels, USE_META_LABEL
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

FEATURES = [
    'sentiment',
    'volume', 'log_volume', 'volume_zscore', 'transactions',
    'vwap', 'obv',
    'gap_vs_prev', 'wick_dominance',
    'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'month'
]

LGBM_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.03,
    'num_leaves': 7,
    'objective': 'binary',
    'random_state': 42,
    'class_weight': 'balanced',
    'min_child_samples': 50,
    'min_split_gain': 0.02,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
    'verbosity': -1
}

def compute_labels(
    df: pd.DataFrame,
    min_return: float | None = None,
    breakout_pct: float = 0.15,
) -> pd.DataFrame:
    df = df.copy()

    if USE_META_LABEL:
        return (
            compute_meta_labels(df)
            .rename(columns={"meta_label": "label"})
            .dropna(subset=FEATURES + ["label"])
            .reset_index(drop=True)
        )

    df["close_t1"] = df["close"].shift(-1)
    df["raw_ret"] = (df["close_t1"] - df["close"]) / df["close"]

    absrets = df["raw_ret"].abs()
    if min_return is None:
        min_return = absrets.quantile(1 - breakout_pct)
        logging.info(
            "[LABEL] Percentile mode → min_return=%.5f (top %.1f%%)",
            min_return,
            breakout_pct * 100,
        )

    df["label"] = (absrets > min_return).astype(int)

    bal = df["label"].value_counts(normalize=True).to_dict()
    logging.info(
        "[LABEL BALANCE] breakout=%.2f  non-breakout=%.2f",
        bal.get(1, 0.0),
        bal.get(0, 0.0),
    )

    return df.dropna(subset=FEATURES + ["label"]).reset_index(drop=True)

def sharpe(returns: np.ndarray) -> float:
    arr = np.asarray(returns)
    if arr.size == 0 or arr.std() == 0:
        return np.nan
    return arr.mean() / arr.std() * np.sqrt(252)

def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    return ((equity - roll_max) / roll_max).min()


def fit(X_train: pd.DataFrame, y_train: pd.Series):
    logging.info("Running FIT on sentiment_volume (TS-CV + calibration)")

    param_grid = {
        "num_leaves":        [7, 15, 31],
        "min_child_samples": [20, 50, 100],
    }

    splitter = TimeSeriesSplit(n_splits=5)

    base = LGBMClassifier(**LGBM_PARAMS)

    grid = GridSearchCV(
            estimator=base,
            param_grid=param_grid,
            cv=splitter,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1,
    ).fit(X_train, y_train)

    logging.info("Best TS-CV params ⇒ %s", grid.best_params_)

    best_lgb = grid.best_estimator_

    iso = CalibratedClassifierCV(
        best_lgb, method="isotonic", cv=TimeSeriesSplit(n_splits=3)
    ).fit(X_train, y_train)

    _best = best_lgb.get_params()

    def _params(deep=True):
        return {k: _best[k] for k in LGBM_PARAMS}

    iso.get_params = _params
    return iso

def predict(model: LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    logging.info("Running PREDICT on sentiment_volume")
    return model.predict_proba(X)[:, 1]
    
