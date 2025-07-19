import logging

import numpy as np

from xgboost import XGBClassifier
from sub.common import compute_meta_labels, USE_META_LABEL
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV

FEATURES = [
    'rsi', 'momentum', 'roc',
    'macd_histogram', 'macd_line', 'macd_signal',
    'rsi_zscore', 'macd_cross', 'macd_hist_flip'
]

PARAM_GRID = {
    'n_estimators':     [250, 500],
    'max_depth':        [6, 10],
    'learning_rate':    [0.01, 0.1],
    'subsample':        [0.8],
    'colsample_bytree': [0.8]
}


def compute_labels(df):
    df = df.copy()

    if USE_META_LABEL:
        return compute_meta_labels(df).rename(columns={'meta_label': 'label'})

    df['close_t1'] = df['close'].shift(-1)
    df['label']   = (df['close_t1'] > df['close']).astype(int)

    return df.dropna(subset=FEATURES + ['label']).reset_index(drop=True)

def sharpe(returns):
    if len(returns) == 0 or returns.std() == 0:
        return np.nan
    return returns.mean() / returns.std() * np.sqrt(252)

def max_drawdown(equity):
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return drawdown.min()


def fit(X_train: np.ndarray, y_train: np.ndarray):
    logging.info("Running FIT on momentum (TS-CV + calibration)")

    splitter = TimeSeriesSplit(n_splits=5)

    base = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
        random_state=42,
        n_jobs=-1,
    )

    grid = GridSearchCV(
        estimator=base,
        param_grid=PARAM_GRID,
        cv=splitter,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    ).fit(X_train, y_train)

    best_xgb = grid.best_estimator_
    logging.info("Best params ⇒ %s", grid.best_params_)

    iso = CalibratedClassifierCV(
        best_xgb, method="isotonic", cv=TimeSeriesSplit(n_splits=3)
    ).fit(X_train, y_train)

    _best = best_xgb.get_params()

    def _get_params(deep=True):
        return {k: _best[k] for k in PARAM_GRID}

    iso.get_params = _get_params
    return iso



def predict(model: XGBClassifier, X: np.ndarray) -> np.ndarray:
    logging.info("Ruinning PREDICT on momentum")
    return model.predict_proba(X)[:, 1]