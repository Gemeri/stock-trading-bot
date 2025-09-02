import numpy as np
import pandas as pd
import logging

from sub.common import compute_meta_labels, USE_META_LABEL
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV

# Feature set used by this strategy (ensure your X arrays/frames are in this order)
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

# Fixed model hyperparameters (replacing the old grid search)
MODEL_PARAMS = {
    'n_estimators': 250,
    'max_depth': 10,
    'min_samples_leaf': 1,
    'max_features': len(FEATURES),  # use all provided features
}


def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute binary labels for training.
    If USE_META_LABEL is True, delegate to meta-label helper (compatible with previous behavior).
    Otherwise, label when move >= ATR occurs in the next 1-2 steps, conditioned on position vs Bollinger mean band.
    """
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

    # Keep only rows where all features and the label are present
    return df.dropna(subset=FEATURES + ['label']).reset_index(drop=True)


def sharpe(returns: pd.Series) -> float:
    if len(returns) == 0 or returns.std() == 0:
        return np.nan
    return returns.mean() / returns.std() * np.sqrt(252)


def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    return ((equity - roll_max) / roll_max).min()


def fit(X_train: np.ndarray, y_train: np.ndarray) -> CalibratedClassifierCV:
    """
    Fit a RandomForest on the provided features with fixed hyperparameters
    and wrap it in an isotonic CalibratedClassifier using time-series CV.
    """
    logging.info("Running FIT on mean_reversion (fixed RF params + calibration)")

    # Base RandomForest using fixed params (no GridSearch)
    rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        **MODEL_PARAMS,
    )

    # Calibrate probabilities with time-series CV
    calib = CalibratedClassifierCV(
        rf, method="isotonic", cv=TimeSeriesSplit(n_splits=3)
    ).fit(X_train, y_train)

    # Expose fixed RF params for external inspection (mirrors prior behavior of exposing best params)
    def _get_params(deep: bool = True):
        return MODEL_PARAMS.copy()

    calib.get_params = _get_params  # type: ignore[attr-defined]

    logging.info("Model trained with params ⇒ %s", MODEL_PARAMS)
    return calib


def predict(model: CalibratedClassifierCV, X: np.ndarray) -> np.ndarray:
    """
    Return calibrated probability of the positive class.
    """
    logging.info("Running PREDICT on mean_reversion")
    return model.predict_proba(X)[:, 1]
