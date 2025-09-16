# momentum.py
# Replace sklearn's HistGradientBoosting with XGBoost + isotonic calibration
# pip install xgboost

import numpy as np
import pandas as pd
from typing import List

from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

FEATURES: List[str] = [
    "close","atr","ema_9","ema_21","ema_50","rsi","rsi_zscore",
    "macd_line","macd_signal","macd_histogram","macd_cross",
    "returns_1","returns_3","returns_5","std_5","std_10",
    "candle_body_ratio","wick_dominance","gap_vs_prev",
    "hour_sin","hour_cos",
    # engineered
    "feat_ema_diff","feat_mom_3norm","feat_mom_5norm","feat_body_over_atr"
]

LABEL_K: float = 0.30  # minimum move as k*ATR/close


class PurgedTimeSeriesCV:
    """
    Time-series CV with a simple purge 'gap' between train and test
    to reduce leakage from adjacent observations.
    """
    def __init__(self, n_splits=5, gap=1):
        self.n_splits = n_splits
        self.gap = int(gap)

    def split(self, X, y=None, groups=None):
        n = len(X)
        base = TimeSeriesSplit(n_splits=self.n_splits)
        for tr, te in base.split(np.arange(n)):
            # purge edges by 'gap'
            tr = tr[tr <= tr[-1] - self.gap]
            te = te[te >= te[0] + self.gap]
            if len(tr) and len(te):
                yield tr, te

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def _safe_div(a, b):
    b = np.asarray(b, float)
    return np.where(np.abs(b) > 1e-12, np.asarray(a, float) / b, 0.0)


def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds engineered features and a binary label:
    1 if next period return > LABEL_K * (ATR/close), 0 if < -LABEL_K * (ATR/close).
    Uncertain rows remain NaN and should be dropped before training.
    """
    d = df.copy()
    close = d["close"].astype(float)
    atr = d.get("atr", pd.Series(np.nan, index=d.index)).astype(float)
    scale = _safe_div(atr, close)

    # engineered features
    d["feat_ema_diff"] = _safe_div(d.get("ema_9", 0) - d.get("ema_21", 0), close)
    d["feat_mom_3norm"] = _safe_div(d.get("returns_3", 0), scale)
    d["feat_mom_5norm"] = _safe_div(d.get("returns_5", 0), scale)
    d["feat_body_over_atr"] = _safe_div(d.get("candle_body_ratio", 0) * close, atr)

    ret1 = (close.shift(-1) - close) / close
    up = (ret1 > LABEL_K * scale)
    dn = (ret1 < -LABEL_K * scale)

    lab = pd.Series(np.nan, index=d.index)
    lab[up] = 1
    lab[dn] = 0
    d["label"] = lab

    return d.reset_index(drop=True)


def fit(X: np.ndarray, y: np.ndarray):
    """
    Train an XGBoost classifier wrapped with isotonic calibration using
    a purged time-series cross-validation splitter.
    """
    # XGBoost can handle NaNs in features; ensure y has no NaNs
    # (assume caller has filtered rows with NaN labels)
    y = np.asarray(y)

    # Handle class imbalance if present
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    scale_pos_weight = float(neg / pos) if pos > 0 else 1.0

    base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=400,
        learning_rate=0.06,
        max_depth=4,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=20,   # not identical to sklearn's min_samples_leaf
        reg_alpha=0.0,
        reg_lambda=0.0,
        tree_method="hist",    # switch to "gpu_hist" if you have a GPU
        n_jobs=-1,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        verbosity=0,
    )

    splitter = PurgedTimeSeriesCV(n_splits=5, gap=1)
    model = CalibratedClassifierCV(base, method="isotonic", cv=splitter, n_jobs=-1)
    model.fit(X, y)
    return model


def predict(model, X: np.ndarray) -> np.ndarray:
    """
    Return calibrated probabilities P(y=1 | X).
    """
    print("Running PREDICT on momentum.py")
    return model.predict_proba(X)[:, 1]
