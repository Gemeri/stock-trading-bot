# momentum.py
import numpy as np
import pandas as pd
from typing import List
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

FEATURES: List[str] = [
    "close","atr","ema_9","ema_21","ema_50","rsi","rsi_zscore",
    "macd_line","macd_signal","macd_histogram",
    "returns_1","returns_3","returns_5","std_5","std_10",
    "candle_body_ratio","wick_dominance","gap_vs_prev",
    "hour_sin","hour_cos",
    "open","high","low","price_change","roc","day_of_week_sin","day_of_week_cos",
    # engineered
    "feat_ema_diff","feat_mom_3norm","feat_mom_5norm","feat_body_over_atr"
]

LABEL_K: float = 0.30  # minimum move as k*ATR/close

class PurgedTimeSeriesCV:
    def __init__(self, n_splits=5, gap=1):
        self.n_splits = n_splits; self.gap = int(gap)
    def split(self, X, y=None, groups=None):
        n = len(X)
        base = TimeSeriesSplit(n_splits=self.n_splits)
        for tr, te in base.split(np.arange(n)):
            tr = tr[tr <= tr[-1] - self.gap]
            te = te[te >= te[0] + self.gap]
            if len(tr) and len(te): yield tr, te
    def get_n_splits(self, X=None, y=None, groups=None): return self.n_splits

def _safe_div(a,b):
    b = np.asarray(b, float)
    return np.where(np.abs(b)>1e-12, np.asarray(a,float)/b, 0.0)

def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    close = d["close"].astype(float)
    atr   = d.get("atr", pd.Series(np.nan, index=d.index)).astype(float)
    scale = _safe_div(atr, close)

    # engineered
    d["feat_ema_diff"]     = _safe_div(d.get("ema_9",0)-d.get("ema_21",0), close)
    d["feat_mom_3norm"]    = _safe_div(d.get("returns_3",0), scale)
    d["feat_mom_5norm"]    = _safe_div(d.get("returns_5",0), scale)
    d["feat_body_over_atr"]= _safe_div(d.get("candle_body_ratio",0)*close, atr)

    ret1 = (close.shift(-1)-close)/close
    up   = (ret1 >  LABEL_K*scale)
    dn   = (ret1 < -LABEL_K*scale)
    lab  = pd.Series(np.nan, index=d.index)
    lab[up] = 1; lab[dn] = 0
    d["label"] = lab
    return d.reset_index(drop=True)

def fit(X: np.ndarray, y: np.ndarray):
    base = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.06, max_iter=400,
        min_samples_leaf=20, l2_regularization=0.0, random_state=42
    )
    splitter = PurgedTimeSeriesCV(n_splits=5, gap=1)
    model = CalibratedClassifierCV(base, method="isotonic", cv=splitter, n_jobs=-1)
    model.fit(X, y)
    return model

def predict(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:,1]