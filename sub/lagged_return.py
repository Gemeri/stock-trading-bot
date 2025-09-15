# lagged_return.py
import numpy as np
import pandas as pd
from typing import List
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

FEATURES: List[str] = [
    "close","atr",
    "lagged_close_1","lagged_close_2","lagged_close_3","lagged_close_5","lagged_close_10",
    "returns_1","returns_3","returns_5",
    "ema_9","ema_21","ema_50","rsi","rsi_zscore",
    "std_5","std_10","candle_body_ratio","wick_dominance","gap_vs_prev",
    # engineered
    "feat_ret_norm_3","feat_ret_norm_5"
]

K_ATR: float = 0.15

class PurgedTimeSeriesCV:
    def __init__(self, n_splits=5, gap=1): self.n_splits=n_splits; self.gap=gap
    def split(self, X, y=None, groups=None):
        n=len(X); base=TimeSeriesSplit(n_splits=self.n_splits)
        for tr,te in base.split(np.arange(n)):
            tr = tr[tr<=tr[-1]-self.gap]; te = te[te>=te[0]+self.gap]
            if len(tr) and len(te): yield tr,te
    def get_n_splits(self, X=None, y=None, groups=None): return self.n_splits

def _safe_div(a,b):
    b=np.asarray(b,float)
    return np.where(np.abs(b)>1e-12, np.asarray(a,float)/b, 0.0)

def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    close = d["close"].astype(float)
    atr   = d.get("atr", pd.Series(np.nan, index=d.index)).astype(float)

    # engineered
    scale = _safe_div(atr, close)
    d["feat_ret_norm_3"] = _safe_div(d.get("returns_3",0), scale)
    d["feat_ret_norm_5"] = _safe_div(d.get("returns_5",0), scale)

    ret1 = (close.shift(-1)-close)/close
    up   = (ret1 >  K_ATR*scale)
    dn   = (ret1 < -K_ATR*scale)
    lab  = pd.Series(np.nan, index=d.index)
    lab[up]=1; lab[dn]=0
    d["label"] = lab
    return d.reset_index(drop=True)

def fit(X: np.ndarray, y: np.ndarray):
    base = XGBClassifier(
        n_estimators=500, learning_rate=0.03, max_depth=4,
        subsample=0.8, colsample_bytree=0.9, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", random_state=42
    )
    splitter = PurgedTimeSeriesCV(n_splits=5, gap=1)
    model = CalibratedClassifierCV(base, method="isotonic", cv=splitter, n_jobs=-1)
    model.fit(X,y)
    return model

def predict(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:,1]