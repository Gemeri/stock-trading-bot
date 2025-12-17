# sentiment_volume.py
import numpy as np
import pandas as pd
from typing import List
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

FEATURES: List[str] = [
    "close","atr","sentiment","volume","volume_zscore",
    "hour_sin","hour_cos",
    "vwap","transactions","greed_index","news_count","news_volume_z","d_sentiment",
    # engineered
    "feat_sent_ema","feat_sent_x_volz","feat_price_over_atr"
]

K_ATR: float = 0.15  # small floor; sentiment often reacts early

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
    scale = _safe_div(atr, close)

    # engineered
    s = d.get("sentiment", pd.Series(0.0, index=d.index)).astype(float)
    d["feat_sent_ema"] = s.ewm(span=10, adjust=False).mean()
    d["feat_sent_x_volz"] = s * d.get("volume_zscore", pd.Series(0.0, index=d.index))
    d["feat_price_over_atr"] = _safe_div(d.get("close",0)-d.get("vwap",0), atr)

    ret1 = (close.shift(-1)-close)/close
    up   = (ret1 >  K_ATR*scale)
    dn   = (ret1 < -K_ATR*scale)
    lab  = pd.Series(np.nan, index=d.index)
    lab[up]=1; lab[dn]=0
    d["label"] = lab
    return d.reset_index(drop=True)

def fit(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
    base = XGBClassifier(
        n_estimators=450, learning_rate=0.035, max_depth=4,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", random_state=42
    )
    splitter = PurgedTimeSeriesCV(n_splits=5, gap=1)
    model = CalibratedClassifierCV(base, method="isotonic", cv=splitter, n_jobs=-1)
    model.fit(X, y, sample_weight=sample_weight)
    return model

def predict(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:,1]
