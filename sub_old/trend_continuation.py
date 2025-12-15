# trend_continuation.py
import numpy as np
import pandas as pd
from typing import List
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

FEATURES: List[str] = [
    "close","atr","ema_9","ema_21","ema_50","ema_200",
    "adx","macd_line","macd_signal","macd_histogram",
    "returns_1","returns_3","std_5","std_10",
    "obv","month","days_since_high","days_since_low",
    # engineered
    "feat_trend_regime","feat_ema_slope"
]

K_ATR: float = 0.25  # continuation floor

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
    d["feat_trend_regime"] = (d.get("ema_21",0) > d.get("ema_50",0)).astype(int)
    d["feat_ema_slope"]    = pd.Series(d.get("ema_21",0)).pct_change().fillna(0.0)

    fwd = (close.shift(-1)-close)/close
    up  = (fwd >  K_ATR*scale)
    dn  = (fwd < -K_ATR*scale)
    lab = pd.Series(np.nan, index=d.index); lab[up]=1; lab[dn]=0
    d["label"] = lab
    return d.reset_index(drop=True)

def fit(X: np.ndarray, y: np.ndarray):
    # class weights for balance
    pos = float((y==1).sum()); neg = float((y==0).sum())
    if pos > 0 and neg > 0:
        w1 = neg/max(pos,1.0); w0 = 1.0
        class_weights = [w0, w1]
    else:
        class_weights = None

    base = CatBoostClassifier(
        iterations=600, depth=6, learning_rate=0.05,
        l2_leaf_reg=3, bagging_temperature=0.5,
        loss_function="Logloss", eval_metric="AUC",
        random_seed=42, verbose=False,
        class_weights=class_weights
    )
    splitter = PurgedTimeSeriesCV(n_splits=5, gap=1)
    model = CalibratedClassifierCV(base, method="isotonic", cv=splitter)
    model.fit(X,y)
    return model

def predict(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:,1]