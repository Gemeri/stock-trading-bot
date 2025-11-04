# mean_reversion.py
import numpy as np
import pandas as pd
from typing import List
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

FEATURES: List[str] = [
    "close","atr","bollinger_percB","bollinger_upper","bollinger_lower",
    "ema_9","ema_21","ema_50","rsi","rsi_zscore","std_5",
    # engineered
    "feat_dist_from_band","feat_atr_slope"
]

K_ATR: float = 0.25  # magnitude floor

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
    percB = d.get("bollinger_percB", pd.Series(np.nan, index=d.index)).astype(float)
    scale = _safe_div(atr, close)

    # engineered
    mid = (d.get("bollinger_upper",0)+d.get("bollinger_lower",0))/2.0
    d["feat_dist_from_band"] = _safe_div(close - mid, atr)
    d["feat_atr_slope"] = pd.Series(atr).pct_change().fillna(0.0)

    ret1 = (close.shift(-1)-close)/close
    ret2 = (close.shift(-2)-close)/close
    fwd  = np.nanmax(np.vstack([ret1.values, ret2.values]), axis=0)
    fwdn = np.nanmin(np.vstack([ret1.values, ret2.values]), axis=0)

    lab = pd.Series(np.nan, index=d.index)
    # upward reversion from lower band
    lab[(percB < 0.10) & (fwd >  K_ATR*scale)] = 1
    # downward reversion from upper band
    lab[(percB > 0.90) & (fwdn < -K_ATR*scale)] = 0
    d["label"] = lab
    return d.reset_index(drop=True)

def fit(X: np.ndarray, y: np.ndarray):
    base = RandomForestClassifier(
        n_estimators=400, max_depth=8, min_samples_leaf=5,
        max_features="sqrt", n_jobs=-1, random_state=42, class_weight="balanced"
    )
    splitter = PurgedTimeSeriesCV(n_splits=5, gap=1)
    model = CalibratedClassifierCV(base, method="isotonic", cv=splitter, n_jobs=-1)
    model.fit(X,y)
    return model

def predict(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:,1]