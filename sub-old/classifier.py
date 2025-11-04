# classifier.py
import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

# Horizons used by Main.py
MULTI_HORIZONS: List[int] = [1, 3, 5, 8]

# Core/source features expected in your CSVs + engineered ones we add here
FEATURES: List[str] = [
    # base (from dataset)
    "close","vwap","atr","ema_9","ema_21","ema_50","ema_200",
    "rsi","rsi_zscore","macd_line","macd_signal","macd_histogram",
    "macd_cross","macd_hist_flip","adx","obv",
    "bollinger_upper","bollinger_lower","bollinger_percB",
    "returns_1","returns_3","returns_5",
    "std_5","std_10","volume","volume_zscore",
    "candle_body_ratio","wick_dominance","gap_vs_prev",
    "hour_sin","hour_cos","day_of_week_sin","day_of_week_cos",
    # engineered (computed below)
    "feat_ema_diff", "feat_price_vwap_atr",
    "feat_macd_pos", "feat_mom_3norm", "feat_mom_5norm"
]

# ATR floor for label construction (fraction of ATR/close)
LABEL_K: float = 0.20  # tune per timeframe if needed

# ---------------- utilities ---------------- #

class PurgedTimeSeriesCV:
    """Purged, gap-aware CV for time series."""
    def __init__(self, n_splits: int = 5, gap: int = 1):
        self.n_splits = n_splits
        self.gap = max(0, int(gap))

    def split(self, X, y=None, groups=None):
        n = len(X)
        base = TimeSeriesSplit(n_splits=self.n_splits)
        for tr_idx, te_idx in base.split(np.arange(n)):
            # purge: drop last `gap` from train, first `gap` from test
            if len(tr_idx) == 0 or len(te_idx) == 0:
                continue
            tr_end = tr_idx[-1] - self.gap
            te_start = te_idx[0] + self.gap
            tr_idx = tr_idx[tr_idx <= tr_end]
            te_idx = te_idx[te_idx >= te_start]
            if len(tr_idx) == 0 or len(te_idx) == 0:
                continue
            yield tr_idx, te_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

def _safe_div(a, b):
    b = np.asarray(b, dtype=float)
    return np.where(np.abs(b) > 1e-12, np.asarray(a, dtype=float) / b, 0.0)

def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # engineered features (robust to missing cols)
    close = d.get("close", pd.Series(np.nan, index=d.index))
    atr   = d.get("atr",   pd.Series(np.nan, index=d.index))
    d["feat_ema_diff"]       = _safe_div(d.get("ema_9",0) - d.get("ema_21",0), close)
    d["feat_price_vwap_atr"] = _safe_div(d.get("close",0) - d.get("vwap",0), atr)
    d["feat_macd_pos"]       = np.sign(d.get("macd_line",0) - d.get("macd_signal",0))
    d["feat_mom_3norm"]      = _safe_div(d.get("returns_3",0), _safe_div(atr, close))
    d["feat_mom_5norm"]      = _safe_div(d.get("returns_5",0), _safe_div(atr, close))
    return d

# ---------------- labels & API ---------------- #

def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds directional labels with a small ATR-relative floor to remove micro-noise.
    For each h: label_h{h} = 1 if ret_h > +k*ATR/close ; 0 if ret_h < -k*ATR/close ; else NaN.
    """
    d = _engineer(df)
    close = d["close"].astype(float)
    atr   = d.get("atr", pd.Series(np.nan, index=d.index)).astype(float)
    scale = _safe_div(atr, close)

    for h in MULTI_HORIZONS:
        ret_h = (close.shift(-h) - close) / close
        up    = (ret_h >  LABEL_K * scale)
        down  = (ret_h < -LABEL_K * scale)
        lab   = pd.Series(np.nan, index=d.index)
        lab[up]   = 1
        lab[down] = 0
        d[f"label_h{h}"] = lab

    return d.reset_index(drop=True)

def fit(X_train: np.ndarray, y_train: np.ndarray, horizon: Optional[int] = None):
    hgap = 1 if horizon is None else int(horizon)
    base = XGBClassifier(
        n_estimators=400, learning_rate=0.03, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", random_state=42
    )
    splitter = PurgedTimeSeriesCV(n_splits=5, gap=hgap)
    model = CalibratedClassifierCV(base, method="isotonic", cv=splitter, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def predict(model, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)[:, 1]
    return np.asarray(proba, dtype=float)
