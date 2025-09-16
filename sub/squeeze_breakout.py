# squeeze_breakout.py
import numpy as np
import pandas as pd
from typing import List
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

# --------------------------------------------------------------------------------------
# Features consumed from your master dataset + engineered here
# (Engineered features are created by compute_labels to keep compatibility with your pipeline)
# --------------------------------------------------------------------------------------
FEATURES: List[str] = [
    # base columns expected to exist in your dataset
    "close","high","low","atr","ema_21",
    "bollinger_upper","bollinger_lower","bollinger_percB",
    "returns_1","returns_3","std_5","std_10",
    # engineered in this module
    "feat_bb_bw",           # Bollinger bandwidth / close
    "feat_keltner_bw",      # Keltner bandwidth / close (uses ema_21 ± k_atr*atr)
    "feat_bw_ratio",        # bb_bw / keltner_bw
    "feat_entropy_20",      # rolling Shannon entropy of returns_1 (20)
    "feat_range_atr",       # (high_low_range or (high-low)) / atr
    "feat_dc_up_dist",      # (donchian_up - close) / (atr/close)
    "feat_dc_dn_dist",      # (close - donchian_dn) / (atr/close)
    "feat_recent_squeeze",  # 0/1 recent squeeze in last L bars
    "feat_squeeze_dur"      # length of current squeeze run in bars
]

# --------------------------------------------------------------------------------------
# Labeling hyperparameters (directional "upside breakout after squeeze")
# --------------------------------------------------------------------------------------
BB_N                 = 20      # Bollinger lookback (expects upper/lower already computed by your feature maker)
DC_N                 = 20      # Donchian lookback for breakout levels
ENT_N                = 20      # window for entropy on 1-bar returns
SQUEEZE_ROLL_Q       = 60      # window over which to compute low-quantile of bandwidth
SQUEEZE_Q            = 0.20    # "low" quantile threshold for bandwidth
KELTNER_ATR_MULT     = 1.5     # k for Keltner channel (ema_21 ± k*atr)
LEAD_SQUEEZE_BARS    = 6       # how many bars back we still consider a squeeze "recent"
BREAKOUT_H           = 5       # look-ahead horizon bars to detect breakout
MARGIN_ATR           = 0.25    # extra margin above Donchian using ATR (in price units via atr)

# --------------------------------------------------------------------------------------
# Purged CV (gap-aware) used across your other submodels
# --------------------------------------------------------------------------------------
class PurgedTimeSeriesCV:
    def __init__(self, n_splits: int = 5, gap: int = 1):
        self.n_splits = n_splits
        self.gap = gap

    def split(self, X, y=None, groups=None):
        n = len(X)
        base = TimeSeriesSplit(n_splits=self.n_splits)
        idx = np.arange(n)
        for tr, te in base.split(idx):
            # Purge 'gap' observations from the end of train and the start of test
            tr_max = tr[-1] - self.gap
            te_min = te[0] + self.gap
            tr_mask = tr <= tr_max
            te_mask = te >= te_min
            tr_idx = tr[tr_mask]
            te_idx = te[te_mask]
            if len(tr_idx) == 0 or len(te_idx) == 0:
                continue
            yield tr_idx, te_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def _safe_div(a, b):
    b = np.asarray(b, dtype=float)
    return np.where(np.abs(b) > 1e-12, np.asarray(a, dtype=float) / b, 0.0)

def _forward_rolling_extreme(arr: np.ndarray, window: int, is_max: bool = True) -> np.ndarray:
    """
    Computes forward-looking rolling max/min over the *next* `window` elements (including the next bar).
    Returns an array aligned with arr (same length). The last (window-1) entries use progressively
    smaller windows; this is fine for labeling but those labels will typically be dropped later.
    """
    a = np.asarray(arr, dtype=float)
    rev = a[::-1]
    ser = pd.Series(rev)
    roll = ser.rolling(window=window, min_periods=1).max() if is_max else ser.rolling(window=window, min_periods=1).min()
    out = roll.values[::-1]
    return out

def _rolling_entropy(x: pd.Series, window: int = ENT_N, bins: int = 10) -> pd.Series:
    """
    Shannon entropy of the distribution of returns over a rolling window.
    """
    x = pd.Series(x).astype(float)
    def _ent(a: np.ndarray) -> float:
        if len(a) == 0 or np.all(~np.isfinite(a)):
            return np.nan
        # robust histogram over finite values
        a = a[np.isfinite(a)]
        if len(a) == 0:
            return np.nan
        hist, _ = np.histogram(a, bins=bins, density=True)
        p = hist / (hist.sum() + 1e-12)
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())
    return x.rolling(window=window, min_periods=1).apply(_ent, raw=True)

def _donchian(high: pd.Series, low: pd.Series, n: int = DC_N):
    dc_up = high.rolling(window=n, min_periods=1).max()
    dc_dn = low.rolling(window=n, min_periods=1).min()
    return dc_up, dc_dn

# --------------------------------------------------------------------------------------
# Feature + label construction
# --------------------------------------------------------------------------------------
def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds engineered features and the binary 'label' column:
    label = 1 when a *recent squeeze* (low BB bandwidth & within Keltner) precedes an
            *upside breakout* beyond Donchian upper + ATR margin within BREAKOUT_H bars.
    Else label = 0.
    """
    d = df.copy()

    # Base columns (robust defaults if missing)
    close = d.get("close", pd.Series(np.nan, index=d.index).astype(float))
    high  = d.get("high",  pd.Series(np.nan, index=d.index).astype(float))
    low   = d.get("low",   pd.Series(np.nan, index=d.index).astype(float))
    atr   = d.get("atr",   pd.Series(np.nan, index=d.index).astype(float))
    ema21 = d.get("ema_21",pd.Series(np.nan, index=d.index).astype(float))

    # --- Bollinger bandwidth (normalize by price to be scale-free) ---
    bb_up = d.get("bollinger_upper", pd.Series(np.nan, index=d.index).astype(float))
    bb_dn = d.get("bollinger_lower", pd.Series(np.nan, index=d.index).astype(float))
    bb_bw = _safe_div((bb_up - bb_dn), close)
    d["feat_bb_bw"] = bb_bw

    # --- Keltner bandwidth (ema_21 ± K*ATR) ---
    kel_up = ema21 + KELTNER_ATR_MULT * atr
    kel_dn = ema21 - KELTNER_ATR_MULT * atr
    kel_bw = _safe_div((kel_up - kel_dn), close)  # = 2*K*atr/close
    d["feat_keltner_bw"] = kel_bw

    # --- Bandwidth ratio + compression proxy ---
    bw_ratio = _safe_div(bb_bw, kel_bw)  # < 1 implies BB inside Keltner (classic "squeeze")
    d["feat_bw_ratio"] = bw_ratio

    # --- Entropy of 1-bar returns ---
    ret1 = d.get("returns_1", (close.pct_change().fillna(0.0)*100.0))
    d["feat_entropy_20"] = _rolling_entropy(pd.Series(ret1), window=ENT_N, bins=10)

    # --- Range / ATR compression (use provided high_low_range if present) ---
    hl_range = d.get("high_low_range", (high - low).abs())
    d["feat_range_atr"] = _safe_div(hl_range, atr)

    # --- Donchian channels (from *past* highs/lows) ---
    dc_up, dc_dn = _donchian(high, low, n=DC_N)
    # normalized distances to levels using ATR normalized by price
    d["feat_dc_up_dist"] = _safe_div((dc_up - close), atr)
    d["feat_dc_dn_dist"] = _safe_div((close - dc_dn), atr)

    # --- Squeeze definition ---
    # 1) "Inside Keltner": bb_bw < kel_bw
    # 2) "Low bandwidth quantile" over last SQUEEZE_ROLL_Q bars
    low_q = d["feat_bb_bw"].rolling(window=SQUEEZE_ROLL_Q, min_periods=5).quantile(SQUEEZE_Q)
    squeeze_now = (d["feat_bb_bw"] < d["feat_keltner_bw"]) & (d["feat_bb_bw"] <= low_q)
    squeeze_now = squeeze_now.fillna(False)

    # Recent squeeze feature (last LEAD_SQUEEZE_BARS bars inclusive of t)
    recent_squeeze = squeeze_now.rolling(window=LEAD_SQUEEZE_BARS, min_periods=1).max().astype(int)
    d["feat_recent_squeeze"] = recent_squeeze

    # Squeeze duration (length of current consecutive squeeze run)
    squeeze_dur = np.zeros(len(d), dtype=int)
    run = 0
    for i, s in enumerate(squeeze_now.values):
        if s:
            run += 1
        else:
            run = 0
        squeeze_dur[i] = run
    d["feat_squeeze_dur"] = squeeze_dur

    # --- Breakout labeling (UP-SIDE) ---
    # Look ahead BREAKOUT_H bars for a high/close that exceeds Donchian upper + ATR margin
    margin = MARGIN_ATR * atr
    # Use high for breakout detection (intrabar)
    future_max = _forward_rolling_extreme(high.values, window=BREAKOUT_H, is_max=True)
    # Detect upside breakout
    up_break = (future_max >= (dc_up.values + margin.values))

    # Final label: need a recent squeeze, then an upside breakout within H bars
    lab = (recent_squeeze.values.astype(bool) & up_break.astype(bool)).astype(int)
    d["label"] = lab

    return d.reset_index(drop=True)

# --------------------------------------------------------------------------------------
# Model training / inference
# --------------------------------------------------------------------------------------
def fit(X: np.ndarray, y: np.ndarray):
    """
    Train an XGBoost classifier and calibrate with isotonic regression using a purged CV.
    """
    base = XGBClassifier(
        n_estimators=450,
        learning_rate=0.035,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
    )
    # Small gap to respect BREAKOUT_H labeling horizon
    splitter = PurgedTimeSeriesCV(n_splits=5, gap=max(1, BREAKOUT_H // 2))
    model = CalibratedClassifierCV(base, method="isotonic", cv=splitter, n_jobs=-1)
    model.fit(X, y)
    return model

def predict(model, X: np.ndarray) -> np.ndarray:
    """
    Return calibrated probabilities for 'upside breakout following a recent squeeze'.
    """
    proba = model.predict_proba(X)[:, 1]
    print("Running PREDICT on squeeze_breakout.py")
    return np.asarray(proba, dtype=float)
