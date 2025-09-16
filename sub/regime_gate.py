# regime_gate.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple
from sklearn.mixture import GaussianMixture
from catboost import CatBoostClassifier

# ======================================================================================
# Public API
# ======================================================================================

# We ONLY reference columns that already exist in your CSV and derive the rest here.
# Base columns required from your CSV:
#   close, atr, rsi, adx, macd_histogram, ema_200, returns_1, std_10
#
# Engineered here:
#   feat_adx_tier, feat_rsi_var_14, feat_acf1, feat_acf3, feat_atr_pct_252,
#   feat_macd_sign_stab_20, feat_realized_vol_10, feat_realized_vol_30,
#   feat_vol_of_vol_30, feat_ret_mean_10, feat_ret_mean_30, feat_above_ema200
#
# Model outputs (probabilities):
#   p_trend, p_high_vol, p_risk_on

FEATURES: List[str] = [
    # --- CSV columns we actually use ---
    "close", "atr", "rsi", "adx", "macd_histogram", "ema_200", "returns_1", "std_10",

    # --- Engineered in this module (computed from the above CSV columns) ---
    "feat_adx_tier",
    "feat_rsi_var_14",
    "feat_acf1",
    "feat_acf3",
    "feat_atr_pct_252",
    "feat_macd_sign_stab_20",
    "feat_realized_vol_10",   # = std_10 (alias to keep interface consistent)
    "feat_realized_vol_30",
    "feat_vol_of_vol_30",
    "feat_ret_mean_10",
    "feat_ret_mean_30",
    "feat_above_ema200",
]

OUTPUT_COLUMNS: List[str] = ["p_trend", "p_high_vol", "p_risk_on"]

@dataclass
class RegimeModel:
    trend_head: CatBoostClassifier
    vol_head: CatBoostClassifier
    risk_head: CatBoostClassifier
    output_columns: List[str]

# ======================================================================================
# Helpers
# ======================================================================================

def _safe_div(a, b):
    b = np.asarray(b, dtype=float)
    return np.where(np.abs(b) > 1e-12, np.asarray(a, dtype=float) / b, 0.0)

def _rolling_autocorr(x: pd.Series, lag: int, window: int = 50) -> pd.Series:
    x = pd.Series(x).astype(float)
    x_dm = x - x.rolling(window, min_periods=1).mean()
    x_lag = x_dm.shift(lag)
    num = (x_dm * x_lag).rolling(window, min_periods=10).mean()
    den = (x_dm.rolling(window, min_periods=10).std() * x_lag.rolling(window, min_periods=10).std())
    return pd.Series(_safe_div(num, den), index=x.index).fillna(0.0)

def _percent_rank(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=max(5, window // 5)).apply(
        lambda a: 0.0 if len(a) == 0 else (pd.Series(a).rank(pct=True).iloc[-1]), raw=False
    ).fillna(0.5)

def _sign_stability(s: pd.Series, window: int = 20) -> pd.Series:
    sign = np.sign(pd.Series(s))
    # fraction of last N bars where sign equals the last sign in the window
    ref = sign.rolling(window, min_periods=1).apply(lambda a: a[-1], raw=True)
    agree = (sign == ref).astype(float)
    return agree.rolling(window, min_periods=3).mean().fillna(0.5)

def _gmm_binary_labels(X: np.ndarray, prefer_higher_idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unsupervised 2-cluster GMM. Returns (labels, proba_for_positive_component).
    If prefer_higher_idx is provided, 'positive' is the component with a higher mean
    on that particular feature (column index w.r.t X).
    """
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
    gmm.fit(X)
    proba = gmm.predict_proba(X)  # (n,2)
    if prefer_higher_idx is not None:
        pos_comp = int(np.argmax(gmm.means_[:, prefer_higher_idx]))
    else:
        pos_comp = int(np.argmax(gmm.means_.mean(axis=1)))
    labels = (np.argmax(proba, axis=1) == pos_comp).astype(int)
    p_pos = proba[:, pos_comp]
    return labels, p_pos

def _ts_last_split(n: int, valid_frac: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    split = int(max(1, n * (1 - valid_frac)))
    return np.arange(0, split), np.arange(split, n)

# ======================================================================================
# Feature + pseudo-label construction (only computes what is NOT already in the CSV)
# ======================================================================================

def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds engineered regime features (derived ONLY from existing CSV columns)
    and unsupervised pseudo-labels:
      - label_trend    (trend=1 vs chop=0)
      - label_highvol  (high-vol=1 vs low-vol=0)
      - label_riskon   (risk-on=1 vs risk-off=0)
    """
    d = df.copy()

    # --- pull ONLY what exists in your CSV ---
    close = d["close"].astype(float)
    atr = d["atr"].astype(float)
    rsi = d["rsi"].astype(float)
    adx = d["adx"].astype(float)
    macd_hist = d["macd_histogram"].astype(float)
    ema200 = d["ema_200"].astype(float)
    ret1 = d["returns_1"].astype(float)
    std10 = d["std_10"].astype(float)

    # --- engineered regime features (derived; not re-computing CSV fields) ---
    # ADX tiers (0..2) via quantiles
    d["feat_adx_tier"] = pd.qcut(adx.fillna(adx.median()), q=3, labels=False, duplicates="drop") \
                            .astype(float).fillna(1.0)

    # RSI variance (14)
    d["feat_rsi_var_14"] = rsi.rolling(14, min_periods=5).var().fillna(method="bfill").fillna(0.0)

    # Autocorrelations of 1-bar returns (trendiness proxy)
    d["feat_acf1"] = _rolling_autocorr(ret1, lag=1, window=50)
    d["feat_acf3"] = _rolling_autocorr(ret1, lag=3, window=50)

    # ATR percentile over 252 bars (normalize by price level)
    atr_frac = _safe_div(atr, close)
    d["feat_atr_pct_252"] = _percent_rank(pd.Series(atr_frac, index=d.index), window=252)

    # MACD sign stability (persistence of histogram sign)
    d["feat_macd_sign_stab_20"] = _sign_stability(macd_hist, window=20)

    # Realized vol features
    d["feat_realized_vol_10"] = std10  # reuse CSV std_10
    d["feat_realized_vol_30"] = ret1.rolling(30, min_periods=10).std().fillna(0.0)
    d["feat_vol_of_vol_30"]   = d["feat_realized_vol_30"].rolling(30, min_periods=10).std().fillna(0.0)

    # Return means (risk-on proxy) + above/below EMA200
    d["feat_ret_mean_10"]  = ret1.rolling(10, min_periods=5).mean().fillna(0.0)
    d["feat_ret_mean_30"]  = ret1.rolling(30, min_periods=10).mean().fillna(0.0)
    d["feat_above_ema200"] = (close > ema200).astype(float)

    # --- Pseudo-labels via GMM clustering on regime descriptors ---
    # 1) Trend vs Chop
    X_tr = np.column_stack([
        d["feat_adx_tier"].values,
        np.abs(d["feat_acf1"].values),
        np.abs(d["feat_acf3"].values),
        d["feat_macd_sign_stab_20"].values,
    ])
    lab_tr, _ = _gmm_binary_labels(X_tr, prefer_higher_idx=0)  # higher ADX tier => trend
    d["label_trend"] = lab_tr.astype(int)

    # 2) High-vol vs Low-vol
    X_vol = np.column_stack([
        d["feat_realized_vol_10"].values,
        d["feat_realized_vol_30"].values,
        d["feat_atr_pct_252"].values,
    ])
    lab_vol, _ = _gmm_binary_labels(X_vol, prefer_higher_idx=2)  # higher ATR pct => high-vol
    d["label_highvol"] = lab_vol.astype(int)

    # 3) Risk-on vs Risk-off
    X_risk = np.column_stack([
        d["feat_ret_mean_10"].values,
        d["feat_ret_mean_30"].values,
        d["feat_above_ema200"].values,
    ])
    lab_risk, _ = _gmm_binary_labels(X_risk, prefer_higher_idx=0)
    d["label_riskon"] = lab_risk.astype(int)

    return d.reset_index(drop=True)

# ======================================================================================
# Training / Inference
# ======================================================================================

def fit(X: np.ndarray, y: Optional[np.ndarray] = None) -> RegimeModel:
    """
    Train three CatBoost binary heads to predict:
      - trend vs chop
      - high-vol vs low-vol
      - risk-on vs risk-off

    Args:
        X: Array with columns in the exact order of FEATURES.
        y: Optional (n,3) array with explicit labels [trend, highvol, riskon].
           If None, unsupervised pseudo-labels are derived from *engineered columns in X*.

    Returns:
        RegimeModel
    """
    n, m = X.shape
    if m != len(FEATURES):
        raise ValueError(f"Expected X to have {len(FEATURES)} columns, got {m}")

    # column indices for engineered subsets within X
    idx = {name: FEATURES.index(name) for name in FEATURES}

    # descriptor subsets used for pseudo-labels (these cols are engineered in compute_labels)
    X_tr = np.column_stack([
        X[:, idx["feat_adx_tier"]],
        np.abs(X[:, idx["feat_acf1"]]),
        np.abs(X[:, idx["feat_acf3"]]),
        X[:, idx["feat_macd_sign_stab_20"]],
    ])
    X_vol = np.column_stack([
        X[:, idx["feat_realized_vol_10"]],
        X[:, idx["feat_realized_vol_30"]],
        X[:, idx["feat_atr_pct_252"]],
    ])
    X_risk = np.column_stack([
        X[:, idx["feat_ret_mean_10"]],
        X[:, idx["feat_ret_mean_30"]],
        X[:, idx["feat_above_ema200"]],
    ])

    if y is None:
        y_tr, _   = _gmm_binary_labels(X_tr,  prefer_higher_idx=0)
        y_vol, _  = _gmm_binary_labels(X_vol, prefer_higher_idx=2)
        y_risk, _ = _gmm_binary_labels(X_risk, prefer_higher_idx=0)
    else:
        if y.ndim != 2 or y.shape[1] != 3:
            raise ValueError("y must be of shape (n_samples, 3): [trend, highvol, riskon]")
        y_tr, y_vol, y_risk = y[:, 0].astype(int), y[:, 1].astype(int), y[:, 2].astype(int)

    def _cb():
        return CatBoostClassifier(
            depth=4,
            learning_rate=0.05,
            iterations=400,
            l2_leaf_reg=3.0,
            random_seed=42,
            loss_function="Logloss",
            eval_metric="Logloss",
            verbose=False,
        )

    tr_idx, va_idx = _ts_last_split(n, valid_frac=0.2)

    trend_head = _cb(); trend_head.fit(X[tr_idx], y_tr[tr_idx], eval_set=(X[va_idx], y_tr[va_idx]))
    vol_head   = _cb(); vol_head.fit(X[tr_idx], y_vol[tr_idx], eval_set=(X[va_idx], y_vol[va_idx]))
    risk_head  = _cb(); risk_head.fit(X[tr_idx], y_risk[tr_idx], eval_set=(X[va_idx], y_risk[va_idx]))

    return RegimeModel(trend_head=trend_head, vol_head=vol_head, risk_head=risk_head, output_columns=OUTPUT_COLUMNS)

def predict(model: RegimeModel, X: np.ndarray) -> np.ndarray:
    """
    Returns an (N, 3) array of probabilities aligned to OUTPUT_COLUMNS:
      [:,0] = p_trend, [:,1] = p_high_vol, [:,2] = p_risk_on
    """
    p_trend   = model.trend_head.predict_proba(X)[:, 1]
    p_highvol = model.vol_head.predict_proba(X)[:, 1]
    p_riskon  = model.risk_head.predict_proba(X)[:, 1]
    print("Running PREDICT on regime_gate.py")
    return np.column_stack([p_trend, p_highvol, p_riskon]).astype(float)
