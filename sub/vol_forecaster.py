# vol_forecaster.py
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from catboost import CatBoostRegressor

# ======================================================================================
# Public API
# ======================================================================================

# We ONLY reference CSV columns you already have, and derive the rest here.
# CSV columns used: 
#   open, high, low, close, returns_1, std_5, std_10, log_volume, volume_zscore, gap_vs_prev,
#   transactions, vwap, adx, rsi_zscore, atr_zscore, days_since_high, days_since_low, bollinger_percB
#
# Engineered here:
#   feat_rv1               ~ realized vol proxy at t (sqrt(pi/2)*|r_t|)
#   feat_rv1_lag{1,5,10}   ~ HAR-RV style lags
#   feat_har_w             ~ "weekly" mean of rv1 (≈30 bars for 4H data)
#   feat_har_m             ~ "monthly" mean of rv1 (≈126 bars for 4H data)
#   feat_parkinson         ~ range-based vol proxy
#   feat_gk                ~ Garman–Klass vol proxy
#   feat_rv1_vol           ~ vol-of-vol (rolling std of rv1)
#
# Target:
#   target_log_sigma_t1 = log( sigma_{t+1} ), where sigma_{t+1} ≈ sqrt(pi/2) * |returns_1 at t+1|
#
# Model outputs (probabilities are not relevant here; we output sigma quantiles):
#   sigma_q10, sigma_q50, sigma_q90

FEATURES: List[str] = [
    # --- CSV cols used directly ---
    "returns_1", "std_5", "std_10", "log_volume", "volume_zscore", "gap_vs_prev",
    "transactions", "vwap", "adx", "rsi_zscore", "atr_zscore",
    "days_since_high", "days_since_low", "bollinger_percB",
    "open", "high", "low", "close",

    # --- Engineered (computed here) ---
    "feat_rv1",
    "feat_rv1_lag1",
    "feat_rv1_lag5",
    "feat_rv1_lag10",
    "feat_har_w",
    "feat_har_m",
    "feat_parkinson",
    "feat_gk",
    "feat_rv1_vol",
]

OUTPUT_COLUMNS: List[str] = ["sigma_q10", "sigma_q50", "sigma_q90"]

# Windows (tuned for 4H bars; adjust if your bar size differs)
HAR_W = 30   # ~ 1 trading week (≈ 5 days * 6 bars)
HAR_M = 126  # ~ 1 trading month (≈ 21 days * 6 bars)

EPS = 1e-12

# ======================================================================================
# Helpers
# ======================================================================================

def _safe_log(x):
    x = np.asarray(x, dtype=float)
    return np.log(np.maximum(x, EPS))

def _sqrt_pos(x):
    return np.sqrt(np.maximum(x, 0.0))

def _rv_from_ret(ret):
    """
    Realized volatility proxy per bar: sigma_t ≈ sqrt(pi/2) * |r_t|
    If returns_1 is in percent, sigma is 'percent-vol per bar'; this is fine as long as you're consistent.
    """
    return np.sqrt(np.pi / 2.0) * np.abs(ret)

def _parkinson_sigma(high: pd.Series, low: pd.Series) -> pd.Series:
    """
    Parkinson range-based volatility (per-bar sigma):
    sigma_P = sqrt( (1 / (4 ln 2)) * (ln(H/L))^2 )
    """
    hl = np.asarray(high) / np.maximum(np.asarray(low), EPS)
    v = (np.log(np.maximum(hl, EPS))) ** 2
    sigma2 = v / (4.0 * np.log(2.0))
    return pd.Series(_sqrt_pos(sigma2), index=high.index)

def _garman_klass_sigma(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Garman–Klass variance per bar:
    var_GK = 0.5*(ln(H/L))^2 - (2ln2 - 1)*(ln(C/O))^2
    sigma_GK = sqrt(max(var_GK, 0))
    """
    hl_term = 0.5 * (np.log(np.maximum(np.asarray(high) / np.maximum(np.asarray(low), EPS), EPS))) ** 2
    co_term = (2.0 * np.log(2.0) - 1.0) * (np.log(np.maximum(np.asarray(close) / np.maximum(np.asarray(open_), EPS), EPS))) ** 2
    var_gk = hl_term - co_term
    return pd.Series(_sqrt_pos(var_gk), index=close.index)

def _last_split(n: int, valid_frac: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    split = int(max(1, n * (1 - valid_frac)))
    return np.arange(0, split), np.arange(split, n)

# ======================================================================================
# Feature + target construction (only computes what is NOT already in the CSV)
# ======================================================================================

def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds engineered volatility features and the training target:
      target_log_sigma_t1 = log( sigma_{t+1} )
    Uses only existing CSV columns; does NOT recompute anything you already provide.
    """
    d = df.copy()

    # Pull required CSV columns
    open_ = d["open"].astype(float)
    high  = d["high"].astype(float)
    low   = d["low"].astype(float)
    close = d["close"].astype(float)

    ret1  = d["returns_1"].astype(float)
    std5  = d["std_5"].astype(float)
    std10 = d["std_10"].astype(float)

    log_vol   = d["log_volume"].astype(float)
    vol_z     = d["volume_zscore"].astype(float)
    gap       = d["gap_vs_prev"].astype(float)
    trans     = d.get("transactions", pd.Series(0.0, index=d.index)).astype(float)
    vwap      = d.get("vwap", pd.Series(0.0, index=d.index)).astype(float)

    adx       = d["adx"].astype(float)
    rsi_z     = d["rsi_zscore"].astype(float)
    atr_z     = d["atr_zscore"].astype(float)
    dsh       = d["days_since_high"].astype(float)
    dsl       = d["days_since_low"].astype(float)
    bb_percB  = d["bollinger_percB"].astype(float)

    # --- Engineered features (HAR-RV and range-based proxies) ---
    rv1 = pd.Series(_rv_from_ret(ret1), index=d.index)  # per-bar realized vol proxy

    d["feat_rv1"]        = rv1
    d["feat_rv1_lag1"]   = rv1.shift(1)
    d["feat_rv1_lag5"]   = rv1.shift(5)
    d["feat_rv1_lag10"]  = rv1.shift(10)
    d["feat_har_w"]      = rv1.rolling(HAR_W, min_periods=HAR_W//3).mean()
    d["feat_har_m"]      = rv1.rolling(HAR_M, min_periods=HAR_M//3).mean()

    d["feat_parkinson"]  = _parkinson_sigma(high, low)
    d["feat_gk"]         = _garman_klass_sigma(open_, high, low, close)
    d["feat_rv1_vol"]    = rv1.rolling(20, min_periods=6).std()

    # Target: log sigma_{t+1}
    sigma_t1 = _rv_from_ret(ret1.shift(-1))  # next bar realized vol proxy
    d["target_log_sigma_t1"] = _safe_log(sigma_t1)

    # Clean NaNs from engineered fields sensibly (keep target NaN where it naturally is at the tail)
    eng_cols = [c for c in d.columns if c.startswith("feat_")]
    d[eng_cols] = d[eng_cols].replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill")

    return d.reset_index(drop=True)

# ======================================================================================
# Training / Inference
# ======================================================================================

def fit(X: np.ndarray, y_log_sigma_t1: np.ndarray, valid_frac: float = 0.2):
    """
    Train CatBoost with MultiQuantile loss to predict log(sigma_{t+1}).
    Returns a fitted CatBoostRegressor.
    """
    n, m = X.shape
    if m != len(FEATURES):
        raise ValueError(f"Expected X to have {len(FEATURES)} columns, got {m}")

    # Blocked (last) split; no shuffling.
    tr_idx, va_idx = _last_split(n, valid_frac=valid_frac)

    model = CatBoostRegressor(
        depth=7,
        learning_rate=0.045,
        l2_leaf_reg=6.0,
        iterations=1200,
        loss_function="MultiQuantile:alpha=0.1,0.5,0.9",
        eval_metric="Quantile:alpha=0.5",
        bootstrap_type="Bayesian",
        subsample=0.8,
        random_strength=1.0,
        random_seed=42,
        od_type="Iter",
        od_wait=100,
        verbose=False,
    )

    model.fit(
        X[tr_idx], y_log_sigma_t1[tr_idx],
        eval_set=(X[va_idx], y_log_sigma_t1[va_idx]),
        use_best_model=True,
    )
    return model

def predict(model: CatBoostRegressor, X: np.ndarray) -> np.ndarray:
    """
    Returns an (N, 3) array of sigma quantiles in the ORIGINAL scale (not log):
      [:,0] = sigma_q10
      [:,1] = sigma_q50 (median)
      [:,2] = sigma_q90
    """
    # CatBoost with MultiQuantile returns predictions shaped (N, n_quantiles) in log space (our target).
    pred_log = np.asarray(model.predict(X))
    # Ensure 2D
    if pred_log.ndim == 1:
        pred_log = pred_log.reshape(-1, 1)

    pred_sigma = np.exp(pred_log)  # back to sigma-space
    # If model returned single column (misconfig), expand safely
    if pred_sigma.shape[1] == 1:
        pred_sigma = np.repeat(pred_sigma, 3, axis=1)
    print("Running PREDICT on vol_forecaster.py")
    return pred_sigma