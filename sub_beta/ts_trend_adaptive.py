# ts_trend_adaptive.py
# ---------------------------------------------------------------------
# TS Trend — Adaptive (state-space)
# - Prediction: Time-Varying Regression (RLS/Kalman-like) on minimal features
# - Trade gate: 2-state Gaussian HMM (trend vs. chop) on simple observables
# - Output: y_pred (bps), signal [-1,1], pred_std (state variance proxy),
#           confidence, trade_mask, updated state.
# ---------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable, Tuple
import numpy as np
import pandas as pd

# Optional HMM (graceful fallback if not installed)
try:
    from hmmlearn.hmm import GaussianHMM
    _HAS_HMM = True
except Exception:
    _HAS_HMM = False


# ------------------------
# Shared result container
# ------------------------
@dataclass
class SubmodelResult:
    # Core
    y_pred: pd.Series                   # expected return over `horizon` (bps)
    signal: pd.Series                   # normalized signal in [-1, 1]

    # Optional
    proba_up: Optional[pd.Series] = None
    pred_std: Optional[pd.Series] = None
    confidence: Optional[pd.Series] = None
    costs: Optional[pd.Series] = None
    trade_mask: Optional[pd.Series] = None

    # Live diagnostics
    live_metrics: Optional[Dict[str, float]] = None

    # Explainability / bookkeeping
    feature_importance: Optional[pd.Series] = None
    used_features: Optional[List[str]] = None
    warmup_bars: int = 0
    model_name: str = ""
    params: Optional[Dict[str, Any]] = None
    state: Optional[Any] = None


# ===========================================================
# Training helper: builds initial `state` for the TV regression
# and (optionally) fits a 2-state HMM for gating.
# ===========================================================
def fit_ts_trend_adaptive_state(
    df_train: pd.DataFrame,
    *,
    features: List[str],  # minimal: ['returns_1','returns_3','rv_20','ema_slope_21','ema200_dist']
    hmm_features: Optional[List[str]] = None,  # e.g., ['returns_1','rv_20','rsi','adx']
    scaler_mode: str = "standard",   # or 'robust'
    rls_lambda: float = 0.98,        # forgetting factor (close to 1)
    rls_delta: float = 1e4,          # init covariance scale (large => weak prior)
    bps_scale: float = 10_000.0,     # convert decimal return to bps
    model_name: str = "ts_trend_adaptive_tvr_hmm",
) -> Dict[str, Any]:
    """
    Prepare a serializable `state` for the adaptive trend model.
    - Builds a scaler on `features`
    - Initializes RLS parameters (beta, P, sigma2)
    - Optionally fits a 2-state Gaussian HMM on `hmm_features` for trade gating.
    """
    X = df_train[features].dropna().copy()
    if X.empty:
        raise ValueError("No training rows after dropping NA in features.")

    # --- build scaler (global stats) ---
    if scaler_mode == "standard":
        mu = X.mean().to_dict()
        sd = X.std(ddof=0).replace(0, 1.0).to_dict()
    elif scaler_mode == "robust":
        mu = X.median().to_dict()
        sd = (X.quantile(0.75) - X.quantile(0.25)).replace(0, 1.0).to_dict()
    else:
        raise ValueError("scaler_mode must be 'standard' or 'robust'")

    # --- init RLS (time-varying regression) ---
    d = len(features) + 1  # +1 for intercept
    beta0 = np.zeros(d)    # [intercept, phi1, phi2, ...]
    P0 = np.eye(d) * rls_delta
    sigma2_ewm = 1e-6      # small starting noise estimate

    # --- optional HMM for gating ---
    hmm_state = None
    if hmm_features:
        H = df_train[hmm_features].dropna().copy()
        if not H.empty:
            if _HAS_HMM:
                # Pre-transform: discretize rsi/adx if present
                Ht = H.copy()
                if "rsi" in Ht.columns:
                    Ht["rsi_bin"] = pd.cut(Ht["rsi"], [-np.inf, 30, 70, np.inf], labels=[0, 1, 2]).astype(int)
                if "adx" in Ht.columns:
                    # stronger trend bins: <15, 15-25, 25-40, >40
                    Ht["adx_bin"] = pd.cut(Ht["adx"], [-np.inf, 15, 25, 40, np.inf], labels=[0, 1, 2, 3]).astype(int)

                # Build observation matrix as numeric
                obs_cols = []
                for c in ["returns_1", "rv_20", "rsi_bin", "adx_bin"]:
                    if c in Ht.columns:
                        obs_cols.append(c)
                if not obs_cols:
                    obs_cols = [col for col in Ht.columns if np.issubdtype(Ht[col].dtype, np.number)]
                Obs = Ht[obs_cols].astype(float).values

                # Fit 2-state Gaussian HMM
                hmm = GaussianHMM(n_components=2, covariance_type="full", n_iter=200, random_state=42)
                hmm.fit(Obs)

                # Serialize HMM params
                hmm_state = {
                    "obs_cols": obs_cols,
                    "startprob_": hmm.startprob_.tolist(),
                    "transmat_": hmm.transmat_.tolist(),
                    "means_": hmm.means_.tolist(),
                    "covars_": [c.tolist() for c in hmm.covars_],
                }
            else:
                # Store heuristic gates config (for fallback)
                hmm_state = {
                    "obs_cols": hmm_features,
                    "heuristic": True,
                    # thresholds tuned later in predict_submodel
                }

    state = {
        "model_name": model_name,
        "feature_order": list(features),
        "scaler_mode": scaler_mode,
        "mu": mu,
        "sd": sd,
        "rls_lambda": float(rls_lambda),
        "bps_scale": float(bps_scale),
        # RLS dynamic state
        "beta": beta0.tolist(),
        "P": P0.tolist(),
        "sigma2_ewm": float(sigma2_ewm),
        # HMM
        "hmm_state": hmm_state,
    }
    return state


# ---------------------------------------------
# Internal utilities
# ---------------------------------------------
def _standardize(X: pd.DataFrame, mu: Dict[str, float], sd: Dict[str, float]) -> pd.DataFrame:
    Z = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
    for c in X.columns:
        s = sd.get(c, 1.0)
        if s == 0:
            s = 1.0
        Z[c] = (X[c] - mu.get(c, 0.0)) / s
    return Z


def _tanh_signal(y_pred_bps: pd.Series, k_bps: float) -> pd.Series:
    return np.tanh(y_pred_bps / float(k_bps))


def _confidence_from_std(pred_std: pd.Series) -> pd.Series:
    if pred_std.isna().all():
        return pd.Series(index=pred_std.index, dtype=float)
    m = pred_std.median()
    mad = (pred_std - m).abs().median()
    z = (pred_std - m) / (mad + 1e-9)
    conf = 1.0 / (1.0 + z.clip(lower=0))
    return conf.clip(0, 1)


def _restore_hmm(hmm_state: Dict[str, Any]) -> Optional[GaussianHMM]:
    if not (_HAS_HMM and hmm_state and not hmm_state.get("heuristic", False)):
        return None
    hmm = GaussianHMM(n_components=2, covariance_type="full", n_iter=1, random_state=42)
    hmm.startprob_ = np.array(hmm_state["startprob_"], dtype=float)
    hmm.transmat_ = np.array(hmm_state["transmat_"], dtype=float)
    hmm.means_ = np.array(hmm_state["means_"], dtype=float)
    hmm.covars_ = np.array(hmm_state["covars_"], dtype=float)
    return hmm


def _predict_hmm_trend_prob(Xhmm: pd.DataFrame, hmm_state: Optional[Dict[str, Any]]) -> pd.Series:
    idx = Xhmm.index
    if hmm_state is None:
        return pd.Series(1.0, index=idx)

    obs_cols = hmm_state.get("obs_cols", [])

    # Start from ALL available cols, then create bins
    ObsDF = Xhmm.copy()

    # Create bins if raw features present (and not already binned)
    if "rsi" in ObsDF.columns and "rsi_bin" not in ObsDF.columns:
        ObsDF["rsi_bin"] = pd.cut(ObsDF["rsi"], [-np.inf, 30, 70, np.inf], labels=[0, 1, 2]).astype(float)
    if "adx" in ObsDF.columns and "adx_bin" not in ObsDF.columns:
        ObsDF["adx_bin"] = pd.cut(ObsDF["adx"], [-np.inf, 15, 25, 40, np.inf], labels=[0, 1, 2, 3]).astype(float)

    # Prefer hmm_state's columns, but only those that actually exist now
    use_cols = [c for c in obs_cols if c in ObsDF.columns]
    if not use_cols:
        use_cols = [c for c in ["returns_1", "rv_20", "rsi_bin", "adx_bin"] if c in ObsDF.columns]
    if not use_cols:
        use_cols = [c for c in ObsDF.columns if np.issubdtype(ObsDF[c].dtype, np.number)]

    Obs = ObsDF[use_cols].astype(float).fillna(method="ffill").fillna(0.0).values

    # If we have a real HMM, compute posterior gamma; else heuristic
    hmm = _restore_hmm(hmm_state)
    if hmm is not None:
        try:
            # score_samples returns (logprob, posteriors); posteriors shape: [n_samples, n_states]
            _, post = hmm.score_samples(Obs)
            # Pick which component is "trend": the one with lower RV and higher |mean return|
            means = hmm.means_.reshape(-1, hmm.means_.shape[-1])
            # heuristic: dimension 0 ~ returns, dimension 1 ~ rv_20 if present
            # choose state with lower mean rv and higher |mean return|
            if len(use_cols) >= 2 and "rv_20" in use_cols:
                rv_idx = use_cols.index("rv_20")
            else:
                rv_idx = None
            ret_idx = 0  # assume returns_1 first if present
            if "returns_1" in use_cols:
                ret_idx = use_cols.index("returns_1")

            if rv_idx is not None:
                trend_state = int(np.argmin(means[:, rv_idx]))
            else:
                trend_state = int(np.argmax(np.abs(means[:, ret_idx])))
            p_trend = post[:, trend_state]
            return pd.Series(p_trend, index=idx, name="p_trend")
        except Exception:
            pass  # fall back to heuristic

    # Heuristic p_trend: higher when |ema_slope| large, ADX high, RV moderate
    # We attempt to use the columns available
    s = pd.Series(0.5, index=idx, dtype=float)
    if "rv_20" in Xhmm.columns:
        rv = Xhmm["rv_20"].astype(float)
        rv_z = (rv - rv.rolling(200, min_periods=20).median()) / (rv.rolling(200, min_periods=20).mad() + 1e-9)
        s *= (1.0 / (1.0 + np.exp(rv_z)))  # lower rv -> higher prob
    if "adx" in Xhmm.columns:
        adx = Xhmm["adx"].astype(float).fillna(0.0)
        s *= (1.0 / (1.0 + np.exp(-(adx - 20) / 5.0)))  # ADX>20 → higher prob
    if "ema_slope_21" in Xhmm.columns:
        slope = Xhmm["ema_slope_21"].astype(float).abs()
        slope_z = (slope - slope.rolling(200, min_periods=20).median()) / (slope.rolling(200, min_periods=20).mad() + 1e-9)
        s *= (1.0 / (1.0 + np.exp(-slope_z)))
    return s.clip(0.0, 1.0).rename("p_trend")


# ===========================================================
# Required submodel API (inference; no future targets read)
# ===========================================================
def predict_submodel(
    df: pd.DataFrame,
    *,
    horizon: int,
    features: List[str],
    as_of: Optional[pd.Timestamp] = None,
    state: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    cost_model: Optional[Callable[[pd.Series], pd.Series]] = None,
) -> SubmodelResult:
    """
    TS Trend — Adaptive:
      - Time-varying regression (RLS/Kalman-like) for edge estimate (y_pred in bps)
      - 2-state HMM (or heuristic) for trade gating (trend vs. chop probability)
    Notes:
      * Uses only present/past features up to `as_of`.
      * Does NOT read future returns / targets.
    """
    if state is None:
        raise ValueError(
            "TS Trend — Adaptive requires a fitted `state`. "
            "Fit once with fit_ts_trend_adaptive_state(...) and pass it here."
        )

    params = params or {}
    clip_k_bps = float(params.get("clip_k_bps", 50.0))  # signal scaler
    p_trend_thresh = float(params.get("p_trend_thresh", 0.6))
    min_abs_edge_bps = params.get("min_abs_edge_bps", None)

    # Slice to as_of
    Xfull = df if as_of is None else df.loc[:as_of]
    # Enforce feature order
    trained_order = state.get("feature_order", features)
    missing = [f for f in trained_order if f not in Xfull.columns]
    if missing:
        raise KeyError(f"Missing required features for TS Trend — Adaptive: {missing}")
    X = Xfull[trained_order].copy()

    # Standardize features
    Z = _standardize(X, state["mu"], state["sd"])
    # Add intercept
    Z["_intercept_"] = 1.0
    cols = ["_intercept_"] + trained_order
    Z = Z[cols]

    # Retrieve dynamic RLS state
    lam = float(state["rls_lambda"])
    bps_scale = float(state.get("bps_scale", 10_000.0))
    beta = np.array(state["beta"], dtype=float)  # shape (d,)
    P = np.array(state["P"], dtype=float)        # shape (d,d)
    sigma2_ewm = float(state.get("sigma2_ewm", 1e-6))

    # Online filter pass (no future leakage): for each t, form one-step-ahead prediction y_pred_t = x_t' beta_{t-1}
    # Observation used for filter update: returns_1 at t (present/past), *not* future returns.
    if "returns_1" not in X.columns:
        raise KeyError("TS Trend — Adaptive needs 'returns_1' in features (decimal returns).")

    y_obs = X["returns_1"].astype(float).fillna(0.0)  # observed current/past returns (decimal)

    y_pred_list = []
    pred_var_list = []

    for t in Z.index:
        x = Z.loc[t, cols].values.astype(float)  # (d,)

        # One-step-ahead prediction (decimal), then convert to bps
        yhat_dec = float(np.dot(x, beta))
        yhat_bps = yhat_dec * bps_scale

        # Predictive variance proxy: v = x' P x
        v = float(np.dot(x, P @ x)) + 1e-9

        y_pred_list.append(yhat_bps)
        pred_var_list.append(v * (sigma2_ewm + 1e-8))  # scale by noise estimate

        # RLS/Kalman-like update using observed current return (decimal)
        y = float(y_obs.loc[t])
        # Innovation
        innov = y - yhat_dec
        # Gain
        K = (P @ x) / (lam + v)
        # Update beta, covariance
        beta = beta + K * innov
        P = (P - np.outer(K, x) @ P) / lam
        # Update noise estimate (EWMA)
        sigma2_ewm = (1 - 0.05) * sigma2_ewm + 0.05 * (innov ** 2)

    y_pred = pd.Series(y_pred_list, index=Z.index, name=f"ts_trend_adaptive_h{horizon}_bps")
    pred_std = pd.Series(np.sqrt(np.maximum(pred_var_list, 0.0)), index=Z.index, name="ts_trend_adaptive_pred_std")

    # Map to signal
    signal = _tanh_signal(y_pred, clip_k_bps)

    # Confidence from pred_std
    confidence = _confidence_from_std(pred_std)

    # --- Trade mask via HMM / heuristic ---
    hmm_state = state.get("hmm_state", None)
    if hmm_state and "obs_cols" in hmm_state:
        need = []
        for c in hmm_state["obs_cols"]:
            if c == "rsi_bin":
                need.append("rsi")      # need raw rsi to build the bin
            elif c == "adx_bin":
                need.append("adx")      # need raw adx to build the bin
            else:
                need.append(c)
        hmm_cols = [c for c in need if c in Xfull.columns]
    else:
        hmm_cols = [c for c in ["returns_1", "rv_20", "rsi", "adx", "ema_slope_21"] if c in Xfull.columns]

    Xhmm = Xfull[hmm_cols].loc[Z.index] if hmm_cols else pd.DataFrame(index=Z.index)
    p_trend = _predict_hmm_trend_prob(Xhmm, hmm_state)
    trade_mask = (p_trend >= p_trend_thresh)
    trade_mask.name = "ts_trend_adaptive_trade_ok"

    # Optional edge gating
    if min_abs_edge_bps is not None:
        trade_mask &= (y_pred.abs() >= float(min_abs_edge_bps))

    # Optional costs pass-through (not used here)
    costs = None
    if callable(cost_model):
        try:
            costs = cost_model(signal)
            if isinstance(costs, (pd.Series, pd.DataFrame)) and hasattr(costs, "reindex"):
                costs = costs.reindex(Z.index)
        except Exception:
            costs = None

    # Update and return dynamic state
    state_out = dict(state)
    state_out["beta"] = beta.tolist()
    state_out["P"] = P.tolist()
    state_out["sigma2_ewm"] = float(sigma2_ewm)

    # Feature "importance": absolute |beta| on standardized scale (ignore intercept), normalized
    beta_series = pd.Series(beta[1:], index=trained_order, name="beta_std")
    if beta_series.abs().sum() > 0:
        beta_imp = (beta_series.abs() / beta_series.abs().sum())
    else:
        beta_imp = beta_series.abs()

    result = SubmodelResult(
        y_pred=y_pred,
        signal=signal,
        proba_up=None,
        pred_std=pred_std,
        confidence=confidence,
        costs=costs,
        trade_mask=trade_mask,
        live_metrics=None,
        feature_importance=beta_imp,
        used_features=trained_order,
        warmup_bars=0,
        model_name=str(state.get("model_name", "ts_trend_adaptive_tvr_hmm")),
        params={
            "horizon": horizon,
            "clip_k_bps": clip_k_bps,
            "p_trend_thresh": p_trend_thresh,
            "min_abs_edge_bps": min_abs_edge_bps,
        },
        state=state_out,
    )
    return result
