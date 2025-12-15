# ts_trend_medium.py
# ---------------------------------------------------------------------
# TS Trend — Medium (1–3 months equiv.) using Ridge (regression)
# Predicts next h-bar return (bps). Outputs y_pred, signal, pred_std, confidence, feature_importance.
# ---------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


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
# Training helper (builds `state` used by predict_submodel)
# ===========================================================
def fit_ts_trend_medium_state(
    df_train: pd.DataFrame,
    target_bps: pd.Series,
    *,
    features: List[str],
    alpha: float = 0.5,                 # Ridge strength (tune via CV)
    fit_intercept: bool = True,
    max_iter: int = 2000,
    tol: float = 1e-4,
    random_state: int = 42,
    clip_coef: Optional[float] = None,  # optional safety clip for stability
    residual_ewm_alpha: float = 0.08,   # uncertainty EWMA smoothing
    model_name: str = "ts_trend_medium_ridge",
) -> Dict[str, Any]:
    """
    Fit Ridge on leakage-safe training data to predict forward returns (bps).
    Returns a serializable `state` dict (scaler, coefficients, feature order, uncertainty config).

    Parameters
    ----------
    df_train : DataFrame
        Must contain `features`; index aligned with target_bps.
    target_bps : Series
        Forward-looking realized returns in bps (aligned to df_train index).
        Construct outside with proper shift/embargo to avoid leakage.
    """
    # Align and drop NA
    X = df_train[features].copy()
    y = target_bps.reindex(X.index)
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X.loc[mask], y.loc[mask]

    # Standardize features
    scaler = StandardScaler()
    Xz = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

    # Ridge fit
    ridge = Ridge(
        alpha=alpha,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
    ridge.fit(Xz.values, y.values)

    # Optional coefficient clipping
    if clip_coef is not None:
        coef = np.clip(ridge.coef_, -clip_coef, clip_coef)
        ridge.coef_ = coef

    # Build serializable state
    state = {
        "scaler_mean_": scaler.mean_.tolist(),
        "scaler_scale_": scaler.scale_.tolist(),
        "scaler_var_": scaler.var_.tolist(),
        "scaler_n_samples_seen_": int(scaler.n_samples_seen_),
        "feature_order": list(features),
        "coef_": ridge.coef_.tolist(),
        "intercept_": float(ridge.intercept_),
        "residual_ewm_alpha": float(residual_ewm_alpha),
        "model_name": model_name,
    }
    return state


# ---------------------------------------------
# Internal utilities (pure, side-effect free)
# ---------------------------------------------
def _restore_scaler(state: Dict[str, Any]) -> StandardScaler:
    sc = StandardScaler()
    sc.mean_ = np.array(state["scaler_mean_"], dtype=float)
    sc.scale_ = np.array(state["scaler_scale_"], dtype=float)
    sc.var_ = np.array(state["scaler_var_"], dtype=float)
    sc.n_samples_seen_ = np.array(state["scaler_n_samples_seen_"], dtype=int)
    return sc


def _restore_ridge(state: Dict[str, Any]) -> Ridge:
    rg = Ridge()
    rg.coef_ = np.array(state["coef_"], dtype=float)
    rg.intercept_ = float(state["intercept_"])
    return rg


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


# ===========================================================
# Required submodel API (inference only; no targets read)
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
    TS Trend — Medium (Ridge, regression).
    Inference-only: requires a pre-fitted `state` (from fit_ts_trend_medium_state).

    Parameters
    ----------
    df : DataFrame
        Feature-engineered, leakage-safe. Must contain `features`.
    horizon : int
        Bars ahead to predict (e.g., 6). Used for naming; model should be trained for that horizon.
    features : list[str]
        Exact feature columns this model expects (order enforced to match training).
    as_of : Timestamp | None
        If provided, slice df to rows <= as_of (live cutoff).
    state : dict | None
        Trained state produced by `fit_ts_trend_medium_state`. If None → raises.
    params : dict | None
        Runtime knobs:
          - 'clip_k_bps' (float, default 60.0): signal scaling for tanh
          - 'min_abs_edge_bps' (float | None): if set, can be used upstream for gating
    cost_model : callable | None
        Optional cost estimator: signals -> per-bar costs in bps (not used here by default).

    Returns
    -------
    SubmodelResult
    """
    if state is None:
        raise ValueError(
            "TS Trend — Medium requires a fitted `state`. "
            "Fit with fit_ts_trend_medium_state(...) and pass it here."
        )

    params = params or {}
    clip_k_bps = float(params.get("clip_k_bps", 60.0))
    min_abs_edge_bps = params.get("min_abs_edge_bps", None)

    # Time slice
    Xfull = df if as_of is None else df.loc[:as_of]
    missing = [f for f in features if f not in Xfull.columns]
    if missing:
        raise KeyError(f"Missing required features for TS Trend — Medium: {missing}")

    # Enforce feature order from training
    trained_order = state.get("feature_order", features)
    if set(trained_order) != set(features):
        raise ValueError(
            "Feature mismatch between training state and runtime features.\n"
            f"Trained: {trained_order}\nRuntime: {features}"
        )
    X = Xfull[trained_order].copy()

    # Restore scaler & model
    scaler = _restore_scaler(state)
    ridge = _restore_ridge(state)

    # Standardize & predict
    Xz = pd.DataFrame(scaler.transform(X.values), index=X.index, columns=trained_order)
    y_pred_vals = Xz.values @ ridge.coef_ + ridge.intercept_
    y_pred = pd.Series(y_pred_vals, index=X.index, name=f"ts_trend_medium_h{horizon}_bps")

    # Map to signal
    signal = _tanh_signal(y_pred, clip_k_bps)

    # Uncertainty proxy: EWMA of deviations (target-free)
    alpha = float(state.get("residual_ewm_alpha", 0.08))
    pred_std = (y_pred - y_pred.ewm(alpha=alpha, adjust=False).mean()).abs().ewm(alpha=alpha, adjust=False).mean()
    pred_std.name = "ts_trend_medium_pred_std"

    # Confidence from pred_std
    confidence = _confidence_from_std(pred_std)

    # Optional costs (not used by this model; provided for symmetry)
    costs = None
    if callable(cost_model):
        try:
            costs = cost_model(signal)
            if isinstance(costs, (pd.Series, pd.DataFrame)) and hasattr(costs, "reindex"):
                costs = costs.reindex(X.index)
        except Exception:
            costs = None

    # Feature importance: absolute standardized coefficients, normalized to sum=1
    coef = np.array(state["coef_"], dtype=float)
    fi = pd.Series(coef, index=trained_order, name="coef")
    if fi.abs().sum() > 0:
        fi = fi / fi.abs().sum()

    result = SubmodelResult(
        y_pred=y_pred,
        signal=signal,
        proba_up=None,
        pred_std=pred_std,
        confidence=confidence,
        costs=costs,
        trade_mask=None,  # Not used for this model
        live_metrics=None,
        feature_importance=fi,
        used_features=trained_order,
        warmup_bars=0,
        model_name=str(state.get("model_name", "ts_trend_medium_ridge")),
        params={
            "horizon": horizon,
            "clip_k_bps": clip_k_bps,
            "min_abs_edge_bps": min_abs_edge_bps,
        },
        state=state,
    )
    return result
