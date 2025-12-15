# ts_trend_short.py
# ---------------------------------------------------------------------
# TS Trend — Short (5–20 bars) using Elastic Net (regression)
# Predicts next-bar (or next h bars) return in bps, plus signal, uncertainty, confidence.
# ---------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

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
def fit_ts_trend_short_state(
    df_train: pd.DataFrame,
    target_bps: pd.Series,
    *,
    features: List[str],
    alpha: float = 0.0005,
    l1_ratio: float = 0.1,
    fit_intercept: bool = True,
    max_iter: int = 2000,
    tol: float = 1e-4,
    random_state: int = 42,
    clip_coef: Optional[float] = None,
    residual_ewm_alpha: float = 0.1,
    model_name: str = "ts_trend_short_enet",
) -> Dict[str, Any]:
    """
    Fit Elastic Net on leakage-safe training data to predict forward returns (bps).
    Returns a serializable `state` dict with the scaler, model, feature order, and uncertainty config.

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

    # Elastic Net fit
    enet = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
    enet.fit(Xz.values, y.values)

    # Optional coefficient clipping (stability on thin data)
    if clip_coef is not None:
        coef = np.clip(enet.coef_, -clip_coef, clip_coef)
        enet.coef_ = coef

    # Build state (serializable)
    state = {
        "scaler_mean_": scaler.mean_.tolist(),
        "scaler_scale_": scaler.scale_.tolist(),
        "scaler_var_": scaler.var_.tolist(),
        "scaler_n_samples_seen_": int(scaler.n_samples_seen_),
        "feature_order": list(features),
        "coef_": enet.coef_.tolist(),
        "intercept_": float(enet.intercept_),
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


def _restore_enet(state: Dict[str, Any]) -> ElasticNet:
    en = ElasticNet()
    en.coef_ = np.array(state["coef_"], dtype=float)
    en.intercept_ = float(state["intercept_"])
    # scikit stores classes/feature_names only in some estimators; for linear model we only need coef_/intercept_
    return en


def _tanh_signal(y_pred_bps: pd.Series, k_bps: float) -> pd.Series:
    return np.tanh(y_pred_bps / float(k_bps))


def _confidence_from_std(pred_std: pd.Series) -> pd.Series:
    # Robust z-score on pred_std → map to [0,1] with logistic
    if pred_std.isna().all():
        return pd.Series(index=pred_std.index, dtype=float)
    m = pred_std.median()
    mad = (pred_std - m).abs().median()
    z = (pred_std - m) / (mad + 1e-9)
    conf = 1.0 / (1.0 + z.clip(lower=0))  # higher std → lower confidence
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
    TS Trend — Short (Elastic Net, regression)
    Inference-only: requires a pre-fitted `state` (from fit_ts_trend_short_state).

    Parameters
    ----------
    df : DataFrame
        Feature-engineered, leakage-safe. Must contain `features`.
    horizon : int
        Bars ahead to predict (e.g., 1). Used for naming; model is trained for a given horizon.
    features : list[str]
        Exact feature columns this model expects (order will be enforced to match training).
    as_of : Timestamp | None
        If provided, slice df to rows <= as_of (live prediction cutoff).
    state : dict | None
        Trained state produced by `fit_ts_trend_short_state`. If None → raises.
    params : dict | None
        Runtime knobs:
          - 'clip_k_bps' (float, default 50.0): signal scaling for tanh
          - 'rv20_max' (float | None): if set and 'rv_20' present, trade_mask = rv_20 <= rv20_max
          - 'min_abs_edge_bps' (float | None): if set, mask out tiny edges
    cost_model : callable | None
        Optional cost estimator: signals -> per-bar costs in bps.

    Returns
    -------
    SubmodelResult
    """
    if state is None:
        raise ValueError(
            "TS Trend — Short requires a fitted `state`. "
            "Fit once with fit_ts_trend_short_state(...) and pass it here."
        )

    params = params or {}
    clip_k_bps = float(params.get("clip_k_bps", 50.0))
    rv20_max = params.get("rv20_max", None)
    min_abs_edge_bps = params.get("min_abs_edge_bps", None)

    # Time slice
    Xfull = df if as_of is None else df.loc[:as_of]
    missing = [f for f in features if f not in Xfull.columns]
    if missing:
        raise KeyError(f"Missing required features for TS Trend — Short: {missing}")

    # Enforce training feature order
    trained_order = state.get("feature_order", features)
    if set(trained_order) != set(features):
        raise ValueError(
            "Feature mismatch between training state and runtime features.\n"
            f"Trained: {trained_order}\nRuntime: {features}"
        )
    X = Xfull[trained_order].copy()

    # Restore scaler and model
    scaler = _restore_scaler(state)
    enet = _restore_enet(state)

    # Standardize & predict
    Xz = pd.DataFrame(scaler.transform(X.values), index=X.index, columns=trained_order)
    # ElasticNet in sklearn doesn't implement predict without fit metadata in object, but we restored coef/intercept_
    # So we compute manually: y = Xz @ coef + intercept
    y_pred_vals = Xz.values @ enet.coef_ + enet.intercept_
    y_pred = pd.Series(y_pred_vals, index=X.index, name=f"ts_trend_short_h{horizon}_bps")

    # Signal mapping
    signal = _tanh_signal(y_pred, clip_k_bps)

    # Uncertainty proxy: EWMA of |y_pred - y_pred.ewm(...)| or use configured residual EWMA (requires residuals in state)
    # Since we cannot read realized returns here, we fall back to EWMA of deviations
    alpha = float(state.get("residual_ewm_alpha", 0.1))
    pred_std = (y_pred - y_pred.ewm(alpha=alpha, adjust=False).mean()).abs().ewm(alpha=alpha, adjust=False).mean()
    pred_std.name = "ts_trend_short_pred_std"

    # Confidence mapping
    confidence = _confidence_from_std(pred_std)

    # Trade mask
    trade_mask = pd.Series(True, index=X.index, name="ts_trend_short_trade_ok")
    if rv20_max is not None and "rv_20" in X.columns:
        trade_mask &= (X["rv_20"] <= float(rv20_max))
    if min_abs_edge_bps is not None:
        trade_mask &= (y_pred.abs() >= float(min_abs_edge_bps))

    # Costs (optional)
    costs = None
    if callable(cost_model):
        try:
            costs = cost_model(signal)
            if isinstance(costs, (pd.Series, pd.DataFrame)) and hasattr(costs, "reindex"):
                costs = costs.reindex(X.index)
        except Exception:
            # Keep costs None if user cost_model fails; avoid breaking inference
            costs = None

    # Feature importance: absolute coefficients (normalized by sum for readability)
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
        trade_mask=trade_mask,
        live_metrics=None,  # fill in caller using realized returns if desired
        feature_importance=fi,
        used_features=trained_order,
        warmup_bars=0,
        model_name=str(state.get("model_name", "ts_trend_short_enet")),
        params={
            "horizon": horizon,
            "clip_k_bps": clip_k_bps,
            "rv20_max": rv20_max,
            "min_abs_edge_bps": min_abs_edge_bps,
        },
        state=state,  # unchanged during pure inference
    )
    return result
