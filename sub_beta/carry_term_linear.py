# carry_term_linear.py
# ---------------------------------------------------------------------
# Carry & Term Structure (futures/FX/perps)
# Model: Linear Regression (Ridge default; OLS optional)
# Label/Horizon: h in {6..12} bars (slow edge) -> target_bps built upstream
# Features (pick per market):
#   - futures/FX: ['carry','curve_slope','adv_ratio','rv_20','bb_bw','dd_63']
#   - crypto perps: ['funding_ann','funding_ma','funding_z','adv_ratio','rv_20','bb_bw','dd_63']
# Outputs: y_pred (bps), signal [-1,1], pred_std (EWMA proxy), confidence, feature_importance
# ---------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression


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
def fit_carry_term_state(
    df_train: pd.DataFrame,
    target_bps: pd.Series,
    *,
    features: List[str],
    model_type: str = "ridge",          # 'ridge' or 'ols'
    alpha: float = 0.25,                # ridge strength (ignored for OLS)
    fit_intercept: bool = True,
    scaler_mode: str = "standard",      # 'standard' | 'robust'
    max_iter: int = 2000,
    tol: float = 1e-4,
    clip_coef: Optional[float] = None,  # optional safety cap on coefficients
    residual_ewm_alpha: float = 0.05,   # slow-moving uncertainty for slow edges
    model_name: str = "carry_term_linear",
) -> Dict[str, Any]:
    """
    Fit a linear model (Ridge/OLS) for carry & term-structure alpha.
    Returns a serializable `state` dict with scaler stats, coefficients, and metadata.
    """
    # Align & basic clean
    X = df_train[features].copy()
    y = target_bps.reindex(X.index)

    # Remove infinities, treat as missing
    X = X.replace([np.inf, -np.inf], np.nan)

    # 1) Drop columns that are completely missing
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)
        features = [f for f in features if f not in all_nan_cols]

    # If nothing left to train on, skip this model
    if X.shape[1] == 0:
        return None

    # 2) Leakage-safe fill (past-only), then robust fallback
    X = X.ffill()
    # Median is computed on the (past-only) training window, so OK for walk-forward
    X = X.fillna(X.median(numeric_only=True))

    # 3) Now align y and drop rows where y is NA
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask].astype(float)

    # Optional: require a minimum sample size for stability
    if len(X) < 25:
        return None


    # Standardize (or robust-standardize) features
    if scaler_mode == "standard":
        scaler = StandardScaler()
        Xz = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
        scaler_state = {
            "scaler_mean_": scaler.mean_.tolist(),
            "scaler_scale_": scaler.scale_.tolist(),
            "scaler_var_": scaler.var_.tolist(),
            "scaler_n_samples_seen_": int(scaler.n_samples_seen_),
        }
    elif scaler_mode == "robust":
        med = X.median()
        iqr = (X.quantile(0.75) - X.quantile(0.25)).replace(0, 1.0)
        Xz = (X - med) / iqr
        scaler_state = {"robust_med_": med.astype(float).to_dict(),
                        "robust_iqr_": iqr.astype(float).to_dict()}
    else:
        raise ValueError("scaler_mode must be 'standard' or 'robust'")

    # Fit linear model
    if model_type.lower() == "ridge":
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, random_state=42)
    elif model_type.lower() == "ols":
        model = LinearRegression(fit_intercept=fit_intercept)
    else:
        raise ValueError("model_type must be 'ridge' or 'ols'")

    model.fit(Xz.values, y.values)

    # Optional coefficient clipping
    coef = model.coef_.astype(float).ravel()
    if clip_coef is not None:
        coef = np.clip(coef, -float(clip_coef), float(clip_coef))
    intercept = float(model.intercept_) if fit_intercept else 0.0

    # Normalized abs-coef feature importance (on standardized space)
    coef_abs = np.abs(coef)
    total = coef_abs.sum() or 1.0
    fi = {f: float(a / total) for f, a in zip(features, coef_abs)}

    # Build serializable state
    state: Dict[str, Any] = {
        "model_name": model_name,
        "feature_order": list(features),
        "model_type": model_type.lower(),
        "alpha": float(alpha),
        "fit_intercept": bool(fit_intercept),
        "coef_": coef.tolist(),
        "intercept_": intercept,
        "scaler_mode": scaler_mode,
        "residual_ewm_alpha": float(residual_ewm_alpha),
        "feature_importance": fi,
        **scaler_state,
    }
    return state


# ---------------------------------------------
# Internal utilities
# ---------------------------------------------
def _standardize(X: pd.DataFrame, state: Dict[str, Any]) -> pd.DataFrame:
    mode = state.get("scaler_mode", "standard")
    if mode == "standard":
        sc = StandardScaler()
        sc.mean_ = np.array(state["scaler_mean_"], dtype=float)
        sc.scale_ = np.array(state["scaler_scale_"], dtype=float)
        sc.var_ = np.array(state["scaler_var_"], dtype=float)
        sc.n_samples_seen_ = np.array(state["scaler_n_samples_seen_"], dtype=int)
        Z = pd.DataFrame(sc.transform(X.values), index=X.index, columns=X.columns)
    else:
        med = pd.Series(state["robust_med_"])
        iqr = pd.Series(state["robust_iqr_"]).replace(0, 1.0)
        Z = (X - med) / iqr
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
    Carry & Term Structure (Linear/Ridge). Inference-only.

    Params (optional):
      - clip_k_bps: float (default 60.0) -> tanh scaling for signal
      - min_abs_edge_bps: float|None -> optional edge threshold for gating (handled upstream if preferred)
    """
    if state is None:
        raise ValueError(
            "Carry/Term model requires a fitted `state`. "
            "Fit with fit_carry_term_state(...) and pass it here."
        )

    params = params or {}
    clip_k_bps = float(params.get("clip_k_bps", 60.0))
    min_abs_edge_bps = params.get("min_abs_edge_bps", None)

    # Slice & feature checks
    Xfull = df if as_of is None else df.loc[:as_of]
    trained_order = state.get("feature_order", features)
    missing = [f for f in trained_order if f not in Xfull.columns]
    if missing:
        raise KeyError(f"Missing required features for Carry/Term model: {missing}")
    X = Xfull[trained_order].copy()

    # Standardize
    Z = _standardize(X, state)

    # Predict via stored coefficients (no sklearn object needed)
    coef = np.array(state["coef_"], dtype=float).ravel()
    intercept = float(state.get("intercept_", 0.0))
    y_pred_vals = Z.values @ coef + intercept
    y_pred = pd.Series(y_pred_vals, index=Z.index, name=f"carry_term_h{horizon}_bps")

    # Signal
    signal = _tanh_signal(y_pred, clip_k_bps)

    # Uncertainty proxy: EWMA of deviations (target-free)
    alpha = float(state.get("residual_ewm_alpha", 0.05))
    pred_std = (y_pred - y_pred.ewm(alpha=alpha, adjust=False).mean()).abs().ewm(alpha=alpha, adjust=False).mean()
    pred_std.name = "carry_term_pred_std"

    # Confidence
    confidence = _confidence_from_std(pred_std)

    # Optional costs passthrough
    costs = None
    if callable(cost_model):
        try:
            costs = cost_model(signal)
            if isinstance(costs, (pd.Series, pd.DataFrame)) and hasattr(costs, "reindex"):
                costs = costs.reindex(Z.index)
        except Exception:
            costs = None

    # Feature importance (from training)
    fi_dict = state.get("feature_importance", {})
    fi = pd.Series({f: float(fi_dict.get(f, 0.0)) for f in trained_order}, name="coef_abs_norm")

    # Optional trade gating at model level?
    trade_mask = None
    if min_abs_edge_bps is not None:
        trade_mask = (y_pred.abs() >= float(min_abs_edge_bps))
        trade_mask.name = "carry_term_trade_ok"

    result = SubmodelResult(
        y_pred=y_pred,
        signal=signal,
        proba_up=None,
        pred_std=pred_std,
        confidence=confidence,
        costs=costs,
        trade_mask=trade_mask,
        live_metrics=None,
        feature_importance=fi,
        used_features=trained_order,
        warmup_bars=50,
        model_name=str(state.get("model_name", "carry_term_linear")),
        params={
            "horizon": horizon,
            "clip_k_bps": clip_k_bps,
            "min_abs_edge_bps": min_abs_edge_bps,
            "model_type": state.get("model_type", "ridge"),
            "alpha": state.get("alpha", 0.25),
        },
        state=state,
    )
    return result
