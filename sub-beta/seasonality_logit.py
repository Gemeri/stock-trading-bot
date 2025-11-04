# seasonality_logit.py
# ---------------------------------------------------------------------
# Seasonality / Turn-of-Month (Logistic Regression, L2)
# Label/Horizon: y = 1[r_{t->t+1} > tau], h=1 bar (tau ≈ costs; label built upstream)
# Features: ['hour_sin','hour_cos','day_of_week_sin','day_of_week_cos',
#            'month_sin','month_cos','rv_20','adv_ratio', ('rsi','bollinger_percB' optional)]
# Outputs: proba_up, y_pred ≈ (2p-1)*move_scale_bps, signal, pred_std=√(p(1-p)), confidence
# ---------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ------------------------
# Shared result container
# ------------------------
@dataclass
class SubmodelResult:
    # Core
    y_pred: pd.Series                   # expected return proxy over `horizon` (bps)
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
def fit_seasonality_state(
    df_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    features: List[str],
    C: float = 1.0,                         # inverse of L2 strength
    class_weight: Optional[str | dict] = "balanced",
    max_iter: int = 1000,
    tol: float = 1e-6,
    random_state: int = 42,
    move_scale_bps: Optional[float] = None, # scales prob edge to bps (see below)
    model_name: str = "seasonality_logit_l2",
    scaler_mode: str = "standard",          # 'standard' | 'robust'
) -> Dict[str, Any]:
    """
    Fit an L2-regularized Logistic Regression on cyclical/time features to predict
    P( profitable next-bar trade ) after costs. Returns a serializable `state`.

    y_train : binary labels already cost-adjusted upstream (0/1).
    move_scale_bps : if None, infer a conservative default (40 bps), or pass per-asset value.
    """
    # Align & drop NA
    X = df_train[features].copy()
    y = y_train.reindex(X.index)
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X.loc[mask], y.loc[mask].astype(int)

    # Build scaler
    if scaler_mode == "standard":
        scaler = StandardScaler()
        Xz = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
    elif scaler_mode == "robust":
        # Robust standardization via med/IQR
        med = X.median()
        iqr = (X.quantile(0.75) - X.quantile(0.25)).replace(0, 1.0)
        scaler = None  # we'll serialize med/iqr manually
        Xz = (X - med) / iqr
    else:
        raise ValueError("scaler_mode must be 'standard' or 'robust'")

    # Logistic Regression (L2)
    logit = LogisticRegression(
        penalty="l2",
        C=C,
        solver="lbfgs",
        max_iter=max_iter,
        tol=tol,
        class_weight=class_weight,
        random_state=random_state,
    )
    logit.fit(Xz.values, y.values)

    # Move scaling (probability -> bps edge)
    if move_scale_bps is None:
        move_scale_bps = 40.0  # conservative default for daily/session bars; tune per asset

    # Serialize state
    state: Dict[str, Any] = {
        "model_name": model_name,
        "feature_order": list(features),
        "coef_": logit.coef_.astype(float).ravel().tolist(),
        "intercept_": float(logit.intercept_.ravel()[0]),
        "C": float(C),
        "class_weight": class_weight if isinstance(class_weight, str) else dict(class_weight),
        "scaler_mode": scaler_mode,
        "move_scale_bps": float(move_scale_bps),
    }
    if scaler_mode == "standard":
        state.update({
            "scaler_mean_": scaler.mean_.tolist(),
            "scaler_scale_": scaler.scale_.tolist(),
            "scaler_var_": scaler.var_.tolist(),
            "scaler_n_samples_seen_": int(scaler.n_samples_seen_),
        })
    else:
        state.update({
            "robust_med_": med.astype(float).to_dict(),
            "robust_iqr_": iqr.astype(float).to_dict(),
        })

    # Store normalized coefficient importances (on standardized space)
    coef = np.array(state["coef_"], dtype=float)
    coef_abs = np.abs(coef)
    total = coef_abs.sum() or 1.0
    fi = {f: float(a/total) for f, a in zip(features, coef_abs)}
    state["feature_importance"] = fi

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


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


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
    Seasonality / Turn-of-Month (Logistic, L2). Inference-only.

    Params (optional):
      - clip_k_bps: float (default 40.0) -> tanh scaling for signal
      - p_floor/p_cap: float (defaults 0.02/0.98) -> probability flooring/capping
      - min_abs_edge_bps: float|None -> optional edge threshold for gating (handled upstream)
    """
    if state is None:
        raise ValueError(
            "Seasonality model requires a fitted `state`. "
            "Fit with fit_seasonality_state(...) and pass it here."
        )

    params = params or {}
    clip_k_bps = float(params.get("clip_k_bps", 40.0))
    p_floor = float(params.get("p_floor", 0.02))
    p_cap = float(params.get("p_cap", 0.98))
    min_abs_edge_bps = params.get("min_abs_edge_bps", None)

    # Slice & feature check
    Xfull = df if as_of is None else df.loc[:as_of]
    trained_order = state.get("feature_order", features)
    missing = [f for f in trained_order if f not in Xfull.columns]
    if missing:
        raise KeyError(f"Missing required features for Seasonality model: {missing}")
    X = Xfull[trained_order].copy()

    # Standardize using stored stats
    Z = _standardize(X, state)

    # Compute logits and probabilities
    coef = np.array(state["coef_"], dtype=float).ravel()
    intercept = float(state["intercept_"])
    logits = Z.values @ coef + intercept
    proba = np.clip(_sigmoid(logits), p_floor, p_cap)
    proba_up = pd.Series(proba, index=Z.index, name="seasonality_proba_up")

    # Uncertainty proxy from classification variance
    pred_std = pd.Series(np.sqrt(proba_up * (1.0 - proba_up)), index=Z.index, name="seasonality_pred_std")

    # Expected return proxy (bps)
    move_scale_bps = float(state.get("move_scale_bps", 40.0))
    y_pred = pd.Series((2.0 * proba_up.values - 1.0) * move_scale_bps, index=Z.index, name=f"seasonality_h{horizon}_bps")

    # Signal
    signal = _tanh_signal(y_pred, clip_k_bps)

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

    # Feature importance (normalized |coef| on standardized space)
    fi_dict = state.get("feature_importance", {})
    fi = pd.Series({f: float(fi_dict.get(f, 0.0)) for f in trained_order}, name="coef_abs_norm")

    # Optional gating (not required by spec). If requested:
    trade_mask = None
    if min_abs_edge_bps is not None:
        trade_mask = (y_pred.abs() >= float(min_abs_edge_bps))
        trade_mask.name = "seasonality_trade_ok"

    result = SubmodelResult(
        y_pred=y_pred,
        signal=signal,
        proba_up=proba_up,
        pred_std=pred_std,
        confidence=confidence,
        costs=costs,
        trade_mask=trade_mask,
        live_metrics=None,
        feature_importance=fi,
        used_features=trained_order,
        warmup_bars=0,
        model_name=str(state.get("model_name", "seasonality_logit_l2")),
        params={
            "horizon": horizon,
            "clip_k_bps": clip_k_bps,
            "p_floor": p_floor,
            "p_cap": p_cap,
            "min_abs_edge_bps": min_abs_edge_bps,
        },
        state=state,
    )
    return result
