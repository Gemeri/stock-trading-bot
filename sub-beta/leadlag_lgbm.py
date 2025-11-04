# leadlag_lgbm.py
# ---------------------------------------------------------------------
# Lead–Lag Cross-Asset
# Model: LightGBM (regression)
# Horizon: h in {1..3} bars (build forward target upstream in bps)
# Features (typical):
#   ['leader_ret_1','leader_ret_2','leader_ret_3',
#    'leader_momentum_5','leader_rv_20','spread_z','rv_20']
# Output: y_pred (bps), signal [-1,1], pred_std (EWMA proxy), confidence, feature_importance
# ---------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable
import numpy as np
import pandas as pd
import lightgbm as lgb


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
def fit_leadlag_state(
    df_train: pd.DataFrame,
    target_bps: pd.Series,
    *,
    features: List[str],
    lgb_params: Optional[Dict[str, Any]] = None,
    n_estimators: int = 800,
    eval_fraction: float = 0.2,             # time-ordered split
    early_stopping_rounds: Optional[int] = 50,
    random_state: int = 42,
    model_name: str = "leadlag_lgbm",
    residual_ewm_alpha: float = 0.06,       # for target-free pred_std proxy
) -> Dict[str, Any]:
    """
    Fit LightGBM regressor mapping leader features -> target asset forward returns (bps).
    Returns a serializable `state` dict with model bytes and metadata.
    """
    # Align & drop NA
    X = df_train[features].copy()
    y = target_bps.reindex(X.index)
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X.loc[mask], y.loc[mask].astype(float)

    # Time-ordered split
    if eval_fraction and 0.0 < eval_fraction < 0.9:
        split_idx = int(len(X) * (1 - eval_fraction))
        X_tr, y_tr = X.iloc[:split_idx], y.iloc[:split_idx]
        X_va, y_va = X.iloc[split_idx:], y.iloc[split_idx:]
        dtr = lgb.Dataset(X_tr, label=y_tr)
        dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
    else:
        dtr = lgb.Dataset(X, label=y)
        dva = None

    # Default LGB params
    params = dict(
        objective="regression",
        metric="l2",
        boosting_type="gbdt",
        learning_rate=0.03,
        num_leaves=31,
        min_data_in_leaf=40,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        lambda_l2=1.0,
        lambda_l1=0.0,
        verbose=-1,
        seed=random_state,
        deterministic=True,
        force_col_wise=True,
    )
    if lgb_params:
        params.update(lgb_params)
    callbacks = []
    if dva is not None and early_stopping_rounds:
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))
    # silence logs (replacement for verbose_eval=False)
    callbacks.append(lgb.log_evaluation(period=0))

    booster = lgb.train(
        params,
        dtr,
        num_boost_round=n_estimators,
        valid_sets=[dtr] + ([dva] if dva is not None else []),
        valid_names=["train"] + (["valid"] if dva is not None else []),
        callbacks=callbacks,           # <-- use callbacks instead of kwargs
    )

    # Serialize model & metadata
    model_str = booster.model_to_string(num_iteration=booster.best_iteration or booster.current_iteration())
    best_iter = int(booster.best_iteration or booster.current_iteration())

    # Gain-based importance (normalized)
    imp_gain = booster.feature_importance(importance_type="gain", iteration=best_iter)
    feat_names = booster.feature_name()
    tot = float(np.sum(imp_gain)) or 1.0
    fi = {f: float(g / tot) for f, g in zip(feat_names, imp_gain)}

    state: Dict[str, Any] = {
        "model_name": model_name,
        "feature_order": list(features),
        "model_str": model_str,
        "best_iteration": best_iter,
        "feature_importance": fi,
        "residual_ewm_alpha": float(residual_ewm_alpha),
        "lgb_params": params,
    }
    return state


# ---------------------------------------------
# Small utilities
# ---------------------------------------------
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
    Lead–Lag Cross-Asset (LightGBM). Inference-only.

    Params (optional):
      - clip_k_bps: float (default 60.0) -> tanh scaling for signal
      - min_abs_edge_bps: float|None     -> optional gate
    """
    if state is None:
        raise ValueError(
            "Lead–Lag model requires a fitted `state`. "
            "Fit with fit_leadlag_state(...) and pass it here."
        )

    params = params or {}
    clip_k_bps = float(params.get("clip_k_bps", 60.0))
    min_abs_edge_bps = params.get("min_abs_edge_bps", None)

    # Slice & feature checks
    Xfull = df if as_of is None else df.loc[:as_of]
    trained_order = state.get("feature_order", features)
    missing = [f for f in trained_order if f not in Xfull.columns]
    if missing:
        raise KeyError(f"Missing required features for Lead–Lag model: {missing}")
    X = Xfull[trained_order].copy()

    # Load booster from string
    booster = lgb.Booster(model_str=state["model_str"])
    ntree = int(state.get("best_iteration", 0)) or None

    # Predict (bps)
    yhat = booster.predict(X, num_iteration=ntree)
    y_pred = pd.Series(yhat, index=X.index, name=f"leadlag_h{horizon}_bps")

    # Signal
    signal = _tanh_signal(y_pred, clip_k_bps)

    # Uncertainty proxy (target-free): EWMA of prediction deviations
    alpha = float(state.get("residual_ewm_alpha", 0.06))
    pred_std = (y_pred - y_pred.ewm(alpha=alpha, adjust=False).mean()).abs().ewm(alpha=alpha, adjust=False).mean()
    pred_std.name = "leadlag_pred_std"

    # Confidence
    confidence = _confidence_from_std(pred_std)

    # Optional costs passthrough
    costs = None
    if callable(cost_model):
        try:
            costs = cost_model(signal)
            if isinstance(costs, (pd.Series, pd.DataFrame)) and hasattr(costs, "reindex"):
                costs = costs.reindex(y_pred.index)
        except Exception:
            costs = None

    # Feature importance from training
    fi_dict = state.get("feature_importance", {})
    fi = pd.Series({f: float(fi_dict.get(f, 0.0)) for f in trained_order}, name="gain_norm")

    # Optional gating
    trade_mask = None
    if min_abs_edge_bps is not None:
        trade_mask = (y_pred.abs() >= float(min_abs_edge_bps))
        trade_mask.name = "leadlag_trade_ok"

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
        model_name=str(state.get("model_name", "leadlag_lgbm")),
        params={
            "horizon": horizon,
            "clip_k_bps": clip_k_bps,
            "min_abs_edge_bps": min_abs_edge_bps,
        },
        state=state,
    )
    return result
