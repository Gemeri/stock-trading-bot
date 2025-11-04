# kday_reversal_catboost.py
# ---------------------------------------------------------------------
# K-day Reversal (1–3 bars) using CatBoost (regression)
# Label/Horizon: h=1 (forward return in bps)
# Features: ['returns_1','returns_3','bollinger_percB','rsi','vwap_dev','volume_zscore',
#            'obv','candle_body_ratio','wick_dominance','adv_ratio','rv_20','hour_sin','hour_cos']
# Outputs: y_pred (bps), signal [-1,1], pred_std (ensemble or rolling), confidence
# ---------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable
import numpy as np
import pandas as pd
import pickle

from catboost import CatBoostRegressor, Pool


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
def fit_kday_reversal_state(
    df_train: pd.DataFrame,
    target_bps: pd.Series,
    *,
    features: List[str],
    cat_params: Optional[Dict[str, Any]] = None,
    n_estimators: int = 600,
    eval_fraction: float = 0.15,
    early_stopping_rounds: int = 50,
    n_ensembles: int = 1,             # >1 enables ensemble-based uncertainty
    random_state: int = 42,
    model_name: str = "kday_reversal_cat",
    residual_ewm_alpha: float = 0.1,  # used if uncertainty_method='rolling'
) -> Dict[str, Any]:
    """
    Fit CatBoost regressor(s) predicting forward return in bps for h=1.
    Returns a serializable `state` dict with one or more pickled models, feature order, and importances.
    """
    # Align & drop NA
    X = df_train[features].copy()
    y = target_bps.reindex(X.index)
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X.loc[mask], y.loc[mask].astype(float)

    # Time-ordered split for eval
    if eval_fraction and 0.0 < eval_fraction < 0.9:
        split = int(len(X) * (1 - eval_fraction))
        X_tr, y_tr = X.iloc[:split], y.iloc[:split]
        X_va, y_va = X.iloc[split:], y.iloc[split:]
        pool_tr = Pool(X_tr, y_tr)
        pool_va = Pool(X_va, y_va)
    else:
        pool_tr = Pool(X, y)
        pool_va = None

    # Default CatBoost params (robust, fast)
    base_params = dict(
        loss_function="RMSE",
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=6.0,
        subsample=0.8,
        rsm=0.8,
        random_strength=1.0,
        bootstrap_type="MVS",
        eval_metric="RMSE",
        verbose=False,
        random_seed=random_state,
        allow_writing_files=False,
    )
    if cat_params:
        base_params.update(cat_params)

    # Train ensemble
    models: List[CatBoostRegressor] = []
    for i in range(max(1, int(n_ensembles))):
        params_i = dict(base_params)
        params_i["random_seed"] = int(base_params.get("random_seed", random_state)) + i
        params_i["iterations"] = int(n_estimators)

        model = CatBoostRegressor(**params_i)

        if pool_va is not None:
            # Newer CatBoost (supports early_stopping_rounds in fit)
            try:
                model.fit(
                    pool_tr,
                    eval_set=pool_va,
                    use_best_model=True,
                    early_stopping_rounds=early_stopping_rounds,
                )
            except TypeError:
                # Older CatBoost fallback: use od_* params instead
                model.set_params(od_type="Iter", od_wait=int(early_stopping_rounds), use_best_model=True)
                model.fit(pool_tr, eval_set=pool_va)
        else:
            model.fit(pool_tr)

        models.append(model)


    # Feature importance (PredictionValuesChange) from first model
    try:
        fi_vals = models[0].get_feature_importance(type="PredictionValuesChange", data=pool_tr)
        fi = {f: float(v) for f, v in zip(X.columns, fi_vals)}
        total = sum(abs(v) for v in fi.values()) or 1.0
        fi = {k: abs(v)/total for k, v in fi.items()}
    except Exception:
        fi = {f: 0.0 for f in X.columns}

    # Serialize models via pickle
    ensemble_pkls = [pickle.dumps(m) for m in models]

    state = {
        "model_name": model_name,
        "feature_order": list(features),
        "ensemble_pkls": ensemble_pkls,               # list of pickled CatBoostRegressor
        "n_ensembles": int(n_ensembles),
        "residual_ewm_alpha": float(residual_ewm_alpha),
        "feature_importance": fi,
    }
    return state


# ---------------------------------------------
# Internal utilities (pure, side-effect free)
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
    K-day Reversal (CatBoost, regression). Inference-only.

    Params (optional):
      - clip_k_bps: float (default 45.0)       -> tanh scaling for signal
      - uncertainty_method: 'ensemble'|'rolling' (default 'ensemble' if n_ensembles>1 else 'rolling')
      - min_abs_edge_bps: float|None           -> optional edge threshold for gating (handled upstream)
    """
    if state is None:
        raise ValueError(
            "K-day Reversal requires a fitted `state`. "
            "Fit with fit_kday_reversal_state(...) and pass it here."
        )

    params = params or {}
    clip_k_bps = float(params.get("clip_k_bps", 45.0))
    uncertainty_method = params.get("uncertainty_method", None)
    min_abs_edge_bps = params.get("min_abs_edge_bps", None)

    # Slice & feature checks
    Xfull = df if as_of is None else df.loc[:as_of]
    trained_order = state.get("feature_order", features)
    missing = [f for f in trained_order if f not in Xfull.columns]
    if missing:
        raise KeyError(f"Missing required features for K-day Reversal: {missing}")
    X = Xfull[trained_order].copy()

    # Restore ensemble
    n_ens = int(state.get("n_ensembles", 1))
    models: List[CatBoostRegressor] = [pickle.loads(b) for b in state["ensemble_pkls"]]
    if uncertainty_method is None:
        uncertainty_method = "ensemble" if n_ens > 1 else "rolling"

    # Predictions
    preds = []
    for m in models:
        preds.append(pd.Series(m.predict(Pool(X)), index=X.index))
    y_pred = pd.concat(preds, axis=1).mean(axis=1) if len(preds) > 1 else preds[0]
    y_pred.name = f"kday_reversal_h{horizon}_bps"

    # Uncertainty
    if uncertainty_method == "ensemble" and len(preds) > 1:
        pred_std = pd.concat(preds, axis=1).std(axis=1)
    else:
        # rolling proxy on prediction innovations (target-free)
        alpha = float(state.get("residual_ewm_alpha", 0.1))
        pred_std = (y_pred - y_pred.ewm(alpha=alpha, adjust=False).mean()).abs().ewm(alpha=alpha, adjust=False).mean()
    pred_std.name = "kday_reversal_pred_std"

    # Signal & confidence
    signal = _tanh_signal(y_pred, clip_k_bps)
    confidence = _confidence_from_std(pred_std)

    # Optional costs pass-through (not used here)
    costs = None
    if callable(cost_model):
        try:
            costs = cost_model(signal)
            if isinstance(costs, (pd.Series, pd.DataFrame)) and hasattr(costs, "reindex"):
                costs = costs.reindex(X.index)
        except Exception:
            costs = None

    # Feature importance from training
    fi_dict = state.get("feature_importance", {})
    fi = pd.Series({f: float(fi_dict.get(f, 0.0)) for f in trained_order}, name="importance_norm")

    # Optional edge gating at this model level? (Spec: no trade_mask; leave to meta/risk layer)
    trade_mask = None
    if min_abs_edge_bps is not None:
        trade_mask = (y_pred.abs() >= float(min_abs_edge_bps))
        trade_mask.name = "kday_reversal_trade_ok"

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
        model_name=str(state.get("model_name", "kday_reversal_cat")),
        params={
            "horizon": horizon,
            "clip_k_bps": clip_k_bps,
            "uncertainty_method": uncertainty_method,
            "min_abs_edge_bps": min_abs_edge_bps,
        },
        state=state,  # unchanged in pure inference
    )
    return result
