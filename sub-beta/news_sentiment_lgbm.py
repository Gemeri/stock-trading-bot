# news_sentiment_lgbm.py
# ---------------------------------------------------------------------
# News / Sentiment (LightGBM, classification)
# Label: y = 1[r_{t->t+h} > tau]; horizon h in {1..3} (target built upstream)
# Features (typical):
#   ['sentiment','d_sentiment','news_volume_z','hour_sin','hour_cos','rv_20', ... PCA_k(embeddings) ...]
# Outputs:
#   - proba_up: P(r_{t->t+h} > tau)
#   - y_pred: (2*proba_up - 1) * move_scale_bps
#   - signal: tanh(y_pred / clip_k_bps)
#   - pred_std: sqrt(p*(1-p))
#   - confidence: mapped from pred_std
#   - feature_importance: normalized gain
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


def fit_news_senti_state(
    df_train: pd.DataFrame,
    y_bin: pd.Series,                      # 0/1 labels for r_{t->t+h} > tau (leakage-safe & aligned)
    *,
    features: List[str],
    lgb_params: Optional[Dict[str, Any]] = None,
    n_estimators: int = 700,
    eval_fraction: float = 0.2,            # time-ordered split
    early_stopping_rounds: Optional[int] = 60,
    random_state: int = 42,
    model_name: str = "news_sentiment_lgbm",
    # calibration for mapping probability -> expected move
    returns_bps: Optional[pd.Series] = None,  # forward returns in bps (same horizon) for move_scale estimate
    move_scale_bps: Optional[float] = None,   # if provided, overrides data-driven estimate
) -> Dict[str, Any]:
    """
    Fit LightGBM classifier mapping sentiment/news features -> P(up move over cost).
    Uses callbacks for early stopping & logging (compatible with LightGBM 3.x/4.x).
    """
    # Align & drop NA
    X = df_train[features].copy()
    y = y_bin.reindex(X.index).astype(float)
    mask = X.notna().all(axis=1) & y.notna()
    if returns_bps is not None:
        r = returns_bps.reindex(X.index).astype(float)
        mask &= r.notna()
    X = X.loc[mask]
    y = y.loc[mask]
    r = returns_bps.loc[mask] if returns_bps is not None else None

    # Time-ordered split
    if eval_fraction and 0.0 < eval_fraction < 0.9 and len(X) >= 10:
        split_idx = int(len(X) * (1 - eval_fraction))
        split_idx = max(1, min(split_idx, len(X) - 1))
        X_tr, y_tr = X.iloc[:split_idx], y.iloc[:split_idx]
        X_va, y_va = X.iloc[split_idx:], y.iloc[split_idx:]
        dtr = lgb.Dataset(X_tr, label=y_tr)
        dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
        has_valid = True
    else:
        dtr = lgb.Dataset(X, label=y)
        dva = None
        has_valid = False

    # Default LightGBM params (binary classification)
    params = dict(
        objective="binary",
        metric="binary_logloss",
        boosting_type="gbdt",
        learning_rate=0.03,
        num_leaves=31,
        min_data_in_leaf=50,
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

    # Handle class imbalance if not specified
    if "scale_pos_weight" not in params:
        pos = float(y.sum())
        neg = float(len(y) - y.sum())
        params["scale_pos_weight"] = (neg / max(pos, 1.0)) if pos > 0 else 1.0

    # ---- Callbacks (LightGBM 4.x style; backward-compatible) ----
    callbacks = [lgb.log_evaluation(period=0)]  # silence logs (replaces verbose_eval)
    if has_valid and early_stopping_rounds and early_stopping_rounds > 0:
        callbacks.append(lgb.early_stopping(stopping_rounds=int(early_stopping_rounds)))

    # Train
    valid_sets = [dtr] + ([dva] if has_valid else [])
    valid_names = ["train"] + (["valid"] if has_valid else [])
    booster = lgb.train(
        params=params,
        train_set=dtr,
        num_boost_round=int(n_estimators),
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,  # <- replaces early_stopping_rounds / verbose_eval
    )

    # Serialize model & metadata
    best_iter = int(getattr(booster, "best_iteration", 0) or booster.current_iteration())
    model_str = booster.model_to_string(num_iteration=best_iter)

    # Gain-based importance (normalized)
    imp_gain = booster.feature_importance(importance_type="gain", iteration=best_iter)
    feat_names = booster.feature_name()
    tot = float(np.sum(imp_gain)) or 1.0
    fi = {f: float(g / tot) for f, g in zip(feat_names, imp_gain)}

    # move_scale calibration
    if move_scale_bps is not None:
        k_bps = float(move_scale_bps)
    elif r is not None and len(r) > 0:
        k_bps = float(0.7 * np.median(np.abs(r.values)))  # robust typical move
        if not np.isfinite(k_bps) or k_bps <= 0:
            k_bps = 40.0
    else:
        k_bps = 40.0

    state: Dict[str, Any] = {
        "model_name": model_name,
        "feature_order": list(features),
        "model_str": model_str,
        "best_iteration": best_iter,
        "feature_importance": fi,
        "lgb_params": params,
        "move_scale_bps": float(k_bps),
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
    News / Sentiment (LightGBM classifier). Inference-only.
    (Unchanged, but kept here in full for completeness.)
    """
    if state is None:
        raise ValueError(
            "News/Sentiment model requires a fitted `state`. "
            "Fit with fit_news_senti_state(...) and pass it here."
        )

    params = params or {}
    clip_k_bps = float(params.get("clip_k_bps", 50.0))
    min_proba_edge = params.get("min_proba_edge", None)
    min_abs_edge_bps = params.get("min_abs_edge_bps", None)

    # Slice & feature checks
    Xfull = df if as_of is None else df.loc[:as_of]
    trained_order = state.get("feature_order", features)
    missing = [f for f in trained_order if f not in Xfull.columns]
    if missing:
        raise KeyError(f"Missing required features for News/Sentiment: {missing}")
    X = Xfull[trained_order].copy()

    # Load booster from string
    booster = lgb.Booster(model_str=state["model_str"])
    ntree = int(state.get("best_iteration", 0)) or None

    # Predict probabilities
    p = booster.predict(X, num_iteration=ntree)           # P(up)
    proba_up = pd.Series(np.clip(p, 1e-6, 1-1e-6), index=X.index, name="news_senti_proba_up")

    # Uncertainty proxy (entropy sigma)
    pred_std = pd.Series(np.sqrt(proba_up * (1.0 - proba_up)), index=X.index, name="news_senti_pred_std")

    # Expected return proxy (bps)
    k_move = float(state.get("move_scale_bps", 40.0))
    y_pred = pd.Series((2.0 * proba_up.values - 1.0) * k_move, index=X.index, name=f"news_senti_h{horizon}_bps")

    # Signal
    signal = pd.Series(np.tanh(y_pred.values / clip_k_bps), index=y_pred.index, name="news_senti_signal")

    # Confidence
    confidence = _confidence_from_std(pred_std)

    # Optional gating
    trade_mask = None
    if (min_proba_edge is not None) or (min_abs_edge_bps is not None):
        trade_mask = pd.Series(True, index=y_pred.index, name="news_senti_trade_ok")
        if min_proba_edge is not None:
            trade_mask &= (proba_up.sub(0.5).abs() >= float(min_proba_edge))
        if min_abs_edge_bps is not None:
            trade_mask &= (y_pred.abs() >= float(min_abs_edge_bps))

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

    return SubmodelResult(
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
        model_name=str(state.get("model_name", "news_sentiment_lgbm")),
        params={
            "horizon": horizon,
            "clip_k_bps": clip_k_bps,
            "min_proba_edge": min_proba_edge,
            "min_abs_edge_bps": min_abs_edge_bps,
            "move_scale_bps": k_move,
        },
        state=state,
    )
