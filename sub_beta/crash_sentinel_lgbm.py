# crash_sentinel_lgbm.py
# ---------------------------------------------------------------------
# 10) Crash-Sentinel (LightGBM, classification)
#
# Purpose:
#   Detect elevated short-horizon crash risk and provide a gating/de-risk
#   signal to the meta layer. We predict P(safe) = 1 - P(crash), then map
#   to a *negative* expected return proxy (bps) when crash risk is high.
#
# Label/Horizon:
#   - y_crash ∈ {0,1} where 1 means "crash" in next h bars (built upstream).
#   - horizon h ∈ {1,2,3} (kept as metadata; model predicts context for h).
#
# Features (typical):
#   ['rv_20','semivar_dn_20','dd_63','bb_bw','adv_ratio','volume_zscore',
#    'transactions','macd_hist_flip','macd_cross', ... (spreads/liquidity if available)]
#
# Outputs:
#   - proba_up:   P(safe)  (i.e., 1 - crash probability)
#   - y_pred:     (proba_up - 1.0) * move_scale_bps   # ≤ 0, more negative when crash risk ↑
#   - signal:     tanh(y_pred / clip_k_bps)           # in [-1, 0]
#   - pred_std:   sqrt(p*(1-p))                       # entropy proxy
#   - confidence: mapped from pred_std
#   - trade_mask: True when de-risk condition holds (P(crash) ≥ threshold)
#
# Notes:
#   - trade_mask semantics here = "DE-RISK" (not 'trade-ok'). We record this
#     explicitly in `params['mask_action']="de_risk"` so the meta layer can
#     interpret it correctly.
# ---------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable
import numpy as np
import pandas as pd
import lightgbm as lgb

try:
    # For optional Platt calibration
    from sklearn.linear_model import LogisticRegression
except Exception:
    LogisticRegression = None


# ------------------------
# Shared result container
# ------------------------
@dataclass
class SubmodelResult:
    # Core
    y_pred: pd.Series                   # expected return over `horizon` (bps; here ≤ 0 on average)
    signal: pd.Series                   # normalized signal in [-1, 1] (here in [-1, 0])

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


# ---------------------------------------------
# Small utilities
# ---------------------------------------------
def _tanh_signal(y_pred_bps: pd.Series, k_bps: float) -> pd.Series:
    return np.tanh(y_pred_bps / float(k_bps))

def _confidence_from_std(pred_std: pd.Series) -> pd.Series:
    # Convert entropy proxy to 0..1 confidence (lower entropy => higher confidence)
    if pred_std.isna().all():
        return pd.Series(index=pred_std.index, dtype=float)
    m = pred_std.median()
    mad = (pred_std - m).abs().median()
    z = (pred_std - m) / (mad + 1e-9)
    conf = 1.0 / (1.0 + z.clip(lower=0))
    return conf.clip(0, 1)

def _safe_prob_platt(p_raw: np.ndarray, platt: Optional[Dict[str, float]]) -> np.ndarray:
    """Optionally recalibrate probabilities via a stored Platt scaler on logits."""
    if not platt:
        return p_raw
    # Guard against 0/1
    p_raw = np.clip(p_raw, 1e-6, 1 - 1e-6)
    logit = np.log(p_raw / (1.0 - p_raw))
    a = float(platt.get("coef", 1.0))
    b = float(platt.get("intercept", 0.0))
    p_cal = 1.0 / (1.0 + np.exp(-(a * logit + b)))
    return np.clip(p_cal, 1e-6, 1 - 1e-6)


# ===========================================================
# Training helper (builds `state` used by predict_submodel)
# ===========================================================
def fit_crash_sentinel_state(
    df_train: pd.DataFrame,
    y_crash: pd.Series,                    # 0/1 label: 1 = crash in next h bars (leakage-safe & aligned)
    *,
    features: List[str],
    lgb_params: Optional[Dict[str, Any]] = None,
    n_estimators: int = 700,
    eval_fraction: float = 0.2,            # time-ordered split
    early_stopping_rounds: Optional[int] = 60,
    random_state: int = 42,
    model_name: str = "crash_sentinel_lgbm",
    # Calibration & gating
    use_platt: bool = True,
    move_scale_bps: Optional[float] = 60.0,  # magnitude of negative edge when crash risk is extreme
    p_crash_gate: float = 0.35,            # de-risk if P(crash) >= this (i.e., p_safe <= 0.65)
) -> Dict[str, Any]:
    """
    Fit LightGBM classifier on crash labels, returning a serializable `state`.

    - We train for y_safe = 1 - y_crash, because the model outputs P(safe).
    - Optional simple Platt calibration is learned on time-held-out validation.
    """
    # Assemble & align
    X = df_train[features].copy()
    y_c = y_crash.reindex(X.index).astype(float)
    mask = X.notna().all(axis=1) & y_c.notna()
    X = X.loc[mask]
    y_c = y_c.loc[mask]
    y_safe = 1.0 - y_c  # target for training: 1 = safe, 0 = crash

    # Time-ordered split
    if eval_fraction and 0.0 < eval_fraction < 0.9:
        split_idx = int(len(X) * (1 - eval_fraction))
        X_tr, y_tr = X.iloc[:split_idx], y_safe.iloc[:split_idx]
        X_va, y_va = X.iloc[split_idx:], y_safe.iloc[split_idx:]
        dtr = lgb.Dataset(X_tr, label=y_tr)
        dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
    else:
        dtr = lgb.Dataset(X, label=y_safe)
        dva = None

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

    # Handle imbalance (crash events are rare → safe dominates)
    if "scale_pos_weight" not in params:
        pos = float(y_safe.sum())                      # count of "safe"
        neg = float(len(y_safe) - y_safe.sum())        # count of "crash"
        # We want the *minority* class to get more weight. If crash is minority, scale_pos_weight < 1 is OK,
        # but more principled is to swap target; however, we'll keep y_safe and adjust as below:
        # Put a cap for numerical stability.
        minority = min(pos, neg)
        majority = max(pos, neg)
        params["scale_pos_weight"] = (majority / max(minority, 1.0))

    # Train booster (supports both new callback API and older kwargs)
    valid_sets = [dtr] + ([dva] if dva is not None else [])
    valid_names = ["train"] + (["valid"] if dva is not None else [])

    use_es = (dva is not None) and (early_stopping_rounds is not None) and (early_stopping_rounds > 0)

    # Prefer callbacks (newer LightGBM)
    callbacks = []
    if use_es and hasattr(lgb, "early_stopping"):
        callbacks.append(lgb.early_stopping(early_stopping_rounds, first_metric_only=True, verbose=False))
    # Silence periodic logging (replacement for verbose_eval)
    if hasattr(lgb, "log_evaluation"):
        callbacks.append(lgb.log_evaluation(period=0))

    try:
        booster = lgb.train(
            params,
            dtr,
            num_boost_round=n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,  # new style
        )
    except TypeError:
        # Fallback for older LightGBM (< callback-only)
        booster = lgb.train(
            params,
            dtr,
            num_boost_round=n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            early_stopping_rounds=early_stopping_rounds if use_es else None,
            verbose_eval=False,
        )

    best_iter = int(booster.best_iteration or booster.current_iteration())
    model_str = booster.model_to_string(num_iteration=best_iter)

    # Gain-based importance (normalized)
    imp_gain = booster.feature_importance(importance_type="gain", iteration=best_iter)
    feat_names = booster.feature_name()
    tot = float(np.sum(imp_gain)) or 1.0
    fi = {f: float(g / tot) for f, g in zip(feat_names, imp_gain)}

    # Optional Platt calibration on validation slice (if available & sklearn present)
    platt: Optional[Dict[str, float]] = None
    if use_platt and (dva is not None) and LogisticRegression is not None:
        p_va = booster.predict(X_va, num_iteration=best_iter)
        # Avoid degenerate probs
        p_va = np.clip(p_va, 1e-6, 1 - 1e-6)
        logit_va = np.log(p_va / (1.0 - p_va)).reshape(-1, 1)
        # Fit a simple LR on logits -> y_safe
        lr = LogisticRegression(solver="lbfgs")
        try:
            lr.fit(logit_va, y_va.values.ravel())
            coef = float(lr.coef_[0, 0])
            intercept = float(lr.intercept_[0])
            platt = {"coef": coef, "intercept": intercept}
        except Exception:
            platt = None

    state: Dict[str, Any] = {
        "model_name": model_name,
        "feature_order": list(features),
        "model_str": model_str,
        "best_iteration": best_iter,
        "feature_importance": fi,
        "lgb_params": params,
        "platt": platt,
        "move_scale_bps": float(move_scale_bps if (move_scale_bps is not None and move_scale_bps > 0) else 60.0),
        "p_crash_gate": float(p_crash_gate),
    }
    return state


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
    Crash-Sentinel (LightGBM classifier). Inference-only.

    Params (optional):
      - clip_k_bps: float (default 50.0)     -> tanh scaling for signal
      - p_crash_gate: float                  -> override state gate; de-risk if P(crash) >= threshold
      - min_abs_edge_bps: float|None         -> optional mask when |y_pred| >= threshold (additional to gate)
    """
    if state is None:
        raise ValueError(
            "Crash-Sentinel model requires a fitted `state`. "
            "Fit with fit_crash_sentinel_state(...) and pass it here."
        )

    params = params or {}
    clip_k_bps = float(params.get("clip_k_bps", 50.0))
    p_crash_gate = float(params.get("p_crash_gate", state.get("p_crash_gate", 0.35)))
    min_abs_edge_bps = params.get("min_abs_edge_bps", None)

    # Slice & feature checks
    Xfull = df if as_of is None else df.loc[:as_of]
    trained_order = state.get("feature_order", features)
    missing = [f for f in trained_order if f not in Xfull.columns]
    if missing:
        raise KeyError(f"Missing required features for Crash-Sentinel: {missing}")
    X = Xfull[trained_order].copy()

    # Load booster from string
    booster = lgb.Booster(model_str=state["model_str"])
    ntree = int(state.get("best_iteration", 0)) or None

    # Predict raw probability of SAFE (not crash)
    p_safe_raw = booster.predict(X, num_iteration=ntree)
    p_safe = _safe_prob_platt(np.asarray(p_safe_raw), state.get("platt", None))
    proba_up = pd.Series(np.clip(p_safe, 1e-6, 1 - 1e-6), index=X.index, name="crash_sentinel_proba_safe")

    # Uncertainty proxy (entropy sigma)
    pred_std = pd.Series(np.sqrt(proba_up * (1.0 - proba_up)), index=X.index, name="crash_sentinel_pred_std")

    # Map to expected return proxy (bps): negative when risk↑
    k_move = float(state.get("move_scale_bps", 60.0))
    # y_pred ≤ 0 typically; equals 0 when p_safe == 1.0
    y_pred = pd.Series((proba_up.values - 1.0) * k_move, index=X.index, name=f"crash_sentinel_h{horizon}_bps")

    # Signal in [-1, 0]
    signal = pd.Series(_tanh_signal(y_pred, clip_k_bps), index=y_pred.index, name="crash_sentinel_signal")

    # Confidence
    confidence = _confidence_from_std(pred_std)

    # Gate for de-risking: True where we recommend de-risk
    p_crash = 1.0 - proba_up
    trade_mask = (p_crash >= p_crash_gate)
    trade_mask.name = "crash_sentinel_de_risk"

    # Optional extra gating on |edge|
    if min_abs_edge_bps is not None:
        trade_mask = trade_mask | (y_pred.abs() >= float(min_abs_edge_bps))

    # Optional costs passthrough (usually not used here)
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

    result = SubmodelResult(
        y_pred=y_pred,
        signal=signal,
        proba_up=proba_up,       # proba_safe
        pred_std=pred_std,
        confidence=confidence,
        costs=costs,
        trade_mask=trade_mask,   # semantics = "DE-RISK"
        live_metrics=None,
        feature_importance=fi,
        used_features=trained_order,
        warmup_bars=0,
        model_name=str(state.get("model_name", "crash_sentinel_lgbm")),
        params={
            "horizon": horizon,
            "clip_k_bps": clip_k_bps,
            "p_crash_gate": p_crash_gate,
            "move_scale_bps": k_move,
            "mask_action": "de_risk",       # IMPORTANT semantics for meta layer
            "min_abs_edge_bps": min_abs_edge_bps,
        },
        state=state,
    )
    return result


# ===========================================================
# (Optional) Helper to build crash labels upstream
# ===========================================================
def build_crash_label_from_returns(
    fwd_ret_bps: pd.Series,
    *,
    vol_ref: Optional[pd.Series] = None,   # e.g., rolling std of returns in bps
    k_sigma: float = 2.5,
    absolute_bps_floor: Optional[float] = None,
) -> pd.Series:
    """
    Convenience method (not used by predict). Create a binary y_crash label with two triggers:
      1) fwd_ret_bps <= -k_sigma * vol_ref
      2) fwd_ret_bps <= -absolute_bps_floor (if provided)

    Returns a 0/1 Series aligned to `fwd_ret_bps`.
    """
    y = pd.Series(0, index=fwd_ret_bps.index, dtype=int)
    cond = pd.Series(False, index=fwd_ret_bps.index)
    if vol_ref is not None:
        cond = cond | (fwd_ret_bps <= -(k_sigma * vol_ref.astype(float)))
    if absolute_bps_floor is not None and absolute_bps_floor > 0:
        cond = cond | (fwd_ret_bps <= -float(absolute_bps_floor))
    y.loc[cond.fillna(False)] = 1
    return y
