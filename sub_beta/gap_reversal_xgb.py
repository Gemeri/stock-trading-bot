# gap_reversal_xgb.py
# ---------------------------------------------------------------------
# Overnight / Gap Reversal (sessioned markets)
# - Model: XGBoost classifier -> P( profitable fill next bar )
# - Label/Horizon: y = 1[r_{t->t+1} > tau], h=1 bar, tau ≈ costs (set in your labeler)
# - Features: ['gap_vs_prev','returns_1','rsi','bollinger_percB','vwap_dev','volume_zscore',
#              'transactions','candle_body_ratio','wick_dominance','rv_20',
#              'hour_sin','hour_cos','day_of_week_sin','day_of_week_cos']
# - Outputs: proba_up, y_pred ≈ (2p-1)*move_scale_bps, signal, pred_std=√(p(1-p)),
#            confidence, trade_mask (avoid extreme rv_20), feature_importance
# ---------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable
import numpy as np
import pandas as pd

# XGBoost
import xgboost as xgb

# Optional: Platt calibration (graceful fallback if sklearn missing)
try:
    from sklearn.linear_model import LogisticRegression
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


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
def fit_gap_reversal_state(
    df_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    features: List[str],
    move_scale_bps: Optional[float] = None,   # if None, auto-estimate from returns_1
    xgb_params: Optional[Dict[str, Any]] = None,
    n_estimators: int = 400,
    early_stopping_rounds: Optional[int] = 50,
    eval_fraction: float = 0.15,
    random_state: int = 42,
    model_name: str = "gap_reversal_xgb",
    calibrate_platt: bool = False,
) -> Dict[str, Any]:
    """
    Fit XGBoost classifier for the gap/overnight reversal model and return a serializable `state`.
    `y_train` must already be cost-adjusted labels: 1 if forward return > tau, else 0.

    Notes:
    - No scaling needed for trees.
    - We compute `scale_pos_weight` automatically if not given.
    - Optional Platt calibration (logistic over raw margins) if sklearn is available.
    """

    # Drop NA rows and align
    X = df_train[features].copy()
    y = y_train.reindex(X.index)
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X.loc[mask], y.loc[mask].astype(int)

    # Auto move scale if not provided (robust median abs move from returns_1 as a proxy)
    if move_scale_bps is None:
        if "returns_1" in X.columns:
            ms = (X["returns_1"].abs().median() * 10_000.0)
            move_scale_bps = float(np.clip(ms, 10.0, 150.0))  # keep within sane bounds
        else:
            move_scale_bps = 50.0  # fallback default

    # Default XGB params
    params = dict(
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1.5,
        reg_lambda=1.5,
        reg_alpha=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        tree_method="hist",   # 'gpu_hist' if GPU is available
    )
    if xgb_params:
        params.update(xgb_params)

    # Class imbalance handling
    if "scale_pos_weight" not in params:
        pos = y.sum()
        neg = max(len(y) - pos, 1)
        params["scale_pos_weight"] = float(neg / max(pos, 1))

    # Train/valid split (time-ordered)
    if eval_fraction and 0.0 < eval_fraction < 0.9:
        split_idx = int(len(X) * (1.0 - eval_fraction))
        X_tr, y_tr = X.iloc[:split_idx], y.iloc[:split_idx]
        X_va, y_va = X.iloc[split_idx:], y.iloc[split_idx:]
        dtrain = xgb.DMatrix(X_tr, label=y_tr, nthread=-1, feature_names=list(X.columns))
        dvalid = xgb.DMatrix(X_va, label=y_va, nthread=-1, feature_names=list(X.columns))
        evals = [(dtrain, "train"), (dvalid, "valid")]
    else:
        dtrain = xgb.DMatrix(X, label=y, nthread=-1, feature_names=list(X.columns))
        evals, dvalid = [(dtrain, "train")], None

    # Train booster
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds if dvalid is not None else None,
        verbose_eval=False,
    )
    best_iter = getattr(booster, "best_iteration", None)
    if best_iter is None:
        # old xgboost only had best_ntree_limit (it’s a *count* of trees)
        best_ntree_limit = int(getattr(booster, "best_ntree_limit", 0) or 0)
        best_iter = best_ntree_limit - 1 if best_ntree_limit > 0 else -1

    # Optional Platt calibration on validation set
    cal = None
    if calibrate_platt and _HAS_SKLEARN and dvalid is not None:
        # Use raw margins (logit) from model as feature for a logistic calibrator
        raw_valid = booster.predict(dvalid, output_margin=True)
        cal_model = LogisticRegression(max_iter=200, solver="lbfgs")
        cal_model.fit(raw_valid.reshape(-1, 1), y_va.values)
        cal = {
            "coef": float(cal_model.coef_.ravel()[0]),
            "intercept": float(cal_model.intercept_.ravel()[0]),
        }

    # Feature importance
    imp_gain = booster.get_score(importance_type="gain")
    # Normalize to sum=1 and include missing features as 0
    total_gain = sum(imp_gain.values()) or 1.0
    fi = {f: float(imp_gain.get(f, 0.0) / total_gain) for f in X.columns}

    # Serialize the booster to bytes
    booster_bytes = booster.save_raw()

    state = {
        "model_name": model_name,
        "feature_order": list(features),
        "xgb_params": params,
        "booster_raw": bytes(booster.save_raw()),
        "best_iteration": int(best_iter),   # <-- store this
        "move_scale_bps": float(move_scale_bps),
        "cal": cal,
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


def _apply_platt(raw_margin: np.ndarray, cal: Optional[Dict[str, float]]) -> np.ndarray:
    # Map raw margins to calibrated probabilities via logistic(a*x + b)
    if cal is None:
        return 1.0 / (1.0 + np.exp(-raw_margin))
    a = cal.get("coef", 1.0)
    b = cal.get("intercept", 0.0)
    z = a * raw_margin + b
    return 1.0 / (1.0 + np.exp(-z))


def _robust_z(series: pd.Series, window: int = 200) -> pd.Series:
    med = series.rolling(window, min_periods=max(20, window // 10)).median()
    mad = series.rolling(window, min_periods=max(20, window // 10)).apply(
        lambda x: np.median(np.abs(x - np.median(x))) if len(x) else np.nan, raw=False
    )
    return (series - med) / (mad + 1e-9)


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
    Overnight / Gap Reversal (XGBoost classification).
    Inference-only: requires a fitted `state` from fit_gap_reversal_state(...).

    Params (optional):
      - clip_k_bps: float (default 50.0) -> tanh scaling for signal
      - rv20_z_max: float (default 3.5)  -> gate out extreme vol regimes
      - p_floor: float (default 0.01)    -> floor for probabilities
      - p_cap: float (default 0.99)      -> cap for probabilities
      - min_abs_edge_bps: float|None     -> optional edge threshold for gating
    """
    if state is None:
        raise ValueError(
            "Gap Reversal requires a fitted `state`. "
            "Fit with fit_gap_reversal_state(...) and pass it here."
        )

    params = params or {}
    clip_k_bps = float(params.get("clip_k_bps", 50.0))
    rv20_z_max = float(params.get("rv20_z_max", 3.5))
    p_floor = float(params.get("p_floor", 0.01))
    p_cap = float(params.get("p_cap", 0.99))
    min_abs_edge_bps = params.get("min_abs_edge_bps", None)

    # Time slice and feature presence checks
    Xfull = df if as_of is None else df.loc[:as_of]
    missing = [f for f in features if f not in Xfull.columns]
    if missing:
        raise KeyError(f"Missing required features for Gap Reversal: {missing}")
    X = Xfull[features].copy()
    booster = xgb.Booster()
    booster.load_model(bytearray(state["booster_raw"]))

    best_iter = int(state.get("best_iteration", -1))
    dmat = xgb.DMatrix(X, nthread=-1, feature_names=list(X.columns))

    try:
        # Newer XGBoost (preferred)
        raw_margin = booster.predict(
            dmat,
            output_margin=True,
            iteration_range=(0, best_iter + 1) if best_iter >= 0 else None,
        )
    except TypeError:
        # Older XGBoost fallback (no iteration_range, but accepts ntree_limit)
        if best_iter >= 0:
            raw_margin = booster.predict(dmat, output_margin=True, ntree_limit=best_iter + 1)
        else:
            raw_margin = booster.predict(dmat, output_margin=True)

    proba = _apply_platt(raw_margin, state.get("cal"))
    proba = np.clip(proba, p_floor, p_cap)
    proba_up = pd.Series(proba, index=X.index, name="gap_reversal_proba_up")

    # Uncertainty proxy (classification entropy std)
    pred_std = pd.Series(np.sqrt(proba_up * (1.0 - proba_up)), index=X.index, name="gap_reversal_pred_std")

    # Expected return proxy in bps
    move_scale_bps = float(state.get("move_scale_bps", 50.0))
    y_pred = pd.Series((2.0 * proba_up.values - 1.0) * move_scale_bps, index=X.index, name=f"gap_reversal_h{horizon}_bps")

    # Signal mapping
    signal = pd.Series(np.tanh(y_pred.values / clip_k_bps), index=X.index, name="gap_reversal_signal")

    # Confidence
    confidence = _confidence_from_std(pred_std)

    # Trade mask: avoid extreme realized vol regimes via robust z on rv_20 if available
    trade_mask = pd.Series(True, index=X.index, name="gap_reversal_trade_ok")
    if "rv_20" in X.columns:
        rvz = _robust_z(X["rv_20"].astype(float))
        trade_mask &= rvz.abs() <= rv20_z_max

    # Optional edge gating
    if min_abs_edge_bps is not None:
        trade_mask &= (y_pred.abs() >= float(min_abs_edge_bps))

    # Optional costs (pass-through)
    costs = None
    if callable(cost_model):
        try:
            costs = cost_model(signal)
            if isinstance(costs, (pd.Series, pd.DataFrame)) and hasattr(costs, "reindex"):
                costs = costs.reindex(X.index)
        except Exception:
            costs = None

    # Feature importance from training (normalized gain)
    fi_dict = state.get("feature_importance", {})
    fi = pd.Series({f: float(fi_dict.get(f, 0.0)) for f in features}, name="gain_norm")

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
        used_features=features,
        warmup_bars=200,  # due to robust z window
        model_name=str(state.get("model_name", "gap_reversal_xgb")),
        params={
            "horizon": horizon,
            "clip_k_bps": clip_k_bps,
            "rv20_z_max": rv20_z_max,
            "p_floor": p_floor,
            "p_cap": p_cap,
            "min_abs_edge_bps": min_abs_edge_bps,
        },
        state=state,
    )
    return result
