# equity_value_quality_xgb.py
# ---------------------------------------------------------------------
# Equity Value–Quality (cross-sectional)
# Model: XGBoost (regression or ranking)
# Horizon: monthly or h in {24..48} bars (build forward target upstream)
# Cross-sectional features (per date across many tickers):
#   value:   ebit_ev, bm, fcf_ev
#   quality: roic, margins, accruals
#   momentum: mom_12_1
#   risk:    vol_z, size_z, beta_z
#
# Inference requires only features. For ranking, training needs grouped labels
# by DATE (e.g., forward return per stock for that month); DMatrix groups set per date.
#
# Output: y_pred (bps alpha proxy), signal [-1,1], pred_std (target-free), confidence,
#         feature_importance (gain-normalized).
# ---------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb

# ------------------------
# Shared result container
# ------------------------
@dataclass
class SubmodelResult:
    # Core
    y_pred: pd.Series
    signal: pd.Series

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
# Helpers
# ===========================================================
def _infer_date_and_asset(df: pd.DataFrame) -> Tuple[pd.Index, Optional[str]]:
    """
    Returns (date_index, asset_col or None).
    - If MultiIndex -> level 0 as date, level 1 as asset.
    - Else -> use df.index as date, and try asset column: 'asset'|'ticker'|'symbol'.
    """
    if isinstance(df.index, pd.MultiIndex):
        date_index = df.index.get_level_values(0)
        return date_index, None  # asset is in index level 1; handled via groupby(level=1)
    # single-level index
    for c in ("asset", "ticker", "symbol"):
        if c in df.columns:
            return df.index, c
    return df.index, None


def _cs_transform(X: pd.DataFrame, method: str = "zscore") -> pd.DataFrame:
    """
    Cross-sectional transform PER DATE index:
    - 'zscore' (default): (x - mean_t) / std_t  (std -> 1 if 0)
    - 'rank_gauss': rank to (0..1) then map to N(0,1) via inverse CDF
    Assumes X's index is date-like; multiple rows per date.
    """
    if method == "zscore":
        def zgrp(g):
            m = g.mean()
            s = g.std(ddof=0).replace(0, 1.0)
            return (g - m) / s
        return X.groupby(level=0).apply(zgrp)
    elif method == "rank_gauss":
        from scipy.stats import norm  # optional dependency; if missing, fall back to zscore
        def rgrp(g):
            # average ranks -> [0,1]; avoid 0/1 for stability
            r = g.rank(pct=True)
            r = r.clip(1e-3, 1 - 1e-3)
            return pd.DataFrame(norm.ppf(r), index=g.index, columns=g.columns)
        try:
            return X.groupby(level=0).apply(rgrp)
        except Exception:
            # Fallback
            return _cs_transform(X, "zscore")
    else:
        return X  # no-op


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


def _make_groups_from_dates(dates: pd.Index) -> np.ndarray:
    # XGBoost expects group sizes (number of rows per group) in order.
    # `dates` must be sorted; groups are consecutive rows per date.
    dser = pd.Series(1, index=dates)
    return dser.groupby(dser.index).size().astype(int).values


# ===========================================================
# Training helper (builds `state` used by predict_submodel)
# ===========================================================
def fit_equity_vq_state(
    df_train: pd.DataFrame,
    target_bps: Optional[pd.Series],
    *,
    features: List[str],
    objective: str = "regression",       # 'regression' | 'rank'
    cs_transform: str = "zscore",        # 'zscore' | 'rank_gauss' | 'none'
    xgb_params: Optional[Dict[str, Any]] = None,
    n_estimators: int = 800,
    early_stopping_rounds: Optional[int] = 50,
    eval_fraction: float = 0.2,
    random_state: int = 42,
    model_name: str = "equity_value_quality_xgb",
    cs_move_scale_bps: float = 40.0,     # used to map rank margins to bps at inference
) -> Dict[str, Any]:
    """
    Fit the cross-sectional Value–Quality model.

    For `objective='regression'`, `target_bps` is required (forward alpha in bps).
    For `objective='rank'`, `target_bps` is also required to form per-date preferences.

    Cross-sectional standardization is applied per DATE during training and
    will be replicated at inference.
    """
    if objective not in ("regression", "rank"):
        raise ValueError("objective must be 'regression' or 'rank'")
    if target_bps is None:
        raise ValueError("target_bps is required (forward cross-sectional alpha in bps).")

    # Ensure MultiIndex [date, asset] or at least repeating dates
    if not isinstance(df_train.index, pd.MultiIndex):
        # Promote to MultiIndex if an asset column exists; otherwise keep single level
        date_idx, asset_col = _infer_date_and_asset(df_train)
        if asset_col is not None:
            df_train = df_train.set_index(asset_col, append=True).sort_index()
            target_bps = target_bps.reindex(df_train.index)
    else:
        asset_col = None  # asset is in index level 1

    # Align & drop NA
    X = df_train[features].copy()
    y = target_bps.reindex(X.index)
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X.loc[mask], y.loc[mask].astype(float)

    # Cross-sectional transform per date
    Xcs = _cs_transform(X, cs_transform)

    # Build train/valid splits (time-ordered by DATE)
    dates = Xcs.index.get_level_values(0) if isinstance(Xcs.index, pd.MultiIndex) else Xcs.index
    order = np.argsort(dates.values)
    Xcs, y = Xcs.iloc[order], y.iloc[order]
    uniq_dates = pd.Index(dates.iloc[order])
    if eval_fraction and 0.0 < eval_fraction < 0.9:
        # split on date boundary
        split_date = uniq_dates[int(len(uniq_dates) * (1 - eval_fraction))]
        tr_idx = uniq_dates <= split_date
        va_idx = uniq_dates > split_date
        X_tr, y_tr = Xcs.loc[tr_idx], y.loc[tr_idx]
        X_va, y_va = Xcs.loc[va_idx], y.loc[va_idx]
    else:
        X_tr, y_tr, X_va, y_va = Xcs, y, None, None

    # XGB parameters
    params = dict(
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=5.0,
        reg_lambda=2.0,
        reg_alpha=0.0,
        random_state=random_state,
        tree_method="hist",
    )
    if objective == "regression":
        params.update(dict(objective="reg:squarederror", eval_metric="rmse"))
    else:
        params.update(dict(objective="rank:pairwise", eval_metric="ndcg"))

    if xgb_params:
        params.update(xgb_params)

    # Build DMatrices (+ groups for ranking)
    dtrain = xgb.DMatrix(X_tr, label=y_tr.values, nthread=-1, feature_names=list(X_tr.columns))
    evals = [(dtrain, "train")]
    if objective == "rank":
        groups_tr = _make_groups_from_dates(X_tr.index.get_level_values(0))
        dtrain.set_group(groups_tr)

    dvalid = None
    if X_va is not None:
        dvalid = xgb.DMatrix(X_va, label=y_va.values, nthread=-1, feature_names=list(X_va.columns))
        if objective == "rank":
            groups_va = _make_groups_from_dates(X_va.index.get_level_values(0))
            dvalid.set_group(groups_va)
        evals.append((dvalid, "valid"))

    # Train booster
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds if dvalid is not None else None,
        verbose_eval=False,
    )

    # Importance
    gain = booster.get_score(importance_type="gain")
    tot = sum(gain.values()) or 1.0
    fi = {f: float(gain.get(f, 0.0) / tot) for f in X_tr.columns}

    # Serialize booster
    booster_raw = booster.save_raw()

    state = {
        "model_name": model_name,
        "feature_order": list(features),
        "objective": objective,
        "booster_raw": bytes(booster_raw),
        "best_ntree_limit": int(getattr(booster, "best_ntree_limit", 0)),
        "cs_transform": cs_transform,
        "feature_importance": fi,
        "asset_col": asset_col,  # if None, assume MultiIndex for asset
        "cs_move_scale_bps": float(cs_move_scale_bps),
        # Uncertainty config
        "residual_ewm_alpha": 0.08,
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
    Equity Value–Quality (cross-sectional, XGBoost). Inference-only.

    Params (optional):
      - clip_k_bps: float (default 60.0)  -> tanh scaling for signal
      - cs_transform: override state (default from state)
      - min_abs_edge_bps: float|None      -> optional gate (handled upstream typically)
    """
    if state is None:
        raise ValueError(
            "Equity Value–Quality requires a fitted `state`. "
            "Fit with fit_equity_vq_state(...) and pass it here."
        )

    params = params or {}
    clip_k_bps = float(params.get("clip_k_bps", 60.0))
    cs_transform = params.get("cs_transform", state.get("cs_transform", "zscore"))
    min_abs_edge_bps = params.get("min_abs_edge_bps", None)

    # Slice & structure
    Xfull = df if as_of is None else df.loc[:as_of]
    trained_order = state.get("feature_order", features)
    missing = [f for f in trained_order if f not in Xfull.columns]
    if missing:
        raise KeyError(f"Missing required features for Equity VQ: {missing}")

    # Conform index for cross-sectional ops
    if not isinstance(Xfull.index, pd.MultiIndex):
        # promote if asset column exists
        asset_col = state.get("asset_col", None)
        if asset_col and asset_col in Xfull.columns:
            Xfull = Xfull.set_index(asset_col, append=True).sort_index()
        else:
            # still allow inference, but CS transforms will behave trivially
            pass

    X = Xfull[trained_order].copy()

    # Cross-sectional transform by DATE
    Xcs = _cs_transform(X, cs_transform)

    # Predict
    booster = xgb.Booster()
    booster.load_model(bytearray(state["booster_raw"]))
    ntree = int(state.get("best_ntree_limit", 0)) or None

    dmat = xgb.DMatrix(Xcs, nthread=-1, feature_names=list(Xcs.columns))
    raw_pred = booster.predict(dmat, ntree_limit=ntree)

    objective = state.get("objective", "regression").lower()
    idx = Xcs.index
    if objective == "regression":
        # Direct alpha proxy in bps
        y_pred = pd.Series(raw_pred, index=idx, name=f"equity_vq_h{horizon}_bps")
    else:
        # Ranking margins -> CS rank percentile per date -> map to bps
        pred_df = pd.Series(raw_pred, index=idx)
        # per-date percentile
        def _rank_to_bps(g):
            pct = g.rank(pct=True)
            return (2.0 * pct - 1.0) * float(state.get("cs_move_scale_bps", 40.0))
        y_pred = pred_df.groupby(level=0).apply(_rank_to_bps).rename(f"equity_vq_h{horizon}_bps")

    # Map to signal
    signal = _tanh_signal(y_pred, clip_k_bps)

    # Uncertainty proxy (target-free):
    # If we have an asset dimension, do per-asset EWMA of deviations; else fallback to per-date dispersion.
    alpha = float(state.get("residual_ewm_alpha", 0.08))
    if isinstance(y_pred.index, pd.MultiIndex) and y_pred.index.nlevels >= 2:
        pred_std = (
            y_pred.groupby(level=1)
            .apply(lambda s: (s - s.ewm(alpha=alpha, adjust=False).mean()).abs().ewm(alpha=alpha, adjust=False).mean())
            .reset_index(level=0, drop=True)
        )
    else:
        # per-date dispersion (assign to each row of that date)
        pred_std = y_pred.groupby(level=0).transform(lambda s: (s - s.mean()).abs().ewm(alpha=alpha, adjust=False).mean())
    pred_std.name = "equity_vq_pred_std"

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
        trade_mask.name = "equity_vq_trade_ok"

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
        warmup_bars=0,
        model_name=str(state.get("model_name", "equity_value_quality_xgb")),
        params={
            "horizon": horizon,
            "clip_k_bps": clip_k_bps,
            "objective": objective,
            "cs_transform": cs_transform,
            "cs_move_scale_bps": state.get("cs_move_scale_bps", 40.0),
            "min_abs_edge_bps": min_abs_edge_bps,
        },
        state=state,
    )
    return result
