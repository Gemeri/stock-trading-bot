# vol_timing_elasticnet.py
# ---------------------------------------------------------------------
# Vol-Timing (Elastic Net)
# Target modes:
#   - 'rv_log' (default): predict log(RV_{t+1}); map to edge via sign & scale
#   - 'risk_adj_return_bps': predict forward risk-adjusted return in bps directly
# Features (typical):
#   ['rv_20','std_10','vov_20','semivar_dn_20','bb_bw','dd_63','adv_ratio','volume_zscore',
#    # optional options-implieds:
#    'IV_7d','IV_30d','IV_RV','term_skew']
# Outputs: y_pred (bps), signal [-1,1], pred_std, confidence, trade_mask (gates calm vs storm)
# ---------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

import warnings
from sklearn.exceptions import ConvergenceWarning

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
def fit_vol_timing_state(
    df_train: pd.DataFrame,
    target: pd.Series,
    *,
    features: List[str],
    target_mode: str = "rv_log",        # 'rv_log' or 'risk_adj_return_bps'
    alpha: float = 0.02,                 # overall regularization strength
    l1_ratio: float = 0.2,               # Elastic Net mixing (0=ridge,1=lasso)
    scaler_mode: str = "standard",       # 'standard' | 'robust'
    max_iter: int = 4000,
    tol: float = 1e-5,
    random_state: int = 42,
    # Mapping from vol forecast to bps edge (used when target_mode='rv_log'):
    k_bps_per_logrv: float = 35.0,       # scale: bps per 1.0 log-RV change
    sign_when_high_vol: int = -1,        # -1 -> negative edge in high vol (mean-rev sleeves)
    residual_ewm_alpha: float = 0.08,    # smoothing for target-free pred_std proxy
    model_name: str = "vol_timing_elasticnet",
) -> Dict[str, Any]:
    """
    Fit Elastic Net for Vol-Timing.
      - target_mode='rv_log': `target` is realized volatility proxy (>=0). We'll log-transform internally.
      - target_mode='risk_adj_return_bps': `target` is forward risk-adjusted return in bps.

    Returns a serializable `state` with scaler stats, coefficients, and mapping config.
    """
    # Align & drop NA
    X = df_train[features].copy()
    y = target.reindex(X.index)
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X.loc[mask], y.loc[mask]

    # Build standardized design matrix
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

    # Prepare target
    mode = target_mode.lower()
    if mode == "rv_log":
        y_pos = y.astype(float).clip(lower=1e-12)
        y_tr = np.log(y_pos.values)
        rv_baseline_log = float(np.median(y_tr))  # robust center for mapping & gating
    elif mode == "risk_adj_return_bps":
        y_tr = y.astype(float).values
        rv_baseline_log = None
    else:
        raise ValueError("target_mode must be 'rv_log' or 'risk_adj_return_bps'")

    stds = Xz.std().replace([np.inf, -np.inf], np.nan)
    keep = (stds > 1e-8) & stds.notna()
    if not bool(keep.all()):
        print(f"[vol_timing] dropping near-constant cols: {list(Xz.columns[~keep])}", flush=True)
        Xz = Xz.loc[:, keep]
        features = list(Xz.columns)

    en = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        selection="random",
    )
    def _fit_with_logging(model, X_arr, y_arr, tag):
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X_arr, y_arr)
        for w in wlist:
            if issubclass(w.category, ConvergenceWarning):
                # Surface the sklearn message + useful context
                print(
                    (
                        f"[vol_timing][WARN] ElasticNet convergence ({tag}): {w.message} | "
                        f"alpha={alpha}, l1_ratio={l1_ratio}, max_iter={model.max_iter}, tol={model.tol}, "
                        f"scaler={scaler_mode}, X_shape={X_arr.shape}, y_std={float(np.nanstd(y_arr)):.3g}"
                    ),
                    flush=True,
                )
        return model

    # 1st attempt
    en = _fit_with_logging(en, Xz.values, y_tr, "try1")

    # If it still warned, try a quick, safe escalation
    if en.n_iter_ is not None and (np.asarray(en.n_iter_) >= en.max_iter - 1).any():
        # Retry with more iterations and a touch more regularization (often resolves it)
        en.set_params(max_iter=int(max_iter * 5), alpha=float(alpha * 1.5), tol=max(tol, 1e-4))
        en = _fit_with_logging(en, Xz.values, y_tr, "try2")

    # Serialize coefficients (optionally clip? Not needed usually for EN)
    coef = en.coef_.astype(float).ravel()
    intercept = float(en.intercept_)

    # Importance (normalized |coef| on standardized space)
    coef_abs = np.abs(coef)
    total = coef_abs.sum() or 1.0
    fi = {f: float(a / total) for f, a in zip(features, coef_abs)}

    state: Dict[str, Any] = {
        "model_name": model_name,
        "feature_order": list(features),
        "scaler_mode": scaler_mode,
        "target_mode": mode,
        "alpha": float(alpha),
        "l1_ratio": float(l1_ratio),
        "coef_": coef.tolist(),
        "intercept_": intercept,
        "feature_importance": fi,
        "residual_ewm_alpha": float(residual_ewm_alpha),
        # mapping config (rv_log only)
        "k_bps_per_logrv": float(k_bps_per_logrv),
        "sign_when_high_vol": int(np.sign(sign_when_high_vol) or -1),
        "rv_baseline_log": rv_baseline_log,
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
    Vol-Timing (Elastic Net). Inference-only.

    Params (optional):
      - clip_k_bps: float (default 50.0)      -> tanh scaling for signal
      - gate_mode: 'calm'|'storm' (default 'calm')
          'calm': trade when predicted vol z <= gate_z
          'storm': trade when predicted vol z >= gate_z (e.g., for hedges)
      - gate_z: float (default 2.0)           -> robust z threshold on predicted log-RV
      - vol_z_window: int (default 200)       -> window for robust z
      - min_abs_edge_bps: float|None          -> optional edge threshold for gating
    """
    if state is None:
        raise ValueError(
            "Vol-Timing requires a fitted `state`. "
            "Fit with fit_vol_timing_state(...) and pass it here."
        )

    params = params or {}
    clip_k_bps = float(params.get("clip_k_bps", 50.0))
    gate_mode = str(params.get("gate_mode", "calm")).lower()
    gate_z = float(params.get("gate_z", 2.0))
    vol_z_window = int(params.get("vol_z_window", 200))
    min_abs_edge_bps = params.get("min_abs_edge_bps", None)

    # Slice & feature checks
    Xfull = df if as_of is None else df.loc[:as_of]
    trained_order = state.get("feature_order", features)
    missing = [f for f in trained_order if f not in Xfull.columns]
    if missing:
        raise KeyError(f"Missing required features for Vol-Timing: {missing}")
    X = Xfull[trained_order].copy()

    # Standardize
    Z = _standardize(X, state)

    # Linear prediction
    coef = np.array(state["coef_"], dtype=float).ravel()
    intercept = float(state.get("intercept_", 0.0))
    lin_pred = pd.Series(Z.values @ coef + intercept, index=Z.index)

    # Map to edge in bps
    mode = state.get("target_mode", "rv_log")
    if mode == "rv_log":
        # lin_pred = predicted log(RV)
        base = float(state.get("rv_baseline_log", float(np.nanmedian(lin_pred.values))))
        k = float(state.get("k_bps_per_logrv", 35.0))
        sgn = int(state.get("sign_when_high_vol", -1))  # -1 => negative edge when vol > baseline
        diff = (lin_pred - base)
        y_pred = pd.Series(sgn * k * diff.values, index=lin_pred.index, name=f"vol_timing_h{horizon}_bps")

        # Predicted vol robust z for gating
        vol_z = _robust_z(lin_pred, window=vol_z_window)
    else:
        # risk-adjusted return in bps directly
        y_pred = pd.Series(lin_pred.values, index=lin_pred.index, name=f"vol_timing_h{horizon}_bps")
        # Use y_pred magnitude as a proxy for activity; gating can still use z of y_pred
        vol_z = _robust_z(y_pred, window=vol_z_window)

    # Signal
    signal = pd.Series(np.tanh(y_pred.values / clip_k_bps), index=y_pred.index, name="vol_timing_signal")

    # Uncertainty proxy (target-free): EWMA of prediction deviations
    alpha = float(state.get("residual_ewm_alpha", 0.08))
    pred_std = (y_pred - y_pred.ewm(alpha=alpha, adjust=False).mean()).abs().ewm(alpha=alpha, adjust=False).mean()
    pred_std.name = "vol_timing_pred_std"

    # Confidence
    confidence = _confidence_from_std(pred_std)

    # Trade mask (vol gating)
    if gate_mode == "storm":
        trade_mask = (vol_z >= gate_z)
    else:  # 'calm'
        trade_mask = (vol_z <= gate_z)
    trade_mask = trade_mask.reindex(y_pred.index).fillna(False)
    trade_mask.name = "vol_timing_trade_ok"

    # Optional edge gating
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

    # Feature importance (from training)
    fi_dict = state.get("feature_importance", {})
    fi = pd.Series({f: float(fi_dict.get(f, 0.0)) for f in trained_order}, name="coef_abs_norm")

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
        warmup_bars=max(50, vol_z_window),  # allow robust z warmup
        model_name=str(state.get("model_name", "vol_timing_elasticnet")),
        params={
            "horizon": horizon,
            "clip_k_bps": clip_k_bps,
            "gate_mode": gate_mode,
            "gate_z": gate_z,
            "vol_z_window": vol_z_window,
            "min_abs_edge_bps": min_abs_edge_bps,
            "target_mode": mode,
        },
        state=state,
    )
    return result
