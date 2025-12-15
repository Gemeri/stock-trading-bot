# macro_regime_gbt_hmm.py
# ---------------------------------------------------------------------
# Macro Regime Classifier
# Model: Gradient-Boosted Trees (XGBoost, binary) -> instantaneous p(risk-on | x_t)
#         + 2-state HMM smoothing (risk-on / risk-off) for regime persistence.
#
# Labels: optional. If provided (risk_on labels 0/1), we train supervised.
#         If not provided, we build an unsupervised proxy score (PC1 of features)
#         and map it through a sigmoid to get p_inst.
#
# Features (external, typical):
#   ['term_spread','dUST10y','vix_level','vix_term','dxy_mom','commods_breadth','rv_median_xasset', ...]
# If you lack externals, you can proxy breadth from your traded set (rv_20, dd_63, percB breadth, etc.).
#
# Outputs:
#   - proba_up: smoothed P(risk-on) from HMM
#   - y_pred: tiny expected-return context (bps) via regime means (optional; default small)
#   - signal: often 0 (context model). Can emit tanh(y_pred/k) if desired.
#   - pred_std: sqrt(p*(1-p)) as uncertainty proxy
#   - confidence: mapped from pred_std
#   - state: contains booster bytes or unsupervised mapping, HMM params, regime means
# ---------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable
import numpy as np
import pandas as pd
import xgboost as xgb

# Optional Platt calibration
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


# =========================
# Small numeric utilities
# =========================
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

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

def _stationary_prior(p_oo: float, p_ii: float) -> np.ndarray:
    """
    Stationary distribution for 2-state Markov chain with P(off->off)=p_oo, P(on->on)=p_ii.
    States order: [off, on]
    """
    # Off->On = 1-p_oo ; On->Off = 1-p_ii
    a = 1.0 - p_oo
    b = 1.0 - p_ii
    pi_on = a / (a + b + 1e-12)
    return np.array([1.0 - pi_on, pi_on], dtype=float)

def _forward_backward(p_like_on: np.ndarray, p_oo: float, p_ii: float, prior: Optional[np.ndarray] = None) -> np.ndarray:
    """
    HMM smoothing with pseudo-likelihoods:
      e_t(state=on)  ∝ p_like_on[t]
      e_t(state=off) ∝ 1 - p_like_on[t]
    Transition matrix T:
      [ [p_oo, 1-p_oo],
        [1-p_ii, p_ii] ]
    Returns smoothed posterior P(state=on | 1:T) for each t (gamma_on).
    """
    T = np.array([[p_oo, 1.0 - p_oo],
                  [1.0 - p_ii, p_ii]], dtype=float)

    n = len(p_like_on)
    eps = 1e-12

    e_on = np.clip(p_like_on, eps, 1.0 - eps)
    e_off = 1.0 - e_on

    # Forward
    if prior is None:
        prior = _stationary_prior(p_oo, p_ii)
    alpha = np.zeros((n, 2), dtype=float)
    # t=0
    emit0 = np.array([e_off[0], e_on[0]])
    alpha[0] = prior * emit0
    alpha[0] /= alpha[0].sum() + eps
    # t>0
    for t in range(1, n):
        emit = np.array([e_off[t], e_on[t]])
        alpha[t] = (alpha[t-1] @ T) * emit
        s = alpha[t].sum()
        if s <= 0:
            alpha[t] = np.array([0.5, 0.5])
        else:
            alpha[t] /= s

    # Backward
    beta = np.zeros((n, 2), dtype=float)
    beta[-1] = 1.0
    for t in range(n - 2, -1, -1):
        emit_next = np.array([e_off[t+1], e_on[t+1]])
        beta[t] = T @ (emit_next * beta[t+1])
        s = beta[t].sum()
        if s <= 0:
            beta[t] = np.array([0.5, 0.5])
        else:
            beta[t] /= s

    # Posterior
    gamma = alpha * beta
    gamma /= (gamma.sum(axis=1, keepdims=True) + eps)
    # Return P(on)
    return gamma[:, 1]


# ===========================================================
# Training helper (builds `state` used by predict_submodel)
# ===========================================================
def fit_macro_regime_state(
    df_train: pd.DataFrame,
    y_risk_on: Optional[pd.Series],    # 0/1 labels for risk-on (optional). If None -> unsupervised proxy.
    *,
    features: List[str],
    xgb_params: Optional[Dict[str, Any]] = None,
    n_estimators: int = 600,
    eval_fraction: float = 0.2,
    early_stopping_rounds: int = 50,
    random_state: int = 42,
    model_name: str = "macro_regime_gbt_hmm",
    # HMM persistence priors (daily-like defaults; tune for your bar size)
    p_stay_off: float = 0.97,
    p_stay_on: float = 0.97,
    # Optional Platt calibration on GBT margins
    calibrate_platt: bool = True,
    # Optional tiny regime means to produce y_pred context (bps)
    returns_bps: Optional[pd.Series] = None,  # forward returns in bps (aligned to df_train.index). Only used to estimate regime averages.
) -> Dict[str, Any]:
    """
    Fit GBT for instantaneous risk-on probability (if labels provided),
    then configure a 2-state HMM for persistence. Returns serializable `state`.

    If `y_risk_on` is None, we build an unsupervised proxy p_inst from the
    first principal component across features (per series) mapped through a sigmoid.
    """
    X = df_train[features].copy()
    # Clean NaNs / infs
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop columns that are completely missing
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)

    # If labels provided, align them; same for returns_bps
    if y_risk_on is not None:
        y = y_risk_on.reindex(X.index)
    else:
        y = None
    if returns_bps is not None:
        r = returns_bps.reindex(X.index)
    else:
        r = None

    # Leakage-safe fill: past-only, then median within training window
    X = X.ffill()
    X = X.fillna(X.median(numeric_only=True))

    # Build mask only for provided series
    mask = pd.Series(True, index=X.index)
    if y is not None:
        mask &= y.notna()
        y = y.loc[mask].astype(int)
    if r is not None:
        mask &= r.notna()
        r = r.loc[mask].astype(float)

    X = X.loc[mask]

    # If still empty or no features left → return neutral state
    if X.shape[0] == 0 or X.shape[1] == 0:
        return {
            "model_name": model_name,
            "feature_order": [],             # train with no features → neutral
            "booster_raw": None,
            "best_ntree_limit": 0,
            "cal": None,
            "feature_importance": {},
            "unsup": {"mu": {}, "sd": {}, "w": [], "scale": 1.0},
            "p_stay_off": float(p_stay_off),
            "p_stay_on": float(p_stay_on),
            "mu_on_bps": 0.0,
            "mu_off_bps": 0.0,
        }

    y = y_risk_on.reindex(X.index).astype(int) if y_risk_on is not None else None
    r = returns_bps.reindex(X.index).astype(float) if returns_bps is not None else None

    booster_raw = None
    best_ntree_limit = 0
    fi = {f: 0.0 for f in features}
    cal = None
    unsup = None

    if y is not None:
        # ---- Supervised GBT (binary: risk-on) ----
        params = dict(
            objective="binary:logistic",
            eval_metric="logloss",
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=2.0,
            reg_lambda=1.5,
            reg_alpha=0.0,
            random_state=random_state,
            tree_method="hist",
        )
        if xgb_params:
            params.update(xgb_params)

        # Handle class imbalance
        if "scale_pos_weight" not in params:
            pos = float(y.sum())
            neg = float(len(y) - y.sum())
            params["scale_pos_weight"] = (neg / max(pos, 1.0)) if pos > 0 else 1.0

        # Time-ordered split
        if eval_fraction and 0.0 < eval_fraction < 0.9:
            split_idx = int(len(X) * (1 - eval_fraction))
            X_tr, y_tr = X.iloc[:split_idx], y.iloc[:split_idx]
            X_va, y_va = X.iloc[split_idx:], y.iloc[split_idx:]
            dtr = xgb.DMatrix(X_tr, label=y_tr.values, nthread=-1, feature_names=list(X.columns))
            dva = xgb.DMatrix(X_va, label=y_va.values, nthread=-1, feature_names=list(X.columns))
            evals = [(dtr, "train"), (dva, "valid")]
        else:
            dtr = xgb.DMatrix(X, label=y.values, nthread=-1, feature_names=list(X.columns))
            dva, evals = None, [(dtr, "train")]

        booster = xgb.train(
            params, dtr, num_boost_round=n_estimators,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds if dva is not None else None,
            verbose_eval=False,
        )
        booster_raw = bytes(booster.save_raw())
        best_ntree_limit = int(getattr(booster, "best_ntree_limit", 0)) or 0

        # Optional Platt over margins
        if calibrate_platt and _HAS_SKLEARN:
            if dva is not None:
                margins = booster.predict(dva, output_margin=True)
                y_va = y.iloc[-len(margins):]
            else:
                margins = booster.predict(dtr, output_margin=True)
                y_va = y
            lr = LogisticRegression(max_iter=200, solver="lbfgs")
            lr.fit(margins.reshape(-1, 1), y_va.values)
            cal = {"coef": float(lr.coef_.ravel()[0]), "intercept": float(lr.intercept_.ravel()[0])}

        # Gain-based importance
        gain = booster.get_score(importance_type="gain")
        tot = sum(gain.values()) or 1.0
        fi = {f: float(gain.get(f, 0.0) / tot) for f in X.columns}

    else:
        # ---- Unsupervised proxy (PC1 -> sigmoid) ----
        if X.shape[0] == 0 or X.shape[1] == 0:
            unsup = {"mu": {}, "sd": {}, "w": [], "scale": 1.0, "cols": []}
            fi = {}
        else:
            mu = X.mean()
            sd = X.std(ddof=0).replace(0, 1.0)
            Z  = (X - mu) / sd
            if Z.shape[0] == 0 or Z.shape[1] == 0:
                unsup = {"mu": {}, "sd": {}, "w": [], "scale": 1.0, "cols": []}
                fi = {}
            else:
                U, S, Vt = np.linalg.svd(Z.values, full_matrices=False)
                w = Vt[0, :]

                # Direction heuristic (flip if VIX loads positive)
                sign = 1.0
                for name in X.columns:
                    if "vix" in name.lower() or "vol" in name.lower():
                        j = list(X.columns).index(name)
                        if w[j] > 0:
                            sign = -1.0
                        break
                w = sign * w

                z = Z.values @ w
                z_std = np.std(z) + 1e-9
                p_inst = _sigmoid(z / z_std)

                unsup = {
                    "mu": mu.astype(float).to_dict(),
                    "sd": sd.astype(float).to_dict(),
                    "w": w.astype(float).tolist(),
                    "scale": float(1.0 / z_std),
                    "cols": list(X.columns),     # <<< save training column order
                }
                fi = {f: float(abs(wi) / np.sum(np.abs(w))) for f, wi in zip(X.columns, w)}


    # HMM persistence params
    p_oo = float(p_stay_off)
    p_ii = float(p_stay_on)

    # If we can estimate regime means, do it (optional context y_pred)
    mu_on_bps = 0.0
    mu_off_bps = 0.0
    if returns_bps is not None:
        # Build instantaneous p_inst on training
        if y is not None:
            dmat = xgb.DMatrix(X, nthread=-1, feature_names=list(X.columns))
            if cal is None:
                p_inst = booster.predict(dmat, ntree_limit=best_ntree_limit or None)
            else:
                margins = booster.predict(dmat, output_margin=True, ntree_limit=best_ntree_limit or None)
                p_inst = _sigmoid(cal["coef"] * margins + cal["intercept"])
        else:
            # unsupervised p_inst already computed above (aligned to X.index)
            p_inst = p_inst  # noqa

        # Smooth with HMM on training
        gamma_on = _forward_backward(np.asarray(p_inst), p_oo, p_ii, prior=None)
        w_on = gamma_on
        w_off = 1.0 - gamma_on
        # Weighted regime means
        mu_on_bps = float(np.sum(w_on * r.values) / (np.sum(w_on) + 1e-9))
        mu_off_bps = float(np.sum(w_off * r.values) / (np.sum(w_off) + 1e-9))

    state = {
        "model_name": model_name,
        "feature_order": list(X.columns),
        # GBT bits (if supervised)
        "booster_raw": booster_raw,
        "best_ntree_limit": best_ntree_limit,
        "cal": cal,
        "feature_importance": fi,
        # Unsupervised mapping if no labels
        "unsup": unsup,  # {mu, sd, w, scale}
        # HMM persistence
        "p_stay_off": p_oo,
        "p_stay_on": p_ii,
        # Regime small expected returns (bps) for context
        "mu_on_bps": float(mu_on_bps),
        "mu_off_bps": float(mu_off_bps),
    }
    return state


# ---------------------------------------------
# Internal: get instantaneous p_on(x_t)
# ---------------------------------------------
def _inst_proba(X: pd.DataFrame, state: Dict[str, Any]) -> np.ndarray:
    # Prefer supervised booster if present
    if state.get("booster_raw", None):
        booster = xgb.Booster()
        booster.load_model(bytearray(state["booster_raw"]))
        dmat = xgb.DMatrix(X, nthread=-1, feature_names=list(X.columns))
        if state.get("cal", None) is None:
            p = booster.predict(dmat, ntree_limit=(state.get("best_ntree_limit") or None))
            return np.clip(p, 1e-6, 1 - 1e-6)
        # Platt-calibrated
        margins = booster.predict(dmat, output_margin=True, ntree_limit=(state.get("best_ntree_limit") or None))
        a = float(state["cal"]["coef"])
        b = float(state["cal"]["intercept"])
        p = _sigmoid(a * margins + b)
        return np.clip(p, 1e-6, 1 - 1e-6)
    # Unsupervised proxy
    unsup = state.get("unsup", None)
    if unsup is None:
        return np.full(len(X), 0.5, dtype=float)

    cols = list(unsup.get("cols", [])) or list(pd.Series(unsup.get("mu", {})).index)
    w = np.array(unsup.get("w", []), dtype=float)
    if len(cols) == 0 or len(w) == 0:
        return np.full(len(X), 0.5, dtype=float)

    # Reindex to training columns (order matters)
    Xsub = X.reindex(columns=cols)
    mu = pd.Series(unsup["mu"]).reindex(cols)
    sd = pd.Series(unsup["sd"]).reindex(cols).replace(0, 1.0)

    # Standardize; missing values → 0 after z-scoring (i.e., equal to training mean)
    Z = ((Xsub - mu) / sd).fillna(0.0)

    # Dimension check; if mismatch, return neutral
    if Z.shape[1] != len(w):
        return np.full(len(X), 0.5, dtype=float)

    scale = float(unsup.get("scale", 1.0))
    z = Z.values @ w
    return np.clip(_sigmoid(z * scale), 1e-6, 1 - 1e-6)



# ===========================================================
# Required submodel API (inference only; HMM smoothing)
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
    Macro Regime Classifier (GBT -> HMM). Inference-only.

    Params (optional):
      - clip_k_bps: float (default 40.0) -> tanh scaling for signal (if emitted)
      - emit_signal: bool (default False) -> if True, produce signal from y_pred; else zeros (context model)
      - p_stay_off / p_stay_on: float -> override HMM persistence
      - prior: 'stationary' | [p_off, p_on] -> HMM prior at start of slice
      - min_abs_edge_bps: float|None -> optional gating (rarely needed here)
    """
    if state is None:
        raise ValueError(
            "Macro Regime Classifier requires a fitted `state`. "
            "Fit with fit_macro_regime_state(...) and pass it here."
        )

    params = params or {}
    clip_k_bps = float(params.get("clip_k_bps", 40.0))
    emit_signal = bool(params.get("emit_signal", False))
    p_oo = float(params.get("p_stay_off", state.get("p_stay_off", 0.97)))
    p_ii = float(params.get("p_stay_on", state.get("p_stay_on", 0.97)))
    prior_param = params.get("prior", "stationary")
    min_abs_edge_bps = params.get("min_abs_edge_bps", None)

    # Slice & feature checks
    Xfull = df if as_of is None else df.loc[:as_of]
    trained_order = state.get("feature_order", features)
    missing = [f for f in trained_order if f not in Xfull.columns]
    if missing:
        raise KeyError(f"Missing required features for Macro Regime: {missing}")
    X = Xfull[trained_order].copy()

    # 1) Instantaneous probabilities from GBT (or unsup proxy)
    p_inst = _inst_proba(X, state)

    # 2) HMM smoothing for persistence
    if isinstance(prior_param, (list, tuple, np.ndarray)) and len(prior_param) == 2:
        prior = np.array(prior_param, dtype=float)
        prior = prior / (prior.sum() + 1e-12)
    else:
        prior = _stationary_prior(p_oo, p_ii)

    p_smoothed = _forward_backward(p_inst, p_oo, p_ii, prior=prior)
    proba_up = pd.Series(p_smoothed, index=X.index, name="macro_regime_proba_on")

    # 3) Tiny expected return context (bps), optional
    mu_on = float(state.get("mu_on_bps", 0.0))
    mu_off = float(state.get("mu_off_bps", 0.0))
    y_pred = pd.Series(mu_on * proba_up.values + mu_off * (1.0 - proba_up.values),
                       index=X.index, name=f"macro_regime_h{horizon}_bps")

    # 4) Signal (often zero for context-only)
    if emit_signal:
        signal = _tanh_signal(y_pred, clip_k_bps)
    else:
        signal = pd.Series(0.0, index=X.index, name="macro_regime_signal")

    # 5) Uncertainty & confidence
    pred_std = pd.Series(np.sqrt(proba_up * (1.0 - proba_up)), index=X.index, name="macro_regime_pred_std")
    confidence = _confidence_from_std(pred_std)

    # Optional gating
    trade_mask = None
    if min_abs_edge_bps is not None:
        trade_mask = (y_pred.abs() >= float(min_abs_edge_bps))
        trade_mask.name = "macro_regime_trade_ok"

    # Feature importance (from booster or unsup w)
    fi_dict = state.get("feature_importance", {})
    fi = pd.Series({f: float(fi_dict.get(f, 0.0)) for f in trained_order}, name="importance")

    result = SubmodelResult(
        y_pred=y_pred,
        signal=signal,
        proba_up=proba_up,
        pred_std=pred_std,
        confidence=confidence,
        costs=None,
        trade_mask=trade_mask,
        live_metrics=None,
        feature_importance=fi,
        used_features=trained_order,
        warmup_bars=0,
        model_name=str(state.get("model_name", "macro_regime_gbt_hmm")),
        params={
            "horizon": horizon,
            "clip_k_bps": clip_k_bps,
            "emit_signal": emit_signal,
            "p_stay_off": p_oo,
            "p_stay_on": p_ii,
            "prior": prior.tolist() if isinstance(prior, np.ndarray) else prior,
        },
        state=state,
    )
    return result
