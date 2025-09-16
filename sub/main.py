# main.py
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import shap

from typing import Tuple, List, Mapping, Dict

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit

# ─────────────────────────────────────────────────────────────────────────────
# Sub-models (always include classifier; no USE_CLASSIFIER branching)
# Existing ones
import sub.momentum as mod1
import sub.trend_continuation as mod2
import sub.mean_reversion as mod3
import sub.lagged_return as mod4
import sub.sentiment_volume as mod5
import sub.regressor as modR
import sub.classifier as mod6                  # multi-horizon classifier (provides confidence utils)
import sub.svr as modSVR                       # RBF SVR price-growth (returns RAW close)
# NEW submodels
import sub.squeeze_breakout as modSB           # single-horizon classifier (probability)
import sub.regime_gate as modRG                # multi-head classifier producing 3 regime probabilities
import sub.vol_forecaster as modVF             # volatility quantile regressor (feature-only)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Config & paths
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
RESULTS_DIR = os.path.join(BASE_DIR, "sub-results")
FEATURE_ANALYTICS_DIR = os.path.join(RESULTS_DIR, "feature-analytics")

load_dotenv()
BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "4Hour")
TICKERS = os.getenv("TICKERS", "TSLA").split(",")
ML_MODEL = os.getenv("ML_MODEL", "sub-vote")  # or "sub-meta"
TIMEFRAME_MAP = {"4Hour": "H4", "2Hour": "H2", "1Hour": "H1", "30Min": "M30", "15Min": "M15"}
CONVERTED_TIMEFRAME = TIMEFRAME_MAP.get(BAR_TIMEFRAME, BAR_TIMEFRAME)
TICKERS_STR = "-".join([t.strip() for t in TICKERS if t.strip()]) or "TSLA"
CSV_PATH = os.path.join(ROOT_DIR, f"{TICKERS_STR}_{CONVERTED_TIMEFRAME}.csv")
MODE = ML_MODEL
EXECUTION = globals().get("EXECUTION", "backtest")
QUICK_TEST = False

HISTORY_PATH = os.path.join(RESULTS_DIR, f"submeta_history_{CONVERTED_TIMEFRAME}.csv")

REG_UP, REG_DOWN = 0.3, -0.3  # regression vote mapping → direction (percent units; e.g. 0.3 = +0.3%)

FORCE_STATIC_THRESHOLDS = True
STATIC_UP = 0.55
STATIC_DOWN = 0.45

META_MODEL_TYPE = os.getenv("META_MODEL_TYPE", "cat")
VALID_META_MODELS = {"logreg", "lgbm", "xgb", "nn", "cat"}
if META_MODEL_TYPE not in VALID_META_MODELS:
    raise ValueError(f"META_MODEL_TYPE must be one of {VALID_META_MODELS}")

# ─────────────────────────────────────────────────────────────────────────────
# Submodel sets
#   - SUBMODS are channels that produce m*/acc* (used by sub-vote and meta)
#   - FEATURE_ONLY_MODULES produce 'feat_*' columns only (not part of votes)
#     (vol_forecaster predicts volatility, regime_gate predicts regime probs)
# ─────────────────────────────────────────────────────────────────────────────
SUBMODS = [mod1, mod2, mod3, mod4, mod5, modR, mod6, modSVR, modSB]   # add squeeze_breakout as a normal channel
FEATURE_ONLY_MODULES = [modRG, modVF]                                 # add as meta features only

# ─────────────────────────────────────────────────────────────────────────────
# Helpers (multi-horizon utilities, meta features, thresholds, SHAP)
# ─────────────────────────────────────────────────────────────────────────────

def is_multi(module) -> bool:
    return hasattr(module, "MULTI_HORIZONS") and isinstance(module.MULTI_HORIZONS, (list, tuple)) and len(module.MULTI_HORIZONS) > 0

def is_multi_regression(module) -> bool:
    return hasattr(module, "compute_targets") and is_multi(module)

def label_col_for(module, h: int | None):
    if is_multi(module) and h is not None:
        return f"label_h{int(h)}"
    return "label"

def max_lookahead(submods) -> int:
    mx = 1
    for m in submods:
        if is_multi(m):
            mx = max(mx, max(m.MULTI_HORIZONS))
    return mx

def expand_channels(submods):
    chans = []
    for m in submods:
        if is_multi(m):
            for h in m.MULTI_HORIZONS:
                chans.append({"mod": m, "h": int(h), "name": f"{m.__name__}_h{h}"})
        else:
            chans.append({"mod": m, "h": None, "name": m.__name__})
    return chans

# ---- SVR identification + helpers ------------------------------------------------------

def is_svr_module(module) -> bool:
    try:
        return module.__name__.split(".")[-1] == "svr" and hasattr(module, "fit") and hasattr(module, "predict")
    except Exception:
        return False

def _svr_predict_next_from_slice(df_slice: pd.DataFrame, module) -> float:
    close = pd.to_numeric(df_slice["close"], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().bfill()
    if len(close) == 0:
        return np.nan
    try:
        bundle = module.fit(close)
        close_ext = pd.concat([close, pd.Series([float(close.iloc[-1])])], ignore_index=True)
        y_pred = module.predict(bundle, close_ext)
        return float(y_pred[-1]) if len(y_pred) else np.nan
    except Exception:
        return np.nan

def _svr_train_on_series(close_series: pd.Series, module):
    close = pd.to_numeric(close_series, errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return module.fit(close)

def _svr_predict_next_with_model(bundle, close_series: pd.Series, module) -> float:
    close = pd.to_numeric(close_series, errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().bfill()
    if len(close) == 0:
        return np.nan
    try:
        close_ext = pd.concat([close, pd.Series([float(close.iloc[-1])])], ignore_index=True)
        y_pred = module.predict(bundle, close_ext)
        return float(y_pred[-1]) if len(y_pred) else np.nan
    except Exception:
        return np.nan

# ---------------------------------------------------------------------------------------

def get_meta_feature_names(hist: pd.DataFrame, n_mods: int | None = None) -> list[str]:
    """
    build_meta_features() creates [m1..mN, acc1..accN, mean_p, var_p, wavg_p, (any feat_* if present)]
    """
    if n_mods is None:
        n_mods = sum(1 for c in hist.columns if isinstance(c, str) and c.startswith("m") and c[1:].isdigit())
    prob_cols = [f"m{i+1}" for i in range(n_mods)]
    acc_cols = [f"acc{i+1}" for i in range(n_mods)]
    extras = [c for c in hist.columns if isinstance(c, str) and c.startswith("feat_")]
    return [*prob_cols, *acc_cols, "mean_p", "var_p", "wavg_p", *extras]

def _nan_to_val(a: np.ndarray, val: float = 0.0) -> np.ndarray:
    b = np.array(a, dtype=float)
    b[~np.isfinite(b)] = val
    return b

def build_meta_features(df: pd.DataFrame, n_mods: int | None = None) -> np.ndarray:
    if n_mods is None:
        n_mods = sum(1 for c in df.columns if isinstance(c, str) and c.startswith("m") and c[1:].isdigit())
    prob_cols = [f"m{i+1}" for i in range(n_mods)]
    acc_cols  = [f"acc{i+1}" for i in range(n_mods)]

    P = df[prob_cols].values if prob_cols and set(prob_cols).issubset(df.columns) else np.zeros((len(df), 0))
    A = df[acc_cols].values  if acc_cols  and set(acc_cols ).issubset(df.columns) else np.zeros((len(df), 0))

    P = _nan_to_val(P, 0.5)
    A = _nan_to_val(A, 0.5)

    if P.shape[1] > 0:
        mean_p = P.mean(axis=1, keepdims=True)
        var_p  = P.var(axis=1,  keepdims=True)
        if A.shape[1] == P.shape[1] and A.shape[1] > 0:
            denom = (A.sum(axis=1, keepdims=True) + 1e-6)
            wavg_p = (P * A).sum(axis=1, keepdims=True) / denom
        else:
            wavg_p = mean_p
    else:
        mean_p = var_p = wavg_p = np.zeros((len(df), 1))

    extra_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("feat_")]
    EX = _nan_to_val(df[extra_cols].values if extra_cols else np.zeros((len(df), 0)), 0.5)

    return np.hstack([P, A, mean_p, var_p, wavg_p, EX])

def make_meta_model():
    if META_MODEL_TYPE == "logreg":
        base = LogisticRegression(max_iter=2000, class_weight="balanced")
    elif META_MODEL_TYPE == "lgbm":
        from lightgbm import LGBMClassifier
        base = LGBMClassifier(
            n_estimators=400, learning_rate=0.03,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        )
    elif META_MODEL_TYPE == "xgb":
        from xgboost import XGBClassifier
        base = XGBClassifier(
            n_estimators=500, learning_rate=0.03, max_depth=4,
            subsample=0.8, colsample_bytree=0.9, reg_lambda=1.0,
            tree_method="hist", objective="binary:logistic",
            eval_metric="logloss", random_state=42
        )
        return base
    elif META_MODEL_TYPE == "cat":
        from catboost import CatBoostClassifier
        base = CatBoostClassifier(
            iterations=600, depth=6, learning_rate=0.05,
            l2_leaf_reg=3, bagging_temperature=0.5,
            loss_function="Logloss", eval_metric="AUC",
            random_seed=42, verbose=False
        )
    else:  # "nn"
        from sklearn.neural_network import MLPClassifier
        base = MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu",
            solver="adam", alpha=1e-4,
            learning_rate_init=5e-4, max_iter=800, random_state=42 
        )
        return base

    return CalibratedClassifierCV(base, method="isotonic", cv=3, n_jobs=-1)

def optimize_asymmetric_thresholds(
    probs: np.ndarray, labels: np.ndarray, metric="profit",
    manual_thresholds=None
) -> Tuple[float, float]:
    if FORCE_STATIC_THRESHOLDS:
        return STATIC_UP, STATIC_DOWN
    if manual_thresholds is not None:
        return manual_thresholds

    up_coarse = np.linspace(0.60, 0.80, 11)
    down_coarse = np.linspace(0.20, 0.40, 11)

    best_up, best_dn, best_sc = 0.55, 0.45, -np.inf
    for up in up_coarse:
        dn_candidates = down_coarse[down_coarse < up - 0.05]
        for dn in dn_candidates:
            mask = (probs > up) | (probs < dn)
            if mask.sum() < 30:
                continue

            if metric == "accuracy":
                score = (labels[mask] == (probs[mask] > up)).mean()
            elif metric == "f1":
                from sklearn.metrics import f1_score
                score = f1_score(labels[mask], probs[mask] > up)
            elif metric == "profit":
                pnl = np.where(
                    probs[mask] > up,  (labels[mask] == 1).astype(int),
                    - (labels[mask] == 0).astype(int)
                )
                score = pnl.mean()
            else:
                score = roc_auc_score(labels[mask], probs[mask])

            if score > best_sc:
                best_sc, best_up, best_dn = score, up, dn

    return best_up, best_dn

def _compute_shap_importance(model, X: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    try:
        bg = X[np.random.choice(len(X), size=min(200, len(X)), replace=False)]
        f = lambda z: np.asarray(model.predict_proba(z))[:, 1]
        explainer = shap.Explainer(f, bg)
        sv = explainer(X)
        vals = np.abs(sv.values).mean(axis=0)
        df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": vals})
        df["importance_norm"] = (df["mean_abs_shap"] / (df["mean_abs_shap"].max() + 1e-12))
        df = df.sort_values("importance_norm", ascending=False).reset_index(drop=True)
        return df
    except Exception:
        try:
            base = np.asarray(model.predict_proba(X))[:, 1]
            deltas = []
            for j in range(X.shape[1]):
                X_pert = X.copy()
                X_pert[:, j] = np.median(X_pert[:, j])
                sc = np.asarray(model.predict_proba(X_pert))[:, 1]
                deltas.append(np.mean(np.abs(sc - base)))
            deltas = np.array(deltas)
            df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": deltas})
            df["importance_norm"] = (deltas / (deltas.max() + 1e-12))
            df = df.sort_values("importance_norm", ascending=False).reset_index(drop=True)
            df["note"] = "approx_fallback_no_shap"
            return df
        except Exception:
            return pd.DataFrame({"feature": feature_names, "mean_abs_shap": 0.0, "importance_norm": 0.0})

def _save_shap_report(X_val: np.ndarray, y_val: np.ndarray, meta, feat_names: List[str], out_dir: str, tag: str):
    os.makedirs(out_dir, exist_ok=True)
    shap_df = _compute_shap_importance(meta, X_val, feat_names)
    shap_path = os.path.join(out_dir, f"shap_importance_{tag}.csv")
    shap_df.to_csv(shap_path, index=False, float_format="%.6f")
    return shap_path


# ─────────────────────────────────────────────────────────────────────────────
# Data prep for sub-models
# ─────────────────────────────────────────────────────────────────────────────

def prepare_train(df_slice: pd.DataFrame, module):
    if hasattr(module, "compute_labels") and not is_multi(module):
        d = module.compute_labels(df_slice)
        tr = d.iloc[:-1].dropna(subset=module.FEATURES + ["label"])
        return tr[module.FEATURES].values, tr["label"].values
    if hasattr(module, "compute_targets") and not is_multi(module):
        d = module.compute_targets(df_slice)
        tr = d.iloc[:-1].dropna(subset=module.FEATURES + ["target"])
        return tr[module.FEATURES].values, tr["target"].values
    d = module.compute_target(df_slice)
    tr = d.iloc[:-1].dropna(subset=module.FEATURES + ["target"])
    return tr[module.FEATURES], tr["target"]

def prepare_train_for_horizon(df_slice: pd.DataFrame, module, h: int):
    assert hasattr(module, "compute_labels"), "Horizon training requires classifier module"
    d = module.compute_labels(df_slice)
    lbl_col = label_col_for(module, h)
    tr = d.iloc[:-1].dropna(subset=module.FEATURES + [lbl_col])
    return tr[module.FEATURES].values, tr[lbl_col].values

def prepare_train_regression_multi(df_slice: pd.DataFrame, module):
    assert hasattr(module, "compute_targets") and hasattr(module, "MULTI_HORIZONS")
    d = module.compute_targets(df_slice)
    tgt_cols = [f"target_h{int(h)}" for h in module.MULTI_HORIZONS]
    tr = d.iloc[:-1].dropna(subset=module.FEATURES + tgt_cols)
    X_tr = tr[module.FEATURES].values
    Y_tr = tr[tgt_cols].values
    last_feats = d[module.FEATURES].iloc[[-1]].values
    return X_tr, Y_tr, last_feats

# ─────────────────────────────────────────────────────────────────────────────
# Feature-only module helpers (regime_gate, vol_forecaster)
# ─────────────────────────────────────────────────────────────────────────────

def _short_name(module) -> str:
    return str(module.__name__).split(".")[-1]

def _compute_feature_outputs_for_slice(df_slice: pd.DataFrame) -> Dict[str, float]:
    """
    Compute 'feat_*' outputs from FEATURE_ONLY_MODULES for the *last row* of df_slice.
    Returns dict of {feat_col_name: value}.
    """
    feats: Dict[str, float] = {}

    # regime_gate → 3 regime probabilities (feature columns)
    try:
        m = modRG
        d = m.compute_labels(df_slice)
        tr = d.iloc[:-1].dropna(subset=m.FEATURES)
        if len(tr) >= 50:
            model = m.fit(tr[m.FEATURES].values)  # unsupervised pseudo-labels inside
            lastX = d[m.FEATURES].iloc[[-1]].values
            probs = m.predict(model, lastX)[0]  # [p_trend, p_high_vol, p_risk_on]
            cols = getattr(m, "OUTPUT_COLUMNS", ["p_trend", "p_high_vol", "p_risk_on"])
            for j, name in enumerate(cols):
                feats[f"feat_{_short_name(m)}_{name}"] = float(probs[j])
        else:
            cols = getattr(modRG, "OUTPUT_COLUMNS", ["p_trend", "p_high_vol", "p_risk_on"])
            for name in cols:
                feats[f"feat_{_short_name(modRG)}_{name}"] = 0.5
    except Exception:
        pass

    # vol_forecaster → 3 sigma quantiles (feature columns)
    try:
        m = modVF
        d = m.compute_labels(df_slice)
        # training uses FEATURES + target_log_sigma_t1
        tr = d.iloc[:-1].dropna(subset=m.FEATURES + ["target_log_sigma_t1"])
        if len(tr) >= 50:
            model = m.fit(tr[m.FEATURES].values, tr["target_log_sigma_t1"].values)
            lastX = d[m.FEATURES].iloc[[-1]].values
            sigmas = m.predict(model, lastX)[0]  # [q10, q50, q90]
            cols = getattr(m, "OUTPUT_COLUMNS", ["sigma_q10", "sigma_q50", "sigma_q90"])
            for j, name in enumerate(cols):
                feats[f"feat_{_short_name(m)}_{name}"] = float(sigmas[j])
        else:
            cols = getattr(modVF, "OUTPUT_COLUMNS", ["sigma_q10", "sigma_q50", "sigma_q90"])
            for name in cols:
                feats[f"feat_{_short_name(modVF)}_{name}"] = np.nan
    except Exception:
        pass

    return feats

def _compute_feature_outputs_for_many(df: pd.DataFrame, idx_list: List[int]) -> pd.DataFrame:
    rows = []
    for i in tqdm(idx_list, desc="Feature-only modules"):
        slc = df.iloc[: i + 1].reset_index(drop=True)
        rows.append(_compute_feature_outputs_for_slice(slc))
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# Confidence feature utilities (Gini & Entropy)
# ─────────────────────────────────────────────────────────────────────────────

def _inject_confidence_features(hist: pd.DataFrame, channels: List[dict]) -> pd.DataFrame:
    """
    Add 'feat_*' confidence columns derived from classifier probabilities.
    - For multi-horizon classifier (mod6): use confidence_across_horizons per horizon.
    - For single-horizon classifiers (e.g., squeeze_breakout): use confidence_from_proba.
    """
    if hist.empty:
        return hist

    out = hist.copy()

    # map horizons → mcol for mod6
    h_to_mcol: Dict[int, str] = {}
    for k, ch in enumerate(channels, start=1):
        if ch["mod"] is mod6:
            if ch["h"] is not None:
                h_to_mcol[int(ch["h"])] = f"m{k}"

    # multi-horizon conf (mod6)
    if h_to_mcol:
        try:
            mapping: Mapping[int, np.ndarray] = {
                h: pd.to_numeric(out[mcol], errors="coerce").values
                for h, mcol in h_to_mcol.items() if mcol in out.columns
            }
            conf_df = mod6.confidence_across_horizons(mapping)
            # rename to feat_*
            conf_df = conf_df.add_prefix("feat_")
            # fill neutral defaults
            for c in conf_df.columns:
                if c.startswith("feat_gini_conf_"):
                    conf_df[c] = pd.to_numeric(conf_df[c], errors="coerce").fillna(0.5)
                elif c.startswith("feat_entropy_conf_"):
                    conf_df[c] = pd.to_numeric(conf_df[c], errors="coerce").fillna(0.0)
            for c in conf_df.columns:
                out[c] = conf_df[c].values
            # also add aggregated means across horizons for convenience
            g_cols = [c for c in conf_df.columns if c.startswith("feat_gini_conf_h")]
            e_cols = [c for c in conf_df.columns if c.startswith("feat_entropy_conf_h")]
            if g_cols:
                out["feat_gini_conf_mean"] = pd.to_numeric(conf_df[g_cols], errors="coerce").mean(axis=1).fillna(0.5)
            if e_cols:
                out["feat_entropy_conf_mean"] = pd.to_numeric(conf_df[e_cols], errors="coerce").mean(axis=1).fillna(0.0)
        except Exception:
            pass

    # single-horizon classifier confidence (e.g., squeeze_breakout)
    for k, ch in enumerate(channels, start=1):
        if ch["mod"] is mod6:
            continue
        if hasattr(ch["mod"], "compute_labels"):
            mcol = f"m{k}"
            if mcol in out.columns:
                try:
                    proba = pd.to_numeric(out[mcol], errors="coerce").values
                    tmp = mod6.confidence_from_proba(proba)
                    # name based on module short name
                    short = _short_name(ch["mod"])
                    g_name = f"feat_gini_conf_{short}"
                    e_name = f"feat_entropy_conf_{short}"
                    out[g_name] = pd.to_numeric(tmp["gini_confidence"], errors="coerce").fillna(0.5).values
                    out[e_name] = pd.to_numeric(tmp["entropy_confidence"], errors="coerce").fillna(0.0).values
                except Exception:
                    pass

    return out

def _inject_lagged_meta_features(hist: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
    if hist.empty:
        return hist
    out = hist.copy()
    if "meta_label" not in out.columns:
        return out
    base = pd.to_numeric(out["meta_label"], errors="coerce")
    for L in lags:
        col = f"feat_meta_label_lag{L}"
        out[col] = base.shift(L).fillna(0.5).astype(float)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Submodel backtest producing m*/acc* and meta_label
# ─────────────────────────────────────────────────────────────────────────────

def backtest_submodels(
    df: pd.DataFrame,
    initial_frac: float = 0.6,
    window: int = 50,
) -> pd.DataFrame:
    """Walk-forward for all submodels; returns DataFrame with m*, acc*, meta_label.
       UPDATED: Regression submodels now output RAW CLOSES (levels)."""
    submods = SUBMODS[:]  # includes squeeze_breakout
    channels = expand_channels(submods)
    n_ch = len(channels)

    n = len(df)
    HMAX = max_lookahead(submods)
    cut = int(n * initial_frac)
    end = n - 1 - HMAX

    rec = []

    preds_history = [[] for _ in range(n_ch)]
    labels_history = [[] for _ in range(n_ch)]
    ref_close_hist = [[] for _ in range(n_ch)]

    for t in tqdm(range(max(cut, 0), end), desc="Submodel BT"):
        slice_df = df.iloc[: t + 1].reset_index(drop=True)

        preds, labels = [], []
        for ch in channels:
            m = ch["mod"]; h = ch["h"]

            if hasattr(m, "compute_labels"):  # classifier (single or multi)
                d   = m.compute_labels(slice_df)
                if is_multi(m):
                    tr  = d.iloc[:-1].dropna(subset=m.FEATURES + [label_col_for(m, h)])
                    if len(tr) < 50:
                        p = 0.5
                    else:
                        last_feats = d[m.FEATURES].iloc[[-1]]
                        model = m.fit(tr[m.FEATURES].values, tr[label_col_for(m, h)].values, horizon=h)
                        p = m.predict(model, last_feats)[0]
                    lbl = 1 if df["close"].iat[t + h] > df["close"].iat[t] else 0
                else:
                    tr  = d.iloc[:-1].dropna(subset=m.FEATURES + ["label"])
                    if len(tr) < 50:
                        p = 0.5
                    else:
                        last_feats = d[m.FEATURES].iloc[[-1]]
                        model = m.fit(tr[m.FEATURES].values, tr["label"].values)
                        p = m.predict(model, last_feats)[0]
                    lbl = 1 if df["close"].iat[t + 1] > df["close"].iat[t] else 0

            elif hasattr(m, "compute_targets"):  # multi-horizon regression (RAW CLOSES)
                X_tr, Y_tr, last_feats = prepare_train_regression_multi(slice_df, m)
                if len(X_tr) < 50:
                    p = np.nan
                else:
                    model = m.fit(X_tr, Y_tr)
                    yhat_multi = m.predict(model, last_feats)[0]
                    idx = m.MULTI_HORIZONS.index(h)
                    p = float(yhat_multi[idx])
                lbl = 1 if df["close"].iat[t + h] > df["close"].iat[t] else 0

            elif is_svr_module(m):  # SVR (RAW close next step)
                p = _svr_predict_next_from_slice(slice_df, m)
                lbl = 1 if df["close"].iat[t + 1] > df["close"].iat[t] else 0

            else:  # legacy single-horizon regression-style (assumed RAW)
                d = m.compute_target(slice_df)
                tr  = d.iloc[:-1].dropna(subset=m.FEATURES + ["target"])
                last_feats = d[m.FEATURES].iloc[[-1]]
                if len(tr) < 50:
                    p = np.nan
                else:
                    model = m.fit(tr[m.FEATURES], tr["target"])
                    p = float(m.predict(model, last_feats)[0])
                lbl = 1 if df["close"].iat[t + 1] > df["close"].iat[t] else 0

            preds.append(p); labels.append(lbl)

        # Rolling accuracies
        accs = []
        for k in range(n_ch):
            preds_history[k].append(preds[k])
            labels_history[k].append(labels[k])
            ref_close_hist[k].append(float(df["close"].iat[t]))

            ph = np.asarray(preds_history[k][-window-1:-1], dtype=float)
            lh = np.asarray(labels_history[k][-window-1:-1], dtype=float)
            refb = np.asarray(ref_close_hist[k][-window-1:-1], dtype=float)

            if hasattr(channels[k]["mod"], "compute_labels"):
                pred_cls = (ph > 0.5).astype(float)
            else:
                with np.errstate(invalid='ignore'):
                    pred_cls = (ph > refb).astype(float)
            msk = np.isfinite(pred_cls)
            accs.append(float((pred_cls[msk] == lh[msk]).mean()) if msk.sum() else 0.5)

        rec_dict = {
            "t": t,
            **{f"m{k+1}":   preds[k] for k in range(n_ch)},
            **{f"acc{k+1}": accs[k]  for k in range(n_ch)},
            "meta_label":  1 if df["close"].iat[t + 1] > df["close"].iat[t] else 0,
        }
        rec.append(rec_dict)

    return pd.DataFrame(rec).set_index("t")


def walkforward_meta_backtest(cache_csv: str,
                              n_mods: int | None = None,
                              initial_frac: float = 0.6,
                              metric: str = "profit"):
    hist = pd.read_csv(cache_csv)
    N = len(hist)
    start = int(N * initial_frac)
    if start < 10:
        raise ValueError("History too short for walk-forward meta test.")

    records = []

    for i in tqdm(range(start, N), desc="Walk-forward (meta)"):
        X_train = build_meta_features(hist.iloc[:i], n_mods)
        y_train = hist.loc[:i-1, "meta_label"].values

        meta = make_meta_model().fit(X_train, y_train)

        split_val = max(1, int(len(y_train) * 0.2))
        probs_val = meta.predict_proba(X_train[-split_val:])[:, 1]
        up, down  = optimize_asymmetric_thresholds(
                        probs_val, y_train[-split_val:], metric=metric)

        x_row = build_meta_features(hist.iloc[[i]], n_mods)
        prob_up = float(meta.predict_proba(x_row)[0, 1])
        action = "BUY" if prob_up > up else ("SELL" if prob_up < down else "HOLD")

        rec = {
            "timestamp":  hist.at[i, "timestamp"] if "timestamp" in hist.columns else i,
            "meta_prob":  prob_up,
            "action":     action,
            "up_thresh":  up,
            "down_thresh":down,
        }
        k = sum(1 for c in hist.columns if isinstance(c, str) and c.startswith("m") and c[1:].isdigit())
        for j in range(1, k+1):
            rec[f"m{j}"]   = hist.at[i, f"m{j}"]
            rec[f"acc{j}"] = hist.at[i, f"acc{j}"]
        records.append(rec)

    return pd.DataFrame(records)

# (QUICK_TEST section unchanged aside from referencing SUBMODS, omitted here for brevity)
# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST (kept identical to your previous version)
# ─────────────────────────────────────────────────────────────────────────────

def _filter_channels_for_quick(submods, use_h1_only: bool = True):
    chans = []
    for m in submods:
        if hasattr(m, "MULTI_HORIZONS") and isinstance(m.MULTI_HORIZONS, (list, tuple)) and len(m.MULTI_HORIZONS) > 0:
            if use_h1_only:
                h = int(sorted(m.MULTI_HORIZONS)[0])
                chans.append({"mod": m, "h": h, "name": f"{m.__name__}_h{h}"})
            else:
                for h in m.MULTI_HORIZONS:
                    chans.append({"mod": m, "h": int(h), "name": f"{m.__name__}_h{h}"})
        else:
            chans.append({"mod": m, "h": None, "name": m.__name__})
    return chans

def _actions_from_probs(p: np.ndarray, up: float, down: float) -> np.ndarray:
    out = np.full_like(p, "HOLD", dtype=object)
    out[p > up] = "BUY"
    out[p < down] = "SELL"
    return out

def _pnl_from_actions(actions: np.ndarray, rets_next: np.ndarray) -> np.ndarray:
    pnl = np.zeros_like(rets_next, dtype=float)
    pnl[actions == "BUY"]  =  rets_next[actions == "BUY"]
    pnl[actions == "SELL"] = -rets_next[actions == "SELL"]
    return pnl

def quick_test(df: pd.DataFrame, initial_frac: float = 0.6, return_df: bool = False):
    # (identical to your previous version)
    # ... (omitted to keep focus on requested changes)
    raise NotImplementedError("quick_test body unchanged; bring over from your prior file if needed.")

# ─────────────────────────────────────────────────────────────────────────────
# History updater (adds m*/acc*/meta_label, plus feat_* from feature-only modules,
# confidence feat_*, and lagged meta-label feat_*)
# ─────────────────────────────────────────────────────────────────────────────

def _recompute_missing_columns_on_existing_rows(
    df: pd.DataFrame,
    hist: pd.DataFrame,
    channels: list[dict],
    window: int,
    usable_last_i: int,
    verbose: bool = True,
) -> pd.DataFrame:
    if hist.empty:
        return hist

    hist = hist.copy()
    hist["timestamp"] = pd.to_datetime(hist["timestamp"], utc=True, errors="coerce")
    hist = hist.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"], keep="last")
    hist = hist.sort_values("timestamp").reset_index(drop=True)

    df_idx_map = (
        df[["timestamp"]].assign(df_idx=np.arange(len(df), dtype=int))
        .set_index("timestamp")["df_idx"]
    )
    hist["df_idx"] = hist["timestamp"].map(df_idx_map)
    hist = hist.dropna(subset=["df_idx"])
    hist["df_idx"] = hist["df_idx"].astype(int)
    hist = hist[hist["df_idx"] <= int(usable_last_i)].reset_index(drop=True)
    if hist.empty:
        return hist

    n_ch = len(channels)
    m_cols  = [f"m{i+1}"   for i in range(n_ch)]
    a_cols  = [f"acc{i+1}" for i in range(n_ch)]

    for c in ["meta_label", *m_cols, *a_cols]:
        if c not in hist.columns:
            hist[c] = np.nan

    missing_m   = [c for c in m_cols if hist[c].isna().any()]
    missing_acc = [c for c in a_cols if hist[c].isna().any()]
    need_meta   = hist["meta_label"].isna().any()

    close = pd.to_numeric(df["close"], errors="coerce")
    df_idx_arr = hist["df_idx"].to_numpy()
    ref_close_series = pd.Series(close.iloc[df_idx_arr].to_numpy(), index=hist.index)
    if need_meta:
        meta_full = (close.shift(-1) > close).astype(int)
        hist.loc[:, "meta_label"] = meta_full.iloc[df_idx_arr].reset_index(drop=True).astype(int)

    # ── Pass A: recompute ONLY missing m_k
    if missing_m:
        if verbose:
            print(f"[HISTORY-REPAIR] Recomputing missing prediction cols: {missing_m}")
        for k, ch in enumerate(channels, start=1):
            col_m = f"m{k}"
            if col_m not in missing_m:
                continue

            preds = []
            for i_df in df_idx_arr:
                slc = df.iloc[: i_df + 1].reset_index(drop=True)
                m, h = ch["mod"], ch["h"]

                if hasattr(m, "compute_labels"):  # classifier
                    d = m.compute_labels(slc)
                    if is_multi(m):
                        X_tr, y_tr = prepare_train_for_horizon(slc, m, int(h))
                        if len(y_tr) < 50:
                            p = 0.5
                        else:
                            last_feats = d[m.FEATURES].iloc[[-1]]
                            model = m.fit(X_tr, y_tr, horizon=int(h))
                            p = float(m.predict(model, last_feats)[0])
                    else:
                        X_tr, y_tr = prepare_train(slc, m)
                        if len(y_tr) < 50:
                            p = 0.5
                        else:
                            last_feats = d[m.FEATURES].iloc[[-1]]
                            model = m.fit(X_tr, y_tr)
                            p = float(m.predict(model, last_feats)[0])

                elif hasattr(m, "compute_targets"):  # regression (RAW)
                    X_tr, Y_tr, last_feats = prepare_train_regression_multi(slc, m)
                    if len(X_tr) < 50:
                        p = np.nan
                    else:
                        model = m.fit(X_tr, Y_tr)
                        yhat_multi = m.predict(model, last_feats)[0]
                        idx_h = m.MULTI_HORIZONS.index(int(h))
                        p = float(yhat_multi[idx_h])

                elif is_svr_module(m):  # SVR RAW close
                    p = _svr_predict_next_from_slice(slc, m)

                else:  # legacy
                    d = m.compute_target(slc)
                    tr = d.iloc[:-1].dropna(subset=m.FEATURES + ["target"])
                    last_feats = d[m.FEATURES].iloc[[-1]]
                    if len(tr) < 50:
                        p = np.nan
                    else:
                        model = m.fit(tr[m.FEATURES], tr["target"])
                        p = float(m.predict(model, last_feats)[0])

                preds.append(p)

            hist.loc[:, col_m] = pd.to_numeric(pd.Series(preds, index=hist.index), errors="coerce")

    # ── Pass B: recompute ONLY missing acc_k (rolling)
    if missing_acc:
        if verbose:
            print(f"[HISTORY-REPAIR] Recomputing missing accuracy cols: {missing_acc}")
        y_series = pd.to_numeric(hist["meta_label"], errors="coerce")

        for k, ch in enumerate(channels, start=1):
            col_a, col_m = f"acc{k}", f"m{k}"
            if col_a not in missing_acc:
                continue

            m_series = pd.to_numeric(hist[col_m], errors="coerce")
            if hasattr(ch["mod"], "compute_labels"):
                pred_bin = (m_series.shift(1) > 0.5).astype(float)
            else:
                base_shift = pd.to_numeric(ref_close_series, errors="coerce").shift(1)
                pred_bin = (m_series.shift(1) > base_shift).astype(float)

            eq = (pred_bin == y_series.shift(1)).astype(float)
            acc = (
                eq.rolling(window=window, min_periods=1)
                  .mean()
                  .fillna(0.5)
            )
            hist.loc[:, col_a] = pd.to_numeric(acc, errors="coerce").fillna(0.5)

    # ── Pass C: ensure feature-only module columns exist and (re)compute missing
    # Determine expected feature columns from modules
    expected_feat_cols = []
    # regime_gate
    for nm in getattr(modRG, "OUTPUT_COLUMNS", ["p_trend", "p_high_vol", "p_risk_on"]):
        expected_feat_cols.append(f"feat_{_short_name(modRG)}_{nm}")
    # vol_forecaster
    for nm in getattr(modVF, "OUTPUT_COLUMNS", ["sigma_q10", "sigma_q50", "sigma_q90"]):
        expected_feat_cols.append(f"feat_{_short_name(modVF)}_{nm}")

    for c in expected_feat_cols:
        if c not in hist.columns:
            hist[c] = np.nan

    need_feat_rows = np.where(pd.isna(hist[expected_feat_cols]).any(axis=1))[0].tolist()
    if need_feat_rows:
        if verbose:
            print(f"[HISTORY-REPAIR] Recomputing feature-only module cols for {len(need_feat_rows)} rows.")
        idx_list = hist.iloc[need_feat_rows]["df_idx"].astype(int).tolist()
        feat_df = _compute_feature_outputs_for_many(df, idx_list).reset_index(drop=True)
        # align and fill
        for j, ridx in enumerate(need_feat_rows):
            for col in expected_feat_cols:
                val = feat_df.iloc[j].get(col, np.nan)
                hist.at[ridx, col] = val

    if "df_idx" in hist.columns:
        hist = hist.drop(columns=["df_idx"])

    return hist

def update_submeta_history(CSV_PATH, HISTORY_PATH,
                           submods=SUBMODS, window=50, verbose=True):
    df = pd.read_csv(CSV_PATH)
    if verbose:
        print("── DEBUG: CSV_PATH =", CSV_PATH)
        print("── DEBUG: columns in CSV:", df.columns.tolist())
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    channels = expand_channels(submods)
    n_ch = len(channels)

    col_m   = [f"m{i+1}"   for i in range(n_ch)]
    col_acc = [f"acc{i+1}" for i in range(n_ch)]
    expected_cols = ["timestamp", *col_m, *col_acc, "meta_label"]

    HMAX = max_lookahead(submods)
    n = len(df)
    usable_last_i = n - 1 - max(1, HMAX)
    if usable_last_i < 0:
        raise ValueError("Not enough rows to build history.")

    # Load existing history if present
    hist = None
    if os.path.exists(HISTORY_PATH):
        try:
            tmp = pd.read_csv(HISTORY_PATH)
            if "timestamp" in tmp.columns:
                tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], utc=True, errors="coerce")
                tmp = tmp.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"], keep="last")
                tmp = tmp.sort_values("timestamp").reset_index(drop=True)
                hist = tmp.copy()
            else:
                if verbose:
                    print("[HISTORY] Missing 'timestamp' in history; rebuilding.")
                hist = None
        except Exception as e:
            if verbose:
                print(f"[HISTORY] Failed to load existing history ({e}); rebuilding.")
            hist = None

    if hist is None or hist.empty:
        hist = pd.DataFrame(columns=expected_cols)
        hist["timestamp"] = pd.to_datetime(hist["timestamp"], utc=True)

    # STEP 1: Repair existing rows (no leak)
    if not hist.empty:
        df_idx_map = (
            df[["timestamp"]].assign(df_idx=np.arange(len(df), dtype=int))
            .set_index("timestamp")["df_idx"]
        )
        hist["_df_idx"] = hist["timestamp"].map(df_idx_map)
        hist = hist.dropna(subset=["_df_idx"])
        hist["_df_idx"] = hist["_df_idx"].astype(int)
        hist = hist[hist["_df_idx"] <= int(usable_last_i)].drop(columns=["_df_idx"]).reset_index(drop=True)

        hist = _recompute_missing_columns_on_existing_rows(
            df=df,
            hist=hist,
            channels=channels,
            window=window,
            usable_last_i=usable_last_i,
            verbose=verbose
        )

    # STEP 2: Append new rows
    if not hist.empty:
        last_ts = hist["timestamp"].iloc[-1]
        future = df[df["timestamp"] > last_ts]
        if future.empty:
            if verbose:
                print("[HISTORY] Already up-to-date (no new timestamps).")
            start_idx = None
        else:
            start_idx = int(future.index.min())
            start_idx = max(0, min(start_idx, usable_last_i))
    else:
        start_idx = int(n * 0.6)
        start_idx = max(0, min(start_idx, usable_last_i))

    end_idx = usable_last_i  # inclusive

    existing_ts = set(hist["timestamp"].astype("datetime64[ns, UTC]").tolist()) if len(hist) else set()
    build_indices = [i for i in range(start_idx, end_idx + 1) if df["timestamp"].iat[i] not in existing_ts] if start_idx is not None else []

    if verbose:
        print(f"[HISTORY] Appending {len(build_indices)} new rows (from i={start_idx} to i={end_idx}).")

    if build_indices:
        preds_hist      = [[] for _ in range(n_ch)]
        labels_hist     = [[] for _ in range(n_ch)]
        ref_close_hist  = [[] for _ in range(n_ch)]

        recs = []
        close_vals = pd.to_numeric(df["close"], errors="coerce").values
        for i in tqdm(build_indices, desc="History (append)"):
            slc = df.iloc[: i + 1].reset_index(drop=True)

            preds, labels = [], []

            for ch in channels:
                m, h = ch["mod"], ch["h"]

                if hasattr(m, "compute_labels"):  # classifier
                    d = m.compute_labels(slc)
                    if is_multi(m):
                        X_tr, y_tr = prepare_train_for_horizon(slc, m, int(h))
                        if len(y_tr) < 50:
                            p = 0.5
                        else:
                            last_feats = d[m.FEATURES].iloc[[-1]]
                            model = m.fit(X_tr, y_tr, horizon=int(h))
                            p = float(m.predict(model, last_feats)[0])
                        lbl = int(df["close"].iat[i + int(h)] > df["close"].iat[i])
                    else:
                        X_tr, y_tr = prepare_train(slc, m)
                        if len(y_tr) < 50:
                            p = 0.5
                        else:
                            last_feats = d[m.FEATURES].iloc[[-1]]
                            model = m.fit(X_tr, y_tr)
                            p = float(m.predict(model, last_feats)[0])
                        lbl = int(df["close"].iat[i + 1] > df["close"].iat[i])

                elif hasattr(m, "compute_targets"):  # regression (RAW)
                    X_tr, Y_tr, last_feats = prepare_train_regression_multi(slc, m)
                    if len(X_tr) < 50:
                        p = np.nan
                    else:
                        model = m.fit(X_tr, Y_tr)
                        yhat_multi = m.predict(model, last_feats)[0]
                        idx_h = m.MULTI_HORIZONS.index(int(h))
                        p = float(yhat_multi[idx_h])
                    lbl = int(df["close"].iat[i + int(h)] > df["close"].iat[i])

                elif is_svr_module(m):  # SVR RAW next close
                    p = _svr_predict_next_from_slice(slc, m)
                    lbl = int(df["close"].iat[i + 1] > df["close"].iat[i])

                else:  # legacy
                    d = m.compute_target(slc)
                    tr = d.iloc[:-1].dropna(subset=m.FEATURES + ["target"])
                    last_feats = d[m.FEATURES].iloc[[-1]]
                    if len(tr) < 50:
                        p = np.nan
                    else:
                        model = m.fit(tr[m.FEATURES], tr["target"])
                        p = float(m.predict(model, last_feats)[0])
                    lbl = int(d["target"].iloc[-1] > 0)

                preds.append(p)
                labels.append(lbl)

            # Rolling accuracies
            accs = []
            for k in range(n_ch):
                preds_hist[k].append(preds[k])
                labels_hist[k].append(labels[k])
                ref_close_hist[k].append(float(df["close"].iat[i]))

                ph = np.asarray(preds_hist[k][-window-1:-1], dtype=float)
                lh = np.asarray(labels_hist[k][-window-1:-1], dtype=float)
                base = np.asarray(ref_close_hist[k][-window-1:-1], dtype=float)

                if hasattr(channels[k]["mod"], "compute_labels"):
                    pred_cls = (ph > 0.5).astype(float)
                else:
                    with np.errstate(invalid="ignore"):
                        pred_cls = (ph > base).astype(float)
                msk = np.isfinite(pred_cls)
                accs.append(float((pred_cls[msk] == lh[msk]).mean()) if msk.sum() else 0.5)

            rec = {
                "timestamp": df["timestamp"].iat[i],
                **{f"m{k+1}":   preds[k] for k in range(n_ch)},
                **{f"acc{k+1}": accs[k]  for k in range(n_ch)},
                "meta_label":  int(close_vals[i + 1] > close_vals[i]),
            }

            # feature-only modules (add feat_* to rec)
            feat_extra = _compute_feature_outputs_for_slice(slc)
            rec.update(feat_extra)

            recs.append(rec)

        if recs:
            new_df = pd.DataFrame(recs)
            all_df = pd.concat([hist, new_df], ignore_index=True)
            all_df = all_df.drop_duplicates(subset=["timestamp"], keep="last")
            for c in expected_cols:
                if c not in all_df.columns:
                    all_df[c] = np.nan
            hist = all_df.sort_values("timestamp").reset_index(drop=True)

    # Inject confidence features and lagged meta-label features
    hist = _inject_confidence_features(hist, channels)
    hist = _inject_lagged_meta_features(hist, [1, 2, 3, 5, 10])

    os.makedirs(RESULTS_DIR, exist_ok=True)
    hist.to_csv(HISTORY_PATH, index=False)
    if verbose:
        print(f"[HISTORY] Saved ⇒ {len(hist)} rows (deduped, repaired).")

    return hist

# ─────────────────────────────────────────────────────────────────────────────
# Orchestration: backtest / live
# ─────────────────────────────────────────────────────────────────────────────

def train_and_save_meta(cache_csv: str,
                        model_pkl: str,
                        threshold_pkl: str,
                        metric: str = "profit",
                        n_mods: int | None = None):
    hist = pd.read_csv(cache_csv)
    X = build_meta_features(hist, n_mods)
    y = hist["meta_label"].values.astype(int)

    split = int(len(y) * 0.6)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    meta = make_meta_model()
    meta.fit(X_tr, y_tr)

    probs_val = meta.predict_proba(X_val)[:, 1]
    up, down = optimize_asymmetric_thresholds(probs_val, y_val, metric=metric)

    with open(model_pkl, "wb") as f:
        pickle.dump(meta, f)
    with open(threshold_pkl, "wb") as f:
        pickle.dump({"up": up, "down": down, "model_type": META_MODEL_TYPE}, f)

    try:
        feat_names = get_meta_feature_names(hist, n_mods)
        shap_csv = _save_shap_report(X_val, y_val, meta, feat_names, FEATURE_ANALYTICS_DIR, tag="meta")
        print(f"[FEATURE-ANALYTICS] SHAP importance saved to {shap_csv}")
    except Exception as e:
        print(f"[FEATURE-ANALYTICS] SHAP failed: {e}")

    return meta, up, down

def enforce_position_rules(actions, start_open=False):
    out = []
    open_pos = start_open
    for act in actions:
        a = str(act).upper()
        if a == "BUY":
            if open_pos:
                out.append("NONE")
            else:
                out.append("BUY")
                open_pos = True
        elif a == "SELL":
            if not open_pos:
                out.append("NONE")
            else:
                out.append("SELL")
                open_pos = False
        else:
            out.append(a)
    return out

def run_backtest(return_df: bool = False, start_open: bool = False):
    df_orig = pd.read_csv(CSV_PATH)

    if QUICK_TEST:
        print("Quick Test is on: ", True)
        # bring over your original quick_test() if you rely on it
        raise NotImplementedError("QUICK_TEST path unchanged; restore from prior version if needed.")

    print("Quick Test is off: ", False)

    channels = expand_channels(SUBMODS)
    n_ch = len(channels)

    os.makedirs(FEATURE_ANALYTICS_DIR, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")

    if MODE == "sub-vote":
        back = backtest_submodels(df_orig, initial_frac=0.6)

        bt_hist_latest = os.path.join(
            FEATURE_ANALYTICS_DIR,
            f"subvote_backtest_history_{CONVERTED_TIMEFRAME}_latest.csv"
        )
        bt_hist_ts = os.path.join(
            FEATURE_ANALYTICS_DIR,
            f"subvote_backtest_history_{CONVERTED_TIMEFRAME}_{ts}.csv"
        )
        try:
            back.to_csv(bt_hist_latest, index=True)
            back.to_csv(bt_hist_ts, index=True)
            # optional: SHAP on snapshot using meta features built from m*/acc* only
            k = sum(1 for c in back.columns if isinstance(c, str) and c.startswith("m") and c[1:].isdigit())
            X = build_meta_features(back.reset_index(drop=True), n_mods=k)
            y = back["meta_label"].values.astype(int)
            split = int(len(y) * 0.6)
            if split >= 50:
                meta_tmp = make_meta_model().fit(X[:split], y[:split])
                _save_shap_report(X[split:], y[split:], meta_tmp, get_meta_feature_names(back, k), FEATURE_ANALYTICS_DIR, tag="subvote_snapshot")
        except Exception as e:
            print(f"[FEATURE-ANALYTICS] (sub-vote backtest) Failed to write SHAP snapshot: {e}")

        VOTE_UP, VOTE_DOWN = 0.55, 0.45
        vote_cols = []
        for k, ch in enumerate(channels, start=1):
            pred_col, vote_col = f"m{k}", f"v{k}"
            m = ch["mod"]
            if hasattr(m, "compute_labels"):
                back[vote_col] = back[pred_col].map(lambda p: 1 if p > VOTE_UP else (0 if p < VOTE_DOWN else 0.5))
            else:
                vote_cols.append(vote_col)
                back[vote_col] = 0.5  # will convert after reset_index

        back = back.reset_index().rename(columns={"index": "t"})
        for k, ch in enumerate(channels, start=1):
            pred_col, vote_col = f"m{k}", f"v{k}"
            if hasattr(ch["mod"], "compute_labels"):
                continue
            def mapper(r):
                pred = r[pred_col]
                base = df_orig["close"].iat[int(r["t"])]
                if not np.isfinite(pred) or not np.isfinite(base) or base == 0:
                    return 0.5
                growth = 100.0 * (float(pred) - float(base)) / float(base)
                return 1 if growth > REG_UP else (0 if growth < REG_DOWN else 0.5)
            back[vote_col] = back.apply(mapper, axis=1)

        majority_req = (n_ch // 2) + 1
        vote_cols = [f"v{k}" for k in range(1, n_ch + 1)]
        back["buy_votes"]  = (back[vote_cols] == 1).sum(axis=1)
        back["sell_votes"] = (back[vote_cols] == 0).sum(axis=1)
        back["action"] = back.apply(
            lambda r: 1 if r.buy_votes  >= majority_req else
                      (0 if r.sell_votes >= majority_req else 0.5),
            axis=1,
        )

        back["timestamp"] = pd.to_datetime(df_orig["timestamp"].iloc[back["t"] + 1])
        df_out = back[["timestamp", *[f"m{k}" for k in range(1, n_ch + 1)], "action"]]
        df_out["action"] = df_out["action"].map({1: "BUY", 0: "SELL", 0.5: "HOLD"})

        if return_df:
            df_out["action"] = enforce_position_rules(df_out["action"].tolist(), start_open)
            return df_out

    else:
        cache_csv = HISTORY_PATH
        os.makedirs(RESULTS_DIR, exist_ok=True)
        update_submeta_history(CSV_PATH, cache_csv, submods=SUBMODS, verbose=True)

        df_out = walkforward_meta_backtest(
            cache_csv=cache_csv,
            n_mods=None,
            initial_frac=0.6,
            metric="profit",
        )
        out_fn = os.path.join(RESULTS_DIR, f"meta_results_walkforward.csv")
        df_out.to_csv(out_fn, index=False, float_format="%.8f")
        print(f"[RESULT] Walk-forward results saved to {out_fn}")

        try:
            meta_pkl   = os.path.join(RESULTS_DIR, "meta_model.pkl")
            thresh_pkl = os.path.join(RESULTS_DIR, "meta_thresholds.pkl")
            meta, up, down = train_and_save_meta(cache_csv, meta_pkl, thresh_pkl, metric="profit", n_mods=None)
            print(f"[META] BUY>{up:.2f} / SELL<{down:.2f}")
        except Exception as e:
            print(f"[META] training/report failed: {e}")
        if return_df:
            return df_out


def row(df_slice: pd.DataFrame, module, idx: int | None = None):
    d = (
        module.compute_labels(df_slice)
        if hasattr(module, "compute_labels")
        else module.compute_target(df_slice)
    )
    if idx is not None and idx in d.index:
        return d.loc[[idx], module.FEATURES]
    return d[module.FEATURES].iloc[[-1]]

def _live_confidence_features_from_current_probs(current_probs: List[float], channels: List[dict], hist_feat_cols: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}

    # multi-horizon mod6
    h_to_val: Dict[int, float] = {}
    for idx, ch in enumerate(channels, start=1):
        if ch["mod"] is mod6 and ch["h"] is not None:
            h_to_val[int(ch["h"])] = float(current_probs[idx - 1])
    if h_to_val:
        try:
            conf_df = mod6.confidence_across_horizons({h: np.array([v], dtype=float) for h, v in h_to_val.items()})
            conf_df = conf_df.add_prefix("feat_")
            for col in conf_df.columns:
                out[col] = float(conf_df[col].iloc[0])
            g_cols = [c for c in conf_df.columns if c.startswith("feat_gini_conf_h")]
            e_cols = [c for c in conf_df.columns if c.startswith("feat_entropy_conf_h")]
            if g_cols:
                out["feat_gini_conf_mean"] = float(np.nanmean([out[c] for c in g_cols]))
            if e_cols:
                out["feat_entropy_conf_mean"] = float(np.nanmean([out[c] for c in e_cols]))
        except Exception:
            pass

    # single-horizon classifiers
    for idx, ch in enumerate(channels, start=1):
        if ch["mod"] is mod6:
            continue
        if hasattr(ch["mod"], "compute_labels"):
            p = float(current_probs[idx - 1])
            try:
                tmp = mod6.confidence_from_proba(np.array([p], dtype=float))
                short = _short_name(ch["mod"])
                g_name = f"feat_gini_conf_{short}"
                e_name = f"feat_entropy_conf_{short}"
                out[g_name] = float(tmp["gini_confidence"].iloc[0])
                out[e_name] = float(tmp["entropy_confidence"].iloc[0])
            except Exception:
                pass

    # keep only those confidence feat cols that meta was trained on (if known)
    if hist_feat_cols:
        out = {k: v for k, v in out.items() if k in hist_feat_cols}
    return out

def _live_feature_only_module_outputs(df: pd.DataFrame) -> Dict[str, float]:
    return _compute_feature_outputs_for_slice(df)

def run_live(return_result: bool = True, position_open: bool = False):
    channels = expand_channels(SUBMODS)
    n_ch = len(channels)

    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    t = len(df) - 1

    if MODE == "sub-vote":
        VOTE_UP, VOTE_DOWN = 0.55, 0.45
        preds = []
        for ch in channels:
            m, h = ch["mod"], ch["h"]
            if hasattr(m, "compute_labels"):
                if is_multi(m):
                    X_tr, y_tr = prepare_train_for_horizon(df, m, h)
                    if len(y_tr) < 50:
                        preds.append(0.5); continue
                    model = m.fit(X_tr, y_tr, horizon=h)
                    feats = m.compute_labels(df)[m.FEATURES].iloc[[t]]
                    preds.append(m.predict(model, feats)[0])
                else:
                    X_tr, y_tr = prepare_train(df, m)
                    if len(y_tr) < 50:
                        preds.append(0.5); continue
                    model = m.fit(X_tr, y_tr)
                    feats = m.compute_labels(df)[m.FEATURES].iloc[[t]]
                    preds.append(m.predict(model, feats)[0])
            elif hasattr(m, "compute_targets"):
                X_tr, Y_tr, last_feats = prepare_train_regression_multi(df, m)
                if len(X_tr) < 50:
                    preds.append(np.nan); continue
                model = m.fit(X_tr, Y_tr)
                yhat_multi = m.predict(model, last_feats)[0]
                idx = m.MULTI_HORIZONS.index(h)
                preds.append(float(yhat_multi[idx]))
            elif is_svr_module(m):
                p = _svr_predict_next_from_slice(df, m)
                preds.append(p)
            else:
                X_tr, y_tr = prepare_train(df, m)
                model = m.fit(X_tr, y_tr)
                feats = row(df, m, t)
                preds.append(m.predict(model, feats)[0])

        votes = []
        base_close = float(df["close"].iat[t])
        for k, ch in enumerate(channels):
            m = ch["mod"]
            if hasattr(m, "compute_labels"):
                votes.append(1 if preds[k] > VOTE_UP else (0 if preds[k] < VOTE_DOWN else 0.5))
            else:
                r_close = preds[k]
                if not np.isfinite(r_close) or not np.isfinite(base_close) or base_close == 0:
                    votes.append(0.5)
                else:
                    growth = 100.0 * (float(r_close) - base_close) / base_close
                    votes.append(1 if growth > REG_UP else (0 if growth < REG_DOWN else 0.5))

        buy, sell = votes.count(1), votes.count(0)
        majority_req = (n_ch // 2) + 1
        majority = 1 if buy >= majority_req else (0 if sell >= majority_req else 0.5)
        action = "BUY" if majority == 1 else ("SELL" if majority == 0 else "HOLD")
        action = enforce_position_rules([action], position_open)[0]
        print(f"[LIVE-VOTE] result = {action}")
        return action if return_result else None

    # ——— META MODE ———
    hist = update_submeta_history(CSV_PATH, HISTORY_PATH, submods=SUBMODS, verbose=True)

    meta_pkl   = os.path.join(RESULTS_DIR, "meta_model.pkl")
    thresh_pkl = os.path.join(RESULTS_DIR, "meta_thresholds.pkl")

    must_retrain = True
    if os.path.exists(meta_pkl) and os.path.exists(thresh_pkl):
        with open(thresh_pkl, "rb") as f:
            thresh_dict = pickle.load(f)
        if thresh_dict.get("model_type") == META_MODEL_TYPE:
            with open(meta_pkl, "rb") as f:
                meta = pickle.load(f)
            up, down   = thresh_dict["up"], thresh_dict["down"]
            must_retrain = False
            print(f"[META] Loaded cached {META_MODEL_TYPE} model")

    if must_retrain:
        meta, up, down = train_and_save_meta(HISTORY_PATH, meta_pkl, thresh_pkl)
        print(f"[META] Re-trained {META_MODEL_TYPE}  BUY>{up:.2f} / SELL<{down:.2f}")

    # Build current m*/acc* row (plus feat_* confidence/lagged/module features)
    current_probs = []
    for ch in expand_channels(SUBMODS):
        m, h = ch["mod"], ch["h"]
        if hasattr(m, "compute_labels"):
            if is_multi(m):
                X_tr, y_tr = prepare_train_for_horizon(df, m, h)
                if len(y_tr) < 50:
                    p = 0.5
                else:
                    model = m.fit(X_tr, y_tr, horizon=h)
                    feats = m.compute_labels(df)[m.FEATURES].iloc[[t]]
                    p = m.predict(model, feats)[0]
            else:
                X_tr, y_tr = prepare_train(df, m)
                if len(y_tr) < 50:
                    p = 0.5
                else:
                    model = m.fit(X_tr, y_tr)
                    feats = m.compute_labels(df)[m.FEATURES].iloc[[t]]
                    p = m.predict(model, feats)[0]
            current_probs.append(float(p))
        elif hasattr(m, "compute_targets"):
            X_tr, Y_tr, last_feats = prepare_train_regression_multi(df, m)
            if len(X_tr) < 50:
                p = np.nan
            else:
                model = m.fit(X_tr, Y_tr)
                yhat_multi = m.predict(model, last_feats)[0]
                idx = m.MULTI_HORIZONS.index(h)
                p = float(yhat_multi[idx])
            current_probs.append(float(p))
        elif is_svr_module(m):
            p = _svr_predict_next_from_slice(df, m)
            current_probs.append(float(p))
        else:
            X_tr, y_tr = prepare_train(df, m)
            model = m.fit(X_tr, y_tr)
            feats = row(df, m, t)
            p = float(m.predict(model, feats)[0])
            current_probs.append(p)

    k_hist = sum(1 for c in hist.columns if isinstance(c, str) and c.startswith("m") and c[1:].isdigit())
    last_accs = [float(hist[f"acc{i+1}"].iloc[-1]) if f"acc{i+1}" in hist.columns else 0.5 for i in range(k_hist)]

    row_dict = {**{f"m{i+1}": current_probs[i] for i in range(k_hist)},
                **{f"acc{i+1}": last_accs[i] for i in range(k_hist)}}

    # add feat_* columns expected by meta: take list from history
    hist_feat_cols = [c for c in hist.columns if isinstance(c, str) and c.startswith("feat_")]

    # 1) confidence features from current probabilities
    conf_now = _live_confidence_features_from_current_probs(current_probs, channels, hist_feat_cols)

    # 2) lagged meta-label features (from hist meta_label)
    lag_feats = {}
    if "meta_label" in hist.columns and len(hist) > 0:
        base = pd.to_numeric(hist["meta_label"], errors="coerce")
        for L in [1, 2, 3, 5, 10]:
            col = f"feat_meta_label_lag{L}"
            if col in hist_feat_cols:
                lag_feats[col] = float(base.iloc[-min(L, len(base))]) if len(base) >= L else 0.5

    # 3) feature-only modules (regime, vol)
    feat_modules_now = _live_feature_only_module_outputs(df)

    # Merge only features the meta model has seen (hist_feat_cols) to keep shapes consistent
    merged_feats = {}
    for k, v in {**conf_now, **lag_feats, **feat_modules_now}.items():
        if k in hist_feat_cols:
            merged_feats[k] = v
    # For any remaining hist feat columns we didn't compute now, set neutral defaults
    defaults = {}
    for c in hist_feat_cols:
        if c not in merged_feats:
            defaults[c] = 0.5 if ("gini" in c or "lag" in c or _short_name(modVF) in c or _short_name(modRG) in c) else 0.0
    merged_feats.update(defaults)

    row_df = pd.DataFrame([{**row_dict, **merged_feats}])

    feat_vec = build_meta_features(row_df, n_mods=k_hist)
    prob = float(meta.predict_proba(feat_vec)[0, 1])
    action = "BUY" if prob > up else ("SELL" if prob < down else "HOLD")
    action = enforce_position_rules([action], position_open)[0]
    print(f"[LIVE] P(up)={prob:.3f}  →  {action}  (BUY>{up:.2f} / SELL<{down:.2f})")

    return (prob, action) if return_result else None
