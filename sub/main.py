# main.py (updated)
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import shap

from typing import Tuple, List

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit

# Sub-models (always include classifier; no USE_CLASSIFIER branching)
import sub.momentum as mod1
import sub.trend_continuation as mod2
import sub.mean_reversion as mod3
import sub.lagged_return as mod4
import sub.sentiment_volume as mod5
import sub.regressor as modR
import sub.classifier as mod6

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
QUICK_TEST = True

HISTORY_PATH = os.path.join(RESULTS_DIR, f"submeta_history_{CONVERTED_TIMEFRAME}.csv")

REG_UP, REG_DOWN = 0.003, -0.003  # for regression vote mapping → direction

FORCE_STATIC_THRESHOLDS = False
STATIC_UP = 0.55
STATIC_DOWN = 0.45

META_MODEL_TYPE = os.getenv("META_MODEL_TYPE", "cat")
VALID_META_MODELS = {"logreg", "lgbm", "xgb", "nn", "cat"}
if META_MODEL_TYPE not in VALID_META_MODELS:
    raise ValueError(f"META_MODEL_TYPE must be one of {VALID_META_MODELS}")

# The sub-mod set is fixed (no USE_CLASSIFIER flag)
SUBMODS = [mod1, mod2, mod3, mod4, mod5, modR, mod6]


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

def build_meta_features(df: pd.DataFrame, n_mods: int | None = None) -> np.ndarray:
    if n_mods is None:
        n_mods = sum(1 for c in df.columns if isinstance(c, str) and c.startswith("m") and c[1:].isdigit())
    prob_cols = [f"m{i+1}" for i in range(n_mods)]
    acc_cols  = [f"acc{i+1}" for i in range(n_mods)]

    P = df[prob_cols].values if prob_cols and set(prob_cols).issubset(df.columns) else np.zeros((len(df), 0))
    A = df[acc_cols].values  if acc_cols  and set(acc_cols ).issubset(df.columns) else np.zeros((len(df), 0))

    if P.shape[1] > 0:
        mean_p = P.mean(axis=1, keepdims=True)
        var_p  = P.var(axis=1,  keepdims=True)
        if A.shape[1] == P.shape[1] and A.shape[1] > 0:
            wavg_p = (P * A).sum(axis=1, keepdims=True) / (A.sum(axis=1, keepdims=True) + 1e-6)
        else:
            wavg_p = mean_p
    else:
        mean_p = var_p = wavg_p = np.zeros((len(df), 1))

    extra_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("feat_")]
    EX = df[extra_cols].values if extra_cols else np.zeros((len(df), 0))

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
    probs: np.ndarray, labels: np.ndarray, metric="accuracy",
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
    """
    SHAP importance for the meta model. Works with calibrated models by wrapping predict_proba.
    Falls back to permutation-like mean abs gradient if SHAP is unavailable.
    """
    try:
        # sample a small background to speed up KernelExplainer/Explainer
        bg = X[np.random.choice(len(X), size=min(200, len(X)), replace=False)]
        # function wrapper → proba of class 1
        f = lambda z: np.asarray(model.predict_proba(z))[:, 1]
        explainer = shap.Explainer(f, bg)
        sv = explainer(X)  # shap.Explanation
        vals = np.abs(sv.values).mean(axis=0)
        df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": vals})
        df["importance_norm"] = (df["mean_abs_shap"] / (df["mean_abs_shap"].max() + 1e-12))
        df = df.sort_values("importance_norm", ascending=False).reset_index(drop=True)
        return df
    except Exception as e:
        # Fallback: variance of model scores when zeroing one feature at a time (cheap proxy)
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
    Y_tr = tr[tgt_cols].values            # (n_samples, n_targets)
    last_feats = d[module.FEATURES].iloc[[-1]].values  # (1, n_features)
    return X_tr, Y_tr, last_feats


# ─────────────────────────────────────────────────────────────────────────────
# Backtests
# ─────────────────────────────────────────────────────────────────────────────

def backtest_submodels(
    df: pd.DataFrame,
    initial_frac: float = 0.6,
    window: int = 50,
) -> pd.DataFrame:
    """Walk-forward for all submodels; returns DataFrame with m*, acc*, meta_label."""
    submods = SUBMODS[:]  # momentum, trend, meanrev, lagged, sentiment, regressor, classifier
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

            elif hasattr(m, "compute_targets"):  # multi-horizon regression (one model, many targets)
                X_tr, Y_tr, last_feats = prepare_train_regression_multi(slice_df, m)
                if len(X_tr) < 50:
                    p = np.nan
                else:
                    model = m.fit(X_tr, Y_tr)
                    yhat_multi = m.predict(model, last_feats)[0]
                    idx = m.MULTI_HORIZONS.index(h)
                    p = float(yhat_multi[idx])
                lbl = 1 if df["close"].iat[t + h] > df["close"].iat[t] else 0

            else:  # legacy single-horizon regression-style
                d = m.compute_target(slice_df)
                tr  = d.iloc[:-1].dropna(subset=m.FEATURES + ["target"])
                last_feats = d[m.FEATURES].iloc[[-1]]
                if len(tr) < 50:
                    p = 0.0
                else:
                    model = m.fit(tr[m.FEATURES], tr["target"])
                    p = float(m.predict(model, last_feats)[0])
                lbl = 1 if d["target"].iloc[-1] > 0 else 0

            preds.append(p); labels.append(lbl)

        # Rolling accuracies: classifier via 0.5; regression via predicted level vs contemporaneous close
        accs = []
        for k in range(n_ch):
            preds_history[k].append(preds[k])
            labels_history[k].append(labels[k])
            ref_close_hist[k].append(float(df["close"].iat[t]))

            ph = np.asarray(preds_history[k][-window-1:-1], dtype=float)
            lh = np.asarray(labels_history[k][-window-1:-1], dtype=float)
            rc = np.asarray(ref_close_hist[k][-window-1:-1], dtype=float)

            if hasattr(channels[k]["mod"], "compute_labels"):
                pred_cls = (ph > 0.5).astype(float)
            else:
                with np.errstate(invalid='ignore'):
                    pred_cls = (ph > rc).astype(float)
            msk = np.isfinite(pred_cls)
            accs.append(float((pred_cls[msk] == lh[msk]).mean()) if msk.sum() else 0.5)

        rec_dict = {
            "t": t,
            **{f"m{k+1}":   preds[k] for k in range(n_ch)},
            **{f"acc{k+1}": accs[k]  for k in range(n_ch)},
            "meta_label": 1 if df["close"].iat[t + 1] > df["close"].iat[t] else 0,
        }
        rec.append(rec_dict)

    return pd.DataFrame(rec).set_index("t")

def walkforward_meta_backtest(cache_csv: str,
                              n_mods: int | None = None,
                              initial_frac: float = 0.6,
                              metric: str = "accuracy"):
    """
    Uses history CSV (m*, acc*, meta_label). Not used when QUICK_TEST=true.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST (no history; SHAP + accuracy + PnL)
# ─────────────────────────────────────────────────────────────────────────────

def _filter_channels_for_quick(submods, use_h1_only: bool = True):
    """
    Build channels but optionally keep only h=1 (or the smallest horizon) for multi-horizon modules.
    That keeps QUICK_TEST fast while still exercising each module.
    """
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
    # ---- quick knobs (env) ----
    quick_max_bars = 1500
    quick_stride   = 5
    quick_h1_only  = os.getenv("QUICK_H1_ONLY", "true").strip().lower() in {"1","true","yes","y"}

    # 1) Tail slice + stride
    df = df.copy()
    if len(df) > quick_max_bars:
        df = df.iloc[-quick_max_bars:].reset_index(drop=True)
    if quick_stride > 1:
        df = df.iloc[::quick_stride].reset_index(drop=True)

    # 2) Channels (optionally keep only the smallest horizon)
    channels = _filter_channels_for_quick(SUBMODS, use_h1_only=quick_h1_only)
    n_ch = len(channels)

    # 3) Global purge gap and split indices
    HMAX = max_lookahead([ch["mod"] for ch in channels])
    N = len(df)
    if N < max(200, 3 * (HMAX + 5)):
        raise ValueError(f"QUICK_TEST slice too small (N={N}, HMAX={HMAX}). Increase QUICK_MAX_BARS or reduce HMAX.")

    cut = int(N * initial_frac)
    # Purged split: [0, cut - HMAX) for train; [cut + HMAX, N - 1 - max(1,HMAX)] for validation rows
    train_end = max(0, cut - HMAX)
    val_start = min(N-1, cut + HMAX)
    val_end   = N - 1 - max(1, HMAX)  # ensure meta_label (t+1) exists and future labels are not peeked

    if val_end <= val_start or train_end < 50:
        raise ValueError("QUICK_TEST split degenerate. Try lowering HMAX or increasing QUICK_MAX_BARS.")

    tr_idx = np.arange(0, train_end, dtype=int)
    va_idx = np.arange(val_start, val_end + 1, dtype=int)

    # 4) Prepare meta_label (direction t+1) for the validation rows
    close = pd.to_numeric(df["close"], errors="coerce").values
    y_meta_full = (close[1:] > close[:-1]).astype(int)
    # y_meta aligned to t; exists for t in [0, N-2]. We will take y_meta for va_idx positions.
    y_meta_va = y_meta_full[va_idx]

    # 5) Build batched predictions per channel (no per-bar training)
    m_cols = {}
    acc_cols = {}
    for k, ch in enumerate(channels, start=1):
        m, h = ch["mod"], ch["h"]

        # Classifier-style modules
        if hasattr(m, "compute_labels"):
            d = m.compute_labels(df).reset_index(drop=True)
            feat_cols = m.FEATURES
            lbl_col = f"label_h{int(h)}" if (h is not None) else "label"

            # Train slice (drop NAs)
            tr = d.loc[tr_idx].dropna(subset=list(feat_cols) + [lbl_col])
            if len(tr) < 50:
                # fallback: constant 0.5 on val
                m_cols[f"m{k}"] = np.full(len(va_idx), 0.5, dtype=float)
                acc_cols[f"acc{k}"] = 0.5
                continue

            X_tr = tr[feat_cols].values
            y_tr = tr[lbl_col].astype(int).values

            # Fit once
            model = m.fit(X_tr, y_tr, horizon=h) if h is not None else m.fit(X_tr, y_tr)

            # Validation slice (keep alignment; predict only where features are finite)
            dv = d.loc[va_idx, feat_cols]
            mask = np.isfinite(dv.values).all(axis=1)
            preds = np.full(len(va_idx), np.nan, dtype=float)
            if mask.any():
                preds[mask] = m.predict(model, dv.loc[mask].values).astype(float)

            # Acc estimate on val (directional), using 0.5 threshold
            acc_mask = mask & np.isfinite(preds)
            if acc_mask.any():
                acc_val = ( (preds[acc_mask] > 0.5).astype(int) == y_meta_va[acc_mask] ).mean()
            else:
                acc_val = 0.5

            m_cols[f"m{k}"] = preds
            acc_cols[f"acc{k}"] = float(acc_val)

        # Multi-horizon regression modules (predict level or return)
        elif hasattr(m, "compute_targets"):
            d = m.compute_targets(df).reset_index(drop=True)
            feat_cols = m.FEATURES
            # choose this horizon's target column
            tgt_col = f"target_h{int(h)}" if (h is not None) else "target"

            # Train slice
            tr = d.loc[tr_idx].dropna(subset=list(feat_cols) + [tgt_col])
            if len(tr) < 50:
                m_cols[f"m{k}"] = np.full(len(va_idx), np.nan, dtype=float)
                acc_cols[f"acc{k}"] = 0.5
                continue

            X_tr = tr[feat_cols].values

            # Multi-target: fit on all targets then pick index of h
            if hasattr(m, "MULTI_HORIZONS") and (h is not None):
                # Build Y_tr as matrix over all chosen horizons, fit once
                tgt_cols_all = [f"target_h{int(hh)}" for hh in m.MULTI_HORIZONS]
                tr_all = d.loc[tr_idx].dropna(subset=list(feat_cols) + tgt_cols_all)
                X_tr = tr_all[feat_cols].values
                Y_tr = tr_all[tgt_cols_all].values
                model = m.fit(X_tr, Y_tr)
                dv = d.loc[va_idx, feat_cols]
                mask = np.isfinite(dv.values).all(axis=1)
                preds = np.full(len(va_idx), np.nan, dtype=float)
                if mask.any():
                    yhat_multi = m.predict(model, dv.loc[mask].values)
                    idx_h = m.MULTI_HORIZONS.index(h)
                    preds_level = np.asarray(yhat_multi)[:, idx_h].astype(float)
                    # Map to directional score via relative move vs current close
                    c0 = close[va_idx[mask]]
                    preds[mask] = (preds_level - c0) / c0  # a signed "return-like" score
                # Acc estimate (sign vs meta_label)
                acc_mask = np.isfinite(preds)
                if acc_mask.any():
                    acc_val = ((preds[acc_mask] > 0).astype(int) == y_meta_va[acc_mask]).mean()
                else:
                    acc_val = 0.5
                m_cols[f"m{k}"] = preds
                acc_cols[f"acc{k}"] = float(acc_val)
            else:
                # Legacy single-target regression
                y_tr = tr[tgt_col].values.astype(float)
                model = m.fit(X_tr, y_tr)
                dv = d.loc[va_idx, feat_cols]
                mask = np.isfinite(dv.values).all(axis=1)
                preds = np.full(len(va_idx), np.nan, dtype=float)
                if mask.any():
                    preds_level = m.predict(model, dv.loc[mask].values).astype(float)
                    c0 = close[va_idx[mask]]
                    preds[mask] = (preds_level - c0) / c0
                acc_mask = np.isfinite(preds)
                acc_val = ((preds[acc_mask] > 0).astype(int) == y_meta_va[acc_mask]).mean() if acc_mask.any() else 0.5
                m_cols[f"m{k}"] = preds
                acc_cols[f"acc{k}"] = float(acc_val)

        else:
            # Shouldn't happen with your current modules; safe default
            m_cols[f"m{k}"] = np.full(len(va_idx), np.nan, dtype=float)
            acc_cols[f"acc{k}"] = 0.5

    # 6) Assemble compact 'back' for the validation slice only
    back = pd.DataFrame({**m_cols, **{f"acc{i+1}": acc_cols[f"acc{i+1}"] for i in range(n_ch)}})
    back["meta_label"] = y_meta_va.astype(int)
    if "timestamp" in df.columns:
        back["timestamp"] = pd.to_datetime(df["timestamp"].values[va_idx], utc=True)

    # 7) Sub-model AUCs vs meta_label (on this val slice)
    sub_auc = {}
    for i in range(1, n_ch + 1):
        col = f"m{i}"
        x = pd.to_numeric(back[col], errors="coerce").values
        y = back["meta_label"].values
        mask = np.isfinite(x)
        auc = roc_auc_score(y[mask], x[mask]) if np.unique(y[mask]).size == 2 else np.nan
        sub_auc[col] = float(auc) if np.isfinite(auc) else np.nan

    # 8) Meta model on the validation slice (60/40 of 'back')
    #    (Keeps quick-test fast; still measures generalization inside val)
    back = back.reset_index(drop=True)
    n_mods = sum(1 for c in back.columns if isinstance(c, str) and c.startswith("m") and c[1:].isdigit())
    X = build_meta_features(back, n_mods=n_mods)
    y = back["meta_label"].values.astype(int)
    if len(y) < 100:
        raise ValueError("QUICK_TEST: validation slice too small for meta. Increase QUICK_MAX_BARS or decrease QUICK_STRIDE.")
    split_b = int(len(y) * 0.6)
    X_tr, y_tr = X[:split_b], y[:split_b]
    X_te, y_te = X[split_b:], y[split_b:]

    meta = make_meta_model().fit(X_tr, y_tr)
    p_tr = meta.predict_proba(X_tr)[:, 1]
    p_te = meta.predict_proba(X_te)[:, 1]

    up, down = optimize_asymmetric_thresholds(p_tr, y_tr, metric="profit")

    # 9) PnL on the meta actions for the test slice (map to original returns)
    # The test slice corresponds to va_idx[split_b:]
    t_idx = va_idx[split_b:]
    ret_next = (close[t_idx + 1] - close[t_idx]) / close[t_idx]
    actions = _actions_from_probs(p_te, up, down)
    pnl_vec = _pnl_from_actions(actions, ret_next)
    pnl_sum = float(np.nansum(pnl_vec))
    pnl_avg = float(np.nanmean(pnl_vec)) if len(pnl_vec) else np.nan

    meta_auc = roc_auc_score(y_te, p_te) if np.unique(y_te).size == 2 else np.nan
    from sklearn.metrics import accuracy_score
    meta_acc = accuracy_score(y_te, (p_te > 0.5).astype(int)) if len(y_te) else np.nan

    # 10) SHAP for the meta (test slice)
    feat_names = get_meta_feature_names(back, n_mods)
    shap_path = _save_shap_report(X_te, y_te, meta, feat_names, FEATURE_ANALYTICS_DIR, tag="quicktest_fast")

    summary = {
        "quick_max_bars": quick_max_bars,
        "quick_stride": quick_stride,
        "quick_h1_only": bool(quick_h1_only),
        "val_rows": int(len(va_idx)),
        "meta_train_rows": int(len(y_tr)),
        "meta_test_rows": int(len(y_te)),
        "meta_auc": float(meta_auc) if np.isfinite(meta_auc) else np.nan,
        "meta_acc@0.5": float(meta_acc) if np.isfinite(meta_acc) else np.nan,
        "threshold_up": float(up),
        "threshold_down": float(down),
        "test_pnl_sum": pnl_sum,
        "test_pnl_mean": pnl_avg,
        "shap_importance_csv": shap_path,
        "submodel_auc_vs_meta_label": sub_auc,
    }

    # Save compact JSON summary
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame([summary]).to_json(os.path.join(RESULTS_DIR, "quicktest_summary.json"), orient="records", indent=2)

    if return_df:
        out = pd.DataFrame({
            "timestamp": pd.to_datetime(df["timestamp"].values[t_idx + 1], utc=True) if "timestamp" in df.columns else np.arange(len(t_idx)),
            "meta_prob": p_te,
            "action": actions,
            "ret_next": ret_next,
            "pnl": pnl_vec
        })
        return summary, out

    return summary

# ─────────────────────────────────────────────────────────────────────────────
# History builder (angles removed)
# ─────────────────────────────────────────────────────────────────────────────

def update_submeta_history(CSV_PATH, HISTORY_PATH,
                           submods=SUBMODS, window=50, verbose=True):
    """
    Builds/updates history CSV with columns:
      timestamp, m1..mN, acc1..accN, meta_label
    (All angle features removed.)
    """
    df = pd.read_csv(CSV_PATH)
    if verbose:
        print("── DEBUG: CSV_PATH =", CSV_PATH)
        print("── DEBUG: columns in CSV:", df.columns.tolist())
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    channels = expand_channels(submods)
    n_ch = len(channels)

    col_m   = [f"m{i+1}"   for i in range(n_ch)]
    col_acc = [f"acc{i+1}" for i in range(n_ch)]

    hist = None
    if os.path.exists(HISTORY_PATH):
        tmp = pd.read_csv(HISTORY_PATH)
        expected_cols = ["timestamp", *col_m, *col_acc, "meta_label"]
        if list(tmp.columns) == expected_cols:
            hist = tmp
            hist["timestamp"] = pd.to_datetime(hist["timestamp"], utc=True)
        else:
            if verbose:
                print("[HISTORY] Schema changed; rebuilding history from scratch.")
            hist = None

    HMAX = max_lookahead(submods)
    n = len(df)
    usable_last_i = n - 1 - max(1, HMAX)
    if usable_last_i < 0:
        raise ValueError("Not enough rows to build history.")

    if hist is not None and len(hist) > 0:
        last_ts = hist["timestamp"].iloc[-1]
        future = df[df["timestamp"] > last_ts]
        if future.empty:
            if verbose: print("[HISTORY] Already up-to-date.")
            return hist
        start_idx = int(future.index.min())
    else:
        hist = pd.DataFrame(columns=["timestamp", *col_m, *col_acc, "meta_label"])
        start_idx = int(n * 0.6)

    start_idx = max(0, min(start_idx, usable_last_i))
    end_idx = usable_last_i  # inclusive

    preds_hist      = [[] for _ in range(n_ch)]
    labels_hist     = [[] for _ in range(n_ch)]
    ref_close_hist  = [[] for _ in range(n_ch)]

    recs = []

    for i in tqdm(range(start_idx, end_idx + 1), desc="History (submods)"):
        slc = df.iloc[: i + 1].reset_index(drop=True)

        preds, labels = [], []

        for ch in channels:
            m, h = ch["mod"], ch["h"]

            if hasattr(m, "compute_labels"):  # classifier (single or multi)
                d = m.compute_labels(slc)
                if is_multi(m):
                    X_tr, y_tr = prepare_train_for_horizon(slc, m, h)
                    if len(y_tr) < 50:
                        p = 0.5
                    else:
                        last_feats = d[m.FEATURES].iloc[[-1]]
                        model = m.fit(X_tr, y_tr, horizon=h)
                        p = m.predict(model, last_feats)[0]
                    lbl = int(df["close"].iat[i + h] > df["close"].iat[i])
                else:
                    X_tr, y_tr = prepare_train(slc, m)
                    if len(y_tr) < 50:
                        p = 0.5
                    else:
                        last_feats = d[m.FEATURES].iloc[[-1]]
                        model = m.fit(X_tr, y_tr)
                        p = m.predict(model, last_feats)[0]
                    lbl = int(df["close"].iat[i + 1] > df["close"].iat[i])

            elif hasattr(m, "compute_targets"):  # multi-horizon regression
                X_tr, Y_tr, last_feats = prepare_train_regression_multi(slc, m)
                if len(X_tr) < 50:
                    p = np.nan
                else:
                    model = m.fit(X_tr, Y_tr)
                    yhat_multi = m.predict(model, last_feats)[0]
                    idx = m.MULTI_HORIZONS.index(h)
                    p = float(yhat_multi[idx])
                lbl = int(df["close"].iat[i + h] > df["close"].iat[i])

            else:  # legacy single-horizon regression-style
                d = m.compute_target(slc)
                tr = d.iloc[:-1].dropna(subset=m.FEATURES + ["target"])
                last_feats = d[m.FEATURES].iloc[[-1]]
                if len(tr) < 50:
                    p = 0.0
                else:
                    model = m.fit(tr[m.FEATURES], tr["target"])
                    p = float(m.predict(model, last_feats)[0])
                lbl = int(d["target"].iloc[-1] > 0)

            preds.append(p)
            labels.append(lbl)

        # Rolling accuracies per channel
        accs = []
        window = 50
        for k in range(n_ch):
            preds_hist[k].append(preds[k])
            labels_hist[k].append(labels[k])
            ref_close_hist[k].append(float(df["close"].iat[i]))

            ph  = np.asarray(preds_hist[k][-window-1:-1], dtype=float)
            lh  = np.asarray(labels_hist[k][-window-1:-1], dtype=float)
            rc  = np.asarray(ref_close_hist[k][-window-1:-1], dtype=float)

            if hasattr(channels[k]["mod"], "compute_labels"):
                pred_cls = (ph > 0.5).astype(float)
            else:
                with np.errstate(invalid='ignore'):
                    pred_cls = (ph > rc).astype(float)
            msk = np.isfinite(pred_cls)
            accs.append(float((pred_cls[msk] == lh[msk]).mean()) if msk.sum() else 0.5)

        rec = {
            "timestamp": df["timestamp"].iat[i],
            **{f"m{k+1}":   preds[k] for k in range(n_ch)},
            **{f"acc{k+1}": accs[k]  for k in range(n_ch)},
            "meta_label":  int(df["close"].iat[i+1] > df["close"].iat[i]),
        }
        recs.append(rec)

        if verbose and (i - start_idx) % 10 == 0:
            print(f"[HISTORY] appended idx={i}")

    if recs:
        hist = pd.concat([hist, pd.DataFrame(recs)], ignore_index=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        hist.to_csv(HISTORY_PATH, index=False)
        if verbose: print(f"[HISTORY] Updated ⇒ {len(hist)} rows.")
    return hist


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration: backtest / live
# ─────────────────────────────────────────────────────────────────────────────

def train_and_save_meta(cache_csv: str,
                        model_pkl: str,
                        threshold_pkl: str,
                        metric: str = "accuracy",
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

    # SHAP importance for meta (validation slice)
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
        result = quick_test(df_orig, initial_frac=0.6, return_df=return_df)
        if return_df:
            summary, df_actions = result
            print("[QUICK_TEST] Summary:", summary)
            return df_actions
        else:
            print("[QUICK_TEST] Summary:", result)
            return

    print("Quick Test is off: ", False)

    # ── Non-quick paths below (unchanged) ──
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
            # optional: SHAP on snapshot (unchanged)
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
                back[vote_col] = back[pred_col].map(lambda r: 1 if r > REG_UP else (0 if r < REG_DOWN else 0.5))
            vote_cols.append(vote_col)

        majority_req = (n_ch // 2) + 1
        back["buy_votes"]  = (back[vote_cols] == 1).sum(axis=1)
        back["sell_votes"] = (back[vote_cols] == 0).sum(axis=1)
        back["action"] = back.apply(
            lambda r: 1 if r.buy_votes  >= majority_req else
                      (0 if r.sell_votes >= majority_req else 0.5),
            axis=1,
        )

        back = back.reset_index().rename(columns={"index": "t"})
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
            metric="accuracy",
        )
        out_fn = os.path.join(RESULTS_DIR, f"meta_results_walkforward.csv")
        df_out.to_csv(out_fn, index=False, float_format="%.8f")
        print(f"[RESULT] Walk-forward results saved to {out_fn}")

        try:
            meta_pkl   = os.path.join(RESULTS_DIR, "meta_model.pkl")
            thresh_pkl = os.path.join(RESULTS_DIR, "meta_thresholds.pkl")
            meta, up, down = train_and_save_meta(cache_csv, meta_pkl, thresh_pkl, metric="accuracy", n_mods=None)
            print(f"[META] BUY>{up:.2f} / SELL<{down:.2f}")
        except Exception as e:
            print(f"[META] training/report failed: {e}")


def row(df_slice: pd.DataFrame, module, idx: int | None = None):
    d = (
        module.compute_labels(df_slice)
        if hasattr(module, "compute_labels")
        else module.compute_target(df_slice)
    )
    if idx is not None and idx in d.index:
        return d.loc[[idx], module.FEATURES]
    return d[module.FEATURES].iloc[[-1]]


def run_live(return_result: bool = True, position_open: bool = False):
    # Build channels
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
            elif hasattr(m, "compute_targets"):  # multi-horizon regression
                X_tr, Y_tr, last_feats = prepare_train_regression_multi(df, m)
                if len(X_tr) < 50:
                    preds.append(np.nan); continue
                model = m.fit(X_tr, Y_tr)
                yhat_multi = m.predict(model, last_feats)[0]
                idx = m.MULTI_HORIZONS.index(h)
                preds.append(float(yhat_multi[idx]))
            else:
                X_tr, y_tr = prepare_train(df, m)
                model = m.fit(X_tr, y_tr)
                feats = row(df, m, t)
                preds.append(m.predict(model, feats)[0])

        votes = []
        close_t = float(df["close"].iat[t])
        for k, ch in enumerate(channels):
            m = ch["mod"]
            if hasattr(m, "compute_labels"):
                votes.append(1 if preds[k] > VOTE_UP else (0 if preds[k] < VOTE_DOWN else 0.5))
            elif hasattr(m, "compute_targets"):
                if not np.isfinite(preds[k]):
                    votes.append(0.5)
                else:
                    rel = (preds[k] - close_t) / close_t
                    votes.append(1 if rel > REG_UP else (0 if rel < REG_DOWN else 0.5))
            else:
                votes.append(1 if preds[k] > REG_UP else (0 if preds[k] < REG_DOWN else 0.5))

        buy, sell = votes.count(1), votes.count(0)
        majority_req = (n_ch // 2) + 1
        majority = 1 if buy >= majority_req else (0 if sell >= majority_req else 0.5)
        action = "BUY" if majority == 1 else ("SELL" if majority == 0 else "HOLD")
        action = enforce_position_rules([action], position_open)[0]
        print(f"[LIVE-VOTE] result = {action}")
        return action if return_result else None

    # ——— META MODE ———
    # (Live path still uses history; QUICK_TEST is for backtesting only.)
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

    # Build current m*/acc* row (no angles)
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
                current_probs.append(float(p))
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
        else:
            X_tr, y_tr = prepare_train(df, m)
            model = m.fit(X_tr, y_tr)
            feats = row(df, m, t)
            p = float(m.predict(model, feats)[0])
            current_probs.append(p)

    k_hist = sum(1 for c in hist.columns if isinstance(c, str) and c.startswith("m") and c[1:].isdigit())
    last_accs = [float(hist[f"acc{i+1}"].iloc[-1]) if f"acc{i+1}" in hist.columns else 0.5 for i in range(k_hist)]

    row_df = pd.DataFrame([{**{f"m{i+1}": current_probs[i] for i in range(k_hist)},
                            **{f"acc{i+1}": last_accs[i] for i in range(k_hist)}}])

    feat_vec = build_meta_features(row_df, n_mods=k_hist)
    prob = float(meta.predict_proba(feat_vec)[0, 1])
    action = "BUY" if prob > up else ("SELL" if prob < down else "HOLD")
    action = enforce_position_rules([action], position_open)[0]
    print(f"[LIVE] P(up)={prob:.3f}  →  {action}  (BUY>{up:.2f} / SELL<{down:.2f})")

    return (prob, action) if return_result else None
