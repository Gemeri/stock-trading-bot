import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

from typing import Tuple


from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import brier_score_loss
import sub.momentum as mod1
import sub.trend_continuation as mod2
import sub.mean_reversion as mod3
import sub.lagged_return as mod4
import sub.sentiment_volume as mod5
import sub.regressor as modR

USE_CLASSIFIER = True
if USE_CLASSIFIER:
    import sub.classifier as mod6
    SUBMODS = [mod1, mod2, mod3, mod4, mod5, modR, mod6]
else:
    SUBMODS = [mod1, mod2, mod3, mod4, mod5, modR]

BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
RESULTS_DIR = os.path.join(BASE_DIR, "sub-results")

load_dotenv()
BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "4Hour")
TICKERS = os.getenv("TICKERS", "TSLA").split(",")
ML_MODEL = os.getenv("ML_MODEL",  "sub-vote")
TIMEFRAME_MAP = {"4Hour":"H4","2Hour":"H2","1Hour":"H1","30Min":"M30","15Min":"M15"}
CONVERTED_TIMEFRAME = TIMEFRAME_MAP.get(BAR_TIMEFRAME, BAR_TIMEFRAME)
HISTORY_PATH = os.path.join(RESULTS_DIR, f"submeta_history_{CONVERTED_TIMEFRAME}.csv")
USE_META_LABEL = os.getenv("USE_META_LABEL", "false")
FEATURE_ANALYTICS_DIR = os.path.join(RESULTS_DIR, "feature-analytics")

TICKERS_STR = "-".join([t.strip() for t in TICKERS if t.strip()]) or "TSLA"
CSV_PATH  = os.path.join(ROOT_DIR, f"{TICKERS_STR}_{CONVERTED_TIMEFRAME}.csv")
MODE = ML_MODEL  # "sub-vote" or "sub-meta"
EXECUTION = globals().get("EXECUTION", "backtest")

REG_UP, REG_DOWN = 0.003, -0.003

FORCE_STATIC_THRESHOLDS = False
STATIC_UP = 0.55
STATIC_DOWN = 0.45

META_MODEL_TYPE = os.getenv("META_MODEL_TYPE", "logreg")
VALID_META_MODELS = {"logreg", "lgbm", "xgb", "nn", "cat"}
if META_MODEL_TYPE not in VALID_META_MODELS:
    raise ValueError(f"META_MODEL_TYPE must be one of {VALID_META_MODELS}")


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

def _linear_angle_generic(y_vals, x_vals=None):
    import math
    y = np.asarray(y_vals, dtype=float)
    if x_vals is None:
        x = np.arange(1, len(y) + 1, dtype=float)
    else:
        x = np.asarray(x_vals, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float('nan')
    slope, _ = np.polyfit(x[mask], y[mask], 1)
    return float(np.degrees(np.arctan(slope)))

def get_meta_feature_names(hist: pd.DataFrame, n_mods: int | None = None) -> list[str]:
    """
    Reconstruct the feature names used by build_meta_features(hist, n_mods).
    Order must exactly match build_meta_features stacking:
      [m1..mN, acc1..accN, mean_p, var_p, wavg_p, feat_*...]
    """
    if n_mods is None:
        n_mods = sum(1 for c in hist.columns if isinstance(c, str) and c.startswith("m") and c[1:].isdigit())
    prob_cols = [f"m{i+1}"   for i in range(n_mods)]
    acc_cols  = [f"acc{i+1}" for i in range(n_mods)]
    extras    = [c for c in hist.columns if isinstance(c, str) and c.startswith("feat_")]
    # build_meta_features appends mean/var/wavg derived from the m* block
    return [*prob_cols, *acc_cols, "mean_p", "var_p", "wavg_p", *extras]


def _avg_coefficients_from_calibrated(meta) -> np.ndarray | None:
    """
    Try to pull averaged linear coefficients out of CalibratedClassifierCV if possible.
    Returns None if unavailable.
    """
    try:
        ccs = getattr(meta, "calibrated_classifiers_", None)
        if not ccs:
            return None
        coefs = []
        for cc in ccs:
            base = getattr(cc, "base_estimator", None)
            if base is None:
                continue
            if hasattr(base, "coef_"):
                coefs.append(np.ravel(base.coef_))
            elif hasattr(base, "feature_importances_"):
                coefs.append(np.ravel(base.feature_importances_))
        if coefs:
            M = np.vstack(coefs)
            return M.mean(axis=0)
    except Exception:
        pass
    return None


def compute_model_feature_importance(meta, X_val: np.ndarray, y_val: np.ndarray,
                                     feature_names: list[str],
                                     random_state: int = 42) -> pd.DataFrame:
    """
    Try multiple routes for importance:
      1) Native .feature_importances_ (tree models)
      2) Averaged linear coefficients (Calibrated + Logistic)
      3) Permutation importance (fallback, model-agnostic)
    Returns a DataFrame with normalized scores.
    """
    importances = None
    method = None

    # 1) direct attribute
    try:
        if hasattr(meta, "feature_importances_"):
            vals = np.asarray(meta.feature_importances_, dtype=float)
            if vals.shape[0] == X_val.shape[1]:
                importances = vals
                method = "model_feature_importances_"
    except Exception:
        pass

    # 2) averaged coefficients from calibrations
    if importances is None:
        coefs = _avg_coefficients_from_calibrated(meta)
        if coefs is not None and coefs.shape[0] == X_val.shape[1]:
            importances = np.abs(coefs)  # magnitude as importance
            method = "avg_linear_coefficients"

    # 3) permutation fallback
    if importances is None:
        try:
            perm = permutation_importance(meta, X_val, y_val, n_repeats=10, random_state=random_state, n_jobs=-1)
            importances = perm.importances_mean
            method = "permutation_importance"
        except Exception:
            # Last resort: zero vector
            importances = np.zeros(X_val.shape[1], dtype=float)
            method = "unavailable"

    imp = pd.DataFrame({
        "feature": feature_names,
        "importance_raw": importances
    })
    # Normalize to [0,1] for readability
    if imp["importance_raw"].max() > 0:
        imp["importance_norm"] = (imp["importance_raw"] - imp["importance_raw"].min()) / (imp["importance_raw"].max() - imp["importance_raw"].min())
    else:
        imp["importance_norm"] = 0.0
    imp["method"] = method
    imp = imp.sort_values("importance_norm", ascending=False).reset_index(drop=True)
    return imp



def build_meta_features(df: pd.DataFrame, n_mods: int | None = None) -> np.ndarray:
    if n_mods is None:
        n_mods = sum(1 for c in df.columns if isinstance(c, str) and c.startswith("m") and c[1:].isdigit())

    prob_cols = [f"m{i+1}"   for i in range(n_mods)]
    acc_cols  = [f"acc{i+1}" for i in range(n_mods)]

    P = df[prob_cols].values if prob_cols and set(prob_cols).issubset(df.columns) else np.zeros((len(df), 0))
    A = df[acc_cols].values  if acc_cols  and set(acc_cols ).issubset(df.columns) else np.zeros((len(df), 0))

    # Basic stats over the available m* columns
    if P.shape[1] > 0:
        mean_p = P.mean(axis=1, keepdims=True)
        var_p  = P.var(axis=1,  keepdims=True)
        # Weighted average over m's by their accuracies when available
        if A.shape[1] == P.shape[1] and A.shape[1] > 0:
            wavg_p = (P * A).sum(axis=1, keepdims=True) / (A.sum(axis=1, keepdims=True) + 1e-6)
        else:
            wavg_p = mean_p
    else:
        mean_p = var_p = wavg_p = np.zeros((len(df), 1))

    # ⬇️ NEW: any engineered extras 'feat_*' (e.g., angles) are appended verbatim
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
            random_state=42, verbosity=-1
        )
    elif META_MODEL_TYPE == "xgb":
        from xgboost import XGBClassifier
        base = XGBClassifier(
            n_estimators=500, learning_rate=0.03, max_depth=4,
            subsample=0.8, colsample_bytree=0.9,
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


def update_submeta_history(CSV_PATH, HISTORY_PATH,
                           submods=SUBMODS, window=50, verbose=True):

    df = pd.read_csv(CSV_PATH)
    if verbose:
        print("── DEBUG: CSV_PATH =", CSV_PATH)
        print("── DEBUG: columns in CSV:", df.columns.tolist())
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    channels = []
    for m in submods:
        if is_multi(m):
            for h in m.MULTI_HORIZONS:
                channels.append({"mod": m, "h": int(h)})
        else:
            channels.append({"mod": m, "h": None})
    n_ch = len(channels)

    col_m   = [f"m{i+1}"   for i in range(n_ch)]
    col_acc = [f"acc{i+1}" for i in range(n_ch)]

    # ⬇️ NEW: engineered features (angles) — no accuracy companions
    engineered_cols = ["feat_angle_clf", "feat_angle_reg", "feat_angle_base"]

    hist = None
    if os.path.exists(HISTORY_PATH):
        tmp = pd.read_csv(HISTORY_PATH)
        expected_cols = ["timestamp", *col_m, *col_acc, *engineered_cols, "meta_label"]
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
        hist = pd.DataFrame(columns=["timestamp", *col_m, *col_acc, *engineered_cols, "meta_label"])
        start_idx = int(n * 0.6)

    start_idx = max(0, min(start_idx, usable_last_i))
    end_idx = usable_last_i  # inclusive

    preds_hist      = [[] for _ in range(n_ch)]
    labels_hist     = [[] for _ in range(n_ch)]
    ref_close_hist  = [[] for _ in range(n_ch)]

    ch_kinds = []
    for ch in channels:
        m, h = ch["mod"], ch["h"]
        if hasattr(m, "compute_labels"):
            ch_kinds.append("clf")
        elif hasattr(m, "compute_targets"):
            ch_kinds.append("reg_multi")
        else:
            ch_kinds.append("reg_single")

    recs = []

    for i in tqdm(range(start_idx, end_idx + 1), desc="History (submods)"):
        slc = df.iloc[: i + 1].reset_index(drop=True)

        preds, labels = [], []

        clf_by_h = {}
        reg_by_h = {}
        base5_vals = []  # momentum, trend_continuation, mean_reversion, lagged_return, sentiment_volume

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
                    # angle collector for classifier
                    if m is mod6:
                        clf_by_h[h] = float(p)
                else:
                    X_tr, y_tr = prepare_train(slc, m)
                    if len(y_tr) < 50:
                        p = 0.5
                    else:
                        last_feats = d[m.FEATURES].iloc[[-1]]
                        model = m.fit(X_tr, y_tr)
                        p = m.predict(model, last_feats)[0]
                    lbl = int(df["close"].iat[i + 1] > df["close"].iat[i])

            elif hasattr(m, "compute_targets"):  # multi-horizon regression: train once on ALL horizons
                if is_multi(m):
                    X_tr, Y_tr, last_feats = prepare_train_regression_multi(slc, m)
                    if len(X_tr) < 50:
                        p = np.nan
                    else:
                        model = m.fit(X_tr, Y_tr)  # ← no horizon kwarg
                        yhat_multi = m.predict(model, last_feats)[0]  # shape (n_targets,)
                        idx = m.MULTI_HORIZONS.index(h)
                        p = float(yhat_multi[idx])  # predicted close_{t+h}
                    lbl = int(df["close"].iat[i + h] > df["close"].iat[i])
                    if m is modR:
                        reg_by_h[h] = p
                else:
                    d = m.compute_targets(slc)
                    tr = d.iloc[:-1].dropna(subset=m.FEATURES + ["target"])
                    last_feats = d[m.FEATURES].iloc[[-1]]
                    if len(tr) < 50:
                        p = np.nan
                    else:
                        model = m.fit(tr[m.FEATURES].values, tr["target"].values)
                        p = float(m.predict(model, last_feats)[0])
                    lbl = int(p > 0)

            else:  # legacy single-horizon regression-style (compute_target)
                d = m.compute_target(slc)
                tr = d.iloc[:-1].dropna(subset=m.FEATURES + ["target"])
                last_feats = d[m.FEATURES].iloc[[-1]]
                if len(tr) < 50:
                    p = 0.0
                else:
                    model = m.fit(tr[m.FEATURES], tr["target"])
                    p = float(m.predict(model, last_feats)[0])
                lbl = int(d["target"].iloc[-1] > 0)

            # collect “base 5” values for angle (mods 1..5 only)
            if m in (mod1, mod2, mod3, mod4, mod5) and h is None:
                base5_vals.append(float(p))

            preds.append(p)
            labels.append(lbl)

        # Rolling accuracies per channel (classifier uses 0.5 threshold, regression uses direction vs close)
        accs = []
        for k in range(n_ch):
            preds_hist[k].append(preds[k])
            labels_hist[k].append(labels[k])
            ref_close_hist[k].append(float(df["close"].iat[i]))

            ph  = np.asarray(preds_hist[k][-window-1:-1], dtype=float)
            lh  = np.asarray(labels_hist[k][-window-1:-1], dtype=float)
            rch = np.asarray(ref_close_hist[k][-window-1:-1], dtype=float)

            if len(ph) == 0:
                accs.append(0.5)
                continue

            kind = ch_kinds[k]
            if kind == "clf":
                pred_cls = (ph > 0.5).astype(float)
            else:
                # For regression, infer direction by comparing predicted level vs contemporaneous close
                # (If NaN predictions exist, mark as NaN)
                with np.errstate(invalid='ignore'):
                    pred_cls = (ph > rch).astype(float)
            mask = np.isfinite(pred_cls)
            if mask.sum() == 0:
                accs.append(0.5)
            else:
                accs.append(float((pred_cls[mask] == lh[mask]).mean()))

        # ⬇️ Build engineered angle features
        angle_clf  = float(mod6.linear_angle([clf_by_h[h]  for h in getattr(mod6, "MULTI_HORIZONS", []) if h in clf_by_h])) if 'mod6' in globals() and len(clf_by_h) >= 2 else float('nan')
        angle_reg  = float(modR.linear_angle([reg_by_h[h]  for h in getattr(modR, "MULTI_HORIZONS", []) if h in reg_by_h])) if len(reg_by_h) >= 2 else float('nan')
        angle_base = float(_linear_angle_generic(base5_vals)) if len(base5_vals) >= 2 else float('nan')

        rec = {
            "timestamp": df["timestamp"].iat[i],
            **{col_m[k]:   preds[k] for k in range(n_ch)},
            **{col_acc[k]: accs[k]  for k in range(n_ch)},
            "feat_angle_clf":  angle_clf,
            "feat_angle_reg":  angle_reg,
            "feat_angle_base": angle_base,
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

def _normalize_feature_for_thresholding(x: np.ndarray) -> np.ndarray:
    """Robustly map a numeric vector to ~[0,1] for threshold optimization."""
    x = np.asarray(x, dtype=float)
    m1, m99 = np.nanpercentile(x, 1), np.nanpercentile(x, 99)
    if not np.isfinite(m1) or not np.isfinite(m99) or m99 <= m1:
        m1, m99 = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(m1) or not np.isfinite(m99) or m99 <= m1:
        return np.zeros_like(x) + 0.5
    z = (x - m1) / (m99 - m1)
    return np.clip(z, 0, 1)


def _rolling_auc(y_true: np.ndarray, score: np.ndarray, window: int) -> np.ndarray:
    vals = []
    for i in range(window, len(y_true)+1):
        y_win = y_true[i-window:i]
        s_win = score[i-window:i]
        if len(np.unique(y_win)) < 2:
            vals.append(np.nan)
            continue
        try:
            vals.append(roc_auc_score(y_win, s_win))
        except Exception:
            vals.append(np.nan)
    return np.array(vals, dtype=float)


def evaluate_raw_feature_usefulness(hist: pd.DataFrame,
                                    label_col: str = "meta_label",
                                    min_samples: int = 200) -> pd.DataFrame:
    """
    Score columns present in history CSV ('m*', 'acc*', any 'feat_*') against the future direction label.
    Produces a ranked DataFrame with many metrics and a composite score.
    """
    cols = [c for c in hist.columns if (
        (isinstance(c, str) and (c.startswith("m") and c[1:].isdigit())) or
        (isinstance(c, str) and c.startswith("acc")) or
        (isinstance(c, str) and c.startswith("feat_"))
    )]
    y = hist[label_col].astype(int).values
    out_rows = []

    for c in cols:
        x_raw = pd.to_numeric(hist[c], errors="coerce").values
        mask = np.isfinite(x_raw) & np.isfinite(y)
        if mask.sum() < min_samples:
            continue

        x = x_raw[mask]
        yy = y[mask]

        # Normalize for thresholding + AUC
        x01 = _normalize_feature_for_thresholding(x)

        # Optimize thresholds (profit-oriented) and evaluate
        up_p, dn_p = optimize_asymmetric_thresholds(x01, yy, metric="profit")
        trigger = (x01 > up_p) | (x01 < dn_p)
        coverage = trigger.mean()

        if trigger.sum() >= 30:
            acc_trig = (yy[trigger] == (x01[trigger] > up_p)).mean()
            pnl = np.where(
                x01[trigger] > up_p,  (yy[trigger] == 1).astype(int),
                - (yy[trigger] == 0).astype(int)
            ).mean()
        else:
            acc_trig, pnl = np.nan, np.nan

        # Directional ROC AUC
        try:
            auc = roc_auc_score(yy, x01)
        except Exception:
            auc = np.nan

        # Mutual information
        try:
            mi = float(mutual_info_classif(x01.reshape(-1, 1), yy, random_state=42))
        except Exception:
            mi = np.nan

        # Calibration skill (only meaningful when x01 ~ prob). Use Brier Skill vs climatology.
        try:
            p_hat = x01
            baseline = yy.mean()
            brier = brier_score_loss(yy, p_hat)
            brier_ref = brier_score_loss(yy, np.full_like(p_hat, baseline, dtype=float))
            bss = 1.0 - (brier / (brier_ref + 1e-12))
        except Exception:
            bss = np.nan

        # Rolling AUC -> learning trend + stability
        W = max(100, int(0.2 * len(yy)))
        rauc = _rolling_auc(yy, x01, window=W)
        if np.isfinite(rauc).sum() >= 5:
            # slope of AUC over time (higher => improving)
            idx = np.arange(len(rauc), dtype=float)
            msk = np.isfinite(rauc)
            if msk.sum() >= 5:
                slope, _ = np.polyfit(idx[msk], rauc[msk], 1)
                learning_trend = float(slope)
                stab = float(np.nanstd(rauc))
                recent_auc = float(np.nanmean(rauc[-max(5, len(rauc)//5):]))
            else:
                learning_trend, stab, recent_auc = np.nan, np.nan, np.nan
        else:
            learning_trend, stab, recent_auc = np.nan, np.nan, np.nan

        # Linear + rank correlations with label (IC)
        try:
            pear = float(np.corrcoef(x01, yy)[0, 1])
        except Exception:
            pear = np.nan
        try:
            from scipy.stats import spearmanr
            spear = float(spearmanr(x01, yy, nan_policy="omit").correlation)
        except Exception:
            spear = np.nan

        out_rows.append({
            "feature": c,
            "auc": auc,
            "mutual_info": mi,
            "brier_skill": bss,
            "trigger_coverage": coverage,
            "trigger_accuracy": acc_trig,
            "profit_like": pnl,
            "pearson_ic": pear,
            "spearman_ic": spear,
            "learning_trend": learning_trend,
            "stability_std": stab,
            "recent_auc": recent_auc,
            "up_thresh": up_p,
            "down_thresh": dn_p,
            "n": int(mask.sum()),
        })

    if not out_rows:
        return pd.DataFrame()

    dfu = pd.DataFrame(out_rows)

    # Normalize/compose a single composite score
    def _minmax(s):
        if s.notna().any():
            lo, hi = s.min(), s.max()
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                return (s - lo) / (hi - lo)
        return pd.Series(np.where(np.isnan(s), np.nan, 0.0), index=s.index)

    # Higher is better: auc, mutual_info, brier_skill, trigger_accuracy, profit_like, recent_auc, learning_trend (positive), (1 - stability)
    s_auc   = _minmax(dfu["auc"])
    s_mi    = _minmax(dfu["mutual_info"])
    s_bss   = _minmax(dfu["brier_skill"])
    s_acc   = _minmax(dfu["trigger_accuracy"])
    s_pnl   = _minmax(dfu["profit_like"])
    s_rec   = _minmax(dfu["recent_auc"])
    s_lrn   = _minmax(dfu["learning_trend"])
    s_cov   = _minmax(dfu["trigger_coverage"])
    s_stab  = _minmax(-dfu["stability_std"])  # lower std is better

    # Weighted blend (tune as needed)
    composite = (
        0.30 * s_auc.fillna(0)   +
        0.20 * s_acc.fillna(0)   +
        0.20 * s_pnl.fillna(0)   +
        0.10 * s_bss.fillna(0)   +
        0.08 * s_lrn.fillna(0)   +
        0.06 * s_stab.fillna(0)  +
        0.04 * s_mi.fillna(0)    +
        0.02 * s_rec.fillna(0)
        # coverage intentionally small; acts as a tie-breaker
        + 0.00 * s_cov.fillna(0)
    )

    dfu["composite_score"] = composite
    dfu = dfu.sort_values(["composite_score", "auc", "trigger_accuracy", "profit_like"], ascending=False).reset_index(drop=True)
    return dfu

def write_feature_usefulness_reports(history_csv: str,
                                     out_dir: str = FEATURE_ANALYTICS_DIR,
                                     meta_model=None,
                                     n_mods: int | None = None) -> dict:
    """
    Loads history CSV, computes:
      A) Per-column usefulness (raw history cols) → ranked table
      B) Meta-model feature importance on build_meta_features() inputs
    Writes CSVs into out_dir and returns paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")

    hist = pd.read_csv(history_csv)
    if "meta_label" not in hist.columns or len(hist) < 300:
        return {"skipped": True, "reason": "insufficient_history"}

    # A) Raw column usefulness
    raw_rank = evaluate_raw_feature_usefulness(hist)
    raw_path_ts = os.path.join(out_dir, f"raw_feature_usefulness_{ts}.csv")
    raw_path_latest = os.path.join(out_dir, "raw_feature_usefulness_latest.csv")
    raw_rank.to_csv(raw_path_ts, index=False, float_format="%.6f")
    raw_rank.to_csv(raw_path_latest, index=False, float_format="%.6f")

    # B) Meta-model input importance (only if model provided)
    meta_paths = {}
    try:
        if meta_model is not None:
            # Construct X,y exactly like training
            X = build_meta_features(hist, n_mods)
            y = hist["meta_label"].values.astype(int)
            split = int(len(y) * 0.6)
            X_val, y_val = X[split:], y[split:]
            if len(y_val) >= 100:
                feat_names = get_meta_feature_names(hist, n_mods)
                meta_imp = compute_model_feature_importance(meta_model, X_val, y_val, feat_names)
                meta_ts = os.path.join(out_dir, f"meta_feature_importance_{ts}.csv")
                meta_latest = os.path.join(out_dir, "meta_feature_importance_latest.csv")
                meta_imp.to_csv(meta_ts, index=False, float_format="%.6f")
                meta_imp.to_csv(meta_latest, index=False, float_format="%.6f")
                meta_paths = {"meta_ts": meta_ts, "meta_latest": meta_latest}
    except Exception as e:
        # don't fail the run if importance can't be computed
        meta_paths = {"meta_error": str(e)}

    # Small JSON-like summary
    summary = {
        "timestamp_utc": ts,
        "raw_usefulness_csv": raw_path_ts,
        "raw_usefulness_latest": raw_path_latest,
        **meta_paths
    }
    with open(os.path.join(out_dir, "summary_latest.txt"), "w") as f:
        f.write(str(summary))
    return summary


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
                from sklearn.metrics import roc_auc_score
                score = roc_auc_score(labels[mask], probs[mask])

            if score > best_sc:
                best_sc, best_up, best_dn = score, up, dn

    return best_up, best_dn


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

    # ⬇️ NEW: write feature usefulness + meta importance reports
    try:
        os.makedirs(FEATURE_ANALYTICS_DIR, exist_ok=True)
        write_feature_usefulness_reports(cache_csv, out_dir=FEATURE_ANALYTICS_DIR, meta_model=meta, n_mods=n_mods)
    except Exception as e:
        print(f"[FEATURE-ANALYTICS] Failed to write reports: {e}")

    return meta, up, down

def backtest_submodels(
    df: pd.DataFrame,
    initial_frac: float = 0.6,
    window: int = 50,
) -> pd.DataFrame:
# in run_backtest()
    submods = [mod1, mod2, mod3, mod4, mod5, modR] + ([mod6] if USE_CLASSIFIER else [])
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
            "timestamp":  hist.at[i, "timestamp"],
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

    # include regressor + optional classifier
    submods = [mod1, mod2, mod3, mod4, mod5, modR] + ([mod6] if USE_CLASSIFIER else [])
    channels = expand_channels(submods)
    n_ch = len(channels)

    # ensure analytics dir exists + a timestamp for snapshots
    os.makedirs(FEATURE_ANALYTICS_DIR, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")

    if MODE == "sub-vote":
        # 1) run submodel backtest (produces m*, acc*, meta_label)
        back = backtest_submodels(df_orig, initial_frac=0.6)

        # 2) save a "history-like" snapshot for analytics (timestamped and latest)
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
            # Run analytics on this snapshot (raw usefulness only; no meta model in vote mode)
            write_feature_usefulness_reports(bt_hist_latest, out_dir=FEATURE_ANALYTICS_DIR, meta_model=None)
        except Exception as e:
            print(f"[FEATURE-ANALYTICS] (sub-vote backtest) Failed to write reports: {e}")

        # 3) voting logic → df_out (unchanged)
        VOTE_UP, VOTE_DOWN = 0.55, 0.45

        vote_cols = []
        for k, ch in enumerate(channels, start=1):
            pred_col, vote_col = f"m{k}", f"v{k}"
            m = ch["mod"]
            if hasattr(m, "compute_labels"):          # classifier style
                back[vote_col] = back[pred_col].map(lambda p: 1 if p > VOTE_UP else (0 if p < VOTE_DOWN else 0.5))
            else:                                     # regression style
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

    else:
        # META backtest path:
        cache_csv = os.path.join(RESULTS_DIR, f"submeta_history_{CONVERTED_TIMEFRAME}.csv")
        os.makedirs(RESULTS_DIR, exist_ok=True)

        # 1) build/refresh history
        update_submeta_history(CSV_PATH, cache_csv, submods=submods, verbose=True)

        # 2) fit a temporary meta model ONLY for importance reporting (doesn't affect BT results)
        try:
            hist = pd.read_csv(cache_csv)
            meta_tmp = None
            if "meta_label" in hist.columns and len(hist) >= 300:
                X = build_meta_features(hist, n_mods=None)
                y = hist["meta_label"].values.astype(int)
                split = int(len(y) * 0.6)
                if split >= 100:
                    meta_tmp = make_meta_model().fit(X[:split], y[:split])
            write_feature_usefulness_reports(cache_csv, out_dir=FEATURE_ANALYTICS_DIR, meta_model=meta_tmp)
        except Exception as e:
            print(f"[FEATURE-ANALYTICS] (meta backtest) Failed to write reports: {e}")

        # 3) run the walk-forward meta backtest (unchanged)
        df_out = walkforward_meta_backtest(
            cache_csv=cache_csv,
            n_mods=None,          # infer from history columns
            initial_frac=0.6,
            metric="accuracy",
        )
        out_fn = os.path.join(RESULTS_DIR, f"meta_results_walkforward.csv")
        df_out.to_csv(out_fn, index=False, float_format="%.8f")
        print(f"[RESULT] Walk-forward results saved to {out_fn}")

    if return_df:
        df_out["action"] = enforce_position_rules(df_out["action"].tolist(), start_open)
        return df_out


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
    # ⬇️ UPDATED: include regressor module
    submods = [mod1, mod2, mod3, mod4, mod5, modR]
    if "USE_CLASSIFIER" in globals() and USE_CLASSIFIER:
        submods.append(mod6)
    channels = []
    for m in submods:
        if is_multi(m):
            for h in m.MULTI_HORIZONS:
                channels.append({"mod": m, "h": int(h)})
        else:
            channels.append({"mod": m, "h": None})
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

        # ⬇️ NEW: keep feature analytics fresh even in vote mode (raw CSV only)
        try:
            write_feature_usefulness_reports(HISTORY_PATH, out_dir=FEATURE_ANALYTICS_DIR, meta_model=None)
        except Exception as e:
            print(f"[FEATURE-ANALYTICS] (vote) Failed to write reports: {e}")

        return action if return_result else None

    # ——— META MODE ———
    hist = update_submeta_history(CSV_PATH, HISTORY_PATH, submods=submods, verbose=True)

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
    else:
        # ⬇️ NEW: even when reusing cache, refresh reports with the loaded model
        try:
            write_feature_usefulness_reports(HISTORY_PATH, out_dir=FEATURE_ANALYTICS_DIR, meta_model=meta)
        except Exception as e:
            print(f"[FEATURE-ANALYTICS] Failed to write reports: {e}")

    # Build the current m*/acc* row + engineered angles
    current_probs = []
    clf_by_h, reg_by_h, base5_vals = {}, {}, []

    for ch in channels:
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
                if m is mod6:
                    clf_by_h[h] = float(p)
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
            if m is modR:
                reg_by_h[h] = float(p)
        else:
            X_tr, y_tr = prepare_train(df, m)
            model = m.fit(X_tr, y_tr)
            feats = row(df, m, t)
            p = float(m.predict(model, feats)[0])
            current_probs.append(p)

        if m in (mod1, mod2, mod3, mod4, mod5) and (h is None):
            base5_vals.append(float(current_probs[-1]))

    k = sum(1 for c in hist.columns if isinstance(c, str) and c.startswith("m") and c[1:].isdigit())
    last_accs = [float(hist[f"acc{i+1}"].iloc[-1]) if f"acc{i+1}" in hist.columns else 0.5 for i in range(k)]

    row_df = pd.DataFrame([{**{f"m{i+1}": current_probs[i] for i in range(k)},
                            **{f"acc{i+1}": last_accs[i] for i in range(k)}}])

    angle_clf  = float(mod6.linear_angle([clf_by_h[h] for h in getattr(mod6, "MULTI_HORIZONS", []) if h in clf_by_h])) if 'mod6' in globals() and len(clf_by_h) >= 2 else float('nan')
    angle_reg  = float(modR.linear_angle([reg_by_h[h] for h in getattr(modR, "MULTI_HORIZONS", []) if h in reg_by_h])) if len(reg_by_h) >= 2 else float('nan')
    angle_base = float(_linear_angle_generic(base5_vals)) if len(base5_vals) >= 2 else float('nan')

    row_df["feat_angle_clf"]  = angle_clf
    row_df["feat_angle_reg"]  = angle_reg
    row_df["feat_angle_base"] = angle_base

    feat_vec = build_meta_features(row_df, n_mods=k)
    prob = float(meta.predict_proba(feat_vec)[0, 1])
    action = "BUY" if prob > up else ("SELL" if prob < down else "HOLD")
    action = enforce_position_rules([action], position_open)[0]
    print(f"[LIVE] P(up)={prob:.3f}  →  {action}  (BUY>{up:.2f} / SELL<{down:.2f})")

    return (prob, action) if return_result else None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Run live prediction mode")
    parser.add_argument("--return_df", action="store_true", help="Return DataFrame instead of writing CSV")
    args = parser.parse_args()

    if args.live:
        run_live()
    else:
        run_backtest(return_df=args.return_df)
