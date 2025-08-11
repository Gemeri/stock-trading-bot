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

import sub.momentum as mod1
import sub.trend_continuation as mod2
import sub.mean_reversion as mod3
import sub.lagged_return as mod4
import sub.sentiment_volume as mod5

USE_CLASSIFIER = True
if USE_CLASSIFIER:
    import sub.classifier as mod6
    SUBMODS = [mod1, mod2, mod3, mod4, mod5, mod6]
else:
    SUBMODS = [mod1, mod2, mod3, mod4, mod5]

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


def build_meta_features(df: pd.DataFrame, n_mods: int | None = None) -> np.ndarray:
    if n_mods is None:
        n_mods = sum(1 for c in df.columns if isinstance(c, str) and c.startswith("m") and c[1:].isdigit())
    prob_cols = [f"m{i+1}"   for i in range(n_mods)]
    acc_cols = [f"acc{i+1}" for i in range(n_mods)]

    P = df[prob_cols].values
    A = df[acc_cols].values

    mean_p = P.mean(axis=1, keepdims=True)
    var_p = P.var(axis=1,  keepdims=True)
    wavg_p = (P * A).sum(axis=1, keepdims=True) / (A.sum(axis=1, keepdims=True) + 1e-6)

    return np.hstack([P, A, mean_p, var_p, wavg_p])

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

    col_m = [f"m{i+1}"   for i in range(n_ch)]
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

    preds_hist = [[] for _ in range(n_ch)]
    labels_hist = [[] for _ in range(n_ch)]
    recs = []

    for i in range(start_idx, end_idx + 1):  # inclusive; i+1 and i+HMAX are safe
        slc = df.iloc[: i + 1].reset_index(drop=True)
        preds, labels = [], []

        for ch in channels:
            m, h = ch["mod"], ch["h"]

            if hasattr(m, "compute_labels"):  # classifier path
                if is_multi(m):
                    X_tr, y_tr = prepare_train_for_horizon(slc, m, h)
                    if len(y_tr) < 50:
                        p = 0.5
                    else:
                        last_feats = m.compute_labels(slc)[m.FEATURES].iloc[[-1]]
                        model = m.fit(X_tr, y_tr, horizon=h)  # safe: only for multi
                        p = m.predict(model, last_feats)[0]
                    lbl = int(df["close"].iat[i + h] > df["close"].iat[i])
                else:
                    X_tr, y_tr = prepare_train(slc, m)
                    if len(y_tr) < 50:
                        p = 0.5
                    else:
                        last_feats = m.compute_labels(slc)[m.FEATURES].iloc[[-1]]
                        model = m.fit(X_tr, y_tr)            # no horizon kwarg
                        p = m.predict(model, last_feats)[0]
                    lbl = int(df["close"].iat[i + 1] > df["close"].iat[i])  # 1-step
            else:
                d = m.compute_target(slc)
                tr  = d.iloc[:-1].dropna(subset=m.FEATURES + ["target"])
                last_feats = d[m.FEATURES].iloc[[-1]]
                if len(tr) < 50:
                    p = 0.0
                else:
                    model = m.fit(tr[m.FEATURES], tr["target"])
                    p = m.predict(model, last_feats)[0]
                d_full = m.compute_target(df)
                lbl = int(d_full["target"].iat[i] > 0)

            preds.append(p)
            labels.append(lbl)

        accs = []
        for k in range(n_ch):
            preds_hist[k].append(preds[k])
            labels_hist[k].append(labels[k])
            ph = np.array(preds_hist[k][-window-1:-1])
            lh = np.array(labels_hist[k][-window-1:-1])
            accs.append(np.mean(np.round(ph) == lh) if len(ph) else 0.5)

        rec = {
            "timestamp": df["timestamp"].iat[i],
            **{col_m[k]:   preds[k] for k in range(n_ch)},
            **{col_acc[k]: accs[k]  for k in range(n_ch)},
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
    y = hist["meta_label"].values

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

    return meta, up, down


def backtest_submodels(
    df: pd.DataFrame,
    initial_frac: float = 0.6,
    window: int = 50,
) -> pd.DataFrame:
    submods = [mod1, mod2, mod3, mod4, mod5] + ([mod6] if USE_CLASSIFIER else [])
    channels = expand_channels(submods)
    n_ch = len(channels)

    n = len(df)
    HMAX = max_lookahead(submods)
    cut = int(n * initial_frac)
    end_limit = n - 1 - HMAX  # ensures t+1 and t+HMAX exist

    rec = []

    preds_history = [[] for _ in range(n_ch)]
    labels_history = [[] for _ in range(n_ch)]

    for t in tqdm(range(max(cut, 0), end_limit), desc="Submodel BT"):
        slice_df = df.iloc[: t + 1].reset_index(drop=True)

        preds, labels = [], []
        for ch in channels:
            m = ch["mod"]; h = ch["h"]

            if hasattr(m, "compute_labels"):  # classifier (single or multi)
                d   = m.compute_labels(slice_df)
                if is_multi(m):
                    lblcol = label_col_for(m, h)         # e.g., label_h3
                    tr  = d.iloc[:-1].dropna(subset=m.FEATURES + [lblcol])
                    if len(tr) < 50:
                        p = 0.5
                    else:
                        last_feats = d[m.FEATURES].iloc[[-1]]
                        model = m.fit(tr[m.FEATURES].values, tr[lblcol].values, horizon=h)
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

            else:  # regression-style submodel
                d = m.compute_target(slice_df)
                tr  = d.iloc[:-1].dropna(subset=m.FEATURES + ["target"])
                last_feats = d[m.FEATURES].iloc[[-1]]
                if len(tr) < 50:
                    p = 0.0
                else:
                    model = m.fit(tr[m.FEATURES], tr["target"])
                    p = m.predict(model, last_feats)[0]
                lbl = 1 if df["close"].iat[t + 1] > df["close"].iat[t] else 0

            preds.append(p); labels.append(lbl)

        accs = []
        for k in range(n_ch):
            preds_history[k].append(preds[k])
            labels_history[k].append(labels[k])
            ph = np.array(preds_history[k][-window - 1 : -1])
            lh = np.array(labels_history[k][-window - 1 : -1])
            accs.append(np.mean(np.round(ph) == lh) if len(ph) else 0.5)

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

    submods = [mod1, mod2, mod3, mod4, mod5] + ([mod6] if USE_CLASSIFIER else [])
    channels = expand_channels(submods)
    n_ch = len(channels)

    if MODE == "sub-vote":
        back = backtest_submodels(df_orig, initial_frac=0.6)
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
        cache_csv = os.path.join(RESULTS_DIR, f"submeta_history_{CONVERTED_TIMEFRAME}.csv")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        update_submeta_history(CSV_PATH, cache_csv, submods=submods, verbose=True)

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
    d = module.compute_target(df_slice)
    tr = d.iloc[:-1].dropna(subset=module.FEATURES + ["target"])
    return tr[module.FEATURES], tr["target"]

def prepare_train_for_horizon(df_slice: pd.DataFrame, module, h: int):
    assert hasattr(module, "compute_labels"), "Horizon training requires classifier module"
    d = module.compute_labels(df_slice)
    lbl_col = label_col_for(module, h)
    tr = d.iloc[:-1].dropna(subset=module.FEATURES + [lbl_col])
    return tr[module.FEATURES].values, tr[lbl_col].values



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
    submods = [mod1, mod2, mod3, mod4, mod5]
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
                        preds.append(0.5)
                        continue
                    model = m.fit(X_tr, y_tr, horizon=h)      # safe: multi only
                    feats = m.compute_labels(df)[m.FEATURES].iloc[[t]]
                    preds.append(m.predict(model, feats)[0])
                else:
                    X_tr, y_tr = prepare_train(df, m)
                    if len(y_tr) < 50:
                        preds.append(0.5)
                        continue
                    model = m.fit(X_tr, y_tr)                 # no horizon kwarg
                    feats = m.compute_labels(df)[m.FEATURES].iloc[[t]]
                    preds.append(m.predict(model, feats)[0])
            else:
                X_tr, y_tr = prepare_train(df, m)
                model = m.fit(X_tr, y_tr)
                feats = row(df, m, t)
                preds.append(m.predict(model, feats)[0])

        votes = []
        for k, ch in enumerate(channels):
            m = ch["mod"]
            if hasattr(m, "compute_labels"):
                votes.append(1 if preds[k] > VOTE_UP else (0 if preds[k] < VOTE_DOWN else 0.5))
            else:
                votes.append(1 if preds[k] > REG_UP else (0 if preds[k] < REG_DOWN else 0.5))

        buy, sell   = votes.count(1), votes.count(0)
        majority_req = (n_ch // 2) + 1
        majority     = 1 if buy  >= majority_req else (0 if sell >= majority_req else 0.5)
        action = "BUY" if majority == 1 else ("SELL" if majority == 0 else "HOLD")
        action = enforce_position_rules([action], position_open)[0]
        print(f"[LIVE-VOTE] result = {action}")
        return action if return_result else None

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

    current_probs = []
    for ch in channels:
        m, h = ch["mod"], ch["h"]
        if hasattr(m, "compute_labels"):
            if is_multi(m):
                X_tr, y_tr = prepare_train_for_horizon(df, m, h)
                if len(y_tr) < 50:
                    current_probs.append(0.5)
                    continue
                model = m.fit(X_tr, y_tr, horizon=h)          # safe: multi only
                feats = m.compute_labels(df)[m.FEATURES].iloc[[t]]
                current_probs.append(m.predict(model, feats)[0])
            else:
                X_tr, y_tr = prepare_train(df, m)
                if len(y_tr) < 50:
                    current_probs.append(0.5)
                    continue
                model = m.fit(X_tr, y_tr)                     # no horizon kwarg
                feats = m.compute_labels(df)[m.FEATURES].iloc[[t]]
                current_probs.append(m.predict(model, feats)[0])
        else:
            X_tr, y_tr = prepare_train(df, m)
            model = m.fit(X_tr, y_tr)
            feats = row(df, m, t)
            current_probs.append(m.predict(model, feats)[0])

    k = sum(1 for c in hist.columns if isinstance(c, str) and c.startswith("m") and c[1:].isdigit())
    last_accs = [float(hist[f"acc{i+1}"].iloc[-1]) if f"acc{i+1}" in hist.columns else 0.5 for i in range(k)]

    row_df = pd.DataFrame([{**{f"m{i+1}": current_probs[i] for i in range(k)},
                            **{f"acc{i+1}": last_accs[i] for i in range(k)}}])

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
