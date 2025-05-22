#!/usr/bin/env python3
"""
main.py

Orchestrates five sub-models in either “sub-vote” or “sub-meta” mode,
and in either “live” or “backtest” execution—configured by environment variables.

Features:
- Asymmetric threshold optimization over validation set (for neutral "hold" zone)
- Consistent close-to-close label alignment
- Persisted model and thresholds as separate artifacts (robust for production)
- Modular code and easy extensibility
"""

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from typing import Tuple

import sub.momentum            as mod1
import sub.trend_continuation  as mod2
import sub.mean_reversion      as mod3
import sub.lagged_return       as mod4
import sub.sentiment_volume    as mod5

# ─── Base Directories ─────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
ROOT_DIR    = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
RESULTS_DIR = os.path.join(BASE_DIR, "sub-results")

# ─── Configuration ───────────────────────────────────────────────────────────
load_dotenv()
BAR_TIMEFRAME       = os.getenv("BAR_TIMEFRAME", "4Hour")
TICKERS             = os.getenv("TICKERS", "TSLA").split(",")
ML_MODEL            = os.getenv("ML_MODEL",  "sub-vote")
TIMEFRAME_MAP       = {"4Hour":"H4","2Hour":"H2","1Hour":"H1","30Min":"M30","15Min":"M15"}
CONVERTED_TIMEFRAME = TIMEFRAME_MAP.get(BAR_TIMEFRAME, BAR_TIMEFRAME)
HISTORY_PATH = os.path.join(RESULTS_DIR, f"submeta_history_{CONVERTED_TIMEFRAME}.csv")

CSV_PATH  = os.path.join(ROOT_DIR, f"{TICKERS}_{CONVERTED_TIMEFRAME}.csv")
MODE       = ML_MODEL  # "sub-vote" or "sub-meta"
EXECUTION  = globals().get("EXECUTION", "backtest")

# Default return thresholds (used only for submodels returning regression)
REG_UP, REG_DOWN = 0.003, -0.003

# Set to True to force thresholds to 0.55/0.45 everywhere
FORCE_STATIC_THRESHOLDS = False
STATIC_UP = 0.55
STATIC_DOWN = 0.45

META_MODEL_TYPE = "logreg"
VALID_META_MODELS = {"logreg", "lgbm", "xgb", "nn"}
if META_MODEL_TYPE not in VALID_META_MODELS:
    raise ValueError(f"META_MODEL_TYPE must be one of {VALID_META_MODELS}")


# ------------------------------------------------------------------------------

def vote_from_prob(p: float, up: float, down: float) -> float:
    """Voting logic for classification-style submodels."""
    return 1 if p > up else (0 if p < down else 0.5)

def vote_from_ret(r: float) -> float:
    """Voting logic for regression-style submodels."""
    return 1 if r > REG_UP else (0 if r < REG_DOWN else 0.5)

def actual_direction(df: pd.DataFrame, t: int) -> int:
    """
    Returns 1 if next close > current close, else 0.
    (Close-to-close direction, as required.)
    """
    return int(df['close'].iat[t+1] > df['close'].iat[t])

# ─── Helper to construct the chosen meta learner ─────────────────────────────
def _make_meta_model():
    """
    Return an *un-fitted* classifier according to META_MODEL_TYPE.
    All models expose .fit(X,y) and .predict_proba(X) just like LogisticRegression.
    """
    if META_MODEL_TYPE == "logreg":
        return LogisticRegression(max_iter=1000, class_weight="balanced")
    elif META_MODEL_TYPE == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=200, learning_rate=0.05,
            num_leaves=31, random_state=42, verbosity=-1
        )
    elif META_MODEL_TYPE == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=300, learning_rate=0.05,
            max_depth=3, subsample=0.8, colsample_bytree=0.9,
            objective="binary:logistic", eval_metric="logloss",
            random_state=42, use_label_encoder=False
        )
    elif META_MODEL_TYPE == "nn":
        # simple two-layer MLP; good enough for ≤ 20 features
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(
            hidden_layer_sizes=(32, 32), activation="relu",
            solver="adam", alpha=1e-4, batch_size="auto",
            learning_rate_init=1e-3, max_iter=500, random_state=42
        )

def update_submeta_history(CSV_PATH, HISTORY_PATH, submods, window=50, verbose=True):
    """
    Updates meta history (submeta_history_X.csv) to match underlying CSV.
    Stores only submodel predictions (m1..m5), rolling accs (acc1..acc5), and meta_label (close-to-close dir).
    Never saves label1-5.
    """
    df = pd.read_csv(CSV_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    if os.path.exists(HISTORY_PATH):
        hist = pd.read_csv(HISTORY_PATH)
        hist['timestamp'] = pd.to_datetime(hist['timestamp'], utc=True)
        last_ts = hist['timestamp'].iloc[-1]
        start_idx = df[df['timestamp'] > last_ts].index.min()
        if pd.isna(start_idx):
            if verbose: print("[HISTORY] Already up-to-date.")
            return hist
    else:
        cols = ['timestamp'] + [f"m{i+1}" for i in range(5)] + [f"acc{i+1}" for i in range(5)] + ['meta_label']
        hist = pd.DataFrame(columns=cols)
        start_idx = int(len(df) * 0.8)

    recs = []
    preds_history = [list(hist[f"m{i+1}"]) if not hist.empty else [] for i in range(5)]
    labels_history = [[] for _ in range(5)]

    for i in range(start_idx, len(df) - 1):
        slice_df = df.iloc[:i+1].reset_index(drop=True)
        preds = []
        labels = []
        for idx, mod in enumerate(submods):
            if hasattr(mod, 'compute_labels'):
                d = mod.compute_labels(slice_df)
                tr = d.iloc[:-1].dropna(subset=mod.FEATURES + ['label'])
                pred = mod.predict(mod.fit(tr[mod.FEATURES].values, tr['label'].values),
                                   d.loc[[len(slice_df)-1], mod.FEATURES])[0]
                label = d['label'].iloc[-1]
            elif hasattr(mod, 'compute_target'):
                d = mod.compute_target(slice_df)
                tr = d.iloc[:-1].dropna(subset=mod.FEATURES + ['target'])
                pred = mod.predict(mod.fit(tr[mod.FEATURES], tr['target']),
                                   d.loc[[len(slice_df)-1], mod.FEATURES])[0]
                label = int(d['target'].iloc[-1] > 0)
            preds.append(pred)
            labels.append(label)
        # Rolling accuracy over *previous* 50
        for idx in range(5):
            preds_history[idx].append(preds[idx])
            labels_history[idx].append(labels[idx])
        accs = []
        for idx in range(5):
            preds_arr = np.array(preds_history[idx][-window-1:-1])
            labels_arr = np.array(labels_history[idx][-window-1:-1])
            if len(preds_arr) > 0 and len(labels_arr) > 0:
                acc = np.mean(np.round(preds_arr) == labels_arr)
            else:
                acc = 0.5
            accs.append(acc)
        meta_label = int(df['close'].iat[i+1] > df['close'].iat[i])
        rec = {'timestamp': slice_df['timestamp'].iat[-1]}
        rec.update({f"m{i+1}": preds[i] for i in range(5)})
        rec.update({f"acc{i+1}": accs[i] for i in range(5)})
        rec['meta_label'] = meta_label
        recs.append(rec)
        if verbose and (i - start_idx) % 10 == 0:
            print(f"[HISTORY] Appended new bar at idx={i} ({slice_df['timestamp'].iat[-1]})")

    if recs:
        new_df = pd.DataFrame(recs)
        hist = pd.concat([hist, new_df], ignore_index=True)
        hist.to_csv(HISTORY_PATH, index=False)
        if verbose: print(f"[HISTORY] Updated. Now {len(hist)} total rows.")
    return hist


def optimize_asymmetric_thresholds(
    probs: np.ndarray, labels: np.ndarray,
    metric: str = "accuracy",
    manual_thresholds: tuple = None
) -> Tuple[float, float]:
    """
    Finds best asymmetric (up, down) threshold for HOLD zone,
    or uses manual_thresholds if provided.
    """
    if FORCE_STATIC_THRESHOLDS:
        print(f"[THRESH] Using STATIC thresholds: BUY > {STATIC_UP:.4f}, SELL < {STATIC_DOWN:.4f}")
        return STATIC_UP, STATIC_DOWN
    if manual_thresholds is not None:
        up, down = manual_thresholds
        print(f"[THRESH] Using MANUAL thresholds: BUY > {up:.4f}, SELL < {down:.4f}")
        return up, down

    up_grid = np.linspace(0.51, 0.80, 30)
    down_grid = np.linspace(0.20, 0.49, 30)
    best_score, best_up, best_down = 0, 0.55, 0.45

    for up in up_grid:
        for down in down_grid:
            if up <= down:
                continue
            preds = np.where(probs > up, 1, np.where(probs < down, 0, 0.5))
            mask = preds != 0.5
            num_confident = np.sum(mask)
            print(f"Thresholds: up={up:.2f}, down={down:.2f}, confident preds={num_confident}/{len(probs)}")
            if not np.any(mask):
                continue
            if metric == "accuracy":
                score = (preds[mask] == labels[mask]).mean()
            elif metric == "f1":
                from sklearn.metrics import f1_score
                score = f1_score(labels[mask], preds[mask])
            else:
                score = (preds[mask] == labels[mask]).mean()
            if score > best_score:
                best_score, best_up, best_down = score, up, down
    print(f"[THRESH] Optimized thresholds: BUY > {best_up:.4f}, SELL < {best_down:.4f}  (valid acc={best_score:.4f})")
    return best_up, best_down


def train_and_save_meta(
    cache_csv: str,
    model_pkl: str,
    threshold_pkl: str,
    metric: str = "accuracy"
):
    """
    Train the *chosen* meta model using sub-model history (preds + rolling accs).
    Persist both the fitted model and the asymmetric thresholds.
    """
    hist = pd.read_csv(cache_csv)
    X = hist[[f"m{i+1}" for i in range(5)] + [f"acc{i+1}" for i in range(5)]].values
    y = hist['meta_label'].values

    # train/validation split (time-ordered)
    split = int(len(y) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val,   y_val   = X[split:], y[split:]

    meta = _make_meta_model()
    meta.fit(X_train, y_train)

    probs_val = meta.predict_proba(X_val)[:, 1]
    up, down  = optimize_asymmetric_thresholds(probs_val, y_val, metric=metric)

    with open(model_pkl, "wb") as f:
        pickle.dump(meta, f)
    with open(threshold_pkl, "wb") as f:
        pickle.dump({"up": up, "down": down, "model_type": META_MODEL_TYPE}, f)

    return meta, up, down


def backtest_submodels(df: pd.DataFrame, initial_frac=0.8, window=50) -> pd.DataFrame:
    """
    Walk-forward backtest of submodels. Outputs only predictions, rolling accuracies, and meta-label.
    """
    n = len(df)
    cut = int(n * initial_frac)
    rec = []

    # Rolling histories for accuracy computation
    preds_history = [[] for _ in range(5)]
    labels_history = [[] for _ in range(5)]

    for t in tqdm(range(cut, n-1), desc="Submodel BT"):
        slice_df = df.iloc[:t+1].reset_index(drop=True)
        preds = []
        labels = []
        # Submodel predictions and true (not stored) labels
        for idx, mod in enumerate([mod1, mod2, mod3, mod4, mod5]):
            if hasattr(mod, 'compute_labels'):
                d = mod.compute_labels(slice_df)
                tr = d.iloc[:-1].dropna(subset=mod.FEATURES + ['label'])
                pred = mod.predict(mod.fit(tr[mod.FEATURES].values, tr['label'].values), d.loc[[t], mod.FEATURES])[0]
                label = d['label'].iloc[-1]
            elif hasattr(mod, 'compute_target'):
                d = mod.compute_target(slice_df)
                tr = d.iloc[:-1].dropna(subset=mod.FEATURES + ['target'])
                pred = mod.predict(mod.fit(tr[mod.FEATURES], tr['target']), d.loc[[t], mod.FEATURES])[0]
                label = int(d['target'].iloc[-1] > 0)
            preds.append(pred)
            labels.append(label)
        # Update rolling histories
        for idx in range(5):
            preds_history[idx].append(preds[idx])
            labels_history[idx].append(labels[idx])
        # Rolling accuracy up to t-1
        accs = []
        for idx in range(5):
            ph = np.array(preds_history[idx][-window-1:-1])
            lh = np.array(labels_history[idx][-window-1:-1])
            if len(ph) > 0 and len(lh) > 0:
                acc = (np.round(ph) == lh).mean()
            else:
                acc = 0.5
            accs.append(acc)
        # Meta-label: did close go up at t+1?
        if t+1 < n:
            meta_label = int(df['close'].iat[t+1] > df['close'].iat[t])
        else:
            meta_label = np.nan
        rec.append({
            't': t,
            'm1': preds[0], 'm2': preds[1], 'm3': preds[2], 'm4': preds[3], 'm5': preds[4],
            'acc1': accs[0], 'acc2': accs[1], 'acc3': accs[2], 'acc4': accs[3], 'acc5': accs[4],
            'meta_label': meta_label
        })
    return pd.DataFrame(rec).set_index('t')


# ───────────────────────── run_backtest ──────────────────────────────────────
def run_backtest(return_df: bool = False):
    """
    Back-test either classic sub-vote or the meta-model pipeline.
    Results are written to sub-results/ and optionally returned.
    """
    df_orig = pd.read_csv(CSV_PATH)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---------- Classic majority vote ---------------------------------------
    if MODE == "sub-vote":
        back = backtest_submodels(df_orig, initial_frac=0.8)
        VOTE_UP, VOTE_DOWN = 0.55, 0.45
        back["v1"] = back["m1"].map(lambda x: vote_from_prob(x, VOTE_UP, VOTE_DOWN))
        back["v2"] = back["m2"].map(lambda x: vote_from_prob(x, VOTE_UP, VOTE_DOWN))
        back["v3"] = back["m3"].map(lambda x: vote_from_prob(x, VOTE_UP, VOTE_DOWN))
        back["v4"] = back["m4"].map(vote_from_ret)
        back["v5"] = back["m5"].map(lambda x: vote_from_prob(x, VOTE_UP, VOTE_DOWN))

        back["buy_votes"] = (back[["v1", "v2", "v3", "v4", "v5"]] == 1).sum(axis=1)
        back["sell_votes"] = (back[["v1", "v2", "v3", "v4", "v5"]] == 0).sum(axis=1)
        back["action"] = back.apply(
            lambda r: 1 if r.buy_votes >= 3 else (0 if r.sell_votes >= 3 else 0.5),
            axis=1,
        )
        back = back.reset_index().rename(columns={"index": "t"})
        back["timestamp"] = pd.to_datetime(df_orig["timestamp"].iloc[back["t"] + 1])
        df_out = back[["timestamp", "m1", "m2", "m3", "m4", "m5", "action"]]

    # ---------- Meta-model pipeline -----------------------------------------
    else:
        cache_csv  = os.path.join(RESULTS_DIR, f"submeta_history_{CONVERTED_TIMEFRAME}.csv")
        meta_pkl   = os.path.join(RESULTS_DIR, "meta_model.pkl")
        thresh_pkl = os.path.join(RESULTS_DIR, "meta_thresholds.pkl")

        submods = [mod1, mod2, mod3, mod4, mod5]
        hist = update_submeta_history(CSV_PATH, cache_csv, submods, verbose=True)

        # ▸ load or retrain ---------------------------------------------------
        must_retrain = True
        if os.path.exists(meta_pkl) and os.path.exists(thresh_pkl):
            with open(thresh_pkl, "rb") as f:
                thresh_dict = pickle.load(f)
            if thresh_dict.get("model_type") == META_MODEL_TYPE:
                with open(meta_pkl, "rb") as f:
                    meta = pickle.load(f)
                up, down = thresh_dict["up"], thresh_dict["down"]
                must_retrain = False
                print(f"[META] Loaded cached {META_MODEL_TYPE} model (BUY>{up:.2f} / SELL<{down:.2f})")

        if must_retrain:
            meta, up, down = train_and_save_meta(cache_csv, meta_pkl, thresh_pkl)
            print(f"[META] Trained new {META_MODEL_TYPE} model (BUY>{up:.2f} / SELL<{down:.2f})")

        # ▸ walk through cached rows -----------------------------------------
        hist = pd.read_csv(cache_csv)  # ensure latest
        X = hist[[f"m{i+1}" for i in range(5)] + [f"acc{i+1}" for i in range(5)]].values

        records = []
        for i in tqdm(range(len(hist)), desc="Meta Backtest"):
            prob = float(meta.predict_proba(X[i].reshape(1, -1))[0, 1])
            action = "BUY" if prob > up else ("SELL" if prob < down else "HOLD")
            records.append(
                {
                    "timestamp": hist["timestamp"].iloc[i],
                    "m1": hist["m1"].iloc[i],
                    "m2": hist["m2"].iloc[i],
                    "m3": hist["m3"].iloc[i],
                    "m4": hist["m4"].iloc[i],
                    "m5": hist["m5"].iloc[i],
                    "acc1": hist["acc1"].iloc[i],
                    "acc2": hist["acc2"].iloc[i],
                    "acc3": hist["acc3"].iloc[i],
                    "acc4": hist["acc4"].iloc[i],
                    "acc5": hist["acc5"].iloc[i],
                    "meta_prob": prob,
                    "action": action,
                }
            )
        df_out = pd.DataFrame(records)
        out_fn = os.path.join(RESULTS_DIR, f"meta_results_{MODE}.csv")
        df_out.to_csv(out_fn, index=False, float_format="%.8f")
        print(f"[RESULT] Results saved to {out_fn}")

    if return_df:
        return df_out
    

# ─── tiny helpers (unchanged) ────────────────────────────────────────────────
def _prepare_train(df_slice: pd.DataFrame, module):
    """Return (X, y) for the given sub-module on df_slice[:-1]."""
    if hasattr(module, "compute_labels"):
        d  = module.compute_labels(df_slice)
        tr = d.iloc[:-1].dropna(subset=module.FEATURES + ["label"])
        return tr[module.FEATURES].values, tr["label"].values
    d  = module.compute_target(df_slice)
    tr = d.iloc[:-1].dropna(subset=module.FEATURES + ["target"])
    return tr[module.FEATURES], tr["target"]


def _row(df_slice: pd.DataFrame, module, idx: int):
    """Single-row feature frame/array for prediction."""
    d = module.compute_labels(df_slice) if hasattr(module, "compute_labels") else module.compute_target(df_slice)
    return d.loc[[idx], module.FEATURES]


# ────────────────────────── run_live (verbose always on) ────────────────────
def run_live(return_result: bool = True):
    """
    Produce one live prediction.

    • MODE == "sub-vote" → majority-vote result (0 / 0.5 / 1)
    • MODE == "sub-meta" → tuple (probability, "BUY"/"HOLD"/"SELL")

    Console output is always enabled.
    """
    submods = [mod1, mod2, mod3, mod4, mod5]
    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    t = len(df) - 1  # last bar

    # ─── Majority-vote path ────────────────────────────────────────────────
    if MODE == "sub-vote":
        VOTE_UP, VOTE_DOWN = 0.55, 0.45
        preds = [
            mod1.predict(mod1.fit(*_prepare_train(df, mod1)), _row(df, mod1, t))[0],
            mod2.predict(mod2.fit(*_prepare_train(df, mod2)), _row(df, mod2, t))[0],
            mod3.predict(mod3.fit(*_prepare_train(df, mod3)), _row(df, mod3, t))[0],
            mod4.predict(mod4.fit(*_prepare_train(df, mod4)), _row(df, mod4, t))[0],
            mod5.predict(mod5.fit(*_prepare_train(df, mod5)), _row(df, mod5, t))[0],
        ]

        votes = [
            vote_from_prob(preds[0], VOTE_UP, VOTE_DOWN),
            vote_from_prob(preds[1], VOTE_UP, VOTE_DOWN),
            vote_from_prob(preds[2], VOTE_UP, VOTE_DOWN),
            vote_from_ret(preds[3]),
            vote_from_prob(preds[4], VOTE_UP, VOTE_DOWN),
        ]
        buy, sell = votes.count(1), votes.count(0)
        majority = 1 if buy >= 3 else (0 if sell >= 3 else 0.5)
        print(f"[LIVE-VOTE] result = {majority}")
        return majority if return_result else None

    # ─── Meta-model path ───────────────────────────────────────────────────
    hist = update_submeta_history(CSV_PATH, HISTORY_PATH, submods, verbose=True)

    meta_pkl   = os.path.join(RESULTS_DIR, "meta_model.pkl")
    thresh_pkl = os.path.join(RESULTS_DIR, "meta_thresholds.pkl")

    must_retrain = True
    if os.path.exists(meta_pkl) and os.path.exists(thresh_pkl):
        with open(thresh_pkl, "rb") as f:
            thresh_dict = pickle.load(f)
        if thresh_dict.get("model_type") == META_MODEL_TYPE:
            with open(meta_pkl, "rb") as f:
                meta = pickle.load(f)
            up, down = thresh_dict["up"], thresh_dict["down"]
            must_retrain = False
            print(f"[META] Loaded cached {META_MODEL_TYPE} model")

    if must_retrain:
        meta, up, down = train_and_save_meta(HISTORY_PATH, meta_pkl, thresh_pkl)
        print(f"[META] Re-trained {META_MODEL_TYPE} model  BUY>{up:.2f} / SELL<{down:.2f}")

    # current sub-model preds + rolling accuracies
    window = 50
    p, accs = [], []
    for mod in submods:
        if hasattr(mod, "compute_labels"):
            d  = mod.compute_labels(df)
            pi = mod.predict(mod.fit(*_prepare_train(df, mod)), _row(df, mod, t))[0]

            past_labels = d["label"].iloc[-(window + 1):-1].values
            past_preds  = d["label"].iloc[-(window + 1):-1].index.map(
                lambda j: mod.predict(
                    mod.fit(*_prepare_train(d.iloc[: j + 1], mod)),
                    _row(d, mod, j)
                )[0]
            ).values if len(d) > window else np.array([])

            acc = (np.round(past_preds) == past_labels).mean() if len(past_preds) else 0.5
        else:  # regression-style
            d  = mod.compute_target(df)
            pi = mod.predict(mod.fit(*_prepare_train(df, mod)), _row(df, mod, t))[0]

            past_labels = (d["target"].iloc[-(window + 1):-1] > 0).astype(int).values
            past_preds  = d["target"].iloc[-(window + 1):-1].index.map(
                lambda j: int(
                    mod.predict(
                        mod.fit(*_prepare_train(d.iloc[: j + 1], mod)),
                        _row(d, mod, j)
                    )[0] > 0
                )
            ).values if len(d) > window else np.array([])

            acc = (np.round(past_preds) == past_labels).mean() if len(past_preds) else 0.5

        p.append(pi)
        accs.append(acc)

    prob   = float(meta.predict_proba(np.array(p + accs).reshape(1, -1))[0, 1])
    action = "BUY" if prob > up else ("SELL" if prob < down else "HOLD")
    print(f"[LIVE] P(up)={prob:.3f}  →  {action}  (BUY>{up:.2f} / SELL<{down:.2f})")

    return (prob, action) if return_result else None


# ─── Entrypoint ──────────────────────────────────────────────────────────────

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

