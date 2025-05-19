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
        meta_label = int(slice_df['close'].iat[-1] > slice_df['close'].iat[-2])
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
    Train meta-model using submodel history (preds + rolling accs only).
    """
    hist = pd.read_csv(cache_csv)
    X = hist[[f"m{i+1}" for i in range(5)] + [f"acc{i+1}" for i in range(5)]].values
    y = hist['meta_label'].values
    split = int(len(y) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val     = X[split:], y[split:]
    meta = LogisticRegression().fit(X_train, y_train)
    probs_val = meta.predict_proba(X_val)[:,1]
    up, down = optimize_asymmetric_thresholds(probs_val, y_val, metric=metric)
    with open(model_pkl, 'wb') as f:
        pickle.dump(meta, f)
    with open(threshold_pkl, 'wb') as f:
        pickle.dump({'up': up, 'down': down}, f)
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



def run_backtest(return_df=False):
    """
    Runs either sub-vote or sub-meta backtest, writes result CSV to RESULTS_DIR.
    """
    df_orig = pd.read_csv(CSV_PATH)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if MODE == 'sub-vote':
        back = backtest_submodels(df_orig, initial_frac=0.8)
        # Thresholds for voting: these can be tuned if you wish
        VOTE_UP, VOTE_DOWN = 0.55, 0.45
        back['v1'] = back['m1'].map(lambda x: vote_from_prob(x, VOTE_UP, VOTE_DOWN))
        back['v2'] = back['m2'].map(lambda x: vote_from_prob(x, VOTE_UP, VOTE_DOWN))
        back['v3'] = back['m3'].map(lambda x: vote_from_prob(x, VOTE_UP, VOTE_DOWN))
        back['v4'] = back['m4'].map(vote_from_ret)
        back['v5'] = back['m5'].map(lambda x: vote_from_prob(x, VOTE_UP, VOTE_DOWN))
        back['buy_votes']  = (back[['v1','v2','v3','v4','v5']] == 1).sum(axis=1)
        back['sell_votes'] = (back[['v1','v2','v3','v4','v5']] == 0).sum(axis=1)
        back['action']     = back.apply(
            lambda r: 1 if r.buy_votes >= 3 else (0 if r.sell_votes >= 3 else 0.5),
            axis=1
        )
        back = back.reset_index().rename(columns={'index':'t'})
        back['timestamp'] = pd.to_datetime(df_orig['timestamp'].iloc[back['t'] + 1])
        df_out = back[['timestamp','m1','m2','m3','m4','m5','meta_prob','action']]

    else:  # sub-meta mode
        cache_csv = os.path.join(RESULTS_DIR, f"submeta_history_{CONVERTED_TIMEFRAME}.csv")
        meta_pkl  = os.path.join(RESULTS_DIR, "meta_model.pkl")
        thresh_pkl = os.path.join(RESULTS_DIR, "meta_thresholds.pkl")

        submods = [mod1, mod2, mod3, mod4, mod5]
        # Always rebuild or update meta-history cache properly
        hist = update_submeta_history(CSV_PATH, cache_csv, submods, verbose=True)
        
        # Train or load meta-model and thresholds
        if os.path.exists(meta_pkl) and os.path.exists(thresh_pkl):
            with open(meta_pkl, 'rb') as f:
                meta = pickle.load(f)
            with open(thresh_pkl, 'rb') as f:
                thresh_dict = pickle.load(f)
            up, down = thresh_dict['up'], thresh_dict['down']
            print(f"[THRESH] Loaded thresholds: BUY > {up:.4f}, SELL < {down:.4f}")
        else:
            meta, up, down = train_and_save_meta(cache_csv, meta_pkl, thresh_pkl)

        # Backtest using only the cached predictions and rolling accuracies
        hist = pd.read_csv(cache_csv)
        X = hist[[f"m{i+1}" for i in range(5)] + [f"acc{i+1}" for i in range(5)]].values
        y = hist['meta_label'].values

        records = []
        for i in tqdm(range(len(hist)), desc="Meta Backtest"):
            meta_input = X[i].reshape(1, -1)
            prob = float(meta.predict_proba(meta_input)[0,1])
            if prob > up:
                action = "BUY"
            elif prob < down:
                action = "SELL"
            else:
                action = "HOLD"
            records.append({
                "timestamp": hist['timestamp'].iloc[i],
                "m1": hist['m1'].iloc[i],
                "m2": hist['m2'].iloc[i],
                "m3": hist['m3'].iloc[i],
                "m4": hist['m4'].iloc[i],
                "m5": hist['m5'].iloc[i],
                "acc1": hist['acc1'].iloc[i],
                "acc2": hist['acc2'].iloc[i],
                "acc3": hist['acc3'].iloc[i],
                "acc4": hist['acc4'].iloc[i],
                "acc5": hist['acc5'].iloc[i],
                "meta_prob": prob,
                "action": action
            })

        df_out = pd.DataFrame(records)
        out_fn = os.path.join(RESULTS_DIR, f"meta_results_{MODE}.csv")
        df_out.to_csv(out_fn, index=False, float_format="%.8f")
        print(f"[RESULT] Results saved to {out_fn}")
        if return_df:
            return df_out

def run_live(return_result=False, verbose=True):
    """
    Live mode: 
    - For sub-vote: outputs final vote (classic).
    - For sub-meta: updates meta-history, retrains meta-model, optimizes thresholds, outputs live action.
    """
    # --- Setup paths and submodels ---
    submods = [mod1, mod2, mod3, mod4, mod5]

    # --- Read data ---
    df = pd.read_csv(CSV_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    n = len(df)
    t = n - 1

    if MODE == 'sub-vote':
        VOTE_UP, VOTE_DOWN = 0.55, 0.45
        d1 = mod1.compute_labels(df)
        tr1 = d1.iloc[:-1].dropna(subset=mod1.FEATURES+['label'])
        p1 = mod1.predict(mod1.fit(tr1[mod1.FEATURES].values, tr1['label'].values), d1.loc[[t], mod1.FEATURES])[0]
        d2 = mod2.compute_labels(df)
        tr2 = d2.iloc[:-1].dropna(subset=mod2.FEATURES+['label'])
        p2 = mod2.predict(mod2.fit(tr2[mod2.FEATURES], tr2['label']), d2.loc[[t], mod2.FEATURES])[0]
        d3 = mod3.compute_labels(df)
        tr3 = d3.iloc[:-1].dropna(subset=mod3.FEATURES+['label'])
        p3 = mod3.predict(mod3.fit(tr3[mod3.FEATURES].values, tr3['label'].values), d3.loc[[t], mod3.FEATURES])[0]
        d4 = mod4.compute_target(df)
        tr4 = d4.iloc[:-1].dropna(subset=mod4.FEATURES+['target'])
        p4 = mod4.predict(mod4.fit(tr4[mod4.FEATURES], tr4['target']), d4.loc[[t], mod4.FEATURES])[0]
        d5 = mod5.compute_labels(df)
        tr5 = d5.iloc[:-1].dropna(subset=mod5.FEATURES+['label'])
        p5 = mod5.predict(mod5.fit(tr5[mod5.FEATURES], tr5['label']), d5.loc[[t], mod5.FEATURES])[0]
        votes = [
            vote_from_prob(p1, VOTE_UP, VOTE_DOWN),
            vote_from_prob(p2, VOTE_UP, VOTE_DOWN),
            vote_from_prob(p3, VOTE_UP, VOTE_DOWN),
            vote_from_ret(p4),
            vote_from_prob(p5, VOTE_UP, VOTE_DOWN)
        ]
        buy, sell = votes.count(1), votes.count(0)
        result = 1 if buy >= 3 else (0 if sell >= 3 else 0.5)
        if return_result:
            return result
        else:
            print(f"Sub-vote live result: {result}")
        return  # don't continue to sub-meta

    # --------- SUB-META MODE ---------
    # 1. Update meta-history with any new bars
    hist = update_submeta_history(CSV_PATH, HISTORY_PATH, submods, verbose=verbose)
    X = hist[[f"m{i+1}" for i in range(5)] + [f"acc{i+1}" for i in range(5)]].values
    y = hist['meta_label'].values
    if len(y) < 20:
        raise RuntimeError("Not enough meta-history for training!")
    split = int(len(y) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val     = X[split:], y[split:]
    meta = LogisticRegression().fit(X_train, y_train)
    probs_val = meta.predict_proba(X_val)[:,1]
    up, down = optimize_asymmetric_thresholds(probs_val, y_val, metric="accuracy")
    if verbose:
        print(f"[THRESH] (Live) Optimized: BUY > {up:.4f}, SELL < {down:.4f}")

    # Current submodel preds
    p = []
    for idx, mod in enumerate(submods):
        if hasattr(mod, 'compute_labels'):
            d = mod.compute_labels(df)
            tr = d.iloc[:-1].dropna(subset=mod.FEATURES + ['label'])
            pi = mod.predict(mod.fit(tr[mod.FEATURES].values, tr['label'].values), d.loc[[t], mod.FEATURES])[0]
        elif hasattr(mod, 'compute_target'):
            d = mod.compute_target(df)
            tr = d.iloc[:-1].dropna(subset=mod.FEATURES + ['target'])
            pi = mod.predict(mod.fit(tr[mod.FEATURES], tr['target']), d.loc[[t], mod.FEATURES])[0]
        p.append(pi)
    # Rolling accuracy (using last 50 points in history)
    accs = []
    for i in range(5):
        preds = hist[f"m{i+1}"].values[-50:] if len(hist) > 1 else []
        labels = hist['meta_label'].values[-50:] if len(hist) > 1 else []
        if len(preds) > 0 and len(labels) > 0:
            acc = (np.round(preds) == labels).mean()
        else:
            acc = 0.5
        accs.append(acc)
    meta_input = np.array(p + accs).reshape(1, -1)

    prob = float(meta.predict_proba(meta_input)[0,1])
    if prob > up:
        action = "BUY"
    elif prob < down:
        action = "SELL"
    else:
        action = "HOLD"
    print(f"[LIVE] Meta-model: Next-close up probability = {prob:.4f} (BUY>{up:.2f}, SELL<{down:.2f}) → {action}")
    if return_result:
        return prob, action


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

