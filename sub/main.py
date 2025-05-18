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

def update_submeta_history(CSV_PATH, HISTORY_PATH, submods, verbose=True):
    """
    Ensures submeta_history_H4.csv is up to date with all bars in the raw CSV.
    Appends any missing submodel predictions + their proper label (for each submodel) + meta label.
    Returns updated history as a DataFrame.
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
        # Now columns: timestamp, m1..m5, label_m1..label_m5, meta_label
        cols = ['timestamp'] + [f"m{i+1}" for i in range(5)] + [f"label_m{i+1}" for i in range(5)] + ['meta_label']
        hist = pd.DataFrame(columns=cols)
        start_idx = int(len(df) * 0.8)

    recs = []
    for i in range(start_idx, len(df)-1):  # -1 so meta_label is defined (next close)
        slice_df = df.iloc[:i+1].reset_index(drop=True)
        # Get predictions and labels for each submodel
        preds = []
        labels = []
        for idx, mod in enumerate(submods):
            if hasattr(mod, 'compute_labels'):
                d = mod.compute_labels(slice_df)
                tr = d.iloc[:-1].dropna(subset=mod.FEATURES + ['label'])
                pred = mod.predict(mod.fit(tr[mod.FEATURES].values, tr['label'].values), d.loc[[len(slice_df)-1], mod.FEATURES])[0]
                # Label for the submodel (on the most recent bar)
                label = d['label'].iloc[-1]
            elif hasattr(mod, 'compute_target'):
                d = mod.compute_target(slice_df)
                tr = d.iloc[:-1].dropna(subset=mod.FEATURES + ['target'])
                pred = mod.predict(mod.fit(tr[mod.FEATURES], tr['target']), d.loc[[len(slice_df)-1], mod.FEATURES])[0]
                # Target for regression as label
                label = d['target'].iloc[-1]
            preds.append(pred)
            labels.append(label)
        # Meta label: will next close go up?
        meta_label = int(slice_df['close'].iat[-1] > slice_df['close'].iat[-2])
        rec = {'timestamp': slice_df['timestamp'].iat[-1]}
        rec.update({f"m{i+1}": preds[i] for i in range(5)})
        rec.update({f"label_m{i+1}": labels[i] for i in range(5)})
        rec['meta_label'] = meta_label
        recs.append(rec)
        if verbose and (i-start_idx)%10==0:
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
) -> Tuple[LogisticRegression, float, float]:
    """
    Train meta-model using submodel history.
    Use last 20% for threshold optimization.
    Save model and thresholds separately.
    Now meta uses m1..m5 and label_m1..label_m5 as features!
    """
    hist = pd.read_csv(cache_csv)
    # Use predictions AND their proper submodel labels as features!
    X = hist[[f"m{i+1}" for i in range(5)] + [f"label_m{i+1}" for i in range(5)]].values
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


def backtest_submodels(df: pd.DataFrame, initial_frac=0.8) -> pd.DataFrame:
    """
    Walk-forward backtest of submodels.
    For each bar, saves: [predictions..., labels..., main close-to-close label]
    """
    n   = len(df)
    cut = int(n * initial_frac)
    rec = []
    for t in tqdm(range(cut, n-1), desc="Submodel BT"):
        slice_df = df.iloc[:t+1].reset_index(drop=True)
        # --- Model 1 (Momentum)
        d1  = mod1.compute_labels(slice_df)
        tr1 = d1.iloc[:-1].dropna(subset=mod1.FEATURES + ['label'])
        p1  = mod1.predict(mod1.fit(tr1[mod1.FEATURES].values, tr1['label'].values),
                           d1.loc[[t], mod1.FEATURES])[0]
        label1 = d1['label'].iloc[-1]

        # --- Model 2 (Trend Continuation)
        d2  = mod2.compute_labels(slice_df)
        tr2 = d2.iloc[:-1].dropna(subset=mod2.FEATURES + ['label'])
        p2  = mod2.predict(mod2.fit(tr2[mod2.FEATURES], tr2['label']),
                           d2.loc[[t], mod2.FEATURES])[0]
        label2 = d2['label'].iloc[-1]

        # --- Model 3 (Mean Reversion)
        d3  = mod3.compute_labels(slice_df)
        tr3 = d3.iloc[:-1].dropna(subset=mod3.FEATURES + ['label'])
        p3  = mod3.predict(mod3.fit(tr3[mod3.FEATURES].values, tr3['label'].values),
                           d3.loc[[t], mod3.FEATURES])[0]
        label3 = d3['label'].iloc[-1]

        # --- Model 4 (Lagged Return)
        d4  = mod4.compute_target(slice_df)
        tr4 = d4.iloc[:-1].dropna(subset=mod4.FEATURES + ['target'])
        p4  = mod4.predict(mod4.fit(tr4[mod4.FEATURES], tr4['target']),
                           d4.loc[[t], mod4.FEATURES])[0]
        # Here, decide if you want a direction label (e.g., up/down) or regression value.
        # For classification-style meta, convert regression to direction label:
        label4 = int(d4['target'].iloc[-1] > 0)

        # --- Model 5 (Sentiment/Volume)
        d5  = mod5.compute_labels(slice_df, min_return=0.002)
        tr5 = d5.iloc[:-1].dropna(subset=mod5.FEATURES + ['label'])
        p5  = mod5.predict(mod5.fit(tr5[mod5.FEATURES], tr5['label']),
                           d5.loc[[t], mod5.FEATURES])[0]
        label5 = d5['label'].iloc[-1]

        # Main close-to-close label (for overall direction)
        label = int(slice_df['close'].iat[-1] > slice_df['close'].iat[-2])

        rec.append({
            't':t, 'm1':p1, 'm2':p2, 'm3':p3, 'm4':p4, 'm5':p5,
            'label1': label1, 'label2': label2, 'label3': label3, 'label4': label4, 'label5': label5,
            'label': label
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

        # Build or load cache
        if os.path.exists(cache_csv):
            rec = pd.read_csv(cache_csv)
            rec['timestamp'] = pd.to_datetime(rec['timestamp'], utc=True)
        else:
            rec = backtest_submodels(df_orig, initial_frac=0.8).reset_index().rename(columns={'index':'t'})
            rec['timestamp'] = pd.to_datetime(df_orig['timestamp'].iloc[rec['t'] + 1])
            rec = rec[['timestamp','m1','m2','m3','m4','m5','label1','label2','label3','label4','label5','label']]
            rec.to_csv(cache_csv, index=False)

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

        # Walk-forward meta backtest (500-bar lookback)
        records = []
        n = len(df_orig)
        window_size = 500
        start = n - window_size
        end   = n - 1

        for i in tqdm(range(start, end), desc="Meta Backtest", total=end - start):
            window_df = df_orig.iloc[i - window_size : i + 1].reset_index(drop=True)
            t = len(window_df) - 1

            d1 = mod1.compute_labels(window_df)
            p1 = mod1.predict(
                mod1.fit(d1.iloc[:-1][mod1.FEATURES].values,
                        d1.iloc[:-1]['label'].values),
                d1.loc[[t], mod1.FEATURES]
            )[0]
            d2 = mod2.compute_labels(window_df)
            p2 = mod2.predict(
                mod2.fit(d2.iloc[:-1][mod2.FEATURES], d2.iloc[:-1]['label']),
                d2.loc[[t], mod2.FEATURES]
            )[0]
            d3 = mod3.compute_labels(window_df)
            p3 = mod3.predict(
                mod3.fit(d3.iloc[:-1][mod3.FEATURES], d3.iloc[:-1]['label']),
                d3.loc[[t], mod3.FEATURES]
            )[0]
            d4 = mod4.compute_target(window_df)
            p4 = mod4.predict(
                mod4.fit(d4.iloc[:-1][mod4.FEATURES], d4.iloc[:-1]['target']),
                d4.loc[[t], mod4.FEATURES]
            )[0]
            d5 = mod5.compute_labels(window_df, min_return=0.002)
            p5 = mod5.predict(
                mod5.fit(d5.iloc[:-1][mod5.FEATURES], d5.iloc[:-1]['label']),
                d5.loc[[t], mod5.FEATURES]
            )[0]

            # GET SUBMODEL LABELS FOR THIS WINDOW
            label1 = d1['label'].iloc[-1]
            label2 = d2['label'].iloc[-1]
            label3 = d3['label'].iloc[-1]
            label4 = int(d4['target'].iloc[-1] > 0)
            label5 = d5['label'].iloc[-1]

            meta_input = np.array([p1, p2, p3, p4, p5, label1, label2, label3, label4, label5]).reshape(1, -1)
            prob = float(meta.predict_proba(meta_input)[0,1])

            if prob > up:
                action = "BUY"
            elif prob < down:
                action = "SELL"
            else:
                action = "HOLD"
            ts = pd.to_datetime(df_orig['timestamp'].iat[i + 1])

            records.append({
                "timestamp": ts,
                "m1":        p1,
                "m2":        p2,
                "m3":        p3,
                "m4":        p4,
                "m5":        p5,
                "label1":    label1,
                "label2":    label2,
                "label3":    label3,
                "label4":    label4,
                "label5":    label5,
                "meta_prob": prob,
                "action":    action
            })

            back = pd.DataFrame(records)
            df_out = back[['timestamp','m1','m2','m3','m4','m5','action']]

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
        d5 = mod5.compute_labels(df, min_return=0.002)
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

    # 2. Retrain meta-model and optimize thresholds
    X = hist[['m1', 'm2', 'm3', 'm4', 'm5', 'label1', 'label2', 'label3', 'label4', 'label5']].values
    y = hist['label'].values
    if len(y) < 20:  # Need at least 20 samples
        raise RuntimeError("Not enough meta-history for training!")
    split = int(len(y) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val     = X[split:], y[split:]
    meta = LogisticRegression().fit(X_train, y_train)
    probs_val = meta.predict_proba(X_val)[:,1]

    # Use your existing optimize_asymmetric_thresholds here
    up, down = optimize_asymmetric_thresholds(probs_val, y_val, metric="accuracy")
    if verbose:
        print(f"[THRESH] (Live) Optimized: BUY > {up:.4f}, SELL < {down:.4f}")

    # 3. Run the *current* submodels on the newest bar
    p = []
    labels = []
    for idx, mod in enumerate(submods):
        if hasattr(mod, 'compute_labels'):
            d = mod.compute_labels(df)
            tr = d.iloc[:-1].dropna(subset=mod.FEATURES + ['label'])
            pi = mod.predict(mod.fit(tr[mod.FEATURES].values, tr['label'].values), d.loc[[t], mod.FEATURES])[0]
            label_i = d['label'].iloc[-1]
        elif hasattr(mod, 'compute_target'):
            d = mod.compute_target(df)
            tr = d.iloc[:-1].dropna(subset=mod.FEATURES + ['target'])
            pi = mod.predict(mod.fit(tr[mod.FEATURES], tr['target']), d.loc[[t], mod.FEATURES])[0]
            label_i = d['target'].iloc[-1]
        p.append(pi)
        labels.append(label_i)
    # Combine for meta input:
    meta_input = np.array(p + labels).reshape(1, -1)
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

