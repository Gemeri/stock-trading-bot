#!/usr/bin/env python3
"""
trend_continuation.py

A full i+1 walk-forward backtest of a LightGBM trend-continuation strategy,
with improved label and model tuning. Refactored for external orchestration.

Exports:
    FEATURES     : list of feature column names
    fit(X, y)    : returns a fitted LGBMClassifier with tuned hyperparameters
    predict(model, X) : returns array of continuation probabilities for X
"""

import os
import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Union

from catboost import CatBoostClassifier, Pool
from sub.common import compute_meta_labels, USE_META_LABEL
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# ─── Config ────────────────────────────────────────────────────────────────────

FEATURES = [
    'ema_9', 'ema_21', 'ema_50', 'ema_200',
    'adx', 'adx_trend', 'obv', 'close',
    'rsi', 'macd_line'
]

PARAM_GRID = {
    'iterations':      [600],
    'depth':           [6],
    'learning_rate':   [0.06],
    'l2_leaf_reg':     [3],
    'bagging_temperature':[0.1]
}

# ─── Helpers ───────────────────────────────────────────────────────────────────

def compute_labels(df, mode='momentum', n_ahead=2, thresh=0.002, quantile=0.4):
    """
    Enhanced label computation.
    mode = 'momentum':  label = 1 if fwd ret over n_ahead bars > thresh
    mode = 'quantile':  label = 1 if top quantile, 0 if bottom quantile, else NaN
    """
    df = df.copy()
    if USE_META_LABEL:
        return compute_meta_labels(df).rename(columns={'meta_label': 'label'})
    df['fwd_ret'] = (df['close'].shift(-n_ahead) - df['open']) / df['open']
    if mode == 'momentum':
        df['label'] = (df['fwd_ret'] > thresh).astype(int)
    elif mode == 'quantile':
        # Assign top quantile=1, bottom quantile=0, ignore middle
        q_hi = df['fwd_ret'].quantile(1 - quantile)
        q_lo = df['fwd_ret'].quantile(quantile)
        df['label'] = np.where(df['fwd_ret'] >= q_hi, 1,
                        np.where(df['fwd_ret'] <= q_lo, 0, np.nan))
    else:
        raise ValueError("Unknown mode for label: choose 'momentum' or 'quantile'")
    return df

def sharpe(returns):
    """Annualized Sharpe assuming 252 trades/year."""
    arr = np.asarray(returns)
    if arr.size == 0 or arr.std() == 0:
        return np.nan
    return arr.mean() / arr.std() * np.sqrt(252)

def max_drawdown(equity):
    """Max drawdown of a cumulative-equity series."""
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return dd.min()

# ─── Fit & Predict API ─────────────────────────────────────────────────────────

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]

def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Utility: convert Series/DataFrame/array-like to a 2-D (features) or 1-D (labels) NumPy array."""
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.values
    return np.asarray(x)

def fit(X_train: ArrayLike, y_train: ArrayLike) -> CatBoostClassifier:
    """
    Manual grid-search over PARAM_GRID using TimeSeriesSplit.
    Uses CatBoost with ordered boosting & early-stopping.
    Returns the best-fitted CatBoostClassifier.
    """
    logging.info("Running FIT (CatBoost) on trend_continuation")

    X_all = _to_numpy(X_train)
    y_all = _to_numpy(y_train).ravel()

    tscv        = TimeSeriesSplit(n_splits=5)
    best_score  = -np.inf
    best_params = None

    # iterate over parameter combinations
    for params in tqdm(ParameterGrid(PARAM_GRID), desc="Hyper-param grid", unit="combo"):
        fold_scores = []

        for tr_idx, val_idx in tscv.split(X_all):
            pool_tr  = Pool(X_all[tr_idx],  y_all[tr_idx])
            pool_val = Pool(X_all[val_idx], y_all[val_idx])

            mdl = CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="AUC",
                random_state=42,
                verbose=False,
                **params
            )
            mdl.fit(
                pool_tr,
                eval_set=pool_val,
                early_stopping_rounds=50,
                use_best_model=True,
            )
            prob_val   = mdl.predict_proba(pool_val)[:, 1]
            fold_scores.append(roc_auc_score(y_all[val_idx], prob_val))

        mean_auc = float(np.mean(fold_scores))
        if mean_auc > best_score:
            best_score, best_params = mean_auc, params

    logging.info(f"Best params: {best_params} | CV ROC-AUC: {best_score:.4f}")

    # retrain best model on *all* data
    full_pool   = Pool(X_all, y_all)
    best_model  = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        random_state=42,
        verbose=False,
        **best_params
    )
    best_model.fit(full_pool, early_stopping_rounds=50, use_best_model=True)
    return best_model


def predict(model: CatBoostClassifier, X: ArrayLike) -> np.ndarray:
    logging.info("Running PREDICT (CatBoost) on trend_continuation")
    X_np = _to_numpy(X)
    return model.predict_proba(X_np)[:, 1]

# ─── Main backtest (unchanged logic, but new label/threshold options) ─────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="CSV with columns 'open','close' + FEATURES")
    parser.add_argument("--output_dir", default="sub_result_trend", help="Where to save results")
    parser.add_argument("--n_back", type=int, default=500, help="Bars in final walk-forward backtest")
    parser.add_argument("--window", type=int, default=500, help="Rolling-window size for training each step")
    parser.add_argument("--holdout_frac", type=float, default=0.2, help="Fraction of backtest slice for threshold tuning")
    parser.add_argument("--commission", type=float, default=0.0005, help="Round-trip commission per trade (pct)")
    parser.add_argument("--slippage", type=float, default=0.0001, help="Round-trip slippage per trade (pct)")
    parser.add_argument("--stop_loss", type=float, default=0.02, help="Stop-loss pct (absolute on raw return)")
    parser.add_argument("--take_profit", type=float, default=0.05, help="Take-profit pct (absolute on raw return)")
    # New label & threshold args
    parser.add_argument("--label_mode", choices=["momentum", "quantile"], default="momentum", help="Label type")
    parser.add_argument("--label_n_ahead", type=int, default=2, help="Bars ahead for forward return in label")
    parser.add_argument("--label_thresh", type=float, default=0.002, help="Return threshold for momentum label")
    parser.add_argument("--label_quantile", type=float, default=0.4, help="Quantile for quantile label")
    parser.add_argument("--thresh_metric", choices=["sharpe", "f1", "roc_auc", "accuracy"], default="sharpe", help="Metric for threshold selection")
    args = parser.parse_args()

    # 1) Load & label
    df = pd.read_csv(args.input_csv).reset_index(drop=True)
    required = {'open','close'} | set(FEATURES)
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    df = compute_labels(
        df,
        mode=args.label_mode,
        n_ahead=args.label_n_ahead,
        thresh=args.label_thresh,
        quantile=args.label_quantile
    )
    df = df.dropna(subset=FEATURES + ['label']).reset_index(drop=True)

    # 2) Hyperparameter tuning on data *before* the last n_back bars
    hyper = df.iloc[:-args.n_back]
    X_h, y_h = hyper[FEATURES], hyper['label']

    best_model = fit(X_h, y_h)
    best_params = best_model.get_params()
    print("Best hyperparameters:", {k: best_params[k] for k in PARAM_GRID})

    # 3) True i+1 walk-forward on last n_back bars
    n     = len(df)
    start = n - args.n_back - 2   # need t+1 and t+2 for exit
    end   = n - 2                 # last t such that t+2 exists

    records = []
    for t in tqdm(range(start, end), desc="WF backtest", unit="bar"):
        # ---------------- rolling train slice ----------------
        tr0 = max(0, t - args.window)
        tr1 = t - 1
        train_df = df.iloc[tr0:tr1 + 1]

        # 15 % of the slice as chronological validation for early-stopping
        val_split = int(len(train_df) * 0.15)
        train_part = train_df.iloc[:-val_split]
        val_part   = train_df.iloc[-val_split:]

        X_tr, y_tr = train_part[FEATURES], train_part['label']
        X_val, y_val = val_part[FEATURES],  val_part['label']

        # ---------- CatBoost model (params taken from “best_params”) ----------
        cat_params = {
            'iterations'            : best_params.get('iterations', 1000),
            'learning_rate'         : best_params.get('learning_rate', 0.05),
            'depth'                 : best_params.get('depth', 6),
            'l2_leaf_reg'           : best_params.get('l2_leaf_reg', 3),
            'loss_function'         : 'Logloss',
            'eval_metric'           : 'AUC',
            'random_seed'           : 42,
            'verbose'               : False,
            'early_stopping_rounds' : 50,     # <-- key change
            'allow_writing_files'   : False   # avoid clutter
        }

        model = CatBoostClassifier(**cat_params)
        model.fit(
            Pool(X_tr, y_tr), 
            eval_set = Pool(X_val, y_val),
            use_best_model = True
        )

        # -------------- one-step-ahead probability --------------
        X_t  = df.loc[[t], FEATURES]
        p_t  = model.predict_proba(X_t)[:, 1][0]    # same shape as before

        # -------------- trade simulation math ------------------
        sign = 1
        op1  = df.at[t+1, 'open']
        op2  = df.at[t+2, 'open']
        raw_ret = sign * (op2 - op1) / op1
        clipped = np.clip(raw_ret, -args.stop_loss, +args.take_profit)
        net_ret = clipped - (args.commission + args.slippage)

        # -------------- logging --------------------------------
        records.append({
            't':           t,
            'prob':        p_t,
            'label':       df.at[t, 'label'],
            'raw_ret':     raw_ret,
            'clipped_ret': clipped,
            'net_ret':     net_ret
        })

    back = pd.DataFrame(records).set_index('t')

    # 4) Threshold tuning on validation slice
    n_val = int(len(back) * args.holdout_frac)
    val, test = back.iloc[:n_val], back.iloc[n_val:]

    best = {'thresh': None, 'score': -np.inf}
    metric = args.thresh_metric
    for thr in np.linspace(0.50, 0.90, 41):
        sel = val[val['prob'] > thr]
        if sel.empty:
            continue
        if metric == "sharpe":
            score = sharpe(sel['net_ret'])
        elif metric == "f1":
            score = f1_score(val['label'], (val['prob'] > thr).astype(int))
        elif metric == "roc_auc":
            score = roc_auc_score(val['label'], val['prob'])
        elif metric == "accuracy":
            score = accuracy_score(val['label'], (val['prob'] > thr).astype(int))
        else:
            raise ValueError("Invalid thresh_metric")
        if score > best['score']:
            best = {'thresh': thr, 'score': score}
    p_star = best['thresh']
    print(f"Optimized p* = {p_star:.3f}  (val {metric}={best['score']:.3f})")

    # 5) Evaluate on the remaining test slice
    sel_test = test[test['prob'] > p_star]
    rets     = sel_test['net_ret']
    equity   = (1 + rets).cumprod()

    total_ret = equity.iloc[-1] - 1 if not equity.empty else 0
    sharpe_t  = sharpe(rets)
    max_dd    = max_drawdown(equity) if not equity.empty else 0
    acc       = accuracy_score(test['label'], (test['prob'] > p_star).astype(int))
    auc       = roc_auc_score(test['label'], test['prob']) if len(test['label'].unique()) > 1 else float('nan')

    # 6) Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    back.to_csv(os.path.join(args.output_dir, "backtest_trend_lgb_full.csv"))

    plt.figure(figsize=(8,4))
    plt.plot(equity.index, equity.values, lw=1.5)
    plt.title("Equity Curve (Trend Continuation Test Slice)")
    plt.xlabel("Index")
    plt.ylabel("Cumulative Return")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "equity_curve_trend.png"))
    plt.close()

    # 7) Summary
    print("\n=== Test-Slice Performance ===")
    print(f"Trades taken       = {len(rets)}")
    print(f"Total return       = {total_ret*100: .2f}%")
    print(f"Sharpe (ann.)      = {sharpe_t:.2f}")
    print(f"Max drawdown       = {max_dd*100:.2f}%")
    print(f"Accuracy @ p*      = {acc:.3f}")
    print(f"ROC-AUC            = {auc:.3f}")
    print(f"All results in     = {args.output_dir}")

if __name__ == "__main__":
    main()
