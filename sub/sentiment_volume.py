#!/usr/bin/env python3
"""
sentiment_volume.py

i+1 walk‐forward backtest of a Decision Tree breakout classifier,
refactored to expose fit() and predict() for external orchestration.

Exports:
    FEATURES     : list of feature column names
    fit(X, y)    : returns a fitted DecisionTreeClassifier via GridSearchCV
    predict(model, X) : returns array of breakout probabilities for X

Original backtest logic remains intact in main().
"""

import os
import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score

# ─── Config ────────────────────────────────────────────────────────────────────

FEATURES = [
    'sentiment', 'volume', 'volume_zscore',
    'obv', 'gap_vs_prev', 'wick_dominance'
]

PARAM_GRID = {
    'criterion':        ['gini', 'entropy'],
    'max_depth':        [3, 5, 10, None],
    'min_samples_leaf': [1, 5, 10],
    'max_features':     ['sqrt', None]
}


# ─── Helpers ───────────────────────────────────────────────────────────────────

def compute_labels(df: pd.DataFrame, min_return: float) -> pd.DataFrame:
    """
    Label a breakout (1) if the next-bar open→open return
    exceeds ±min_return (absolute), else 0.
    Also store raw next-bar return.
    """
    df = df.copy()
    df['open_t1'] = df['open'].shift(-1)
    df['raw_ret'] = (df['open_t1'] - df['open']) / df['open']
    df['label']   = (df['raw_ret'].abs() > min_return).astype(int)
    return df

def sharpe(returns: np.ndarray) -> float:
    """Annualized Sharpe assuming 252 trades/year."""
    arr = np.asarray(returns)
    if arr.size == 0 or arr.std() == 0:
        return np.nan
    return arr.mean() / arr.std() * np.sqrt(252)

def max_drawdown(equity: pd.Series) -> float:
    """Max drawdown of a cumulative-equity series."""
    roll_max = equity.cummax()
    return ((equity - roll_max) / roll_max).min()


# ─── Fit & Predict API ─────────────────────────────────────────────────────────

def fit(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    """
    Grid-search CV on (X_train, y_train) to tune DecisionTree hyperparameters,
    then return the best-fitted classifier.
    """
    logging.info("Ruinning FIT on sentiment_volume")
    tscv = TimeSeriesSplit(n_splits=5)
    base = DecisionTreeClassifier(random_state=42)
    grid = GridSearchCV(
        estimator=base,
        param_grid=PARAM_GRID,
        cv=tscv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def predict(model: DecisionTreeClassifier, X: pd.DataFrame) -> np.ndarray:
    """
    Given a fitted DecisionTreeClassifier and DataFrame X,
    return the probability of breakout (class=1) for each row.
    """
    logging.info("Ruinning PREDICT on sentiment_volume")
    return model.predict_proba(X)[:, 1]


# ─── Main backtest (uses fit()/predict()) ──────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Backtest sentiment‐volume anomaly breakout detector"
    )
    p.add_argument("input_csv",
                   help="CSV with 'open' + FEATURES")
    p.add_argument("--output_dir",   default="results-sentvol",
                   help="Where to save results")
    p.add_argument("--n_back",       type=int,   default=500,
                   help="Bars in final walk‐forward backtest")
    p.add_argument("--window",       type=int,   default=500,
                   help="Rolling‐window size for training each step")
    p.add_argument("--holdout_frac", type=float, default=0.2,
                   help="Fraction of backtest slice for threshold tuning")
    p.add_argument("--min_return",   type=float, default=0.002,
                   help="Minimum abs. return to count as breakout (e.g. 0.002=0.2%)")
    p.add_argument("--commission",   type=float, default=0.0005,
                   help="Round‐trip commission per trade (pct)")
    p.add_argument("--slippage",     type=float, default=0.0001,
                   help="Round‐trip slippage per trade (pct)")
    p.add_argument("--stop_loss",    type=float, default=0.02,
                   help="Stop‐loss pct on raw return")
    p.add_argument("--take_profit",  type=float, default=0.05,
                   help="Take‐profit pct on raw return")
    args = p.parse_args()

    # 1) Load & label
    df = pd.read_csv(args.input_csv).reset_index(drop=True)
    required = {'open'} | set(FEATURES)
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing column(s): {missing}")
    df = compute_labels(df, args.min_return)\
           .dropna(subset=FEATURES + ['label'])\
           .reset_index(drop=True)

    # 2) Hyperparameter tuning on data *before* last n_back bars
    hyper = df.iloc[:-args.n_back]
    X_h, y_h = hyper[FEATURES], hyper['label']

    best_model = fit(X_h, y_h)
    best_params = best_model.get_params()
    print("Best hyperparameters:", {k: best_params[k] for k in PARAM_GRID})

    # 3) True i+1 walk‐forward backtest
    n     = len(df)
    start = n - args.n_back - 2   # need t+2 for exit
    end   = n - 2                 # last t such that t+2 exists

    records = []
    for t in tqdm(range(start, end),
                  desc="WF backtest", unit="bar", dynamic_ncols=True):

        tr0, tr1 = max(0, t - args.window), t - 1
        train = df.iloc[tr0:tr1+1]
        X_tr, y_tr = train[FEATURES], train['label']

        # retrain on window with best_params
        model = DecisionTreeClassifier(**{k: best_params[k] for k in PARAM_GRID},
                                       random_state=42)
        model.fit(X_tr, y_tr)

        # predict probability on bar t
        X_t = df.loc[[t], FEATURES]
        p_t = predict(model, X_t)[0]

        raw  = df.at[t, 'raw_ret']
        sign = np.sign(raw) if raw != 0 else 1

        # simulate PnL: entry at t+1 open, exit at t+2 open
        op1 = df.at[t+1, 'open']
        op2 = df.at[t+2, 'open']
        ret = sign * (op2 - op1) / op1

        clipped = np.clip(ret,
                          -args.stop_loss,
                          args.take_profit)
        net_ret = clipped - (args.commission + args.slippage)

        records.append({
            't':       t,
            'prob':    p_t,
            'label':   df.at[t, 'label'],
            'raw_ret': raw,
            'net_ret': net_ret
        })

    back = pd.DataFrame(records).set_index('t')

    # 4) Threshold tuning on validation slice
    n_val = int(len(back) * args.holdout_frac)
    val, test = back.iloc[:n_val], back.iloc[n_val:]

    best = {'thresh': None, 'sharpe': -np.inf}
    for thr in np.linspace(0.5, 0.9, 41):
        sel = val[val['prob'] > thr]
        if sel.empty:
            continue
        sh = sharpe(sel['net_ret'])
        if sh > best['sharpe']:
            best = {'thresh': thr, 'sharpe': sh}
    p_star = best['thresh']
    print(f"Optimized p* = {p_star:.3f}  (val Sharpe={best['sharpe']:.2f})")

    # 5) Evaluate on test slice
    sel_test = test[test['prob'] > p_star]
    rets     = sel_test['net_ret']
    equity   = (1 + rets).cumprod()

    total_ret = equity.iloc[-1] - 1
    sharpe_t  = sharpe(rets)
    max_dd    = max_drawdown(equity)
    acc       = accuracy_score(test['label'], (test['prob'] > p_star).astype(int))
    auc       = roc_auc_score(test['label'], test['prob'])

    # 6) Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    back.to_csv(os.path.join(args.output_dir, "backtest_sentvol_dt_full.csv"))

    plt.figure(figsize=(8,4))
    plt.plot(equity.index, equity.values, lw=1.5)
    plt.title("Equity Curve (Sentiment-Volume Anomaly)")
    plt.xlabel("Index"); plt.ylabel("Cumulative Return")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "equity_curve.png"))
    plt.close()

    # 7) Summary
    print("\n=== Test-Slice Performance ===")
    print(f"Trades taken       = {len(rets)}")
    print(f"Total return       = {total_ret*100:.2f}%")
    print(f"Sharpe (ann.)      = {sharpe_t:.2f}")
    print(f"Max drawdown       = {max_dd*100:.2f}%")
    print(f"Accuracy @ p*      = {acc:.3f}")
    print(f"ROC-AUC            = {auc:.3f}")
    print(f"All results in     = {args.output_dir}")

if __name__ == "__main__":
    main()
