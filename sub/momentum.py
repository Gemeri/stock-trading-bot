#!/usr/bin/env python3
"""
momentum.py

A full i+1 walk‐forward backtest of an XGBoost momentum strategy,
refactored to expose fit() and predict() for external orchestration.

Exports:
    FEATURES     : list of feature column names
    fit(X, y)    : returns a fitted XGBClassifier with tuned hyperparameters
    predict(model, X) : returns array of up‐move probabilities for X

Original backtest logic remains intact in main().
"""

import os
import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score

# ─── Config ────────────────────────────────────────────────────────────────────

# 1) Features the momentum model uses
FEATURES = [
    'rsi', 'momentum', 'roc',
    'macd_histogram', 'macd_line', 'macd_signal',
    'rsi_zscore', 'macd_cross', 'macd_hist_flip'
]

# 2) Hyperparameter grid for tuning
PARAM_GRID = {
    'n_estimators':     [100],
    'max_depth':        [10],
    'learning_rate':    [0.05],
    'subsample':        [0.8],
    'colsample_bytree': [0.8]
}


# ─── Helpers ───────────────────────────────────────────────────────────────────

def compute_labels(df):
    """
    Compute binary momentum label:
      label = 1 if next bar's open > this bar's open, else 0.
    """
    df = df.copy()
    df['open_t1'] = df['open'].shift(-1)
    df['label']   = (df['open_t1'] > df['open']).astype(int)
    return df

def sharpe(returns):
    """Annualized Sharpe assuming 252 trades/year."""
    if len(returns) == 0 or returns.std() == 0:
        return np.nan
    return returns.mean() / returns.std() * np.sqrt(252)

def max_drawdown(equity):
    """Max drawdown of a cumulative-equity series."""
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return drawdown.min()


# ─── Fit & Predict API ─────────────────────────────────────────────────────────

def fit(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    """
    Grid-search CV on (X_train, y_train) to tune hyperparameters,
    then return the best-fitted XGBClassifier (best_estimator_).
    """
    logging.info("Running FIT on momentum")
    tscv = TimeSeriesSplit(n_splits=5)
    base = XGBClassifier(use_label_encoder=False,
                         eval_metric='logloss',
                         verbosity=0,
                         random_state=42)
    grid = GridSearchCV(
        estimator=base,
        param_grid=PARAM_GRID,
        cv=tscv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    # best_estimator_ is already fitted on the hyperparameter slice
    return grid.best_estimator_


def predict(model: XGBClassifier, X: np.ndarray) -> np.ndarray:
    """
    Given a fitted XGBClassifier and feature matrix X (n_samples×n_features),
    return the probability of the 'up' class for each row.
    """
    logging.info("Running PREDICT on momentum")
    return model.predict_proba(X)[:, 1]


# ─── Main backtest (unchanged logic, now uses fit()/predict()) ────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_csv",
                   help="CSV with 'open' + FEATURES")
    p.add_argument("--output_dir",   default=".",    help="Where to save results")
    p.add_argument("--n_back",       type=int, default=500,
                   help="Bars in final walk‐forward backtest")
    p.add_argument("--window",       type=int, default=500,
                   help="Rolling‐window size for training each step")
    p.add_argument("--holdout_frac", type=float, default=0.2,
                   help="Frac of backtest slice used for threshold tuning")
    p.add_argument("--commission",   type=float, default=0.0005,
                   help="Round‐trip commission per trade (pct)")
    p.add_argument("--slippage",     type=float, default=0.0001,
                   help="Round‐trip slippage per trade (pct)")
    p.add_argument("--stop_loss",    type=float, default=0.02,
                   help="Stop‐loss pct (absolute on raw return)")
    p.add_argument("--take_profit",  type=float, default=0.05,
                   help="Take‐profit pct (absolute on raw return)")
    args = p.parse_args()

    # 1) Load & label
    df = pd.read_csv(args.input_csv).reset_index(drop=True)
    required = {'open'} | set(FEATURES)
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    df = compute_labels(df).dropna(subset=FEATURES + ['label']).reset_index(drop=True)

    # 2) Hyperparameter tuning on data *before* the last n_back bars
    hyper = df.iloc[:-args.n_back]
    X_h, y_h = hyper[FEATURES].values, hyper['label'].values

    # use our refactored fit()
    best_model = fit(X_h, y_h)
    best_params = best_model.get_params()
    print("Best hyperparameters:", {k:best_params[k] for k in PARAM_GRID})

    # 3) True i+1 walk‐forward on last n_back bars
    n      = len(df)
    start  = n - args.n_back - 2  # need t+1 and t+2
    end    = n - 2
    records = []

    for t in tqdm(range(start, end),
                  desc="WF backtest", unit="bar", dynamic_ncols=True):

        # rolling-window train
        tr0, tr1 = max(0, t - args.window), t - 1
        train    = df.iloc[tr0:tr1+1]
        X_tr     = train[FEATURES].values
        y_tr     = train['label'].values

        # fit a fresh model with the best params
        model = XGBClassifier(**{k: best_params[k] for k in PARAM_GRID},
                              use_label_encoder=False,
                              eval_metric='logloss',
                              verbosity=0,
                              random_state=42)
        model.fit(X_tr, y_tr)

        # predict probability on bar t
        x_t = df.loc[t, FEATURES].values.reshape(1, -1)
        p_t = predict(model, x_t)[0]

        # simulate PnL: entry at t+1 open, exit at t+2 open
        op1 = df.at[t+1, 'open']
        op2 = df.at[t+2, 'open']
        raw_ret = (op2 - op1) / op1

        # apply stops/takes & costs
        clipped = np.clip(raw_ret,
                          -args.stop_loss,
                          +args.take_profit)
        net_ret = clipped - (args.commission + args.slippage)

        records.append({
            't':          t,
            'prob':       p_t,
            'label':      df.at[t, 'label'],
            'raw_ret':    raw_ret,
            'clipped_ret':clipped,
            'net_ret':    net_ret
        })

    back = pd.DataFrame(records).set_index('t')

    # 4) Threshold tuning on validation slice
    n_val = int(len(back) * args.holdout_frac)
    val   = back.iloc[:n_val]
    test  = back.iloc[n_val:]

    best = {'thresh': None, 'sharpe': -np.inf}
    for thr in np.linspace(0.50, 0.90, 41):
        sel = val[val['prob'] > thr]
        if sel.empty:
            continue
        sh = sharpe(sel['net_ret'])
        if sh > best['sharpe']:
            best = {'thresh': thr, 'sharpe': sh}
    p_star = best['thresh']
    print(f"Optimized p* = {p_star:.3f}  (val Sharpe={best['sharpe']:.2f})")

    # 5) Evaluate on the remaining test slice
    sel_test = test[test['prob'] > p_star]
    rets     = sel_test['net_ret']
    equity   = (1 + rets).cumprod()

    total_ret = equity.iloc[-1] - 1
    sharpe_t  = sharpe(rets)
    max_dd    = max_drawdown(equity)
    acc       = accuracy_score(test['label'],
                               (test['prob'] > p_star).astype(int))
    auc       = roc_auc_score(test['label'], test['prob'])

    # 6) Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    back.to_csv(os.path.join(args.output_dir, "backtest_momentum_xgb_full.csv"))

    plt.figure(figsize=(8,4))
    plt.plot(equity.index, equity.values, lw=1.5)
    plt.title("Equity Curve (Momentum Test Slice)")
    plt.xlabel("Index")
    plt.ylabel("Cumulative Return")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "equity_curve_momentum.png"))
    plt.close()

    # 7) Summary
    print("\n=== Test-Slice Performance ===")
    print(f"Trades taken       = {len(rets)}")
    print(f"Total return       = {total_ret*100: .2f}%")
    print(f"Sharpe (ann.)      = {sharpe_t: .2f}")
    print(f"Max drawdown       = {max_dd*100: .2f}%")
    print(f"Accuracy @ p*      = {acc: .3f}")
    print(f"ROC-AUC            = {auc:.3f}")
    print(f"All results in     = {args.output_dir}")

if __name__ == "__main__":
    main()
