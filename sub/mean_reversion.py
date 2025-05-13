#!/usr/bin/env python3
"""
mean_reversion.py

A full i+1 walk‐forward backtest of a Random Forest mean‐reversion strategy,
refactored to expose fit() and predict() for external orchestration.

Exports:
    FEATURES     : list of feature column names
    fit(X, y)    : returns a fitted RandomForestClassifier with tuned hyperparameters
    predict(model, X) : returns array of reversal probabilities for X

Original backtest logic remains intact in main().
"""

import os
import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score

# ─── Config ────────────────────────────────────────────────────────────────────

FEATURES = [
    'bollinger_upper', 'bollinger_lower',
    'atr', 'atr_zscore',
    'candle_body_ratio', 'wick_dominance',
    'rsi_zscore', 'macd_histogram', 'macd_hist_flip'
]

PARAM_GRID = {
    'n_estimators':     [100, 250, 500, 1000],
    'max_depth':        [20],
    'min_samples_leaf':[10],
    'max_features':     ['sqrt', len(FEATURES)]
}


# ─── Helpers ───────────────────────────────────────────────────────────────────

def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 0/1 reversal label vs Bollinger mean.
    """
    df = df.copy()
    df['mean_band'] = (df['bollinger_upper'] + df['bollinger_lower']) / 2
    close = df['close']
    next_close = close.shift(-1)

    up_snap   = (close > df['mean_band']) & (next_close < close)
    down_snap = (close < df['mean_band']) & (next_close > close)
    df['label'] = (up_snap | down_snap).astype(int)
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
    drawdown = (equity - roll_max) / roll_max
    return drawdown.min()


# ─── Fit & Predict API ─────────────────────────────────────────────────────────

def fit(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Grid-search CV on (X_train, y_train) to tune hyperparameters,
    then return the best-fitted RandomForestClassifier.
    """
    logging.info("Ruinning FIT on mean_reversion")
    tscv = TimeSeriesSplit(n_splits=3)
    base = RandomForestClassifier(random_state=42, n_jobs=-1)
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

def predict(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    """
    Given a fitted RandomForestClassifier and feature matrix X
    (n_samples×n_features), return the probability of reversal (class=1).
    """
    logging.info("Ruinning PREDICT on mean_reversion")
    return model.predict_proba(X)[:, 1]


# ─── Main backtest (unchanged logic, now uses fit()/predict()) ────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_csv", help="CSV with 'open','close' + FEATURES")
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
    required = {'open','close','bollinger_upper','bollinger_lower'} | set(FEATURES)
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    df = compute_labels(df).dropna(subset=FEATURES + ['label']).reset_index(drop=True)

    # 2) Hyperparameter tuning on data *before* the last n_back bars
    hyper = df.iloc[:-args.n_back]
    X_h, y_h = hyper[FEATURES].values, hyper['label'].values

    best_model = fit(X_h, y_h)
    best_params = best_model.get_params()
    print("Best hyperparameters:",
          {k: best_params[k] for k in PARAM_GRID})

    # 3) True i+1 walk‐forward on last n_back bars
    n      = len(df)
    start  = n - args.n_back - 2   # need t+1 and t+2
    end    = n - 2
    records = []

    for t in tqdm(range(start, end),
                  desc="WF backtest", unit="bar", dynamic_ncols=True):

        tr0, tr1 = max(0, t - args.window), t - 1
        train    = df.iloc[tr0:tr1+1]
        X_tr     = train[FEATURES].values
        y_tr     = train['label'].values

        # retrain on window using best_params
        model = RandomForestClassifier(**{k: best_params[k] for k in PARAM_GRID},
                                       random_state=42, n_jobs=-1)
        model.fit(X_tr, y_tr)

        # predict prob of reversal on bar t
        x_t = df.loc[t, FEATURES].values.reshape(1, -1)
        p_t = predict(model, x_t)[0]

        # simulate PnL: entry at t+1 open, exit at t+2 open
        op1 = df.at[t+1, 'open']
        op2 = df.at[t+2, 'open']
        sign = +1 if df.at[t, 'close'] < df.at[t, 'mean_band'] else -1
        raw_ret = sign * (op2 - op1) / op1

        # apply stop‐loss / take‐profit & costs
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
    val, test = back.iloc[:n_val], back.iloc[n_val:]

    best = {'thresh': None, 'sharpe': -np.inf}
    for thr in np.linspace(0.50, 0.90, 41):
        sel = val[val['prob'] > thr]
        if sel.empty:
            continue
        s = sharpe(sel['net_ret'])
        if s > best['sharpe']:
            best = {'thresh': thr, 'sharpe': s}
    p_star = best['thresh']
    print(f"Optimized p* = {p_star:.3f}  (val Sharpe={best['sharpe']:.2f})")

    # 5) Evaluate on test slice
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
    back.to_csv(os.path.join(args.output_dir, "backtest_mean_reversion_rf_full.csv"))

    plt.figure(figsize=(8,4))
    plt.plot(equity.index, equity.values, lw=1.5)
    plt.title("Equity Curve (Mean‐Reversion Test Slice)")
    plt.xlabel("Index")
    plt.ylabel("Cumulative Return")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "equity_curve_mean_reversion.png"))
    plt.close()

    # 7) Summary
    print("\n=== Test‐Slice Performance ===")
    print(f"Trades taken       = {len(rets)}")
    print(f"Total return       = {total_ret*100: .2f}%")
    print(f"Sharpe (ann.)      = {sharpe_t:.2f}")
    print(f"Max drawdown       = {max_dd*100:.2f}%")
    print(f"Accuracy @ p*      = {acc:.3f}")
    print(f"ROC-AUC            = {auc:.3f}")
    print(f"All results in     = {args.output_dir}")

if __name__ == "__main__":
    main()
