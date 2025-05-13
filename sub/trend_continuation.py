#!/usr/bin/env python3
"""
trend_continuation.py

A full i+1 walk-forward backtest of a LightGBM trend-continuation strategy,
refactored to expose fit() and predict() for external orchestration.

Exports:
    FEATURES     : list of feature column names
    fit(X, y)    : returns a fitted LGBMClassifier with tuned hyperparameters
    predict(model, X) : returns array of continuation probabilities for X

Original backtest logic remains intact in main().
"""

import os
import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score

# ─── Silence LightGBM warnings ─────────────────────────────────────────────────
logging.getLogger('lightgbm').setLevel(logging.ERROR)

# ─── Config ────────────────────────────────────────────────────────────────────

FEATURES = [
    'ema_9', 'ema_21', 'ema_50', 'ema_200',
    'adx', 'adx_trend', 'obv', 'close',
    'rsi', 'macd_line'
]

PARAM_GRID = {
    'n_estimators':      [250],
    'num_leaves':        [31],
    'learning_rate':     [0.1],
    'subsample':         [0.7],
    'colsample_bytree':  [0.9],
    'min_child_samples': [20]
}

# ─── Helpers ───────────────────────────────────────────────────────────────────

def compute_labels(df):
    """
    Label = 1 if the direction of the next bar (open→open)
    matches the direction of the current bar; else 0.
    """
    df = df.copy()
    df['open_t1']      = df['open'].shift(-1)
    df['direction_t0'] = np.sign(df['open'] - df['open'].shift(1))
    df['direction_t1'] = np.sign(df['open_t1'] - df['open'])
    df['label']        = (df['direction_t1'] == df['direction_t0']).astype(int)
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

def fit(X_train: pd.DataFrame, y_train: pd.Series) -> LGBMClassifier:
    """
    Grid-search CV on (X_train, y_train) to tune hyperparameters,
    then return the best-fitted LGBMClassifier (best_estimator_).
    """
    logging.info("Ruinning FIT on trend_continuation")
    tscv = TimeSeriesSplit(n_splits=3)
    base = LGBMClassifier(random_state=42, verbosity=-1)
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

def predict(model: LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    """
    Given a fitted LGBMClassifier and DataFrame X (n_samples×n_features),
    return the probability of continuation (class=1) for each row.
    """
    logging.info("Ruinning PREDICT on trend_continuation")
    return model.predict_proba(X)[:, 1]


# ─── Main backtest (unchanged logic, now uses fit()/predict()) ────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv",
                        help="CSV with columns 'open','close' + FEATURES")
    parser.add_argument("--output_dir",   default=".",    help="Where to save results")
    parser.add_argument("--n_back",       type=int, default=500,
                        help="Bars in final walk-forward backtest")
    parser.add_argument("--window",       type=int, default=500,
                        help="Rolling-window size for training each step")
    parser.add_argument("--holdout_frac", type=float, default=0.2,
                        help="Fraction of backtest slice for threshold tuning")
    parser.add_argument("--commission",   type=float, default=0.0005,
                        help="Round-trip commission per trade (pct)")
    parser.add_argument("--slippage",     type=float, default=0.0001,
                        help="Round-trip slippage per trade (pct)")
    parser.add_argument("--stop_loss",    type=float, default=0.02,
                        help="Stop-loss pct (absolute on raw return)")
    parser.add_argument("--take_profit",  type=float, default=0.05,
                        help="Take-profit pct (absolute on raw return)")
    args = parser.parse_args()

    # 1) Load & label
    df = pd.read_csv(args.input_csv).reset_index(drop=True)
    required = {'open','close'} | set(FEATURES)
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    df = compute_labels(df).dropna(subset=FEATURES + ['label']).reset_index(drop=True)

    # 2) Hyperparameter tuning on data *before* the last n_back bars
    hyper = df.iloc[:-args.n_back]
    X_h, y_h = hyper[FEATURES], hyper['label']

    # use our refactored fit()
    best_model = fit(X_h, y_h)
    best_params = best_model.get_params()
    print("Best hyperparameters:", {k: best_params[k] for k in PARAM_GRID})

    # 3) True i+1 walk-forward on last n_back bars
    n     = len(df)
    start = n - args.n_back - 2   # need t+1 and t+2 for exit
    end   = n - 2                 # last t such that t+2 exists

    records = []
    for t in tqdm(range(start, end),
                  desc="WF backtest", unit="bar"):
        tr0 = max(0, t - args.window)
        tr1 = t - 1
        train = df.iloc[tr0:tr1+1]
        X_tr, y_tr = train[FEATURES], train['label']

        # fit with best params, quiet verbosity
        model = LGBMClassifier(**{k: best_params[k] for k in PARAM_GRID},
                               random_state=42,
                               verbosity=-1)
        model.fit(X_tr, y_tr)

        # predict prob of continuation on bar t
        X_t = df.loc[[t], FEATURES]
        p_t = predict(model, X_t)[0]

        # trade direction = direction_t0
        sign = int(df.at[t, 'direction_t0'])

        # simulate PnL: entry at t+1 open, exit at t+2 open
        op1 = df.at[t+1, 'open']
        op2 = df.at[t+2, 'open']
        raw_ret = sign * (op2 - op1) / op1

        # stop-loss / take-profit clipping
        clipped = np.clip(raw_ret,
                          -args.stop_loss,
                          +args.take_profit)

        net_ret = clipped - (args.commission + args.slippage)

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
