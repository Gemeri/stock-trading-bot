#!/usr/bin/env python3
"""
backtest_mean_reversion_rf_full_refined.py

A full i+1 walk‐forward backtest of a Random Forest mean‐reversion strategy:
 1) “Meaningful” snap‐back labels (≥1×ATR, 2‐bar horizon)
 2) Grid‐search CV for hyperparameters (no future leakage)
 3) True i+1 walk‐forward on last n_back bars
 4) Quantile‐filter threshold (top 10% signals)
 5) PnL sim: next‐open entry, following‐open exit
    with commission, slippage, full‐capital sizing,
    and simple stop‐loss / take‐profit
 6) Metrics: total return, Sharpe, max drawdown, accuracy, ROC‐AUC

Also exposes two functions for meta‐modeling:
  • fit(df, n_back, window, holdout_frac)
  • predict(df, t, state, window)
"""

import os
import numpy as np
import pandas as pd
import logging
import argparse

import matplotlib.pyplot as plt
from sub.common import compute_meta_labels, USE_META_LABEL
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ─── Config ────────────────────────────────────────────────────────────────────

FEATURES = [
    'bollinger_upper', 'bollinger_lower',
    'atr', 'atr_zscore',
    'candle_body_ratio', 'wick_dominance',
    'rsi_zscore', 'macd_histogram', 'macd_hist_flip'
]

PARAM_GRID = {
    'n_estimators':     [250],
    'max_depth':        [5, 10, 20],
    'min_samples_leaf': [1],
    'max_features':     [len(FEATURES)]
}

# ─── Helpers ───────────────────────────────────────────────────────────────────

def compute_labels(df):
    """
    Same snap-back definition, now guarantees the returned frame
    has no missing values in FEATURES + label and still supports
    USE_META_LABEL override.
    """
    df = df.copy()

    if USE_META_LABEL:
        # hand off to meta-label helper exactly as before
        return compute_meta_labels(df).rename(columns={'meta_label': 'label'})

    df['mean_band'] = (df['bollinger_upper'] + df['bollinger_lower']) / 2
    c0, c1, c2 = df['close'], df['close'].shift(-1), df['close'].shift(-2)

    up1 = (c0 > df['mean_band']) & ((c0 - c1) >= df['atr'])
    up2 = (c0 > df['mean_band']) & ((c0 - c2) >= df['atr'])
    dn1 = (c0 < df['mean_band']) & ((c1 - c0) >= df['atr'])
    dn2 = (c0 < df['mean_band']) & ((c2 - c0) >= df['atr'])

    df['label'] = (up1 | up2 | dn1 | dn2).astype(int)

    # Drop any row still missing required features or label
    return df.dropna(subset=FEATURES + ['label']).reset_index(drop=True)

def sharpe(returns):
    """Annualized Sharpe assuming 252 trades/year."""
    if len(returns)==0 or returns.std()==0:
        return np.nan
    return returns.mean()/returns.std()*np.sqrt(252)

def max_drawdown(equity):
    """Max drawdown of a cumulative‐equity series."""
    roll_max = equity.cummax()
    return ((equity - roll_max)/roll_max).min()

# ─── Fit & Predict for Meta‐Model ───────────────────────────────────────────────

def fit(X_train: np.ndarray, y_train: np.ndarray):
    """
    • Time-series GridSearchCV tunes the Random Forest hyper-params.  
    • Best RF is then **isotonic-calibrated** with another TS split.  
    • Returns the calibrated model while still exposing .get_params()
      for external cloning inside main.py.
    """
    logging.info("Running FIT on mean_reversion (TS-CV + calibration)")

    tscv = TimeSeriesSplit(n_splits=5)

    base_rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    grid = GridSearchCV(
        estimator=base_rf,
        param_grid=PARAM_GRID,
        cv=tscv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    ).fit(X_train, y_train)

    best_rf = grid.best_estimator_
    logging.info("Best params ⇒ %s", grid.best_params_)

    # ── calibrate with isotonic regression using *another* TS split ──
    calib = CalibratedClassifierCV(
        best_rf, method="isotonic", cv=TimeSeriesSplit(n_splits=3)
    ).fit(X_train, y_train)

    # expose RF params so walk-forward code can re-instantiate fresh RFs
    _rf_params = best_rf.get_params()

    def _get_params(deep=True):
        return {k: _rf_params[k] for k in PARAM_GRID}

    calib.get_params = _get_params  # monkey-patch for compatibility
    return calib


def predict(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    """
    Given a fitted RandomForestClassifier and feature matrix X
    (n_samples × n_features), return the probability of the 'snap-back'
    class for each row.
    """
    logging.info("Running PREDICT on mean_reversion")
    return model.predict_proba(X)[:, 1]

# ─── Stand‐Alone Backtest Entrypoint ───────────────────────────────────────────

if __name__=="__main__":
    from tqdm import tqdm
    from sklearn.metrics import accuracy_score, roc_auc_score

    p = argparse.ArgumentParser()
    p.add_argument("input_csv", help="CSV with 'open','close'+FEATURES")
    p.add_argument("--output_dir",   default=".",    help="Where to save results")
    p.add_argument("--n_back",       type=int,   default=500,
                   help="Bars in final walk‐forward backtest")
    p.add_argument("--window",       type=int,   default=500,
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
    df = compute_labels(df).reset_index(drop=True)

    # 2) Hyperparameter tuning on all but the last n_back bars
    df2 = df.copy()
    hyper = df2.iloc[:-args.n_back]
    X_h, y_h = hyper[FEATURES].values, hyper['label'].values

    # fit returns a fitted RandomForestClassifier
    model = fit(X_h, y_h)
    best_params = model.get_params()
    print("Best hyperparameters:", {k: best_params[k] for k in PARAM_GRID})

    # 3) True i+1 walk‐forward on last n_back bars
    records = []
    start = len(df) - args.n_back - 2
    end   = len(df) - 2

    for t in tqdm(range(start, end), desc="WF backtest", unit="bar"):
        # rolling‐window train slice
        tr0, tr1 = max(0, t - args.window), t - 1
        train = df.iloc[tr0:tr1+1]
        X_tr, y_tr = train[FEATURES].values, train['label'].values

        # retrain fresh model with best_params
        m = RandomForestClassifier(
            **{k: best_params[k] for k in PARAM_GRID},
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        m.fit(X_tr, y_tr)

        # predict snap‐back prob on bar t
        x_t = df.loc[[t], FEATURES].values
        p_t = predict(m, x_t)[0]

        # simulate PnL: entry at open t+1, exit at open t+2
        op1 = df.at[t+1, 'open']
        op2 = df.at[t+2, 'open']
        raw   = ((op2 - op1) / op1)
        clipped = np.clip(raw, -args.stop_loss, args.take_profit)
        net_ret = clipped - (args.commission + args.slippage)

        records.append({'t':t, 'prob':p_t, 'label':df.at[t,'label'], 'net_ret':net_ret})

    back = pd.DataFrame(records).set_index('t')

    # 4) Threshold tuning on validation slice
    nv  = int(len(back) * args.holdout_frac)
    val = back.iloc[:nv]
    best = {'sharpe': -np.inf, 'thresh': None}
    for thr in np.linspace(0.50, 0.90, 41):
        sel = val[val['prob'] > thr]
        if sel.empty: continue
        sh = sharpe(sel['net_ret'])
        if sh > best['sharpe']:
            best = {'sharpe': sh, 'thresh': thr}
    p_star = best['thresh']
    print(f"Optimized p* = {p_star:.3f}  (val Sharpe={best['sharpe']:.2f})")

    # 5) Evaluate on test slice
    test = back.iloc[nv:]
    sel  = test[test['prob'] > p_star]
    rets = sel['net_ret']
    eq   = (1 + rets).cumprod()

    # metrics
    total = eq.iloc[-1] - 1
    sr    = sharpe(rets)
    mdd   = max_drawdown(eq)
    acc   = accuracy_score(test['label'], (test['prob']>p_star).astype(int))
    auc   = roc_auc_score(test['label'], test['prob'])

    # save
    os.makedirs(args.output_dir, exist_ok=True)
    back.to_csv(os.path.join(args.output_dir, "backtest_rf_refined.csv"))
    plt.figure(figsize=(8,4))
    plt.plot(eq, lw=1.5)
    plt.title("Equity Curve (Test Slice)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "equity_curve.png"))

    # summary
    print("\n=== Test‐Slice Performance ===")
    print(f"Trades taken       = {len(rets)}")
    print(f"Total return       = {total*100: .2f}%")
    print(f"Sharpe (ann.)      = {sr:.2f}")
    print(f"Max drawdown       = {mdd*100:.2f}%")
    print(f"Accuracy @ p*      = {acc:.3f}")
    print(f"ROC-AUC            = {auc:.3f}")
    print(f"Results saved to   = {args.output_dir}")
