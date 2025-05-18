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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# ─── Config ────────────────────────────────────────────────────────────────────

FEATURES = [
    'bollinger_upper', 'bollinger_lower',
    'atr', 'atr_zscore',
    'candle_body_ratio', 'wick_dominance',
    'rsi_zscore', 'macd_histogram', 'macd_hist_flip'
]

PARAM_GRID = {
    'n_estimators':     [250, 500, 1000],
    'max_depth':        [5, 10, 20],
    'min_samples_leaf': [1, 3, 5],
    'max_features':     [len(FEATURES)]
}

# ─── Helpers ───────────────────────────────────────────────────────────────────

def compute_labels(df):
    """
    Label = 1 if any of the next 2 bars snaps back toward the mean 
    by ≥ 1×ATR, else 0.  Also stores `mean_band`.
    """
    df = df.copy()
    df['mean_band'] = (df['bollinger_upper'] + df['bollinger_lower']) / 2
    c0 = df['close']
    c1 = c0.shift(-1)
    c2 = c0.shift(-2)
    # upstream snaps
    up1 = (c0 > df['mean_band']) & ((c0 - c1) >= df['atr'])
    up2 = (c0 > df['mean_band']) & ((c0 - c2) >= df['atr'])
    # downstream snaps
    dn1 = (c0 < df['mean_band']) & ((c1 - c0) >= df['atr'])
    dn2 = (c0 < df['mean_band']) & ((c2 - c0) >= df['atr'])
    df['label'] = (up1|up2|dn1|dn2).astype(int)
    return df.dropna(subset=FEATURES + ['label'])

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

def fit(df, n_back=500, window=500, holdout_frac=0.2):
    """
    1) compute_labels on entire df
    2) tune RF on df[:-n_back]
    3) simulate walk‐forward on last n_back bars to get probabilities
    4) pick 90th percentile of val‐slice as threshold

    Returns a `state` dict with:
      • state['best_params']
      • state['p_star']
    """
    df2 = compute_labels(df).reset_index(drop=True)

    # 2) hyperparam tuning
    hyper = df2.iloc[:-n_back]
    Xh, yh = hyper[FEATURES].values, hyper['label'].values
    tscv = TimeSeriesSplit(n_splits=5)
    rf   = RandomForestClassifier(
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    grid = GridSearchCV(rf, PARAM_GRID, cv=tscv,
                        scoring='roc_auc', n_jobs=-1, verbose=1)
    grid.fit(Xh, yh)
    best = grid.best_params_

    # 3) walk‐forward to collect probs
    records = []
    start = len(df2) - n_back - 2
    end   = len(df2) - 2
    for t in range(start, end):
        tr0 = max(0, t - window)
        tr1 = t - 1
        train = df2.iloc[tr0:tr1+1]
        model = RandomForestClassifier(
            **best, class_weight='balanced', random_state=42, n_jobs=-1
        )
        model.fit(train[FEATURES], train['label'])
        pt = model.predict_proba(df2.loc[[t], FEATURES])[:,1][0]
        records.append(pt)
    back = pd.DataFrame({'prob': records})
    
    # 4) 90th‐percentile threshold on first val slice
    nv = int(len(back) * holdout_frac)
    p_star = back['prob'].iloc[:nv].quantile(0.90)

    return {'best_params': best, 'p_star': float(p_star)}

def predict(df, t, state, window=500):
    """
    For bar index t, retrain on the preceding `window` bars
    and return the probability of snap‐back.
    """
    df2 = compute_labels(df).reset_index(drop=True)
    best = state['best_params']
    tr0  = max(0, t - window)
    tr1  = t - 1
    train = df2.iloc[tr0:tr1+1]
    model = RandomForestClassifier(
        **best, class_weight='balanced', random_state=42, n_jobs=-1
    )
    model.fit(train[FEATURES], train['label'])
    return model.predict_proba(df2.loc[[t], FEATURES])[:,1][0]

# ─── Stand‐Alone Backtest Entrypoint ───────────────────────────────────────────

if __name__=="__main__":
    import argparse
    import matplotlib.pyplot as plt
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
                   help="Take-profit pct (absolute on raw return)")
    args = p.parse_args()

    # full backtest, exactly as before
    df = pd.read_csv(args.input_csv).reset_index(drop=True)
    df = compute_labels(df).reset_index(drop=True)

    state = fit(df,
                n_back=args.n_back,
                window=args.window,
                holdout_frac=args.holdout_frac)

    records = []
    start = len(df) - args.n_back - 2
    end   = len(df) - 2
    for t in tqdm(range(start, end),
                  desc="WF backtest", unit="bar"):
        pt = predict(df, t, state, window=args.window)
        op1 = df.at[t+1, 'open']
        op2 = df.at[t+2, 'open']
        sign = +1 if df.at[t, 'close'] < df.at[t, 'mean_band'] else -1
        raw = sign * (op2 - op1) / op1
        clipped = np.clip(raw, -args.stop_loss, args.take_profit)
        net = clipped - (args.commission + args.slippage)
        records.append({'t':t, 'prob':pt, 'label':df.at[t,'label'], 'net_ret':net})

    back = pd.DataFrame(records).set_index('t')
    # apply threshold
    nv   = int(len(back)*args.holdout_frac)
    val  = back.iloc[:nv]
    test = back.iloc[nv:]
    p_star = state['p_star']
    sel = test[test['prob'] > p_star]
    rets = sel['net_ret']
    eq   = (1+rets).cumprod()

    # metrics
    total   = eq.iloc[-1] - 1
    sr      = sharpe(rets)
    mdd     = max_drawdown(eq)
    acc     = accuracy_score(test['label'], (test['prob']>p_star).astype(int))
    auc     = roc_auc_score(test['label'], test['prob'])

    # save
    os.makedirs(args.output_dir, exist_ok=True)
    back.to_csv(os.path.join(args.output_dir, "backtest_rf_refined.csv"))
    plt.figure(figsize=(8,4))
    plt.plot(eq, lw=1.5)
    plt.title("Equity Curve (Test Slice)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "equity_curve.png"))

    print("\n=== Test‐Slice Performance ===")
    print(f"Trades taken       = {len(rets)}")
    print(f"Total return       = {total*100: .2f}%")
    print(f"Sharpe (ann.)      = {sr:.2f}")
    print(f"Max drawdown       = {mdd*100: .2f}%")
    print(f"Accuracy @ p*      = {acc:.3f}")
    print(f"ROC-AUC            = {auc:.3f}")
    print(f"Results saved to   = {args.output_dir}")
