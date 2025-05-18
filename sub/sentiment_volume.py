#!/usr/bin/env python3
"""
sentiment_volume_lightgbm.py

i+1 walk‐forward backtest of a LightGBM breakout classifier.
Now chooses threshold by maximum total return (not just Sharpe) and uses a larger validation slice.
"""

import os
import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

# ─── Config ────────────────────────────────────────────────────────────────────

FEATURES = [
    'sentiment', 'volume', 'volume_zscore',
    'obv', 'gap_vs_prev', 'wick_dominance'
]

LGBM_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.03,
    'num_leaves': 7,  # Strong regularization
    'objective': 'binary',
    'random_state': 42,
    'class_weight': 'balanced',  # helps if labels are imbalanced
    'min_child_samples': 50,     # prevent overfitting to noise
    'min_split_gain': 0.02,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
    'verbosity': -1
}

# ─── Helpers ───────────────────────────────────────────────────────────────────

def compute_labels(df: pd.DataFrame, min_return: float = None, breakout_pct: float = 0.15) -> pd.DataFrame:
    """
    Label a breakout (1) if next-bar open→open return is in the top X% of abs returns.
    By default, top 15% abs returns are breakouts.
    """
    df = df.copy()
    df['open_t1'] = df['open'].shift(-1)
    df['raw_ret'] = (df['open_t1'] - df['open']) / df['open']
    absrets = df['raw_ret'].abs()

    # Use percentile-based threshold if min_return not provided
    if min_return is None:
        min_return = absrets.quantile(1 - breakout_pct)
        print(f"[LABEL] Using percentile-based min_return: {min_return:.5f} for top {breakout_pct*100:.1f}% breakouts")

    df['label'] = (absrets > min_return).astype(int)

    # Print class balance
    label_counts = df['label'].value_counts(normalize=True)
    print(f"[LABEL BALANCE] Fraction breakout=1: {label_counts.get(1,0):.2f}, non-breakout=0: {label_counts.get(0,0):.2f}")
    return df

def sharpe(returns: np.ndarray) -> float:
    arr = np.asarray(returns)
    if arr.size == 0 or arr.std() == 0:
        return np.nan
    return arr.mean() / arr.std() * np.sqrt(252)

def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    return ((equity - roll_max) / roll_max).min()

def fit(X_train: pd.DataFrame, y_train: pd.Series) -> LGBMClassifier:
    model = LGBMClassifier(**LGBM_PARAMS)
    model.fit(X_train, y_train)
    return model

def predict(model: LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]

# ─── Main backtest ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Backtest sentiment‐volume anomaly breakout detector using LightGBM"
    )
    p.add_argument("input_csv", help="CSV with 'open' + FEATURES")
    p.add_argument("--output_dir", default="results-sentvol-lgbm", help="Where to save results")
    p.add_argument("--n_back", type=int, default=500, help="Bars in final walk‐forward backtest")
    p.add_argument("--window", type=int, default=500, help="Rolling‐window size for training each step")
    p.add_argument("--holdout_frac", type=float, default=0.4, help="Fraction of backtest slice for threshold tuning (use more for stability!)")
    p.add_argument("--min_return", type=float, default=None, help="Minimum abs. return to count as breakout (if not set, uses top 15% of returns)")
    p.add_argument("--breakout_pct", type=float, default=0.15, help="Breakout percentile if min_return not set (e.g. 0.15 = top 15%)")
    p.add_argument("--commission", type=float, default=0.0005, help="Round‐trip commission per trade (pct)")
    p.add_argument("--slippage", type=float, default=0.0001, help="Round‐trip slippage per trade (pct)")
    p.add_argument("--stop_loss", type=float, default=0.02, help="Stop‐loss pct on raw return")
    p.add_argument("--take_profit", type=float, default=0.05, help="Take‐profit pct on raw return")
    args = p.parse_args()

    # 1) Load & label
    df = pd.read_csv(args.input_csv).reset_index(drop=True)
    required = {'open'} | set(FEATURES)
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing column(s): {missing}")
    df = compute_labels(df, args.min_return, args.breakout_pct) \
           .dropna(subset=FEATURES + ['label']) \
           .reset_index(drop=True)

    # 2) Train on data before backtest
    hyper = df.iloc[:-args.n_back]
    X_h, y_h = hyper[FEATURES], hyper['label']
    model = fit(X_h, y_h)

    # 3) Walk‐forward backtest
    n     = len(df)
    start = n - args.n_back - 2
    end   = n - 2
    records = []

    for t in tqdm(range(start, end), desc="WF backtest", unit="bar", dynamic_ncols=True):
        tr0, tr1 = max(0, t - args.window), t - 1
        window_df = df.iloc[tr0:tr1+1]
        X_tr, y_tr = window_df[FEATURES], window_df['label']

        wnd_model = LGBMClassifier(**LGBM_PARAMS)
        wnd_model.fit(X_tr, y_tr)

        X_t = df.loc[[t], FEATURES]
        p_t = predict(wnd_model, X_t)[0]

        raw = df.at[t, 'raw_ret']
        sign = np.sign(raw) if raw != 0 else 1
        op1 = df.at[t+1, 'open']
        op2 = df.at[t+2, 'open']
        ret = sign * (op2 - op1) / op1

        clipped = np.clip(ret, -args.stop_loss, args.take_profit)
        net_ret = clipped - (args.commission + args.slippage)

        records.append({
            't':       t,
            'prob':    p_t,
            'label':   df.at[t, 'label'],
            'raw_ret': raw,
            'net_ret': net_ret
        })

    back = pd.DataFrame(records).set_index('t')

    # 4) Threshold tuning (choose threshold with maximum total return on validation slice)
    n_val = int(len(back) * args.holdout_frac)
    val, test = back.iloc[:n_val], back.iloc[n_val:]

    thresholds = np.linspace(0.5, 0.9, 41)
    summary_rows = []
    best = {'thresh': None, 'total_ret': -np.inf, 'sharpe': -np.inf}
    for thr in thresholds:
        sel = val[val['prob'] > thr]
        if sel.empty:
            continue
        sh = sharpe(sel['net_ret'])
        total_ret = (1 + sel['net_ret']).prod() - 1
        acc = accuracy_score(sel['label'], np.ones_like(sel['label'])) if len(sel) > 0 else np.nan
        summary_rows.append({
            "Threshold": thr,
            "Trades": len(sel),
            "Sharpe": sh,
            "TotalReturn": total_ret,
            "Accuracy": acc
        })
        if total_ret > best['total_ret']:
            best = {'thresh': thr, 'total_ret': total_ret, 'sharpe': sh}
    # Fallback if all are negative total return
    if best['thresh'] is None and summary_rows:
        best_row = max(summary_rows, key=lambda r: r["TotalReturn"])
        best = {'thresh': best_row["Threshold"], 'total_ret': best_row["TotalReturn"], 'sharpe': best_row["Sharpe"]}

    # Print sweep summary
    print("\nThreshold sweep summary (validation slice):")
    print(f"{'Threshold':>10} {'Trades':>7} {'Sharpe':>8} {'TotalReturn':>12} {'Accuracy':>8}")
    for row in summary_rows:
        print(f"{row['Threshold']:10.3f} {row['Trades']:7d} {row['Sharpe']:8.3f} {row['TotalReturn']:12.3f} {row['Accuracy']:8.3f}")

    p_star = best['thresh']
    print(f"\n[OPTIMIZED] Chosen p* = {p_star:.3f}  (Validation Total Return={best['total_ret']:.2f}, Sharpe={best['sharpe']:.2f})")

    # 5) Evaluate on test slice
    sel_test = test[test['prob'] > p_star]
    rets     = sel_test['net_ret']
    equity   = (1 + rets).cumprod()

    total_ret = equity.iloc[-1] - 1 if len(equity) > 0 else 0
    sharpe_t  = sharpe(rets)
    max_dd    = max_drawdown(equity)
    acc       = accuracy_score(test['label'], (test['prob'] > p_star).astype(int)) if len(test) > 0 else np.nan
    auc       = roc_auc_score(test['label'], test['prob']) if len(test) > 0 else np.nan

    # 6) Save and plot
    os.makedirs(args.output_dir, exist_ok=True)
    back.to_csv(os.path.join(args.output_dir, "backtest_sentvol_lgbm.csv"))

    plt.figure(figsize=(8,4))
    plt.plot(equity.index, equity.values, lw=1.5)
    plt.title("Equity Curve (Sentiment-Volume Anomaly with LightGBM)")
    plt.xlabel("Index"); plt.ylabel("Cumulative Return")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "equity_curve_lgbm.png"))
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
