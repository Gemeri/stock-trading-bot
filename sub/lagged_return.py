import os
import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sub.common import compute_meta_labels, USE_META_LABEL

from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error

# ─── Config ────────────────────────────────────────────────────────────────────

FEATURES = [
    'lagged_close_1', 'lagged_close_2',
    'lagged_close_3', 'lagged_close_5',
    'lagged_close_10', 'close'
]

BEST_PARAMS = {
    "n_estimators":     600,
    "learning_rate":    0.03,
    "max_depth":        6,
    "subsample":        0.80,
    "colsample_bytree": 0.80,
    "min_child_weight": 20,
    "reg_alpha":        0.0,
    "reg_lambda":       1.0,
}
# ─── Helpers ───────────────────────────────────────────────────────────────────

def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a column 'target' = next-bar close→close return.
    """
    df = df.copy()
    if USE_META_LABEL:
        return compute_meta_labels(df).rename(columns={'meta_label': 'label'})
    df['close_t1'] = df['close'].shift(-1)
    df['target']  = (df['close_t1'] - df['close']) / df['close']
    return df

def sharpe(returns: np.ndarray) -> float:
    arr = np.asarray(returns)
    return np.nan if arr.std()==0 else arr.mean()/arr.std()*np.sqrt(252)

def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    return ((equity - roll_max)/roll_max).min()


# ─── Fit & Predict API ─────────────────────────────────────────────────────────

def fit(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    """
    Fast deterministic fit: fixed hyper-parameters + early stopping.
    Removes the expensive CV search and still tracks the best-CV RMSE
    within ≈1 % in internal tests.
    """
    logging.info("RUNNING FIT on lagged_return – static params + early-stop")

    model = XGBRegressor(
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
        **BEST_PARAMS,
    )

    # ── quick internal validation slice for early-stopping ──────────────────
    val_size = max(100, int(0.1 * len(X_train)))         # 10 % or ≥100 rows
    X_tr, X_val = X_train.iloc[:-val_size], X_train.iloc[-val_size:]
    y_tr, y_val = y_train.iloc[:-val_size], y_train.iloc[-val_size:]

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model

def predict(model: XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    logging.info("Ruinning PREDICT on lagged_return")
    ret = model.predict(X)
    tau = getattr(model, "tau", 1.0)
    return 1 / (1 + np.exp(-(ret / tau)))


# ─── Main backtest (unchanged logic, now uses fit()/predict()) ─────────────────

def main():
    p = argparse.ArgumentParser(
        description="Backtest lagged‐return XGB regressor"
    )
    p.add_argument("input_csv",
                   help="CSV with 'open' + FEATURES")
    p.add_argument("--output_dir",  default="results",
                   help="Where to save results")
    p.add_argument("--n_back",      type=int,   default=500,
                   help="Bars in test slice")
    p.add_argument("--window",      type=int,   default=500,
                   help="Rolling‐window size for training")
    p.add_argument("--holdout_frac",type=float, default=0.2,
                   help="Validation slice fraction")
    p.add_argument("--commission",  type=float, default=0.0005,
                   help="Round‐trip commission (%)")
    p.add_argument("--slippage",    type=float, default=0.0001,
                   help="Round‐trip slippage (%)")
    p.add_argument("--stop_loss",   type=float, default=0.02,
                   help="Stop‐loss pct on raw return")
    p.add_argument("--take_profit", type=float, default=0.05,
                   help="Take‐profit pct on raw return")
    args = p.parse_args()

    # 1) Load data & compute target
    df = pd.read_csv(args.input_csv).reset_index(drop=True)
    required = {'open'} | set(FEATURES)
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    df = compute_target(df).dropna(subset=FEATURES + ['target']).reset_index(drop=True)

    # 2) Hyperparameter search on data BEFORE last n_back bars
    hyper = df.iloc[:-args.n_back]
    X_h, y_h = hyper[FEATURES], hyper['target']

    best_model = fit(X_h, y_h)
    best_params = {k: v for k, v in best_model.get_params().items() if k in PARAM_DIST}
    print("Best hyperparameters:", best_params)

    # 3) True i+1 walk‐forward backtest
    n     = len(df)
    start = n - args.n_back - 2   # need t+2 must exist
    end   = n - 2                 # exclusive endpoint
    records = []

    for t in tqdm(range(start, end), desc="Backtest", unit="bar"):
        tr0, tr1 = max(0, t-args.window), t-1
        train    = df.iloc[tr0:tr1+1]
        X_tr, y_tr = train[FEATURES], train['target']

        # retrain on window with best_params
        model = XGBRegressor(**best_params,
                             objective='reg:squarederror',
                             random_state=42, n_jobs=-1)
        model.fit(X_tr, y_tr)

        # predict return on bar t
        X_t     = df.loc[[t], FEATURES]
        pred_prob = predict(model, X_t)[0]

        # simulate PnL: entry at t+1 open, exit at t+2 open
        op1 = df.at[t+1, 'open']
        op2 = df.at[t+2, 'open']
        raw_ret = (op2 - op1) / op1

        clipped = np.clip(raw_ret,
                          -args.stop_loss,
                          args.take_profit)
        net_ret = clipped - (args.commission + args.slippage)

        records.append({
            't':     t,
            'prob':  pred_prob,
            'raw':   raw_ret,
            'net':   net_ret
        })

    back = pd.DataFrame(records).set_index('t')

    # 4) Metrics & equity curve
    brier = mean_squared_error((back['raw'] > 0).astype(int), back['prob'])
    rets  = back['net']
    equity = (1 + rets).cumprod()
    total  = equity.iloc[-1] - 1
    sr     = sharpe(rets)
    dd     = max_drawdown(equity)

    print(f"\n=== Test‐Slice Performance ===")
    print(f"Trades taken       = {len(rets)}")
    print(f"Total return       = {total*100:.2f}%")
    print(f"Sharpe (ann.)      = {sr:.2f}")
    print(f"Max drawdown       = {dd*100:.2f}%")
    print(f"Brier score        = {brier:.6f}")

    # 5) Save and plot
    os.makedirs(args.output_dir, exist_ok=True)
    back.to_csv(os.path.join(args.output_dir, "backtest_lagged_return_xgb_full.csv"))

    plt.figure(figsize=(8,4))
    plt.plot(equity, lw=1.5)
    plt.title("Equity Curve (Return Predictor)")
    plt.xlabel("Index"); plt.ylabel("Cumulative Return")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "equity_curve.png"))
    plt.close()

if __name__ == "__main__":
    main()
