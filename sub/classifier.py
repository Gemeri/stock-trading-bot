import logging
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score

# ─────────────────────────────────────────────────────────────────────────────
# Multi-horizon classifier: predict P(close_{t+h} > close_t) for h ∈ {1,3,5,8}
# ─────────────────────────────────────────────────────────────────────────────

FEATURES = [
    'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions', 'sentiment',
    'price_change', 'high_low_range', 'log_volume', 'macd_line', 'macd_signal', 'macd_histogram',
    'rsi', 'roc', 'atr', 'ema_9', 'ema_21', 'ema_50', 'ema_200', 'adx', 'obv', 'bollinger_upper', 
    'bollinger_lower', 'bollinger_percB', 'returns_1', 'returns_3', 'returns_5', 'std_5', 'std_10',
    'lagged_close_1', 'lagged_close_2', 'lagged_close_3', 'lagged_close_5', 'lagged_close_10',
    'candle_body_ratio', 'wick_dominance', 'gap_vs_prev', 'volume_zscore', 'atr_zscore', 'rsi_zscore',
    'macd_cross', 'macd_hist_flip', 'month', 'hour_sin', 'hour_cos', 'day_of_week_sin', 
    'day_of_week_cos', 'days_since_high', 'days_since_low',
]

# Signals to the main pipeline that this module emits *four* predictions
MULTI_HORIZONS = [1, 3, 5, 8]

def _label_col(h: int) -> str:
    return f"label_h{h}"

def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for h in MULTI_HORIZONS:
        df[_label_col(h)] = (df["close"].shift(-h) > df["close"]).astype(int)
    # Backward-compat single-horizon name (used elsewhere defensively)
    df["label"] = df[_label_col(1)]
    return df.reset_index(drop=True)

def fit(X_train: np.ndarray, y_train: np.ndarray, *, horizon: int | None = None):
    htxt = f"h={horizon}" if horizon is not None else "h=?"
    logging.info(f"FIT all_features_xgboost ({htxt}) – static params + isotonic calibration")

    # Simple fixed split for early stopping-like behavior
    val_size = max(200, int(0.1 * len(X_train)))
    X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
    y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

    core = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    core.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    calib = CalibratedClassifierCV(
        core,
        method="isotonic",
        cv=TimeSeriesSplit(n_splits=3),
    ).fit(X_train, y_train)

    # provide a helper so legacy CLI can recreate core params if needed
    calib.get_xgb_params = core.get_xgb_params
    return calib

def predict(model, X: np.ndarray) -> np.ndarray:
    logging.info("PREDICT all_features_xgboost (calibrated)")
    return model.predict_proba(X)[:, 1]

def linear_angle(y_vals, x_vals=None):
    if x_vals is None:
        x_vals = MULTI_HORIZONS

    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float('nan')

    slope, _ = np.polyfit(x[mask], y[mask], 1)
    return float(np.degrees(np.arctan(slope)))



# ─────────────────────────────── CLI (optional) ──────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input_csv", help="CSV containing raw bars + all features")
    p.add_argument("--output", default="xgb_backtest_multi.csv", help="Where to dump walk-forward results")
    p.add_argument("--n_back", type=int, default=500, help="Bars for final walk-forward slice")
    p.add_argument("--window", type=int, default=500, help="Rolling window for training each step")
    args = p.parse_args()

    df_raw = pd.read_csv(args.input_csv)
    df = compute_labels(df_raw)

    recs = []
    start = len(df) - args.n_back - 1
    end   = len(df) - 2

    for t in tqdm(range(start, end), desc="Walk-forward (multi-horizon)"):
        row_feats = df.loc[[t], FEATURES].values

        probs = {}
        labels = {}
        for h in MULTI_HORIZONS:
            lbl_col = _label_col(h)
            train_slice = df.iloc[max(0, t - args.window): t].dropna(subset=FEATURES + [lbl_col])
            if len(train_slice) < 50:
                probs[h] = np.nan
                labels[h] = np.nan
                continue
            X_tr = train_slice[FEATURES].values
            y_tr = train_slice[lbl_col].values
            clf  = fit(X_tr, y_tr, horizon=h)
            probs[h]  = float(predict(clf, row_feats)[0])
            labels[h] = int(df.at[t, lbl_col])
            
        angle_prob_deg = linear_angle([probs.get(h, np.nan) for h in MULTI_HORIZONS])
        rec = {"t": t, "angle_prob_deg": angle_prob_deg}
        rec = {"t": t}
        for h in MULTI_HORIZONS:
            rec[f"prob_h{h}"]  = probs[h]
            rec[f"label_h{h}"] = labels[h]
        recs.append(rec)

    res = pd.DataFrame(recs).set_index("t")
    res.to_csv(args.output)

    # Print simple metrics for h=1 as a sanity check
    mask = res["prob_h1"].notna()
    if mask.sum() > 0:
        acc = accuracy_score(res.loc[mask, "label_h1"], (res.loc[mask, "prob_h1"] > 0.5).astype(int))
        auc = roc_auc_score(res.loc[mask, "label_h1"], res.loc[mask, "prob_h1"])
        print(f"Saved walk-forward results to {args.output}")
        print(f"h=1 Accuracy = {acc:.3f}\nAUC = {auc:.3f}")
    else:
        print(f"Saved walk-forward results to {args.output} (no valid slice for metrics)")
