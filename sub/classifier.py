import logging
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

from sub.common import compute_meta_labels, USE_META_LABEL

# ─── Feature Set ────────────────────────────────────────────────────────────
FEATURES = [
    # basic price/vol
    "open", "high", "low", "close", "volume", "vwap", "transactions",
    # sentiment / price derived
    "sentiment", "price_change", "high_low_range", "log_volume",
    # oscillators & trend
    "macd_line", "macd_signal", "macd_histogram", "rsi", "momentum", "roc", "atr",
    # EMAs
    "ema_9", "ema_21", "ema_50", "ema_200",
    # ADX & volume
    "adx", "obv",
    # bands
    "bollinger_upper", "bollinger_lower",
    # lagged closes
    "lagged_close_1", "lagged_close_2", "lagged_close_3", "lagged_close_5", "lagged_close_10",
    # candle anatomy & gaps
    "candle_body_ratio", "wick_dominance", "gap_vs_prev",
    # z‑scores & transforms
    "volume_zscore", "atr_zscore", "rsi_zscore",
    # engineered flags
    "adx_trend", "macd_cross", "macd_hist_flip",
    # calendar / rolling extremes
    "day_of_week", "days_since_high", "days_since_low"
]

BEST_XGB_PARAMS = {
    "n_estimators":     700,
    "learning_rate":    0.05,
    "max_depth":        4,
    "min_child_weight": 5,
    "subsample":        0.80,
    "colsample_bytree": 0.80,
    "gamma":            0.0,
    "reg_alpha":        0.0,
    "reg_lambda":       1.0,
}

# ─── Label Construction ─────────────────────────────────────────────────────

def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if USE_META_LABEL:
        # keep behavior identical to previous pipeline
        return compute_meta_labels(df).rename(columns={"meta_label": "label"})

    df["label"] = (df["close"].shift(-1) > df["close"]).astype(int)
    # drop rows that still contain NaNs in feature set or label
    return df.dropna(subset=FEATURES + ["label"]).reset_index(drop=True)

def fit(X_train: np.ndarray, y_train: np.ndarray):
    """
    Fast deterministic fit:
      • fixed hyper-params (BEST_XGB_PARAMS) – no grid search
      • early-stopping on the last 10 % of the training slice
      • isotonic calibration for well-behaved probabilities
    """
    logging.info("Running FIT on all_features_xgboost – static params + early-stop")

    # 1) split last 10 % for early-stopping
    val_size = max(200, int(0.1 * len(X_train)))
    X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
    y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

    core = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        **BEST_XGB_PARAMS,
    )

    core.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False,
    )

    # 2) isotonic calibration with a *time-series aware* CV
    calib = CalibratedClassifierCV(
        core,
        method="isotonic",
        cv=TimeSeriesSplit(n_splits=3),
    ).fit(X_train, y_train)

    # expose raw XGB params so downstream code that clones works unchanged
    calib.get_xgb_params = core.get_xgb_params
    return calib


def predict(model, X: np.ndarray) -> np.ndarray:
    logging.info("Running PREDICT on all_features_xgboost (calibrated)")
    return model.predict_proba(X)[:, 1]

# ─── Optional Stand‑Alone Test Harness ──────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    from sklearn.metrics import accuracy_score, roc_auc_score

    p = argparse.ArgumentParser()
    p.add_argument("input_csv", help="CSV containing raw bars + all features")
    p.add_argument("--output", default="xgb_backtest.csv", help="Where to dump simple walk‑forward results")
    p.add_argument("--n_back", type=int, default=500, help="Bars for final walk‑forward slice")
    p.add_argument("--window", type=int, default=500, help="Rolling window size for training each step")
    args = p.parse_args()

    # 1) Load & label
    df = pd.read_csv(args.input_csv)
    df = compute_labels(df)

    # 2) Hyper‑param tuning on all but last n_back
    df_tune = df.iloc[:-args.n_back]
    X_tune, y_tune = df_tune[FEATURES].values, df_tune["label"].values
    model = fit(X_tune, y_tune)

    # 3) Simple i+1 walk‑forward probability evaluation on last n_back
    recs = []
    start = len(df) - args.n_back - 1
    end   = len(df) - 2  # need *next* bar for label

    for t in tqdm(range(start, end), desc="Walk‑forward"):
        train_slice = df.iloc[max(0, t - args.window): t]
        X_tr, y_tr  = train_slice[FEATURES].values, train_slice["label"].values
        clf = XGBClassifier(**model.get_xgb_params(), objective="binary:logistic", eval_metric="logloss", n_jobs=-1)
        clf.fit(X_tr, y_tr, verbose=False)

        p_up = predict(clf, df.loc[[t], FEATURES].values)[0]
        true = df.at[t, "label"]
        recs.append({"t": t, "prob": p_up, "label": true})

    res = pd.DataFrame(recs).set_index("t")
    res.to_csv(args.output)

    acc = accuracy_score(res["label"], (res["prob"] > 0.5).astype(int))
    auc = roc_auc_score(res["label"], res["prob"])
    print(f"Saved walk‑forward results to {args.output}")
    print(f"Accuracy  = {acc:.3f}\nROC‑AUC   = {auc:.3f}")
