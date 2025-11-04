import argparse
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ─────────────────────────────────────────────────────────────────────────────
# Features & horizons
# ─────────────────────────────────────────────────────────────────────────────

FEATURES: List[str] = [
    'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions', 'sentiment',
    'price_change', 'high_low_range', 'log_volume', 'macd_line', 'macd_signal', 'macd_histogram',
    'rsi', 'roc', 'atr', 'ema_9', 'ema_21', 'ema_50', 'ema_200', 'adx', 'obv', 'bollinger_upper',
    'bollinger_lower', 'bollinger_percB', 'returns_1', 'returns_3', 'returns_5', 'std_5', 'std_10',
    'lagged_close_1', 'lagged_close_2', 'lagged_close_3', 'lagged_close_5', 'lagged_close_10',
    'candle_body_ratio', 'wick_dominance', 'gap_vs_prev', 'volume_zscore', 'atr_zscore', 'rsi_zscore',
    'macd_cross', 'macd_hist_flip', 'month', 'hour_sin', 'hour_cos', 'day_of_week_sin',
    'day_of_week_cos', 'days_since_high', 'days_since_low',
]

MULTI_HORIZONS: List[int] = [1, 3, 5, 8]


def _target_col(h: int) -> str:
    return f"target_h{h}"


def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add target columns target_h{h} = close.shift(-h)."""
    df = df.copy()
    for h in MULTI_HORIZONS:
        df[_target_col(h)] = df['close'].shift(-h)
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model API (CatBoost Multi-target with MultiRMSE)
# ─────────────────────────────────────────────────────────────────────────────

def fit(X_train: np.ndarray, Y_train: np.ndarray):
    """Fit a single CatBoostRegressor on (X_train, Y_train) with MultiRMSE.
    A small holdout tail is used for early stopping. Returns the trained model.
    Y_train must be shape (n_samples, n_targets) ordered by MULTI_HORIZONS.
    """
    logging.info("FIT catboost_regressor (MultiRMSE) – static params, val tail split")

    # Prevent degenerate splits
    val_size = max(200, int(0.1 * len(X_train)))

    X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
    Y_tr, Y_val = Y_train[:-val_size], Y_train[-val_size:]

    train_pool = Pool(X_tr, Y_tr)
    val_pool = Pool(X_val, Y_val)

    model = CatBoostRegressor(iterations=600, depth=6, learning_rate=0.05,
            l2_leaf_reg=3, bagging_temperature=0.5,
            loss_function="MultiRMSE", eval_metric="MultiRMSE",
            random_seed=42, verbose=False)
    model.fit(
        train_pool,
        eval_set=val_pool,
        verbose=False,
        use_best_model=True,
        early_stopping_rounds=50
    )
    return model


def predict(model, X: np.ndarray) -> np.ndarray:
    """Predict multi-horizon targets. Returns shape (n_samples, n_targets)."""
    logging.info("PREDICT catboost_regressor (multi)")
    preds = model.predict(X)
    # CatBoost returns list of lists; ensure np.ndarray
    return np.asarray(preds, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Angle utility
# ─────────────────────────────────────────────────────────────────────────────

def linear_angle(y_vals: List[float], x_vals: Optional[List[float]] = None) -> float:
    """Compute the angle (degrees) of the best-fit line through (x, y).
    Default x = [1, 3, 5, 8]. Returns NaN if fewer than 2 finite points exist.
    """
    if x_vals is None:
        x_vals = MULTI_HORIZONS
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float('nan')

    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    angle_rad = np.arctan(slope)  # slope is in "price per horizon step"
    return float(np.degrees(angle_rad))


# ─────────────────────────────── CLI (optional) ──────────────────────────────

@dataclass
class WalkForwardConfig:
    n_back: int = 500
    window: int = 500


def _walk_forward(df: pd.DataFrame, cfg: WalkForwardConfig, output_path: str) -> pd.DataFrame:
    logging.info("Starting walk-forward (multi-horizon regression via CatBoost + MultiRMSE)")

    # Ensure targets exist
    df = compute_targets(df)

    # Precompute target column list in the canonical order
    tgt_cols = [_target_col(h) for h in MULTI_HORIZONS]

    recs = []
    start = len(df) - cfg.n_back - 1
    end = len(df) - 2
    start = max(0, start)

    for t in tqdm(range(start, end), desc="Walk-forward (multi-horizon reg)"):
        row_feats = df.loc[[t], FEATURES].values

        # Train on strictly prior rows [t - window, t)
        train_slice = df.iloc[max(0, t - cfg.window): t].dropna(subset=FEATURES + tgt_cols)
        if len(train_slice) < 50:
            # Not enough data: record NaNs for this step
            rec = {"t": t, "angle_pred_deg": float('nan')}
            for h in MULTI_HORIZONS:
                rec[f"pred_h{h}"] = float('nan')
                rec[f"actual_h{h}"] = float('nan')
            recs.append(rec)
            continue

        X_tr = train_slice[FEATURES].values
        Y_tr = train_slice[tgt_cols].values  # shape (n_samples, n_targets)

        model = fit(X_tr, Y_tr)
        yhat_multi = predict(model, row_feats)[0]  # shape (n_targets,)

        preds: Dict[int, float] = {}
        actuals: Dict[int, float] = {}

        for i, h in enumerate(MULTI_HORIZONS):
            preds[h] = float(yhat_multi[i])
            val = df.at[t, _target_col(h)]
            actuals[h] = float(val) if pd.notna(val) else float('nan')

        # Angle over predicted points (requires ≥2 finite points; ideally 4)
        y_list = [preds.get(h, np.nan) for h in MULTI_HORIZONS]
        angle_deg = linear_angle(y_list)

        rec = {"t": t, "angle_pred_deg": angle_deg}
        for h in MULTI_HORIZONS:
            rec[f"pred_h{h}"] = preds[h]
            rec[f"actual_h{h}"] = actuals[h]
        recs.append(rec)

    res = pd.DataFrame(recs).set_index("t")
    res.to_csv(output_path)

    # Simple sanity metrics for h=1 (align each t prediction with close_{t+1})
    mask_h1 = res["pred_h1"].notna() & res["actual_h1"].notna()
    if mask_h1.sum() > 0:
        mae = mean_absolute_error(res.loc[mask_h1, "actual_h1"], res.loc[mask_h1, "pred_h1"])
        rmse = mean_squared_error(res.loc[mask_h1, "actual_h1"], res.loc[mask_h1, "pred_h1"], squared=False)
        print(f"Saved walk-forward results to {output_path}")
        print(f"h=1 MAE = {mae:.4f}\nRMSE = {rmse:.4f}")
    else:
        print(f"Saved walk-forward results to {output_path} (no valid slice for h=1 metrics)")

    # Optional: aggregated MultiRMSE across all horizons where all preds/actuals are present
    pred_cols = [f"pred_h{h}" for h in MULTI_HORIZONS]
    act_cols = [f"actual_h{h}" for h in MULTI_HORIZONS]
    mask_all = res[pred_cols + act_cols].notna().all(axis=1)
    if mask_all.sum() > 0:
        y_true = res.loc[mask_all, act_cols].values
        y_pred = res.loc[mask_all, pred_cols].values
        mse_per_target = np.mean((y_true - y_pred) ** 2, axis=0)
        multi_rmse = float(np.sqrt(np.mean(mse_per_target)))
        rmse_per_target = np.sqrt(mse_per_target)
        per_target_str = ", ".join(
            f"h={h}: {rmse_per_target[i]:.4f}" for i, h in enumerate(MULTI_HORIZONS)
        )
        print(f"MultiRMSE (all horizons) = {multi_rmse:.4f}")
        print(f"Per-horizon RMSE -> {per_target_str}")

    return res


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("input_csv", help="CSV containing raw bars + all features")
    p.add_argument("--output", default="catboost_backtest_reg_multi.csv", help="Where to dump walk-forward results")
    p.add_argument("--n_back", type=int, default=500, help="Bars for final walk-forward slice")
    p.add_argument("--window", type=int, default=500, help="Rolling window for training each step")
    args = p.parse_args()

    df_raw = pd.read_csv(args.input_csv)
    # Defensive: ensure required columns exist
    missing = [f for f in FEATURES if f not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    cfg = WalkForwardConfig(n_back=args.n_back, window=args.window)
    _walk_forward(df_raw, cfg, args.output)