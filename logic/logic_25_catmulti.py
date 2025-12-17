import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm  
import numpy as np
import pandas as pd
import config
from catboost import CatBoostClassifier
from sklearn.metrics import roc_curve, auc

# =====================================================
# CONFIGURATION
# =====================================================

logger = logging.getLogger(__name__)

# Timeframe and data path configuration
BAR_TIMEFRAME = config.BAR_TIMEFRAME
TIMEFRAME_MAP = {
    "4Hour": "H4", "2Hour": "H2", "1Hour": "H1",
    "30Min": "M30", "15Min": "M15"
}
CONVERTED_TIMEFRAME = TIMEFRAME_MAP.get(BAR_TIMEFRAME, BAR_TIMEFRAME)

DATA_DIR = "data"
CATBOOST_DIR = "catboost-multi"
os.makedirs(CATBOOST_DIR, exist_ok=True)

# Base features (strict gate)
BASE_FEATURES = [
    'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions', 'sentiment',
    'greed_index', 'news_count', 'news_volume_z', 'price_change', 'high_low_range',
    'macd_line', 'macd_signal', 'macd_histogram', 'rsi', 'roc', 'atr', 'ema_9',
    'ema_21', 'ema_50', 'ema_200', 'adx', 'obv', 'bollinger_percB', 'returns_1',
    'returns_3', 'returns_5', 'std_5', 'std_10', 'lagged_close_1', 'lagged_close_2',
    'lagged_close_3', 'lagged_close_5', 'lagged_close_10', 'candle_body_ratio',
    'wick_dominance', 'gap_vs_prev', 'volume_zscore', 'atr_zscore', 'rsi_zscore',
    'month', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
    'days_since_high', 'days_since_low', "d_sentiment"
]

# CatBoost and ensemble hyperparameters
TRAINING_WINDOW = 4000
RETRAIN_EVERY = 1
PER_CLASS_AUC_THRESHOLD = 0.0
GINI_THRESHOLD = 0.09
ENTROPY_THRESHOLD = 0.90
VAL_FRACTION = 0.2
LOOKAHEAD_HORIZONS = [2, 4, 6, 8, 10, 12]
QUANTILE_WINDOW = 32

# Trading / risk parameters
SPREAD_RATE = 0.0005
LOSS_CLOSE_RATE = 0.05
GAIN_CLOSE_RATE = 0.10
COOL_OFF_AFTER_CLOSE_TRADE = 3
GRID_SEARCH_ENABLED = True
GRID_SEARCH_WINDOW_CANDLES = 100

# =====================================================
# PATH HELPERS
# =====================================================

DATA_DIR = "data"
def timeframe_subdir(tf_code: str) -> str:
    """Return the directory path for a given timeframe code, creating it if needed."""

    path = os.path.join(DATA_DIR, tf_code)
    os.makedirs(path, exist_ok=True)
    return path

def get_csv_filename(ticker):
    return os.path.join(timeframe_subdir(CONVERTED_TIMEFRAME), f"{ticker}_{CONVERTED_TIMEFRAME}.csv")


def _get_model_meta_path(ticker: str) -> str:
    return os.path.join(CATBOOST_DIR, f"{ticker}_meta.json")


def _get_model_path(ticker: str, offset: int) -> str:
    return os.path.join(CATBOOST_DIR, f"{ticker}_target_{offset}.cbm")


def _get_risk_state_path(ticker: str) -> str:
    return os.path.join(CATBOOST_DIR, f"{ticker}_risk_state.json")


# =====================================================
# MATH / CONFIDENCE HELPERS
# =====================================================

def calculate_gini(probs: np.ndarray) -> float:
    p = np.asarray(probs, dtype=float)
    p = p / p.sum()  # Ensure sum = 1
    K = len(p)

    # Gini index
    G = 1.0 - np.sum(p ** 2)

    # Normalize by maximum possible Gini for K classes
    G_norm = G / (1.0 - 1.0 / K)

    # Confidence is inverse of normalized Gini
    confidence_gini = 1.0 - G_norm
    return float(confidence_gini)


def calibrate_scores(probs: np.ndarray) -> Tuple[float, float]:
    p = np.asarray(probs, dtype=float)
    p = p / p.sum()  # Ensure sum = 1
    K = len(p)

    # Gini-based confidence
    confidence_gini = calculate_gini(p)

    # Normalized entropy
    H = -np.sum(p * np.log(p + 1e-12))
    H_norm = H / np.log(K)

    return confidence_gini, float(H_norm)


# =====================================================
# DATA LOADING / PREP
# =====================================================

def _load_raw_data(ticker: str) -> pd.DataFrame:
    csv_path = get_csv_filename(ticker)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data CSV not found for {ticker}: {csv_path}")

    df = pd.read_csv(csv_path)

    # Normalise timestamp column name
    if "time" in df.columns:
        time_col = "time"
    elif "timestamp" in df.columns:
        time_col = "timestamp"
    else:
        raise ValueError(f"No 'time' or 'timestamp' column found in {csv_path}")

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)
    df = df.rename(columns={time_col: "time"})

    # Drop predicted_close if present (do NOT use it for training)
    if "predicted_close" in df.columns:
        df = df.drop(columns=["predicted_close"])

    return df


def _ensure_multi_horizon_targets(df: pd.DataFrame) -> pd.DataFrame:
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain 'close' column to build targets.")

    close = df["close"].astype(float)

    for t in LOOKAHEAD_HORIZONS:
        col = f"target_{t}"
        if col in df.columns:
            continue  # already present, don't overwrite

        fwd_ret = close.shift(-t) / close - 1.0

        # Rolling quantiles over the *past* QUANTILE_WINDOW values of fwd_ret
        rolling_low = (
            fwd_ret.rolling(QUANTILE_WINDOW, min_periods=QUANTILE_WINDOW)
                  .quantile(0.33)
        )
        rolling_high = (
            fwd_ret.rolling(QUANTILE_WINDOW, min_periods=QUANTILE_WINDOW)
                  .quantile(0.67)
        )

        target = np.zeros(len(df), dtype=int)

        valid = (~fwd_ret.isna()) & (~rolling_low.isna()) & (~rolling_high.isna())
        target[(fwd_ret > rolling_high) & valid] = 1
        target[(fwd_ret < rolling_low) & valid] = -1

        df[col] = target

    return df


def _get_feature_and_target_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    # First, make sure target_* columns are present (created in-memory if needed)
    df = _ensure_multi_horizon_targets(df)

    # Features: strict intersection with BASE_FEATURES
    feature_cols = [c for c in BASE_FEATURES if c in df.columns]

    # Targets: exactly the configured horizon range
    target_cols = []
    for t in LOOKAHEAD_HORIZONS:
        col = f"target_{t}"
        if col in df.columns:
            target_cols.append(col)

    return feature_cols, target_cols



# =====================================================
# MODEL TRAINING AND CACHING
# =====================================================

def _train_models_for_timestamp(
    df: pd.DataFrame,
    current_time: datetime,
    feature_cols: List[str],
    target_cols: List[str],
    is_backtest: bool = False,
) -> Dict[str, Any]:

    df_cut = df[df["time"] <= current_time].copy()
    n = len(df_cut)

    # Hard minimum: need > 600 candles total
    if n <= 600:
        logger.info(
            f"Not enough data to train (need > 600 candles, have {n})."
        )
        return {"models": [], "feature_cols": feature_cols, "target_offsets": []}

    # Decide training window slice
    if n <= TRAINING_WINDOW:
        # Use all data up to current_time (no trim to TRAINING_WINDOW)
        window_df = df_cut.copy()
        logger.info(
            f"Using full history up to {current_time} for training "
            f"(len={n} <= TRAINING_WINDOW={TRAINING_WINDOW})."
        )
    else:
        # Use the last TRAINING_WINDOW rows
        window_df = df_cut.iloc[-TRAINING_WINDOW:].copy()
        logger.info(
            f"Using rolling window of size TRAINING_WINDOW={TRAINING_WINDOW} "
            f"out of total {n} candles up to {current_time}."
        )

    X_window_full = window_df[feature_cols]

    # Label mapping
    label_mapping = {-1: 0, 0: 1, 1: 2}
    reverse_mapping = {v: k for k, v in label_mapping.items()}

    models_cache: List[Dict[str, Any]] = []
    target_offsets: List[int] = []

    for target_col in target_cols:
        offset = int(target_col.split("_", 1)[1])

        # Raw labels for this horizon
        y_window_raw = window_df[target_col].map(label_mapping)

        # --- Horizon-specific cut (backtest only) ---
        # For horizon t, we drop the last t candles from both X and y.
        # This ensures that for all remaining rows, their forward label
        # uses information that would be available at current_time in a
        # realistic backtest.
        if is_backtest and offset > 0:
            effective_len = len(window_df) - offset
            if effective_len <= 0:
                logger.info(
                    f"Skipping horizon {offset}: "
                    f"not enough data after horizon cut (len={len(window_df)}, offset={offset})."
                )
                continue
            X_window = X_window_full.iloc[:effective_len]
            y_window = y_window_raw.iloc[:effective_len]
        else:
            X_window = X_window_full
            y_window = y_window_raw

        if len(X_window) <= 1:
            logger.info(
                f"Skipping horizon {offset}: insufficient samples after cut (len={len(X_window)})."
            )
            continue

        split_index = int(len(X_window) * (1.0 - VAL_FRACTION))
        if split_index <= 1 or split_index >= len(X_window):
            logger.warning(
                f"Validation split degenerate for horizon {offset}; adjusting split index."
            )
            split_index = max(1, len(X_window) - 1)

        X_train = X_window.iloc[:split_index]
        X_val = X_window.iloc[split_index:]
        y_train = y_window.iloc[:split_index]
        y_val = y_window.iloc[split_index:]

        logger.info(
            f"Training CatBoost for {target_col} (offset={offset}), "
            f"class distribution: {y_train.value_counts().to_dict()}"
        )

        model = CatBoostClassifier(
            iterations=200,
            learning_rate=0.05,
            depth=5,
            l2_leaf_reg=5,
            loss_function="MultiClass",
            eval_metric="MultiClass",
            random_seed=42,
            early_stopping_rounds=20,
            verbose=False
        )
        ages = np.arange(len(X_train)-1, -1, -1)     # oldest gets largest age, newest gets 0
        weights = np.exp(-np.log(2) * ages / HALF_LIFE)
        train_pool = Pool(X_train, y_train, weight=weights)
        model.fit(train_pool)

        # Per-class AUC on validation
        y_proba_val = model.predict_proba(X_val)
        per_class_auc: Dict[int, float] = {}
        for class_idx in range(3):
            y_true_bin = (y_val == class_idx).astype(int)
            if y_true_bin.sum() == 0:
                per_class_auc[class_idx] = np.nan
                continue
            fpr, tpr, _ = roc_curve(y_true_bin, y_proba_val[:, class_idx])
            per_class_auc[class_idx] = auc(fpr, tpr)

        logger.info(f"Validation per-class AUC for {target_col}: {per_class_auc}")

        if all(
            v >= PER_CLASS_AUC_THRESHOLD
            for v in per_class_auc.values()
            if not np.isnan(v)
        ):
            models_cache.append(
                {"model": model, "offset": offset, "per_class_auc": per_class_auc}
            )
            target_offsets.append(offset)
            logger.info(
                f"Model for {target_col} retained (all AUC ≥ {PER_CLASS_AUC_THRESHOLD})."
            )
        else:
            logger.info(
                f"Model for {target_col} discarded (AUC below threshold)."
            )

    logger.info(f"Total models retained: {len(models_cache)}")

    return {
        "models": models_cache,
        "feature_cols": feature_cols,
        "target_offsets": target_offsets,
        "label_mapping": {-1: 0, 0: 1, 1: 2},
        "reverse_mapping": {0: -1, 1: 0, 2: 1},
        "trained_until": current_time.isoformat()
    }



def _save_models_to_disk(ticker: str, train_result: Dict[str, Any]) -> None:
    models_cache = train_result.get("models", [])
    feature_cols = train_result.get("feature_cols", [])
    target_offsets = train_result.get("target_offsets", [])
    label_mapping = train_result.get("label_mapping", {-1: 0, 0: 1, 1: 2})
    reverse_mapping = train_result.get("reverse_mapping", {0: -1, 1: 0, 2: 1})
    trained_until = train_result.get("trained_until", None)

    for m in models_cache:
        offset = m["offset"]
        model = m["model"]
        model_path = _get_model_path(ticker, offset)
        model.save_model(model_path)

    meta = {
        "ticker": ticker,
        "feature_cols": feature_cols,
        "target_offsets": target_offsets,
        "label_mapping": label_mapping,
        "reverse_mapping": reverse_mapping,
        "trained_until": trained_until,
        "runs_since_retrain": 0,
        "retrain_every": RETRAIN_EVERY
    }

    with open(_get_model_meta_path(ticker), "w") as f:
        json.dump(meta, f)


def _load_models_from_disk(ticker: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    meta_path = _get_model_meta_path(ticker)
    if not os.path.exists(meta_path):
        return [], {}

    with open(meta_path, "r") as f:
        meta = json.load(f)

    feature_cols = meta.get("feature_cols", [])
    target_offsets = meta.get("target_offsets", [])
    label_mapping = {int(k): v for k, v in meta.get("label_mapping", {}).items()}
    reverse_mapping = {int(k): v for k, v in meta.get("reverse_mapping", {}).items()}

    meta["label_mapping"] = label_mapping
    meta["reverse_mapping"] = reverse_mapping
    models_cache: List[Dict[str, Any]] = []

    for offset in target_offsets:
        model_path = _get_model_path(ticker, offset)
        if not os.path.exists(model_path):
            continue
        model = CatBoostClassifier()
        model.load_model(model_path)
        models_cache.append({"model": model, "offset": offset})

    return models_cache, meta


def _update_meta_retrain_counter(ticker: str, meta: Dict[str, Any], reset: bool) -> None:
    if not meta:
        meta = {
            "ticker": ticker,
            "feature_cols": [],
            "target_offsets": [],
            "label_mapping": {-1: 0, 0: 1, 1: 2},
            "reverse_mapping": {0: -1, 1: 0, 2: 1},
            "trained_until": None,
            "runs_since_retrain": 0,
            "retrain_every": RETRAIN_EVERY
        }

    if reset:
        meta["runs_since_retrain"] = 0
    else:
        meta["runs_since_retrain"] = int(meta.get("runs_since_retrain", 0)) + 1

    if "retrain_every" not in meta:
        meta["retrain_every"] = RETRAIN_EVERY

    with open(_get_model_meta_path(ticker), "w") as f:
        json.dump(meta, f)


# =====================================================
# RISK STATE PERSISTENCE
# =====================================================

def _load_risk_state(ticker: str) -> Dict[str, Any]:
    path = _get_risk_state_path(ticker)
    if not os.path.exists(path):
        return {
            "last_entry_price": None,
            "last_entry_timestamp": None,
            "last_direction": None,  # "long" or "short"
            "stop_loss_price": None,
            "take_profit_price": None,
            "loss_close_rate": LOSS_CLOSE_RATE,
            "gain_close_rate": GAIN_CLOSE_RATE,
            "cooloff_after_close_trade": COOL_OFF_AFTER_CLOSE_TRADE,  # NEW: per-ticker cooloff
            "last_close_timestamp": None,
            "last_action": None  # "close_long" or "close_short"
        }
    with open(path, "r") as f:
        state = json.load(f)

    # Ensure new keys always exist
    if "loss_close_rate" not in state:
        state["loss_close_rate"] = LOSS_CLOSE_RATE
    if "gain_close_rate" not in state:
        state["gain_close_rate"] = GAIN_CLOSE_RATE
    if "cooloff_after_close_trade" not in state:
        state["cooloff_after_close_trade"] = COOL_OFF_AFTER_CLOSE_TRADE

    return state



def _save_risk_state(ticker: str, state: Dict[str, Any]) -> None:
    """
    Persist risk state JSON to disk.
    """
    path = _get_risk_state_path(ticker)
    with open(path, "w") as f:
        json.dump(state, f)


# =====================================================
# ENSEMBLE PREDICTION
# =====================================================
def _grid_search_execution_params(
    df_hist: pd.DataFrame,
    ticker: str,
    window_candles: int = GRID_SEARCH_WINDOW_CANDLES,
) -> Tuple[float, float, int]:
    # Ensure we have enough history
    df_hist = df_hist.sort_values("time").copy()
    n = len(df_hist)
    min_needed = max(window_candles, 50)
    if n < min_needed:
        logger.info(
            f"[{ticker}] Grid search skipped: only {n} candles available (<{min_needed})."
        )
        return LOSS_CLOSE_RATE, GAIN_CLOSE_RATE, COOL_OFF_AFTER_CLOSE_TRADE

    # Take only most recent window_candles
    df_window = df_hist.iloc[-window_candles:].copy()

    # Load models and meta
    models_cache, meta = _load_models_from_disk(ticker)
    if not models_cache or not meta:
        logger.info(f"[{ticker}] Grid search skipped: no models/meta available.")
        return LOSS_CLOSE_RATE, GAIN_CLOSE_RATE, COOL_OFF_AFTER_CLOSE_TRADE

    feature_cols = meta.get("feature_cols", [])
    if not feature_cols:
        logger.info(f"[{ticker}] Grid search skipped: no feature_cols in meta.")
        return LOSS_CLOSE_RATE, GAIN_CLOSE_RATE, COOL_OFF_AFTER_CLOSE_TRADE

    reverse_mapping = meta.get("reverse_mapping", {0: -1, 1: 0, 2: 1})

    # -------- Build signals using current ensemble (NO retraining here) --------
    signals: List[Tuple[datetime, float, int]] = []

    for _, row in df_window.iterrows():
        t = row["time"]
        price = float(row["close"])

        X_row = row[feature_cols].to_frame().T

        votes: Dict[int, List[int]] = {1: [], 0: [], -1: []}

        for m in models_cache:
            model = m["model"]
            probs = model.predict_proba(X_row).flatten()
            g, e = calibrate_scores(probs)

            if g >= GINI_THRESHOLD or e <= ENTROPY_THRESHOLD:
                pred_class = int(model.predict(X_row)[0])
                pred_label = reverse_mapping.get(pred_class, 0)
                votes[pred_label].append(1)

        total_votes = sum(len(v) for v in votes.values())
        if total_votes < 1:
            pred_target = 0
        else:
            pred_target = max(votes.items(), key=lambda x: len(x[1]))[0]

        signals.append((t, price, pred_target))

    # If every signal is flat, nothing to optimize
    if not any(pred != 0 for (_, _, pred) in signals):
        logger.info(f"[{ticker}] Grid search skipped: all signals in window are flat.")
        return LOSS_CLOSE_RATE, GAIN_CLOSE_RATE, COOL_OFF_AFTER_CLOSE_TRADE

    # -------- Parameter grid --------
    sl_values = [x / 100.0 for x in range(1, 6)]   # 0.01 → 0.05
    tp_values = [x / 100.0 for x in range(1, 16)]  # 0.01 → 0.15
    cool_values = list(range(1, 11))               # 1 → 10 bars

    best_pnl = -np.inf
    best_tuple = (LOSS_CLOSE_RATE, GAIN_CLOSE_RATE, COOL_OFF_AFTER_CLOSE_TRADE)

    param_grid = [
        (sl, tp, cool)
        for sl in sl_values
        for tp in tp_values
        for cool in cool_values
    ]

    logger.info(
        f"[{ticker}] Starting grid search over "
        f"{len(sl_values)}x{len(tp_values)}x{len(cool_values)}={len(param_grid)} combos "
        f"on {len(signals)} candles."
    )

    for sl, tp, cool in tqdm(param_grid, desc=f"{ticker} grid search", leave=False):
        pnl = 0.0
        q = 0.0                 # position (+1 long, -1 short, 0 flat)
        entry_price: Optional[float] = None
        last_close_ts: Optional[datetime] = None
        last_action: Optional[str] = None  # "close_long" or "close_short"

        for t, price, pred in signals:
            # 1) Check existing position for stop-loss / take-profit exit
            if q != 0 and entry_price is not None:
                pl = q * (price - entry_price)
                trade_value = abs(q * entry_price)
                close_trade = False
                if trade_value > 0:
                    pct_pl = pl / trade_value
                    close_trade = (
                        (pl < 0 and abs(pct_pl) > sl) or
                        (pl > 0 and abs(pct_pl) > tp)
                    )
                if close_trade:
                    pnl += pl
                    last_close_ts = t
                    last_action = "close_long" if q > 0 else "close_short"
                    q = 0.0
                    entry_price = None

            # 2) Cool-off logic
            cooloff_buy = False
            cooloff_sell = False
            if last_close_ts is not None:
                gap_bars = _gap_in_4h_bars(last_close_ts, t)
                if gap_bars < cool:
                    if last_action == "close_long":
                        cooloff_buy = True
                    elif last_action == "close_short":
                        cooloff_sell = True

            # 3) Directional logic (mirrors core engine qualitatively, 1-share size)
            if pred > 0 and not cooloff_buy:
                if q < 0:
                    # Close short, realize PL
                    if entry_price is not None:
                        pnl += q * (price - entry_price)
                    last_close_ts = t
                    last_action = "close_short"
                    q = 0.0
                    entry_price = None
                if q == 0:
                    q = 1.0
                    entry_price = price

            elif pred < 0 and not cooloff_sell:
                if q > 0:
                    # Close long, realize PL
                    if entry_price is not None:
                        pnl += q * (price - entry_price)
                    last_close_ts = t
                    last_action = "close_long"
                    q = 0.0
                    entry_price = None
                if q == 0:
                    q = -1.0
                    entry_price = price
            # pred == 0 → hold / flat

        # 4) Mark-to-market any open position at the end of the window
        if q != 0 and entry_price is not None and len(signals) > 0:
            last_price = signals[-1][1]
            pnl += q * (last_price - entry_price)

        if pnl > best_pnl:
            best_pnl = pnl
            best_tuple = (sl, tp, cool)

    logger.info(
        f"[{ticker}] Grid search best params: "
        f"stop_loss={best_tuple[0]:.3f}, take_profit={best_tuple[1]:.3f}, "
        f"cooloff_bars={best_tuple[2]}, pnl={best_pnl:.4f}"
    )

    return best_tuple



def _get_pred_target_for_timestamp(
    df: pd.DataFrame,
    current_time: datetime,
    ticker: str,
    is_backtest: bool = False,
) -> Tuple[int, int, int, int, Dict[str, Any]]:
    if df.empty:
        return 0, 0, 0, 0, {
            "total_models": 0,
            "total_strong": 0,
            "vote_counts": {"up": 0, "down": 0, "flat": 0},
            "avg_gini": {"up": None, "down": None, "flat": None},
            "avg_entropy": {"up": None, "down": None, "flat": None},
        }

    # Filter data up to current_time (inclusive)
    df_cut = df[df["time"] <= current_time].copy()
    if df_cut.empty:
        return 0, 0, 0, 0, {
            "total_models": 0,
            "total_strong": 0,
            "vote_counts": {"up": 0, "down": 0, "flat": 0},
            "avg_gini": {"up": None, "down": None, "flat": None},
            "avg_entropy": {"up": None, "down": None, "flat": None},
        }

    # Ensure features + targets exist
    feature_cols, target_cols = _get_feature_and_target_columns(df_cut)
    if not target_cols:
        logger.warning(f"No target_* columns could be generated for {ticker}; returning neutral signal.")
        return 0, 0, 0, 0, {
            "total_models": 0,
            "total_strong": 0,
            "vote_counts": {"up": 0, "down": 0, "flat": 0},
            "avg_gini": {"up": None, "down": None, "flat": None},
            "avg_entropy": {"up": None, "down": None, "flat": None},
        }

    # Load cached models and meta
    models_cache, meta = _load_models_from_disk(ticker)

    # Parse trained_until
    trained_until_str = meta.get("trained_until") if meta else None
    trained_until: Optional[datetime] = None
    if trained_until_str:
        try:
            trained_until = datetime.fromisoformat(trained_until_str)
        except Exception:
            trained_until = None

    runs_since_retrain = int(meta.get("runs_since_retrain", 0)) if meta else 0
    retrain_every = int(meta.get("retrain_every", RETRAIN_EVERY)) if meta else RETRAIN_EVERY

    need_retrain = False
    if not models_cache:
        need_retrain = True
        logger.info(f"No cached models found for {ticker}, training from scratch.")
    elif trained_until is None:
        need_retrain = True
        logger.info(f"No trained_until in meta for {ticker}, training.")
    elif current_time < trained_until:
        # Backtest earlier than last training -> retrain to avoid look-ahead
        need_retrain = True
        logger.info(
            f"Current timestamp {current_time} earlier than trained_until={trained_until}, retraining."
        )
    elif runs_since_retrain >= retrain_every:
        need_retrain = True
        logger.info(
            f"runs_since_retrain={runs_since_retrain} ≥ retrain_every={retrain_every}, retraining."
        )

    if need_retrain:
        # 1) Train models
        train_result = _train_models_for_timestamp(
            df=df_cut,
            current_time=current_time,
            feature_cols=feature_cols,
            target_cols=target_cols,
            is_backtest=is_backtest,
        )
        _save_models_to_disk(ticker, train_result)

        # 2) Optional grid search for execution hyperparameters
        if GRID_SEARCH_ENABLED:
            best_sl, best_tp, best_cool = _grid_search_execution_params(
                df_hist=df_cut,
                ticker=ticker,
                window_candles=GRID_SEARCH_WINDOW_CANDLES,
            )
            risk_state = _load_risk_state(ticker)
            risk_state["loss_close_rate"] = float(best_sl)
            risk_state["gain_close_rate"] = float(best_tp)
            risk_state["cooloff_after_close_trade"] = int(best_cool)
            _save_risk_state(ticker, risk_state)

        # 3) Reload models & meta after saving
        models_cache, meta = _load_models_from_disk(ticker)

    # At this point, we (should) have models on disk; if not, neutral
    if not models_cache:
        logger.info(f"No viable models available for {ticker} at {current_time}; neutral signal.")
        _update_meta_retrain_counter(ticker, meta, reset=need_retrain)
        return 0, 0, 0, 0, {
            "total_models": 0,
            "total_strong": 0,
            "vote_counts": {"up": 0, "down": 0, "flat": 0},
            "avg_gini": {"up": None, "down": None, "flat": None},
            "avg_entropy": {"up": None, "down": None, "flat": None},
        }

    # Use last row (current_time or latest <= current_time)
    df_cut = df_cut.sort_values("time")
    current_row = df_cut.iloc[-1]
    X_current = current_row[feature_cols].to_frame().T

    # Reverse mapping from meta (ensure proper types)
    reverse_mapping = meta.get("reverse_mapping", {0: -1, 1: 0, 2: 1})

    votes: Dict[int, List[int]] = {1: [], 0: [], -1: []}
    gini_list: Dict[int, List[float]] = {1: [], 0: [], -1: []}
    entropy_list: Dict[int, List[float]] = {1: [], 0: [], -1: []}

    for m in models_cache:
        model = m["model"]
        offset = m.get("offset")
        probs = model.predict_proba(X_current).flatten()
        g, e = calibrate_scores(probs)

        if g >= GINI_THRESHOLD or e <= ENTROPY_THRESHOLD:
            pred_class = int(model.predict(X_current)[0])
            pred_label = reverse_mapping.get(pred_class, 0)
            logger.debug(
                f"STRONG pred offset={offset} g={g:.3f} e={e:.3f} -> {probs} -> {pred_label}"
            )
            votes[pred_label].append(1)
            gini_list[pred_label].append(g)
            entropy_list[pred_label].append(e)
        else:
            logger.debug(
                f"WEAK pred offset={offset} g={g:.3f} e={e:.3f} -> {probs}"
            )

    total_votes = sum(len(v) for v in votes.values())
    if total_votes < 1:
        pred_target = 0
    else:
        pred_target = max(votes.items(), key=lambda x: len(x[1]))[0]

    models_up = len(votes[1])
    models_down = len(votes[-1])
    models_stable = len(votes[0])

    total_models = len(models_cache)
    total_strong = total_votes

    def _safe_mean(values: List[float]) -> Optional[float]:
        return float(np.mean(values)) if values else None

    avg_gini_raw = {
        1: _safe_mean(gini_list[1]),
        0: _safe_mean(gini_list[0]),
        -1: _safe_mean(gini_list[-1]),
    }
    avg_entropy_raw = {
        1: _safe_mean(entropy_list[1]),
        0: _safe_mean(entropy_list[0]),
        -1: _safe_mean(entropy_list[-1]),
    }

    model_stats: Dict[str, Any] = {
        "total_models": total_models,
        "total_strong": total_strong,
        "vote_counts": {
            "up": models_up,
            "down": models_down,
            "flat": models_stable,
        },
        "avg_gini": {
            "up": avg_gini_raw[1],
            "down": avg_gini_raw[-1],
            "flat": avg_gini_raw[0],
        },
        "avg_entropy": {
            "up": avg_entropy_raw[1],
            "down": avg_entropy_raw[-1],
            "flat": avg_entropy_raw[0],
        },
    }

    _update_meta_retrain_counter(ticker, meta, reset=need_retrain)

    return pred_target, models_up, models_down, models_stable, model_stats

def _compute_action_confidence(
    action: str,
    pred_target: int,
    close_trade: bool,
    pct_pl: float,
    stop_loss_rate: float,
    take_profit_rate: float,
    model_stats: Dict[str, Any],
) -> float:
    """
    Heuristic confidence score in [0, 1], combining:
      - Ensemble agreement and coverage
      - Gini / entropy (probability sharpness)
      - Risk / stop-loss dynamics for exit actions
    """

    total_models = int(model_stats.get("total_models", 0) or 0)
    total_strong = int(model_stats.get("total_strong", 0) or 0)
    vote_counts = model_stats.get("vote_counts", {}) or {}
    avg_gini = model_stats.get("avg_gini", {}) or {}
    avg_entropy = model_stats.get("avg_entropy", {}) or {}

    # Direction key for stats lookup
    if pred_target > 0:
        dir_key = "up"
    elif pred_target < 0:
        dir_key = "down"
    else:
        dir_key = "flat"

    dir_votes = float(vote_counts.get(dir_key, 0) or 0)
    total_votes = float(total_strong) if total_strong > 0 else 1.0
    vote_agreement = dir_votes / total_votes

    coverage = float(total_strong) / float(total_models) if total_models > 0 else 0.0

    g = float(avg_gini.get(dir_key, 0.0) or 0.0)
    ent = float(avg_entropy.get(dir_key, 1.0) or 1.0)

    # Model-based component: majority agreement, coverage, and sharpness of probabilities.
    prob_sharpness = 0.5 * g + 0.5 * (1.0 - ent)
    model_conf = (
        0.4 * vote_agreement +
        0.3 * coverage +
        0.3 * prob_sharpness
    )
    model_conf = max(0.0, min(1.0, model_conf))

    # Risk / stop-loss component: if we're exiting because of thresholds,
    # confidence is higher the more decisively we crossed them.
    risk_conf = 0.5
    if close_trade and pct_pl is not None:
        threshold = take_profit_rate if pct_pl > 0 else stop_loss_rate
        if threshold and threshold > 0:
            ratio = min(abs(pct_pl) / threshold, 2.0)
            # Map ratio in [0,2] → risk_conf in [0.5, 1.0]
            risk_conf = 0.5 + 0.25 * ratio
            risk_conf = max(0.5, min(1.0, risk_conf))

    # Combine model and risk components
    final_conf = 0.7 * model_conf + 0.3 * risk_conf

    # Extra dampening for "NONE" with almost no model support
    if action == "NONE" and pred_target == 0 and total_strong <= 1:
        final_conf *= 0.7

    return float(max(0.0, min(1.0, final_conf)))


# =====================================================
# TRADING LOGIC (CORE ENGINE)
# =====================================================

def _gap_in_4h_bars(start: datetime, end: datetime) -> int:
    if start is None:
        return 10**9
    if end < start:
        start, end = end, start
    diff_hours = (end - start).total_seconds() / 3600.0
    return int(diff_hours // 4)


def _decide_trade_action(
    current_price: float,
    current_timestamp: datetime,
    position_qty: float,
    ticker: str,
    df: pd.DataFrame,
    is_backtest: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    risk_state = _load_risk_state(ticker)
    entry_price = risk_state.get("last_entry_price")
    last_entry_ts_str = risk_state.get("last_entry_timestamp")
    last_direction = risk_state.get("last_direction")
    stop_loss_rate = float(risk_state.get("loss_close_rate", LOSS_CLOSE_RATE))
    take_profit_rate = float(risk_state.get("gain_close_rate", GAIN_CLOSE_RATE))
    cooloff_after_close_trade = int(
        risk_state.get("cooloff_after_close_trade", COOL_OFF_AFTER_CLOSE_TRADE)
    )
    last_close_ts_str = risk_state.get("last_close_timestamp")
    last_action = risk_state.get("last_action")

    last_entry_ts = (
        datetime.fromisoformat(last_entry_ts_str)
        if last_entry_ts_str else None
    )
    last_close_ts = (
        datetime.fromisoformat(last_close_ts_str)
        if last_close_ts_str else None
    )

    # 1) Get model-based directional signal
    (
        pred_target,
        models_up,
        models_down,
        models_stable,
        model_stats,
    ) = _get_pred_target_for_timestamp(
        df=df,
        current_time=current_timestamp,
        ticker=ticker,
        is_backtest=is_backtest,
    )

    # 2) Compute P&L and exit trigger
    q = float(position_qty)
    pl = 0.0
    pct_pl = 0.0
    close_trade = False

    if q != 0 and entry_price is not None:
        entry_price = float(entry_price)
        # PL = q * (p - entry)
        pl = q * (current_price - entry_price)
        trade_value = abs(q * entry_price)
        if trade_value > 0:
            pct_pl = pl / trade_value
            # Stop-loss / Take-profit
            close_trade = (
                (pl < 0 and abs(pct_pl) > stop_loss_rate) or
                (pl > 0 and abs(pct_pl) > take_profit_rate)
            )

    # 3) Cool-off after forced closes (in 4H bars)
    cooloff_buy = False
    cooloff_sell = False
    gap_since_last_forced = _gap_in_4h_bars(last_close_ts, current_timestamp)

    if last_close_ts is not None and gap_since_last_forced < cooloff_after_close_trade:
        if last_action == "close_long":
            cooloff_buy = True
        elif last_action == "close_short":
            cooloff_sell = True

    action = "NONE"
    reason = ""

    # 4) Forced exit has highest priority
    if q < 0 and close_trade:
        # Close short via BUY
        action = "BUY"
        reason = f"EXIT: close short after {pct_pl*100:.3f}% loss/gain"
        risk_state["last_close_timestamp"] = current_timestamp.isoformat()
        risk_state["last_action"] = "close_short"
        risk_state["last_entry_price"] = None
        risk_state["last_entry_timestamp"] = None
        risk_state["last_direction"] = None

    elif q > 0 and close_trade:
        # Close long via SELL
        action = "SELL"
        reason = f"EXIT: close long after {pct_pl*100:.3f}% loss/gain"
        risk_state["last_close_timestamp"] = current_timestamp.isoformat()
        risk_state["last_action"] = "close_long"
        risk_state["last_entry_price"] = None
        risk_state["last_entry_timestamp"] = None
        risk_state["last_direction"] = None

    # 5) Long entry logic (pred_target > 0) if not in buy cool-off
    elif pred_target > 0 and not cooloff_buy:
        if q < 0:
            # First step: close short via BUY
            action = "BUY"
            reason = "LONG: close short"
            risk_state["last_close_timestamp"] = current_timestamp.isoformat()
            risk_state["last_action"] = "close_short"
            risk_state["last_entry_price"] = None
            risk_state["last_entry_timestamp"] = None
            risk_state["last_direction"] = None

        elif q == 0:
            # Open long via BUY (parameters already tuned in risk_state, if grid search is enabled)
            action = "BUY"
            reason = "LONG: open long"
            risk_state["last_entry_price"] = float(current_price)
            risk_state["last_entry_timestamp"] = current_timestamp.isoformat()
            risk_state["last_direction"] = "long"
            # store the concrete stop-loss and take-profit price levels
            risk_state["stop_loss_price"] = current_price * (1.0 - stop_loss_rate)
            risk_state["take_profit_price"] = current_price * (1.0 + take_profit_rate)

    # 6) Short entry logic (pred_target < 0) if not in sell cool-off
    elif pred_target < 0 and not cooloff_sell:
        if q > 0:
            # First step: close long via SELL
            action = "SELL"
            reason = "SHORT: close long"
            risk_state["last_close_timestamp"] = current_timestamp.isoformat()
            risk_state["last_action"] = "close_long"
            risk_state["last_entry_price"] = None
            risk_state["last_entry_timestamp"] = None
            risk_state["last_direction"] = None
        elif q == 0:
            # Open short via SELL
            action = "SELL"
            reason = "SHORT: open short"
            risk_state["last_entry_price"] = float(current_price)
            risk_state["last_entry_timestamp"] = current_timestamp.isoformat()
            risk_state["last_direction"] = "short"
            # store the concrete stop-loss and take-profit price levels for short
            risk_state["stop_loss_price"] = current_price * (1.0 + stop_loss_rate)
            risk_state["take_profit_price"] = current_price * (1.0 - take_profit_rate)

    # 7) Persist updated risk state
    _save_risk_state(ticker, risk_state)

    # 8) Compute overall confidence score for this action
    confidence_score = _compute_action_confidence(
        action=action,
        pred_target=pred_target,
        close_trade=close_trade,
        pct_pl=pct_pl,
        stop_loss_rate=stop_loss_rate,
        take_profit_rate=take_profit_rate,
        model_stats=model_stats,
    )

    debug_info = {
        "pred_target": pred_target,
        "models_up": models_up,
        "models_down": models_down,
        "models_stable": models_stable,
        "position_qty": q,
        "entry_price": entry_price,
        "pl": pl,
        "pct_pl": pct_pl,
        "close_trade": close_trade,
        "cooloff_buy": cooloff_buy,
        "cooloff_sell": cooloff_sell,
        "stop_loss_rate": stop_loss_rate,
        "take_profit_rate": take_profit_rate,
        "cooloff_after_close_trade": cooloff_after_close_trade,
        "reason": reason,
        "confidence_score": confidence_score,
        "model_stats": model_stats,
    }

    return action, debug_info



# =====================================================
# PUBLIC ENTRYPOINTS
# =====================================================

def run_logic(current_price, predicted_price, ticker):
    from forest import api, buy_shares, sell_shares  # imported here to keep script importable

    logger = logging.getLogger(__name__)

    try:
        account = api.get_account()
        cash = float(account.cash)
    except Exception as e:
        logger.error(f"[{ticker}] Error fetching account details: {e}")
        return

    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception:
        position_qty = 0.0

    # Load full data and assume the last row is the current candle
    df = _load_raw_data(ticker)
    if df.empty:
        logger.info(f"[{ticker}] No data available; skipping logic.")
        return

    df = df.sort_values("time")
    latest_row = df.iloc[-1]
    current_timestamp = latest_row["time"]

    # Sanity check: ensure passed current_price is consistent (optional)
    # but do not enforce strictly to avoid accidental runtime issues.
    # Decide action using shared engine
    action, debug_info = _decide_trade_action(
        current_price=float(current_price),
        current_timestamp=current_timestamp,
        position_qty=position_qty,
        ticker=ticker,
        df=df
    )

    logger.info(
        f"[{ticker}] Live Decision: action={action}, "
        f"debug={debug_info}"
    )

    if action == "BUY":
        # If currently short, cover full; if flat, open long with trade_rate * cash notionally
        if position_qty < 0:
            qty_to_buy = abs(position_qty)
        elif position_qty == 0:
            to_invest = 3.0 * cash  # use trade_rate = 3 by default; adjust if desired
            qty_to_buy = int(to_invest // current_price)
        else:
            # Already long; no additional buy in this logic
            qty_to_buy = 0

        if qty_to_buy > 0:
            logger.info(f"[{ticker}] Executing BUY {qty_to_buy} @ {current_price}")
            buy_shares(ticker, qty_to_buy, current_price, predicted_price)

    elif action == "SELL":
        # If currently long, sell full; if flat, open short with trade_rate * cash notionally
        if position_qty > 0:
            qty_to_sell = position_qty
        elif position_qty == 0:
            to_short = 3.0 * cash  # trade_rate = 3
            qty_to_sell = int(to_short // current_price)
        else:
            # Already short; no additional sell in this logic
            qty_to_sell = 0

        if qty_to_sell > 0:
            logger.info(f"[{ticker}] Executing SELL {qty_to_sell} @ {current_price}")
            sell_shares(ticker, qty_to_sell, current_price, predicted_price)

    else:
        logger.info(f"[{ticker}] Action=NONE; no trade executed.")


def run_backtest(
    current_price,
    predicted_price,
    position_qty,
    current_timestamp,
    candles,
    ticker,
    confidence
):
    """
    Backtest logic.

    - Ignores predicted_price and candles for decision making to keep parity
      with run_logic.
    - Uses only data up to and including `current_timestamp` from the CSV.
    - Uses the same CatBoost voting + trading engine as run_logic.
    - Returns:
        * if confidence=False: "BUY" | "SELL" | "NONE"
        * if confidence=True: (action, confidence_score in [0,1])
    """
    logger = logging.getLogger(__name__)

    # Ensure current_timestamp is datetime
    if isinstance(current_timestamp, str):
        current_timestamp = datetime.fromisoformat(current_timestamp)

    df = _load_raw_data(ticker)
    df = df[df["time"] <= current_timestamp].copy()
    if df.empty:
        logger.info(f"[{ticker}] No data up to {current_timestamp}; return NONE.")
        if confidence:
            return "NONE", 0.0
        return "NONE"

    action, debug_info = _decide_trade_action(
        current_price=float(current_price),
        current_timestamp=current_timestamp,
        position_qty=float(position_qty),
        ticker=ticker,
        df=df,
        is_backtest=True
    )

    logger.debug(
        f"[{ticker}] Backtest Decision at {current_timestamp}: action={action}, "
        f"debug={debug_info}"
    )

    if confidence:
        print("Results: ", action, float(debug_info.get("confidence_score", 0.0)))
        return action, float(debug_info.get("confidence_score", 0.0))

    return action

