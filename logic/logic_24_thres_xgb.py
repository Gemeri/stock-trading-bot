import os
import json
import math
import logging
from typing import Optional, Tuple, Dict, Any

import logic.tools as tools
from bot.trading.orders import buy_shares, sell_shares
import numpy as np
import pandas as pd

# ---- Env & Globals -----------------------------------------------------------
def get_csv_filename(ticker):
    return  tools.get_csv_filename(ticker)

BASE_FEATURES = tools.FEATURES

XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.05
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE = 0.8
XGB_MIN_CHILD_WEIGHT = 1.0
XGB_EARLY_STOP_ROUNDS = 25
XGB_RANDOM_STATE = 42
XGB_PROBA_THRESHOLD = 0.60

# Trading / risk params
STOP_LOSS_PCT = 0.02
MIN_TRAIN_ROWS = 300
VAL_SPLIT = 0.2

# Cache directory (required to be in 'xgboost' folder)
CACHE_DIR = os.path.join("xgboost")
os.makedirs(CACHE_DIR, exist_ok=True)

# Logger
logger = logging.getLogger(__name__)

HALF_LIFE = 1500
# ---- Utilities ---------------------------------------------------------------
def add_weights(n_samples: int) -> np.ndarray:
    ages = np.arange(n_samples - 1, -1, -1)
    return np.exp(-np.log(2) * ages / HALF_LIFE)

def _state_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"{ticker}_{tools.CONVERTED_TIMEFRAME}_state.json")

def _load_state(ticker: str) -> Dict[str, Any]:
    path = _state_path(ticker)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_state(ticker: str, state: Dict[str, Any]) -> None:
    path = _state_path(ticker)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)

def _clear_state_if_flat(ticker: str, position_qty: float) -> None:
    """If the portfolio is flat, clear entry/stop to avoid stale state."""
    if position_qty <= 0:
        st = _load_state(ticker)
        changed = False
        for k in ["entry_price", "stop_loss", "last_action", "last_signal_ts"]:
            if k in st:
                st.pop(k, None)
                changed = True
        if changed:
            _save_state(ticker, st)

def _parse_current_ts(ts_in) -> Optional[pd.Timestamp]:
    if ts_in is None:
        return None
    try:
        # Accept pd.Timestamp, int/float epoch, or str-like
        if isinstance(ts_in, pd.Timestamp):
            return ts_in
        if isinstance(ts_in, (int, float)):
            return pd.to_datetime(int(ts_in), unit="s", utc=True)
        return pd.to_datetime(ts_in, utc=True)
    except Exception:
        return None


# ---- Data Loading & Preparation ---------------------------------------------
def _load_dataset_upto(ticker: str, cutoff_ts: Optional[pd.Timestamp]) -> Optional[pd.DataFrame]:
    """Load CSV and trim to rows where timestamp <= cutoff_ts (if provided)."""
    csv_path = get_csv_filename(ticker)
    if not os.path.exists(csv_path):
        logger.error(f"[{ticker}] Data CSV not found at: {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"[{ticker}] Failed to read CSV: {e}")
        return None

    # Ensure essential columns
    for col in ["timestamp", "close"]:
        if col not in df.columns:
            logger.error(f"[{ticker}] Missing required column '{col}' in CSV.")
            return None

    # Normalize timestamp
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    except Exception as e:
        logger.error(f"[{ticker}] Failed to parse 'timestamp': {e}")
        return None

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if cutoff_ts is not None:
        df = df.loc[df["timestamp"] <= cutoff_ts].reset_index(drop=True)

    # Gate features: use only those available
    available_feats = [c for c in BASE_FEATURES if (c in df.columns and c != "timestamp")]
    if len(available_feats) == 0:
        logger.error(f"[{ticker}] No model features available after gating.")
        return None

    # Build targets: next candle up/down (relative to current close)
    df["future_close"] = df["close"].shift(-1)
    df["y_up"] = (df["future_close"] > df["close"]).astype(np.int8)
    df["y_down"] = (df["future_close"] < df["close"]).astype(np.int8)

    # Drop last row for training (no future)
    df_train = df.iloc[:-1].copy()
    if df_train.shape[0] < MIN_TRAIN_ROWS:
        logger.warning(f"[{ticker}] Not enough rows to train (have {df_train.shape[0]}, need {MIN_TRAIN_ROWS}).")
        return None

    # Ensure no NaN in features subset for training
    feat_cols = available_feats
    train_cols = feat_cols + ["y_up", "y_down", "timestamp"]
    df_train = df_train.dropna(subset=feat_cols).reset_index(drop=True)

    # Keep final row (current context) for live prediction
    df_live = df.iloc[-1:].copy()
    if df_live[feat_cols].isnull().any().any():
        logger.warning(f"[{ticker}] Live row has NaNs in features; cannot infer.")
        return None

    return df, df_train, df_live, feat_cols


def _split_train_valid(df_train: pd.DataFrame, feat_cols: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Time-aware train/valid split (last VAL_SPLIT portion used as validation)."""
    n = df_train.shape[0]
    val_size = max(1, int(n * VAL_SPLIT))
    split_idx = n - val_size
    return df_train.iloc[:split_idx], df_train.iloc[split_idx:]


# ---- Modeling ----------------------------------------------------------------
def _train_two_xgb(df_train: pd.DataFrame, df_valid: pd.DataFrame, feat_cols: list):
    try:
        from xgboost import XGBClassifier
    except Exception as e:
        logger.error(f"XGBoost is not available: {e}")
        return None, None

    # Prepare datasets
    X_tr = df_train[feat_cols].values
    y_up_tr = df_train["y_up"].values
    y_dn_tr = df_train["y_down"].values

    X_va = df_valid[feat_cols].values
    y_up_va = df_valid["y_up"].values
    y_dn_va = df_valid["y_down"].values

    # Imbalance handling
    def _spw(y):
        pos = float((y == 1).sum())
        neg = float((y == 0).sum())
        return (neg / max(pos, 1.0)) if pos > 0 else 1.0

    params = dict(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE,
        min_child_weight=XGB_MIN_CHILD_WEIGHT,
        early_stopping_rounds=XGB_EARLY_STOP_ROUNDS,
        objective="binary:logistic",
        eval_metric=["auc", "logloss"],
        tree_method="auto",
        random_state=XGB_RANDOM_STATE,
        n_jobs=int(str(os.cpu_count() or 4)),
        verbosity=0
    )

    up_clf = XGBClassifier(**params, scale_pos_weight=_spw(y_up_tr))
    dn_clf = XGBClassifier(**params, scale_pos_weight=_spw(y_dn_tr))

    weights = add_weights(len(X_tr))
    # Early stopping with validation
    up_clf.fit(
        X_tr, y_up_tr,
        sample_weight=weights,
        eval_set=[(X_va, y_up_va)],
        verbose=False,
    )
    dn_clf.fit(
        X_tr, y_dn_tr,
        sample_weight=weights,
        eval_set=[(X_va, y_dn_va)],
        verbose=False,
    )
    return up_clf, dn_clf


def _infer_signal(up_clf, dn_clf, x_live: np.ndarray, proba_thr: float) -> Tuple[str, Dict[str, float]]:
    """Return 'BUY', 'SELL', or 'NONE' based on agreement rule with thresholds."""
    p_up = float(up_clf.predict_proba(x_live)[0, 1]) if up_clf else 0.0
    p_dn = float(dn_clf.predict_proba(x_live)[0, 1]) if dn_clf else 0.0

    pred_up = int(p_up >= proba_thr)
    pred_dn = int(p_dn >= proba_thr)

    # Agreement rule:
    #  - BUY  if up==1 and down==0
    #  - SELL if up==0 and down==1
    #  - NONE otherwise (including both 1 or both 0)
    if pred_up == 1 and pred_dn == 0:
        action = "BUY"
    elif pred_up == 0 and pred_dn == 1:
        action = "SELL"
    else:
        action = "NONE"

    return action, {"p_up": p_up, "p_down": p_dn, "pred_up": pred_up, "pred_down": pred_dn}


# ---- Trading Decision Core (shared by live & backtest) -----------------------
def _decide_action(
    ticker: str,
    current_price: float,
    position_qty: float,
    cutoff_ts: Optional[pd.Timestamp],
) -> Tuple[str, Dict[str, Any]]:
    telemetry: Dict[str, Any] = {"reason": "", "details": {}}

    # 0) Clear stale state if flat
    _clear_state_if_flat(ticker, position_qty)

    # 1) Trailing stop check (always identical in live/backtest)
    state = _load_state(ticker)
    if position_qty > 0 and "stop_loss" in state:
        stop_loss = float(state.get("stop_loss", 0.0))
        if current_price <= stop_loss and stop_loss > 0:
            telemetry["reason"] = "STOP_LOSS"
            telemetry["details"] = {"stop_loss": stop_loss, "current_price": current_price}
            # clear state on exit
            for k in ["entry_price", "stop_loss"]:
                state.pop(k, None)
            state["last_action"] = "SELL"
            _save_state(ticker, state)
            return "SELL", telemetry

    # 2) Load dataset up to cutoff
    loaded = _load_dataset_upto(ticker, cutoff_ts)
    if not loaded:
        telemetry["reason"] = "NO_DATA"
        return "NONE", telemetry
    df, df_train, df_live, feat_cols = loaded

    # Last feature row for live inference (no look-ahead)
    x_live = df_live[feat_cols].values

    # Time-aware split
    df_tr, df_va = _split_train_valid(df_train, feat_cols)
    if df_va.empty or df_tr.empty:
        telemetry["reason"] = "SPLIT_TOO_SMALL"
        return "NONE", telemetry

    # 3) Train fresh models (no caching)
    up_clf, dn_clf = _train_two_xgb(df_tr, df_va, feat_cols)
    if up_clf is None or dn_clf is None:
        telemetry["reason"] = "MODEL_ERROR"
        return "NONE", telemetry

    # 4) Agreement signal
    action, probs = _infer_signal(up_clf, dn_clf, x_live, XGB_PROBA_THRESHOLD)
    telemetry["probs"] = probs
    telemetry["features_used"] = len(feat_cols)
    telemetry["reason"] = "MODEL_SIGNAL"

    # 5) Apply position-aware gating and trailing stop maintenance
    if action == "BUY":
        if position_qty > 0:
            # Already long; ignore buy
            telemetry["reason"] = "ALREADY_LONG"
            return "NONE", telemetry
        # initialize trailing stop
        entry = float(current_price)
        stop = entry * (1.0 - STOP_LOSS_PCT)
        state["entry_price"] = entry
        state["stop_loss"] = stop
        state["last_action"] = "BUY"
        state["last_signal_ts"] = str(df_live["timestamp"].iloc[0])
        _save_state(ticker, state)
        telemetry["details"] = {"entry_price": entry, "stop_loss": stop}
        return "BUY", telemetry

    elif action == "SELL":
        if position_qty <= 0:
            telemetry["reason"] = "NO_POSITION_TO_SELL"
            return "NONE", telemetry
        # clear entry/stop on sell
        for k in ["entry_price", "stop_loss"]:
            state.pop(k, None)
        state["last_action"] = "SELL"
        state["last_signal_ts"] = str(df_live["timestamp"].iloc[0])
        _save_state(ticker, state)
        return "SELL", telemetry

    # Update trailing stop if still long (raise stop only)
    if position_qty > 0 and "entry_price" in state:
        new_stop = current_price * (1.0 - STOP_LOSS_PCT)
        if float(state.get("stop_loss", 0.0)) < new_stop:
            state["stop_loss"] = new_stop
            _save_state(ticker, state)
            telemetry["details"]["trail_stop_raised_to"] = new_stop

    return "NONE", telemetry


# ---- Live Trading Entrypoint -------------------------------------------------
def run_logic(current_price, predicted_price, ticker):
    from forest import api

    logger = logging.getLogger(__name__)
    # Account & position
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

    # Decide action using latest CSV (no cutoff => up to most recent row)
    action, info = _decide_action(
        ticker=ticker,
        current_price=float(current_price),
        position_qty=position_qty,
        cutoff_ts=None,
    )

    logger.info(
        f"[{ticker}] Price={current_price} | Action={action} | Info={info}"
    )

    # Execute
    if action == "BUY" and position_qty == 0:
        max_shares = int(cash // float(current_price))
        if max_shares > 0:
            logger.info(f"[{ticker}] BUY {max_shares} @ {current_price}")
            try:
                # Keep signature compatibility; predicted_price is ignored everywhere
                buy_shares(ticker, max_shares, float(current_price), float(current_price))
            except Exception as e:
                logger.error(f"[{ticker}] BUY failed: {e}")
        else:
            logger.info(f"[{ticker}] Insufficient cash to buy.")
    elif action == "SELL" and position_qty > 0:
        logger.info(f"[{ticker}] SELL {position_qty} @ {current_price}")
        try:
            sell_shares(ticker, float(position_qty), float(current_price), float(current_price))
        except Exception as e:
            logger.error(f"[{ticker}] SELL failed: {e}")
    else:
        logger.info(f"[{ticker}] NONE")


# ---- Backtest Entrypoint -----------------------------------------------------
def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles, ticker):
    cutoff_ts = _parse_current_ts(current_timestamp)
    action, info = _decide_action(
        ticker=ticker,
        current_price=float(current_price),
        position_qty=float(position_qty),
        cutoff_ts=cutoff_ts,
    )
    # For transparency while backtesting (you can capture logs externally)
    logger = logging.getLogger(__name__)
    logger.info(f"[BT {ticker}] ts={current_timestamp} price={current_price} pos={position_qty} -> {action} {info}")
    return action
