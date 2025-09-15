import os
import logging
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- ENV SETUP ---
load_dotenv()
BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "1Hour")

TIMEFRAME_MAP = {
    "4Hour": "H4",
    "2Hour": "H2",
    "1Hour": "H1",
    "30Min": "M30",
    "15Min": "M15"
}
CONVERTED_TIMEFRAME = TIMEFRAME_MAP.get(BAR_TIMEFRAME, BAR_TIMEFRAME)
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

ACTIONS = ["NONE", "BUY", "SELL"]  # Action label mapping

def get_csv_filename(ticker: str) -> str:
    return os.path.join("data", f"{ticker}_{CONVERTED_TIMEFRAME}.csv")

# --- LOGGER SETUP ---
logger = logging.getLogger("logic_nn_walkforward")
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# --- MODEL UTILS ---

def build_model(input_dim: int):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')  # Output: NONE, BUY, SELL (one-hot)
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def get_latest_model_and_scaler(ticker, cache, force_retrain, X_train, y_train):
    if cache.get("model") is not None and not force_retrain:
        return cache["model"], cache["scaler"]
    # Build or retrain model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    y_cat = np.zeros((len(y_train), 3))
    for i, y in enumerate(y_train):
        y_cat[i, ACTIONS.index(y)] = 1
    model = build_model(X_scaled.shape[1])
    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    model.fit(
        X_scaled, y_cat,
        epochs=40, batch_size=32,
        validation_split=0.15, verbose=0,
        callbacks=[early_stop]
    )
    cache["model"], cache["scaler"] = model, scaler
    return model, scaler

# --- DATA PREP ---

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def label_actions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["next_close"] = df["close"].shift(-1)
    df["action"] = "NONE"
    df.loc[df["next_close"] > df["close"], "action"] = "BUY"
    df.loc[df["next_close"] < df["close"], "action"] = "SELL"
    return df

# --- MAIN LOGIC FUNCTIONS ---

# Model and scaler cache (to avoid retraining if same data)
_MODEL_CACHE = {}

def run_logic(current_price, predicted_price, ticker):
    from forest import api, buy_shares, sell_shares

    # Load CSV, train model on ALL data (for live)
    csv_path = get_csv_filename(ticker)
    if not os.path.exists(csv_path):
        logger.error(f"Data CSV not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    df = prepare_data(df)
    df = label_actions(df)
    if len(df) < 100:
        logger.warning("Not enough data for model training.")
        return

    # Build model/scaler or use cache
    cache = _MODEL_CACHE.setdefault(ticker, {})
    X_train = df[FEATURES].values
    y_train = df["action"].values
    model, scaler = get_latest_model_and_scaler(ticker, cache, force_retrain=True, X_train=X_train, y_train=y_train)

    # Prepare latest features for prediction
    # Find last row matching current_price (safeguard)
    latest_row = df.iloc[-1].copy()
    latest_row["predicted_close"] = predicted_price  # update for live
    features = latest_row[FEATURES].values.reshape(1, -1)
    X_scaled = scaler.transform(features)

    pred = model.predict(X_scaled)[0]
    action_idx = int(np.argmax(pred))
    action = ACTIONS[action_idx]
    logger.info(f"[{ticker}] Model signal: {action} (prob={pred[action_idx]:.2f}), Current Price: {current_price}, Predicted Price: {predicted_price}")

    # Retrieve account/position info
    try:
        account = api.get_account()
        cash = float(account.cash)
    except Exception as e:
        logger.error(f"[{ticker}] Error fetching account: {e}")
        return

    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception:
        position_qty = 0.0

    # Trading logic (long only)
    if action == "BUY" and position_qty == 0:
        max_shares = int(cash // current_price)
        if max_shares > 0:
            logger.info(f"[{ticker}] Executing BUY: {max_shares} shares.")
            buy_shares(ticker, max_shares, current_price, predicted_price)
    elif action == "SELL" and position_qty > 0:
        logger.info(f"[{ticker}] Executing SELL: {position_qty} shares.")
        sell_shares(ticker, position_qty, current_price, predicted_price)
    else:
        logger.info(f"[{ticker}] No action taken.")

def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles, ticker):
    csv_path = get_csv_filename(ticker)
    if not os.path.exists(csv_path):
        logger.error(f"Data CSV not found: {csv_path}")
        return "NONE"
    df = pd.read_csv(csv_path)
    df = prepare_data(df)
    df = label_actions(df)
    # Use only data up to and including the current_timestamp
    df_hist = df[df["timestamp"] <= current_timestamp]
    if len(df_hist) < 100:
        logger.info(f"[{ticker}] Not enough data at timestamp {current_timestamp}")
        return "NONE"

    # Build model/scaler or use cache (cache by length of df_hist)
    cache_key = f"{ticker}_{current_timestamp}"
    if cache_key not in _MODEL_CACHE:
        # Use full history up to current candle
        X_train = df_hist[FEATURES].values
        y_train = df_hist["action"].values
        cache = {}
        model, scaler = get_latest_model_and_scaler(ticker, cache, force_retrain=True, X_train=X_train, y_train=y_train)
        _MODEL_CACHE[cache_key] = {"model": model, "scaler": scaler}
    else:
        model = _MODEL_CACHE[cache_key]["model"]
        scaler = _MODEL_CACHE[cache_key]["scaler"]

    # Get the feature row for this timestamp (from df_hist)
    row = df_hist[df_hist["timestamp"] == current_timestamp]
    if row.empty:
        logger.warning(f"[{ticker}] Timestamp {current_timestamp} not found.")
        return "NONE"
    row = row.iloc[0].copy()
    row["predicted_close"] = predicted_price  # Update with given predicted_price
    features = row[FEATURES].values.reshape(1, -1)
    X_scaled = scaler.transform(features)
    pred = model.predict(X_scaled)[0]
    action_idx = int(np.argmax(pred))
    action = ACTIONS[action_idx]
    logger.info(f"[{ticker}] Backtest @ {current_timestamp}: Model signal {action} (prob={pred[action_idx]:.2f})")
    # Long only: BUY only if no position, SELL only if holding
    if action == "BUY" and position_qty == 0:
        return "BUY"
    elif action == "SELL" and position_qty > 0:
        return "SELL"
    else:
        return "NONE"