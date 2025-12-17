import os
import logging
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import logic.tools as tools

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

# === Environment Setup ===
load_dotenv()

def get_csv_filename(ticker):
    return tools.get_csv_filename(ticker)

# === Feature List ===
FEATURES = tools.FEATURES
TARGET = "action"

# === Logging Setup ===
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s")

# === Neural Network/Scaler Model Cache (for efficiency during backtesting) ===
_MODEL_CACHE = {}

def _encode_action(action):
    return {"BUY": 0, "SELL": 1, "NONE": 2}[action]

def _decode_action(label):
    return {0: "BUY", 1: "SELL", 2: "NONE"}[label]

def _prepare_training_data(df):
    # Rule-based labeling:
    # If predicted_close > close: BUY (if not long), else NONE
    # If predicted_close < close: SELL (if long), else NONE
    # If predicted_close == close: NONE

    df = df.copy()
    df["action"] = "NONE"
    position = 0  # 0=flat, 1=long

    for idx in df.index:
        pred = df.loc[idx, "predicted_close"]
        close = df.loc[idx, "close"]

        if pred > close:
            if position == 0:
                df.at[idx, "action"] = "BUY"
                position = 1
            else:
                df.at[idx, "action"] = "NONE"
        elif pred < close:
            if position == 1:
                df.at[idx, "action"] = "SELL"
                position = 0
            else:
                df.at[idx, "action"] = "NONE"
        else:
            df.at[idx, "action"] = "NONE"
    df["action"] = df["action"].map(_encode_action)
    return df

def _load_data_up_to_timestamp(ticker, current_timestamp):
    fname = get_csv_filename(ticker)
    if not os.path.isfile(fname):
        logger.error(f"CSV file not found: {fname}")
        return None
    df = pd.read_csv(fname)
    # Ensure timestamp is sorted and filter
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df[df["timestamp"] <= current_timestamp].reset_index(drop=True)
    return df

def _train_or_update_model(ticker, df_train):
    X = df_train[FEATURES].astype(np.float32)
    X = X.fillna(0.0)
    y = df_train[TARGET].astype(int)

    # Use a fixed NN structure for stability and speed
    if ticker in _MODEL_CACHE:
        scaler, model = _MODEL_CACHE[ticker]
    else:
        scaler = StandardScaler()
        model = MLPClassifier(
            hidden_layer_sizes=(32, 16), activation='relu',
            solver='adam', batch_size=32,
            max_iter=20, warm_start=True, random_state=42
        )
        # Need to call partial_fit once with all possible classes
        scaler.fit(X)
        model.partial_fit(scaler.transform(X), y, classes=[0, 1, 2])
        _MODEL_CACHE[ticker] = (scaler, model)
        return scaler, model

    scaler.partial_fit(X)
    model.partial_fit(scaler.transform(X), y)
    return scaler, model

def run_logic(current_price, predicted_price, ticker):
    from forest import api, buy_shares, sell_shares

    # Load/cached model and scaler for the ticker
    global _MODEL_CACHE
    scaler, model = _MODEL_CACHE.get(ticker, (None, None))
    if scaler is None or model is None:
        logger.error(f"No trained model found for ticker: {ticker}. Run backtest first.")
        return

    # Get features from latest available row in the CSV
    fname = get_csv_filename(ticker)
    if not os.path.isfile(fname):
        logger.error(f"CSV file not found for ticker: {ticker}")
        return
    df = pd.read_csv(fname)
    df = df.sort_values("timestamp").reset_index(drop=True)
    features_row = df.iloc[-1][FEATURES].astype(np.float32).fillna(0.0).values.reshape(1, -1)
    # Model predicts one of: 0=BUY, 1=SELL, 2=NONE
    try:
        pred_action = model.predict(scaler.transform(features_row))[0]
        action = _decode_action(pred_action)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return

    # Retrieve account details and position
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
        position_qty = 0.0  # Assume no position if none exists

    logger.info(f"[{ticker}] Live | Price: {current_price:.2f}, Predicted: {predicted_price:.2f}, "
                f"Position: {position_qty}, Cash: {cash:.2f}, ModelAction: {action}")

    if action == "BUY" and position_qty == 0:
        max_shares = int(cash // current_price)
        if max_shares > 0:
            logger.info(f"[{ticker}] BUY {max_shares} shares at {current_price:.2f}")
            buy_shares(ticker, max_shares, current_price, predicted_price)
        else:
            logger.info(f"[{ticker}] Insufficient cash to BUY")
    elif action == "SELL" and position_qty > 0:
        logger.info(f"[{ticker}] SELL {position_qty} shares at {current_price:.2f}")
        sell_shares(ticker, position_qty, current_price, predicted_price)
    else:
        logger.info(f"[{ticker}] NO ACTION ({action})")

def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles, ticker):
    # Step 1: Load all data up to and including current_timestamp
    df_hist = _load_data_up_to_timestamp(ticker, current_timestamp)
    if df_hist is None or len(df_hist) < 50:
        logger.warning(f"[{ticker}] Not enough data to train at {current_timestamp}")
        return "NONE"

    # Step 2: Prepare training data (label ground truth using rules)
    df_train = _prepare_training_data(df_hist.copy())

    # Step 3: Train or update NN model and scaler
    scaler, model = _train_or_update_model(ticker, df_train)

    # Step 4: Extract latest row (current candle) as features for prediction
    features_row = df_hist.iloc[-1][FEATURES].astype(np.float32).fillna(0.0).values.reshape(1, -1)
    # Step 5: Model inference
    pred_action = model.predict(scaler.transform(features_row))[0]
    action = _decode_action(pred_action)

    logger.info(f"[{ticker}] Backtest | Time: {current_timestamp}, Price: {current_price:.2f}, "
                f"Predicted: {predicted_price:.2f}, Position: {position_qty}, NN_Action: {action}")

    # Step 6: Action constraint checks (no shorting)
    # Only BUY if no position, only SELL if long, else NONE
    if action == "BUY" and position_qty == 0:
        return "BUY"
    elif action == "SELL" and position_qty > 0:
        return "SELL"
    else:
        return "NONE"

# === End of Script ===
