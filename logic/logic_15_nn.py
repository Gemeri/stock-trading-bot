import os
import logging
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.optim as optim

# --------------- ENVIRONMENT CONFIGURATION -----------------
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

def get_csv_filename(ticker: str) -> str:
    return os.path.join("data", f"{ticker}_{CONVERTED_TIMEFRAME}.csv")

# ----------------- LOGGER SETUP -----------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ----------------- NEURAL NETWORK ----------------------
class TradeNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=48, output_dim=3):
        super(TradeNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim//2),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim//2, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ----------------- FEATURE/UTILITY FUNCTIONS -------------------
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

ACTION_MAP = {0: "NONE", 1: "BUY", 2: "SELL"}
ACTION_MAP_INV = {"NONE": 0, "BUY": 1, "SELL": 2}

def _preprocess_df(df: pd.DataFrame):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(method='ffill').fillna(0)
    # Ensure all features are float32 (prevents object dtype issues)
    for col in FEATURES:
        df[col] = df[col].astype(np.float32)
    return df

def _label_targets(df: pd.DataFrame, threshold=0.002):
    y = []
    closes = df['close'].values
    for i in range(len(df)-1):
        future = closes[i+1]
        curr = closes[i]
        pct = (future - curr) / curr
        if pct > threshold:
            y.append(ACTION_MAP_INV["BUY"])
        elif pct < -threshold:
            y.append(ACTION_MAP_INV["SELL"])
        else:
            y.append(ACTION_MAP_INV["NONE"])
    y.append(ACTION_MAP_INV["NONE"])  # Last row: can't know future
    return np.array(y)

def _standardize(train_df: pd.DataFrame, test_row: pd.Series):
    X_train = train_df[FEATURES].values.astype(np.float32)
    means = np.mean(X_train, axis=0)
    stds = np.std(X_train, axis=0) + 1e-8
    X_train_std = ((X_train - means) / stds).astype(np.float32)
    X_test_std = (((test_row[FEATURES].values.astype(np.float32)) - means) / stds).reshape(1, -1).astype(np.float32)
    return X_train_std, X_test_std


# ----------------- MAIN LOGIC -----------------------
_model_cache = {}  # Cache models for re-use during a single run (backtest only)

def run_logic(current_price, predicted_price, ticker):
    from forest import api, buy_shares, sell_shares

    try:
        filename = get_csv_filename(ticker)
        df = pd.read_csv(filename)
    except Exception as e:
        logger.error(f"[{ticker}] Could not load CSV: {e}")
        return

    df = _preprocess_df(df)
    if len(df) < 60:
        logger.error(f"[{ticker}] Not enough data for model.")
        return

    X = df[FEATURES]
    y = _label_targets(df)
    X_std, X_live = _standardize(df, df.iloc[-1])

    device = torch.device("cpu")
    model = TradeNN(input_dim=len(FEATURES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    # Short, efficient live retrain (use only last 500 rows for speed)
    X_train = torch.tensor(X_std[-500:], dtype=torch.float32)
    y_train = torch.tensor(y[-500:], dtype=torch.long)

    model.train()
    for epoch in range(16):
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()

    # Predict for current
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(np.asarray(X_live, dtype=np.float32)))
        action = torch.argmax(logits, dim=1).item()
    logger.info(f"[{ticker}] Live model predicts: {ACTION_MAP[action]}. Current price: {current_price}, Predicted: {predicted_price}")

    # Position management
    try:
        account = api.get_account()
        cash = float(account.cash)
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception as e:
        position_qty = 0.0
        cash = 0.0

    if action == ACTION_MAP_INV["BUY"] and position_qty == 0 and cash >= current_price:
        max_shares = int(cash // current_price)
        logger.info(f"[{ticker}] Buying {max_shares} shares at {current_price}.")
        buy_shares(ticker, max_shares, current_price, predicted_price)
    elif action == ACTION_MAP_INV["SELL"] and position_qty > 0:
        logger.info(f"[{ticker}] Selling {position_qty} shares at {current_price}.")
        sell_shares(ticker, position_qty, current_price, predicted_price)
    else:
        logger.info(f"[{ticker}] No action taken. Holding position: {position_qty}.")

def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles, ticker):
    try:
        filename = get_csv_filename(ticker)
        df_full = pd.read_csv(filename)
    except Exception as e:
        logger.error(f"[{ticker}] Could not load CSV: {e}")
        return "NONE"

    # Preprocess with timestamp as datetime
    df_full = _preprocess_df(df_full)

    # Parse current_timestamp to datetime if not already
    if not np.issubdtype(type(current_timestamp), np.datetime64):
        try:
            ts_compare = pd.to_datetime(current_timestamp)
        except Exception as e:
            logger.error(f"Could not parse current_timestamp '{current_timestamp}': {e}")
            return "NONE"
    else:
        ts_compare = current_timestamp

    # Select rows up to and including current_timestamp
    df_step = df_full[df_full['timestamp'] <= ts_compare]
    if len(df_step) < 60:
        logger.warning(f"[{ticker}] Not enough history for backtest at ts={current_timestamp}")
        return "NONE"

    # Use model cache for speed-up (if possible)
    cache_key = (ticker, str(ts_compare))
    if cache_key in _model_cache:
        model, means, stds = _model_cache[cache_key]
    else:
        y = _label_targets(df_step)
        X_std, X_live = _standardize(df_step, df_step.iloc[-1])
        device = torch.device("cpu")
        model = TradeNN(input_dim=len(FEATURES)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003)

        X_train = torch.tensor(X_std[-500:], dtype=torch.float32)
        y_train = torch.tensor(y[-500:], dtype=torch.long)
        model.train()
        for epoch in range(10):  # Keep epochs low for backtest speed
            optimizer.zero_grad()
            out = model(X_train)
            loss = criterion(out, y_train)
            loss.backward()
            optimizer.step()
        means = np.mean(df_step[FEATURES].values, axis=0)
        stds = np.std(df_step[FEATURES].values, axis=0) + 1e-8
        _model_cache[cache_key] = (model, means, stds)

    # Prepare input for current step
    test_row = df_step.iloc[-1]
    X_input = ((test_row[FEATURES].values - means) / stds).reshape(1, -1)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(np.asarray(X_input, dtype=np.float32)))
        action = torch.argmax(logits, dim=1).item()

    # Trading constraints: Only BUY if flat, SELL if long
    if action == ACTION_MAP_INV["BUY"] and position_qty == 0:
        logger.info(f"[{ticker}] Backtest: BUY at {current_price} (ts={current_timestamp})")
        return "BUY"
    elif action == ACTION_MAP_INV["SELL"] and position_qty > 0:
        logger.info(f"[{ticker}] Backtest: SELL at {current_price} (ts={current_timestamp})")
        return "SELL"
    else:
        logger.info(f"[{ticker}] Backtest: NONE at {current_price} (ts={current_timestamp})")
        return "NONE"
