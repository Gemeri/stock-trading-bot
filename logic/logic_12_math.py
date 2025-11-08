import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from datetime import datetime
from pysr import PySRRegressor
from sklearn.model_selection import TimeSeriesSplit
from skopt import gp_minimize
from scipy.stats import zscore

# ========== ENV VARS, PATHS, LOGGING ==========

load_dotenv()
BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "1Hour")
TIMEFRAME_MAP = {
    "4Hour": "H4", "2Hour": "H2", "1Hour": "H1",
    "30Min": "M30", "15Min": "M15"
}
CONVERTED_TIMEFRAME = TIMEFRAME_MAP.get(BAR_TIMEFRAME, BAR_TIMEFRAME)

def get_csv_filename(ticker):
    return os.path.join("data", f"{ticker}_{CONVERTED_TIMEFRAME}.csv")

REPORTS_DIR = "trade_reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, filename=os.path.join(REPORTS_DIR, 'trading.log'),
                    filemode='a', format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ========== UTILS ==========

FEATURES = [
    'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions', 'sentiment',
    'greed_index', 'news_count', 'news_volume_z', 'price_change', 'high_low_range', 
    'macd_line', 'macd_signal', 'macd_histogram', 'rsi', 'roc', 'atr', 'ema_9', 'ema_21', 
    'ema_50', 'ema_200', 'adx', 'obv', 'bollinger_percB', 'returns_1', 
    'returns_3', 'returns_5', 'std_5', 'std_10', 'lagged_close_1', 'lagged_close_2', 
    'lagged_close_3', 'lagged_close_5', 'lagged_close_10', 'candle_body_ratio', 
    'wick_dominance', 'gap_vs_prev', 'volume_zscore', 'atr_zscore', 'rsi_zscore',
    'month', 'hour_sin', 'hour_cos', 'day_of_week_sin',  'day_of_week_cos', 
    'days_since_high', 'days_since_low', "d_sentiment"
]

def parse_timestamp(x):
    # Accept both string and pd.Timestamp
    if isinstance(x, pd.Timestamp):
        return x
    try:
        return pd.to_datetime(x)
    except Exception:
        return pd.to_datetime(x, unit='s')

def load_full_csv(ticker):
    csv = get_csv_filename(ticker)
    df = pd.read_csv(csv)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
    else:
        raise ValueError('CSV must have "timestamp" column.')
    return df

def get_historical_data_up_to(timestamp, ticker):
    df = load_full_csv(ticker)
    df = df[df['timestamp'] <= parse_timestamp(timestamp)]
    df = df.reset_index(drop=True)
    return df

def get_data_for_live(ticker):
    return load_full_csv(ticker)

def clean_features(df):
    df = df.copy()
    # Keep only columns in FEATURES and drop NaN/inf rows
    features = [f for f in FEATURES if f in df.columns]
    df = df[features + ['timestamp', 'close']]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

# ========== TARGET CREATION (NO LOOKAHEAD!) ==========

def create_target(df, n_candles_ahead=3, threshold=0.002):
    future_close = df['close'].shift(-n_candles_ahead)
    ret = (future_close - df['close']) / df['close']
    y = np.zeros_like(ret)
    y[ret > threshold] = 1
    y[ret < -threshold] = -1
    return y

# ========== PY SR TUNING ==========

def hyperparameter_search(X, y, n_trials=10, time_limit=180):
    def score(params):
        niterations, maxsize, populations = params
        reg = PySRRegressor(
            niterations=int(niterations),
            populations=int(populations),
            maxsize=int(maxsize),
            unary_operators=["neg", "abs", "square", "log", "exp", "sin", "cos"],
            binary_operators=["+", "-", "*", "/", "pow"],
            select_k_features=10,
            progress=False,
            temp_equation_file=True,
            model_selection="best",
            verbosity=0,
            random_state=42
        )
        try:
            reg.fit(X, y)
            pred = reg.predict(X)
            score = np.mean(np.sign(pred) == np.sign(y))
            # Penalize complexity
            expr_complex = len(str(reg.get_best().sympy_expr))
            penalty = 0.002 * expr_complex
            return -(score - penalty)
        except Exception as e:
            logger.warning(f"PySR failed: {e}")
            return 1e6  # Large penalty

    space = [
        (50, 150),    # niterations
        (10, 30),     # maxsize
        (5, 20),      # populations
    ]
    res = gp_minimize(score, space, n_calls=n_trials, random_state=42)
    best = res.x
    return {
        "niterations": int(best[0]),
        "maxsize": int(best[1]),
        "populations": int(best[2])
    }

# ========== CROSS-VALIDATION & THRESHOLD ==========

def time_series_cv_score(X, y, params, n_splits=3):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, test_idx in tscv.split(X):
        reg = PySRRegressor(
            niterations=params['niterations'],
            maxsize=params['maxsize'],
            populations=params['populations'],
            unary_operators=["neg", "abs", "square", "log", "exp", "sin", "cos"],
            binary_operators=["+", "-", "*", "/", "pow"],
            select_k_features=10,
            progress=False,
            temp_equation_file=True,
            model_selection="best",
            verbosity=0,
            random_state=42
        )
        reg.fit(X.iloc[train_idx], y[train_idx])
        pred = reg.predict(X.iloc[test_idx])
        score = np.mean(np.sign(pred) == np.sign(y[test_idx]))
        scores.append(score)
    return np.mean(scores)

def find_optimal_threshold(y_true, preds, min_signals=10):
    # Optimize threshold for best Sharpe
    candidate_thresholds = np.linspace(0.01, 1, 30)
    best_sharpe = -np.inf
    best_threshold = 0.1
    for t in candidate_thresholds:
        sig = np.zeros_like(preds)
        sig[preds > t] = 1
        sig[preds < -t] = -1
        # Enough signals?
        if np.sum(sig != 0) < min_signals:
            continue
        ret = (y_true * sig)
        mean = np.mean(ret)
        std = np.std(ret) + 1e-8
        sharpe = mean / std
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_threshold = t
    return best_threshold

# ========== FIT SYMBOLIC TRADER ==========

def fit_symbolic_trader(X, y):
    # Tune hyperparams
    params = hyperparameter_search(X, y, n_trials=12)
    # Validate via CV
    cv_score = time_series_cv_score(X, y, params)
    # Train on all data
    reg = PySRRegressor(
        niterations=params['niterations'],
        maxsize=params['maxsize'],
        populations=params['populations'],
        unary_operators=["neg", "abs", "square", "log", "exp", "sin", "cos"],
        binary_operators=["+", "-", "*", "/", "pow"],
        select_k_features=10,
        progress=False,
        temp_equation_file=True,
        model_selection="best",
        verbosity=0,
        random_state=42
    )
    reg.fit(X, y)
    pred = reg.predict(X)
    threshold = find_optimal_threshold(y, pred)
    return reg, threshold, params, cv_score

# ========== TRADE SIGNAL GENERATION ==========

def equation_predict_action(reg, threshold, features):
    pred = reg.predict(features)[0]
    if pred > threshold:
        return "BUY"
    elif pred < -threshold:
        return "SELL"
    else:
        return "NONE"

# ========== EVALUATION ==========

def compute_metrics(trade_log, initial_cash=10000):
    df = pd.DataFrame(trade_log)
    df = df.sort_values('timestamp').reset_index(drop=True)
    # Compute equity curve
    cash = initial_cash
    position = 0
    entry_price = 0
    portfolio = []
    for i, row in df.iterrows():
        action = row['action']
        price = row['price']
        qty = row['shares']
        if action == 'BUY':
            position += qty
            cash -= qty * price
            entry_price = price
        elif action == 'SELL':
            cash += qty * price
            position -= qty
        # Mark-to-market
        port_val = cash + position * price
        portfolio.append(port_val)
    # Metrics
    returns = pd.Series(portfolio).pct_change().dropna()
    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
    cumret = portfolio[-1] / initial_cash - 1
    rollmax = np.maximum.accumulate(portfolio)
    drawdown = (portfolio - rollmax) / rollmax
    max_dd = np.min(drawdown)
    return {
        'sharpe': sharpe,
        'total_return': cumret,
        'max_drawdown': max_dd,
        'curve': portfolio
    }

def plot_equity_curve(curve, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(curve)
    plt.title('Portfolio Curve')
    plt.xlabel('Trade')
    plt.ylabel('Equity')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ========== LIVE TRADE INTERFACE ==========

def run_logic(current_price, predicted_price, ticker):
    from forest import api, buy_shares, sell_shares

    logger.info(f"[{ticker}] Live logic: Price={current_price}, Predicted={predicted_price}")
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

    # Get all features for last row in CSV (realistically this would be streamed/assembled live)
    df = get_data_for_live(ticker)
    df = clean_features(df)
    # Use last available row (simulate "now")
    X_last = df[FEATURES].iloc[[-1]]

    # Train equation on full CSV with correct targets
    y = create_target(df)
    X = df[FEATURES]
    reg, threshold, params, cv_score = fit_symbolic_trader(X, y)

    # Save human-readable equation
    eqn_path = os.path.join(REPORTS_DIR, f"equation_{ticker}_live.txt")
    with open(eqn_path, "w") as f:
        f.write(str(reg.get_best().sympy_expr))

    action = equation_predict_action(reg, threshold, X_last)

    logger.info(f"[{ticker}] Action: {action} (Thresh={threshold:.4f})")

    if action == "BUY":
        if position_qty == 0:
            max_shares = int(cash // current_price)
            if max_shares > 0:
                buy_shares(ticker, max_shares, current_price, predicted_price)
                logger.info(f"[{ticker}] Buy {max_shares} @ {current_price}")
    elif action == "SELL":
        if position_qty > 0:
            sell_shares(ticker, position_qty, current_price, predicted_price)
            logger.info(f"[{ticker}] Sell {position_qty} @ {current_price}")
    # NONE: do nothing

# ========== BACKTEST LOGIC ==========

def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles, ticker):
    # 1. Get all data up to this candle
    hist = get_historical_data_up_to(current_timestamp, ticker)
    hist = clean_features(hist)
    if len(hist) < 100:
        return "NONE"  # Not enough data

    # 2. Fit model up to this candle
    y = create_target(hist)
    X = hist[FEATURES]
    reg, threshold, params, cv_score = fit_symbolic_trader(X, y)

    # 3. Use current features (last row in 'candles')
    last_row = candles.iloc[[-1]].copy()
    last_row = clean_features(last_row)
    if last_row.empty:
        return "NONE"
    X_now = last_row[FEATURES]

    action = equation_predict_action(reg, threshold, X_now)
    return action

# ========== BACKTEST DRIVER (FOR EXTERNAL USE/REPORTS) ==========

def run_full_backtest(ticker):
    df = load_full_csv(ticker)
    df = clean_features(df)
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:].reset_index(drop=True)

    y_train = create_target(train_df)
    X_train = train_df[FEATURES]
    reg, threshold, params, cv_score = fit_symbolic_trader(X_train, y_train)

    # Save equation
    eqn_path = os.path.join(REPORTS_DIR, f"equation_final.txt")
    with open(eqn_path, "w") as f:
        f.write(str(reg.get_best().sympy_expr))

    # Train and test Sharpe, Return, Drawdown
    y_test = create_target(test_df)
    X_test = test_df[FEATURES]
    pred_train = reg.predict(X_train)
    pred_test = reg.predict(X_test)

    # Apply threshold
    sig_train = np.zeros_like(pred_train)
    sig_train[pred_train > threshold] = 1
    sig_train[pred_train < -threshold] = -1
    sig_test = np.zeros_like(pred_test)
    sig_test[pred_test > threshold] = 1
    sig_test[pred_test < -threshold] = -1

    # Train/test metrics
    def get_perf(y, sig):
        r = y * sig
        sharpe = np.mean(r) / (np.std(r) + 1e-8) * np.sqrt(252)
        ret = np.sum(r)
        return sharpe, ret

    sharpe_train, ret_train = get_perf(y_train, sig_train)
    sharpe_test, ret_test = get_perf(y_test, sig_test)

    # Simulate trade log, single position, $10,000, no shorting, log all trades
    trade_log = []
    cash = 10000
    position = 0
    entry_price = 0
    portfolio = []
    for i, row in test_df.iterrows():
        features = row[FEATURES].values.reshape(1, -1)
        pred = reg.predict(features)[0]
        ts = row['timestamp']
        px = row['close']
        act = "NONE"
        # BUY
        if pred > threshold and position == 0:
            qty = int(cash // px)
            if qty > 0:
                trade_log.append({'timestamp': ts, 'action': 'BUY', 'price': px, 'shares': qty, 'current_price': px})
                position += qty
                cash -= qty * px
                entry_price = px
                act = "BUY"
        # SELL
        elif pred < -threshold and position > 0:
            trade_log.append({'timestamp': ts, 'action': 'SELL', 'price': px, 'shares': position, 'current_price': px})
            cash += position * px
            position = 0
            entry_price = 0
            act = "SELL"
        # Portfolio curve
        port_val = cash + position * px
        portfolio.append(port_val)
        # Log "NONE" actions too for completeness
        if act == "NONE":
            trade_log.append({'timestamp': ts, 'action': 'NONE', 'price': px, 'shares': 0, 'current_price': px})

    # Save logs
    trade_log_path = os.path.join(REPORTS_DIR, "trade_log.csv")
    pd.DataFrame(trade_log).to_csv(trade_log_path, index=False)

    # Compute metrics and plot
    metrics = compute_metrics(trade_log)
    plot_file = os.path.join(REPORTS_DIR, "equity_curve.png")
    plot_equity_curve(metrics['curve'], plot_file)

    # Save summary
    report_file = os.path.join(REPORTS_DIR, "summary.txt")
    with open(report_file, "w") as f:
        f.write(f"Train Sharpe: {sharpe_train:.2f}\n")
        f.write(f"Test Sharpe: {sharpe_test:.2f}\n")
        f.write(f"Test Total Return: {metrics['total_return']:.2%}\n")
        f.write(f"Test Max Drawdown: {metrics['max_drawdown']:.2%}\n")
        f.write(f"Equation: {str(reg.get_best().sympy_expr)}\n")
        f.write(f"Hyperparameters: {params}\n")

# ========== MODULE EXPOSES ONLY THE MAIN INTERFACES ==========

__all__ = ['run_logic', 'run_backtest']
