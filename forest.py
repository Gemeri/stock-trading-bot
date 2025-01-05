import os
import sys
import time
import logging
import threading
import pytz
import schedule
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import alpaca_trade_api as tradeapi

# --------------------------------------------------------------------------
# 1. Configuration (Loaded from .env)
# --------------------------------------------------------------------------
load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")
API_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Comma-separated tickers, e.g. "TSLA,AAPL,MSFT"
TICKERS = os.getenv("TICKERS", "TSLA").split(",")

# Default timeframe (if not provided)
BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "4Hour")

# Number of bars/candles to fetch
N_BARS = int(os.getenv("N_BARS", "5000"))

TRADE_LOG_FILENAME = "trade_log.csv"

# RandomForest parameters
N_ESTIMATORS = 100
RANDOM_SEED = 42

# We'll use New York local time
NY_TZ = pytz.timezone("America/New_York")

# Global shutdown flag
SHUTDOWN = False

# --------------------------------------------------------------------------
# 2. Logging Setup
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# --------------------------------------------------------------------------
# 3. Alpaca API Setup
# --------------------------------------------------------------------------
api = tradeapi.REST(API_KEY, API_SECRET, API_BASE_URL, api_version='v2')

# --------------------------------------------------------------------------
# 4. Helper: Candle Fetching + Utility
# --------------------------------------------------------------------------

def timeframe_to_code(tf: str) -> str:
    """Convert timeframe string to short code for filenames."""
    mapping = {
        "15Min": "M15",
        "30Min": "M30",
        "1Hour": "H1",
        "2Hour": "H2",
        "4Hour": "H4",
        "1Day": "D1"
    }
    return mapping.get(tf, tf)

def fetch_candles(ticker: str, bars: int = 5000, timeframe: str = None) -> pd.DataFrame:
    if not timeframe:
        timeframe = BAR_TIMEFRAME

    end_dt = datetime.now(tz=pytz.utc)
    start_dt = end_dt - timedelta(days=1826)  # ~5 years

    logging.info(f"[{ticker}] Fetching {bars} {timeframe} bars from Alpaca.")
    try:
        barset = api.get_bars(
            symbol=ticker,
            timeframe=timeframe,
            limit=bars,
            start=start_dt.isoformat(),
            end=end_dt.isoformat(),
            adjustment='raw',
            feed='iex'
        )
    except Exception as e:
        logging.error(f"[{ticker}] Error fetching bars: {e}")
        return pd.DataFrame()

    df = barset.df
    if df.empty:
        logging.warning(f"[{ticker}] No data returned from get_bars().")
        return pd.DataFrame()

    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(ticker, level='symbol', drop_level=False)

    df = df.reset_index()
    df.rename(columns={
        'timestamp': 'timestamp',
        'trade_count': 'transactions',
        'Volume': 'volume'
    }, inplace=True)

    if 'vwap' not in df.columns:
        if 'vw' in df.columns:
            df['vwap'] = df['vw']
        else:
            df['vwap'] = np.nan

    if 'transactions' not in df.columns:
        df['transactions'] = np.nan

    final_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions']
    for c in final_cols:
        if c not in df.columns:
            df[c] = np.nan

    df = df[final_cols]
    logging.info(f"[{ticker}] Fetched {len(df)} bars.")
    return df

# --------------------------------------------------------------------------
# 5. Feature Engineering
# --------------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few naive features: price_change, high_low_range, log_volume."""
    df['price_change'] = df['close'] - df['open']
    df['high_low_range'] = df['high'] - df['low']
    df['log_volume'] = np.log1p(df['volume'])
    return df

# --------------------------------------------------------------------------
# 6. Single-Step Next-Candle Prediction (Used by Live Trading)
# --------------------------------------------------------------------------
def train_and_predict(df: pd.DataFrame) -> float:
    """
    Train on entire dataset (minus final shift) and predict the *next candle's close*.
    Used by the scheduled "run_job" for live trading logic.
    """
    try:
        df = add_features(df)
        df['target'] = df['close'].shift(-1)
        df.dropna(inplace=True)

        if len(df) < 10:
            logging.error("Not enough rows after shift to train. Need more candles.")
            return None

        features = [
            'open','high','low','close','volume','vwap',
            'price_change','high_low_range','log_volume'
        ]
        X = df[features]
        y = df['target']

        model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
        model.fit(X, y)

        # The last row is right before the "next" candle
        last_row_features = df.iloc[-1][features]
        last_row_df = pd.DataFrame([last_row_features], columns=features)

        predicted_close = model.predict(last_row_df)[0]
        return predicted_close
    except Exception as e:
        logging.error(f"Error during train_and_predict: {e}")
        return None

# --------------------------------------------------------------------------
# 7. Trading Logic
# --------------------------------------------------------------------------
def trade_logic(current_price: float, predicted_price: float, ticker: str):
    try:
        positions = api.list_positions()
    except Exception as e:
        logging.error(f"[{ticker}] Error listing positions: {e}")
        return

    position_qty = 0
    position_side = None

    for p in positions:
        if p.symbol == ticker:
            position_qty = float(p.qty)
            position_side = 'long' if position_qty > 0 else 'short'
            break

    try:
        account = api.get_account()
        cash = float(account.cash)
    except Exception as e:
        logging.error(f"[{ticker}] Error getting account info: {e}")
        return

    # 1) predicted > current & no long => BUY
    if predicted_price is not None and current_price is not None and (predicted_price > current_price):
        if position_side != 'long':
            if position_side == 'short':
                close_short(ticker, abs(position_qty), predicted_price)
            shares_to_buy = int(cash // current_price)
            if shares_to_buy > 0:
                buy_shares(ticker, shares_to_buy, current_price, predicted_price)

    # 2) predicted < current & we have a long => SELL
    if predicted_price is not None and current_price is not None and (predicted_price < current_price):
        if position_side == 'long':
            sell_shares(ticker, position_qty, current_price, predicted_price)

    # 3) predicted < current & no short => SHORT
    if predicted_price is not None and current_price is not None and (predicted_price < current_price):
        if position_side != 'short':
            if position_side == 'long':
                sell_shares(ticker, position_qty, current_price, predicted_price)
            shares_to_short = int(cash // current_price)
            if shares_to_short > 0:
                short_shares(ticker, shares_to_short, current_price, predicted_price)

    # 4) predicted > current & short => COVER
    if predicted_price is not None and current_price is not None and (predicted_price > current_price):
        if position_side == 'short':
            close_short(ticker, abs(position_qty), predicted_price)

def buy_shares(ticker, qty, buy_price, predicted_price):
    if qty <= 0:
        return
    try:
        api.submit_order(
            symbol=ticker,
            qty=qty,
            side='buy',
            type='market',
            time_in_force='day'
        )
        logging.info(f"[{ticker}] BUY {qty} at {buy_price:.2f} (Pred: {predicted_price:.2f})")
        log_trade("BUY", ticker, qty, buy_price, predicted_price, None)
    except Exception as e:
        logging.error(f"[{ticker}] Buy order failed: {e}")

def sell_shares(ticker, qty, sell_price, predicted_price):
    if qty <= 0:
        return
    try:
        pos = None
        try:
            pos = api.get_position(ticker)
        except:
            pass
        avg_entry = float(pos.avg_entry_price) if pos else 0.0

        api.submit_order(
            symbol=ticker,
            qty=qty,
            side='sell',
            type='market',
            time_in_force='day'
        )
        pl = (sell_price - avg_entry) * qty
        logging.info(f"[{ticker}] SELL {qty} at {sell_price:.2f} (Pred: {predicted_price:.2f}, P/L: {pl:.2f})")
        log_trade("SELL", ticker, qty, sell_price, predicted_price, pl)
    except Exception as e:
        logging.error(f"[{ticker}] Sell order failed: {e}")

def short_shares(ticker, qty, short_price, predicted_price):
    if qty <= 0:
        return
    try:
        api.submit_order(
            symbol=ticker,
            qty=qty,
            side='sell',
            type='market',
            time_in_force='day'
        )
        logging.info(f"[{ticker}] SHORT {qty} at {short_price:.2f} (Pred: {predicted_price:.2f})")
        log_trade("SHORT", ticker, qty, short_price, predicted_price, None)
    except Exception as e:
        logging.error(f"[{ticker}] Short order failed: {e}")

def close_short(ticker, qty, cover_price):
    if qty <= 0:
        return
    try:
        pos = None
        try:
            pos = api.get_position(ticker)
        except:
            pass
        avg_short_price = float(pos.avg_entry_price) if pos else 0.0

        api.submit_order(
            symbol=ticker,
            qty=qty,
            side='buy',
            type='market',
            time_in_force='day'
        )
        pl = (avg_short_price - cover_price) * qty
        logging.info(f"[{ticker}] COVER SHORT {qty} at {cover_price:.2f} (P/L: {pl:.2f})")
        log_trade("COVER", ticker, qty, cover_price, None, pl)
    except Exception as e:
        logging.error(f"[{ticker}] Cover short failed: {e}")

def log_trade(action, ticker, qty, current_price, predicted_price, profit_loss):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "timestamp": now_str,
        "action": action,
        "ticker": ticker,
        "quantity": qty,
        "current_price": current_price,
        "predicted_price": predicted_price,
        "profit_loss": profit_loss
    }
    df = pd.DataFrame([row])
    if not os.path.exists(TRADE_LOG_FILENAME):
        df.to_csv(TRADE_LOG_FILENAME, index=False, mode='w')
    else:
        df.to_csv(TRADE_LOG_FILENAME, index=False, mode='a', header=False)

# --------------------------------------------------------------------------
# 8. Scheduling
# --------------------------------------------------------------------------
TIMEFRAME_SCHEDULE = {
    "15Min": [
        "09:30", "09:45", "10:00", "10:15", "10:30", "10:45",
        "11:00", "11:15", "11:30", "11:45", "12:00", "12:15",
        "12:30", "12:45", "13:00", "13:15", "13:30", "13:45",
        "14:00", "14:15", "14:30", "14:45", "15:00", "15:15",
        "15:30", "15:45", "16:00"
    ],
    "30Min": [
        "09:30", "10:00", "10:30", "11:00", "11:30", "12:00",
        "12:30", "13:00", "13:30", "14:00", "14:30", "15:00",
        "15:30", "16:00"
    ],
    "1Hour": [
        "09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"
    ],
    "2Hour": [
        "09:30", "11:30", "13:30", "15:30"
    ],
    "4Hour": [
        "09:30", "12:00"
    ],
    "1Day": [
        "09:30"
    ]
}

def setup_schedule_for_timeframe(timeframe: str):
    """
    Clears existing schedule and sets up times based on TIMEFRAME_SCHEDULE dict.
    If timeframe is invalid, falls back to '1Day'.
    """
    import schedule
    
    valid_keys = list(TIMEFRAME_SCHEDULE.keys())
    if timeframe not in valid_keys:
        logging.warning(f"{timeframe} not recognized. Falling back to 1Day schedule at 09:30.")
        timeframe = "1Day"

    times_list = TIMEFRAME_SCHEDULE[timeframe]

    schedule.clear()  # remove existing jobs
    for t in times_list:
        schedule.every().day.at(t).do(run_job)

    logging.info(f"Scheduled run_job() for timeframe={timeframe} at these NY times: {times_list}")

def run_job():
    """Scheduled job for live trading: fetch data, predict next candle, do trade."""
    logging.info("Starting scheduled job...")

    # Check if market is open
    try:
        clock = api.get_clock()
        if not clock.is_open:
            logging.info("Market not open. Skipping job.")
            return
    except Exception as e:
        logging.error(f"Error checking market status: {e}")
        return

    # Perform trades
    _perform_trading_job()

    logging.info("Scheduled job finished.")


def _perform_trading_job():
    for ticker in TICKERS:
        df = fetch_candles(ticker, bars=N_BARS, timeframe=BAR_TIMEFRAME)
        if df.empty:
            logging.error(f"[{ticker}] Data fetch returned empty DataFrame. Skipping.")
            continue

        tf_code = timeframe_to_code(BAR_TIMEFRAME)
        csv_filename = f"{ticker}_{tf_code}.csv"
        try:
            df.to_csv(csv_filename, index=False)
            logging.info(f"[{ticker}] Saved candle data to {csv_filename}")
        except Exception as e:
            logging.error(f"[{ticker}] Error saving CSV: {e}")
            continue

        pred_close = train_and_predict(df)
        if pred_close is None:
            logging.error(f"[{ticker}] Model training or prediction failed. Skipping trade logic.")
            continue

        current_price = df.iloc[-1]['close']
        logging.info(f"[{ticker}] Current Price = {current_price:.2f}, Predicted Next Close = {pred_close:.2f}")

        trade_logic(current_price, pred_close, ticker)

# --------------------------------------------------------------------------
# 9. Console Listener
# --------------------------------------------------------------------------
def console_listener():
    global SHUTDOWN
    while not SHUTDOWN:
        cmd_line = sys.stdin.readline().strip()
        if not cmd_line:
            continue

        parts = cmd_line.split()
        cmd = parts[0].lower()

        if cmd == "turnoff":
            logging.info("Received 'turnoff' command. Shutting down gracefully...")
            schedule.clear()
            SHUTDOWN = True

        elif cmd == "api-test":
            logging.info("Testing Alpaca API keys...")
            try:
                account = api.get_account()
                logging.info(f"Account Cash: {account.cash}")
                logging.info("Alpaca API keys are valid.")
            except Exception as e:
                logging.error(f"Alpaca API test failed: {e}")

        elif cmd == "get-data":
            timeframe = BAR_TIMEFRAME
            if len(parts) > 1:
                timeframe = parts[1]
            logging.info(f"Received 'get-data' command with timeframe={timeframe}.")

            for ticker in TICKERS:
                df = fetch_candles(ticker, bars=N_BARS, timeframe=timeframe)
                if df.empty:
                    logging.error(f"[{ticker}] No data. Could not save CSV.")
                    continue
                tf_code = timeframe_to_code(timeframe)
                csv_filename = f"{ticker}_{tf_code}.csv"
                df.to_csv(csv_filename, index=False)
                logging.info(f"[{ticker}] Saved candle data to {csv_filename}.")

        elif cmd == "feature-importance":
            for ticker in TICKERS:
                tf_code = timeframe_to_code(BAR_TIMEFRAME)
                csv_filename = f"{ticker}_{tf_code}.csv"
                if not os.path.exists(csv_filename):
                    logging.info(f"[{ticker}] {csv_filename} not found. Fetching data.")
                    df = fetch_candles(ticker, bars=N_BARS, timeframe=BAR_TIMEFRAME)
                    if df.empty:
                        logging.error(f"[{ticker}] Empty data, skip.")
                        continue
                    df.to_csv(csv_filename, index=False)
                else:
                    df = pd.read_csv(csv_filename)

                df = add_features(df)
                df['target'] = df['close'].shift(-1)
                df.dropna(inplace=True)
                if len(df) < 10:
                    logging.error(f"[{ticker}] Not enough data for feature importance.")
                    continue

                features = [
                    'open','high','low','close','volume','vwap',
                    'price_change','high_low_range','log_volume'
                ]
                X = df[features]
                y = df['target']

                model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
                model.fit(X, y)

                fi = model.feature_importances_
                fi_dict = dict(zip(features, fi))

                logging.info(f"Feature Importances for {ticker} ({BAR_TIMEFRAME}):")
                for k, v in fi_dict.items():
                    logging.info(f"   {k}: {v:.5f}")

        elif cmd == "predict-next":
            """
            predict-next
            - For each ticker, fetch data, train, predict the next close, print it
            - No multi-step forecast, just a single step
            """
            for ticker in TICKERS:
                df = fetch_candles(ticker, bars=N_BARS, timeframe=BAR_TIMEFRAME)
                if df.empty:
                    logging.error(f"[{ticker}] Empty data, skipping predict-next.")
                    continue

                pred_close = train_and_predict(df)
                if pred_close is None:
                    logging.error(f"[{ticker}] No prediction generated.")
                    continue

                current_price = df.iloc[-1]['close']
                logging.info(f"[{ticker}] Current Price={current_price:.2f}, Predicted Next Close={pred_close:.2f}")

        elif cmd == "force-run":
            """
            force-run
            - Bypass the market open check and run trades immediately
            """
            logging.info("Force-running job now (ignoring market open check).")
            _perform_trading_job()
            logging.info("Force-run job finished.")

        elif cmd == "backtest":
            if len(parts) < 2:
                logging.warning("Usage: backtest <N> [timeframe optional]")
                continue

            try:
                test_size = int(parts[1])
            except ValueError:
                logging.error("Invalid test_size for 'backtest' command.")
                continue

            timeframe = BAR_TIMEFRAME
            if len(parts) > 2:
                timeframe = parts[2]

            for ticker in TICKERS:
                df = fetch_candles(ticker, bars=N_BARS, timeframe=timeframe)
                if df.empty:
                    logging.error(f"[{ticker}] No data to backtest.")
                    continue

                # We'll do a single-step backtest from your existing code (not changed)
                df = add_features(df)
                df['target'] = df['close'].shift(-1)
                df.dropna(inplace=True)

                if len(df) <= test_size + 1:
                    logging.error("Not enough rows for backtest split.")
                    continue

                features = [
                    'open','high','low','close','volume','vwap',
                    'price_change','high_low_range','log_volume'
                ]
                train_end_idx = len(df) - test_size
                train_data = df.iloc[:train_end_idx]
                test_data  = df.iloc[train_end_idx:]

                X_train = train_data[features]
                y_train = train_data['target']

                X_test = test_data[features]
                y_test = test_data['close']

                model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                rmse = math.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                avg_close = y_test.mean()
                accuracy = 100 - (mae / avg_close * 100)

                logging.info(f"[{ticker}] Backtest: test_size={test_size}, RMSE={rmse:.4f}, "
                             f"MAE={mae:.4f}, Accuracy={accuracy:.2f}%")

                out_df = pd.DataFrame({
                    'timestamp': test_data['timestamp'].values,
                    'actual_close': y_test.values,
                    'predicted_close': y_pred
                })
                tf_code = timeframe_to_code(timeframe)
                out_csv = f"backtest_{ticker}_{test_size}_{tf_code}.csv"
                out_df.to_csv(out_csv, index=False)
                logging.info(f"[{ticker}] Saved backtest predictions to {out_csv}.")

                # Quick plot
                plt.figure(figsize=(10, 6))
                plt.plot(out_df['timestamp'], out_df['actual_close'], label='Actual')
                plt.plot(out_df['timestamp'], out_df['predicted_close'], label='Predicted')
                plt.title(f"{ticker} Backtest (Last {test_size} rows) - {timeframe}")
                plt.xlabel("Timestamp")
                plt.ylabel("Close Price")
                plt.legend()
                plt.grid(True)
                if test_size > 30:
                    plt.xticks(rotation=45)
                plt.tight_layout()
                out_img = f"backtest_{ticker}_{test_size}_{tf_code}.png"
                plt.savefig(out_img)
                plt.close()
                logging.info(f"[{ticker}] Saved backtest plot to {out_img}.")

        else:
            logging.warning(f"Unrecognized command: {cmd_line}")

# --------------------------------------------------------------------------
# 10. Main Loop
# --------------------------------------------------------------------------
def main():
    setup_schedule_for_timeframe(BAR_TIMEFRAME)
    listener_thread = threading.Thread(target=console_listener, daemon=True)
    listener_thread.start()

    logging.info("Bot started. Running schedule in local system time (NY).")
    logging.info("Commands:")
    logging.info("  turnoff -> stop script gracefully")
    logging.info("  api-test -> check Alpaca credentials")
    logging.info("  get-data [timeframe optional] -> fetch & save CSV for each ticker")
    logging.info("  feature-importance -> train & show feature importances")
    logging.info("  predict-next -> predict the next candle's close for each ticker")
    logging.info("  force-run -> run trades ignoring market open check")
    logging.info("  backtest <N> [timeframe optional] -> single-step approach on last N rows")

    while not SHUTDOWN:
        try:
            schedule.run_pending()
            time.sleep(20)
        except Exception as e:
            logging.error(f"Main loop error: {e}")
            time.sleep(60)

    logging.info("Main loop exited. Bye!")

if __name__ == "__main__":
    main()
