import os
import sys
import time
import math
import logging
import threading
import pytz
import schedule
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

import alpaca_trade_api as tradeapi

# -------------------------------
# 1. Load Configuration (from .env)
# -------------------------------
load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")
API_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

TICKERS = os.getenv("TICKERS", "TSLA").split(",")
BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "4Hour")
N_BARS = int(os.getenv("N_BARS", "5000"))
TRADE_LOG_FILENAME = "trade_log.csv"

N_ESTIMATORS = 100
RANDOM_SEED = 42
NY_TZ = pytz.timezone("America/New_York")

SHUTDOWN = False

# -------------------------------
# 2. Logging Setup
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# -------------------------------
# 3. Alpaca API Setup
# -------------------------------
api = tradeapi.REST(API_KEY, API_SECRET, API_BASE_URL, api_version='v2')

# -------------------------------
# 4. Sentiment Analysis Model Setup
# -------------------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

def predict_sentiment(text):
    """
    Returns predicted sentiment details.
    Mapping: {0: 'Negative', 1: 'Neutral', 2: 'Positive'}.
    We calculate sentiment_score as (positive confidence – negative confidence).
    """
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    sentiment_class = outputs.logits.argmax(dim=1).item()
    confidence_scores = outputs.logits.softmax(dim=1)[0].tolist()
    sentiment_score = confidence_scores[2] - confidence_scores[0]
    return sentiment_class, confidence_scores, sentiment_score

# -------------------------------
# 5. Candles Helper Functions
# -------------------------------
def timeframe_to_code(tf: str) -> str:
    mapping = {
        "15Min": "M15",
        "30Min": "M30",
        "1Hour": "H1",
        "2Hour": "H2",
        "4Hour": "H4",
        "1Day": "D1"
    }
    return mapping.get(tf, tf)

def get_bars_per_day(tf: str) -> float:
    mapping = {
        "15Min": 32,
        "30Min": 16,
        "1Hour": 8,
        "2Hour": 4,
        "4Hour": 2,
        "1Day": 1
    }
    return mapping.get(tf, 1)

def fetch_candles(ticker: str, bars: int = 5000, timeframe: str = None) -> pd.DataFrame:
    if not timeframe:
        timeframe = BAR_TIMEFRAME
    bars_per_day = get_bars_per_day(timeframe)
    required_days = math.ceil((bars / bars_per_day) * 1.25)
    end_dt = datetime.now(tz=pytz.utc)
    start_dt = end_dt - timedelta(days=required_days)
    logging.info(f"[{ticker}] Fetching {bars} {timeframe} bars from {start_dt.isoformat()} to {end_dt.isoformat()}.")
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
    for c in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
        if c not in df.columns:
            df[c] = np.nan
    if 'vwap' not in df.columns:
        df['vwap'] = np.nan
    if 'trade_count' in df.columns:
        df.rename(columns={'trade_count': 'transactions'}, inplace=True)
    else:
        df['transactions'] = np.nan
    final_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions']
    df = df[final_cols]
    logging.info(f"[{ticker}] Fetched {len(df)} bars.")
    return df

# -------------------------------
# 6. News & Sentiment Functions
# -------------------------------
# These mappings determine how many days back and how many articles per day based on the timeframe.
NUM_DAYS_MAPPING = {
    "1Day": 1650,
    "4Hour": 1650,
    "2Hour": 1600,
    "1Hour": 800,
    "30Min": 400,
    "15Min": 200
}
ARTICLES_PER_DAY_MAPPING = {
    "1Day": 1,
    "4Hour": 2,
    "2Hour": 4,
    "1Hour": 8,
    "30Min": 16,
    "15Min": 32
}

def fetch_news_sentiments(ticker: str, num_days: int, articles_per_day: int):
    """
    For a given ticker, fetch news articles from Alpaca over the past num_days.
    For each day, up to articles_per_day are processed.
    Each article's sentiment is computed (sentiment_score = positive - negative).
    Returns a sorted list of dictionaries with keys 'created_at' and 'sentiment'.
    """
    news_list = []
    start_date_news = datetime.now(timezone.utc) - timedelta(days=num_days)
    today_dt = datetime.now(timezone.utc)
    total_days = (today_dt.date() - start_date_news.date()).days + 1
    for day_offset in range(total_days):
        current_day = start_date_news + timedelta(days=day_offset)
        if current_day > today_dt:
            break
        next_day = current_day + timedelta(days=1)
        start_str = current_day.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str   = next_day.strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            articles = api.get_news(ticker, start=start_str, end=end_str)
            if articles:
                count = min(len(articles), articles_per_day)
                for article in articles[:count]:
                    headline = article.headline if article.headline else ""
                    summary  = article.summary if article.summary else ""
                    combined_text = f"{headline} {summary}"
                    _ , confidence, sentiment_score = predict_sentiment(combined_text)
                    created_time = article.created_at if article.created_at else current_day
                    news_list.append({
                        "created_at": created_time,
                        "sentiment": sentiment_score
                    })
        except Exception as e:
            logging.error(f"Error fetching news for {ticker} on {current_day.date()}: {e}")
    news_list = sorted(news_list, key=lambda x: x['created_at'])
    return news_list

def assign_sentiment_to_candles(df: pd.DataFrame, news_list: list):
    """
    For each candle (row in df), assign the sentiment from the news article whose created_at
    is closest to the candle’s timestamp. If no article is found, carry forward the previous sentiment.
    Returns a list of sentiment values (floats).
    """
    sentiments = []
    last_sentiment = 0.0
    for idx, row in df.iterrows():
        try:
            candle_time = pd.to_datetime(row['timestamp']).replace(tzinfo=timezone.utc)
        except Exception:
            candle_time = datetime.now(timezone.utc)
        best = None
        best_diff = None
        for article in news_list:
            diff = abs((article['created_at'] - candle_time).total_seconds())
            if best is None or diff < best_diff:
                best = article
                best_diff = diff
        if best is not None:
            sentiment = best['sentiment']
            last_sentiment = sentiment
        else:
            sentiment = last_sentiment
        sentiments.append(sentiment)
    return sentiments

# New function: run only the sentiment update job.
def run_sentiment_job():
    """
    For each ticker, fetch the current candles,
    update the sentiment from news (using the timeframe-dependent parameters),
    and save the updated CSV.
    """
    for ticker in TICKERS:
        df = fetch_candles(ticker, bars=N_BARS, timeframe=BAR_TIMEFRAME)
        if df.empty:
            logging.error(f"[{ticker}] Unable to fetch candle data for sentiment update.")
            continue
        news_num_days = NUM_DAYS_MAPPING.get(BAR_TIMEFRAME, 1650)
        articles_per_day = ARTICLES_PER_DAY_MAPPING.get(BAR_TIMEFRAME, 1)
        news_list = fetch_news_sentiments(ticker, news_num_days, articles_per_day)
        df['sentiment'] = assign_sentiment_to_candles(df, news_list)
        tf_code = timeframe_to_code(BAR_TIMEFRAME)
        csv_filename = f"{ticker}_{tf_code}.csv"
        df.to_csv(csv_filename, index=False)
        logging.info(f"[{ticker}] Updated sentiment in candle CSV: {csv_filename}")

# -------------------------------
# 7. Feature Engineering (Candles)
# -------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df['price_change'] = df['close'] - df['open']
    df['high_low_range'] = df['high'] - df['low']
    df['log_volume'] = np.log1p(df['volume'])
    return df

# -------------------------------
# 8. Train and Predict (Including Sentiment Feature)
# -------------------------------
def train_and_predict(df: pd.DataFrame) -> float:
    try:
        df = add_features(df)
        df['target'] = df['close'].shift(-1)
        df.dropna(inplace=True)
        if len(df) < 10:
            logging.error("Not enough rows after shift to train. Need more candles.")
            return None
        features = [
            'open','high','low','close','volume','vwap',
            'price_change','high_low_range','log_volume','sentiment'
        ]
        X = df[features]
        y = df['target']
        model_rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
        model_rf.fit(X, y)
        last_row_features = df.iloc[-1][features]
        last_row_df = pd.DataFrame([last_row_features], columns=features)
        predicted_close = model_rf.predict(last_row_df)[0]
        return predicted_close
    except Exception as e:
        logging.error(f"Error during train_and_predict: {e}")
        return None

# -------------------------------
# 9. Improved Trading Logic & Order Functions
# -------------------------------
def trade_logic(current_price: float, predicted_price: float, ticker: str):
    try:
        try:
            pos = api.get_position(ticker)
            position_qty = float(pos.qty)
            current_position = "long" if position_qty > 0 else "short"
        except Exception:
            current_position = "none"
            position_qty = 0

        account = api.get_account()
        cash = float(account.cash)

        if predicted_price > current_price:
            # Want to go long.
            if current_position == "long":
                logging.info(f"[{ticker}] Already long. No action required.")
            elif current_position == "short":
                logging.info(f"[{ticker}] Currently short; covering short first.")
                close_short(ticker, abs(position_qty), current_price)
                account = api.get_account()
                cash = float(account.cash)
                shares_to_buy = int(cash // current_price)
                if shares_to_buy > 0:
                    buy_shares(ticker, shares_to_buy, current_price, predicted_price)
                else:
                    logging.info(f"[{ticker}] Insufficient buying power after covering short.")
            elif current_position == "none":
                shares_to_buy = int(cash // current_price)
                if shares_to_buy > 0:
                    buy_shares(ticker, shares_to_buy, current_price, predicted_price)
                else:
                    logging.info(f"[{ticker}] Insufficient buying power to go long.")
        elif predicted_price < current_price:
            # Want to go short.
            if current_position == "short":
                logging.info(f"[{ticker}] Already short. No action required.")
            elif current_position == "long":
                logging.info(f"[{ticker}] Currently long; selling long position first.")
                sell_shares(ticker, position_qty, current_price, predicted_price)
                account = api.get_account()
                cash = float(account.cash)
                shares_to_short = int(cash // current_price)
                if shares_to_short > 0:
                    short_shares(ticker, shares_to_short, current_price, predicted_price)
                else:
                    logging.info(f"[{ticker}] Insufficient funds to short after selling long.")
            elif current_position == "none":
                shares_to_short = int(cash // current_price)
                if shares_to_short > 0:
                    short_shares(ticker, shares_to_short, current_price, predicted_price)
                else:
                    logging.info(f"[{ticker}] Insufficient funds to open a short position.")
    except Exception as e:
        logging.error(f"[{ticker}] Error in trade_logic: {e}")

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
        logging.info(f"[{ticker}] BUY {qty} at {buy_price:.2f} (Predicted: {predicted_price:.2f})")
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
        logging.info(f"[{ticker}] SELL {qty} at {sell_price:.2f} (Predicted: {predicted_price:.2f}, P/L: {pl:.2f})")
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
        logging.info(f"[{ticker}] SHORT {qty} at {short_price:.2f} (Predicted: {predicted_price:.2f})")
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

# -------------------------------
# 10. Trading Job (with integrated sentiment update)
# -------------------------------
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

        # Update candles with sentiment information.
        news_num_days = NUM_DAYS_MAPPING.get(BAR_TIMEFRAME, 1650)
        articles_per_day = ARTICLES_PER_DAY_MAPPING.get(BAR_TIMEFRAME, 1)
        news_list = fetch_news_sentiments(ticker, news_num_days, articles_per_day)
        df['sentiment'] = assign_sentiment_to_candles(df, news_list)
        df.to_csv(csv_filename, index=False)
        logging.info(f"[{ticker}] Updated candle CSV with sentiment.")

        pred_close = train_and_predict(df)
        if pred_close is None:
            logging.error(f"[{ticker}] Model training or prediction failed. Skipping trade logic.")
            continue
        current_price = df.iloc[-1]['close']
        logging.info(f"[{ticker}] Current Price = {current_price:.2f}, Predicted Next Close = {pred_close:.2f}")
        trade_logic(current_price, pred_close, ticker)

# -------------------------------
# 11. Scheduling & Console Listener
# -------------------------------
# Times (in NY local time) for scheduled trading runs.
TIMEFRAME_SCHEDULE = {
    "15Min": ["09:30", "09:45", "10:00", "10:15", "10:30", "10:45",
              "11:00", "11:15", "11:30", "11:45", "12:00", "12:15",
              "12:30", "12:45", "13:00", "13:15", "13:30", "13:45",
              "14:00", "14:15", "14:30", "14:45", "15:00", "15:15",
              "15:30", "15:45", "16:00"],
    "30Min": ["09:30", "10:00", "10:30", "11:00", "11:30", "12:00",
              "12:30", "13:00", "13:30", "14:00", "14:30", "15:00",
              "15:30", "16:00"],
    "1Hour": ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"],
    "2Hour": ["09:30", "11:30", "13:30", "15:30"],
    "4Hour": ["09:30", "12:00"],
    "1Day": ["09:30"]
}

def setup_schedule_for_timeframe(timeframe: str):
    import schedule
    valid_keys = list(TIMEFRAME_SCHEDULE.keys())
    if timeframe not in valid_keys:
        logging.warning(f"{timeframe} not recognized. Falling back to 1Day schedule at 09:30.")
        timeframe = "1Day"
    times_list = TIMEFRAME_SCHEDULE[timeframe]
    schedule.clear()
    for t in times_list:
        schedule.every().day.at(t).do(run_job)
        # Also schedule the sentiment job 15 minutes prior:
        try:
            base_time = datetime.strptime(t, "%H:%M")
            sentiment_time = (base_time - timedelta(minutes=15)).strftime("%H:%M")
            schedule.every().day.at(sentiment_time).do(run_sentiment_job)
            logging.info(f"Scheduled sentiment update at {sentiment_time} NY time and trading at {t}.")
        except Exception as e:
            logging.error(f"Error scheduling sentiment job: {e}")
    logging.info(f"Scheduled run_job() for timeframe={timeframe} at NY times: {times_list}")

def run_job():
    logging.info("Starting scheduled trading job...")
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            clock = api.get_clock()
            if not clock.is_open:
                logging.info("Market not open. Skipping job.")
                return
            break
        except Exception as e:
            logging.error(f"Error checking market status (Attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(60)
            else:
                logging.error("Max retries exceeded. Skipping the scheduled job.")
                return
    _perform_trading_job()
    logging.info("Scheduled trading job finished.")

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
        elif cmd == "predict-next":
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
        elif cmd == "run-sentiment":
            logging.info("Running sentiment update job only...")
            run_sentiment_job()
            logging.info("Sentiment update job completed.")
        elif cmd == "force-run":
            logging.info("Force-running job now (ignoring market open check).")
            # Run sentiment update first
            run_sentiment_job()
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
                df = add_features(df)
                df['target'] = df['close'].shift(-1)
                df.dropna(inplace=True)
                if len(df) <= test_size + 1:
                    logging.error("Not enough rows for backtest split.")
                    continue
                features = ['open','high','low','close','volume','vwap','price_change','high_low_range','log_volume','sentiment']
                train_end_idx = len(df) - test_size
                train_data = df.iloc[:train_end_idx]
                test_data  = df.iloc[train_end_idx:]
                X_train = train_data[features]
                y_train = train_data['target']
                X_test = test_data[features]
                y_test = test_data['close']
                model_rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
                model_rf.fit(X_train, y_train)
                y_pred = model_rf.predict(X_test)
                rmse = math.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                avg_close = y_test.mean()
                accuracy = 100 - (mae / avg_close * 100)
                logging.info(f"[{ticker}] Backtest: test_size={test_size}, RMSE={rmse:.4f}, MAE={mae:.4f}, Accuracy={accuracy:.2f}%")
                out_df = pd.DataFrame({
                    'timestamp': test_data['timestamp'].values,
                    'actual_close': y_test.values,
                    'predicted_close': y_pred
                })
                tf_code = timeframe_to_code(timeframe)
                out_csv = f"backtest_{ticker}_{test_size}_{tf_code}.csv"
                out_df.to_csv(out_csv, index=False)
                logging.info(f"[{ticker}] Saved backtest predictions to {out_csv}.")
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

def main():
    setup_schedule_for_timeframe(BAR_TIMEFRAME)
    listener_thread = threading.Thread(target=console_listener, daemon=True)
    listener_thread.start()
    logging.info("Bot started. Running schedule in local NY time.")
    logging.info("Commands: turnoff, api-test, get-data [timeframe], run-sentiment, predict-next, force-run, backtest <N> [timeframe]")
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
