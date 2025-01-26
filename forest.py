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

# A new environment variable controlling whether news fetching is on or off
NEWS_MODE = os.getenv("NEWS_MODE", "on").lower().strip()  # "on" or "off"

# A new environment variable controlling which features are disabled
DISABLED_FEATURES = os.getenv("DISABLED_FEATURES", "").strip()
if DISABLED_FEATURES:
    DISABLED_FEATURES_SET = set([f.strip() for f in DISABLED_FEATURES.split(",") if f.strip()])
else:
    DISABLED_FEATURES_SET = set()

# Make sure sentiment is NEVER disabled
if "sentiment" in DISABLED_FEATURES_SET:
    DISABLED_FEATURES_SET.remove("sentiment")

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

        logging.info(f"[{ticker}] Fetching news for day={current_day.date()} (start={start_str}, end={end_str})...")
        try:
            articles = api.get_news(ticker, start=start_str, end=end_str)
            if articles:
                logging.info(f"[{ticker}] Found {len(articles)} articles; processing up to {articles_per_day}")
                count = min(len(articles), articles_per_day)
                for article in articles[:count]:
                    headline = article.headline if article.headline else ""
                    summary  = article.summary if article.summary else ""
                    combined_text = f"{headline} {summary}"
                    _, _, sentiment_score = predict_sentiment(combined_text)
                    created_time = article.created_at if article.created_at else current_day
                    news_list.append({
                        "created_at": created_time,
                        "sentiment": sentiment_score,
                        "headline": headline,
                        "summary": summary
                    })
        except Exception as e:
            logging.error(f"Error fetching news for {ticker} on {current_day.date()}: {e}")
    news_list = sorted(news_list, key=lambda x: x['created_at'])
    logging.info(f"[{ticker}] Total news articles across all days: {len(news_list)}")
    return news_list

def assign_sentiment_to_candles(df: pd.DataFrame, news_list: list):
    logging.info("Assigning sentiment to candles...")
    sentiments = []
    last_sentiment = 0.0
    for _, row in df.iterrows():
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
    logging.info("Finished assigning sentiment to candles.")
    return sentiments

def save_news_to_csv(ticker: str, news_list: list):
    if not news_list:
        logging.info(f"[{ticker}] No articles to save.")
        return
    rows = []
    for item in news_list:
        item_sentiment_str = f"{item.get('sentiment', 0.0):.15f}"
        row = {
            "created_at": item.get("created_at", ""),
            "headline": item.get("headline", ""),
            "summary": item.get("summary", ""),
            "sentiment": item_sentiment_str
        }
        rows.append(row)

    df_news = pd.DataFrame(rows)
    csv_filename = f"{ticker}_articles_sentiment.csv"
    df_news.to_csv(csv_filename, index=False)
    logging.info(f"[{ticker}] Saved articles with sentiment to {csv_filename}")

def run_sentiment_job(skip_data=False):
    for ticker in TICKERS:
        if skip_data:
            tf_code = timeframe_to_code(BAR_TIMEFRAME)
            csv_filename = f"{ticker}_{tf_code}.csv"
            if not os.path.exists(csv_filename):
                logging.info(f"[{ticker}] run-sentiment -r: CSV not found, we must create it. Not skipping.")
            else:
                df_check = pd.read_csv(csv_filename)
                if 'sentiment' in df_check.columns and not df_check['sentiment'].isnull().values.any():
                    logging.info(f"[{ticker}] run-sentiment -r: Candle CSV has full sentiment. Skipping.")
                    continue
                else:
                    logging.info(f"[{ticker}] run-sentiment -r: Found CSV but missing/no sentiment. Will fill.")
        if NEWS_MODE == "off":
            logging.info(f"[{ticker}] NEWS_MODE=off, skipping any news fetch/assignment.")
            continue
        df = fetch_candles(ticker, bars=N_BARS, timeframe=BAR_TIMEFRAME)
        if df.empty:
            logging.error(f"[{ticker}] Unable to fetch candle data for sentiment update.")
            continue
        news_num_days = NUM_DAYS_MAPPING.get(BAR_TIMEFRAME, 1650)
        articles_per_day = ARTICLES_PER_DAY_MAPPING.get(BAR_TIMEFRAME, 1)
        news_list = fetch_news_sentiments(ticker, news_num_days, articles_per_day)

        save_news_to_csv(ticker, news_list)

        sentiments = assign_sentiment_to_candles(df, news_list)
        df["sentiment"] = sentiments
        df["sentiment"] = df["sentiment"].apply(lambda x: f"{x:.15f}")

        tf_code = timeframe_to_code(BAR_TIMEFRAME)
        csv_filename = f"{ticker}_{tf_code}.csv"
        df.to_csv(csv_filename, index=False)
        logging.info(f"[{ticker}] Updated sentiment in candle CSV: {csv_filename}")

# -------------------------------
# 7. Additional Feature Engineering
# -------------------------------
def compute_custom_features(df: pd.DataFrame) -> pd.DataFrame:

    # Example: price_return needs 'close' to compute percentage change
    if 'price_return' not in DISABLED_FEATURES_SET and 'close' in df.columns:
        df['price_return'] = df['close'].pct_change().fillna(0)

    # candle_rise = high - low
    if 'candle_rise' not in DISABLED_FEATURES_SET and 'high' in df.columns and 'low' in df.columns:
        df['candle_rise'] = df['high'] - df['low']

    # body_size = close - open
    if 'body_size' not in DISABLED_FEATURES_SET and 'close' in df.columns and 'open' in df.columns:
        df['body_size'] = df['close'] - df['open']

    # wick_to_body = (high - low)/(close - open)
    if 'wick_to_body' not in DISABLED_FEATURES_SET and all(col in df.columns for col in ['high','low','close','open']):
        body = (df['close'] - df['open']).replace(0, np.nan)  # avoid div-zero
        df['wick_to_body'] = ((df['high'] - df['low']) / body).replace(np.nan, 0)

    # macd_line
    if 'macd_line' not in DISABLED_FEATURES_SET and 'close' in df.columns:
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        df['macd_line'] = macd - signal

    # rsi
    if 'rsi' not in DISABLED_FEATURES_SET and 'close' in df.columns:
        window = 14
        delta = df['close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=window - 1, adjust=False).mean()
        ema_down = down.ewm(com=window - 1, adjust=False).mean()
        rs = ema_up / ema_down.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)

    # momentum = close - close.shift(14)
    if 'momentum' not in DISABLED_FEATURES_SET and 'close' in df.columns:
        df['momentum'] = df['close'] - df['close'].shift(14)
        df['momentum'] = df['momentum'].fillna(0)

    # rate of change (roc)
    if 'roc' not in DISABLED_FEATURES_SET and 'close' in df.columns:
        shifted = df['close'].shift(14)
        df['roc'] = ((df['close'] - shifted) / shifted.replace(0, np.nan)) * 100
        df['roc'] = df['roc'].fillna(0)

    # atr
    if 'atr' not in DISABLED_FEATURES_SET and all(col in df.columns for col in ['high','low','close']):
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        tr = high_low.combine(high_close, max).combine(low_close, max)
        df['atr'] = tr.ewm(span=14, adjust=False).mean().fillna(0)

    # hist_vol
    if 'hist_vol' not in DISABLED_FEATURES_SET and 'close' in df.columns:
        ret = df['close'].pct_change()
        rolling_std = ret.rolling(window=14).std()
        df['hist_vol'] = rolling_std * np.sqrt(14)
        df['hist_vol'] = df['hist_vol'].fillna(0)

    # obv and volume_change both need 'volume'
    if 'volume' in df.columns:
        # obv
        if 'obv' not in DISABLED_FEATURES_SET:
            df['obv'] = 0
            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i - 1]:
                    df.loc[i, 'obv'] = df.loc[i - 1, 'obv'] + df.loc[i, 'volume']
                elif df['close'].iloc[i] < df['close'].iloc[i - 1]:
                    df.loc[i, 'obv'] = df.loc[i - 1, 'obv'] - df.loc[i, 'volume']
                else:
                    df.loc[i, 'obv'] = df.loc[i - 1, 'obv']

        # volume_change
        if 'volume_change' not in DISABLED_FEATURES_SET:
            df['volume_change'] = df['volume'].pct_change().fillna(0)

    else:
        # If 'volume' is missing or disabled, skip them
        if 'obv' not in DISABLED_FEATURES_SET:
            logging.info("Cannot compute OBV: 'volume' missing or disabled.")
        if 'volume_change' not in DISABLED_FEATURES_SET:
            logging.info("Cannot compute volume_change: 'volume' missing or disabled.")

    # stoch_k
    if 'stoch_k' not in DISABLED_FEATURES_SET and all(col in df.columns for col in ['high','low','close']):
        low14 = df['low'].rolling(window=14).min()
        high14 = df['high'].rolling(window=14).max()
        stoch_k = (df['close'] - low14) / (high14 - low14.replace(0, np.nan)) * 100
        df['stoch_k'] = stoch_k.fillna(50)

    # bollinger
    if 'bollinger_upper' not in DISABLED_FEATURES_SET or 'bollinger_lower' not in DISABLED_FEATURES_SET:
        if 'close' in df.columns:
            ma20 = df['close'].rolling(window=20).mean()
            std20 = df['close'].rolling(window=20).std()
            if 'bollinger_upper' not in DISABLED_FEATURES_SET:
                df['bollinger_upper'] = ma20 + 2 * std20
            if 'bollinger_lower' not in DISABLED_FEATURES_SET:
                df['bollinger_lower'] = ma20 - 2 * std20

    # lagged_close
    if any(f'lagged_close_{lag}' not in DISABLED_FEATURES_SET for lag in [1,2,3,5,10]):
        if 'close' in df.columns:
            for lag in [1, 2, 3, 5, 10]:
                if f'lagged_close_{lag}' not in DISABLED_FEATURES_SET:
                    df[f'lagged_close_{lag}'] = df['close'].shift(lag).fillna(df['close'].iloc[0])

    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely add features that depend on columns which may be disabled or missing.
    """
    logging.info("Adding features (price_change, high_low_range, log_volume)...")

    # 'price_change' depends on 'open' and 'close'
    if 'price_change' not in DISABLED_FEATURES_SET:
        if 'open' in df.columns and 'close' in df.columns:
            df['price_change'] = df['close'] - df['open']

    # 'high_low_range' depends on 'high' and 'low'
    if 'high_low_range' not in DISABLED_FEATURES_SET:
        if 'high' in df.columns and 'low' in df.columns:
            df['high_low_range'] = df['high'] - df['low']

    # 'log_volume' depends on 'volume'
    if 'volume' not in DISABLED_FEATURES_SET:
        if 'volume' in df.columns:
            df['log_volume'] = np.log1p(df['volume'])
        else:
            logging.info("Volume column missing, skipping log_volume feature.")
    
    return df

# -------------------------------
# 8. Enhanced Train & Predict Pipeline
# -------------------------------
def train_and_predict(df: pd.DataFrame) -> float:
    # Make sure we forcibly remove 'sentiment' from disabled if present
    if "sentiment" in DISABLED_FEATURES_SET:
        DISABLED_FEATURES_SET.remove("sentiment")

    if 'sentiment' in df.columns and df["sentiment"].dtype == object:
        try:
            df["sentiment"] = df["sentiment"].astype(float)
        except Exception as e:
            logging.error(f"Cannot convert sentiment column to float: {e}")
            return None

    df = add_features(df)
    df = compute_custom_features(df)

    possible_cols = [
        'open','high','low','close','volume','vwap',
        'price_change','high_low_range','log_volume','sentiment',
        'price_return','candle_rise','body_size','wick_to_body','macd_line',
        'rsi','momentum','roc','atr','hist_vol','obv','volume_change',
        'stoch_k','bollinger_upper','bollinger_lower',
        'lagged_close_1','lagged_close_2','lagged_close_3','lagged_close_5','lagged_close_10'
    ]
    enabled_cols = []
    for c in possible_cols:
        if c in df.columns:
            if c not in DISABLED_FEATURES_SET:
                enabled_cols.append(c)

    df['target'] = df['close'].shift(-1)
    df.dropna(inplace=True)
    if len(df) < 10:
        logging.error("Not enough rows after shift to train. Need more candles.")
        return None

    X = df[enabled_cols]
    y = df['target']
    logging.info(f"Training RandomForestRegressor with {len(X)} rows and {len(enabled_cols)} features.")
    model_rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
    model_rf.fit(X, y)

    last_row_features = df.iloc[-1][enabled_cols]
    last_row_df = pd.DataFrame([last_row_features], columns=enabled_cols)
    predicted_close = model_rf.predict(last_row_df)[0]
    return predicted_close

# -------------------------------
# 9. Trading Logic & Order Functions (Unchanged)
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
# 10. Trading Job
# -------------------------------
def _perform_trading_job(skip_data=False):
    for ticker in TICKERS:
        tf_code = timeframe_to_code(BAR_TIMEFRAME)
        csv_filename = f"{ticker}_{tf_code}.csv"

        if not skip_data:
            df = fetch_candles(ticker, bars=N_BARS, timeframe=BAR_TIMEFRAME)
            if df.empty:
                logging.error(f"[{ticker}] Data fetch returned empty DataFrame. Skipping.")
                continue

            df = add_features(df)
            df = compute_custom_features(df)

            # drop disabled columns from the final CSV
            for col in list(df.columns):
                # never drop sentiment forcibly
                if col in DISABLED_FEATURES_SET and col != "sentiment":
                    df.drop(columns=[col], inplace=True)

            try:
                df.to_csv(csv_filename, index=False)
                logging.info(f"[{ticker}] Saved candle data (with advanced features minus disabled) to {csv_filename}")
            except Exception as e:
                logging.error(f"[{ticker}] Error saving CSV: {e}")
                continue

            if NEWS_MODE == "on":
                news_num_days = NUM_DAYS_MAPPING.get(BAR_TIMEFRAME, 1650)
                articles_per_day = ARTICLES_PER_DAY_MAPPING.get(BAR_TIMEFRAME, 1)
                news_list = fetch_news_sentiments(ticker, news_num_days, articles_per_day)
                save_news_to_csv(ticker, news_list)
                sentiments = assign_sentiment_to_candles(df, news_list)
                df["sentiment"] = sentiments
                df["sentiment"] = df["sentiment"].apply(lambda x: f"{x:.15f}")
                if "sentiment" in DISABLED_FEATURES_SET:
                    # never actually drop sentiment
                    pass
                df.to_csv(csv_filename, index=False)
                logging.info(f"[{ticker}] Updated candle CSV with sentiment & features.")
            else:
                logging.info(f"[{ticker}] NEWS_MODE=off, skipping sentiment assignment.")
        else:
            logging.info(f"[{ticker}] skip_data=True. Using existing CSV {csv_filename} for trade job.")
            if not os.path.exists(csv_filename):
                logging.error(f"[{ticker}] CSV file {csv_filename} does not exist. Cannot trade.")
                continue
            df = pd.read_csv(csv_filename)
            if df.empty:
                logging.error(f"[{ticker}] Existing CSV is empty. Skipping.")
                continue

        pred_close = train_and_predict(df)
        if pred_close is None:
            logging.error(f"[{ticker}] Model training or prediction failed. Skipping trade logic.")
            continue
        current_price = float(df.iloc[-1]['close'])
        logging.info(f"[{ticker}] Current Price = {current_price:.2f}, Predicted Next Close = {pred_close:.2f}")
        trade_logic(current_price, pred_close, ticker)

# -------------------------------
# 11. Scheduling
# -------------------------------
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
        if NEWS_MODE == "on":
            try:
                base_time = datetime.strptime(t, "%H:%M")
                sentiment_time = (base_time - timedelta(minutes=15)).strftime("%H:%M")
                schedule.every().day.at(sentiment_time).do(run_sentiment_job)
                logging.info(f"Scheduled sentiment update at {sentiment_time} NY time and trading at {t} (NEWS_MODE=on).")
            except Exception as e:
                logging.error(f"Error scheduling sentiment job: {e}")
        else:
            logging.info(f"NEWS_MODE={NEWS_MODE}. No sentiment scheduling before {t}.")
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
    _perform_trading_job(skip_data=False)
    logging.info("Scheduled trading job finished.")

# -------------------------------
# 12. .env Updating Logic
# -------------------------------
def update_env_variable(key: str, value: str):
    env_path = ".env"
    lines = []
    found_key = False

    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = f.readlines()

    new_key_val = f"{key}={value}\n"
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = new_key_val
            found_key = True
            break

    if not found_key:
        lines.append(new_key_val)

    with open(env_path, "w") as f:
        f.writelines(lines)

    os.environ[key] = value
    logging.info(f"Updated .env: {key}={value}")

# -------------------------------
# 13. Backtest with Trading Simulation
# -------------------------------
def simulate_backtest_trades(ticker, df, initial_balance=100000.0):
    """
    We do a candle-by-candle backtest:
     1) For each i in [0..len(df)-1], we train a model on [0..i-1], predict row i,
        decide to go long/short/none, place or hold position, etc.
     2) We track trades in backtest_trades_{ticker}.csv
     3) We track a daily equity curve in backtest_equity_{ticker}.csv
    We'll do a simple approach with 1 share trades for demonstration.
    """
    if 'sentiment' in df.columns and df["sentiment"].dtype == object:
        try:
            df["sentiment"] = df["sentiment"].astype(float)
        except:
            pass

    # We'll store a list of dicts for trades, plus portfolio snapshots
    trades = []
    portfolio_curve = []

    # position can be: 'long', 'short', or 'none'
    position = 'none'
    shares_held = 0
    balance = initial_balance
    entry_price = 0.0

    # We'll define a function to compute total equity
    def current_equity(price):
        if position == 'none':
            return balance
        elif position == 'long':
            return balance + (price - entry_price) * shares_held
        elif position == 'short':
            return balance + (entry_price - price) * shares_held

    # We'll walk candle by candle
    for i in range(len(df) - 1):
        # Train on data [0..i-1]
        train_df = df.iloc[:i].copy()
        if len(train_df) < 10:
            # Not enough data to train
            continue

        # Meanwhile, we want to do next candle's open or close as the "current price"
        current_price = df.iloc[i]['close']  # or open, your choice

        # Create a subset that doesn't include row i
        train_df['target'] = train_df['close'].shift(-1)
        train_df.dropna(inplace=True)
        if len(train_df) < 10:
            continue

        # gather columns that remain (skip disabled)
        possible_cols = [
            'open','high','low','close','volume','vwap',
            'price_change','high_low_range','log_volume','sentiment',
            'price_return','candle_rise','body_size','wick_to_body','macd_line',
            'rsi','momentum','roc','atr','hist_vol','obv','volume_change',
            'stoch_k','bollinger_upper','bollinger_lower',
            'lagged_close_1','lagged_close_2','lagged_close_3','lagged_close_5','lagged_close_10'
        ]
        enabled_cols = [c for c in possible_cols if c in train_df.columns and c not in DISABLED_FEATURES_SET]

        X = train_df[enabled_cols]
        y = train_df['target']

        if len(X) < 10:
            continue

        # Train model
        model_rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
        model_rf.fit(X, y)

        # Now predict row i's next close
        row_i = df.iloc[i][enabled_cols]
        row_i_df = pd.DataFrame([row_i], columns=enabled_cols)
        pred_close = model_rf.predict(row_i_df)[0]

        # Our trade logic:
        # if pred_close > current_price => we want to go long
        # if pred_close < current_price => we want to go short
        if pred_close > current_price:
            # want to be long
            if position == 'long':
                # do nothing
                pass
            elif position == 'short':
                # close short
                pl = (entry_price - current_price) * shares_held
                balance += pl
                trades.append({
                    'timestamp': str(df.iloc[i]['timestamp']),
                    'action': 'COVER',
                    'price': current_price,
                    'pl': pl,
                    'balance': balance
                })
                # open new long
                shares_held = 1
                position = 'long'
                entry_price = current_price
                trades.append({
                    'timestamp': str(df.iloc[i]['timestamp']),
                    'action': 'BUY',
                    'price': current_price,
                    'pl': None,
                    'balance': balance
                })
            elif position == 'none':
                # open new long
                shares_held = 1
                position = 'long'
                entry_price = current_price
                trades.append({
                    'timestamp': str(df.iloc[i]['timestamp']),
                    'action': 'BUY',
                    'price': current_price,
                    'pl': None,
                    'balance': balance
                })

        else:
            # want to be short
            if position == 'short':
                # do nothing
                pass
            elif position == 'long':
                # close long
                pl = (current_price - entry_price) * shares_held
                balance += pl
                trades.append({
                    'timestamp': str(df.iloc[i]['timestamp']),
                    'action': 'SELL',
                    'price': current_price,
                    'pl': pl,
                    'balance': balance
                })
                # open new short
                shares_held = 1
                position = 'short'
                entry_price = current_price
                trades.append({
                    'timestamp': str(df.iloc[i]['timestamp']),
                    'action': 'SHORT',
                    'price': current_price,
                    'pl': None,
                    'balance': balance
                })
            elif position == 'none':
                # open short
                shares_held = 1
                position = 'short'
                entry_price = current_price
                trades.append({
                    'timestamp': str(df.iloc[i]['timestamp']),
                    'action': 'SHORT',
                    'price': current_price,
                    'pl': None,
                    'balance': balance
                })

        # record portfolio equity
        eq = current_equity(current_price)
        portfolio_curve.append({
            'timestamp': df.iloc[i]['timestamp'],
            'equity': eq
        })

    # at the end, we close any open position
    final_price = df.iloc[-1]['close']
    eq = 0
    if position == 'long':
        pl = (final_price - entry_price) * shares_held
        balance += pl
        eq = balance
        trades.append({
            'timestamp': str(df.iloc[-1]['timestamp']),
            'action': 'SELL (final)',
            'price': final_price,
            'pl': pl,
            'balance': balance
        })
    elif position == 'short':
        pl = (entry_price - final_price) * shares_held
        balance += pl
        eq = balance
        trades.append({
            'timestamp': str(df.iloc[-1]['timestamp']),
            'action': 'COVER (final)',
            'price': final_price,
            'pl': pl,
            'balance': balance
        })
    else:
        eq = balance

    # Save trades
    trades_df = pd.DataFrame(trades)
    trades_csv = f"backtest_trades_{ticker}.csv"
    trades_df.to_csv(trades_csv, index=False)
    logging.info(f"[{ticker}] Saved backtest trades to {trades_csv}")

    # Save portfolio
    portfolio_df = pd.DataFrame(portfolio_curve)
    # append a final row for the last candle
    portfolio_df = portfolio_df.append({
        'timestamp': df.iloc[-1]['timestamp'],
        'equity': eq
    }, ignore_index=True)

    portfolio_csv = f"backtest_equity_{ticker}.csv"
    portfolio_df.to_csv(portfolio_csv, index=False)
    logging.info(f"[{ticker}] Saved backtest equity curve to {portfolio_csv}")

    # Plot equity
    plt.figure(figsize=(10,6))
    plt.plot(portfolio_df['timestamp'], portfolio_df['equity'], label='Equity')
    plt.title(f"{ticker} Backtest Equity")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    equity_png = f"backtest_equity_{ticker}.png"
    plt.savefig(equity_png)
    plt.close()
    logging.info(f"[{ticker}] Saved backtest equity chart to {equity_png}")

    total_pl = eq - initial_balance
    logging.info(f"[{ticker}] Backtest final P/L = {total_pl:.2f} (Start={initial_balance:.2f}, End={eq:.2f})")

    return trades_df, portfolio_df

# -------------------------------
# 14. Console Commands
# -------------------------------
def console_listener():
    global SHUTDOWN, TICKERS, BAR_TIMEFRAME, N_BARS, NEWS_MODE, DISABLED_FEATURES_SET
    while not SHUTDOWN:
        cmd_line = sys.stdin.readline().strip()
        if not cmd_line:
            continue
        parts = cmd_line.split()
        cmd = parts[0].lower()
        skip_data = ('-r' in parts)

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
            if len(parts) > 1 and parts[1] != '-r':
                timeframe = parts[1]
            logging.info(f"Received 'get-data' command with timeframe={timeframe}.")
            for ticker in TICKERS:
                df = fetch_candles(ticker, bars=N_BARS, timeframe=timeframe)
                if df.empty:
                    logging.error(f"[{ticker}] No data. Could not save CSV with features.")
                    continue
                df = add_features(df)
                df = compute_custom_features(df)
                # drop disabled columns except sentiment
                for col in list(df.columns):
                    if col in DISABLED_FEATURES_SET and col != "sentiment":
                        df.drop(columns=[col], inplace=True)
                tf_code = timeframe_to_code(timeframe)
                csv_filename = f"{ticker}_{tf_code}.csv"
                df.to_csv(csv_filename, index=False)
                logging.info(f"[{ticker}] Saved candle data + advanced features (minus disabled) to {csv_filename}.")

        elif cmd == "predict-next":
            for ticker in TICKERS:
                tf_code = timeframe_to_code(BAR_TIMEFRAME)
                csv_filename = f"{ticker}_{tf_code}.csv"
                if skip_data:
                    logging.info(f"[{ticker}] predict-next -r: Using existing CSV {csv_filename}")
                    if not os.path.exists(csv_filename):
                        logging.error(f"[{ticker}] CSV does not exist, skipping.")
                        continue
                    df = pd.read_csv(csv_filename)
                    if df.empty:
                        logging.error(f"[{ticker}] CSV is empty, skipping.")
                        continue
                else:
                    df = fetch_candles(ticker, bars=N_BARS, timeframe=BAR_TIMEFRAME)
                    if df.empty:
                        logging.error(f"[{ticker}] Empty data, skipping predict-next.")
                        continue
                    df = add_features(df)
                    df = compute_custom_features(df)
                    for col in list(df.columns):
                        if col in DISABLED_FEATURES_SET and col != "sentiment":
                            df.drop(columns=[col], inplace=True)
                    df.to_csv(csv_filename, index=False)
                    logging.info(f"[{ticker}] Fetched new data + advanced features (minus disabled), saved to {csv_filename}")

                pred_close = train_and_predict(df)
                if pred_close is None:
                    logging.error(f"[{ticker}] No prediction generated.")
                    continue
                current_price = float(df.iloc[-1]['close'])
                logging.info(f"[{ticker}] Current Price={current_price:.2f}, Predicted Next Close={pred_close:.2f}")

        elif cmd == "run-sentiment":
            logging.info("Running sentiment update job...")
            run_sentiment_job(skip_data=skip_data)
            logging.info("Sentiment update job completed.")

        elif cmd == "force-run":
            logging.info("Force-running job now (ignoring market open check).")
            if not skip_data:
                run_sentiment_job(skip_data=False)
                _perform_trading_job(skip_data=False)
            else:
                logging.info("force-run -r: skipping data fetch, using existing CSV + skipping new sentiment fetch.")
                _perform_trading_job(skip_data=True)
            logging.info("Force-run job finished.")

        elif cmd == "backtest":
            """
            backtest <N> [timeframe optional] [-r optional]
             -> now uses a step-by-step approach that simulates actual trades
            """
            if len(parts) < 2:
                logging.warning("Usage: backtest <N> [timeframe optional] [-r optional]")
                continue
            test_size_str = parts[1]
            try:
                test_size = int(test_size_str)
            except ValueError:
                logging.error("Invalid test_size for 'backtest' command.")
                continue
            timeframe = BAR_TIMEFRAME
            if len(parts) > 2 and parts[2] != '-r':
                timeframe = parts[2]

            for ticker in TICKERS:
                tf_code = timeframe_to_code(timeframe)
                csv_filename = f"{ticker}_{tf_code}.csv"

                if skip_data:
                    logging.info(f"[{ticker}] backtest -r: using existing CSV {csv_filename}")
                    if not os.path.exists(csv_filename):
                        logging.error(f"[{ticker}] CSV file {csv_filename} does not exist, skipping.")
                        continue
                    df = pd.read_csv(csv_filename)
                    if df.empty:
                        logging.error(f"[{ticker}] CSV is empty, skipping.")
                        continue
                else:
                    df = fetch_candles(ticker, bars=N_BARS, timeframe=timeframe)
                    if df.empty:
                        logging.error(f"[{ticker}] No data to backtest.")
                        continue
                    # If NEWS_MODE=on, get sentiment
                    if NEWS_MODE == "on":
                        news_num_days = NUM_DAYS_MAPPING.get(timeframe, 1650)
                        articles_per_day = ARTICLES_PER_DAY_MAPPING.get(timeframe, 1)
                        news_list = fetch_news_sentiments(ticker, news_num_days, articles_per_day)
                        df["sentiment"] = assign_sentiment_to_candles(df, news_list)
                        df["sentiment"] = df["sentiment"].apply(lambda x: f"{x:.15f}")

                    df = add_features(df)
                    df = compute_custom_features(df)
                    # drop disabled except sentiment
                    for col in list(df.columns):
                        if col in DISABLED_FEATURES_SET and col != "sentiment":
                            df.drop(columns=[col], inplace=True)
                    df.to_csv(csv_filename, index=False)
                    logging.info(f"[{ticker}] Saved updated CSV with sentiment & features (minus disabled) to {csv_filename} before backtest.")

                if 'sentiment' in df.columns and df["sentiment"].dtype == object:
                    try:
                        df["sentiment"] = df["sentiment"].astype(float)
                    except Exception as e:
                        logging.error(f"[{ticker}] Could not convert sentiment to float: {e}")
                        continue

                # If user wants partial data or test_size, we do it differently. 
                # We'll do a step-by-step approach with "simulate_backtest_trades"

                # Optionally, if we want the last N rows only. But let's do a full approach.
                # We'll just pass the entire df. The step by step function will not skip the last test_size. 
                if len(df) < 15:
                    logging.error(f"[{ticker}] Not enough data to do step-by-step backtest.")
                    continue

                logging.info(f"[{ticker}] Starting step-by-step backtest with test_size={test_size}.")
                simulate_backtest_trades(ticker, df, initial_balance=100000.0)

        elif cmd == "feature-importance":
            timeframe = BAR_TIMEFRAME
            if len(parts) > 1 and parts[1] != '-r':
                timeframe = parts[1]
            for ticker in TICKERS:
                logging.info(f"[{ticker}] Running 'feature-importance' with timeframe={timeframe}, skip_data={skip_data}")
                tf_code = timeframe_to_code(timeframe)
                csv_filename = f"{ticker}_{tf_code}.csv"

                if skip_data:
                    logging.info(f"[{ticker}] feature-importance -r: using CSV {csv_filename}")
                    if not os.path.exists(csv_filename):
                        logging.error(f"[{ticker}] CSV does not exist. Cannot run feature-importance.")
                        continue
                    df = pd.read_csv(csv_filename)
                    if df.empty:
                        logging.error(f"[{ticker}] CSV is empty. Cannot run feature-importance.")
                        continue
                else:
                    df = fetch_candles(ticker, bars=N_BARS, timeframe=timeframe)
                    if df.empty:
                        logging.error(f"[{ticker}] No data for 'feature-importance'.")
                        continue
                    logging.info(f"[{ticker}] Completed candle fetch.")
                    if NEWS_MODE == "on":
                        logging.info(f"[{ticker}] Fetching news sentiments for {timeframe}.")
                        news_num_days = NUM_DAYS_MAPPING.get(timeframe, 1650)
                        articles_per_day = ARTICLES_PER_DAY_MAPPING.get(timeframe, 1)
                        news_list = fetch_news_sentiments(ticker, news_num_days, articles_per_day)
                        df["sentiment"] = assign_sentiment_to_candles(df, news_list)
                        df["sentiment"] = df["sentiment"].apply(lambda x: f"{x:.15f}")
                    df = add_features(df)
                    df = compute_custom_features(df)
                    for col in list(df.columns):
                        if col in DISABLED_FEATURES_SET and col != "sentiment":
                            df.drop(columns=[col], inplace=True)
                    df.to_csv(csv_filename, index=False)
                    logging.info(f"[{ticker}] Saved updated CSV with data & features (minus disabled) for feature-importance.")

                if 'sentiment' in df.columns and df["sentiment"].dtype == object:
                    try:
                        df["sentiment"] = df["sentiment"].astype(float)
                    except Exception as e:
                        logging.error(f"[{ticker}] Could not convert sentiment to float: {e}")
                        continue

                # re-add in memory for training
                df = add_features(df)
                df = compute_custom_features(df)
                for col in list(df.columns):
                    if col in DISABLED_FEATURES_SET and col != "sentiment":
                        df.drop(columns=[col], inplace=True)

                df['target'] = df['close'].shift(-1)
                df.dropna(inplace=True)
                if len(df) < 10:
                    logging.error(f"[{ticker}] Not enough data to train for feature importance.")
                    continue

                possible_cols = [
                    'open','high','low','close','volume','vwap',
                    'price_change','high_low_range','log_volume','sentiment',
                    'price_return','candle_rise','body_size','wick_to_body','macd_line',
                    'rsi','momentum','roc','atr','hist_vol','obv','volume_change',
                    'stoch_k','bollinger_upper','bollinger_lower',
                    'lagged_close_1','lagged_close_2','lagged_close_3','lagged_close_5','lagged_close_10'
                ]
                available_cols = [c for c in possible_cols if c in df.columns]

                X = df[available_cols]
                y = df['target']

                logging.info(f"[{ticker}] Training RandomForestRegressor to compute feature importance. Rows={len(X)}, Features={len(available_cols)}")
                try:
                    model_rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
                    model_rf.fit(X, y)
                    importances = model_rf.feature_importances_
                    logging.info(f"[{ticker}] Feature Importances (timeframe={timeframe}):")
                    for feature_name, importance in zip(available_cols, importances):
                        logging.info(f"   {feature_name}: {importance:.4f}")
                except Exception as e:
                    logging.error(f"[{ticker}] Could not compute feature importance: {e}")

        elif cmd == "commands":
            logging.info("Available commands:")
            logging.info("  turnoff")
            logging.info("  api-test")
            logging.info("  get-data [timeframe]")
            logging.info("  predict-next [-r]")
            logging.info("  run-sentiment [-r]")
            logging.info("  force-run [-r]")
            logging.info("  backtest <N> [timeframe optional] [-r]")
            logging.info("  feature-importance [timeframe optional] [-r]")
            logging.info("  set-tickers (tickers)")
            logging.info("  set-timeframe (timeframe)")
            logging.info("  set-nbars (Number of candles)")
            logging.info("  set-news (on/off)")
            logging.info("  disable-feature <comma-separated features>")
            logging.info("  auto-feature")
            logging.info("  commands")

        # -------------------------------
        # Extended "disable-feature" logic
        # -------------------------------
        elif cmd == "disable-feature":
            """
            If user sets "disable-feature main", only the following remain enabled:
              open, high, low, close, vwap,
              lagged_close_1, lagged_close_2, lagged_close_3, lagged_close_5, lagged_close_10,
              bollinger_upper, bollinger_lower,
              momentum, obv,
              sentiment (always forced)

            If user sets "disable-feature base", only:
              timestamp, open, high, low, close, volume, vwap, transactions,
              sentiment

            Otherwise, we do the normal approach
            """
            if len(parts) < 2:
                logging.warning("Usage: disable-feature <comma-separated features> OR disable-feature main/base")
                continue

            new_disabled = parts[1].strip().lower()
            if new_disabled == "main":
                # enable only main set
                # that means we disable everything except the main set + sentiment
                main_set = {
                    "timestamp","open","high","low","close","vwap",
                    "lagged_close_1","lagged_close_2","lagged_close_3","lagged_close_5","lagged_close_10",
                    "bollinger_upper","bollinger_lower",
                    "momentum","obv",
                    "sentiment"
                }
                # We'll gather the full possible and remove from main_set
                all_possible = {
                    "timestamp","open","high","low","close","volume","vwap","transactions","sentiment",
                    "price_change","high_low_range","log_volume","price_return","candle_rise","body_size",
                    "wick_to_body","macd_line","rsi","momentum","roc","atr","hist_vol","obv","volume_change",
                    "stoch_k","bollinger_upper","bollinger_lower",
                    "lagged_close_1","lagged_close_2","lagged_close_3","lagged_close_5","lagged_close_10"
                }
                # everything not in main_set is disabled
                final_disabled = all_possible - main_set
                # remove 'sentiment' if present
                if "sentiment" in final_disabled:
                    final_disabled.remove("sentiment")
                new_disabled_str = ",".join(sorted(final_disabled))
                update_env_variable("DISABLED_FEATURES", new_disabled_str)
                # also update in memory
                new_set = set(final_disabled)
                if "sentiment" in new_set:
                    new_set.remove("sentiment")
                DISABLED_FEATURES_SET = new_set
                logging.info(f"disable-feature main => DISABLED_FEATURES={new_disabled_str}")
                continue
            elif new_disabled == "base":
                # enable only base set
                # base set is {timestamp, open, high, low, close, volume, vwap, transactions, sentiment}
                base_set = {
                    "timestamp","open","high","low","close","volume","vwap","transactions","sentiment"
                }
                all_possible = {
                    "timestamp","open","high","low","close","volume","vwap","transactions","sentiment",
                    "price_change","high_low_range","log_volume","price_return","candle_rise","body_size",
                    "wick_to_body","macd_line","rsi","momentum","roc","atr","hist_vol","obv","volume_change",
                    "stoch_k","bollinger_upper","bollinger_lower",
                    "lagged_close_1","lagged_close_2","lagged_close_3","lagged_close_5","lagged_close_10"
                }
                final_disabled = all_possible - base_set
                if "sentiment" in final_disabled:
                    final_disabled.remove("sentiment")
                new_disabled_str = ",".join(sorted(final_disabled))
                update_env_variable("DISABLED_FEATURES", new_disabled_str)
                new_set = set(final_disabled)
                if "sentiment" in new_set:
                    new_set.remove("sentiment")
                DISABLED_FEATURES_SET = new_set
                logging.info(f"disable-feature base => DISABLED_FEATURES={new_disabled_str}")
                continue
            else:
                # normal approach
                # parse user input
                # forcibly remove "sentiment" if present
                features_list = [f.strip() for f in new_disabled.split(",") if f.strip()]
                if "sentiment" in features_list:
                    features_list.remove("sentiment")
                # turn into string
                new_disabled_str = ",".join(features_list)
                update_env_variable("DISABLED_FEATURES", new_disabled_str)
                if new_disabled_str.strip():
                    new_set = set(features_list)
                else:
                    new_set = set()
                # again remove sentiment if user tried
                if "sentiment" in new_set:
                    new_set.remove("sentiment")
                DISABLED_FEATURES_SET = new_set
                logging.info(f"Updated DISABLED_FEATURES in memory to {DISABLED_FEATURES_SET}")

        elif cmd == "auto-feature":
            threshold = 0.01
            logging.info("Running auto-feature with threshold=%.2f." % threshold)
            combined_importances = {}
            combined_counts = {}
            for ticker in TICKERS:
                tf_code = timeframe_to_code(BAR_TIMEFRAME)
                csv_filename = f"{ticker}_{tf_code}.csv"
                if not os.path.exists(csv_filename):
                    logging.warning(f"[{ticker}] CSV not found. Skipping auto-feature on this ticker.")
                    continue
                df = pd.read_csv(csv_filename)
                if df.empty:
                    logging.warning(f"[{ticker}] CSV empty. Skipping.")
                    continue
                if 'sentiment' in df.columns and df["sentiment"].dtype == object:
                    try:
                        df["sentiment"] = df["sentiment"].astype(float)
                    except:
                        pass
                # re-add features in memory
                df = add_features(df)
                df = compute_custom_features(df)
                # do not drop from memory yet if we want to see all columns
                # we do want to skip them if they're not in the CSV though
                df['target'] = df['close'].shift(-1)
                df.dropna(inplace=True)
                if len(df) < 10:
                    logging.warning(f"[{ticker}] Not enough rows after shift.")
                    continue

                possible_cols = {
                    "timestamp","open","high","low","close","volume","vwap","transactions","sentiment",
                    "price_change","high_low_range","log_volume","price_return","candle_rise","body_size",
                    "wick_to_body","macd_line","rsi","momentum","roc","atr","hist_vol","obv","volume_change",
                    "stoch_k","bollinger_upper","bollinger_lower",
                    "lagged_close_1","lagged_close_2","lagged_close_3","lagged_close_5","lagged_close_10"
                }
                # see which ones exist in df
                available_cols = [c for c in possible_cols if c in df.columns]

                X = df[available_cols]
                y = df['target']
                if len(X) < 10:
                    logging.warning(f"[{ticker}] Not enough data to compute feature importances.")
                    continue
                try:
                    model_rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
                    model_rf.fit(X, y)
                    importances = model_rf.feature_importances_
                    logging.info(f"[{ticker}] Feature importances from auto-feature run:")
                    for fcol, fimp in zip(available_cols, importances):
                        logging.info(f"   {fcol}: {fimp:.4f}")

                    for col, imp in zip(available_cols, importances):
                        combined_importances[col] = combined_importances.get(col, 0.0) + imp
                        combined_counts[col] = combined_counts.get(col, 0) + 1
                except Exception as e:
                    logging.error(f"[{ticker}] auto-feature error: {e}")

            if not combined_importances:
                logging.info("No features or tickers processed in auto-feature. Possibly no data.")
                continue

            # average
            averaged = {}
            for col, tot in combined_importances.items():
                avg_imp = tot / combined_counts[col]
                averaged[col] = avg_imp

            logging.info("Final average feature importances across tickers:")
            for col in sorted(averaged.keys()):
                logging.info(f"  {col}: {averaged[col]:.4f}")

            to_disable = []
            for col, imp in averaged.items():
                if col == "sentiment":
                    continue  # never disable sentiment
                if imp < threshold:
                    to_disable.append(col)

            if not to_disable:
                logging.info("No features found below threshold. Doing nothing.")
                continue

            new_disabled_str = ",".join(to_disable)
            logging.info(f"auto-feature: disabling features below threshold: {new_disabled_str}")
            update_env_variable("DISABLED_FEATURES", new_disabled_str)
            new_set = set(to_disable)
            if "sentiment" in new_set:
                new_set.remove("sentiment")
            DISABLED_FEATURES_SET = new_set
            logging.info(f"Updated DISABLED_FEATURES in memory to {DISABLED_FEATURES_SET}")

        else:
            logging.warning(f"Unrecognized command: {cmd_line}")

def main():
    setup_schedule_for_timeframe(BAR_TIMEFRAME)
    listener_thread = threading.Thread(target=console_listener, daemon=True)
    listener_thread.start()
    logging.info("Bot started. Running schedule in local NY time.")
    logging.info("Commands: turnoff, api-test, get-data [timeframe], predict-next [-r], run-sentiment [-r], force-run [-r], backtest <N> [timeframe] [-r], feature-importance [timeframe] [-r], commands")
    logging.info("Also: set-tickers, set-timeframe, set-nbars, set-news (on/off), disable-feature <list>, auto-feature")
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
