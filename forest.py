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

# We define special sets for "main" and "base". 
# If DISABLED_FEATURES == "main" or "base", we'll do a special keep-only logic.
MAIN_FEATURES = {
    "timestamp","open","high","low","close","vwap",
    "lagged_close_1","lagged_close_2","lagged_close_3","lagged_close_5","lagged_close_10",
    "bollinger_upper","bollinger_lower","momentum","obv"
}
BASE_FEATURES = {
    "timestamp","open","high","low","close","volume","vwap","transactions"
}

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
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds some basic features, but checks for column existence to avoid KeyErrors.
    """
    logging.info("Adding features (price_change, high_low_range, log_volume)...")
    # Only compute if columns exist:
    if 'close' in df.columns and 'open' in df.columns:
        df['price_change'] = df['close'] - df['open']
    if 'high' in df.columns and 'low' in df.columns:
        df['high_low_range'] = df['high'] - df['low']
    if 'volume' in df.columns:
        df['log_volume'] = np.log1p(df['volume'])
    return df

def compute_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a variety of technical indicators and features.
    Checks for needed columns before computing each, to avoid KeyErrors.
    """
    # price_return
    if 'close' in df.columns:
        df['price_return'] = df['close'].pct_change().fillna(0)
    # candle_rise
    if all(x in df.columns for x in ['high','low']):
        df['candle_rise'] = df['high'] - df['low']
    # body_size
    if all(x in df.columns for x in ['close','open']):
        df['body_size'] = df['close'] - df['open']
        body = (df['close'] - df['open']).replace(0, np.nan)
        if all(x in df.columns for x in ['high','low']):
            df['wick_to_body'] = ((df['high'] - df['low']) / body).replace(np.nan, 0)
    # macd_line
    if 'close' in df.columns:
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        df['macd_line'] = (macd - signal)
    # rsi
    if 'close' in df.columns:
        window = 14
        delta = df['close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=window-1, adjust=False).mean()
        ema_down = down.ewm(com=window-1, adjust=False).mean()
        rs = ema_up / ema_down.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)
    # momentum
    if 'close' in df.columns:
        df['momentum'] = df['close'] - df['close'].shift(14)
        df['momentum'] = df['momentum'].fillna(0)
    # roc
    if 'close' in df.columns:
        shifted = df['close'].shift(14)
        df['roc'] = ((df['close'] - shifted) / shifted.replace(0, np.nan)) * 100
        df['roc'] = df['roc'].fillna(0)
    # atr
    if all(x in df.columns for x in ['high','low','close']):
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        tr = high_low.combine(high_close, max).combine(low_close, max)
        df['atr'] = tr.ewm(span=14, adjust=False).mean().fillna(0)
    # hist_vol
    if 'close' in df.columns:
        ret = df['close'].pct_change()
        rolling_std = ret.rolling(window=14).std()
        df['hist_vol'] = rolling_std * np.sqrt(14)
        df['hist_vol'] = df['hist_vol'].fillna(0)
    # obv
    if 'close' in df.columns and 'volume' in df.columns:
        df['obv'] = 0
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                df.loc[i, 'obv'] = df.loc[i-1, 'obv'] + df.loc[i, 'volume']
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                df.loc[i, 'obv'] = df.loc[i-1, 'obv'] - df.loc[i, 'volume']
            else:
                df.loc[i, 'obv'] = df.loc[i-1, 'obv']
    # volume_change
    if 'volume' in df.columns:
        df['volume_change'] = df['volume'].pct_change().fillna(0)
    # stoch_k
    if all(x in df.columns for x in ['close','high','low']):
        low14 = df['low'].rolling(window=14).min()
        high14 = df['high'].rolling(window=14).max()
        stoch_k = (df['close'] - low14) / (high14 - low14.replace(0, np.nan)) * 100
        df['stoch_k'] = stoch_k.fillna(50)
    # bollinger
    if 'close' in df.columns:
        ma20 = df['close'].rolling(window=20).mean()
        std20 = df['close'].rolling(window=20).std()
        df['bollinger_upper'] = ma20 + 2*std20
        df['bollinger_lower'] = ma20 - 2*std20
    # lagged_close_*
    if 'close' in df.columns:
        for lag in [1,2,3,5,10]:
            df[f'lagged_close_{lag}'] = df['close'].shift(lag).fillna(df['close'].iloc[0])
    return df

# A helper to apply the "DISABLED_FEATURES" logic,
# including the special "main" or "base" modes,
# and ensuring 'sentiment' is never disabled.
def drop_disabled_features(df: pd.DataFrame) -> pd.DataFrame:
    global DISABLED_FEATURES, DISABLED_FEATURES_SET
    # If the mode is "main" or "base", we keep only certain columns + "sentiment".
    if DISABLED_FEATURES == "main":
        keep_set = MAIN_FEATURES.union({"sentiment"})
        # Only keep columns in keep_set (if they exist in df)
        keep_cols = [c for c in df.columns if c in keep_set]
        return df[keep_cols]
    elif DISABLED_FEATURES == "base":
        keep_set = BASE_FEATURES.union({"sentiment"})
        keep_cols = [c for c in df.columns if c in keep_set]
        return df[keep_cols]
    else:
        # Normal custom approach: remove columns if they are in DISABLED_FEATURES_SET, except 'sentiment'
        temp = {c for c in DISABLED_FEATURES_SET if c.lower() != 'sentiment'}
        final_cols = [c for c in df.columns if c not in temp]
        return df[final_cols]

# -------------------------------
# 8. Enhanced Train & Predict Pipeline
# -------------------------------
def train_and_predict(df: pd.DataFrame) -> float:
    # Ensure sentiment is float if it exists
    if 'sentiment' in df.columns and df["sentiment"].dtype == object:
        try:
            df["sentiment"] = df["sentiment"].astype(float)
        except Exception as e:
            logging.error(f"Cannot convert sentiment column to float: {e}")
            return None

    # The script previously calls add_features and compute_custom_features, 
    # so we typically do it outside. But let's keep the calls here to maintain consistency
    df = add_features(df)
    df = compute_custom_features(df)
    # Drop disabled features
    df = drop_disabled_features(df)

    possible_cols = [
        'open','high','low','close','volume','vwap',
        'price_change','high_low_range','log_volume','sentiment',
        'price_return','candle_rise','body_size','wick_to_body','macd_line',
        'rsi','momentum','roc','atr','hist_vol','obv','volume_change',
        'stoch_k','bollinger_upper','bollinger_lower',
        'lagged_close_1','lagged_close_2','lagged_close_3','lagged_close_5','lagged_close_10'
    ]
    # Only use columns that actually exist
    available_cols = [c for c in possible_cols if c in df.columns]

    # We want to predict the next close => create target
    if 'close' not in df.columns:
        logging.error("No 'close' in DataFrame, cannot create target.")
        return None
    df['target'] = df['close'].shift(-1)
    df.dropna(inplace=True)
    if len(df) < 10:
        logging.error("Not enough rows after shift to train. Need more candles.")
        return None

    X = df[available_cols]
    y = df['target']
    logging.info(f"Training RandomForestRegressor with {len(X)} rows and {len(available_cols)} features (others are disabled).")
    model_rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
    model_rf.fit(X, y)

    last_row_features = df.iloc[-1][available_cols]
    last_row_df = pd.DataFrame([last_row_features], columns=available_cols)
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
# 10. Trading Job (with integrated sentiment update)
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

            # Generate features
            df = add_features(df)
            df = compute_custom_features(df)
            # Now drop disabled
            df = drop_disabled_features(df)

            try:
                df.to_csv(csv_filename, index=False)
                logging.info(f"[{ticker}] Saved candle data (w/out disabled) to {csv_filename}")
            except Exception as e:
                logging.error(f"[{ticker}] Error saving CSV: {e}")
                continue

            if NEWS_MODE == "on":
                news_num_days = NUM_DAYS_MAPPING.get(BAR_TIMEFRAME, 1650)
                articles_per_day = ARTICLES_PER_DAY_MAPPING.get(BAR_TIMEFRAME, 1)
                news_list = fetch_news_sentiments(ticker, news_num_days, articles_per_day)
                save_news_to_csv(ticker, news_list)
                sentiments = assign_sentiment_to_candles(df, news_list)
                # Add sentiment
                df["sentiment"] = sentiments
                df["sentiment"] = df["sentiment"].apply(lambda x: f"{x:.15f}")
                # If sentiment is not allowed to be disabled, we keep it anyway
                # but if "sentiment" was in the user-limited set, do not drop it.
                # We'll just re-drop disabled (which won't remove sentiment).
                df = drop_disabled_features(df)

                df.to_csv(csv_filename, index=False)
                logging.info(f"[{ticker}] Updated candle CSV with sentiment & features, minus disabled.")
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

        # Model train & predict
        pred_close = train_and_predict(df)
        if pred_close is None:
            logging.error(f"[{ticker}] Model training or prediction failed. Skipping trade logic.")
            continue
        current_price = float(df.iloc[-1]['close'])
        logging.info(f"[{ticker}] Current Price = {current_price:.2f}, Predicted Next Close = {pred_close:.2f}")
        trade_logic(current_price, pred_close, ticker)

# -------------------------------
# 11. Scheduling & Console Listener
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
# 13. Additional Commands
# -------------------------------
def console_listener():
    global SHUTDOWN, TICKERS, BAR_TIMEFRAME, N_BARS, NEWS_MODE, DISABLED_FEATURES, DISABLED_FEATURES_SET
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
                df = drop_disabled_features(df)

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
                    df = drop_disabled_features(df)

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
            # Usage:
            #   backtest <N> [simple|complex] [timeframe optional] [-r optional]
            # Example:
            #   backtest 500 simple -r
            #   backtest 500 complex 1Hour
            if len(parts) < 2:
                logging.warning("Usage: backtest <N> [simple|complex] [timeframe?] [-r?]")
                continue

            test_size_str = parts[1]
            approach = "simple"  # default
            timeframe = BAR_TIMEFRAME
            # next parts might be 'simple' or 'complex' or a timeframe
            # parse them in order
            possible_approaches = ["simple", "complex"]
            idx = 2
            while idx < len(parts):
                val = parts[idx]
                if val in possible_approaches:
                    approach = val
                elif val != "-r":  # might be timeframe
                    timeframe = val
                idx += 1

            try:
                test_size = int(test_size_str)
            except ValueError:
                logging.error("Invalid test_size for 'backtest' command.")
                continue

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
                    logging.info(f"[{ticker}] Fetching news for backtest sentiment (timeframe={timeframe})...")
                    if NEWS_MODE == "on":
                        news_num_days = NUM_DAYS_MAPPING.get(timeframe, 1650)
                        articles_per_day = ARTICLES_PER_DAY_MAPPING.get(timeframe, 1)
                        news_list = fetch_news_sentiments(ticker, news_num_days, articles_per_day)
                        save_news_to_csv(ticker, news_list)
                        sentiments = assign_sentiment_to_candles(df, news_list)
                        df["sentiment"] = sentiments
                        df["sentiment"] = df["sentiment"].apply(lambda x: f"{x:.15f}")

                    df = add_features(df)
                    df = compute_custom_features(df)
                    df = drop_disabled_features(df)

                    df.to_csv(csv_filename, index=False)
                    logging.info(f"[{ticker}] Saved updated CSV with sentiment & features (minus disabled) to {csv_filename} before backtest.")

                # Convert sentiment if needed
                if 'sentiment' in df.columns and df["sentiment"].dtype == object:
                    try:
                        df["sentiment"] = df["sentiment"].astype(float)
                    except Exception as e:
                        logging.error(f"[{ticker}] Could not convert sentiment to float: {e}")
                        continue

                # We do another pass of add_features + compute_custom_features 
                # to mirror the typical pipeline. Then drop disabled again:
                df = add_features(df)
                df = compute_custom_features(df)
                df = drop_disabled_features(df)

                # Make sure we can do the target shift
                if 'close' not in df.columns:
                    logging.error(f"[{ticker}] No 'close' column after feature processing. Cannot backtest.")
                    continue
                df['target'] = df['close'].shift(-1)
                df.dropna(inplace=True)
                if len(df) <= test_size + 1:
                    logging.error(f"[{ticker}] Not enough rows for backtest split. Need more data than test_size.")
                    continue

                # We define the train_end as len(df) - test_size
                # so the last test_size rows are our "test set" in either approach.
                total_len = len(df)
                train_end = total_len - test_size
                if train_end < 1:
                    logging.error(f"[{ticker}] train_end < 1. Not enough data for that test_size.")
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

                # Prepare arrays for storing predictions
                predictions = []
                actuals = []
                timestamps = []

                # We'll also do a local "backtest trading simulation"
                # Start with $10,000, no position
                start_balance = 10000.0
                cash = start_balance
                position_qty = 0.0
                position_type = "none"  # "none", "long", or "short"
                avg_entry_price = 0.0

                # We'll keep logs of trades
                trade_records = []
                # We'll keep a record of portfolio value after each candle
                portfolio_records = []

                def record_trade(action, tstamp, shares, curr_price, pred_price, pl):
                    trade_records.append({
                        "timestamp": tstamp,
                        "action": action,
                        "shares": shares,
                        "current_price": curr_price,
                        "predicted_price": pred_price,
                        "profit_loss": pl
                    })

                def get_portfolio_value(p_type, p_qty, csh, curr_price):
                    """
                    If long: total = cash + (current_price - avg_entry_price) * p_qty
                    If short: total = cash + (avg_entry_price - current_price) * p_qty
                    """
                    if p_type == "none":
                        return csh
                    elif p_type == "long":
                        return csh + (curr_price - avg_entry_price) * p_qty
                    elif p_type == "short":
                        return csh + (avg_entry_price - curr_price) * p_qty
                    return csh

                def backtest_trade_logic(c_price, p_price, row_time):
                    nonlocal cash, position_qty, position_type, avg_entry_price

                    # if predicted_price > current_price => go long
                    # if predicted_price < current_price => go short
                    # close old position if it differs from new direction
                    shares_traded = 0
                    pl = None

                    if p_price > c_price:
                        # should be long
                        if position_type == "long":
                            # do nothing
                            pass
                        elif position_type == "short":
                            # close short
                            pl = (avg_entry_price - c_price) * position_qty
                            cash += pl
                            record_trade("COVER", row_time, position_qty, c_price, p_price, pl)
                            position_qty = 0.0
                            position_type = "none"
                            avg_entry_price = 0.0

                            # open long
                            shares_to_buy = int(cash // c_price)
                            if shares_to_buy > 0:
                                position_qty = shares_to_buy
                                position_type = "long"
                                avg_entry_price = c_price
                                record_trade("LONG", row_time, shares_to_buy, c_price, p_price, None)
                                cash = cash
                        else:
                            # none -> open long
                            shares_to_buy = int(cash // c_price)
                            if shares_to_buy > 0:
                                position_qty = shares_to_buy
                                position_type = "long"
                                avg_entry_price = c_price
                                record_trade("LONG", row_time, position_qty, c_price, p_price, None)
                                cash = cash

                    elif p_price < c_price:
                        # should be short
                        if position_type == "short":
                            # do nothing
                            pass
                        elif position_type == "long":
                            pl = (c_price - avg_entry_price) * position_qty
                            cash += pl
                            record_trade("SELL", row_time, position_qty, c_price, p_price, pl)
                            position_qty = 0.0
                            position_type = "none"
                            avg_entry_price = 0.0

                            # open short
                            shares_to_short = int(cash // c_price)
                            if shares_to_short > 0:
                                position_qty = shares_to_short
                                position_type = "short"
                                avg_entry_price = c_price
                                record_trade("SHORT", row_time, position_qty, c_price, p_price, None)
                                cash = cash
                        else:
                            # none -> open short
                            shares_to_short = int(cash // c_price)
                            if shares_to_short > 0:
                                position_qty = shares_to_short
                                position_type = "short"
                                avg_entry_price = c_price
                                record_trade("SHORT", row_time, position_qty, c_price, p_price, None)
                                cash = cash

                    # at the end of this candle, record portfolio
                    port_value = get_portfolio_value(position_type, position_qty, cash, c_price)
                    portfolio_records.append({
                        "timestamp": row_time,
                        "portfolio_value": port_value
                    })

                # ---------------------------
                #  "simple" approach: train once, predict for each candle in the test set
                # ---------------------------
                if approach == "simple":
                    # Train once on [0..train_end-1]
                    train_df = df.iloc[:train_end]
                    test_df  = df.iloc[train_end:]
                    if len(train_df) < 2:
                        logging.error(f"[{ticker}] Not enough training rows for simple approach.")
                        continue

                    X_train = train_df[available_cols]
                    y_train = train_df['target']
                    model_rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
                    model_rf.fit(X_train, y_train)

                    for i in range(len(test_df)):
                        row_idx = test_df.index[i]
                        row_data = df.loc[row_idx]
                        row_features = row_data[available_cols].values.reshape(1, -1)
                        pred_price = model_rf.predict(row_features)[0]
                        real_close = row_data['close']
                        timestamps.append(row_data['timestamp'])
                        predictions.append(pred_price)
                        actuals.append(real_close)

                        backtest_trade_logic(real_close, pred_price, row_data['timestamp'])

                # ---------------------------
                #  "complex" approach: 
                #   For each candle i in the last test_size, re-train on [0..i-1], then predict i
                #   Add a progress bar printing in console
                # ---------------------------
                elif approach == "complex":
                    count = total_len - train_end
                    logging.info(f"[{ticker}] Starting COMPLEX backtest. Steps to process = {count}")
                    bar_length = 20

                    for step_index, i in enumerate(range(train_end, total_len), start=1):
                        # Print a rudimentary progress bar
                        progress = int(bar_length * step_index / count)
                        bar = "#"*progress + "-"*(bar_length - progress)
                        logging.info(f"[{ticker}] complex backtest progress: Step {step_index}/{count} [{bar}]")

                        row_data = df.iloc[i]
                        # train on [0..i-1]
                        sub_train = df.iloc[:i]
                        if len(sub_train) < 2:
                            logging.warning(f"[{ticker}] Not enough data to train at iteration i={i}. Skipping.")
                            continue
                        X_sub = sub_train[available_cols]
                        y_sub = sub_train['target']
                        model_rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
                        model_rf.fit(X_sub, y_sub)

                        # Now predict row i
                        x_test = row_data[available_cols].values.reshape(1, -1)
                        pred_price = model_rf.predict(x_test)[0]
                        real_close = row_data['close']
                        timestamps.append(row_data['timestamp'])
                        predictions.append(pred_price)
                        actuals.append(real_close)

                        backtest_trade_logic(real_close, pred_price, row_data['timestamp'])
                else:
                    logging.warning(f"[{ticker}] Unknown approach={approach}. Skipping backtest.")
                    continue

                # Evaluate and produce RMSE, MAE, Accuracy
                y_pred = np.array(predictions)
                y_test = np.array(actuals)
                rmse = math.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                avg_close = y_test.mean() if len(y_test) > 0 else 1e-6
                accuracy = 100 - (mae / avg_close * 100)

                logging.info(f"[{ticker}] Backtest ({approach}): test_size={test_size}, RMSE={rmse:.4f}, MAE={mae:.4f}, Accuracy={accuracy:.2f}%")

                # Save predictions vs actual
                out_df = pd.DataFrame({
                    'timestamp': timestamps,
                    'actual_close': actuals,
                    'predicted_close': predictions
                })
                out_csv = f"backtest_predictions_{ticker}_{test_size}_{tf_code}_{approach}.csv"
                out_df.to_csv(out_csv, index=False)
                logging.info(f"[{ticker}] Saved backtest predictions to {out_csv}.")

                # Plot
                plt.figure(figsize=(10, 6))
                plt.plot(out_df['timestamp'], out_df['actual_close'], label='Actual')
                plt.plot(out_df['timestamp'], out_df['predicted_close'], label='Predicted')
                plt.title(f"{ticker} Backtest ({approach} - Last {test_size} rows) - {timeframe}")
                plt.xlabel("Timestamp")
                plt.ylabel("Close Price")
                plt.legend()
                plt.grid(True)
                if test_size > 30:
                    plt.xticks(rotation=45)
                plt.tight_layout()
                out_img = f"backtest_predictions_{ticker}_{test_size}_{tf_code}_{approach}.png"
                plt.savefig(out_img)
                plt.close()
                logging.info(f"[{ticker}] Saved backtest predictions plot to {out_img}.")

                # Save the trade log
                trade_log_df = pd.DataFrame(trade_records)
                trade_log_csv = f"backtest_trades_{ticker}_{test_size}_{tf_code}_{approach}.csv"
                if not trade_log_df.empty:
                    trade_log_df.to_csv(trade_log_csv, index=False)
                logging.info(f"[{ticker}] Saved backtest trade log to {trade_log_csv}.")

                # Save the portfolio values
                port_df = pd.DataFrame(portfolio_records)
                port_csv = f"backtest_portfolio_{ticker}_{test_size}_{tf_code}_{approach}.csv"
                if not port_df.empty:
                    port_df.to_csv(port_csv, index=False)
                logging.info(f"[{ticker}] Saved backtest portfolio records to {port_csv}.")

                # Plot the portfolio growth
                if not port_df.empty:
                    plt.figure(figsize=(10, 6))
                    plt.plot(port_df['timestamp'], port_df['portfolio_value'], label='Portfolio Value')
                    plt.title(f"{ticker} Portfolio Value Backtest ({approach})")
                    plt.xlabel("Timestamp")
                    plt.ylabel("Portfolio Value (USD)")
                    plt.legend()
                    plt.grid(True)
                    if test_size > 30:
                        plt.xticks(rotation=45)
                    plt.tight_layout()
                    out_port_img = f"backtest_portfolio_{ticker}_{test_size}_{tf_code}_{approach}.png"
                    plt.savefig(out_port_img)
                    plt.close()
                    logging.info(f"[{ticker}] Saved backtest portfolio plot to {out_port_img}.")

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
                        save_news_to_csv(ticker, news_list)
                        sentiments = assign_sentiment_to_candles(df, news_list)
                        df["sentiment"] = sentiments
                        df["sentiment"] = df["sentiment"].apply(lambda x: f"{x:.15f}")

                    df = add_features(df)
                    df = compute_custom_features(df)
                    df = drop_disabled_features(df)

                    df.to_csv(csv_filename, index=False)
                    logging.info(f"[{ticker}] Saved updated CSV with data & features (minus disabled) for feature-importance.")

                if 'sentiment' in df.columns and df["sentiment"].dtype == object:
                    try:
                        df["sentiment"] = df["sentiment"].astype(float)
                    except Exception as e:
                        logging.error(f"[{ticker}] Could not convert sentiment to float: {e}")
                        continue

                df = add_features(df)
                df = compute_custom_features(df)
                df = drop_disabled_features(df)

                if 'close' not in df.columns:
                    logging.error(f"[{ticker}] No 'close' column. Cannot run feature-importance.")
                    continue
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
            logging.info("  backtest <N> [simple|complex] [timeframe?] [-r?]")
            logging.info("  feature-importance [timeframe optional] [-r]")
            logging.info("  set-tickers (tickers)  # usage: set-tickers TSLA,APPL")
            logging.info("  set-timeframe (timeframe)  # usage: set-timeframe 4Hour")
            logging.info("  set-nbars (Number of candles)  # usage: set-nbars 5000")
            logging.info("  set-news (on/off)  # usage: set-news on OR set-news off")
            logging.info("  disable-feature <comma-separated/features or main/base>")
            logging.info("  auto-feature")
            logging.info("  commands")

        # ---- NEW COMMANDS IMPLEMENTATION ----
        elif cmd == "set-tickers":
            if len(parts) < 2:
                logging.info("Usage: set-tickers TICKER1,TICKER2,...")
                continue
            new_tick_str = parts[1]
            update_env_variable("TICKERS", new_tick_str)
            # Also update in memory
            TICKERS = [s.strip() for s in new_tick_str.split(",") if s.strip()]
            logging.info(f"Updated TICKERS in memory to {TICKERS}")

        elif cmd == "set-timeframe":
            if len(parts) < 2:
                logging.info("Usage: set-timeframe 4Hour/1Day/etc.")
                continue
            new_tf = parts[1]
            update_env_variable("BAR_TIMEFRAME", new_tf)
            BAR_TIMEFRAME = new_tf
            logging.info(f"Updated BAR_TIMEFRAME in memory to {BAR_TIMEFRAME}")

        elif cmd == "set-nbars":
            if len(parts) < 2:
                logging.info("Usage: set-nbars 5000")
                continue
            new_nbars_str = parts[1]
            try:
                new_nbars = int(new_nbars_str)
                update_env_variable("N_BARS", str(new_nbars))
                N_BARS = new_nbars
                logging.info(f"Updated N_BARS in memory to {N_BARS}")
            except Exception as e:
                logging.error(f"Cannot parse set-nbars: {e}")

        elif cmd == "set-news":
            if len(parts) < 2:
                logging.info("Usage: set-news on/off")
                continue
            new_news = parts[1].lower()
            if new_news not in ['on','off']:
                logging.warning("set-news expects 'on' or 'off'.")
                continue
            update_env_variable("NEWS_MODE", new_news)
            NEWS_MODE = new_news
            logging.info(f"Updated NEWS_MODE in memory to {NEWS_MODE}")
        # ---- END OF NEW COMMANDS ----

        elif cmd == "disable-feature":
            if len(parts) < 2:
                logging.warning("Usage: disable-feature <comma-separated features OR 'main'/'base'>")
                continue
            new_disabled = parts[1]
            update_env_variable("DISABLED_FEATURES", new_disabled)
            DISABLED_FEATURES = new_disabled
            # Rebuild the set
            if new_disabled in ["main", "base"]:
                # We'll store the actual text "main" or "base" in the env 
                # The drop_disabled_features logic will handle it specially.
                DISABLED_FEATURES_SET = set()
            else:
                if new_disabled.strip():
                    new_set = set([f.strip() for f in new_disabled.split(",") if f.strip()])
                else:
                    new_set = set()
                # but do not allow "sentiment" to be disabled:
                new_set.discard("sentiment")
                DISABLED_FEATURES_SET = new_set

            logging.info(f"Updated DISABLED_FEATURES in memory to {DISABLED_FEATURES}")
            logging.info(f"Updated DISABLED_FEATURES_SET to {DISABLED_FEATURES_SET}")

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

                # Re-add features in memory
                df = add_features(df)
                df = compute_custom_features(df)
                # skip dropping for now, because we want to see importances for everything
                if 'close' not in df.columns:
                    logging.warning(f"[{ticker}] No close column for auto-feature. Skipping.")
                    continue
                df['target'] = df['close'].shift(-1)
                df.dropna(inplace=True)
                if len(df) < 10:
                    logging.warning(f"[{ticker}] Not enough rows after shift.")
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

            averaged = {}
            for col, tot in combined_importances.items():
                avg_imp = tot / combined_counts[col]
                averaged[col] = avg_imp

            logging.info("Final average feature importances across tickers:")
            for col in sorted(averaged.keys()):
                logging.info(f"  {col}: {averaged[col]:.4f}")

            to_disable = []
            for col, imp in averaged.items():
                if imp < threshold:
                    to_disable.append(col)

            if not to_disable:
                logging.info("No features found below threshold. Doing nothing.")
                continue

            new_disabled_str = ",".join(to_disable)
            logging.info(f"auto-feature: disabling features below threshold: {new_disabled_str}")
            update_env_variable("DISABLED_FEATURES", new_disabled_str)
            DISABLED_FEATURES = new_disabled_str
            # again, remove 'sentiment' from the set
            new_set = set(to_disable)
            new_set.discard("sentiment")
            DISABLED_FEATURES_SET = new_set
            logging.info(f"Updated DISABLED_FEATURES in memory to {DISABLED_FEATURES_SET}")

        else:
            logging.warning(f"Unrecognized command: {cmd_line}")

def main():
    setup_schedule_for_timeframe(BAR_TIMEFRAME)
    listener_thread = threading.Thread(target=console_listener, daemon=True)
    listener_thread.start()
    logging.info("Bot started. Running schedule in local NY time.")
    logging.info("Commands: turnoff, api-test, get-data [timeframe], predict-next [-r], run-sentiment [-r], force-run [-r], backtest <N> [simple|complex] [timeframe?] [-r?], feature-importance [timeframe] [-r], commands")
    logging.info("Also: set-tickers, set-timeframe, set-nbars, set-news, disable-feature <list or main/base>, auto-feature")
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
