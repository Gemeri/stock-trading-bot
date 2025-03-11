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
import importlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from openai import OpenAI
import requests
import json
import re
import discord
import xgboost as xgb

# -------------------------------
# 1. Load Configuration (from .env)
# -------------------------------
load_dotenv()

ML_MODEL = os.getenv("ML_MODEL", "forest")

API_KEY = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")
API_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

DISCORD_MODE = os.getenv("DISCORD_MODE", "off").lower().strip()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")
DISCORD_USER_ID = os.getenv("DISCORD_USER_ID", "")

BING_API_KEY = os.getenv("BING_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
AI_TICKER_COUNT = int(os.getenv("AI_TICKER_COUNT", "0"))
AI_TICKERS = []

TICKERS = os.getenv("TICKERS", "TSLA").split(",")
BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "4Hour")
N_BARS = int(os.getenv("N_BARS", "5000"))
TRADE_LOG_FILENAME = "trade_log.csv"

N_ESTIMATORS = 100
RANDOM_SEED = 42
NY_TZ = pytz.timezone("America/New_York")

NEWS_MODE = os.getenv("NEWS_MODE", "on").lower().strip()

if DISCORD_MODE == "on":

    discord_client = discord.Client()

    @discord_client.event
    async def on_ready():
        logging.info(f"Discord bot logged in as {discord_client.user}")

    def run_discord_bot():
        try:
            discord_client.run(DISCORD_TOKEN)
        except Exception as e:
            logging.error(f"Discord bot failed to run: {e}")

    discord_thread = threading.Thread(target=run_discord_bot, daemon=True)
    discord_thread.start()

def get_model():
    if ML_MODEL.lower() == "xgboost":
        logging.info("Using XGBoost model as per ML_MODEL configuration.")
        return xgb.XGBRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
    else:
        logging.info("Using Random Forest model as per ML_MODEL configuration.")
        return RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)

DISABLED_FEATURES = os.getenv("DISABLED_FEATURES", "").strip()
if DISABLED_FEATURES:
    DISABLED_FEATURES_SET = set([f.strip() for f in DISABLED_FEATURES.split(",") if f.strip()])
else:
    DISABLED_FEATURES_SET = set()

TRADE_LOGIC = os.getenv("TRADE_LOGIC", "15").strip()

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

import alpaca_trade_api as tradeapi
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
# 4b. Dictionary to map TRADE_LOGIC to actual module filenames
# -------------------------------
json_file_path = os.path.join("logic", "logic_scripts.json")

with open(json_file_path, "r") as file:
    LOGIC_MODULE_MAP = json.load(file)

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
    df = pd.DataFrame()
    if hasattr(barset, 'df'):
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
        df = add_features(df)
        df = compute_custom_features(df)
        df = drop_disabled_features(df)

        news_num_days = NUM_DAYS_MAPPING.get(BAR_TIMEFRAME, 1650)
        articles_per_day = ARTICLES_PER_DAY_MAPPING.get(BAR_TIMEFRAME, 1)
        news_list = fetch_news_sentiments(ticker, news_num_days, articles_per_day)

        save_news_to_csv(ticker, news_list)

        sentiments = assign_sentiment_to_candles(df, news_list)
        df["sentiment"] = sentiments
        df["sentiment"] = df["sentiment"].apply(lambda x: f"{x:.15f}")
        df = drop_disabled_features(df)

        tf_code = timeframe_to_code(BAR_TIMEFRAME)
        csv_filename = f"{ticker}_{tf_code}.csv"
        df.to_csv(csv_filename, index=False)
        logging.info(f"[{ticker}] Updated sentiment in candle CSV: {csv_filename}")

# -------------------------------
# 7. Additional Feature Engineering
# -------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Adding features (price_change, high_low_range, log_volume)...")
    if 'close' in df.columns and 'open' in df.columns:
        df['price_change'] = df['close'] - df['open']
    if 'high' in df.columns and 'low' in df.columns:
        df['high_low_range'] = df['high'] - df['low']
    if 'volume' in df.columns:
        df['log_volume'] = np.log1p(df['volume'])
    return df

def compute_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    if 'close' in df.columns:
        df['price_return'] = df['close'].pct_change().fillna(0)
    if all(x in df.columns for x in ['high','low']):
        df['candle_rise'] = df['high'] - df['low']
    if all(x in df.columns for x in ['close','open']):
        df['body_size'] = df['close'] - df['open']
        body = (df['close'] - df['open']).replace(0, np.nan)
        if all(x in df.columns for x in ['high','low']):
            df['wick_to_body'] = ((df['high'] - df['low']) / body).replace(np.nan, 0)
    if 'close' in df.columns:
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        df['macd_line'] = (macd - signal)
    if 'close' in df.columns:
        window = 14
        delta = df['close'].diff()
        up = delta.clip(lower=0)
        down = -1*delta.clip(upper=0)
        ema_up = up.ewm(com=window-1, adjust=False).mean()
        ema_down = down.ewm(com=window-1, adjust=False).mean()
        rs = ema_up / ema_down.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)
    if 'close' in df.columns:
        df['momentum'] = df['close'] - df['close'].shift(14)
        df['momentum'] = df['momentum'].fillna(0)
    if 'close' in df.columns:
        shifted = df['close'].shift(14)
        df['roc'] = ((df['close'] - shifted) / shifted.replace(0, np.nan))*100
        df['roc'] = df['roc'].fillna(0)
    if all(x in df.columns for x in ['high','low','close']):
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        tr = high_low.combine(high_close, max).combine(low_close, max)
        df['atr'] = tr.ewm(span=14, adjust=False).mean().fillna(0)
    if 'close' in df.columns:
        ret = df['close'].pct_change()
        rolling_std = ret.rolling(window=14).std()
        df['hist_vol'] = rolling_std * np.sqrt(14)
        df['hist_vol'] = df['hist_vol'].fillna(0)
    if 'close' in df.columns and 'volume' in df.columns:
        df['obv'] = 0
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                df.loc[i, 'obv'] = df.loc[i-1, 'obv'] + df.loc[i, 'volume']
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                df.loc[i, 'obv'] = df.loc[i-1, 'obv'] - df.loc[i, 'volume']
            else:
                df.loc[i, 'obv'] = df.loc[i-1, 'obv']
    if 'volume' in df.columns:
        df['volume_change'] = df['volume'].pct_change().fillna(0)
    if all(x in df.columns for x in ['close','high','low']):
        low14 = df['low'].rolling(window=14).min()
        high14 = df['high'].rolling(window=14).max()
        stoch_k = (df['close'] - low14) / (high14 - low14.replace(0, np.nan))*100
        df['stoch_k'] = stoch_k.fillna(50)
    if 'close' in df.columns:
        ma20 = df['close'].rolling(window=20).mean()
        std20 = df['close'].rolling(window=20).std()
        df['bollinger_upper'] = ma20 + 2*std20
        df['bollinger_lower'] = ma20 - 2*std20
    if 'close' in df.columns:
        for lag in [1,2,3,5,10]:
            df[f'lagged_close_{lag}'] = df['close'].shift(lag).fillna(df['close'].iloc[0])
    return df

def drop_disabled_features(df: pd.DataFrame) -> pd.DataFrame:
    global DISABLED_FEATURES, DISABLED_FEATURES_SET
    if DISABLED_FEATURES == "main":
        keep_set = MAIN_FEATURES.union({"sentiment"})
        keep_cols = [c for c in df.columns if c in keep_set]
        return df[keep_cols]
    elif DISABLED_FEATURES == "base":
        keep_set = BASE_FEATURES.union({"sentiment"})
        keep_cols = [c for c in df.columns if c in keep_set]
        return df[keep_cols]
    else:
        temp = {c for c in DISABLED_FEATURES_SET if c.lower() != 'sentiment'}
        final_cols = [c for c in df.columns if c not in temp]
        return df[final_cols]

# -------------------------------
# 8. Enhanced Train & Predict Pipeline
# -------------------------------
def train_and_predict(df: pd.DataFrame) -> float:
    if 'sentiment' in df.columns and df["sentiment"].dtype == object:
        try:
            df["sentiment"] = df["sentiment"].astype(float)
        except Exception as e:
            logging.error(f"Cannot convert sentiment column to float: {e}")
            return None

    df = add_features(df)
    df = compute_custom_features(df)
    df = drop_disabled_features(df)

    possible_cols = [
        'open','high','low','close','volume','vwap',
        'price_change','high_low_range','log_volume','sentiment',
        'price_return','candle_rise','body_size','wick_to_body','macd_line',
        'rsi','momentum','roc','atr','hist_vol','obv','volume_change',
        'stoch_k','bollinger_upper','bollinger_lower',
        'lagged_close_1','lagged_close_2','lagged_close_3','lagged_close_5','lagged_close_10'
    ]
    available_cols = [c for c in possible_cols if c in df.columns]

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
    logging.info(f"Training model with {len(X)} rows and {len(available_cols)} features (others are disabled).")
    model = get_model()
    model.fit(X, y)

    last_row_features = df.iloc[-1][available_cols]
    last_row_df = pd.DataFrame([last_row_features], columns=available_cols)
    predicted_close = model.predict(last_row_df)[0]
    return predicted_close

###################################################################################################
# 1) New helper function to fetch AI tickers via OpenAI + Bing, inspired by your provided example.
#    This function attempts to follow the same structure of making a query, optionally using Bing
#    results, then generating a final list of stock tickers from OpenAI. It is fully automatic.
###################################################################################################

def fetch_new_ai_tickers(num_needed, exclude_tickers):
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    def bing_web_search(query):
        """Simple Bing Web Search to get some snippet info for each query."""
        search_url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
        params = {"q": query, "textDecorations": True, "textFormat": "HTML", "count": 2}
        try:
            r = requests.get(search_url, headers=headers, params=params)
            r.raise_for_status()
            data = r.json()
            found = []
            if 'webPages' in data:
                for v in data['webPages']['value']:
                    found.append({
                        'name': v['name'],
                        'url': v['url'],
                        'snippet': v['snippet'],
                        'displayUrl': v['displayUrl']
                    })
            return found
        except Exception as e:
            logging.error(f"Bing search failed: {e}")
            return []

    search_queries = [
        "best US stocks expected to rise soon",
        "top bullish stocks to watch in the US market"
    ]

    snippet_contexts = []
    for one_query in search_queries:
        results = bing_web_search(one_query)
        for item in results:
            snippet_contexts.append(f"{item['name']}: {item['snippet']}")

    joined_snippets = "\n".join(snippet_contexts)

    exclude_list_str = ", ".join(sorted(exclude_tickers)) if exclude_tickers else "None"
    prompt_text = (
        f"You are an AI that proposes exactly {num_needed} unique US stock tickers (one per line)\n"
        f"that are likely to rise soon, based on fundamental/technical analysis.\n"
        f"Use the following context from Bing if helpful:\n{joined_snippets}\n\n"
        f"Do NOT include these tickers: {exclude_list_str}\n"
        f"Output only {num_needed} lines, each line is a ticker symbol only, no extra text."
    )

    try:
        messages = [
            {"role": "system", "content": "You are a financial assistant that suggests promising tickers."},
            {"role": "user", "content": prompt_text}
        ]

        completion = openai_client.chat.completions.create(
            model="o1",
            messages=messages,
            store=True
        )

        content = completion.choices[0].message.content.strip()

    except Exception as e:
        logging.error(f"Error calling OpenAI ChatCompletion: {e}")
        return []

    lines = content.split('\n')
    candidate_tickers = []
    for ln in lines:
        tck = ln.strip().upper()
        tck = tck.replace('.', '').replace('-', '').replace(' ', '')
        if tck and tck not in exclude_tickers:
            candidate_tickers.append(tck)

    candidate_tickers = candidate_tickers[:num_needed]
    return candidate_tickers


def _ensure_ai_tickers():
    global AI_TICKERS, AI_TICKER_COUNT

    if AI_TICKER_COUNT <= 0:
        return

    still_open = set()
    try:
        positions = api.list_positions()
        for p in positions:
            if p.symbol in AI_TICKERS and abs(float(p.qty)) > 0:
                still_open.add(p.symbol)
    except Exception as e:
        logging.error(f"Error retrieving positions in _ensure_ai_tickers: {e}")

    AI_TICKERS = [t for t in AI_TICKERS if t in still_open]

    needed = AI_TICKER_COUNT - len(AI_TICKERS)
    if needed > 0:
        exclude = set(TICKERS) | set(AI_TICKERS)
        new_ai_tickers = fetch_new_ai_tickers(needed, exclude)
        for new_tck in new_ai_tickers:
            if new_tck not in AI_TICKERS:
                AI_TICKERS.append(new_tck)
        logging.info(f"Fetched {len(new_ai_tickers)} new AI tickers: {new_ai_tickers}")
    else:
        logging.info("No need to fetch new AI tickers; current AI Tickers are sufficient.")

# -------------------------------
# 9. Trading Logic & Order Functions
# -------------------------------
def send_discord_order_message(action, ticker, price, predicted_price, extra_info=""):
    message = (f"Order Action: {action}\nTicker: {ticker}\n"
               f"Price: {price:.2f}\nPredicted: {predicted_price:.2f}\n{extra_info}")
    if DISCORD_MODE == "on" and DISCORD_USER_ID:
        async def discord_send_dm():
            try:
                user = await discord_client.fetch_user(int(DISCORD_USER_ID))
                await user.send(message)
                logging.info(f"Sent Discord DM for {ticker} order: {action}")
            except Exception as e:
                logging.error(f"Discord DM failed: {e}")
        discord_client.loop.create_task(discord_send_dm())
    else:
        logging.info("Discord mode is off or DISCORD_USER_ID not set.")

def buy_shares(ticker, qty, buy_price, predicted_price):
    if qty <= 0:
        return
    try:
        if DISCORD_MODE == "on":
            send_discord_order_message("BUY", ticker, buy_price, predicted_price,
                                       extra_info="Buying shares via Discord bot.")
        else:
            account = api.get_account()
            available_cash = float(account.cash)
            base_tickers_count = len(TICKERS) if TICKERS else 0
            total_ticker_slots = base_tickers_count + AI_TICKER_COUNT
            if total_ticker_slots <= 0:
                total_ticker_slots = 1
            split_cash = available_cash / total_ticker_slots
            max_shares_for_this_ticker = int(split_cash // buy_price)
            final_qty = min(qty, max_shares_for_this_ticker)
            if final_qty <= 0:
                logging.info(f"[{ticker}] Not enough split cash to buy any shares. Skipping.")
                return
            api.submit_order(
                symbol=ticker,
                qty=final_qty,
                side='buy',
                type='market',
                time_in_force='day'
            )
            logging.info(f"[{ticker}] BUY {final_qty} at {buy_price:.2f} (Predicted: {predicted_price:.2f})")
            log_trade("BUY", ticker, final_qty, buy_price, predicted_price, None)
            if ticker not in TICKERS and ticker not in AI_TICKERS:
                AI_TICKERS.append(ticker)
    except Exception as e:
        logging.error(f"[{ticker}] Buy order failed: {e}")

def sell_shares(ticker, qty, sell_price, predicted_price):
    if qty <= 0:
        return
    try:
        if DISCORD_MODE == "on":
            send_discord_order_message("SELL", ticker, sell_price, predicted_price,
                                       extra_info="Selling shares via Discord bot.")
        else:
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
            try:
                new_pos = api.get_position(ticker)
                if float(new_pos.qty) == 0 and ticker in AI_TICKERS:
                    AI_TICKERS.remove(ticker)
                    _ensure_ai_tickers()
            except:
                if ticker in AI_TICKERS:
                    AI_TICKERS.remove(ticker)
                    _ensure_ai_tickers()
    except Exception as e:
        logging.error(f"[{ticker}] Sell order failed: {e}")

def short_shares(ticker, qty, short_price, predicted_price):
    if qty <= 0:
        return
    try:
        if DISCORD_MODE == "on":
            send_discord_order_message("SHORT", ticker, short_price, predicted_price,
                                       extra_info="Shorting shares via Discord bot.")
        else:
            account = api.get_account()
            available_cash = float(account.cash)
            base_tickers_count = len(TICKERS) if TICKERS else 0
            total_ticker_slots = base_tickers_count + AI_TICKER_COUNT
            if total_ticker_slots <= 0:
                total_ticker_slots = 1
            split_cash = available_cash / total_ticker_slots
            max_shares_for_this_ticker = int(split_cash // short_price)
            final_qty = min(qty, max_shares_for_this_ticker)
            if final_qty <= 0:
                logging.info(f"[{ticker}] Not enough split cash/margin to short any shares. Skipping.")
                return
            api.submit_order(
                symbol=ticker,
                qty=final_qty,
                side='sell',
                type='market',
                time_in_force='day'
            )
            logging.info(f"[{ticker}] SHORT {final_qty} at {short_price:.2f} (Predicted: {predicted_price:.2f})")
            log_trade("SHORT", ticker, final_qty, short_price, predicted_price, None)
            if ticker not in TICKERS and ticker not in AI_TICKERS:
                AI_TICKERS.append(ticker)
    except Exception as e:
        logging.error(f"[{ticker}] Short order failed: {e}")

def close_short(ticker, qty, cover_price):
    if qty <= 0:
        return
    try:
        if DISCORD_MODE == "on":
            send_discord_order_message("COVER", ticker, cover_price, 0,
                                       extra_info="Covering short via Discord bot.")
        else:
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
            try:
                new_pos = api.get_position(ticker)
                if float(new_pos.qty) == 0 and ticker in AI_TICKERS:
                    AI_TICKERS.remove(ticker)
                    _ensure_ai_tickers()
            except:
                if ticker in AI_TICKERS:
                    AI_TICKERS.remove(ticker)
                    _ensure_ai_tickers()
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
        "profit_loss": profit_loss,
        "trade_logic": TRADE_LOGIC
    }
    df = pd.DataFrame([row])
    if not os.path.exists(TRADE_LOG_FILENAME):
        df.to_csv(TRADE_LOG_FILENAME, index=False, mode='w')
    else:
        df.to_csv(TRADE_LOG_FILENAME, index=False, mode='a', header=False)


def _update_logic_json():
    logic_dir = "logic"
    if not os.path.isdir(logic_dir):
        os.makedirs(logic_dir)

    pattern = re.compile(r"^logic_(\d+)_(\w+)\.py$")

    scripts_map = {}
    for fname in os.listdir(logic_dir):
        match = pattern.match(fname)
        if match:
            num_str = match.group(1)
            script_base = fname[:-3]
            try:
                num_int = int(num_str)
                scripts_map[num_int] = script_base
            except ValueError:
                pass

    if not scripts_map:
        with open(os.path.join(logic_dir, "logic_scripts.json"), "w") as f:
            json.dump({}, f, indent=2)
        return

    sorted_nums = sorted(scripts_map.keys())
    final_dict = {}
    for i in sorted_nums:
        final_dict[str(i)] = scripts_map[i]

    with open(os.path.join(logic_dir, "logic_scripts.json"), "w") as f:
        json.dump(final_dict, f, indent=2)


def _get_logic_script_name(logic_id: str) -> str:
    logic_dir = "logic"
    json_path = os.path.join(logic_dir, "logic_scripts.json")
    if not os.path.isfile(json_path):
        _update_logic_json()

    if not os.path.isfile(json_path):
        return "logic_15_forecast_driven"

    with open(json_path, "r") as f:
        data = json.load(f)
        if logic_id in data:
            return data[logic_id]
        else:
            return "logic_15_forecast_driven"

# -------------------------------
# 9b. The central "trade_logic" placeholder
# -------------------------------
def trade_logic(current_price: float, predicted_price: float, ticker: str):
    """
    Dynamically reads the JSON file in logic/ to dispatch to the correct sub-script
    based on the string in TRADE_LOGIC from the .env file (e.g. "15").
    """
    try:
        import importlib
        logic_module_name = _get_logic_script_name(TRADE_LOGIC)
        module_path = f"logic.{logic_module_name}"
        logic_module = importlib.import_module(module_path)
        logic_module.run_logic(current_price, predicted_price, ticker)
    except Exception as e:
        logging.error(f"Error dispatching to trade logic '{TRADE_LOGIC}': {e}")

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
                df["sentiment"] = sentiments
                df["sentiment"] = df["sentiment"].apply(lambda x: f"{x:.15f}")
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
                sentiment_time = (base_time - timedelta(minutes=20)).strftime("%H:%M")
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
# 13. Additional Commands & Console
# -------------------------------
def console_listener():
    global SHUTDOWN, TICKERS, BAR_TIMEFRAME, N_BARS, NEWS_MODE, DISABLED_FEATURES, DISABLED_FEATURES_SET, TRADE_LOGIC, AI_TICKER_COUNT, AI_TICKERS
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
                logging.info("Alpaca API keys are valid (or we are in fake mode).")
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
            if len(parts) < 2:
                logging.warning("Usage: backtest <N> [simple|complex] [timeframe?] [-r?]")
                continue

            test_size_str = parts[1]
            approach = "simple"
            timeframe_for_backtest = BAR_TIMEFRAME
            possible_approaches = ["simple", "complex"]
            skip_data = ('-r' in parts)
            idx = 2

            while idx < len(parts):
                val = parts[idx]
                if val in possible_approaches:
                    approach = val
                elif val != '-r':
                    timeframe_for_backtest = val
                idx += 1

            try:
                test_size = int(test_size_str)
            except ValueError:
                logging.error("Invalid test_size for 'backtest' command.")
                continue

            logic_module_name = LOGIC_MODULE_MAP.get(TRADE_LOGIC, "logic_15_forecast_driven")
            logging.info("Trading logic: " + logic_module_name)
            try:
                logic_module = importlib.import_module(f"logic.{logic_module_name}")
            except Exception as e:
                logging.error(f"Unable to import logic module {logic_module_name}: {e}")
                continue

            for ticker in TICKERS:
                tf_code = timeframe_to_code(timeframe_for_backtest)
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
                    df = fetch_candles(ticker, bars=N_BARS, timeframe=timeframe_for_backtest)
                    if df.empty:
                        logging.error(f"[{ticker}] No data to backtest.")
                        continue
                    df = add_features(df)
                    df = compute_custom_features(df)
                    df = drop_disabled_features(df)
                    df.to_csv(csv_filename, index=False)
                    logging.info(f"[{ticker}] Saved updated CSV with sentiment & features (minus disabled) to {csv_filename} before backtest.")

                if 'close' not in df.columns:
                    logging.error(f"[{ticker}] No 'close' column after feature processing. Cannot backtest.")
                    continue
                df['target'] = df['close'].shift(-1)
                df.dropna(inplace=True)
                if len(df) <= test_size + 1:
                    logging.error(f"[{ticker}] Not enough rows for backtest split. Need more data than test_size.")
                    continue

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

                predictions = []
                actuals = []
                timestamps = []
                trade_records = []
                portfolio_records = []
                start_balance = 10000.0
                cash = start_balance
                position_qty = 0
                avg_entry_price = 0.0

                def record_trade(action, tstamp, shares, curr_price, pred_price, pl):
                    trade_records.append({
                        "timestamp": tstamp,
                        "action": action,
                        "shares": shares,
                        "current_price": curr_price,
                        "predicted_price": pred_price,
                        "profit_loss": pl
                    })

                def get_portfolio_value(pos_qty, csh, c_price, avg_price):
                    if pos_qty > 0:
                        return csh + (c_price - avg_price) * pos_qty
                    elif pos_qty < 0:
                        return csh + (avg_price - c_price) * abs(pos_qty)
                    else:
                        return csh

                backtest_candles = df.iloc[train_end:].copy()

                if approach == "simple":
                    train_df = df.iloc[:train_end]
                    test_df  = df.iloc[train_end:]
                    if len(train_df) < 2:
                        logging.error(f"[{ticker}] Not enough training rows for simple approach.")
                        continue

                    X_train = train_df[available_cols]
                    y_train = train_df['target']
                    model = get_model()
                    model.fit(X_train, y_train)


                    for i_ in range(len(test_df)):
                        row_idx = test_df.index[i_]
                        row_data = df.loc[row_idx]
                        x_test = row_data[available_cols].values.reshape(1, -1)
                        pred_price = model.predict(x_test)[0]
                        real_close = row_data['close']
                        timestamps.append(row_data['timestamp'])
                        predictions.append(pred_price)
                        actuals.append(real_close)

                        action = logic_module.run_backtest(
                            current_price=real_close,
                            predicted_price=pred_price,
                            position_qty=position_qty,
                            current_timestamp=row_data['timestamp'],
                            candles=test_df
                        )

                        if action == "BUY":
                            if position_qty < 0:
                                pl = (avg_entry_price - real_close) * abs(position_qty)
                                cash += pl
                                record_trade("COVER", row_data['timestamp'], abs(position_qty), real_close, pred_price, pl)
                                position_qty = 0
                                avg_entry_price = 0.0
                            shares_to_buy = int(cash // real_close)
                            if shares_to_buy > 0:
                                position_qty = shares_to_buy
                                avg_entry_price = real_close
                                record_trade("BUY", row_data['timestamp'], shares_to_buy, real_close, pred_price, None)
                        elif action == "SELL":
                            if position_qty > 0:
                                pl = (real_close - avg_entry_price) * position_qty
                                cash += pl
                                record_trade("SELL", row_data['timestamp'], position_qty, real_close, pred_price, pl)
                                position_qty = 0
                                avg_entry_price = 0.0
                        elif action == "SHORT":
                            if position_qty > 0:
                                pl = (real_close - avg_entry_price) * position_qty
                                cash += pl
                                record_trade("SELL", row_data['timestamp'], position_qty, real_close, pred_price, pl)
                                position_qty = 0
                                avg_entry_price = 0.0
                            shares_to_short = int(cash // real_close)
                            if shares_to_short > 0:
                                position_qty = -shares_to_short
                                avg_entry_price = real_close
                                record_trade("SHORT", row_data['timestamp'], shares_to_short, real_close, pred_price, None)
                        elif action == "COVER":
                            if position_qty < 0:
                                pl = (avg_entry_price - real_close) * abs(position_qty)
                                cash += pl
                                record_trade("COVER", row_data['timestamp'], abs(position_qty), real_close, pred_price, pl)
                                position_qty = 0
                                avg_entry_price = 0.0

                        val = get_portfolio_value(position_qty, cash, real_close, avg_entry_price)
                        portfolio_records.append({
                            "timestamp": row_data['timestamp'],
                            "portfolio_value": val
                        })

                elif approach == "complex":
                    count = total_len - train_end
                    logging.info(f"[{ticker}] Starting COMPLEX backtest. Steps to process = {count}")
                    bar_length = 20

                    for step_index, i_ in enumerate(range(train_end, total_len), start=1):
                        progress = int(bar_length * step_index / count)
                        bar_str = "#"*progress + "-"*(bar_length - progress)
                        logging.info(f"[{ticker}] complex backtest progress: Step {step_index}/{count} [{bar_str}]")
                        sub_train = df.iloc[:i_]
                        if len(sub_train) < 2:
                            logging.warning(f"[{ticker}] Not enough data to train at iteration {i_}. Skipping.")
                            continue

                        X_sub = sub_train[available_cols]
                        y_sub = sub_train['target']
                        model_rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
                        model_rf.fit(X_sub, y_sub)

                        row_data = df.iloc[i_]
                        pred_price = model_rf.predict(row_data[available_cols].values.reshape(1, -1))[0]
                        real_close = row_data['close']
                        timestamps.append(row_data['timestamp'])
                        predictions.append(pred_price)
                        actuals.append(real_close)

                        action = logic_module.run_backtest(
                            current_price=real_close,
                            predicted_price=pred_price,
                            position_qty=position_qty,
                            current_timestamp=row_data['timestamp'],
                            candles=backtest_candles
                        )

                        if action == "BUY":
                            if position_qty < 0:
                                pl = (avg_entry_price - real_close) * abs(position_qty)
                                cash += pl
                                record_trade("COVER", row_data['timestamp'], abs(position_qty), real_close, pred_price, pl)
                                position_qty = 0
                                avg_entry_price = 0.0
                            shares_to_buy = int(cash // real_close)
                            if shares_to_buy > 0:
                                position_qty = shares_to_buy
                                avg_entry_price = real_close
                                record_trade("BUY", row_data['timestamp'], shares_to_buy, real_close, pred_price, None)
                        elif action == "SELL":
                            if position_qty > 0:
                                pl = (real_close - avg_entry_price) * position_qty
                                cash += pl
                                record_trade("SELL", row_data['timestamp'], position_qty, real_close, pred_price, pl)
                                position_qty = 0
                                avg_entry_price = 0.0
                        elif action == "SHORT":
                            if position_qty > 0:
                                pl = (real_close - avg_entry_price) * position_qty
                                cash += pl
                                record_trade("SELL", row_data['timestamp'], position_qty, real_close, pred_price, pl)
                                position_qty = 0
                                avg_entry_price = 0.0
                            shares_to_short = int(cash // real_close)
                            if shares_to_short > 0:
                                position_qty = -shares_to_short
                                avg_entry_price = real_close
                                record_trade("SHORT", row_data['timestamp'], shares_to_short, real_close, pred_price, None)
                        elif action == "COVER":
                            if position_qty < 0:
                                pl = (avg_entry_price - real_close) * abs(position_qty)
                                cash += pl
                                record_trade("COVER", row_data['timestamp'], abs(position_qty), real_close, pred_price, pl)
                                position_qty = 0
                                avg_entry_price = 0.0

                        val = get_portfolio_value(position_qty, cash, real_close, avg_entry_price)
                        portfolio_records.append({
                            "timestamp": row_data['timestamp'],
                            "portfolio_value": val
                        })
                else:
                    logging.warning(f"[{ticker}] Unknown approach={approach}. Skipping backtest.")
                    continue
                
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

                # Plot predictions
                plt.figure(figsize=(10, 6))
                plt.plot(out_df['timestamp'], out_df['actual_close'], label='Actual')
                plt.plot(out_df['timestamp'], out_df['predicted_close'], label='Predicted')
                plt.title(f"{ticker} Backtest ({approach} - Last {test_size} rows) - {timeframe_for_backtest}")
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

                trade_log_df = pd.DataFrame(trade_records)
                trade_log_csv = f"backtest_trades_{ticker}_{test_size}_{tf_code}_{approach}.csv"
                if not trade_log_df.empty:
                    trade_log_df.to_csv(trade_log_csv, index=False)
                logging.info(f"[{ticker}] Saved backtest trade log to {trade_log_csv}.")

                port_df = pd.DataFrame(portfolio_records)
                port_csv = f"backtest_portfolio_{ticker}_{test_size}_{tf_code}_{approach}.csv"
                if not port_df.empty:
                    port_df.to_csv(port_csv, index=False)
                logging.info(f"[{ticker}] Saved backtest portfolio records to {port_csv}.")

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
            if skip_data:
                logging.info("feature-importance -r: Will use existing CSV data for each ticker.")
            else:
                logging.info("feature-importance: Will fetch fresh data for each ticker, then compute importance.")

            for ticker in TICKERS:
                tf_code = timeframe_to_code(BAR_TIMEFRAME)
                csv_filename = f"{ticker}_{tf_code}.csv"

                if not skip_data:
                    df = fetch_candles(ticker, bars=N_BARS, timeframe=BAR_TIMEFRAME)
                    if df.empty:
                        logging.error(f"[{ticker}] Unable to fetch data for feature-importance.")
                        continue
                    df = add_features(df)
                    df = compute_custom_features(df)
                    df = drop_disabled_features(df)
                    df.to_csv(csv_filename, index=False)
                    logging.info(f"[{ticker}] Fetched data & saved CSV for feature-importance.")
                else:
                    if not os.path.exists(csv_filename):
                        logging.error(f"[{ticker}] CSV file {csv_filename} not found, skipping.")
                        continue
                    df = pd.read_csv(csv_filename)
                    if df.empty:
                        logging.error(f"[{ticker}] CSV is empty, skipping.")
                        continue

                # 2) Convert 'sentiment' if present
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
                    logging.error(f"[{ticker}] No 'close' column after processing. Cannot compute importance.")
                    continue
                df['target'] = df['close'].shift(-1)
                df.dropna(inplace=True)
                if len(df) < 10:
                    logging.error(f"[{ticker}] Not enough rows after shift to train. Skipping.")
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

                model_rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
                model_rf.fit(X, y)

                importances = model_rf.feature_importances_
                feats_importances = sorted(zip(available_cols, importances), key=lambda x: x[1], reverse=True)
                logging.info(f"[{ticker}] Feature importances (descending):")
                for feat, imp in feats_importances:
                    logging.info(f"   {feat}: {imp:.3f}")

        elif cmd == "commands":
            logging.info("Available commands:")
            logging.info("  turnoff")
            logging.info("  api-test")
            logging.info("  get-data [timeframe]")
            logging.info("  predict-next [-r]")
            logging.info("  run-sentiment [-r]")
            logging.info("  force-run [-r]")
            logging.info("  backtest <N> [simple|complex] [timeframe?] [-r?]")
            logging.info("  feature-importance [-r]")
            logging.info("  set-tickers (tickers)")
            logging.info("  set-timeframe (timeframe)")
            logging.info("  set-nbars (Number of candles)")
            logging.info("  set-news (on/off)")
            logging.info("  trade-logic <logic>")
            logging.info("  disable-feature <comma-separated features or main/base>")
            logging.info("  auto-feature [-r]")
            logging.info("  set-ntickers (Number)")
            logging.info("  ai-tickers")
            logging.info("  create-script (name)")
            logging.info("  commands")

        elif cmd == "set-tickers":
            if len(parts) < 2:
                logging.info("Usage: set-tickers TICKER1,TICKER2,...")
                continue
            new_tick_str = parts[1]
            update_env_variable("TICKERS", new_tick_str)
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

        elif cmd == "trade-logic":
            if len(parts) < 2:
                logging.info("Usage: trade-logic <logicValue>  # e.g. 1..15")
                continue
            new_logic = parts[1]
            update_env_variable("TRADE_LOGIC", new_logic)
            TRADE_LOGIC = new_logic
            logging.info(f"Updated TRADE_LOGIC in memory to {TRADE_LOGIC}")

        elif cmd == "disable-feature":
            if len(parts) < 2:
                logging.warning("Usage: disable-feature <comma-separated features OR 'main'/'base'>")
                continue
            new_disabled = parts[1]
            update_env_variable("DISABLED_FEATURES", new_disabled)
            DISABLED_FEATURES = new_disabled
            if new_disabled in ["main", "base"]:
                DISABLED_FEATURES_SET = set()
            else:
                if new_disabled.strip():
                    new_set = set([f.strip() for f in new_disabled.split(",") if f.strip()])
                else:
                    new_set = set()
                new_set.discard("sentiment")
                DISABLED_FEATURES_SET = new_set
            logging.info(f"Updated DISABLED_FEATURES in memory to {DISABLED_FEATURES}")
            logging.info(f"Updated DISABLED_FEATURES_SET to {DISABLED_FEATURES_SET}")

        elif cmd == "auto-feature":
            low_import_set = set()

            if skip_data:
                logging.info("auto-feature -r: Will use existing CSV data for each ticker.")
            else:
                logging.info("auto-feature: Will fetch fresh data for each ticker, then compute importance & disable low ones.")

            for ticker in TICKERS:
                tf_code = timeframe_to_code(BAR_TIMEFRAME)
                csv_filename = f"{ticker}_{tf_code}.csv"

                if not skip_data:
                    df = fetch_candles(ticker, bars=N_BARS, timeframe=BAR_TIMEFRAME)
                    if df.empty:
                        logging.error(f"[{ticker}] Unable to fetch data for auto-feature.")
                        continue
                    df = add_features(df)
                    df = compute_custom_features(df)
                    df = drop_disabled_features(df)
                    df.to_csv(csv_filename, index=False)
                    logging.info(f"[{ticker}] Fetched data & saved CSV for auto-feature.")
                else:
                    if not os.path.exists(csv_filename):
                        logging.error(f"[{ticker}] CSV file {csv_filename} not found, skipping.")
                        continue
                    df = pd.read_csv(csv_filename)
                    if df.empty:
                        logging.error(f"[{ticker}] CSV is empty, skipping.")
                        continue

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
                    logging.error(f"[{ticker}] No 'close' column after processing. Cannot compute auto-feature.")
                    continue
                df['target'] = df['close'].shift(-1)
                df.dropna(inplace=True)
                if len(df) < 10:
                    logging.error(f"[{ticker}] Not enough rows for training. Skipping.")
                    continue

                # Prepare features
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

                # Train RandomForest
                model_rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
                model_rf.fit(X, y)

                # 5) Evaluate feature importances
                importances = model_rf.feature_importances_
                feats_importances = sorted(zip(available_cols, importances), key=lambda x: x[1], reverse=True)

                # Check which are < 0.05
                below_threshold = [feat for feat, imp in feats_importances if imp < 0.050]
                if below_threshold:
                    logging.info(f"[{ticker}] The following features are < 0.050 importance: {below_threshold}")
                    for ft in below_threshold:
                        low_import_set.add(ft)
                else:
                    logging.info(f"[{ticker}] No features with < 0.050 importance found.")

            if low_import_set:
                logging.info(f"Combining newly found low-importance features: {low_import_set}")

                current_disabled = DISABLED_FEATURES if DISABLED_FEATURES else ""
                if current_disabled in ["main", "base"]:
                    logging.warning(f"DISABLED_FEATURES is set to '{current_disabled}', ignoring auto-feature changes.")
                else:
                    disabled_features_set_local = set()
                    if current_disabled.strip():
                        disabled_features_set_local = set([f.strip() for f in current_disabled.split(",") if f.strip()])
                    # Union with new low-importance features
                    final_disabled = disabled_features_set_local.union(low_import_set)

                    # Rebuild the string
                    new_disabled_str = ",".join(sorted(final_disabled))

                    update_env_variable("DISABLED_FEATURES", new_disabled_str)
                    DISABLED_FEATURES = new_disabled_str
                    new_set_for_memory = set([f.strip() for f in new_disabled_str.split(",") if f.strip()])
                    new_set_for_memory.discard("sentiment")
                    DISABLED_FEATURES_SET = new_set_for_memory

                    logging.info(f"auto-feature updated DISABLED_FEATURES to: {DISABLED_FEATURES}")
            else:
                logging.info("No features fell below 0.050 threshold across all tickers. No .env changes made.")
            
        elif cmd == "set-ntickers":
            if len(parts) < 2:
                logging.info("Usage: set-ntickers <intValue>")
                continue
            new_val_str = parts[1]
            try:
                new_val_int = int(new_val_str)
            except ValueError:
                logging.error(f"Invalid integer for set-ntickers: {new_val_str}")
                continue

            update_env_variable("AI_TICKER_COUNT", str(new_val_int))
            AI_TICKER_COUNT = new_val_int
            logging.info(f"Updated AI_TICKER_COUNT in memory to {AI_TICKER_COUNT}")


        elif cmd == "ai-tickers":
            if AI_TICKER_COUNT <= 0:
                logging.info("AI_TICKER_COUNT is 0 or not set. No AI tickers to fetch.")
                continue

            # Call the function with no excludes.
            new_ai_list = fetch_new_ai_tickers(AI_TICKER_COUNT, exclude_tickers=[])
            if not new_ai_list:
                logging.info("No AI tickers returned or an error occurred.")
            else:
                logging.info(f"AI recommended tickers: {new_ai_list}")

        elif cmd == "create-script":
            if len(parts) < 2:
                logging.info("Usage: create-script <name>")
                continue

            user_provided_name = parts[1]
            import re
            pattern_name = re.compile(r'^[a-z0-9_]+$')
            if not pattern_name.match(user_provided_name):
                logging.error("Invalid script name. Use only lowercase letters, digits, or underscores.")
                continue

            logic_dir = "logic"
            if not os.path.isdir(logic_dir):
                os.makedirs(logic_dir)

            existing_nums = []
            for fname in os.listdir(logic_dir):
                match = re.match(r'^logic_(\d+)_.*\.py$', fname)
                if match:
                    try:
                        existing_nums.append(int(match.group(1)))
                    except ValueError:
                        pass

            highest_num = max(existing_nums) if existing_nums else 0
            new_num = highest_num + 1
            new_script_name = f"logic_{new_num}_{user_provided_name}.py"
            new_script_path = os.path.join(logic_dir, new_script_name)

            template_content = """def run_logic(current_price, predicted_price, ticker):
            from forest import api, buy_shares, sell_shares, short_shares, close_short

            def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
                return "NONE"
        """
            try:
                with open(new_script_path, "w") as f:
                    f.write(template_content)
                logging.info(f"Created new logic script: {new_script_path}")
            except Exception as e:
                logging.error(f"Unable to create new logic script: {e}")
                continue

            _update_logic_json()
            logging.info("Updated logic_scripts.json to include the newly created script.")

        else:
            logging.warning(f"Unrecognized command: {cmd_line}")

def main():
    _update_logic_json()
    setup_schedule_for_timeframe(BAR_TIMEFRAME)
    listener_thread = threading.Thread(target=console_listener, daemon=True)
    listener_thread.start()
    logging.info("Bot started. Running schedule in local NY time.")
    logging.info("Commands: turnoff, api-test, get-data [timeframe], predict-next [-r], run-sentiment [-r], force-run [-r], backtest <N> [simple|complex] [timeframe?] [-r?], feature-importance [-r], commands, set-tickers, set-timeframe, set-nbars, set-news, trade-logic <1..15>, disable-feature <list or main/base>, auto-feature [-r], set-ntickers (Number), ai-tickers, create-script (name)")

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
