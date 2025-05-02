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
from timeframe import TIMEFRAME_SCHEDULE
import ast
from sklearn.linear_model import RidgeCV
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from alpaca_trade_api.rest import REST, QuoteV2, APIError
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance


# LSTM/TF/Keras for deep learning models
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

from transformers import BertModel, BertConfig
import torch
import torch.nn as nn

# -------------------------------
# 1. Load Configuration (from .env)
# -------------------------------
load_dotenv()

# trading / model selection ------------------------------------------------------------------
ML_MODEL  = os.getenv("ML_MODEL", "forest")

API_KEY   = os.getenv("ALPACA_API_KEY", "")
API_SECRET= os.getenv("ALPACA_API_SECRET", "")
API_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# scheduling & run‑mode ----------------------------------------------------------------------
RUN_SCHEDULE            = os.getenv("RUN_SCHEDULE", "on").lower().strip()
SENTIMENT_OFFSET_MINUTES= int(os.getenv("SENTIMENT_OFFSET_MINUTES", "20"))

BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "4Hour")
N_BARS        = int(os.getenv("N_BARS", "5000"))

# discord / ai / news ------------------------------------------------------------------------
DISCORD_MODE  = os.getenv("DISCORD_MODE", "off").lower().strip()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")
DISCORD_USER_ID = os.getenv("DISCORD_USER_ID", "")
BING_API_KEY  = os.getenv("BING_API_KEY", "")
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY", "")
AI_TICKER_COUNT = int(os.getenv("AI_TICKER_COUNT", "0"))
AI_TICKERS: list[str] = []

TICKERS = [t.strip().upper() for t in os.getenv("TICKERS", "TSLA").split(",") if t.strip()]
TRADE_LOG_FILENAME = "trade_log.csv"

# misc ---------------------------------------------------------------------------------------
N_ESTIMATORS = 100
RANDOM_SEED  = 42
NY_TZ = pytz.timezone("America/New_York")

NEWS_MODE = os.getenv("NEWS_MODE", "on").lower().strip()

# -----------------------------------------------------------------------------
# list of every feature column your models know about (including the new ones)
# -----------------------------------------------------------------------------
POSSIBLE_FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'volume', 'vwap',
    'price_change', 'high_low_range', 'log_volume', 'sentiment',
    'price_return', 'candle_rise', 'body_size', 'wick_to_body',
    'macd_line', 'macd_signal', 'macd_histogram',
    'rsi', 'momentum', 'roc', 'atr', 'hist_vol', 'obv', 'volume_change',
    'stoch_k', 'bollinger_upper', 'bollinger_lower',
    'ema_9', 'ema_21', 'ema_50', 'ema_200', 'adx',
    'lagged_close_1', 'lagged_close_2', 'lagged_close_3',
    'lagged_close_5', 'lagged_close_10'
]


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

def parse_ml_models():
    models = [m.strip().lower() for m in ML_MODEL.split(",") if m.strip()]
    if "all" in models:
        # Add special logic: all 5 (forest, xgboost, lstm, transformer_reg, transformer_cls)
        models = ["forest", "xgboost", "lstm", "transformer", "transformer_cls"]
    return models

def get_single_model(model_name, input_shape=None, num_features=None, lstm_seq=60):
    if model_name in ["forest", "rf", "randomforest"]:
        return RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
    elif model_name == "xgboost":
        return xgb.XGBRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
    elif model_name == "lstm":
        # --- Deep, robust LSTM stack with regularization ---
        if input_shape is None and num_features is not None:
            input_shape = (lstm_seq, num_features)
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Masking(mask_value=0.),  # handles missing/zero padding if present
            layers.LSTM(128, return_sequences=True, dropout=0.25, recurrent_dropout=0.25),
            layers.BatchNormalization(),
            layers.LSTM(64, return_sequences=True, dropout=0.20, recurrent_dropout=0.20),
            layers.BatchNormalization(),
            layers.LSTM(32, return_sequences=False, dropout=0.10, recurrent_dropout=0.10),
            layers.BatchNormalization(),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.15),
            layers.Dense(1)
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mae")
        return model
    elif model_name == "transformer":
        # --- Deep Transformer for regression with regularization ---
        class LargeTransformerRegressor(nn.Module):
            def __init__(self, num_features, seq_len=60):
                super().__init__()
                d_model = 64
                self.embedding = nn.Linear(num_features, d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=8,
                    dim_feedforward=256,
                    dropout=0.20,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
                self.dropout = nn.Dropout(0.20)
                self.fc1 = nn.Linear(seq_len * d_model, 32)
                self.relu = nn.ReLU()
                self.out = nn.Linear(32, 1)
            def forward(self, x):
                # x: [batch, seq, features]
                em = self.embedding(x)
                out = self.transformer(em)
                out = self.dropout(out)
                out = out.reshape(out.size(0), -1)
                out = self.fc1(out)
                out = self.relu(out)
                return self.out(out)
        return LargeTransformerRegressor
    elif model_name == "transformer_cls":
        # --- Large Transformer for classification (up/down, binary) ---
        class LargeTransformerClassifier(nn.Module):
            def __init__(self, num_features, seq_len=60):
                super().__init__()
                d_model = 64
                self.embedding = nn.Linear(num_features, d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=8,
                    dim_feedforward=256,
                    dropout=0.20,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
                self.dropout = nn.Dropout(0.20)
                self.fc1 = nn.Linear(seq_len * d_model, 32)
                self.relu = nn.ReLU()
                self.out = nn.Linear(32, 2)
            def forward(self, x):
                em = self.embedding(x)
                out = self.transformer(em)
                out = self.dropout(out)
                out = out.reshape(out.size(0), -1)
                out = self.fc1(out)
                out = self.relu(out)
                return self.out(out)
        return LargeTransformerClassifier
    else:
        logging.warning(f"Unknown model type {model_name}. Using RandomForest.")
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

_CANONICAL_TF = {
    "15min":  "15Min",
    "30min":  "30Min",
    "1h":     "1Hour",  "1hour": "1Hour",
    "2h":     "2Hour",  "2hour": "2Hour",
    "4h":     "4Hour",  "4hour": "4Hour",
    "1d":     "1Day",   "1day":  "1Day",
}

def canonical_timeframe(tf: str) -> str:
    """
    Return the canonical camel-cased timeframe (e.g. '4Hour')
    regardless of the capitalisation or abbreviation the user typed.
    Unknown strings are returned unchanged.
    """
    if not tf:
        return tf
    tf_clean = tf.strip().lower()
    return _CANONICAL_TF.get(tf_clean, tf)


# ==============================================================
# 1.  timeframe_to_code
# ==============================================================

def timeframe_to_code(tf: str) -> str:
    tf = canonical_timeframe(tf)
    mapping = {
        "15Min": "M15",
        "30Min": "M30",
        "1Hour": "H1",
        "2Hour": "H2",
        "4Hour": "H4",
        "1Day":  "D1",
    }
    return mapping.get(tf, tf)


# ==============================================================
# 2.  get_bars_per_day
# ==============================================================

def get_bars_per_day(tf: str) -> float:
    tf = canonical_timeframe(tf)
    mapping = {
        "15Min": 32,
        "30Min": 16,
        "1Hour":  8,
        "2Hour":  4,
        "4Hour":  2,
        "1Day":   1,
    }
    return mapping.get(tf, 1)

BAR_TIMEFRAME = canonical_timeframe(BAR_TIMEFRAME)

def fetch_candles(ticker: str, bars: int = 10000, timeframe: str = None) -> pd.DataFrame:
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
            adjustment='all',
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
        if NEWS_MODE == "off":
            logging.info(f"[{ticker}] NEWS_MODE=off, skipping news sentiment update.")
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

        df_sentiment = df[['timestamp']].copy()
        df_sentiment['sentiment'] = sentiments
        df_sentiment['sentiment'] = df_sentiment['sentiment'].apply(lambda x: f"{x:.15f}")
        tf_code = timeframe_to_code(BAR_TIMEFRAME)
        sentiment_csv_filename = f"{ticker}_sentiment_{tf_code}.csv"
        df_sentiment.to_csv(sentiment_csv_filename, index=False)
        logging.info(f"[{ticker}] Updated sentiment CSV: {sentiment_csv_filename}")

def merge_sentiment_from_csv(df, ticker):
    tf_code = timeframe_to_code(BAR_TIMEFRAME)
    sentiment_csv_filename = f"{ticker}_sentiment_{tf_code}.csv"
    if not os.path.exists(sentiment_csv_filename):
        logging.error(f"Sentiment CSV {sentiment_csv_filename} not found for ticker {ticker}.")
        return df
    
    try:
        df_sentiment = pd.read_csv(sentiment_csv_filename)
    except Exception as e:
        logging.error(f"Error reading sentiment CSV {sentiment_csv_filename}: {e}")
        return df

    df_sentiment['timestamp'] = pd.to_datetime(df_sentiment['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    news_list = []
    for _, row in df_sentiment.iterrows():
        news_list.append({
            "created_at": row['timestamp'],
            "sentiment": float(row['sentiment'])
        })
    
    new_sentiments = assign_sentiment_to_candles(df, news_list)
    df["sentiment"] = new_sentiments
    df["sentiment"] = df["sentiment"].apply(lambda x: f"{x:.15f}")
    logging.info(f"Merged sentiment from {sentiment_csv_filename} into latest data for {ticker}.")
    return df


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
    """
    Compute a variety of technical features on price data:
      - price_return, candle_rise, body_size, wick_to_body
      - MACD (12,26,9), ADX (14), EMAs (9,21,50,200)
      - RSI, momentum, ROC, ATR, historical volatility, OBV, volume_change
      - stochastic K, Bollinger Bands, lagged closes
    """
    # --- PRICE-BASED FEATURES ---
    if 'close' in df.columns:
        df['price_return'] = df['close'].pct_change().fillna(0)
    if all(c in df.columns for c in ['high', 'low']):
        df['candle_rise'] = df['high'] - df['low']
    if all(c in df.columns for c in ['close', 'open']):
        df['body_size'] = df['close'] - df['open']
        body = df['body_size'].replace(0, np.nan)
        if all(c in df.columns for c in ['high', 'low']):
            df['wick_to_body'] = ((df['high'] - df['low']) / body).replace(np.nan, 0)

    # --- MACD (12,26,9) ---
    if 'close' in df.columns:
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd_line    = ema12 - ema26
        signal_line  = macd_line.ewm(span=9, adjust=False).mean()
        df['macd_line']      = macd_line
        df['macd_signal']    = signal_line
        df['macd_histogram'] = macd_line - signal_line

    # --- EMAs (9,21,50,200) ---
    if 'close' in df.columns:
        for span in [9, 21, 50, 200]:
            df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()

    # --- ADX (14) ---
    if all(c in df.columns for c in ['high', 'low', 'close']):
        period = 14
        high  = df['high']
        low   = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low  - close.shift(1)).abs()
        tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        up_move   = high.diff()
        down_move = low.shift(1) - low
        plus_dm   = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm  = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        tr_smooth       = tr.rolling(window=period).sum()
        plus_dm_smooth  = plus_dm.rolling(window=period).sum()
        minus_dm_smooth = minus_dm.rolling(window=period).sum()

        plus_di  = 100 * plus_dm_smooth  / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        dx       = (plus_di.subtract(minus_di).abs() / (plus_di + minus_di)) * 100

        df['adx'] = dx.rolling(window=period).mean()

    # --- RSI (14) ---
    if 'close' in df.columns:
        window = 14
        delta = df['close'].diff()
        up    = delta.clip(lower=0)
        down  = -delta.clip(upper=0)
        ema_up   = up.ewm(com=window-1, adjust=False).mean()
        ema_down = down.ewm(com=window-1, adjust=False).mean()
        rs = ema_up / ema_down.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)

    # --- MOMENTUM and ROC ---
    if 'close' in df.columns:
        df['momentum'] = (df['close'] - df['close'].shift(14)).fillna(0)
        shifted = df['close'].shift(14)
        df['roc'] = ((df['close'] - shifted) / shifted.replace(0, np.nan) * 100).fillna(0)

    # --- ATR (14) ---
    if all(c in df.columns for c in ['high', 'low', 'close']):
        high_low   = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close  = (df['low'] - df['close'].shift(1)).abs()
        tr_calc    = high_low.combine(high_close, max).combine(low_close, max)
        df['atr']  = tr_calc.ewm(span=14, adjust=False).mean().fillna(0)

    # --- HISTORICAL VOLATILITY ---
    if 'close' in df.columns:
        ret = df['close'].pct_change()
        vol_std = ret.rolling(window=14).std()
        df['hist_vol'] = (vol_std * np.sqrt(14)).fillna(0)

    # --- ON-BALANCE VOLUME ---
    if all(c in df.columns for c in ['close', 'volume']):
        df['obv'] = 0
        for i in range(1, len(df)):
            prev = df.at[i-1, 'obv']
            if df.at[i, 'close'] > df.at[i-1, 'close']:
                df.at[i, 'obv'] = prev + df.at[i, 'volume']
            elif df.at[i, 'close'] < df.at[i-1, 'close']:
                df.at[i, 'obv'] = prev - df.at[i, 'volume']
            else:
                df.at[i, 'obv'] = prev

    # --- VOLUME CHANGE ---
    if 'volume' in df.columns:
        df['volume_change'] = df['volume'].pct_change().fillna(0)

    # --- STOCHASTIC %K (14) ---
    if all(c in df.columns for c in ['close', 'high', 'low']):
        low14  = df['low'].rolling(window=14).min()
        high14 = df['high'].rolling(window=14).max()
        stoch_k = ((df['close'] - low14) / (high14 - low14.replace(0, np.nan)) * 100).fillna(50)
        df['stoch_k'] = stoch_k

    # --- BOLLINGER BANDS (20,2) ---
    if 'close' in df.columns:
        ma20 = df['close'].rolling(window=20).mean()
        std20= df['close'].rolling(window=20).std()
        df['bollinger_upper'] = ma20 + 2 * std20
        df['bollinger_lower'] = ma20 - 2 * std20

    # --- LAGGED CLOSES ---
    if 'close' in df.columns:
        for lag in [1, 2, 3, 5, 10]:
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
def train_and_predict(df: pd.DataFrame, return_model_stack=False):
    if 'sentiment' in df.columns and df["sentiment"].dtype == object:
        try:
            df["sentiment"] = df["sentiment"].astype(float)
        except Exception as e:
            logging.error(f"Cannot convert sentiment column to float: {e}")
            return None

    df = add_features(df)
    df = compute_custom_features(df)
    df = drop_disabled_features(df)

    available_cols = [c for c in POSSIBLE_FEATURE_COLS if c in df.columns]

    if 'close' not in df.columns:
        logging.error("No 'close' in DataFrame, cannot create target.")
        return None

    df = df.copy()
    df['target'] = df['close'].shift(-1)
    df.dropna(inplace=True)
    if len(df) < 70:  # more needed for deep models
        logging.error("Not enough rows after shift to train. Need more candles.")
        return None

    X = df[available_cols]
    y = df['target']

    # --- Check for NaN or Inf in data or target. Skip transformer if found
    if np.isnan(X.values).any() or np.isnan(y.values).any() or np.isinf(X.values).any() or np.isinf(y.values).any():
        logging.error("NaN or Inf detected in features or targets! Will skip any deep models (LSTM/Transformer) for this call.")
        use_transformer = False
    else:
        use_transformer = True

    last_row_features = df.iloc[-1][available_cols]
    last_row_df = pd.DataFrame([last_row_features], columns=available_cols)
    last_X_np = np.array(last_row_df)

    ml_models = parse_ml_models()
    out_preds = []
    out_names = []

    # --- SEQUENCE Models (LSTM, Transformer, Transformer_cls only) need sequences
    seq_len = 60

    def series_to_supervised(Xvalues, yvalues, seq_length):
        n_samples = Xvalues.shape[0]
        n_features = Xvalues.shape[1]
        Xs, ys = [], []
        for i in range(n_samples - seq_length):
            Xs.append(Xvalues[i:i+seq_length,:])
            ys.append(yvalues[i+seq_length])
        Xs, ys = np.array(Xs), np.array(ys)
        return Xs, ys

    # --- LSTM -- Tensorflow/Keras (will be handled only if no NaN/Inf)
    if "lstm" in ml_models and use_transformer:
        X_lstm_np = np.array(X)
        y_lstm_np = np.array(y)
        X_lstm_win, y_lstm_win = series_to_supervised(X_lstm_np, y_lstm_np, seq_len)
        lstm_model = get_single_model("lstm", input_shape=(seq_len, X_lstm_np.shape[1]), num_features=X_lstm_np.shape[1], lstm_seq=seq_len)
        es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, verbose=1, restore_best_weights=True)
        cp = keras.callbacks.ModelCheckpoint("lstm_best_model.keras", save_best_only=True, monitor="val_loss", mode="min", verbose=0)
        lstm_model.fit(
            X_lstm_win, y_lstm_win,
            epochs=80,
            batch_size=32,
            verbose=0,
            validation_split=0.18,
            callbacks=[es, cp]
        )
        lstm_model.load_weights("lstm_best_model.keras")
        last_seq = X_lstm_np[-seq_len:,:].reshape(1,seq_len,-1)
        lstm_pred = lstm_model.predict(last_seq, verbose=0)[0][0]
        out_preds.append(lstm_pred)
        out_names.append("lstm_pred")

    # --- XGBoost
    if "xgboost" in ml_models:
        xgb_model = xgb.XGBRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
        xgb_model.fit(X, y)
        xgb_pred = xgb_model.predict(last_row_df)[0]
        out_preds.append(xgb_pred)
        out_names.append("xgb_pred")

    # --- Random Forest
    if "forest" in ml_models or "rf" in ml_models or "randomforest" in ml_models:
        rf_model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
        rf_model.fit(X, y)
        rf_pred = rf_model.predict(last_row_df)[0]
        out_preds.append(rf_pred)
        out_names.append("rf_pred")

    # ========== Transformer REGRESSION (PyTorch) ========== #
    if "transformer" in ml_models and use_transformer:
        try:
            # --- Scale input features --- (required for deep NNs)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X.values)
            y_scale_mean = y.mean()
            y_scale_std = y.std() if y.std() > 1e-5 else 1.0

            Xtr_np = X_scaled
            ytr_np = ((y.values - y_scale_mean) / y_scale_std)

            if np.isnan(Xtr_np).any() or np.isnan(ytr_np).any():
                logging.error("NaN in scaled Transformer input or target, skipping Transformer!")
                use_this_transformer = False
            else:
                use_this_transformer = True

            if use_this_transformer:
                Xtr_win, ytr_win = series_to_supervised(Xtr_np, ytr_np, seq_len)
                Xtr_win_torch = torch.tensor(Xtr_win, dtype=torch.float32)
                ytr_win_torch = torch.tensor(ytr_win, dtype=torch.float32)

                TransformerReg = get_single_model("transformer", num_features=Xtr_np.shape[1])
                tr_reg_model = TransformerReg(num_features=Xtr_np.shape[1], seq_len=seq_len)
                opt = torch.optim.Adam(tr_reg_model.parameters(), lr=0.0005)  # smaller learning rate
                loss_fn = torch.nn.L1Loss()
                tr_reg_model.train()

                for epoch in range(50):
                    opt.zero_grad()
                    y_pred = tr_reg_model(Xtr_win_torch).squeeze()
                    if torch.isnan(y_pred).any():
                        logging.error("NaN in TransformerReg model output! Skipping transformer this fit.")
                        break
                    loss = loss_fn(y_pred, ytr_win_torch)
                    if torch.isnan(loss):
                        logging.error("NaN loss in TransformerReg training loop, skipping this Transformer fit!")
                        break
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(tr_reg_model.parameters(), max_norm=2.0)
                    opt.step()
                    if epoch % 10 == 0:
                        logging.info(f"TransformerReg (epoch {epoch}): train loss = {loss.item():.5f}")
                tr_reg_model.eval()
                # Predict last seq using scaling
                last_x_seq = X_scaled[-seq_len:,:].reshape(1,seq_len,-1)
                last_seq_torch = torch.tensor(last_x_seq, dtype=torch.float32)
                with torch.no_grad():
                    y_pred_scaled = tr_reg_model(last_seq_torch).cpu().numpy()[0,0]
                tr_reg_pred = (y_pred_scaled * y_scale_std) + y_scale_mean  # invert scaling
                if not np.isnan(tr_reg_pred) and not np.isinf(tr_reg_pred):
                    out_preds.append(tr_reg_pred)
                    out_names.append("tr_reg_pred")
        except Exception as e:
            logging.error(f"TransformerReg failed to train: {e}")

    # ========== Transformer CLASSIFIER (PyTorch, scaled) ========== #
    if "transformer_cls" in ml_models and use_transformer:
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X.values)
            Xtr_np = X_scaled
            ytr_np = y.values
            Xtr_win, ytr_win = series_to_supervised(Xtr_np, ytr_np, seq_len)
            ytr_cls = (ytr_win > Xtr_win[:,-1,-1])
            ytr_cls = ytr_cls.astype(np.long)
            Xtr_win_torch = torch.tensor(Xtr_win, dtype=torch.float32)
            ytr_win_torch = torch.tensor(ytr_cls, dtype=torch.long)

            TransformerCls = get_single_model("transformer_cls", num_features=Xtr_np.shape[1])
            tr_cls_model = TransformerCls(num_features=Xtr_np.shape[1], seq_len=seq_len)
            opt = torch.optim.Adam(tr_cls_model.parameters(), lr=0.0005)
            loss_fn = torch.nn.CrossEntropyLoss()
            tr_cls_model.train()

            for epoch in range(30):
                opt.zero_grad()
                out_logits = tr_cls_model(Xtr_win_torch)
                if torch.isnan(out_logits).any():
                    logging.error("NaN outputs in TransformerCls! Skipping.")
                    break
                loss = loss_fn(out_logits, ytr_win_torch)
                if torch.isnan(loss):
                    logging.error("NaN loss in TransformerCls! Skipping fit.")
                    break
                loss.backward()
                torch.nn.utils.clip_grad_norm_(tr_cls_model.parameters(), max_norm=2.0)
                opt.step()
                if epoch % 8 == 0:
                    logging.info(f"TransformerCls (epoch {epoch}): train loss = {loss.item():.5f}")
            tr_cls_model.eval()
            last_x_seq = X_scaled[-seq_len:,:].reshape(1,seq_len,-1)
            last_seq_torch = torch.tensor(last_x_seq, dtype=torch.float32)
            with torch.no_grad():
                logits = tr_cls_model(last_seq_torch)[0]
                softmax_scores = torch.softmax(logits,dim=0).cpu().numpy()
                cls_dir_pred = softmax_scores[1]-softmax_scores[0]
            if not np.isnan(cls_dir_pred) and not np.isinf(cls_dir_pred):
                out_preds.append(cls_dir_pred)
                out_names.append("tr_cls_pred")
        except Exception as e:
            logging.error(f"TransformerCls failed to train: {e}")

    # Final output - meta model?
    if len(out_preds)==1:
        if return_model_stack:
            return out_preds[0], {out_names[0]:out_preds[0]}
        else:
            return out_preds[0]

    # Otherwise, meta-stack
    meta_X_list = []
    meta_y_list = []
    valid_idx_start = 69   # skip period lost by sequence windows
    for i in range(valid_idx_start, len(df)-1):  # for stacking meta-train
        features_i = df.iloc[i][available_cols].values
        meta_features = []
        for m_idx, mname in enumerate(out_names):
            val = None
            if mname == "lstm_pred":
                seq = np.array(df.iloc[i-seq_len+1:i+1][available_cols])
                if seq.shape[0]==seq_len:
                    seq = seq.reshape(1,seq_len,-1)
                    lstm_pred_meta = lstm_model.predict(seq,verbose=0)[0][0]
                    meta_features.append(lstm_pred_meta)
                else:
                    meta_features.append(np.nan)
            elif mname == "xgb_pred":
                meta_features.append(xgb_model.predict(features_i.reshape(1,-1))[0])
            elif mname == "rf_pred":
                meta_features.append(rf_model.predict(features_i.reshape(1,-1))[0])
            elif mname == "tr_reg_pred":
                seq = np.array(df.iloc[i-seq_len+1:i+1][available_cols])
                if seq.shape[0]==seq_len and use_transformer:
                    seq_scaled = scaler.transform(seq)
                    seq_t = torch.tensor(seq_scaled.reshape(1,seq_len,-1),dtype=torch.float32)
                    with torch.no_grad():
                        y_pred_scaled = tr_reg_model(seq_t).cpu().numpy()[0,0]
                    y_pred_rescaled = (y_pred_scaled * y_scale_std) + y_scale_mean
                    meta_features.append(y_pred_rescaled)
                else:
                    meta_features.append(np.nan)
            elif mname == "tr_cls_pred":
                seq = np.array(df.iloc[i-seq_len+1:i+1][available_cols])
                if seq.shape[0]==seq_len and use_transformer:
                    seq_scaled = scaler.transform(seq)
                    seq_t = torch.tensor(seq_scaled.reshape(1,seq_len,-1),dtype=torch.float32)
                    with torch.no_grad():
                        logits = tr_cls_model(seq_t)[0]
                        softmax_scores = torch.softmax(logits,dim=0).cpu().numpy()
                        cls_dir_pred = softmax_scores[1]-softmax_scores[0]
                        meta_features.append(cls_dir_pred)
                else:
                    meta_features.append(np.nan)
            else:
                meta_features.append(np.nan)
        if not any(np.isnan(meta_features)):
            meta_X_list.append(meta_features)
            meta_y_list.append(df.iloc[i+1]['close'])
    if len(meta_X_list)<3:
        logging.warning(f"Meta-model: Not enough meta training data, falling back to mean of predictions.")
        pred = np.mean([p for p in out_preds])
        if return_model_stack:
            return pred, dict(zip(out_names,out_preds))
        else:
            return pred

    meta_X = np.array(meta_X_list)
    meta_y = np.array(meta_y_list)
    meta_model = RidgeCV().fit(meta_X, meta_y)
    pred_stack = np.array(out_preds).reshape(1,-1)
    final_pred = float(meta_model.predict(pred_stack)[0])
    if return_model_stack:
        pred_details = dict(zip(out_names, out_preds))
        return final_pred, pred_details
    else:
        return final_pred

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
            model="",
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
        pos = None
        try:
            pos = api.get_position(ticker)
            pos_qty = float(pos.qty)
        except Exception:
            pos = None
            pos_qty = 0.0

        already_long = pos_qty > 0
        already_short = pos_qty < 0
        abs_short_qty = abs(pos_qty) if already_short else 0.0

        # Step 1: If short, fully cover first (to flat)
        if already_short and abs_short_qty > 0:
            if DISCORD_MODE == "on":
                send_discord_order_message(
                    "COVER", ticker, buy_price, predicted_price,
                    extra_info="Auto-covering short before BUY."
                )
            else:
                api.submit_order(
                    symbol=ticker,
                    qty=int(abs_short_qty),
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
                logging.info(f"[{ticker}] Auto-COVER {int(abs_short_qty)} at {buy_price:.2f} before BUY")
                log_trade("COVER", ticker, abs_short_qty, buy_price, None, None)
            pos_qty = 0.0   # Now flat

        # Step 2: If already long, avoid duplicate buy
        if already_long:
            logging.info(f"[{ticker}] Already long {int(pos_qty)} shares. Skipping new BUY to prevent duplicate long position.")
            return

        # Step 3: Proceed to new BUY for desired qty
        if DISCORD_MODE == "on":
            send_discord_order_message(
                "BUY", ticker, buy_price, predicted_price,
                extra_info="Buying shares via Discord bot."
            )
        else:
            # Risk allocation as before
            account = api.get_account()
            available_cash = float(account.cash)
            total_ticker_slots = (len(TICKERS) if TICKERS else 0) + AI_TICKER_COUNT
            total_ticker_slots = max(total_ticker_slots, 1)
            split_cash = available_cash / total_ticker_slots
            max_shares = int(split_cash // buy_price)
            final_qty = min(qty, max_shares)
            if final_qty <= 0:
                logging.info(f"[{ticker}] Not enough split cash to buy any shares. Skipping.")
                return
            api.submit_order(
                symbol=ticker,
                qty=int(final_qty),
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            logging.info(f"[{ticker}] BUY {int(final_qty)} at {buy_price:.2f} (Predicted: {predicted_price:.2f})")
            log_trade("BUY", ticker, final_qty, buy_price, predicted_price, None)
            if ticker not in TICKERS and ticker not in AI_TICKERS:
                AI_TICKERS.append(ticker)
    except Exception as e:
        logging.error(f"[{ticker}] Buy order failed: {e}")

def sell_shares(ticker, qty, sell_price, predicted_price):
    if qty <= 0:
        return
    try:
        pos = None
        try:
            pos = api.get_position(ticker)
            pos_qty = float(pos.qty)
        except Exception:
            pos = None
            pos_qty = 0.0

        already_long = pos_qty > 0
        already_short = pos_qty < 0
        abs_long_qty = pos_qty if already_long else 0.0

        if not already_long or abs_long_qty <= 0:
            logging.info(f"[{ticker}] No long position to SELL. Skipping.")
            return

        sellable_qty = min(qty, abs_long_qty)
        if sellable_qty <= 0:
            logging.info(f"[{ticker}] No shares to SELL.")
            return

        if DISCORD_MODE == "on":
            send_discord_order_message(
                "SELL", ticker, sell_price, predicted_price,
                extra_info="Selling shares via Discord bot."
            )
        else:
            avg_entry = float(pos.avg_entry_price) if pos else 0.0
            api.submit_order(
                symbol=ticker,
                qty=int(sellable_qty),
                side='sell',
                type='market',
                time_in_force='gtc'
            )
            pl = (sell_price - avg_entry) * sellable_qty
            logging.info(f"[{ticker}] SELL {int(sellable_qty)} at {sell_price:.2f} (Predicted: {predicted_price:.2f}, P/L: {pl:.2f})")
            log_trade("SELL", ticker, sellable_qty, sell_price, predicted_price, pl)
            try:
                new_pos = api.get_position(ticker)
                if float(new_pos.qty) == 0 and ticker in AI_TICKERS:
                    AI_TICKERS.remove(ticker)
                    _ensure_ai_tickers()
            except Exception:
                if ticker in AI_TICKERS:
                    AI_TICKERS.remove(ticker)
                    _ensure_ai_tickers()
    except Exception as e:
        logging.error(f"[{ticker}] Sell order failed: {e}")

def short_shares(ticker, qty, short_price, predicted_price):
    if qty <= 0:
        return
    try:
        pos = None
        try:
            pos = api.get_position(ticker)
            pos_qty = float(pos.qty)
        except Exception:
            pos = None
            pos_qty = 0.0

        already_long = pos_qty > 0
        already_short = pos_qty < 0
        abs_long_qty = pos_qty if already_long else 0.0
        abs_short_qty = abs(pos_qty) if already_short else 0.0

        # Step 1: If long, fully sell to flat
        if already_long and abs_long_qty > 0:
            if DISCORD_MODE == "on":
                send_discord_order_message(
                    "SELL", ticker, short_price, predicted_price,
                    extra_info="Auto-selling long before SHORT."
                )
            else:
                avg_entry = float(pos.avg_entry_price)
                api.submit_order(
                    symbol=ticker,
                    qty=int(abs_long_qty),
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                pl = (short_price - avg_entry) * abs_long_qty
                logging.info(f"[{ticker}] Auto-SELL {int(abs_long_qty)} at {short_price:.2f} before SHORT")
                log_trade("SELL", ticker, abs_long_qty, short_price, predicted_price, pl)
            pos_qty = 0.0  # Flat

        # Step 2: If already short, avoid duplicate short
        if already_short:
            logging.info(f"[{ticker}] Already short {int(abs_short_qty)} shares. Skipping new SHORT to prevent duplicate short position.")
            return

        # Step 3: Proceed to open new SHORT for qty
        if DISCORD_MODE == "on":
            send_discord_order_message(
                "SHORT", ticker, short_price, predicted_price,
                extra_info="Shorting shares via Discord bot."
            )
        else:
            account = api.get_account()
            available_cash = float(account.cash)
            total_ticker_slots = (len(TICKERS) if TICKERS else 0) + AI_TICKER_COUNT
            total_ticker_slots = max(total_ticker_slots, 1)
            split_cash = available_cash / total_ticker_slots
            max_shares = int(split_cash // short_price)
            final_qty = min(qty, max_shares)
            if final_qty <= 0:
                logging.info(f"[{ticker}] Not enough split cash/margin to short any shares. Skipping.")
                return
            api.submit_order(
                symbol=ticker,
                qty=int(final_qty),
                side='sell',
                type='market',
                time_in_force='gtc'
            )
            logging.info(f"[{ticker}] SHORT {int(final_qty)} at {short_price:.2f} (Predicted: {predicted_price:.2f})")
            log_trade("SHORT", ticker, final_qty, short_price, predicted_price, None)
            if ticker not in TICKERS and ticker not in AI_TICKERS:
                AI_TICKERS.append(ticker)
    except Exception as e:
        logging.error(f"[{ticker}] Short order failed: {e}")

def close_short(ticker, qty, cover_price):
    if qty <= 0:
        return
    try:
        pos = None
        try:
            pos = api.get_position(ticker)
            pos_qty = float(pos.qty)
        except Exception:
            pos = None
            pos_qty = 0.0

        already_short = pos_qty < 0
        abs_short_qty = abs(pos_qty) if already_short else 0.0

        if not already_short or abs_short_qty <= 0:
            logging.info(f"[{ticker}] No short position to COVER. Skipping.")
            return

        coverable_qty = min(qty, abs_short_qty)
        if coverable_qty <= 0:
            logging.info(f"[{ticker}] No shares to COVER.")
            return

        if DISCORD_MODE == "on":
            send_discord_order_message(
                "COVER", ticker, cover_price, 0,
                extra_info="Covering short via Discord bot."
            )
        else:
            avg_entry = float(pos.avg_entry_price) if pos else 0.0
            api.submit_order(
                symbol=ticker,
                qty=int(coverable_qty),
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            pl = (avg_entry - cover_price) * coverable_qty
            logging.info(f"[{ticker}] COVER SHORT {int(coverable_qty)} at {cover_price:.2f} (P/L: {pl:.2f})")
            log_trade("COVER", ticker, coverable_qty, cover_price, None, pl)
            try:
                new_pos = api.get_position(ticker)
                if float(new_pos.qty) == 0 and ticker in AI_TICKERS:
                    AI_TICKERS.remove(ticker)
                    _ensure_ai_tickers()
            except Exception:
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

def get_current_price() -> float:
    """
    Fetches the latest quote for TSLA and returns its estimated current price
    (midpoint of bid and ask) as a float.
    Expects ALPACA_API_KEY and ALPACA_API_SECRET in the environment.
    """
    key    = os.getenv("ALPACA_API_KEY", "")
    secret = os.getenv("ALPACA_API_SECRET", "")
    if not key or not secret:
        print("Error: set ALPACA_API_KEY and ALPACA_API_SECRET in your environment.", file=sys.stderr)
        sys.exit(1)

    api = REST(key, secret, "https://paper-api.alpaca.markets", api_version="v2")

    try:
        quote: QuoteV2 = api.get_latest_quote("TSLA")
    except APIError as e:
        print(f"API error fetching quote: {e}", file=sys.stderr)
        sys.exit(2)

    # return a float, not a formatted string
    return (quote.bid_price + quote.ask_price) / 2

# -------------------------------
# 9b. The central "trade_logic" placeholder
# -------------------------------
def trade_logic(current_price: float, predicted_price: float, ticker: str):
    try:
        import importlib
        logic_module_name = _get_logic_script_name(TRADE_LOGIC)
        module_path = f"logic.{logic_module_name}"
        logic_module = importlib.import_module(module_path)
        logging.info(f"This is a test with {logic_module}")
        real_current = get_current_price()
        logic_module.run_logic(current_price, predicted_price, ticker)
    except Exception as e:
        logging.error(f"Error dispatching to trade logic '{TRADE_LOGIC}': {e}")

# -------------------------------
# 10. Trading Job
# -------------------------------

def check_latest_candle_condition(df: pd.DataFrame, timeframe: str, scheduled_time_ny: str) -> bool:
    if df.empty:
        return False

    last_ts = pd.to_datetime(df.iloc[-1]['timestamp'])
    last_ts_str = last_ts.strftime("%H:%M:%S+00:00")
    
    expected = last_ts_str

    if timeframe == "4Hour":
        if scheduled_time_ny == "20:01":
            expected = "20:00:00+00:00"
    elif timeframe == "2Hour":
        if scheduled_time_ny == "10:01":
            expected = "12:00:00+00:00"
        elif scheduled_time_ny == "18:01":
            expected = "20:00:00+00:00"
    elif timeframe == "1Hour":
        if scheduled_time_ny == "11:01":
            expected = "12:00:00+00:00"
        elif scheduled_time_ny == "12:01":
            expected = "13:00:00+00:00"
        elif scheduled_time_ny == "16:01":
            expected = "20:00:00+00:00"
        elif scheduled_time_ny == "17:01":
            expected = "21:00:00+00:00"
    else:
        return True

    if last_ts_str == expected:
        return True
    else:
        logging.info(f"Latest candle time {last_ts_str} does not match expected {expected} for timeframe {timeframe} at scheduled NY time {scheduled_time_ny}.")
        return False



def _perform_trading_job(skip_data=False, scheduled_time_ny: str = None):
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
                df = merge_sentiment_from_csv(df, ticker)
                df = drop_disabled_features(df)
                df.to_csv(csv_filename, index=False)
                logging.info(f"[{ticker}] Updated candle CSV with merged sentiment & features, minus disabled.")
            else:
                logging.info(f"[{ticker}] NEWS_MODE=off, skipping sentiment merge.")
        else:
            logging.info(f"[{ticker}] skip_data=True. Using existing CSV {csv_filename} for trade job.")
            if not os.path.exists(csv_filename):
                logging.error(f"[{ticker}] CSV file {csv_filename} does not exist. Cannot trade.")
                continue
            df = pd.read_csv(csv_filename)
            if df.empty:
                logging.error(f"[{ticker}] Existing CSV is empty. Skipping.")
                continue

        # --- NEW: Check the latest candle condition based on the scheduled NY time ---
        if scheduled_time_ny is not None:
            if not check_latest_candle_condition(df, BAR_TIMEFRAME, scheduled_time_ny):
                logging.info(f"[{ticker}] Latest candle condition not met for timeframe {BAR_TIMEFRAME} at scheduled time {scheduled_time_ny}. Skipping trade.")
                continue
        # ---------------------------------------------------------------------------

        pred_close = train_and_predict(df)
        if pred_close is None:
            logging.error(f"[{ticker}] Model training or prediction failed. Skipping trade logic.")
            continue
        current_price = float(df.iloc[-1]['close'])
        logging.info(f"[{ticker}] Current Price = {current_price:.2f}, Predicted Next Close = {pred_close:.2f}")
        trade_logic(current_price, pred_close, ticker)


def setup_schedule_for_timeframe(timeframe: str) -> None:
    """
    Build the daily schedule for the requested bar length.

    • Skips all scheduling when RUN_SCHEDULE == "off"
    • Uses SENTIMENT_OFFSET_MINUTES from .env for the news‑model lead‑in
    • Relies on TIMEFRAME_SCHEDULE imported from timeframe.py
    """
    import schedule

    # --- honour master switch --------------------------------------------------------------
    if RUN_SCHEDULE == "off":
        schedule.clear()
        logging.info("RUN_SCHEDULE=off – no jobs have been queued.")
        return
    # ---------------------------------------------------------------------------------------

    # choose a known table key --------------------------------------------------------------
    if timeframe not in TIMEFRAME_SCHEDULE:
        logging.warning(f"{timeframe} not recognised; defaulting to 1Day.")
        timeframe = "1Day"

    times_list = TIMEFRAME_SCHEDULE[timeframe]
    schedule.clear()

    # build the jobs ------------------------------------------------------------------------
    for t in times_list:
        # trading run
        schedule.every().day.at(t).do(lambda t=t: run_job(t))

        # optional sentiment lead‑in
        if NEWS_MODE == "on":
            try:
                base_time      = datetime.strptime(t, "%H:%M")
                offset_minutes = SENTIMENT_OFFSET_MINUTES          # ← env‑controlled
                sentiment_time = (base_time - timedelta(minutes=offset_minutes)).strftime("%H:%M")
                schedule.every().day.at(sentiment_time).do(run_sentiment_job)
                logging.info(f"Scheduled sentiment update {sentiment_time} NY & trade {t} NY.")
            except Exception as e:
                logging.error(f"Error scheduling sentiment lead‑in for {t}: {e}")
        else:
            logging.info(f"NEWS_MODE={NEWS_MODE}. Sentiment run skipped before {t}.")

    logging.info(f"Scheduling prepared for {timeframe}: {times_list}")



def run_job(scheduled_time_ny: str):
    logging.info(f"Starting scheduled trading job at NY time {scheduled_time_ny}...")
    
    today_str = datetime.now().date().isoformat()
    try:
        calendar = api.get_calendar(start=today_str, end=today_str)
        if not calendar:
            logging.info("Market is not scheduled to open today. Skipping job.")
            return
    except Exception as e:
        logging.error(f"Error fetching market calendar: {e}")
        return

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            clock = api.get_clock()
            break
        except Exception as e:
            logging.error(f"Error checking market clock (Attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(60)
            else:
                logging.error("Max retries exceeded. Skipping the scheduled job.")
                return

    _perform_trading_job(skip_data=False, scheduled_time_ny=scheduled_time_ny)
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
            # After run_sentiment_job, merge the sentiment data into the full candle CSV
            for ticker in TICKERS:
                tf_code = timeframe_to_code(BAR_TIMEFRAME)
                candle_csv = f"{ticker}_{tf_code}.csv"
                if not os.path.exists(candle_csv):
                    logging.error(f"[{ticker}] Candle CSV {candle_csv} not found, skipping merge.")
                    continue
                try:
                    df = pd.read_csv(candle_csv)
                except Exception as e:
                    logging.error(f"[{ticker}] Error reading candle CSV: {e}")
                    continue

                df_updated = merge_sentiment_from_csv(df, ticker)
                try:
                    df_updated.to_csv(candle_csv, index=False)
                    logging.info(f"[{ticker}] Updated unified candle CSV with sentiment: {candle_csv}")
                except Exception as e:
                    logging.error(f"[{ticker}] Error saving unified candle CSV: {e}")
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

                available_cols = [c for c in POSSIBLE_FEATURE_COLS if c in df.columns]

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
                    if len(train_df) < 70:
                        logging.error(f"[{ticker}] Not enough training rows for simple approach.")
                        continue

                    # Fit models, but only on train
                    pred_stack = []
                    for i_ in range(len(test_df)):
                        sub_df = pd.concat([train_df, test_df.iloc[:i_+1]], axis=0, ignore_index=True)
                        pred_close = train_and_predict(sub_df)
                        row_idx = test_df.index[i_]
                        row_data = df.loc[row_idx]
                        real_close = row_data['close']
                        timestamps.append(row_data['timestamp'])
                        predictions.append(pred_close)
                        actuals.append(real_close)

                        action = logic_module.run_backtest(
                            current_price=real_close,
                            predicted_price=pred_close,
                            position_qty=position_qty,
                            current_timestamp=row_data['timestamp'],
                            candles=test_df
                        )

                        if action == "BUY":
                            if position_qty < 0:
                                pl = (avg_entry_price - real_close) * abs(position_qty)
                                cash += pl
                                record_trade("COVER", row_data['timestamp'], abs(position_qty), real_close, pred_close, pl)
                                position_qty = 0
                                avg_entry_price = 0.0
                            shares_to_buy = int(cash // real_close)
                            if shares_to_buy > 0:
                                position_qty = shares_to_buy
                                avg_entry_price = real_close
                                record_trade("BUY", row_data['timestamp'], shares_to_buy, real_close, pred_close, None)
                        elif action == "SELL":
                            if position_qty > 0:
                                pl = (real_close - avg_entry_price) * position_qty
                                cash += pl
                                record_trade("SELL", row_data['timestamp'], position_qty, real_close, pred_close, pl)
                                position_qty = 0
                                avg_entry_price = 0.0
                        elif action == "SHORT":
                            if position_qty > 0:
                                pl = (real_close - avg_entry_price) * position_qty
                                cash += pl
                                record_trade("SELL", row_data['timestamp'], position_qty, real_close, pred_close, pl)
                                position_qty = 0
                                avg_entry_price = 0.0
                            shares_to_short = int(cash // real_close)
                            if shares_to_short > 0:
                                position_qty = -shares_to_short
                                avg_entry_price = real_close
                                record_trade("SHORT", row_data['timestamp'], shares_to_short, real_close, pred_close, None)
                        elif action == "COVER":
                            if position_qty < 0:
                                pl = (avg_entry_price - real_close) * abs(position_qty)
                                cash += pl
                                record_trade("COVER", row_data['timestamp'], abs(position_qty), real_close, pred_close, pl)
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
                        if len(sub_train) < 70:
                            logging.warning(f"[{ticker}] Not enough data to train at iteration {i_}. Skipping.")
                            continue

                        pred_close = train_and_predict(sub_train)
                        row_data = df.iloc[i_]
                        real_close = row_data['close']
                        timestamps.append(row_data['timestamp'])
                        predictions.append(pred_close)
                        actuals.append(real_close)

                        action = logic_module.run_backtest(
                            current_price=real_close,
                            predicted_price=pred_close,
                            position_qty=position_qty,
                            current_timestamp=row_data['timestamp'],
                            candles=backtest_candles
                        )

                        if action == "BUY":
                            if position_qty < 0:
                                pl = (avg_entry_price - real_close) * abs(position_qty)
                                cash += pl
                                record_trade("COVER", row_data['timestamp'], abs(position_qty), real_close, pred_close, pl)
                                position_qty = 0
                                avg_entry_price = 0.0
                            shares_to_buy = int(cash // real_close)
                            if shares_to_buy > 0:
                                position_qty = shares_to_buy
                                avg_entry_price = real_close
                                record_trade("BUY", row_data['timestamp'], shares_to_buy, real_close, pred_close, None)
                        elif action == "SELL":
                            if position_qty > 0:
                                pl = (real_close - avg_entry_price) * position_qty
                                cash += pl
                                record_trade("SELL", row_data['timestamp'], position_qty, real_close, pred_close, pl)
                                position_qty = 0
                                avg_entry_price = 0.0
                        elif action == "SHORT":
                            if position_qty > 0:
                                pl = (real_close - avg_entry_price) * position_qty
                                cash += pl
                                record_trade("SELL", row_data['timestamp'], position_qty, real_close, pred_close, pl)
                                position_qty = 0
                                avg_entry_price = 0.0
                            shares_to_short = int(cash // real_close)
                            if shares_to_short > 0:
                                position_qty = -shares_to_short
                                avg_entry_price = real_close
                                record_trade("SHORT", row_data['timestamp'], shares_to_short, real_close, pred_close, None)
                        elif action == "COVER":
                            if position_qty < 0:
                                pl = (avg_entry_price - real_close) * abs(position_qty)
                                cash += pl
                                record_trade("COVER", row_data['timestamp'], abs(position_qty), real_close, pred_close, pl)
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

                out_df = pd.DataFrame({
                    'timestamp': timestamps,
                    'actual_close': actuals,
                    'predicted_close': predictions
                })
                out_csv = f"backtest_predictions_{ticker}_{test_size}_{tf_code}_{approach}.csv"
                out_df.to_csv(out_csv, index=False)
                logging.info(f"[{ticker}] Saved backtest predictions to {out_csv}.")

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
                tf_code  = timeframe_to_code(BAR_TIMEFRAME)
                csv_file = f"{ticker}_{tf_code}.csv"

                # ─── FETCH OR LOAD ──────────────────────────────────────────────────────
                if not skip_data:
                    df = fetch_candles(ticker, bars=N_BARS, timeframe=BAR_TIMEFRAME)
                    if df.empty:
                        logging.error(f"[{ticker}] Unable to fetch data for feature-importance.")
                        continue
                    df = add_features(df)
                    df = compute_custom_features(df)
                    df = drop_disabled_features(df)
                    df.to_csv(csv_file, index=False)
                    logging.info(f"[{ticker}] Fetched data & saved CSV for feature-importance.")
                else:
                    if not os.path.exists(csv_file):
                        logging.error(f"[{ticker}] CSV {csv_file} not found, skipping.")
                        continue
                    df = pd.read_csv(csv_file)
                    if df.empty:
                        logging.error(f"[{ticker}] CSV is empty, skipping.")
                        continue

                # ─── CLEAN & ENGINEER ───────────────────────────────────────────────────
                if 'sentiment' in df.columns and df['sentiment'].dtype == object:
                    try:
                        df['sentiment'] = df['sentiment'].astype(float)
                    except Exception as e:
                        logging.error(f"[{ticker}] Could not convert sentiment: {e}")
                        continue

                df = add_features(df)
                df = compute_custom_features(df)
                df = drop_disabled_features(df)

                # ─── DEFINE TARGET AS NEXT-DAY RETURN ───────────────────────────────────
                df['target'] = df['close'].pct_change().shift(-1)
                df.dropna(inplace=True)
                if len(df) < 30:
                    logging.error(f"[{ticker}] Not enough data after shift, skipping.")
                    continue

                # ─── PREPARE FEATURES (INCLUDING RAW CLOSE) ─────────────────────────────
                features = [c for c in POSSIBLE_FEATURE_COLS if c in df.columns]
                X_all   = df[features]
                y_all   = df['target']

                # ─── EXPANDING-WINDOW ROLLING FORECAST & IMPORTANCE ACCUMULATION ────────
                train_size = int(len(df) * 0.8)           # e.g. 2394 of 2992
                imp_accum  = np.zeros(len(features))
                n_models   = 0

                for i in range(train_size, len(df)):
                    X_train = X_all.iloc[:i]
                    y_train = y_all.iloc[:i]
                    X_test  = X_all.iloc[i:i+1]
                    y_test  = y_all.iloc[i]

                    # train on [0:i), predict i, then include i in next train
                    model = RandomForestRegressor(
                        n_estimators=N_ESTIMATORS,
                        random_state=RANDOM_SEED,
                        n_jobs=-1
                    )
                    model.fit(X_train, y_train)

                    # optional: capture rolling predictions
                    y_pred = model.predict(X_test)[0]
                    # (you could store y_pred vs y_test here for metrics)

                    imp_accum += model.feature_importances_
                    n_models  += 1

                # ─── AVERAGE IMPORTANCES & LOG ─────────────────────────────────────────
                avg_imps = imp_accum / n_models
                ranked   = sorted(zip(features, avg_imps), key=lambda x: x[1], reverse=True)

                logging.info(f"[{ticker}] Rolling-window feature importances (avg over {n_models} models):")
                for feat, imp in ranked:
                    logging.info(f"   {feat}: {imp:.4f}")

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

                available_cols = [c for c in POSSIBLE_FEATURE_COLS if c in df.columns]
                X = df[available_cols]
                y = df['target']

                model_rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
                model_rf.fit(X, y)

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
                    final_disabled = disabled_features_set_local.union(low_import_set)

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
