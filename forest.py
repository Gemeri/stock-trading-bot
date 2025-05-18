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
import lightgbm as lgb
import time


# LSTM/TF/Keras for deep learning models
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import importlib.util

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
REWRITE = os.getenv("REWRITE", "off").strip().lower()


# ----------------------------------------------------
# list of every feature column your models know about
# ----------------------------------------------------
POSSIBLE_FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'volume', 'vwap', 'sentiment',
    'macd_line', 'macd_signal', 'macd_histogram',
    'rsi', 'momentum', 'roc', 'atr', 'obv',
    'bollinger_upper', 'bollinger_lower',
    'ema_9', 'ema_21', 'ema_50', 'ema_200', 'adx',
    'lagged_close_1', 'lagged_close_2', 'lagged_close_3',
    'lagged_close_5', 'lagged_close_10',
    'candle_body_ratio',
    'wick_dominance',
    'gap_vs_prev',
    'volume_zscore',
    'atr_zscore',
    'rsi_zscore',
    'adx_trend',
    'macd_cross',
    'macd_hist_flip',
    'day_of_week',
    'days_since_high',
    'days_since_low'
]

def call_sub_main(mode, df, execution):
    sub_main_path = os.path.join(os.path.dirname(__file__), "sub", "main.py")
    spec = importlib.util.spec_from_file_location("sub_main", sub_main_path)
    sub_main = importlib.util.module_from_spec(spec)
    sys.modules["sub_main"] = sub_main
    spec.loader.exec_module(sub_main)
    sub_main.MODE       = mode
    sub_main.EXECUTION  = execution
    if execution == "live":
        csv_path = "_sub_tmp_live.csv"
        df.to_csv(csv_path, index=False)
        sub_main.CSV_PATH = csv_path
        result = sub_main.run_live(return_result=True)
        os.remove(csv_path)
        return result
    else:
        csv_path = "_sub_tmp_backtest.csv"
        df.to_csv(csv_path, index=False)
        sub_main.CSV_PATH = csv_path
        results_df = sub_main.run_backtest(return_df=True)
        os.remove(csv_path)
        return results_df

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
        models = ["forest", "xgboost", "lightgbm", "lstm", "transformer"]
    canonical = []
    for m in models:
        if m in ["lgbm", "lightgbm"]:
            canonical.append("lightgbm")
        elif m in ["sub-vote", "sub_vote"]:
            canonical.append("sub-vote")
        elif m in ["sub-meta", "sub_meta"]:
            canonical.append("sub-meta")
        else:
            canonical.append(m)
    return canonical

def get_single_model(model_name, input_shape=None, num_features=None, lstm_seq=60):
    if model_name in ["forest", "rf", "randomforest"]:
        return RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
    elif model_name == "xgboost":
        return xgb.XGBRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
    elif model_name == "lightgbm":
        return lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
    elif model_name == "lstm":
        if input_shape is None and num_features is not None:
            input_shape = (lstm_seq, num_features)
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Masking(mask_value=0.),
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
                em = self.embedding(x)
                out = self.transformer(em)
                out = self.dropout(out)
                out = out.reshape(out.size(0), -1)
                out = self.fc1(out)
                out = self.relu(out)
                return self.out(out)
        return LargeTransformerRegressor
    elif model_name == "classifier":
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
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    sentiment_class = outputs.logits.argmax(dim=1).item()
    confidence_scores = outputs.logits.softmax(dim=1)[0].tolist()
    sentiment_score = confidence_scores[2] - confidence_scores[0]
    return sentiment_class, confidence_scores, sentiment_score

# -------------------------------
# 4b. Dictionary to map TRADE_LOGIC to actual module filenames
# -------------------------------
CLASSIFIER_MODELS = {"classifier", "sub-vote", "sub_vote", "sub-meta", "sub_meta"}

def get_logic_dir_and_json():
    logic_dir = "logic"
    json_path = os.path.join(logic_dir, "logic_scripts.json")
    return logic_dir, json_path

def _update_logic_json():
    logic_dir, json_path = get_logic_dir_and_json()
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
        with open(json_path, "w") as f:
            json.dump({}, f, indent=2)
        return
    sorted_nums = sorted(scripts_map.keys())
    final_dict = {}
    for i in sorted_nums:
        final_dict[str(i)] = scripts_map[i]
    with open(json_path, "w") as f:
        json.dump(final_dict, f, indent=2)

def _get_logic_script_name(logic_id: str) -> str:
    logic_dir, json_path = get_logic_dir_and_json()
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

_CANONICAL_TF = {
    "15min":  "15Min",
    "30min":  "30Min",
    "1h":     "1Hour",  "1hour": "1Hour",
    "2h":     "2Hour",  "2hour": "2Hour",
    "4h":     "4Hour",  "4hour": "4Hour",
    "1d":     "1Day",   "1day":  "1Day",
}

def canonical_timeframe(tf: str) -> str:
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

def fetch_candles(
    ticker: str,
    bars: int = 10_000,
    timeframe: str | None = None,
    last_timestamp: pd.Timestamp | None = None
) -> pd.DataFrame:
    if not timeframe:
        timeframe = BAR_TIMEFRAME
    end_dt = datetime.now(tz=pytz.utc)

    # -------- determine start date -----------------------------------------
    if last_timestamp is not None:
        # add one second so we never duplicate the previous bar
        start_dt = (pd.to_datetime(last_timestamp, utc=True) +
                    timedelta(seconds=1))
    else:
        bars_per_day = get_bars_per_day(timeframe)
        required_days = math.ceil((bars / bars_per_day) * 1.25)
        start_dt = end_dt - timedelta(days=required_days)

    logging.info(
        f"[{ticker}] Fetching up-to-date {timeframe} bars "
        f"from {start_dt.isoformat()} to {end_dt.isoformat()} "
        f"(limit={bars})."
    )

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

    # normalise expected columns -------------------------------------------
    for c in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
        if c not in df.columns:
            df[c] = np.nan
    if 'vwap' not in df.columns:
        df['vwap'] = np.nan
    if 'trade_count' in df.columns:
        df.rename(columns={'trade_count': 'transactions'}, inplace=True)
    else:
        df['transactions'] = np.nan

    final_cols = [
        'timestamp', 'open', 'high', 'low', 'close',
        'volume', 'vwap', 'transactions'
    ]
    df = df[final_cols]

    logging.info(f"[{ticker}] Fetched {len(df)} new bar(s).")
    return df

# ---------------------------------------------------------------------------
# FINAL  fetch_candles_plus_features_and_predclose (with rolling-safe features)
# ---------------------------------------------------------------------------
def fetch_candles_plus_features_and_predclose(
    ticker: str,
    bars: int,
    timeframe: str,
    rewrite_mode: str
) -> pd.DataFrame:

    tf_code       = timeframe_to_code(timeframe)
    csv_filename  = f"{ticker}_{tf_code}.csv"
    sentiment_csv = f"{ticker}_sentiment_{tf_code}.csv"

    # helper: usable ML features (excludes predicted_close placeholder)
    def get_features(df: pd.DataFrame):
        exclude = {'predicted_close'}
        return [c for c in POSSIBLE_FEATURE_COLS if c in df.columns and c not in exclude]

    # ───── 1. load historical CSV (if present & not rewriting) ─────────────
    if rewrite_mode == "off" and os.path.exists(csv_filename):
        try:
            df_existing = pd.read_csv(csv_filename, parse_dates=['timestamp'])
            last_ts     = df_existing['timestamp'].max()
            if pd.notna(last_ts) and last_ts.tzinfo is None:
                last_ts = last_ts.tz_localize('UTC')
        except Exception as e:
            logging.error(f"[{ticker}] Problem loading old CSV: {e}")
            df_existing = pd.DataFrame()
            last_ts     = None
    else:
        df_existing = pd.DataFrame()
        last_ts     = None

    # ───── 2. figure out how many fresh bars we need ───────────────────────
    if last_ts is not None:
        now_utc      = datetime.now(pytz.utc)
        bars_per_day = get_bars_per_day(timeframe)
        days_gap     = max(0.0, (now_utc - last_ts).total_seconds() / 86_400)
        bars_needed  = int(math.ceil(days_gap * bars_per_day * 1.10))  # +10 %
    else:
        bars_needed = bars

    if bars_needed == 0:
        logging.info(f"[{ticker}] Data already up-to-date – no new candles.")
        return df_existing.copy()

    # ───── 3. fetch ONLY the missing candles ───────────────────────────────
    df_candles_new = fetch_candles(
        ticker,
        bars=bars_needed,
        timeframe=timeframe,
        last_timestamp=last_ts
    )
    if df_candles_new.empty:
        logging.warning(f"[{ticker}] No new bars fetched.")
        return df_existing.copy()

    df_candles_new['timestamp'] = pd.to_datetime(
        df_candles_new['timestamp'], utc=True)

    # protection in case the API returned duplicates
    if last_ts is not None:
        df_candles_new = df_candles_new[df_candles_new['timestamp'] > last_ts]
        if df_candles_new.empty:
            logging.info(f"[{ticker}] No candles newer than {last_ts}.")
            return df_existing.copy()

    # ───── 4. attach *incremental* sentiment to the new slice ──────────────
    if last_ts is not None:
        news_days = 1
        start_dt  = last_ts
    else:
        news_days = NUM_DAYS_MAPPING.get(BAR_TIMEFRAME, 1650)
        start_dt  = None

    articles_per_day = ARTICLES_PER_DAY_MAPPING.get(BAR_TIMEFRAME, 1)
    news_list = fetch_news_sentiments(
        ticker,
        news_days,
        articles_per_day,
        start_dt=start_dt
    )
    sentiments = assign_sentiment_to_candles(df_candles_new, news_list)
    df_candles_new['sentiment'] = sentiments

    # update / append the *_sentiment_*.csv file
    try:
        if os.path.exists(sentiment_csv):
            df_sent_old = pd.read_csv(sentiment_csv, parse_dates=['timestamp'])
        else:
            df_sent_old = pd.DataFrame(columns=['timestamp', 'sentiment'])

        df_sent_new = pd.DataFrame({
            "timestamp": df_candles_new['timestamp'],
            "sentiment": [f"{s:.15f}" for s in sentiments]
        })
        (pd.concat([df_sent_old, df_sent_new])
           .drop_duplicates(subset=['timestamp'])
           .sort_values('timestamp')
           .to_csv(sentiment_csv, index=False))
    except Exception as e:
        logging.error(f"[{ticker}] Unable to update sentiment CSV: {e}")

    # ───── 5. combine old + new *before* engineering features ──────────────
    df_combined = (pd.concat([df_existing, df_candles_new], ignore_index=True)
                     .drop_duplicates(subset=['timestamp'])
                     .sort_values('timestamp')
                     .reset_index(drop=True))

    # ───── 6. (re-)run feature engineering on the FULL frame ───────────────
    # ensures rolling windows/EMAs see complete history
    df_combined = add_features(df_combined)
    df_combined = compute_custom_features(df_combined)

    for col in df_combined.columns:
        if col != 'timestamp':
            df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')

    # ───── 7. regenerate predicted_close only for the new rows ──────────────
    ml_models     = parse_ml_models()
    has_regressor = any(m in ["xgboost", "forest", "rf", "randomforest",
                              "lstm", "transformer"] for m in ml_models)

    if has_regressor:
        if 'predicted_close' not in df_combined.columns:
            df_combined['predicted_close'] = np.nan

        n_start = len(df_combined) - len(df_candles_new)  # first new index

        if len(df_combined) - n_start > 0:
            logging.info(f"[{ticker}] Rolling ML predicted_close for "
                         f"{len(df_combined) - n_start} new row(s)…")

            for i in range(max(1, n_start), len(df_combined)):
                try:
                    X_train = df_combined.iloc[:i][get_features(df_combined)]
                    y_train = df_combined.iloc[:i]['close']

                    # use the first selected model
                    m = ml_models[0]
                    if m in ["forest", "rf", "randomforest"]:
                        mdl = RandomForestRegressor(
                            n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
                    elif m == "xgboost":
                        mdl = xgb.XGBRegressor(
                            n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
                    elif m in ["lightgbm", "lgbm"]:
                        mdl = lgb.LGBMRegressor(
                            n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
                    elif m == "lstm":
                        mdl = get_single_model(
                            "lstm",
                            input_shape=(X_train.shape[0], X_train.shape[1]),
                            num_features=X_train.shape[1],
                            lstm_seq=X_train.shape[0]
                        )
                    elif m == "transformer":
                        Transformer = get_single_model(
                            "transformer",
                            num_features=X_train.shape[1],
                            lstm_seq=X_train.shape[0])
                        mdl = Transformer(
                            num_features=X_train.shape[1],
                            seq_len=X_train.shape[0])
                    else:
                        mdl = RandomForestRegressor(
                            n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)

                    mdl.fit(X_train, y_train)
                    X_pred = df_combined.iloc[[i-1]][get_features(df_combined)]
                    df_combined.at[i, 'predicted_close'] = mdl.predict(X_pred)[0]
                except Exception as e:
                    logging.error(f"[{ticker}] predicted_close error idx={i}: {e}")
                    df_combined.at[i, 'predicted_close'] = np.nan

    # ───── 8. save & return  ───────────────────────────────────────────────
    try:
        df_combined.to_csv(csv_filename, index=False)
        logging.info(f"[{ticker}] CSV updated – now {len(df_combined)} rows.")
    except Exception as e:
        logging.error(f"[{ticker}] Unable to write {csv_filename}: {e}")

    return df_combined

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

def fetch_news_sentiments(
    ticker: str,
    num_days: int,
    articles_per_day: int,
    start_dt: datetime | None = None
):
    news_list = []

    # ----------------------------------------------------------------------
    if start_dt is not None:
        start_date_news = pd.to_datetime(start_dt, utc=True)
        logging.info(f"[{ticker}] Incremental news pull from {start_date_news}.")
    else:
        start_date_news = datetime.now(timezone.utc) - timedelta(days=num_days)
        logging.info(f"[{ticker}] Full-range news pull (≈{num_days} days).")

    today_dt = datetime.now(timezone.utc)
    total_days = (today_dt.date() - start_date_news.date()).days + 1

    for day_offset in range(total_days):
        current_day = start_date_news + timedelta(days=day_offset)
        if current_day > today_dt:
            break
        next_day = current_day + timedelta(days=1)
        start_str = current_day.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str   = next_day.strftime("%Y-%m-%dT%H:%M:%SZ")

        logging.info(f"[{ticker}]   ↳ {current_day.date()} …")
        try:
            articles = api.get_news(ticker, start=start_str, end=end_str)
            if articles:
                count = min(len(articles), articles_per_day)
                for article in articles[:count]:
                    headline = article.headline or ""
                    summary  = article.summary or ""
                    combined = f"{headline} {summary}"
                    _, _, sentiment_score = predict_sentiment(combined)
                    created_at = article.created_at or current_day
                    news_list.append({
                        "created_at": created_at,
                        "sentiment": sentiment_score,
                        "headline": headline,
                        "summary": summary
                    })
        except Exception as e:
            logging.error(f"Error fetching news for {ticker}: {e}")

    news_list.sort(key=lambda x: x['created_at'])
    logging.info(f"[{ticker}] Total new articles: {len(news_list)}")
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
    df = df.copy()
    logging.info("Adding features (price_change, high_low_range, log_volume)...")
    if 'close' in df.columns and 'open' in df.columns:
        df['price_change'] = df['close'] - df['open']
    if 'high' in df.columns and 'low' in df.columns:
        df['high_low_range'] = df['high'] - df['low']
    if 'volume' in df.columns:
        df['log_volume'] = np.log1p(df['volume'])
    return df

def compute_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    # --- MACD (12,26,9) & classic indicators ---
    if 'close' in df.columns:
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd_line    = ema12 - ema26
        signal_line  = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist    = macd_line - signal_line
        df['macd_line']      = macd_line
        df['macd_signal']    = signal_line
        df['macd_histogram'] = macd_hist

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
        tr_calc    = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr']  = tr_calc.ewm(span=14, adjust=False).mean().fillna(0)

    # --- EMAs (9,21,50,200) ---
    if 'close' in df.columns:
        for span in [9, 21, 50, 200]:
            df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()

    # --- ADX (14) ---
    if all(c in df.columns for c in ['high', 'low', 'close']):
        period = 14
        high   = df['high']
        low    = df['low']
        close  = df['close']

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low  - close.shift(1)).abs()
        tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        up_move   = high.diff()
        down_move = low.shift(1) - low
        plus_dm   = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm  = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        tr_smooth       = tr.rolling(window=period, min_periods=1).sum()
        plus_dm_smooth  = plus_dm.rolling(window=period, min_periods=1).sum()
        minus_dm_smooth = minus_dm.rolling(window=period, min_periods=1).sum()

        plus_di  = 100 * plus_dm_smooth  / tr_smooth.replace(0, np.nan)
        minus_di = 100 * minus_dm_smooth / tr_smooth.replace(0, np.nan)
        dx       = (plus_di.subtract(minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100

        df['adx'] = dx.rolling(window=period, min_periods=1).mean()

    # --- ON-BALANCE VOLUME ---
    if all(c in df.columns for c in ['close', 'volume']):
        df['obv'] = 0.0
        for i in range(1, len(df)):
            prev = df.at[i-1, 'obv']
            if df.at[i, 'close'] > df.at[i-1, 'close']:
                df.at[i, 'obv'] = prev + df.at[i, 'volume']
            elif df.at[i, 'close'] < df.at[i-1, 'close']:
                df.at[i, 'obv'] = prev - df.at[i, 'volume']
            else:
                df.at[i, 'obv'] = prev

    # --- BOLLINGER BANDS (20,2) ---
    if 'close' in df.columns:
        ma20  = df['close'].rolling(window=20, min_periods=1).mean()
        std20 = df['close'].rolling(window=20, min_periods=1).std()
        df['bollinger_upper'] = (ma20 + 2 * std20).fillna(0.0)
        df['bollinger_lower'] = (ma20 - 2 * std20).fillna(0.0)

    # --- LAGGED CLOSES ---
    if 'close' in df.columns:
        for lag in [1, 2, 3, 5, 10]:
            df[f'lagged_close_{lag}'] = df['close'].shift(lag).fillna(df['close'].iloc[0])

    # ========== CUSTOM FEATURES REQUIRED ==========

    # 1. Candle Body Ratio: (close – open) / (high – low)
    if all(x in df.columns for x in ['close', 'open', 'high', 'low']):
        denominator = (df['high'] - df['low']).replace(0, np.nan)
        df['candle_body_ratio'] = ((df['close'] - df['open']) / denominator).replace([np.inf, -np.inf], 0).fillna(0)

    # 2. Wick Dominance: max(upper wick, lower wick) as % of full candle
    if all(x in df.columns for x in ['close', 'open', 'high', 'low']):
        upper_wick = (df[['close', 'open']].max(axis=1))
        lower_wick = (df[['close', 'open']].min(axis=1))
        upper_wick_len = df['high'] - upper_wick
        lower_wick_len = lower_wick - df['low']
        candle_range = (df['high'] - df['low']).replace(0, np.nan)
        wick_dom = np.maximum(upper_wick_len, lower_wick_len) / candle_range
        df['wick_dominance'] = wick_dom.replace([np.inf, -np.inf], 0).fillna(0)

    # 3. Gap vs Previous Close: open - lagged_close_1
    if 'open' in df.columns and 'lagged_close_1' in df.columns:
        df['gap_vs_prev'] = df['open'] - df['lagged_close_1']

    # 4. Z-score for volume, ATR, RSI (window=30)
    for feat, z_feat in [('volume', 'volume_zscore'), ('atr', 'atr_zscore'), ('rsi', 'rsi_zscore')]:
        if feat in df.columns:
            roll = df[feat].rolling(window=30, min_periods=10)
            mean = roll.mean()
            std = roll.std().replace(0, np.nan)
            df[z_feat] = ((df[feat] - mean) / std).fillna(0)

    # 5. ADX Regime flag: 1 if ADX>30 (trending), else 0
    df['adx_trend'] = (df.get('adx', 0) > 30).astype(int)

    # 6A. MACD crossing (signal): 1 if crossing up (MACD crosses above signal), -1 if crossing below, 0 otherwise
    if 'macd_line' in df.columns and 'macd_signal' in df.columns:
        macd_cross = np.where(
            (df['macd_line'].shift(1) < df['macd_signal'].shift(1)) &
            (df['macd_line'] >= df['macd_signal']), 1,
            np.where(
                (df['macd_line'].shift(1) > df['macd_signal'].shift(1)) &
                (df['macd_line'] <= df['macd_signal']), -1, 0
            )
        )
        df['macd_cross'] = macd_cross

        # 6B. MACD histogram sign change
        macd_hist_flip = np.where(
            (df['macd_histogram'].shift(1) * df['macd_histogram']) < 0,
            1, 0
        )
        df['macd_hist_flip'] = macd_hist_flip
    else:
        df['macd_cross'] = 0
        df['macd_hist_flip'] = 0

    # 7. Day of week (0=Mon..4=Fri, NaN for missing timestamp)
    if 'timestamp' in df.columns:
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek.fillna(-1).astype(int)

    # 8. Days since last N-bar high/low (window=14)
    N = 14  # lookback for swing
    if 'high' in df.columns:
        highs = df['high'].rolling(window=N, min_periods=1)
        last_high = highs.apply(lambda x: np.argmax(x[::-1]), raw=True)
        df['days_since_high'] = last_high
    if 'low' in df.columns:
        lows = df['low'].rolling(window=N, min_periods=1)
        last_low = lows.apply(lambda x: np.argmax(x[::-1]), raw=True)
        df['days_since_low'] = last_low

    return df

# -------------------------------
# 8. Enhanced Train & Predict Pipeline
# -------------------------------
def train_and_predict(df: pd.DataFrame, return_model_stack=False):
    # 1. Convert sentiment column if needed
    if 'sentiment' in df.columns and df["sentiment"].dtype == object:
        try:
            df["sentiment"] = df["sentiment"].astype(float)
        except Exception as e:
            logging.error(f"Cannot convert sentiment column to float: {e}")
            return None

    # 2. Feature engineering on raw data
    df_feat = add_features(df)
    df_feat = compute_custom_features(df_feat)

    # 3. Determine which columns to use as features
    available_cols = [c for c in POSSIBLE_FEATURE_COLS if c in df_feat.columns]

    # 4. Ensure we have price data
    if 'close' not in df_feat.columns:
        logging.error("No 'close' in DataFrame, cannot create target.")
        return None

    # 5. Build df_full (including the newest row with NaN target)
    df_full = df_feat.copy()
    df_full['target'] = df_full['close'].shift(-1)

    # 6. Build df_train by dropping the row(s) without a valid target
    df_train = df_full.dropna(subset=['target']).copy()
    if len(df_train) < 70:
        logging.error("Not enough rows after shift to train. Need more candles.")
        return None

    # 7. Prepare training feature matrix and target vector
    X = df_train[available_cols]
    y = df_train['target']
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # 8. NaN/Inf diagnostics for downstream model choice
    nan_counts    = X.isna().sum()
    inf_counts    = X.applymap(np.isinf).sum()
    target_nan    = y.isna().sum()
    target_inf    = np.isinf(y).sum()
    logging.info(f"Features NaN per column:\n{nan_counts[nan_counts>0].to_dict()}")
    logging.info(f"Features Inf per column:\n{inf_counts[inf_counts>0].to_dict()}")
    logging.info(f"Target NaNs: {target_nan}, Target INFs: {target_inf}")

    if nan_counts.any() or inf_counts.any() or target_nan > 0 or target_inf > 0:
        logging.error("→ Skipping LSTM/Transformer due to NaN/Inf above.")
        use_transformer = False
    else:
        use_transformer = True

    # 9. Extract the very last candle's features for any “last-row” prediction
    last_row_features = df_full.iloc[-1][available_cols]
    last_row_df       = pd.DataFrame([last_row_features], columns=available_cols)
    last_X_np         = np.array(last_row_df)

    # 10. Decide which models to run
    ml_models = parse_ml_models()

    print(df_full)
    # 11. If using sub-logic, hand off the FULL df_full (with newest row) to sub/main.py
    if "sub-vote" in ml_models or "sub-meta" in ml_models:
        mode   = "sub-vote" if "sub-vote" in ml_models else "sub-meta"
        action = call_sub_main(mode, df_full, execution="live")
        logging.info(f"Sub-{mode} run_live action: {action}")
        return action  # "BUY", "SELL", or "NONE"
    out_preds = []
    out_names = []

    # Sequence models
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

    regression_model_ids = ["forest", "rf", "randomforest", "xgboost", "lightgbm", "lstm", "transformer"]
    use_meta_model = sorted(ml_models) == sorted(["forest", "xgboost", "lightgbm", "lstm", "transformer"])

    # -- Separate classifier logic --
    if "classifier" in ml_models:
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

            TransformerCls = get_single_model("classifier", num_features=Xtr_np.shape[1])
            tr_cls_model = TransformerCls(num_features=Xtr_np.shape[1], seq_len=seq_len)
            opt = torch.optim.Adam(tr_cls_model.parameters(), lr=0.0005)
            loss_fn = torch.nn.CrossEntropyLoss()
            tr_cls_model.train()
            for epoch in range(30):
                opt.zero_grad()
                out_logits = tr_cls_model(Xtr_win_torch)
                if torch.isnan(out_logits).any():
                    logging.error("NaN outputs in Classifier! Skipping.")
                    break
                loss = loss_fn(out_logits, ytr_win_torch)
                if torch.isnan(loss):
                    logging.error("NaN loss in Classifier! Skipping fit.")
                    break
                loss.backward()
                torch.nn.utils.clip_grad_norm_(tr_cls_model.parameters(), max_norm=2.0)
                opt.step()
                if epoch % 8 == 0:
                    logging.info(f"Classifier (epoch {epoch}): train loss = {loss.item():.5f}")
            tr_cls_model.eval()
            last_x_seq = X_scaled[-seq_len:,:].reshape(1,seq_len,-1)
            last_seq_torch = torch.tensor(last_x_seq, dtype=torch.float32)
            with torch.no_grad():
                logits = tr_cls_model(last_seq_torch)[0]
                softmax_scores = torch.softmax(logits,dim=0).cpu().numpy()
                cls_dir_pred = softmax_scores[1]-softmax_scores[0]
            if return_model_stack:
                return cls_dir_pred, {"classifier_pred": cls_dir_pred}
            else:
                return cls_dir_pred
        except Exception as e:
            logging.error(f"Classifier failed to train: {e}")
            return None

    # --- If meta model (all 5 regressors selected) ---
    if use_meta_model:
        model_names_for_meta = ["forest", "xgboost", "lightgbm", "lstm", "transformer"]
        N = len(df)
        start_idx = max(seq_len+1, 70)
        preds_dict = {name:[] for name in model_names_for_meta}
        y_true = []
        logging.info(f"Meta model expanding-window walk-forward: {start_idx} ... {N-2}")
        for i in range(start_idx, N-1):
            train_idx = range(i)
            local_X = X.iloc[train_idx]
            local_y = y.iloc[train_idx]
            X_pred_row = X.iloc[[i]]
            # FOREST
            try:
                rf_model_meta = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
                rf_model_meta.fit(local_X, local_y)
                rf_pred = rf_model_meta.predict(X_pred_row)[0]
            except:
                rf_pred = np.nan
            preds_dict["forest"].append(rf_pred)
            # XGBOOST
            try:
                xgb_model_meta = xgb.XGBRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
                xgb_model_meta.fit(local_X, local_y)
                xgb_pred = xgb_model_meta.predict(X_pred_row)[0]
            except:
                xgb_pred = np.nan
            preds_dict["xgboost"].append(xgb_pred)
            # LIGHTGBM
            try:
                lgbm_model_meta = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
                lgbm_model_meta.fit(local_X, local_y)
                lgbm_pred = lgbm_model_meta.predict(X_pred_row)[0]
            except:
                lgbm_pred = np.nan
            preds_dict["lightgbm"].append(lgbm_pred)
            # LSTM
            try:
                if use_transformer and i > seq_len+1:
                    X_lstm_np = local_X.values
                    y_lstm_np = local_y.values
                    X_lstm_win, y_lstm_win = series_to_supervised(X_lstm_np, y_lstm_np, seq_len)
                    lstm_model_meta = get_single_model("lstm", input_shape=(seq_len, X_lstm_np.shape[1]), num_features=X_lstm_np.shape[1], lstm_seq=seq_len)
                    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, verbose=0, restore_best_weights=True)
                    cp = keras.callbacks.ModelCheckpoint("lstm_meta_model.keras", save_best_only=True, monitor="val_loss", mode="min", verbose=0)
                    lstm_model_meta.fit(
                        X_lstm_win, y_lstm_win,
                        epochs=10,
                        batch_size=32,
                        verbose=0,
                        validation_split=0.18,
                        callbacks=[es, cp]
                    )
                    last_seq = local_X.iloc[-seq_len:].values.reshape(1,seq_len,-1)
                    lstm_pred = lstm_model_meta.predict(last_seq, verbose=0)[0][0]
                else:
                    lstm_pred = np.nan
            except:
                lstm_pred = np.nan
            preds_dict["lstm"].append(lstm_pred)
            # TRANSFORMER
            try:
                if use_transformer and i > seq_len+1:
                    scaler = StandardScaler()
                    tr_X = scaler.fit_transform(local_X.values)
                    tr_y = local_y.values
                    Xtr_win, ytr_win = series_to_supervised(tr_X, tr_y, seq_len)
                    TransformerRegMeta = get_single_model("transformer", num_features=tr_X.shape[1])
                    tr_reg_model_meta = TransformerRegMeta(num_features=tr_X.shape[1], seq_len=seq_len)
                    opt = torch.optim.Adam(tr_reg_model_meta.parameters(), lr=0.0008)
                    loss_fn = torch.nn.L1Loss()
                    tr_reg_model_meta.train()
                    Xtr_win_torch = torch.tensor(Xtr_win, dtype=torch.float32)
                    ytr_win_torch = torch.tensor(ytr_win, dtype=torch.float32)
                    for epoch in range(5):
                        opt.zero_grad()
                        loss = loss_fn(tr_reg_model_meta(Xtr_win_torch).squeeze(), ytr_win_torch)
                        loss.backward()
                        opt.step()
                    last_seq = scaler.transform(local_X.iloc[-seq_len:].values).reshape(1,seq_len,-1)
                    last_seq_torch = torch.tensor(last_seq, dtype=torch.float32)
                    tr_pred = tr_reg_model_meta(last_seq_torch).detach().numpy()[0,0]
                else:
                    tr_pred = np.nan
            except:
                tr_pred = np.nan
            preds_dict["transformer"].append(tr_pred)
            # Save true y
            y_true.append(y.iloc[i])
        preds_arr = np.vstack([preds_dict["forest"],
                               preds_dict["xgboost"],
                               preds_dict["lightgbm"],
                               preds_dict["lstm"],
                               preds_dict["transformer"]]).T
        y_true = np.array(y_true)
        valid_mask = ~np.isnan(preds_arr).any(axis=1) & ~np.isnan(y_true)
        preds_arr_valid = preds_arr[valid_mask]
        y_true_valid = y_true[valid_mask]
        if len(y_true_valid) < 5:
            logging.error("Meta model: Not enough valid rows for fitting.")
            pred = np.nanmean([preds_arr_valid[-1,i] for i in range(preds_arr_valid.shape[1]) if not np.isnan(preds_arr_valid[-1,i])])
            if return_model_stack:
                return pred, {k: preds_arr_valid[-1, idx] for idx, k in enumerate(model_names_for_meta)}
            else:
                return pred
        meta_model = RidgeCV().fit(preds_arr_valid[:-1], y_true_valid[:-1])  # last one = test
        pred_stack = preds_arr_valid[-1].reshape(1,-1)
        final_pred = float(meta_model.predict(pred_stack)[0])
        if return_model_stack:
            out = {k: preds_arr_valid[-1, idx] for idx, k in enumerate(model_names_for_meta)}
            out['meta_pred'] = final_pred
            return final_pred, out
        else:
            return final_pred

    # If not meta: just pick single selected regressor
    if "lstm" in ml_models and use_transformer:
        X_lstm_np = np.array(X)
        y_lstm_np = np.array(y)
        X_lstm_win, y_lstm_win = series_to_supervised(X_lstm_np, y_lstm_np, seq_len)
        lstm_model = get_single_model("lstm", input_shape=(seq_len, X_lstm_np.shape[1]), num_features=X_lstm_np.shape[1], lstm_seq=seq_len)
        es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, verbose=0, restore_best_weights=True)
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

    if "xgboost" in ml_models:
        xgb_model = xgb.XGBRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
        xgb_model.fit(X, y)
        xgb_pred = xgb_model.predict(last_row_df)[0]
        out_preds.append(xgb_pred)
        out_names.append("xgb_pred")

    if "lightgbm" in ml_models or "lgbm" in ml_models:
        lgbm_model = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
        lgbm_model.fit(X, y)
        lgbm_pred = lgbm_model.predict(last_row_df)[0]
        out_preds.append(lgbm_pred)
        out_names.append("lgbm_pred")

    if "forest" in ml_models or "rf" in ml_models or "randomforest" in ml_models:
        rf_model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
        rf_model.fit(X, y)
        rf_pred = rf_model.predict(last_row_df)[0]
        out_preds.append(rf_pred)
        out_names.append("rf_pred")

    if "transformer" in ml_models and use_transformer:
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X.values)
            y_scale_mean = y.mean()
            y_scale_std = y.std() if y.std() > 1e-5 else 1.0
            Xtr_np = X_scaled
            ytr_np = ((y.values - y_scale_mean) / y_scale_std)
            if np.isnan(Xtr_np).any() or np.isnan(ytr_np).any():
                use_this_transformer = False
            else:
                use_this_transformer = True
            if use_this_transformer:
                Xtr_win, ytr_win = series_to_supervised(Xtr_np, ytr_np, seq_len)
                Xtr_win_torch = torch.tensor(Xtr_win, dtype=torch.float32)
                ytr_win_torch = torch.tensor(ytr_win, dtype=torch.float32)
                TransformerReg = get_single_model("transformer", num_features=Xtr_np.shape[1])
                tr_reg_model = TransformerReg(num_features=Xtr_np.shape[1], seq_len=seq_len)
                opt = torch.optim.Adam(tr_reg_model.parameters(), lr=0.0005)
                loss_fn = torch.nn.L1Loss()
                tr_reg_model.train()
                for epoch in range(50):
                    opt.zero_grad()
                    y_pred = tr_reg_model(Xtr_win_torch).squeeze()
                    if torch.isnan(y_pred).any():
                        break
                    loss = loss_fn(y_pred, ytr_win_torch)
                    if torch.isnan(loss):
                        break
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(tr_reg_model.parameters(), max_norm=2.0)
                    opt.step()
                tr_reg_model.eval()
                last_x_seq = X_scaled[-seq_len:,:].reshape(1,seq_len,-1)
                last_seq_torch = torch.tensor(last_x_seq, dtype=torch.float32)
                with torch.no_grad():
                    y_pred_scaled = tr_reg_model(last_seq_torch).cpu().numpy()[0,0]
                tr_reg_pred = (y_pred_scaled * y_scale_std) + y_scale_mean
                if not np.isnan(tr_reg_pred) and not np.isinf(tr_reg_pred):
                    out_preds.append(tr_reg_pred)
                    out_names.append("tr_reg_pred")
        except Exception as e:
            logging.error(f"TransformerReg failed to train: {e}")

    # Output
    if len(out_preds)==1:
        if return_model_stack:
            return out_preds[0], {out_names[0]:out_preds[0]}
        else:
            return out_preds[0]
    elif len(out_preds) > 1:
        pred = np.mean(out_preds)
        if return_model_stack:
            return pred, dict(zip(out_names,out_preds))
        else:
            return pred
    else:
        return None

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
    logging.info("BUY: ", ticker)
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
            pos_qty = 0.0

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
    logging.info("SELL: ", ticker)
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
    logging.info("SHORT: ", ticker)
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
            pos_qty = 0.0

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
    logging.info("COVER: ", ticker)
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


def get_current_price() -> float:
    key    = API_KEY
    secret = API_SECRET
    link   = API_BASE_URL
    if not key or not secret:
        print("Error: set ALPACA_API_KEY and ALPACA_API_SECRET in your environment.", file=sys.stderr)
        sys.exit(1)

    api = REST(key, secret, link, api_version="v2")

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
        ml_models = parse_ml_models()
        classifier_stack = {"classifier", "sub-vote", "sub_meta", "sub-meta", "sub_vote"}
        logic_dir, _ = get_logic_dir_and_json()

        # ─── choose module ───────────────────────────────────────────────────
        if classifier_stack & set(ml_models):
            logic_module_name = "classifier"          # logic/classifier.py
        else:
            logic_module_name = _get_logic_script_name(TRADE_LOGIC)

        module_path   = f"{logic_dir}.{logic_module_name}"
        logic_module  = importlib.import_module(module_path)

        real_current  = get_current_price()
        logic_module.run_logic(real_current, predicted_price, ticker)

    except Exception as e:
        logging.error(f"Error dispatching to trade logic '{logic_module_name}': {e}")

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
            df = fetch_candles_plus_features_and_predclose(
                ticker,
                bars=N_BARS,
                timeframe=BAR_TIMEFRAME,
                rewrite_mode=REWRITE
            )
            try:
                df.to_csv(csv_filename, index=False)
                logging.info(f"[{ticker}] Saved candle data (w/out disabled) to {csv_filename}")
            except Exception as e:
                logging.error(f"[{ticker}] Error saving CSV: {e}")
                continue

            if NEWS_MODE == "on":
                df = merge_sentiment_from_csv(df, ticker)
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

        allowed = set(POSSIBLE_FEATURE_COLS) | {"timestamp"}
        df = df.loc[:, [c for c in df.columns if c in allowed]]

        # --- NEW: Check the latest candle condition based on the scheduled NY time ---
        if scheduled_time_ny is not None:
            if not check_latest_candle_condition(df, BAR_TIMEFRAME, scheduled_time_ny):
                logging.info(f"[{ticker}] Latest candle condition not met for timeframe {BAR_TIMEFRAME} at scheduled time {scheduled_time_ny}. Skipping trade.")
                continue
        # ---------------------------------------------------------------------------
        raw_pred = train_and_predict(df)
        if raw_pred is None:
            logging.error(f"[{ticker}] Model training or prediction failed. Skipping trade logic.")
            continue
        
        try:
            pred_close = float(raw_pred)
        except (TypeError, ValueError) as e:
            logging.error(f"[{ticker}] Cannot convert prediction to float: {raw_pred!r} ({e}). Skipping.")
            continue
        
        current_price = float(df.iloc[-1]['close'])
        logging.info(f"[{ticker}] Current Price = {current_price:.2f}, Predicted Next Close = {pred_close:.2f}")
        trade_logic(current_price, pred_close, ticker)


def setup_schedule_for_timeframe(timeframe: str) -> None:

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
                offset_minutes = SENTIMENT_OFFSET_MINUTES
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
    global SHUTDOWN, TICKERS, BAR_TIMEFRAME, N_BARS, NEWS_MODE, TRADE_LOGIC, AI_TICKER_COUNT, AI_TICKERS
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
                df = fetch_candles_plus_features_and_predclose(
                    ticker,
                    bars=N_BARS,
                    timeframe=BAR_TIMEFRAME,
                    rewrite_mode=REWRITE
                )

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
                    df = fetch_candles_plus_features_and_predclose(
                        ticker,
                        bars=N_BARS,
                        timeframe=BAR_TIMEFRAME,
                        rewrite_mode=REWRITE
                    )
                    df.to_csv(csv_filename, index=False)
                    logging.info(f"[{ticker}] Fetched new data + advanced features (minus disabled), saved to {csv_filename}")

                allowed = set(POSSIBLE_FEATURE_COLS) | {"timestamp"}
                df = df.loc[:, [c for c in df.columns if c in allowed]]

                pred_close = train_and_predict(df)
                if pred_close is None:
                    logging.error(f"[{ticker}] No prediction generated.")
                    continue
                current_price = float(df.iloc[-1]['close'])
                logging.info(f"[{ticker}] Current Price={current_price:.2f}, Predicted Next Close={pred_close:.2f}")

        elif cmd == "run-sentiment":
            logging.info("Running sentiment update job...")
            run_sentiment_job(skip_data=skip_data)
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

                    logic_dir, json_path = get_logic_dir_and_json()
                    if not os.path.isfile(json_path):
                        _update_logic_json()

                    with open(json_path, "r") as f:
                        LOGIC_MODULE_MAP = json.load(f) if os.path.getsize(json_path) else {}

                    ml_models = parse_ml_models()
                    if {"classifier", "sub-vote", "sub_meta", "sub-meta", "sub_vote"} & set(ml_models):
                        logic_module_name = "classifier"               # always use classifier.py
                    else:
                        logic_module_name = LOGIC_MODULE_MAP.get(TRADE_LOGIC, "logic_15_forecast_driven")

                    logging.info("Trading logic: " + logic_module_name)
                    try:
                        logic_module = importlib.import_module(f"{logic_dir}.{logic_module_name}")
                    except Exception as e:
                        logging.error(f"Unable to import logic module {logic_module_name}: {e}")
                        continue

                    match_script = re.match(r"^logic_(\d+)_", logic_module_name)
                    logic_num_str = match_script.group(1) if match_script else "unknown"
                    results_dir = "results"
                    if not os.path.exists(results_dir):
                        os.makedirs(results_dir)
                    # Subdir for this logic script number
                    logic_subdir = os.path.join(results_dir, logic_num_str)
                    if not os.path.exists(logic_subdir):
                        os.makedirs(logic_subdir)

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
                            df = fetch_candles_plus_features_and_predclose(
                                ticker,
                                bars=N_BARS,
                                timeframe=BAR_TIMEFRAME,
                                rewrite_mode=REWRITE
                            )
                            df.to_csv(csv_filename, index=False)
                            logging.info(f"[{ticker}] Saved updated CSV with sentiment & features (minus disabled) to {csv_filename} before backtest.")

                        allowed = set(POSSIBLE_FEATURE_COLS) | {"timestamp"}
                        df = df.loc[:, [col for col in df.columns if col in allowed]]

                        if 'close' not in df.columns:
                            logging.error(f"[{ticker}] No 'close' column after feature processing. Cannot backtest.")
                            continue
                        df['target'] = df['close'].shift(-1)
                        df.dropna(subset=['target'], inplace=True)
                        if len(df) <= test_size + 1:
                            logging.error(f"[{ticker}] Not enough rows for backtest split. Need more data than test_size.")
                            continue

                        total_len = len(df)
                        train_end = total_len - test_size
                        if train_end < 1:
                            logging.error(f"[{ticker}] train_end < 1. Not enough data for that test_size.")
                            continue

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

                        if approach in ["simple", "complex"]:
                            ml_models = parse_ml_models()
                            if approach == "simple":
                                train_df = df.iloc[:train_end]
                                test_df  = df.iloc[train_end:]
                                idx_list = list(test_df.index)
                            else: # complex
                                idx_list = list(range(train_end, total_len))
                                test_df = df.iloc[idx_list]
                        
                            if len(df.iloc[:train_end]) < 70:
                                logging.error(f"[{ticker}] Not enough training rows for {approach} approach.")
                                continue
                            if "sub-vote" in ml_models or "sub-meta" in ml_models:
                                mode = "sub-vote" if "sub-vote" in ml_models else "sub-meta"
                                signal_df = call_sub_main(mode, df, execution="backtest")

                                trade_actions = signal_df[signal_df["action"].isin(["BUY", "SELL"])].reset_index(drop=True)
                                predictions = []
                                actuals = []
                                timestamps = []
                                trade_records = []
                                portfolio_records = []
                                start_balance = 10000.0
                                cash = start_balance
                                position_qty = 0
                                avg_entry_price = 0.0

                                df['timestamp'] = pd.to_datetime(df['timestamp'])
                                trade_actions['timestamp'] = pd.to_datetime(trade_actions['timestamp'])
                                df_sorted = df.sort_values('timestamp').reset_index(drop=True)
                                trade_actions_sorted = trade_actions.sort_values('timestamp').reset_index(drop=True)
                                merged_actions = pd.merge_asof(
                                    trade_actions_sorted, df_sorted, on='timestamp', direction="backward"
                                )

                                def record_trade(action, tstamp, shares, curr_price, pl):
                                    trade_records.append({
                                        "timestamp": tstamp,
                                        "action": action,
                                        "shares": shares,
                                        "current_price": curr_price,
                                        "profit_loss": pl
                                    })

                                for idx, row in merged_actions.iterrows():
                                    action = row["action"]
                                    ts = row["timestamp"]
                                    price = row["close"]
                                    
                                    if action == "BUY":
                                        if position_qty < 0:
                                            pl = (avg_entry_price - price) * abs(position_qty)
                                            cash += pl
                                            record_trade("COVER", ts, abs(position_qty), price, pl)
                                            position_qty = 0
                                            avg_entry_price = 0.0
                                        shares_to_buy = int(cash // price)
                                        if shares_to_buy > 0:
                                            position_qty = shares_to_buy
                                            avg_entry_price = price
                                            record_trade("BUY", ts, shares_to_buy, price, None)
                                    elif action == "SELL":
                                        if position_qty > 0:
                                            pl = (price - avg_entry_price) * position_qty
                                            cash += pl
                                            record_trade("SELL", ts, position_qty, price, pl)
                                            position_qty = 0
                                            avg_entry_price = 0.0
                                    val = cash + (position_qty * (price - avg_entry_price) if position_qty > 0 else 0)
                                    portfolio_records.append({"timestamp": ts, "portfolio_value": val})

                                results_dir = os.path.join("sub", "sub-results")
                                if not os.path.exists(results_dir):
                                    os.makedirs(results_dir)

                            # ----------- ALL OTHER ML MODES -----------
                            else:
                                pred_stack = []
                                for i_, row_idx in enumerate(idx_list):
                                    if approach == "simple":
                                        sub_df = pd.concat([train_df, test_df.iloc[:i_+1]], axis=0, ignore_index=True)
                                    else:
                                        sub_df = df.iloc[:row_idx]
                                    pred_close = train_and_predict(sub_df)
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
                                        candles=test_df if approach == "simple" else df.iloc[idx_list]
                                    )

                                    # --- Prevent duplicate trades ---
                                    if action == "BUY":
                                        if position_qty > 0:
                                            action = "NONE"
                                    elif action == "SHORT":
                                        if position_qty < 0:
                                            action = "NONE"
                                    elif action == "SELL":
                                        if position_qty <= 0:
                                            action = "NONE"
                                    elif action == "COVER":
                                        if position_qty >= 0:
                                            action = "NONE"

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

                        out_csv      = os.path.join(logic_subdir, f"backtest_predictions_{ticker}_{test_size}_{tf_code}_{approach}.csv")
                        out_img      = os.path.join(logic_subdir, f"backtest_predictions_{ticker}_{test_size}_{tf_code}_{approach}.png")
                        trade_log_csv= os.path.join(logic_subdir, f"backtest_trades_{ticker}_{test_size}_{tf_code}_{approach}.csv")
                        port_csv     = os.path.join(logic_subdir, f"backtest_portfolio_{ticker}_{test_size}_{tf_code}_{approach}.csv")
                        out_port_img = os.path.join(logic_subdir, f"backtest_portfolio_{ticker}_{test_size}_{tf_code}_{approach}.png")

                        out_df = pd.DataFrame({
                            'timestamp': timestamps,
                            'actual_close': actuals,
                            'predicted_close': predictions
                        })
                        out_df['timestamp'] = pd.to_datetime(out_df['timestamp'])
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
                        plt.savefig(out_img)
                        plt.close()
                        logging.info(f"[{ticker}] Saved backtest predictions plot to {out_img}.")

                        trade_log_df = pd.DataFrame(trade_records)
                        if not trade_log_df.empty:
                            trade_log_df.to_csv(trade_log_csv, index=False)
                        logging.info(f"[{ticker}] Saved backtest trade log to {trade_log_csv}.")

                        port_df = pd.DataFrame(portfolio_records)
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
                    df = fetch_candles_plus_features_and_predclose(
                        ticker,
                        bars=N_BARS,
                        timeframe=BAR_TIMEFRAME,
                        rewrite_mode=REWRITE
                    )
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
                train_size = int(len(df) * 0.8)
                imp_accum  = np.zeros(len(features))
                n_models   = 0

                for i in range(train_size, len(df)):
                    X_train = X_all.iloc[:i]
                    y_train = y_all.iloc[:i]
                    X_test  = X_all.iloc[i:i+1]
                    y_test  = y_all.iloc[i]

                    model = RandomForestRegressor(
                        n_estimators=N_ESTIMATORS,
                        random_state=RANDOM_SEED,
                        n_jobs=-1
                    )
                    model.fit(X_train, y_train)

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
            pattern_name = re.compile(r'^[a-z0-9_]+$')
            if not pattern_name.match(user_provided_name):
                logging.error("Invalid script name. Use only lowercase letters, digits, or underscores.")
                continue

            logic_dir, _ = get_logic_dir_and_json()
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
    logging.info("Commands: turnoff, api-test, get-data [timeframe], predict-next [-r], run-sentiment [-r], force-run [-r], backtest <N> [simple|complex] [timeframe?] [-r?], feature-importance [-r], commands, set-tickers, set-timeframe, set-nbars, set-news, trade-logic <1..15>, set-ntickers (Number), ai-tickers, create-script (name)")

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
