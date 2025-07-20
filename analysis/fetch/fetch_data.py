import pandas as pd
import os
import pytz
import logging
import math
from datetime import datetime, timedelta, timezone
from time_util import timeframe_to_code, get_bars_per_day
from config import DATA_DIR
from alpaca_client import fetch_candles, fetch_news
from sentiment import predict_fin_sentiment, assign_sentiment_to_candles
from custom_features import add_features, compute_custom_features

ARTICLES_PER_DAY_MAPPING = {
    "1Day": 1,
    "4Hour": 2,
    "2Hour": 4,
    "1Hour": 8,
    "30Min": 16,
    "15Min": 32
}

NY_TZ = pytz.timezone("America/New_York")

def fetch_candles_plus_features(
    ticker: str,
    bars: int,
    timeframe: str,
    rewrite_mode: str = "",
    fetch_sentiment: bool = False
) -> pd.DataFrame:

    tf_code       = timeframe_to_code(timeframe)
    csv_filename  = os.path.join(DATA_DIR, f"{ticker}_{tf_code}.csv")
    sentiment_csv = os.path.join(DATA_DIR, f"{ticker}_sentiment_{tf_code}.csv")

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

    if last_ts is not None:
        df_candles_new = df_candles_new[df_candles_new['timestamp'] > last_ts]
        if df_candles_new.empty:
            logging.info(f"[{ticker}] No candles newer than {last_ts}.")
            return df_existing.copy()

    # we calculate the oldest candle ts from the dataset we have just downloaded
    oldest_candle_ts = df_candles_new['timestamp'].min()
    days_ago = (datetime.now(NY_TZ) - oldest_candle_ts).days

    if fetch_sentiment is True:
    
        # ───── 4. attach incremental sentiment to the new slice ──────────────
        if last_ts is not None:
            news_days = 1
            start_dt  = last_ts
        else:
            news_days = days_ago
            start_dt  = None

        articles_per_day = ARTICLES_PER_DAY_MAPPING.get(timeframe, 1)
        news_list = fetch_news(
            ticker,
            news_days,
            articles_per_day,
            start_dt=start_dt
        )

        # we add the sentiment for each piece of news
        for news_item in news_list:
            combined = f"{news_item['headline']} {news_item['summary']}"
            _, _, sentiment_score = predict_fin_sentiment(combined)
            news_item['sentiment'] = sentiment_score


        sentiments = assign_sentiment_to_candles(df_candles_new, news_list)
        df_candles_new['sentiment'] = sentiments

        try:
            if os.path.exists(sentiment_csv):
                df_sent_old = pd.read_csv(sentiment_csv, parse_dates=['timestamp'])
            else:
                df_sent_old = pd.DataFrame(columns=['timestamp', 'sentiment'])

            df_sent_new = pd.DataFrame({
                "timestamp": df_candles_new['timestamp'],
                "sentiment": sentiments
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

    # ────── 6. run feature engineering on the FULL frame ───────────────
    df_combined = add_features(df_combined)
    df_combined = compute_custom_features(df_combined)

    for col in df_combined.columns:
        if col != 'timestamp':
            df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')

    # ───── 7. save & return  ───────────────────────────────────────────────
    try:
        df_combined.to_csv(csv_filename, index=False)
        logging.info(f"[{ticker}] CSV updated – now {len(df_combined)} rows.")
    except Exception as e:
        logging.error(f"[{ticker}] Unable to write {csv_filename}: {e}")

    return df_combined