import pandas as pd
import datetime
import pytz
import math
import logging
import numpy as np

import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta, timezone
from config import API_KEY, API_SECRET, API_BASE_URL
from time_util import get_bars_per_day


api = tradeapi.REST(API_KEY, API_SECRET, API_BASE_URL, api_version='v2')

def fetch_candles(
    ticker: str,
    bars: int = 10_000,
    timeframe: str | None = None,
    last_timestamp: pd.Timestamp | None = None
) -> pd.DataFrame:
    if not timeframe:
        timeframe = '4Hour'
    end_dt = datetime.now(tz=pytz.utc)

    # -------- determine start date -----------------------------------------
    if last_timestamp is not None:
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

def fetch_news(
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
                    # combined = f"{headline} {summary}"
                    #_, _, sentiment_score = predict_sentiment(combined)
                    created_at = article.created_at or current_day
                    news_list.append({
                        "created_at": created_at,
                        #"sentiment": sentiment_score,
                        "headline": headline,
                        "summary": summary
                    })
        except Exception as e:
            logging.error(f"Error fetching news for {ticker}: {e}")

    news_list.sort(key=lambda x: x['created_at'])
    logging.info(f"[{ticker}] Total new articles: {len(news_list)}")
    return news_list