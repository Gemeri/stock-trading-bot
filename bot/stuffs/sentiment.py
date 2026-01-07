from transformers import AutoTokenizer, AutoModelForSequenceClassification
import bot.stuffs.candles as candles
import pandas as pd
from datetime import datetime, timedelta, timezone, date
import requests
import math
import logging
import os
import numpy as np
from forest import API_KEY, API_SECRET, DATA_DIR, BAR_TIMEFRAME
logger = logging.getLogger(__name__)

try:
    # ProsusAI FinBERT (financial sentiment)
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    # Resolve label indices dynamically in case the config changes in the future
    _LABEL2ID = getattr(model.config, "label2id", {"positive": 0, "negative": 1, "neutral": 2})
    _POS_IDX = int(_LABEL2ID.get("positive", 0))
    _NEG_IDX = int(_LABEL2ID.get("negative", 1))
except Exception as e:
    tokenizer = None
    model = None
    _LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}
    _POS_IDX, _NEG_IDX = 0, 1
    logging.info(f"Offline / FinBERT load failed: {e}. Sentiment disabled and will return neutral-like values.")

def predict_sentiment(text: str):
    """
    Uses ProsusAI/finbert to produce:
      - sentiment_class: argmax index from model logits
      - confidence_scores: softmax(probabilities) as a float list [P(pos), P(neg), P(neu)] *order resolved via model.config*
      - sentiment_score: P(positive) - P(negative)
    """
    if not text:
        return 2, [0.0, 0.0, 1.0], 0.0  # neutral-ish fallback for empty text

    if tokenizer is None or model is None:
        # Fallback if model couldn't be loaded
        logging.warning("predict_sentiment called while FinBERT is unavailable. Returning neutral-like values.")
        return 2, [0.0, 0.0, 1.0], 0.0

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=1)[0].tolist()

    # FinBERT's config defines indices (resolved above); compute positive-minus-negative
    sentiment_score = probs[_POS_IDX] - probs[_NEG_IDX]
    sentiment_class = outputs.logits.argmax(dim=1).item()
    return sentiment_class, probs, sentiment_score

def fetch_news_sentiments(
    ticker: str,
    num_days: int,
    articles_per_day: int,
    start_dt: datetime | None = None,
    base_url: str = "https://data.alpaca.markets/v1beta1/news",
    days_of_interest: set[date] | None = None,
):
    """
    Pulls Alpaca v1beta1 news via REST (no api.get_news), including full content,
    and feeds FinBERT the combined 'headline + content + summary'.
    """
    news_list = []

    # --------- date range setup ----------
    if days_of_interest:
        day_sequence = sorted(days_of_interest)
        logging.info(f"[{ticker}] News pull limited to {len(day_sequence)} day(s) from candle timestamps.")
    else:
        if start_dt is not None:
            start_date_news = pd.to_datetime(start_dt, utc=True).normalize()
            logging.info(f"[{ticker}] Incremental news pull from {start_date_news.date()} (full days).")
        else:
            start_date_news = (datetime.now(timezone.utc) - timedelta(days=num_days)).replace(hour=0, minute=0, second=0, microsecond=0)
            logging.info(f"[{ticker}] Full-range news pull (≈{num_days} days).")

        today_dt = datetime.now(timezone.utc)
        total_days = (today_dt.date() - start_date_news.date()).days + 1
        day_sequence = [(start_date_news + timedelta(days=offset)).date() for offset in range(total_days)]

    # --------- headers (use keys if available) ----------
    headers = {"accept": "application/json"}

    api_key = API_KEY
    api_secret = API_SECRET

    if api_key and api_secret:
        headers["APCA-API-KEY-ID"] = api_key
        headers["APCA-API-SECRET-KEY"] = api_secret
    else:
        logging.warning("Missing API_KEY or API_SECRET in environment; Alpaca requests may fail (401).")

    session = requests.Session()

    for day_item in day_sequence:
        current_day = datetime.combine(day_item, datetime.min.time(), tzinfo=timezone.utc)
        if current_day > datetime.now(timezone.utc):
            break
        next_day = current_day + timedelta(days=1)

        start_str = current_day.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str   = next_day.strftime("%Y-%m-%dT%H:%M:%SZ")
        logging.info(f"[{ticker}]   ↳ {current_day.date()} …")

        # --------- query Alpaca News (REST) ----------
        per_request_limit = max(int(articles_per_day), 50)
        page_token = None
        fetched_for_day = 0
        news_symbol = ticker
        if "/" in ticker:
            base, quote = ticker.split("/", 1)
            news_symbol = f"{base}{quote}"
    # optionally also try base-only fallback: base
        while True:
            params = {
                "start": start_str,
                "end": end_str,
                "sort": "desc",
                "symbols": news_symbol,
                "limit": per_request_limit,
                "include_content": "true",
                "exclude_contentless": "true",
            }
            if page_token:
                params["page_token"] = page_token

            try:
                resp = session.get(base_url, headers=headers, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                items = []
                if isinstance(data, dict) and "news" in data:
                    items = data["news"]
                elif isinstance(data, list):
                    items = data

                if not items:
                    break

                for article in items:
                    headline = article.get("headline", "") or ""
                    summary  = article.get("summary", "") or ""
                    content  = article.get("content", "") or ""

                    combined = " ".join([seg for seg in (headline, content, summary) if seg]).strip()

                    created_at_raw = article.get("created_at") or article.get("updated_at")
                    created_at = pd.to_datetime(created_at_raw, utc=True, errors="coerce") if created_at_raw else None
                    if created_at is None or pd.isna(created_at):
                        created_at = current_day

                    _, _, sentiment_score = predict_sentiment(combined)

                    news_list.append({
                        "created_at": created_at,
                        "sentiment": sentiment_score,
                        "headline": headline,
                        "summary": summary,
                        "content": content,
                        "url": article.get("url", ""),
                        "source": (article.get("source") or {}).get("name") if isinstance(article.get("source"), dict) else article.get("source", "")
                    })
                    fetched_for_day += 1

                page_token = data.get("next_page_token") if isinstance(data, dict) else None
                if not page_token:
                    break

            except Exception as e:
                logging.error(f"Error fetching news for {ticker}: {e}")
                break

        logging.info(f"[{ticker}]     ↳ fetched {fetched_for_day} article(s) on {current_day.date()}.")

    news_list.sort(key=lambda x: x['created_at'])
    logging.info(f"[{ticker}] Total new articles: {len(news_list)}")
    return news_list

def assign_sentiment_to_candles(
    df: pd.DataFrame,
    news_list: list,
    *,
    last_sentiment_fallback: float | None = None,
):
    logging.info("Assigning sentiment to candles (daily average)...")

    day_to_scores: dict[date, list[float]] = {}
    for article in news_list:
        created_at = pd.to_datetime(article.get('created_at'), utc=True, errors='coerce')
        if created_at is None or pd.isna(created_at):
            continue
        day_key = created_at.date()
        score = article.get('sentiment', 0.0)
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        day_to_scores.setdefault(day_key, []).append(score)

    day_to_avg = {day_key: float(np.mean(scores)) for day_key, scores in day_to_scores.items()}
    day_to_count = {day_key: len(scores) for day_key, scores in day_to_scores.items()}

    sentiments: list[float] = []
    news_counts: list[int] = []
    fallback = None
    if last_sentiment_fallback is not None and not math.isnan(last_sentiment_fallback):
        fallback = float(last_sentiment_fallback)

    for ts in df['timestamp']:
        stamp = pd.Timestamp(ts)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize('UTC')
        else:
            stamp = stamp.tz_convert('UTC')
        day_key = stamp.date()

        if day_key in day_to_avg:
            sentiment = day_to_avg[day_key]
            fallback = sentiment
            count = day_to_count.get(day_key, 0)
        else:
            if fallback is None:
                fallback = 0.0
            sentiment = fallback
            count = 0

        sentiments.append(sentiment)
        news_counts.append(int(count))

    logging.info("Finished assigning sentiment to candles.")
    return sentiments, news_counts

def save_news_to_csv(ticker: str, news_list: list):
    """
    Now also persists 'content' (full article text) for traceability.
    """
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
            "content": item.get("content", ""),     # NEW
            "sentiment": item_sentiment_str
        }
        rows.append(row)

    df_news = pd.DataFrame(rows)
    ticker_fs = candles.fs_safe_ticker(ticker)
    csv_filename = os.path.join(DATA_DIR, f"{ticker_fs}_articles_sentiment.csv")
    df_news.to_csv(csv_filename, index=False)
    logging.info(f"[{ticker}] Saved articles with sentiment to {csv_filename}")

def merge_sentiment_from_csv(df, ticker):
    tf_code = candles.timeframe_to_code(BAR_TIMEFRAME)
    ticker_fs = candles.fs_safe_ticker(ticker)
    sentiment_csv_filename = candles.sentiment_csv_path(ticker_fs, tf_code)
    if not os.path.exists(sentiment_csv_filename):
        logging.error(f"Sentiment CSV {sentiment_csv_filename} not found for ticker {ticker_fs}.")
        return df
    
    try:
        df_sentiment = pd.read_csv(sentiment_csv_filename)
    except Exception as e:
        logging.error(f"Error reading sentiment CSV {sentiment_csv_filename}: {e}")
        return df

    df_sentiment['timestamp'] = pd.to_datetime(df_sentiment['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df_sentiment = (df_sentiment
                    .sort_values('timestamp')
                    .drop_duplicates(subset=['timestamp'], keep='last'))
    if 'sentiment' in df_sentiment.columns:
        df_sentiment['sentiment'] = pd.to_numeric(df_sentiment['sentiment'], errors='coerce')
    if 'news_count' not in df_sentiment.columns:
        df_sentiment['news_count'] = 0
    df_sentiment['news_count'] = pd.to_numeric(df_sentiment['news_count'], errors='coerce').fillna(0).astype(int)

    df = df.merge(df_sentiment[['timestamp', 'sentiment', 'news_count']], on='timestamp', how='left')
    df['sentiment'] = df['sentiment'].fillna(method='ffill').fillna(0.0)
    df['news_count'] = df['news_count'].fillna(0).astype(int)
    df['sentiment'] = df['sentiment'].apply(lambda x: f"{float(x):.15f}")
    logging.info(f"Merged sentiment from {sentiment_csv_filename} into latest data for {ticker}.")
    return df