from forest import DATA_DIR, ROLLING_CANDLES, BAR_TIMEFRAME, DISABLE_PRED_CLOSE
import re
import os
import pandas as pd
import bot.ml.pipelines as pipelines
from datetime import datetime, timedelta, timezone, date
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.historical import CryptoHistoricalDataClient
import numpy as np
import numpy as np
import logging
import math
import pytz
import alpaca_trade_api as tradeapi
import bot.features.engineering as engineering
import bot.stuffs.greed_index as index
import bot.stuffs.sentiment as sentiment
import forest as forest

api = forest.api

logger = logging.getLogger(__name__)

crypto_client = CryptoHistoricalDataClient()

_WINDOWS_RESERVED = {
    "CON","PRN","AUX","NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

def fs_safe_ticker(ticker: str) -> str:
    s = (ticker or "").strip()

    # Common pairs: BTC/USD -> BTC-USD
    s = s.replace("/", "-").replace("\\", "-")

    # Replace any remaining illegal Windows filename chars + control chars
    s = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", s)

    # Windows can't end filenames with dot or space
    s = s.rstrip(" .")

    if not s:
        s = "UNKNOWN"

    # Avoid reserved device names
    if s.upper() in _WINDOWS_RESERVED:
        s = "_" + s

    return s

def timeframe_subdir(tf_code: str) -> str:
    """Return the directory path for a given timeframe code, creating it if needed."""

    path = os.path.join(DATA_DIR, tf_code)
    os.makedirs(path, exist_ok=True)
    return path

def candle_csv_path(ticker: str, tf_code: str) -> str:
    """Build the candle CSV path for ticker/timeframe inside its subdirectory."""

    return os.path.join(timeframe_subdir(tf_code), f"{ticker}_{tf_code}.csv")


def sentiment_csv_path(ticker: str, tf_code: str) -> str:
    """Build the sentiment CSV path for ticker/timeframe inside its subdirectory."""

    return os.path.join(timeframe_subdir(tf_code), f"{ticker}_sentiment_{tf_code}.csv")


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

def normalize_timestamp_utc(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    """
    Ensure df[col] is timezone-aware UTC datetime, and remove unparseable timestamps.
    This prevents tz-mixed sorting bugs and NaT rows accidentally surviving rolling trims.
    """
    if df is None or df.empty or col not in df.columns:
        return df if df is not None else pd.DataFrame()

    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    df = df[df[col].notna()]
    return df

def limit_df_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trim DataFrame to keep the most recent ROLLING_CANDLES rows.

    Key guarantees:
    - timestamps are normalized to UTC (prevents tz-aware/naive ordering bugs)
    - duplicates on timestamp keep the *latest* row (prevents old candle overwriting new)
    - rolling keeps the newest timestamps
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df

    if ROLLING_CANDLES <= 0:
        # Still normalize + dedupe for correctness, but do not trim.
        if "timestamp" in df.columns:
            df = normalize_timestamp_utc(df, "timestamp")
            df = df.sort_values("timestamp", kind="mergesort")
            df = df.drop_duplicates(subset=["timestamp"], keep="last")
        return df.reset_index(drop=True)

    if len(df) <= ROLLING_CANDLES:
        # Still normalize + dedupe to avoid future issues.
        if "timestamp" in df.columns:
            df = normalize_timestamp_utc(df, "timestamp")
            df = df.sort_values("timestamp", kind="mergesort")
            df = df.drop_duplicates(subset=["timestamp"], keep="last")
        return df.reset_index(drop=True)

    df = df.copy()
    if "timestamp" in df.columns:
        df = normalize_timestamp_utc(df, "timestamp")
        df = df.sort_values("timestamp", kind="mergesort")
        df = df.drop_duplicates(subset=["timestamp"], keep="last")

        if len(df) > ROLLING_CANDLES:
            df = df.iloc[-ROLLING_CANDLES:]
    else:
        df = df.iloc[-ROLLING_CANDLES:]

    return df.reset_index(drop=True)

def read_csv_limited(path: str) -> pd.DataFrame:
    """Read a CSV and apply rolling candle limit (with UTC timestamp normalization)."""
    try:
        df = pd.read_csv(path)
    except Exception as e:
        logging.error(f"Error reading CSV {path}: {e}")
        return pd.DataFrame()

    if "timestamp" in df.columns:
        df = normalize_timestamp_utc(df, "timestamp")

    return limit_df_rows(df)

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

def fetch_candles(
    ticker: str,
    bars: int = 10_000,
    timeframe: str | None = None,
    last_timestamp: pd.Timestamp | None = None
) -> pd.DataFrame:
    tf_str = canonical_timeframe(timeframe or BAR_TIMEFRAME)

    ALPACA_TF_MAP: dict[str, TimeFrame] = {
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "30Min": TimeFrame(30, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "2Hour": TimeFrame(2, TimeFrameUnit.Hour),
        "4Hour": TimeFrame(4, TimeFrameUnit.Hour),
        "1Day":  TimeFrame(1, TimeFrameUnit.Day),
    }

    is_crypto = "/" in ticker
    end_dt = datetime.now(tz=pytz.utc)

    if last_timestamp is not None:
        start_dt = pd.to_datetime(last_timestamp, utc=True).to_pydatetime() + timedelta(seconds=1)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=pytz.utc)
        else:
            start_dt = start_dt.astimezone(pytz.utc)
    else:
        bars_per_day = get_bars_per_day(tf_str)
        required_days = math.ceil((bars / bars_per_day) * 1.25)
        start_dt = end_dt - timedelta(days=required_days)

    logging.info(
        f"[{ticker}] Fetching up-to-date {tf_str} bars "
        f"from {start_dt.isoformat()} to {end_dt.isoformat()} "
        f"(limit={bars})."
    )

    try:
        if is_crypto:
            if tf_str not in ALPACA_TF_MAP:
                raise ValueError(f"Unsupported crypto timeframe: {tf_str}")
            request = CryptoBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=ALPACA_TF_MAP[tf_str],
                start=start_dt,
                end=end_dt,
                limit=bars
            )
            barset = crypto_client.get_crypto_bars(request)

        else:
            barset = api.get_bars(
                symbol=ticker,
                timeframe=tf_str,
                limit=bars,
                start=start_dt.isoformat(),
                end=end_dt.isoformat(),
                adjustment="all",
                feed="iex",
            )

    except Exception as e:
        logging.error(f"[{ticker}] Error fetching bars: {e}")
        return pd.DataFrame()

    df = pd.DataFrame()
    if hasattr(barset, "df"):
        df = barset.df
    elif isinstance(barset, (list, tuple)):
        df = pd.DataFrame(barset)

    if df.empty:
        logging.warning(f"[{ticker}] No data returned while fetching bars.")
        return pd.DataFrame()

    if isinstance(df.index, pd.MultiIndex):
        lvl = "symbol" if "symbol" in df.index.names else 0
        try:
            df = df.xs(ticker, level=lvl)
        except Exception:
            pass

    df = df.reset_index()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    else:
        df["timestamp"] = pd.NaT

    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            df[c] = np.nan

    if "vwap" not in df.columns:
        df["vwap"] = np.nan

    if "trade_count" in df.columns and "transactions" not in df.columns:
        df.rename(columns={"trade_count": "transactions"}, inplace=True)
    elif "transactions" not in df.columns:
        df["transactions"] = np.nan

    final_cols = ["timestamp", "open", "high", "low", "close", "volume", "vwap", "transactions"]
    df = df[final_cols]

    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    logging.info(f"[{ticker}] Fetched {len(df)} new bar(s).")
    return df

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

def fetch_candles_plus_features(
    ticker: str,
    bars: int,
    timeframe: str,
    rewrite_mode: str
) -> pd.DataFrame:
    """
    Fetches/updates candles + sentiment + features, then (optionally) rolls predicted_close
    via roll_predicted_close(). Keeps the same DISABLE_PRED_CLOSE gating and final drop logic.
    """
    tf_code       = timeframe_to_code(timeframe)
    ticker_fs     = fs_safe_ticker(ticker)
    csv_filename  = candle_csv_path(ticker_fs, tf_code)
    sentiment_csv = sentiment_csv_path(ticker_fs, tf_code)

    # keep the same "drop at the end" behavior, but apply it consistently to early returns too
    def _finalize(df: pd.DataFrame) -> pd.DataFrame:
        if DISABLE_PRED_CLOSE:
            df = df.drop(columns=["predicted_close"], errors="ignore")
        return df

    # ───── 1. load historical CSV (if present & not rewriting) ─────────────
    if rewrite_mode == "off" and os.path.exists(csv_filename):
        try:
            df_existing = read_csv_limited(csv_filename)
            if not df_existing.empty and "timestamp" in df_existing.columns:
                df_existing = normalize_timestamp_utc(df_existing, "timestamp")
                last_ts = df_existing["timestamp"].max()
                if pd.isna(last_ts):
                    last_ts = None
                else:
                    last_ts = pd.Timestamp(last_ts)
                    if last_ts.tzinfo is None:
                        last_ts = last_ts.tz_localize("UTC")
                    else:
                        last_ts = last_ts.tz_convert("UTC")
            else:
                last_ts = None
        except Exception as e:
            logging.error(f"[{ticker}] Problem loading old CSV: {e}")
            df_existing = pd.DataFrame()
            last_ts = None
    else:
        df_existing = pd.DataFrame()
        last_ts = None

    first_missing_pred_idx = None
    if "predicted_close" in df_existing.columns:
        na_mask = df_existing["predicted_close"].isna().to_numpy()
        if na_mask.any():
            first_missing_pred_idx = int(np.argmax(na_mask))

    # ───── 2. figure out how many fresh bars we need ───────────────────────
    if last_ts is not None:
        now_utc      = datetime.now(pytz.utc)
        bars_per_day = get_bars_per_day(timeframe)
        days_gap     = max(0.0, (now_utc - last_ts).total_seconds() / 86_400)
        bars_needed  = int(math.ceil(days_gap * bars_per_day * 1.10))  # +10 %
    else:
        bars_needed = bars

    have_preds = (
        "predicted_close" in df_existing.columns
        and df_existing["predicted_close"].notna().all()
    )

    if bars_needed == 0 and have_preds:
        logging.info(f"[{ticker}] Data up-to-date and preds present – skipping update.")
        return _finalize(df_existing.copy())

    df_candles_new = fetch_candles(
        ticker,
        bars=bars_needed,
        timeframe=timeframe,
        last_timestamp=last_ts
    )

    has_new = not df_candles_new.empty
    is_backfill_only = (not has_new) and (first_missing_pred_idx is not None)

    # no new bars AND nothing to backfill → exit early
    if not has_new and not is_backfill_only:
        logging.info(f"[{ticker}] No new bars fetched and no missing predicted_close.")
        return _finalize(df_existing.copy())

    # normalize timestamps only if we actually have new rows
    if has_new and "timestamp" in df_candles_new.columns:
        df_candles_new = normalize_timestamp_utc(df_candles_new, "timestamp")

    # if we do have new rows, keep only those newer than last_ts
    if has_new and last_ts is not None and "timestamp" in df_candles_new.columns:
        df_candles_new = df_candles_new[df_candles_new["timestamp"] > last_ts]
        has_new = not df_candles_new.empty
        if not has_new and not is_backfill_only:
            logging.info(f"[{ticker}] No candles newer than {last_ts}.")
            return _finalize(df_existing.copy())

    if has_new or is_backfill_only:
        try:
            df_existing, df_candles_new = index.attach_greed_index(ticker, df_existing, df_candles_new)
        except Exception as e:
            logging.error(f"[{ticker}] Failed to attach greed index feature: {e}")

    df_sent_old = pd.DataFrame(columns=["timestamp", "sentiment", "news_count"])
    last_sentiment_known = None
    if os.path.exists(sentiment_csv):
        try:
            df_sent_old = pd.read_csv(sentiment_csv, parse_dates=["timestamp"])
            if "timestamp" in df_sent_old.columns:
                df_sent_old = normalize_timestamp_utc(df_sent_old, "timestamp")

            if "sentiment" in df_sent_old.columns:
                df_sent_old["sentiment"] = pd.to_numeric(df_sent_old["sentiment"], errors="coerce")
                series = df_sent_old["sentiment"].dropna()
                if not series.empty:
                    last_sentiment_known = float(series.iloc[-1])
        except Exception as e:
            logging.error(f"[{ticker}] Problem loading old sentiment CSV: {e}")
            df_sent_old = pd.DataFrame(columns=["timestamp", "sentiment", "news_count"])

    # ───── 4) attach incremental sentiment only when adding new candles ─────
    if has_new:
        if last_ts is not None:
            news_days = 1
            start_dt = last_ts
        else:
            news_days = NUM_DAYS_MAPPING.get(BAR_TIMEFRAME, 1650)
            start_dt = None

        articles_per_day = ARTICLES_PER_DAY_MAPPING.get(BAR_TIMEFRAME, 1)
        candle_days: set[date] = set()
        for ts in df_candles_new["timestamp"]:
            stamp = pd.Timestamp(ts)
            if stamp.tzinfo is None:
                stamp = stamp.tz_localize("UTC")
            else:
                stamp = stamp.tz_convert("UTC")
            candle_days.add(stamp.date())

        news_list = sentiment.fetch_news_sentiments(
            ticker,
            news_days,
            articles_per_day,
            start_dt=start_dt,
            days_of_interest=candle_days
        )

        sentiments, news_counts = sentiment.assign_sentiment_to_candles(
            df_candles_new,
            news_list,
            last_sentiment_fallback=last_sentiment_known
        )
        df_candles_new["sentiment"] = sentiments
        df_candles_new["news_count"] = news_counts

    if has_new:
        try:
            df_sent_new = pd.DataFrame({
                "timestamp": df_candles_new["timestamp"],
                "sentiment": [f"{s:.15f}" for s in sentiments],
                "news_count": news_counts
            })

            df_sent_all = pd.concat([df_sent_old, df_sent_new], ignore_index=True)
            df_sent_all = normalize_timestamp_utc(df_sent_all, "timestamp")
            df_sent_all = df_sent_all.sort_values("timestamp", kind="mergesort")
            df_sent_all = df_sent_all.drop_duplicates(subset=["timestamp"], keep="last")
            df_sent_all.to_csv(sentiment_csv, index=False)
        except Exception as e:
            logging.error(f"[{ticker}] Unable to update sentiment CSV: {e}")

    # ───── 5) combine (stable sort + keep newest duplicates) ───────────────
    df_combined = pd.concat([df_existing, df_candles_new], ignore_index=True, sort=False)

    if "timestamp" in df_combined.columns:
        df_combined = normalize_timestamp_utc(df_combined, "timestamp")
        df_combined = df_combined.sort_values("timestamp", kind="mergesort")
        df_combined = df_combined.drop_duplicates(subset=["timestamp"], keep="last")
        df_combined = df_combined.reset_index(drop=True)
    else:
        df_combined = df_combined.reset_index(drop=True)

    # 6) features
    df_combined = engineering.add_features(df_combined)
    df_combined = engineering.compute_custom_features(df_combined)

    for col in df_combined.columns:
        if col != "timestamp":
            df_combined[col] = pd.to_numeric(df_combined[col], errors="coerce")

    if "predicted_close" not in df_combined.columns:
        df_combined["predicted_close"] = np.nan

    # 7) predicted_close (moved out)
    first_new_ts = None
    if has_new and "timestamp" in df_candles_new.columns and not df_candles_new.empty:
        first_new_ts = df_candles_new["timestamp"].min()

    df_combined = pipelines.roll_predicted_close(
        ticker=ticker,
        df=df_combined,
        first_missing_pred_idx=first_missing_pred_idx,
        first_new_ts=first_new_ts,
    )

    # ───── 8. trim, save & return  ─────────────────────────────────────────
    df_combined = limit_df_rows(df_combined)

    try:
        df_combined.to_csv(csv_filename, index=False)
        logging.info(f"[{ticker}] CSV updated – now {len(df_combined)} rows.")
    except Exception as e:
        logging.error(f"[{ticker}] Unable to write {csv_filename}: {e}")

    return _finalize(df_combined)