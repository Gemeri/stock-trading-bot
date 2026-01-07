import pandas as pd
import requests
import os
import datetime
import logging
logger = logging.getLogger(__name__)
import numpy as np
from forest import DATA_DIR


CNN_FEAR_GREED_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
ALTERNATIVE_FNG_CSV_URL = "https://api.alternative.me/fng/?limit=0&format=csv"

def _load_primary_greed_series() -> pd.DataFrame:
    """Fetch the CNN Fear & Greed time series from production.dataviz.cnn.io,
    robust to 418/anti-bot blocks and minor schema variations."""
    import time

    empty = pd.DataFrame(columns=["timestamp", "value"])

    headers = {
        # CNN’s CDN often blocks "botty" clients; use a realistic UA + typical headers.
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://money.cnn.com/data/fear-and-greed/",
        "Origin": "https://money.cnn.com",
        "Connection": "keep-alive",
    }

    session = requests.Session()
    url = CNN_FEAR_GREED_URL  # expected: https://production.dataviz.cnn.io/index/fearandgreed/graphdata

    payload = None
    for attempt in range(3):
        try:
            # small cache-buster on retries
            params = {"_": int(time.time())} if attempt else None
            resp = session.get(url, headers=headers, timeout=12, params=params)
            if resp.status_code == 418:
                # backoff and try again
                time.sleep(0.6 * (attempt + 1))
                continue
            resp.raise_for_status()
            payload = resp.json()
            break
        except Exception as e:
            logging.error(f"Failed to fetch CNN fear & greed data (try {attempt+1}/3): {e}")
            if attempt == 2:
                return empty
            time.sleep(0.6 * (attempt + 1))

    if not isinstance(payload, dict):
        return empty

    # Historical series can appear either as a dict containing "data": [{x,y},...]
    # or directly as a list of {x,y}/{"timestamp","score"}/etc.
    hist = payload.get("fear_and_greed_historical", [])
    if isinstance(hist, dict):
        data = hist.get("data", hist.get("timeline", []))
    else:
        data = hist

    if not isinstance(data, list) or not data:
        return empty

    rows = []
    for item in data:
        if not isinstance(item, dict):
            continue
        # Timestamp field candidates
        ts_val = item.get("x", item.get("timestamp", item.get("date")))
        # Value field candidates
        v_val = item.get("y", item.get("score", item.get("value")))
        # Parse timestamp
        ts_parsed = pd.NaT
        if ts_val is not None:
            try:
                if isinstance(ts_val, (int, float)) and not isinstance(ts_val, bool):
                    # Heuristic: ms vs s
                    unit = "ms" if float(ts_val) > 1e11 else "s"
                    ts_parsed = pd.to_datetime(ts_val, unit=unit, utc=True, errors="coerce")
                else:
                    ts_parsed = pd.to_datetime(ts_val, utc=True, errors="coerce")
            except Exception:
                ts_parsed = pd.NaT

        # Parse value
        v_num = pd.to_numeric(v_val, errors="coerce")

        if pd.notna(ts_parsed) and pd.notna(v_num):
            rows.append((ts_parsed, float(v_num)))

    if not rows:
        return empty

    df = pd.DataFrame(rows, columns=["timestamp", "value"]).dropna()
    if df.empty:
        return empty

    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    return df.reset_index(drop=True)

def _load_local_greed_map() -> dict:
    """Load locally cached fear & greed values keyed by date."""
    path = os.path.join(DATA_DIR, "greed-index.csv")
    if not os.path.exists(path):
        return {}

    try:
        # Be tolerant of ragged rows or stray delimiters.
        df = pd.read_csv(path, on_bad_lines="skip")
    except Exception as e:
        logging.error(f"Failed to read local greed index CSV {path}: {e}")
        return {}

    if df.empty:
        return {}

    date_col = None
    value_col = None
    for col in df.columns:
        lowered = str(col).strip().lower()
        if lowered in {"date", "timestamp"} and date_col is None:
            date_col = col
        elif lowered in {"greed", "value", "index", "fear_and_greed", "fng"} and value_col is None:
            value_col = col

    if date_col is None:
        date_col = df.columns[0]
    if value_col is None:
        value_col = df.columns[-1]

    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["date", value_col])
    if df.empty:
        return {}

    grouped = df.groupby("date")[value_col].mean()
    return {idx: float(val) for idx, val in grouped.items()}

def _load_alt_greed_map() -> dict:
    """Fetch the Alternative.me Fear & Greed (CRYPTO) as a fallback, using JSON (not CSV)."""
    # JSON endpoint: https://api.alternative.me/fng/?limit=0
    # Docs: https://alternative.me/crypto/fear-and-greed-index/  (see API section)
    alt_url = "https://api.alternative.me/fng/?limit=0"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
    }

    try:
        response = requests.get(alt_url, headers=headers, timeout=15)
        response.raise_for_status()
        payload = response.json()
    except Exception as e:
        logging.error(f"Failed to fetch Alternative.me fear & greed JSON: {e}")
        return {}

    data = payload.get("data", []) if isinstance(payload, dict) else []
    if not isinstance(data, list) or not data:
        return {}

    # Build date -> values map, averaging same-day duplicates.
    day_to_vals: dict[datetime.date, list] = {}

    for rec in data:
        if not isinstance(rec, dict):
            continue
        ts = rec.get("timestamp")
        val = rec.get("value")
        ts_num = pd.to_numeric(ts, errors="coerce")
        v_num = pd.to_numeric(val, errors="coerce")
        if pd.isna(ts_num) or pd.isna(v_num):
            continue
        # Alt.me timestamp is seconds since epoch.
        dt = pd.to_datetime(int(ts_num), unit="s", utc=True, errors="coerce")
        if pd.isna(dt):
            continue
        d = dt.date()
        day_to_vals.setdefault(d, []).append(float(v_num))

    if not day_to_vals:
        return {}

    return {d: float(np.mean(vals)) for d, vals in day_to_vals.items()}


def attach_greed_index(ticker, df_existing: pd.DataFrame, df_new: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach greed_index feature to candle data using multiple data sources.

    Uses CNN time series when available; falls back to local CSV (by date) and then
    Alternative.me (JSON) daily values keyed by date. Forward-fills across gaps.

    If ticker contains '/', only Alternative.me (_load_alt_greed_map) is used.
    """
    if (df_existing is None or df_existing.empty) and (df_new is None or df_new.empty):
        return df_existing, df_new

    frames: list[pd.Series] = []
    if df_existing is not None and not df_existing.empty and 'timestamp' in df_existing.columns:
        df_existing['timestamp'] = pd.to_datetime(df_existing['timestamp'], utc=True, errors='coerce')
        frames.append(df_existing['timestamp'])
    if df_new is not None and not df_new.empty and 'timestamp' in df_new.columns:
        df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], utc=True, errors='coerce')
        frames.append(df_new['timestamp'])

    if not frames:
        return df_existing, df_new

    timestamps = pd.concat(frames).dropna().drop_duplicates().sort_values()
    if timestamps.empty:
        return df_existing, df_new

    use_alt_only = isinstance(ticker, str) and ('/' in ticker)

    greed_mapping: dict[pd.Timestamp, float] = {}
    last_value: float | None = None

    if use_alt_only:
        alt_map = _load_alt_greed_map()

        for ts in timestamps:
            value: float | None = None
            date_key = ts.date()
            fallback = alt_map.get(date_key) if alt_map else None

            if fallback is not None and pd.notna(fallback):
                try:
                    value = float(fallback)
                except (TypeError, ValueError):
                    value = None

            if value is None and last_value is not None:
                value = last_value

            greed_mapping[ts] = float(value) if value is not None else np.nan
            if value is not None:
                last_value = value
    else:
        primary_df = _load_primary_greed_series()
        if not primary_df.empty:
            primary_df = primary_df[primary_df['timestamp'] >= timestamps.iloc[0]].reset_index(drop=True)

        primary_times = primary_df['timestamp'].tolist() if not primary_df.empty else []
        primary_values = primary_df['value'].tolist() if not primary_df.empty else []
        primary_idx = 0

        local_map = _load_local_greed_map()
        alt_map = None

        prev_ts: pd.Timestamp | None = None

        for ts in timestamps:
            interval_values: list[float] = []
            while primary_idx < len(primary_times) and primary_times[primary_idx] <= ts:
                candidate_time = primary_times[primary_idx]
                candidate_value = primary_values[primary_idx]
                if (prev_ts is None or candidate_time > prev_ts) and pd.notna(candidate_value):
                    interval_values.append(float(candidate_value))
                primary_idx += 1

            value: float | None = None
            if interval_values:
                value = float(np.mean(interval_values))
            else:
                date_key = ts.date()
                fallback = local_map.get(date_key)
                if fallback is None:
                    if alt_map is None:
                        alt_map = _load_alt_greed_map()
                    fallback = alt_map.get(date_key) if alt_map else None
                if fallback is not None and pd.notna(fallback):
                    try:
                        value = float(fallback)
                    except (TypeError, ValueError):
                        value = None

            if value is None and last_value is not None:
                value = last_value

            greed_mapping[ts] = float(value) if value is not None else np.nan
            if value is not None:
                last_value = value
            prev_ts = ts

    if df_existing is not None and not df_existing.empty and 'timestamp' in df_existing.columns:
        df_existing['greed_index'] = df_existing['timestamp'].map(greed_mapping)
    if df_new is not None and not df_new.empty and 'timestamp' in df_new.columns:
        df_new['greed_index'] = df_new['timestamp'].map(greed_mapping)

    return df_existing, df_new

