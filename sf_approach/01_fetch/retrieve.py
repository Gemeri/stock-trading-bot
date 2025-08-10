import pandas as pd
import datetime
import pytz
import logging
import math
from datetime import datetime, timedelta, timezone
from time_util import get_bars_per_day

from config import BAR_TIMEFRAME

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