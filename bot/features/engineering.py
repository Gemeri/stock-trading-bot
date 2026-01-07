import logging
import pandas as pd
import numpy as np
logger = logging.getLogger(__name__)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    logging.info("Adding features (price_change, high_low_range)...")
    if 'close' in df.columns and 'open' in df.columns:
        df['price_change'] = df['close'] - df['open']
    if 'high' in df.columns and 'low' in df.columns:
        df['high_low_range'] = df['high'] - df['low']
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

    # --- ROC ---
    if 'close' in df.columns:
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

        # Temporary (not written to df)
        bollinger_upper = (ma20 + 2 * std20).fillna(0.0)
        bollinger_lower = (ma20 - 2 * std20).fillna(0.0)

        # Only this is written to df
        band_width = (bollinger_upper - bollinger_lower).replace(0, np.nan)
        df['bollinger_percB'] = ((df['close'] - bollinger_lower) / band_width).fillna(0.0)

        df['returns_1'] = df['close'].pct_change()
        df['returns_3'] = df['close'].pct_change(3)
        df['returns_5'] = df['close'].pct_change(5)
        df['std_5'] = df['close'].rolling(5).std()
        df['std_10'] = df['close'].rolling(10).std()

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

    if 'news_count' in df.columns:
        counts = pd.to_numeric(df['news_count'], errors='coerce').fillna(0.0)
        df['news_count'] = counts.round().astype(int)
        log_counts = np.log1p(df['news_count'].astype(float))
        window = 63
        roll = log_counts.rolling(window=window, min_periods=window)
        mean = roll.mean()
        std = roll.std().clip(lower=1e-6)
        z = (log_counts - mean) / std
        z = z.where(roll.count() >= window, 0.0)
        z = z.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        df['news_volume_z'] = z.clip(-3.0, 3.0)
    else:
        df['news_volume_z'] = 0.0

    if 'sentiment' in df.columns:
        # raw one-bar change Δ_t
        delta = df['sentiment'].diff()

        # rolling standardization over window w
        roll = delta.rolling(window=63, min_periods=63)
        mean = roll.mean()
        std  = roll.std().replace(0, np.nan)

        z = (delta - mean) / std
        # Before we have enough history, set 0; also clean inf/nan
        z = z.where(roll.count() >= 63, 0.0)
        z = z.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        # clip extremes
        df['d_sentiment'] = z.clip(-float(3.0), float(3.0))
    else:
        df['d_sentiment'] = 0.0

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

    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'], errors='coerce')

        df['month'] = ts.dt.month.astype('Int64')

        hour_raw = ts.dt.hour
        dow_raw  = ts.dt.dayofweek

        hour_safe = hour_raw.fillna(0)
        dow_safe  = dow_raw.fillna(0)

        df['hour_sin'] = np.sin(2 * np.pi * hour_safe / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour_safe / 24)
        df['day_of_week_sin'] = np.sin(2 * np.pi * dow_safe / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * dow_safe / 7)

        df = df.drop(columns=['day_of_week', 'hour'], errors='ignore')


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

    return df.fillna(0.0)

def series_to_supervised(Xvalues: np.ndarray,
                         yvalues: np.ndarray,
                         seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    n_samples, n_features = Xvalues.shape
    if n_samples <= seq_length:
        return np.empty((0, seq_length, n_features)), np.empty(0)
    Xs = np.stack([
        Xvalues[i : i + seq_length, :]
        for i in range(n_samples - seq_length)
    ])
    ys = yvalues[seq_length:]
    return Xs, ys
