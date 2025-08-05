import os

DATA_DIR = "data"

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
    'candle_body_ratio', "transactions",
    'wick_dominance', "price_change",
    'gap_vs_prev', "high_low_range",
    'volume_zscore', "log_volume",
    'atr_zscore', 'rsi_zscore',
    'adx_trend', 'macd_cross',
    'macd_hist_flip', 
    'day_of_week',
    'days_since_high',
    'days_since_low'
]

# paper trading / SF account
API_KEY   = os.getenv("ALPACA_API_KEY", "PKX2YGROYCB3773UTOFP")
API_SECRET= os.getenv("ALPACA_API_SECRET", "Fd2ARJkeU36FPyugrzKKVmK4SUMRHAQK4ifql9UX")
API_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
