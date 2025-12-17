import config
import os
import forest
BAR_TIMEFRAME = config.BAR_TIMEFRAME
TIMEFRAME_MAP = {
    "4Hour": "H4", "2Hour": "H2", "1Hour": "H1",
    "30Min": "M30", "15Min": "M15"
}
CONVERTED_TIMEFRAME = TIMEFRAME_MAP.get(BAR_TIMEFRAME, BAR_TIMEFRAME)
FEATURES = forest.POSSIBLE_FEATURE_COLS

DATA_DIR = "data"
def timeframe_subdir(tf_code: str) -> str:
    """Return the directory path for a given timeframe code, creating it if needed."""

    path = os.path.join(DATA_DIR, tf_code)
    os.makedirs(path, exist_ok=True)
    return path

def get_csv_filename(ticker):
    ticker_fs = forest.fs_safe_ticker(ticker)
    return os.path.join(timeframe_subdir(CONVERTED_TIMEFRAME), f"{ticker_fs}_{CONVERTED_TIMEFRAME}.csv")