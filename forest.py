import os
import time
import threading
import pytz
import schedule
import matplotlib
matplotlib.use("Agg")
from datetime import datetime
from dotenv import load_dotenv
from timeframe import TIMEFRAME_SCHEDULE
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names",
    category=UserWarning,
)

from alpaca.data.historical import CryptoHistoricalDataClient
import config
from bot.logging import setup_logging
import logging

setup_logging(logging.INFO)

logger = logging.getLogger(__name__)
import alpaca_trade_api as tradeapi
import logic.logic_25_catmulti as catmulti

# -------------------------------
# 1. Load Configuration (from .env)
# -------------------------------
load_dotenv()

POSSIBLE_FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions', 'sentiment',
    'greed_index', 'news_count', 'news_volume_z', 'price_change', 'high_low_range', 
    'macd_line', 'macd_signal', 'macd_histogram', 'rsi', 'roc', 'atr', 'ema_9', 'ema_21', 
    'ema_50', 'ema_200', 'adx', 'obv', 'bollinger_percB', 'returns_1', 
    'returns_3', 'returns_5', 'std_5', 'std_10', 'lagged_close_1', 'lagged_close_2', 
    'lagged_close_3', 'lagged_close_5', 'lagged_close_10', 'candle_body_ratio', 
    'wick_dominance', 'gap_vs_prev', 'volume_zscore', 'atr_zscore', 'rsi_zscore',
    'month', 'hour_sin', 'hour_cos', 'day_of_week_sin',  'day_of_week_cos', 
    'days_since_high', 'days_since_low', "d_sentiment"
]

# trading / model selection ------------------------------------------------------------------
ML_MODEL  = config.ML_MODEL
SUB_VERSION = config.SUB_VERSION
API_KEY = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")
API_BASE_URL = config.API_BASE_URL
SUBMETA_USE_ACTION = os.getenv("SUBMETA_USE_ACTION", "false").strip().lower() == "true"

# scheduling & run‑mode ----------------------------------------------------------------------
RUN_SCHEDULE = config.RUN_SCHEDULE
BACKTEST_TICKER = config.BACKTEST_TICKER
DISABLE_PRED_CLOSE = config.DISABLE_PRED_CLOSE
USE_SHORT = config.USE_SHORT
HORIZON = max(1, int(getattr(config, "HORIZON", 1)))
USE_HALF_LIFE = config.USE_HALF_LIFE

_raw = config.STATIC_TICKERS
STATIC_TICKERS = {t.strip().upper()
                  for t in _raw.split(",")
                  if t.strip()}      
BAR_TIMEFRAME = config.BAR_TIMEFRAME
N_BARS = config.N_BARS
ROLLING_CANDLES = config.ROLLING_CANDLES
TRADE_LOGIC = config.TRADE_LOGIC
SHUTDOWN = False


# discord / ai / news ------------------------------------------------------------------------
DISCORD_MODE  = config.DISCORD_MODE
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")
DISCORD_USER_ID = os.getenv("DISCORD_USER_ID", "")
BING_API_KEY  = os.getenv("BING_API_KEY", "")
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY", "")
AI_TICKER_COUNT = config.AI_TICKER_COUNT
AI_TICKERS: list[str] = []
USE_TICKER_SELECTION = config.USE_TICKER_SELECTION

# maximum number of distinct tickers allowed in portfolio (0=unlimited)
MAX_TICKERS = config.MAX_TICKERS

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
TICKERLIST_PATH = os.path.join(DATA_DIR, "tickerlist.txt")
BEST_TICKERS_CACHE = os.path.join(DATA_DIR, "best_tickers_cache.json")

BACKTEST_CACHE_VERSION = 1
BACKTEST_CACHE_DIR = os.path.join(DATA_DIR, "backtest_cache")
os.makedirs(BACKTEST_CACHE_DIR, exist_ok=True)

USE_FULL_SHARES = config.USE_FULL_SHARES

TICKERLIST_TOP_N = 3
TICKERS: list[str] = []
TICKER_ML_OVERRIDES: dict[str, str] = {}
SELECTION_MODEL: str | None = None

TRADE_LOG_FILENAME = os.path.join(DATA_DIR, "trade_log.csv")

CNN_FEAR_GREED_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
ALTERNATIVE_FNG_CSV_URL = "https://api.alternative.me/fng/?limit=0&format=csv"

# misc ---------------------------------------------------------------------------------------
N_ESTIMATORS = 100
RANDOM_SEED  = 42
NY_TZ = pytz.timezone("America/New_York")
api = tradeapi.REST(API_KEY, API_SECRET, API_BASE_URL, api_version='v2')

REWRITE = config.REWRITE

crypto_client = CryptoHistoricalDataClient()

import bot.cli.console as console
import bot.selection.ranking as ranking
import bot.selection.loader as loader
import bot.stuffs.candles as candles
import bot.trading.logic as logicScript
import bot.ml.stacking as stacking


def _perform_trading_job(skip_data=False, scheduled_time_ny: str = None):
    import bot.trading.orders as orders
    # Pick tickers based on selection mode
    if USE_TICKER_SELECTION:
        selected, check = ranking.load_best_ticker_cache()
        if check is None:
            selected = ranking.select_best_tickers(
                top_n=TICKERLIST_TOP_N,
                skip_data=skip_data
            )
    else:
        selected = loader.load_tickerlist()

    loader._ensure_ai_tickers()
    owned = list(ranking.get_owned_tickers())
    tickers_run = sorted(set(selected + AI_TICKERS + owned))

    global TICKERS
    TICKERS = tickers_run

    batching = (USE_TICKER_SELECTION is False)
    if batching:
        orders.begin_trade_batch()

    try:
        for ticker in tickers_run:
            tf_code = candles.timeframe_to_code(BAR_TIMEFRAME)
            ticker_fs = candles.fs_safe_ticker(ticker)
            csv_filename = candles.candle_csv_path(ticker_fs, tf_code)

            if not skip_data:
                df = candles.fetch_candles_plus_features(
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
            else:
                logging.info(f"[{ticker}] skip_data=True. Using existing CSV {csv_filename} for trade job.")
                if not os.path.exists(csv_filename):
                    logging.error(f"[{ticker}] CSV file {csv_filename} does not exist. Cannot trade.")
                    continue
                df = candles.read_csv_limited(csv_filename)
                if df.empty:
                    logging.error(f"[{ticker}] Existing CSV is empty. Skipping.")
                    continue

            allowed = set(POSSIBLE_FEATURE_COLS) | {"timestamp"}
            df = df.loc[:, [c for c in df.columns if c in allowed]]

            if scheduled_time_ny is not None:
                if not logicScript.check_latest_candle_condition(df, BAR_TIMEFRAME, scheduled_time_ny):
                    logging.info(
                        f"[{ticker}] Latest candle condition not met for timeframe {BAR_TIMEFRAME} "
                        f"at scheduled time {scheduled_time_ny}. Skipping trade."
                    )
                    continue

            raw_pred = stacking.train_and_predict(df, ticker=ticker)

            if isinstance(raw_pred, str) and raw_pred.upper() in {"BUY", "SELL", "HOLD", "NONE"}:
                action_str = raw_pred.upper()
                live_price = logicScript.get_current_price(ticker)

                if action_str == "BUY":
                    account = api.get_account()
                    max_qty = int(float(account.cash) // live_price)
                    if max_qty > 0:
                        orders.buy_shares(ticker, max_qty, live_price, live_price)
                    else:
                        logging.info(f"[{ticker}] No cash available for BUY.")

                elif action_str == "SELL":
                    try:
                        pos_qty = int(float(api.get_position(ticker).qty))
                    except Exception:
                        pos_qty = 0
                    if pos_qty > 0:
                        orders.sell_shares(ticker, pos_qty, live_price, live_price)
                    else:
                        logging.info(f"[{ticker}] Nothing to SELL for {ticker}.")

                else:
                    logging.info(f"[{ticker}] Action={action_str}. No trade executed.")

                continue

            if isinstance(raw_pred, tuple) and len(raw_pred) == 2:
                raw_pred = raw_pred[0]

            if raw_pred is None:
                logging.error(f"[{ticker}] Model failed – skipping.")
                continue

            try:
                pred_close = float(raw_pred)
            except (TypeError, ValueError):
                logging.error(f"[{ticker}] Prediction not numeric ({raw_pred!r}). Skipping.")
                continue

            current_price = float(df.iloc[-1]["close"])
            logging.info(f"[{ticker}] Current = {current_price:.2f}  •  Predicted = {pred_close:.2f}")
            logicScript.trade_logic(current_price, pred_close, ticker)

    finally:
        if batching:
            orders.end_trade_batch_and_flush()


def setup_schedule_for_timeframe(timeframe: str) -> None:

    # --- weekly job that must ALWAYS be scheduled -----------------------------------------
    # Runs every Saturday at 06:00 New York time, regardless of RUN_SCHEDULE.
    schedule.every().saturday.at("06:00").do(catmulti.get_best_half_life)

    # --- honour master switch --------------------------------------------------------------
    # If RUN_SCHEDULE is off, we still keep the ALWAYS-on Saturday job above.
    if RUN_SCHEDULE == "off":
        logging.info("RUN_SCHEDULE=off – daily timeframe jobs not queued (Saturday half-life job still queued).")
        return

    # choose a known table key --------------------------------------------------------------
    if timeframe not in TIMEFRAME_SCHEDULE:
        logging.warning(f"{timeframe} not recognised; defaulting to 1Day.")
        timeframe = "1Day"

    times_list = TIMEFRAME_SCHEDULE[timeframe]

    # build the jobs ------------------------------------------------------------------------
    for t in times_list:
        schedule.every().day.at(t).do(lambda t=t: run_job(t))
        schedule.every().day.at(t).do(lambda t=t: ranking.maybe_update_best_tickers(t))

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
            logging.error(f"Error checking market clock (Attempt {attempt}/{max_retries})")
            if attempt < max_retries:
                time.sleep(60)
            else:
                logging.error("Max retries exceeded. Skipping the scheduled job.")
                return

    global TICKERS
    if USE_TICKER_SELECTION:
        cached, _ = ranking.load_best_ticker_cache()
        if cached:
            TICKERS = cached
        else:
            TICKERS = ranking.compute_and_cache_best_tickers()
    else:
        TICKERS = loader.load_tickerlist()

    _perform_trading_job(skip_data=False, scheduled_time_ny=scheduled_time_ny)

    logging.info("Scheduled trading job finished.")

def main():
    logicScript._update_logic_json()
    setup_schedule_for_timeframe(BAR_TIMEFRAME)
    listener_thread = threading.Thread(target=console.console_listener, daemon=True)
    listener_thread.start()
    logging.info("Bot started. Running schedule in local NY time.")
    logging.info("Write 'commands' for command list")
    try:
        account = api.get_account()
        positions = api.list_positions()
        if positions:
            for pos in positions:
                logging.info(f"{pos.symbol}: {pos.qty} shares @ {pos.avg_entry_price}")
        else:
            logging.info("No shares owned.")
    except Exception as e:
        logging.error(f"Alpaca API test failed: {e}")
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
    
