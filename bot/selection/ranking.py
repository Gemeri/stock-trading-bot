from datetime import datetime, timedelta, timezone, date
import bot.ml.stacking as stacking
import bot.trading.logic as logic
from forest import api, MAX_TICKERS, BAR_TIMEFRAME, POSSIBLE_FEATURE_COLS, TRADE_LOGIC, N_BARS, REWRITE
import logging
import numpy as np
import importlib
import pandas as pd
import bot.ml.registry as registry
import forest
import os
import json
import pytz
import bot.stuffs.candles as candles
import bot.selection.loader as loader

logger = logging.getLogger(__name__)

NY_TZ = pytz.timezone("America/New_York")


def get_owned_tickers() -> set[str]:
    try:
        positions = api.list_positions()
        return {p.symbol for p in positions if abs(float(p.qty)) > 0}
    except Exception as e:
        logging.error(f"Error retrieving positions: {e}")
        return set()


def owned_ticker_count() -> int:
    return len(get_owned_tickers())


def available_cash() -> float:
    try:
        account = api.get_account()
        return float(account.cash)
    except Exception as e:
        logging.error(f"Error retrieving account cash: {e}")
        return 0.0


def max_tickers_reached() -> bool:
    return MAX_TICKERS > 0 and owned_ticker_count() >= MAX_TICKERS


def cash_below_minimum() -> bool:
    return available_cash() < 10


def select_best_tickers(top_n: int | None = None, skip_data: bool = False) -> list[str]:
    if top_n is None:
        top_n = forest.TICKERLIST_TOP_N

    tickers = loader.load_tickerlist()
    if not tickers:
        return []

    print(forest.SELECTION_MODEL)

    # ────────────────────────────────────────────────────────────────────
    # 1. Trade-logic branch ─ use run_backtest for scoring
    # ────────────────────────────────────────────────────────────────────
    if forest.SELECTION_MODEL and forest.SELECTION_MODEL.endswith('.py'):
        # Dynamically load the trade-logic module (from module path or file)
        try:
            logic_module = importlib.import_module(forest.SELECTION_MODEL)
        except ModuleNotFoundError:
            spec = importlib.util.spec_from_file_location("trade_logic_module", forest.SELECTION_MODEL)
            if spec is None or spec.loader is None:
                logging.error("Unable to find trade-logic script: %s", forest.SELECTION_MODEL)
                return []
            logic_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(logic_module)

        # For others we just keep ticker symbols in discovery order.
        results_with_conf: list[tuple[float, str]] = []
        results_plain: list[str] = []

        for tck in tickers:
            tf_code  = candles.timeframe_to_code(BAR_TIMEFRAME)
            csv_file = candles.candle_csv_path(tck, tf_code)

            # ── Load or refresh candles + features ────────────────────────
            if skip_data:
                if not os.path.exists(csv_file):
                    continue
                df = pd.read_csv(csv_file)
                if df.empty:
                    continue
            else:
                df = candles.fetch_candles_plus_features(
                    tck, bars=N_BARS, timeframe=BAR_TIMEFRAME, rewrite_mode=REWRITE
                )
                df.to_csv(csv_file, index=False)

            allowed_cols = set(POSSIBLE_FEATURE_COLS) | {"timestamp"}
            df = df.loc[:, [c for c in df.columns if c in allowed_cols]]

            # ── Fresh prediction for latest row ───────────────────────────
            pred_close = stacking.train_and_predict(df, ticker=tck, for_selection=True)
            if isinstance(pred_close, dict):
                pred_close = list(pred_close.values())[-1]
            try:
                pred_close = float(np.asarray(pred_close).flatten()[-1])
            except (TypeError, ValueError):
                pred_close = np.nan

            if "predicted_close" not in df.columns:
                df["predicted_close"] = np.nan
            df["predicted_close"] = df["predicted_close"].astype(float)
            df.at[df.index[-1], "predicted_close"] = pred_close

            # ── Run back-test once on full data ───────────────────────────
            try:
                if TRADE_LOGIC == 25 or TRADE_LOGIC == "logic_25_catmulti.py" or TRADE_LOGIC == 9 or TRADE_LOGIC == "logic_9_tcn.py":
                    # New signature: return (action, confidence) when confidence=True
                    rb_result = logic_module.run_backtest(
                        current_price     = logic.get_current_price(tck),
                        predicted_price   = pred_close,
                        position_qty      = 0,
                        current_timestamp = df.iloc[-1]["timestamp"],
                        candles           = df.copy(),
                        ticker            = tck,
                        confidence        = True,
                    )
                else:
                    # Legacy behaviour: no confidence flag
                    rb_result = logic_module.run_backtest(
                        current_price     = logic.get_current_price(tck),
                        predicted_price   = pred_close,
                        position_qty      = 0,
                        current_timestamp = df.iloc[-1]["timestamp"],
                        candles           = df.copy(),
                        ticker            = tck,
                    )
            except Exception as exc:
                logging.exception("Back-test failed for %s: %s", tck, exc)
                continue

            # ── Unpack action / confidence depending on TRADE_LOGIC ───────
            action = rb_result
            confidence_score: float | None = None

            if TRADE_LOGIC == 25 or TRADE_LOGIC == "logic_25_catmulti.py" or TRADE_LOGIC == 9 or TRADE_LOGIC == "logic_9_tcn.py":
                # Expected shape: (action, confidence)
                if isinstance(rb_result, tuple) and len(rb_result) >= 2:
                    action, confidence_score = rb_result[0], rb_result[1]
                elif isinstance(rb_result, dict):
                    # Fallback: dict-style result
                    action = rb_result.get("action", rb_result.get("signal", "NONE"))
                    confidence_score = rb_result.get("confidence")
                else:
                    # Fallback: only action returned
                    action = rb_result
                    confidence_score = None

                # Normalise confidence_score to float if possible
                if confidence_score is not None:
                    try:
                        confidence_score = float(confidence_score)
                    except (TypeError, ValueError):
                        confidence_score = None

            # Normalise action to upper-case string
            action_str = str(action).strip().upper()
            logging.info("%s: %s", tck, action_str)

            # Keep tickers with a BUY signal
            if action_str == "BUY":
                if TRADE_LOGIC == 25 or TRADE_LOGIC == "logic_25_catmulti.py" or TRADE_LOGIC == 9 or TRADE_LOGIC == "logic_9_tcn.py":
                    # If confidence is missing, treat as 0.0 so it sorts to the bottom.
                    if confidence_score is None or not np.isfinite(confidence_score):
                        confidence_score = 0.0
                    results_with_conf.append((confidence_score, tck))
                else:
                    results_plain.append(tck)

        # ── Finalise trade-logic selection ───────────────────────────────
        if TRADE_LOGIC == 25 or TRADE_LOGIC == "logic_25_catmulti.py" or TRADE_LOGIC == 9 or TRADE_LOGIC == "logic_9_tcn.py":
            if not results_with_conf:
                logging.info("No tickers produced a BUY signal.")
                return []
            # Sort BUY tickers by confidence (descending) and apply top_n
            results_with_conf.sort(key=lambda x: x[0], reverse=True)
            return [t for _, t in results_with_conf[:max(1, top_n)]]
        else:
            if not results_plain:
                logging.info("No tickers produced a BUY signal.")
                return []
            # Preserve discovery order but cap at top_n
            return results_plain[:max(1, top_n)]

    # ────────────────────────────────────────────────────────────────────
    # 2. ML-ranking branch (unchanged from original)
    # ────────────────────────────────────────────────────────────────────
    ml_models = registry.get_ml_models_for_ticker(for_selection=True)
    classifier_set = {"forest_cls", "xgboost_cls", "lightgbm_cls", "transformer_cls", "catboost_cls"}
    results: list[tuple[float, str]] = []

    for tck in tickers:
        tf_code  = candles.timeframe_to_code(BAR_TIMEFRAME)
        csv_file = candles.candle_csv_path(tck, tf_code)

        if skip_data:
            if not os.path.exists(csv_file):
                continue
            df = pd.read_csv(csv_file)
            if df.empty:
                continue
        else:
            df = candles.fetch_candles_plus_features(
                tck, bars=N_BARS, timeframe=BAR_TIMEFRAME, rewrite_mode=REWRITE
            )
            df.to_csv(csv_file, index=False)

        allowed = set(POSSIBLE_FEATURE_COLS) | {"timestamp"}
        df      = df.loc[:, [c for c in df.columns if c in allowed]]
        pred    = stacking.train_and_predict(df, ticker=tck, for_selection=True)
        if pred is None:
            continue

        try:
            pred_val = float(pred)
        except (TypeError, ValueError):
            continue

        current_price = float(df.iloc[-1]["close"])
        if set(ml_models) & classifier_set:                 # classifier prob
            metric = pred_val
        else:                                               # regression %
            metric = (pred_val - current_price) / current_price
        print(tck, ": ", metric)
        results.append((metric, tck))

    # ── Apply thresholds ────────────────────────────────────────────────
    if set(ml_models) & classifier_set:
        results = [(m, t) for m, t in results if m >= 0.55]
    else:
        results = [(m, t) for m, t in results if m >= 0.01]

    if not results:
        logging.info("No tickers met selection thresholds.")
        return []

    results.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in results[:max(1, top_n)]]

def load_best_ticker_cache() -> tuple[list[str], datetime | None]:
    if not os.path.exists(forest.BEST_TICKERS_CACHE):
        return [], None
    try:
        with open(forest.BEST_TICKERS_CACHE, "r") as f:
            data = json.load(f)
        tickers = data.get("tickers", [])
        ts_raw = data.get("timestamp")
        ts = datetime.fromisoformat(ts_raw) if ts_raw else None
        return tickers, ts
    except Exception as e:
        logging.error(f"Error reading best ticker cache: {e}")
        return [], None


def save_best_ticker_cache(tickers: list[str]) -> None:
    data = {
        "tickers": tickers,
        "timestamp": datetime.now(NY_TZ).isoformat()
    }
    try:
        with open(forest.BEST_TICKERS_CACHE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        logging.error(f"Unable to save best ticker cache: {e}")


def compute_and_cache_best_tickers() -> list[str]:

    owned = list(get_owned_tickers())

    if cash_below_minimum() or max_tickers_reached():
        forest.TICKERS = owned
        save_best_ticker_cache(owned)
        return owned

    selected = select_best_tickers()
    combined = sorted({*selected, *owned})

    forest.TICKERS = combined
    save_best_ticker_cache(selected)

    return combined

def maybe_update_best_tickers(scheduled_time_ny: str | None = None):
    today_str = datetime.now().date().isoformat()
    try:
        calendar = api.get_calendar(start=today_str, end=today_str)
        if not calendar:
            logging.info("Market is not scheduled to open today. Skipping job.")
            return
    except Exception as e:
        logging.error(f"Clock check failed")
        return

    _, cache_ts = load_best_ticker_cache()

    if scheduled_time_ny is None:            # start-up path
        run_dt_ny = datetime.now(NY_TZ)
    else:                                    # scheduled path
        h, m = map(int, scheduled_time_ny.split(":"))
        run_dt_ny = datetime.now(NY_TZ).replace(
            hour=h, minute=m, second=0, microsecond=0
        )

    if cache_ts is None or cache_ts < run_dt_ny:
        compute_and_cache_best_tickers()