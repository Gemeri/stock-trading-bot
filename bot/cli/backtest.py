from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import bot.trading.logic as logicScript
from catboost import CatBoostRegressor
import bot.stuffs.candles as candles
import bot.ml.pipelines as pipelines
import bot.ml.registry as registry
import bot.ml.stacking as stacking
import market_timer.timer as market_timer
import matplotlib.pyplot as plt
import bot.ml.models as models
from datetime import datetime
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np
import importlib
import logging
import hashlib
import forest
import math
import json
import re
import os


logger = logging.getLogger(__name__)

N_ESTIMATORS = 100
RANDOM_SEED = 42
BAR_TIMEFRAME = forest.BAR_TIMEFRAME
N_BARS = forest.N_BARS
REWRITE = forest.REWRITE

def _timestamp_to_str(value):
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    try:
        return pd.to_datetime(value).isoformat()
    except Exception:
        return str(value)

def _deserialize_timestamp(value):
    if value is None:
        return None
    try:
        return pd.to_datetime(value)
    except Exception:
        return value


def _serialize_records(records):
    serialized = []
    for rec in records or []:
        new_rec = {}
        for key, val in rec.items():
            if key == "timestamp":
                new_rec[key] = _timestamp_to_str(val)
            elif isinstance(val, (np.integer, np.int64)):
                new_rec[key] = int(val)
            elif isinstance(val, (np.floating, np.float32, np.float64)):
                new_rec[key] = float(val)
            else:
                new_rec[key] = val
        serialized.append(new_rec)
    return serialized


def _deserialize_records(records):
    restored = []
    for rec in records or []:
        new_rec = rec.copy()
        if "timestamp" in new_rec:
            new_rec["timestamp"] = _deserialize_timestamp(new_rec["timestamp"])
        restored.append(new_rec)
    return restored


def get_backtest_cache_path(meta: dict) -> str:
    key_str = json.dumps(meta, sort_keys=True)
    cache_key = hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:16]
    return os.path.join(forest.BACKTEST_CACHE_DIR, f"backtest_{cache_key}.json")

def update_backtest_cache(path: str, meta: dict, *, mode: str, total_iterations: int,
                          next_index: int, predictions: list, actuals: list,
                          timestamps: list, trade_records: list, portfolio_records: list,
                          cash: float, position_qty, avg_entry_price: float,
                          last_action, pending_trades: list = None) -> None:
    remaining = max(0, total_iterations - next_index)
    state = {
        "version": forest.BACKTEST_CACHE_VERSION,
        "mode": mode,
        "total_iterations": total_iterations,
        "next_index": next_index,
        "remaining_candles": remaining,
        "predictions": [float(x) for x in predictions],
        "actuals": [float(x) for x in actuals],
        "timestamps": [_timestamp_to_str(ts) for ts in timestamps],
        "trade_records": _serialize_records(trade_records),
        "portfolio_records": _serialize_records(portfolio_records),
        "cash": float(cash),
        "position_qty": float(position_qty),
        "avg_entry_price": float(avg_entry_price),
        "last_action": last_action,
        "pending_trades": _serialize_records(pending_trades or []),
    }
    state.update(meta)
    save_backtest_cache(path, state)

def load_backtest_cache(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"Failed to load backtest cache {path}: {e}")
        return None

def backtest_cache_matches(state, meta: dict) -> bool:
    if not state or state.get("version") != forest.BACKTEST_CACHE_VERSION:
        return False
    for key, val in meta.items():
        if state.get(key) != val:
            return False
    return True

def save_backtest_cache(path: str, state: dict) -> None:
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, path)
    except Exception as e:
        logging.warning(f"Failed to save backtest cache {path}: {e}")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

def restore_backtest_state(state, *, mode: str, total_iterations: int,
                           start_balance: float):
    if not state or state.get("mode") != mode:
        return None
    cached_total = state.get("total_iterations")
    if cached_total is not None and cached_total != total_iterations:
        return None

    predictions = [float(x) for x in state.get("predictions", [])]
    actuals = [float(x) for x in state.get("actuals", [])]
    timestamps = [_deserialize_timestamp(ts) for ts in state.get("timestamps", [])]
    trade_records = _deserialize_records(state.get("trade_records", []))
    portfolio_records = _deserialize_records(state.get("portfolio_records", []))
    pending_trades = _deserialize_records(state.get("pending_trades", []))

    cash = float(state.get("cash", start_balance))
    position_qty = state.get("position_qty", 0)
    try:
        position_qty = int(position_qty)
    except Exception:
        position_qty = float(position_qty)
    avg_entry_price = float(state.get("avg_entry_price", 0.0))
    next_index = int(state.get("next_index", 0))
    last_action = state.get("last_action")

    return (
        predictions,
        actuals,
        timestamps,
        trade_records,
        portfolio_records,
        cash,
        position_qty,
        avg_entry_price,
        next_index,
        last_action,
        pending_trades,
    )

def clear_backtest_cache(path: str) -> None:
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logging.warning(f"Unable to remove backtest cache {path}: {e}")


def check_for_wait(ticker: str, direction: int):
    """Check if market_timer says to execute or wait."""
    df = candles.fetch_candles_plus_features(
        ticker,
        bars=N_BARS,
        timeframe=BAR_TIMEFRAME,
        rewrite_mode=REWRITE
    )
    decision = market_timer.get_execution_decision(df, ticker, 3, "cat", direction)
    if decision == "EXECUTE":
        return True
    else:
        return False


def run_backtest(parts, use_ml=True):
    logging.info(parts)
    final_values_by_logic = {}
    if len(parts) < 2:
        logging.warning(
            "Usage: backtest <N> [simple|complex] [timeframe?] [-r?] "
            "or backtest all <N> [simple|complex] [timeframe?] [-r?]"
        )
        return

    # --------------------------------------------------------------
    #  Detect "backtest all" vs single-logic backtest
    # --------------------------------------------------------------
    run_all = False
    arg_offset = 0

    if parts[1].lower() == "all":
        run_all = True
        arg_offset = 1
        if len(parts) < 3:
            logging.warning(
                "Usage: backtest all <N> [simple|complex] [timeframe?] [-r?]"
            )
            return
        test_size_str = parts[2]
    else:
        test_size_str = parts[1]

    approach = "simple"
    timeframe_for_backtest = forest.BAR_TIMEFRAME
    possible_approaches = ["simple", "complex"]
    skip_data = ('-r' in parts)

    # start parsing optional args after "all" + <N> or just <N>
    idx = 2 + arg_offset
    while idx < len(parts):
        val = parts[idx]
        if val in possible_approaches:
            approach = val
        elif val != '-r':
            timeframe_for_backtest = val
        idx += 1

    try:
        test_size = int(test_size_str)
    except ValueError:
        logging.error("Invalid test_size for 'backtest' command.")
        return

    logic_dir, json_path = logicScript.get_logic_dir_and_json()
    if not os.path.isfile(json_path):
        logicScript._update_logic_json()

    with open(json_path, "r") as f:
        LOGIC_MODULE_MAP = json.load(f) if os.path.getsize(json_path) else {}

    # Precompute ML models for ticker once (still reused below)
    ml_models = registry.get_ml_models_for_ticker(forest.BACKTEST_TICKER) if use_ml else []

    # --------------------------------------------------------------
    #  Determine which logic module(s) to run
    # --------------------------------------------------------------
    if run_all:
        if not LOGIC_MODULE_MAP:
            logging.error("No logic scripts available for 'backtest all'.")
            return

        numbered = []
        for key, module_name in LOGIC_MODULE_MAP.items():
            try:
                num = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(module_name, str) and module_name:
                numbered.append((num, module_name))

        if not numbered:
            logging.error(
                "No numbered logic scripts found in LOGIC_MODULE_MAP for 'backtest all'."
            )
            return

        numbered.sort(key=lambda x: x[0])
        logic_modules_to_run = [m for _, m in numbered]

        logging.info(
            f"[{forest.BACKTEST_TICKER}] backtest all: running logic scripts in order: "
            f"{', '.join(logic_modules_to_run)}"
        )
    else:
        # Original single-logic behaviour
        if use_ml and {
            "forest_cls", "xgboost_cls", "lightgbm_cls",
            "transformer_cls", "sub-vote", "sub_meta",
            "sub-meta", "sub_vote", "catboost_cls"
        } & set(ml_models):
            logic_modules_to_run = ["classifier"]
        else:
            logic_modules_to_run = [logicScript._get_logic_script_name(str(forest.TRADE_LOGIC))]

    # --------------------------------------------------------------
    #  Prepare / load data ONCE, reused for all logic scripts
    # --------------------------------------------------------------
    tf_code = candles.timeframe_to_code(timeframe_for_backtest)
    ticker_fs = candles.fs_safe_ticker(forest.BACKTEST_TICKER)
    csv_filename = candles.candle_csv_path(ticker_fs, tf_code)

    if skip_data:
        logging.info(
            f"[{forest.BACKTEST_TICKER}] backtest -r: using existing CSV {csv_filename}"
        )
        if not os.path.exists(csv_filename):
            logging.error(
                f"[{forest.BACKTEST_TICKER}] CSV file {csv_filename} does not exist, skipping."
            )
            return
        df = candles.read_csv_limited(csv_filename)
        if df.empty:
            logging.error(f"[{forest.BACKTEST_TICKER}] CSV is empty, skipping.")
            return
    else:
        df = candles.fetch_candles_plus_features(
            forest.BACKTEST_TICKER,
            bars=forest.N_BARS,
            timeframe=forest.BAR_TIMEFRAME,
            rewrite_mode=forest.REWRITE
        )
        df.to_csv(csv_filename, index=False)
        logging.info(
            f"[{forest.BACKTEST_TICKER}] Saved updated CSV with sentiment & features "
            f"(minus disabled) to {csv_filename} before backtest."
        )

    allowed = set(forest.POSSIBLE_FEATURE_COLS) | {"timestamp"}
    df = df.loc[:, [col for col in df.columns if col in allowed]]

    if 'close' not in df.columns:
        logging.error(
            f"[{forest.BACKTEST_TICKER}] No 'close' column after feature processing. Cannot backtest."
        )
        return

    horizon_gap = forest.HORIZON
    df['target'] = df['close'].shift(-horizon_gap)
    df.dropna(subset=['target'], inplace=True)

    if len(df) <= test_size + horizon_gap:
        logging.error(
            f"[{forest.BACKTEST_TICKER}] Not enough rows for backtest split. Need more data than test_size."
        )
        return

    total_len = len(df)
    train_end = total_len - test_size
    if train_end < 1:
        logging.error(
            f"[{forest.BACKTEST_TICKER}] train_end < 1. Not enough data for that test_size."
        )
        return

    # ==============================================================
    #  MAIN LOOP: run the backtest for each requested logic module
    # ==============================================================
    for logic_module_name in logic_modules_to_run:
        logging.info("Trading logic: " + logic_module_name)
        try:
            logic_module = importlib.import_module(f"{logic_dir}.{logic_module_name}")
        except Exception as e:
            logging.error(
                f"Unable to import logic module {logic_module_name}: {e}"
            )
            continue

        # Per-logic cache meta/state (so each logic has its own cache file)
        cache_meta = {
            "ticker": forest.BACKTEST_TICKER,
            "approach": approach,
            "timeframe": timeframe_for_backtest,
            "test_size": test_size,
            "logic_module": logic_module_name,
            "ml_models": sorted(ml_models),
        }
        cache_path = get_backtest_cache_path(cache_meta)
        cache_state = load_backtest_cache(cache_path)
        if cache_state and not backtest_cache_matches(cache_state, cache_meta):
            logging.info(
                f"[{forest.BACKTEST_TICKER}] Ignoring cached backtest state for "
                f"{logic_module_name} due to configuration change."
            )
            cache_state = None

        classifier_names = {
            "forest_cls", "xgboost_cls",
            "lightgbm_cls", "catboost_cls"
        }

        # Only treat as "classifier" result dir when the logic module itself is "classifier"
        if (
            logic_module_name == "classifier"
            and classifier_names & set(ml_models)
        ):
            logic_num_str = ",".join(ml_models)
        else:
            match_script  = re.match(r"^logic_(\d+)_", logic_module_name)
            logic_num_str = match_script.group(1) if match_script else "unknown"

        results_dir = os.path.join("results", tf_code, logic_num_str)

        os.makedirs(results_dir, exist_ok=True)

        # ----------------------------------------------------------
        #  Per-logic backtest state variables
        # ----------------------------------------------------------
        predictions = []
        actuals = []
        timestamps = []
        trade_records = []
        portfolio_records = []
        position_qty = 0
        avg_entry_price = 0.0
        last_action = None
        start_iter = 0
        pending_trades = []

        ticker_has_slash = "/" in str(forest.BACKTEST_TICKER)
        allow_shorting = forest.USE_SHORT and (not ticker_has_slash)
        buy_fee_rate = 0.0025 if ticker_has_slash else 0.0

        start_balance = 10000.0 if not ticker_has_slash else 1000000.0
        cash = start_balance

        def record_trade(action, tstamp, shares, curr_price, pred_price, pl):
            trade_records.append({
                "timestamp": tstamp,
                "action": action,
                "shares": shares,
                "current_price": curr_price,
                "predicted_price": pred_price,
                "profit_loss": pl
            })

        def get_portfolio_value(pos_qty, csh, c_price, avg_price):
            if pos_qty > 0:
                return csh + (c_price - avg_price) * pos_qty
            elif pos_qty < 0:
                return csh + (avg_price - c_price) * abs(pos_qty)
            else:
                return csh

        if approach in ["simple", "complex"]:
            if use_ml:
                # (Re)fetch ml models inside approach block, as original code did
                ml_models = registry.get_ml_models_for_ticker(forest.BACKTEST_TICKER)
                if approach == "simple":
                    horizon_buffer = max(0, horizon_gap - 1)
                    train_df = df.iloc[:train_end]
                    if horizon_buffer > 0:
                        if len(train_df) <= horizon_buffer:
                            logging.error(
                                f"[{forest.BACKTEST_TICKER}] Not enough training rows after horizon cut."
                            )
                            continue
                        train_df = train_df.iloc[:-horizon_buffer]
                    test_df  = df.iloc[train_end:]
                    idx_list = list(test_df.index)

                    feature_cols = [c for c in forest.POSSIBLE_FEATURE_COLS if c in train_df.columns]
                    X_train = train_df[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
                    y_train = train_df['target']

                    model_stack = []
                    for m in ml_models:
                        if m in ["forest", "rf", "randomforest"]:
                            mdl = RandomForestRegressor(
                                n_estimators=N_ESTIMATORS,
                                random_state=RANDOM_SEED
                            )
                        elif m == "xgboost":
                            mdl = xgb.XGBRegressor(
                                n_estimators=114,
                                max_depth=9,
                                learning_rate=0.14264252588219034,
                                subsample=0.5524803023252148,
                                colsample_bytree=0.7687841723045249,
                                gamma=0.5856035407199236,
                                reg_alpha=0.5063880221467401,
                                reg_lambda=0.0728996118523866,
                            )
                        elif m == "catboost":
                            mdl = CatBoostRegressor(
                                iterations=600, depth=6, learning_rate=0.05,
                                l2_leaf_reg=3, bagging_temperature=0.5,
                                loss_function="RMSE", eval_metric="RMSE",
                                random_seed=42, verbose=False
                            )
                        elif m in ["lightgbm", "lgbm"]:
                            mdl = lgb.LGBMRegressor(
                                n_estimators=N_ESTIMATORS,
                                random_state=RANDOM_SEED
                            )
                        else:
                            continue
                        pipelines.fixed_fit(X_train, y_train, mdl)
                        model_stack.append(mdl)

                    if not model_stack:
                        logging.error(
                            f"[{forest.BACKTEST_TICKER}] No supported model for simple backtest."
                        )
                        continue
                else:  # complex
                    idx_list = list(range(train_end, total_len))

                min_train_rows = len(train_df) if approach == "simple" else len(df.iloc[:train_end]) - max(0, horizon_gap - 1)
                if min_train_rows < 70:
                    logging.error(
                        f"[{forest.BACKTEST_TICKER}] Not enough training rows for {approach} approach."
                    )
                    continue
            else:
                test_df = df.iloc[train_end:]
                idx_list = list(test_df.index)

            total_iterations = len(idx_list)
            resume_data = restore_backtest_state(
                cache_state,
                mode="ml",
                total_iterations=total_iterations,
                start_balance=start_balance,
            )
            if resume_data:
                (
                    predictions,
                    actuals,
                    timestamps,
                    trade_records,
                    portfolio_records,
                    cash,
                    position_qty,
                    avg_entry_price,
                    start_iter,
                    last_action,
                    pending_trades,
                ) = resume_data
                logging.info(
                    f"[{forest.BACKTEST_TICKER}] Resuming backtest for {logic_module_name} "
                    f"at candle {start_iter} of {total_iterations}."
                )

            if use_ml and ("sub-vote" in ml_models or "sub-meta" in ml_models):
                mode       = "sub-vote" if "sub-vote" in ml_models else "sub-meta"
                signal_df  = models.call_sub_main(forest.BACKTEST_TICKER, df, 500)

                # --- keep only actionable rows and order chronologically ---------------
                trade_actions = (
                    signal_df[signal_df["action"].isin(["BUY", "SELL"])]
                    .copy()
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )

                # -----------------------------------------------------------------------
                #  Portfolio state variables
                # -----------------------------------------------------------------------
                trade_records      = []
                portfolio_records  = []
                predictions        = []
                actuals            = []
                timestamps         = []
                pending_trades     = []

                START_BALANCE  = 10_000.0 if not ticker_has_slash else 1_000_000.0
                cash = START_BALANCE
                position_qty = 0
                avg_entry_price = 0.0

                # convenience -----------------------------------------------------------
                def record_trade(act, ts, qty, px, pl=None):
                    trade_records.append({
                        "timestamp"     : ts,
                        "action"        : act,
                        "shares"        : qty,
                        "current_price" : px,
                        "profit_loss"   : pl
                    })

                # timestamp alignment ---------------------------------------------------
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                trade_actions['timestamp'] = pd.to_datetime(trade_actions['timestamp'])

                df_sorted = df.sort_values('timestamp').reset_index(drop=True)
                actions_sorted = trade_actions.sort_values('timestamp').reset_index(drop=True)

                merged_actions = pd.merge_asof(
                    actions_sorted,
                    df_sorted,
                    on='timestamp',
                    direction='backward'
                )

                total_iterations = len(merged_actions)
                resume_data = restore_backtest_state(
                    cache_state,
                    mode="signal",
                    total_iterations=total_iterations,
                    start_balance=START_BALANCE,
                )
                if resume_data:
                    (
                        predictions,
                        actuals,
                        timestamps,
                        trade_records,
                        portfolio_records,
                        cash,
                        position_qty,
                        avg_entry_price,
                        start_iter,
                        last_action,
                        pending_trades,
                    ) = resume_data
                    logging.info(
                        f"[{forest.BACKTEST_TICKER}] Resuming signal backtest for {logic_module_name} "
                        f"at step {start_iter} of {total_iterations}."
                    )
                else:
                    start_iter = 0
                    last_action = None

                # -----------------------------------------------------------------------
                #  Iterate over each (action, candle) row – enforce position sanity
                # -----------------------------------------------------------------------
                for iter_idx, (_, row) in enumerate(merged_actions.iterrows()):
                    if iter_idx < start_iter:
                        continue
                    raw_act = row["action"]
                    ts      = row["timestamp"]
                    price   = row["close"]

                    if raw_act == "BUY":
                        # If already long, ignore duplicate BUY
                        if position_qty > 0:
                            raw_act = "NONE"
                        # If short, COVER first
                        elif position_qty < 0:
                            pl = (avg_entry_price - price) * abs(position_qty)
                            cash += pl
                            record_trade("COVER", ts, abs(position_qty), price, pl)
                            position_qty    = 0
                            avg_entry_price = 0.0

                    elif raw_act == "SELL":
                        if position_qty <= 0:
                            raw_act = "NONE"

                    # ------------------------------------------------ execute action ---
                    if raw_act == "BUY":
                        shares_to_buy = int(cash // price)
                        if shares_to_buy > 0:
                            position_qty    = shares_to_buy
                            avg_entry_price = price
                            # 0.25% fee on BUY only for cyrpto
                            fee = shares_to_buy * price * buy_fee_rate
                            if fee:
                                cash -= fee
                            record_trade("BUY", ts, shares_to_buy, price, pl=(-fee if fee else None))

                    elif raw_act == "SELL":
                        if position_qty > 0:
                            pl   = (price - avg_entry_price) * position_qty
                            cash += pl
                            record_trade("SELL", ts, position_qty, price, pl)
                            position_qty    = 0
                            avg_entry_price = 0.0

                    # ------------------------------------------------ portfolio value --
                    port_val = (
                        cash
                        + (price - avg_entry_price) * position_qty
                        if position_qty != 0 else cash
                    )
                    portfolio_records.append({"timestamp": ts, "portfolio_value": port_val})

                    last_action = raw_act
                    update_backtest_cache(
                        cache_path,
                        cache_meta,
                        mode="signal",
                        total_iterations=total_iterations,
                        next_index=iter_idx + 1,
                        predictions=predictions,
                        actuals=actuals,
                        timestamps=timestamps,
                        trade_records=trade_records,
                        portfolio_records=portfolio_records,
                        cash=cash,
                        position_qty=position_qty,
                        avg_entry_price=avg_entry_price,
                        last_action=last_action,
                        pending_trades=pending_trades,
                    )

                # -----------------------------------------------------------------------
                #  Ensure results directory exists (unchanged)
                # -----------------------------------------------------------------------
                results_dir = os.path.join("sub", "sub-results")
                os.makedirs(results_dir, exist_ok=True)

            # ----------- ALL OTHER ML MODES -----------------------------------
            else:
                ROLL_WIN = test_size

                for i_, row_idx in enumerate(idx_list):

                    if i_ < start_iter:
                        continue

                    if use_ml:
                        if approach == "simple":
                            if row_idx == 0:
                                continue
                            feat_row = df.loc[row_idx - 1, feature_cols]
                            X_pred = pd.DataFrame([feat_row], columns=feature_cols)
                            X_pred = X_pred.replace([np.inf, -np.inf], 0.0).fillna(0.0)
                            preds = [mdl.predict(X_pred)[0] for mdl in model_stack]
                            pred_close = float(np.mean(preds))
                        else:
                            sub_df = df.iloc[:row_idx]
                            if horizon_gap > 1:
                                if len(sub_df) <= horizon_gap - 1:
                                    logging.error(
                                        f"[{forest.BACKTEST_TICKER}] Not enough rows for training after horizon cut."
                                    )
                                    break
                                sub_df_for_training = sub_df.iloc[: -(horizon_gap - 1)]
                            else:
                                sub_df_for_training = sub_df
                            pred_close = stacking.train_and_predict(
                                sub_df_for_training,
                                ticker=forest.BACKTEST_TICKER
                            )
                    else:
                        pred_close = 0.0
                    row_data   = df.loc[row_idx]
                    real_close = row_data["close"]

                    # ───────────────────────────────────────────────────────────
                    # 2. Build the ROLLING candles slice handed to run_backtest
                    # ───────────────────────────────────────────────────────────
                    start_idx  = max(0, row_idx - ROLL_WIN + 1)      # never < 0
                    candles_bt = df.iloc[start_idx : row_idx + 1].copy()

                    # make sure the prediction column exists & is float-typed
                    if "predicted_close" not in candles_bt.columns:
                        candles_bt["predicted_close"] = np.nan
                    candles_bt["predicted_close"] = candles_bt[
                        "predicted_close"
                    ].astype(float)

                    # ensure pred_close is a scalar, then set it
                    if isinstance(pred_close, dict):
                        pred_close = list(pred_close.values())[-1]  # grab last value
                    pred_close = float(np.asarray(pred_close).flatten()[-1])
                    candles_bt.at[
                        candles_bt.index[-1], "predicted_close"
                    ] = pred_close

                    # ───────────────────────────────────────────────────────────
                    # 3. Call the user's trading-logic module
                    # ───────────────────────────────────────────────────────────
                    print(logic_module)
                    if forest.TRADE_LOGIC == 25 or forest.TRADE_LOGIC == "logic_25_catmulti.py" or forest.TRADE_LOGIC == 9 or forest.TRADE_LOGIC == "logic_9_tcn.py":
                        action = logic_module.run_backtest(
                            current_price = real_close,
                            predicted_price = pred_close,
                            position_qty = position_qty,
                            current_timestamp = row_data["timestamp"],
                            candles = candles_bt,
                            ticker = forest.BACKTEST_TICKER,
                            confidence = False
                        )
                    else:
                        action = logic_module.run_backtest(
                            current_price     = real_close,
                            predicted_price   = pred_close,
                            position_qty      = position_qty,
                            current_timestamp = row_data["timestamp"],
                            candles           = candles_bt,
                            ticker            = forest.BACKTEST_TICKER
                        )

                    # ───────────────────────────────────────────────────────────
                    # 4. Log the prediction so RMSE/plots still work
                    # ───────────────────────────────────────────────────────────
                    predictions.append(pred_close)
                    actuals.append(real_close)
                    timestamps.append(row_data["timestamp"])

                    # ───────────────────────────────────────────────────────────
                    # 5. PROCESS PENDING TRADES (when action is NONE)
                    # ───────────────────────────────────────────────────────────
                    if action == "NONE" and forest.USE_TICKER_SELECTION == 2:
                        executed_indices = []
                        for idx, pending in enumerate(pending_trades):
                            direction = pending["direction"]
                            if check_for_wait(forest.BACKTEST_TICKER, direction):
                                # EXECUTE this pending trade
                                pending_action = pending["action"]
                                
                                # Execute the trade using current price
                                if pending_action == "BUY":
                                    shares_to_buy = int(cash // real_close)
                                    if shares_to_buy > 0:
                                        position_qty = shares_to_buy
                                        avg_entry_price = real_close
                                        fee = shares_to_buy * real_close * buy_fee_rate
                                        if fee:
                                            cash -= fee
                                        record_trade(
                                            "BUY",
                                            row_data["timestamp"],
                                            shares_to_buy,
                                            real_close,
                                            pred_close,
                                            (-fee if fee else None),
                                        )
                                        logging.info(f"Executed pending BUY at {real_close}")
                                
                                elif pending_action == "SELL":
                                    if position_qty > 0:
                                        pl = (real_close - avg_entry_price) * position_qty
                                        cash += pl
                                        record_trade(
                                            "SELL",
                                            row_data["timestamp"],
                                            position_qty,
                                            real_close,
                                            pred_close,
                                            pl,
                                        )
                                        position_qty = 0
                                        avg_entry_price = 0.0
                                        logging.info(f"Executed pending SELL at {real_close}")
                                
                                elif pending_action == "SHORT" and allow_shorting:
                                    shares_to_short = int(cash // real_close)
                                    if shares_to_short > 0:
                                        position_qty = -shares_to_short
                                        avg_entry_price = real_close
                                        record_trade(
                                            "SHORT",
                                            row_data["timestamp"],
                                            shares_to_short,
                                            real_close,
                                            pred_close,
                                            None,
                                        )
                                        logging.info(f"Executed pending SHORT at {real_close}")
                                
                                elif pending_action == "COVER":
                                    if position_qty < 0:
                                        qty = abs(position_qty)
                                        pl = (avg_entry_price - real_close) * qty
                                        cash += pl
                                        record_trade(
                                            "COVER",
                                            row_data["timestamp"],
                                            qty,
                                            real_close,
                                            pred_close,
                                            pl,
                                        )
                                        position_qty = 0
                                        avg_entry_price = 0.0
                                        logging.info(f"Executed pending COVER at {real_close}")
                                
                                executed_indices.append(idx)
                            else:
                                # WAIT - increment counter
                                pending["wait_count"] += 1
                                if pending["wait_count"] >= 7:
                                    logging.info(f"Abandoning pending {pending['action']} after 7 waits")
                                    executed_indices.append(idx)
                        
                        # Remove executed/abandoned trades from pending list
                        for idx in sorted(executed_indices, reverse=True):
                            del pending_trades[idx]

                    # ───────────────────────────────────────────────────────────
                    # 6. NORMALIZE ACTION
                    # ───────────────────────────────────────────────────────────
                    if action == "BUY":
                        if position_qty > 0:
                            action = "NONE"
                        elif position_qty < 0:
                            action = "COVER"

                    elif action == "SELL":
                        if position_qty > 0:
                            pass
                        elif position_qty == 0:
                            # shorting disabled for crypto
                            if allow_shorting:
                                action = "SHORT"
                            else:
                                action = "NONE"
                        else:
                            action = "NONE"

                    elif action == "SHORT":
                        # shorting disabled for crypto
                        if (not allow_shorting) or position_qty != 0:
                            action = "NONE"

                    elif action == "COVER":
                        if position_qty >= 0:
                            action = "NONE"

                    # ───────────────────────────────────────────────────────────
                    # 7. CHECK MARKET TIMER (USE_TICKER_SELECTION == 2)
                    # ───────────────────────────────────────────────────────────
                    if action != "NONE" and forest.USE_TICKER_SELECTION == 2:
                        # Determine direction: 1 for BUY/COVER, 0 for SELL/SHORT
                        direction = 1 if action in ["BUY", "COVER"] else 0
                        
                        if not check_for_wait(forest.BACKTEST_TICKER, direction):
                            # WAIT - add to pending trades
                            pending_trades.append({
                                "action": action,
                                "timestamp": row_data["timestamp"],
                                "price": real_close,
                                "pred_price": pred_close,
                                "direction": direction,
                                "wait_count": 0
                            })
                            logging.info(f"Added {action} to pending trades (WAIT signal)")
                            action = "NONE"  # Don't execute now
                        # else: EXECUTE - proceed normally

                    # ───────────────────────────────────────────────────────────
                    # 8. EXECUTE ACTION
                    # ───────────────────────────────────────────────────────────
                    if action == "BUY":
                        shares_to_buy = int(cash // real_close)
                        if shares_to_buy > 0:
                            position_qty    = shares_to_buy
                            avg_entry_price = real_close

                            # 0.25% fee on BUY only for crypto
                            fee = shares_to_buy * real_close * buy_fee_rate
                            if fee:
                                cash -= fee

                            record_trade(
                                "BUY",
                                row_data["timestamp"],
                                shares_to_buy,
                                real_close,
                                pred_close,
                                (-fee if fee else None),
                            )

                    elif action == "SELL":
                        # close long
                        if position_qty > 0:
                            pl = (real_close - avg_entry_price) * position_qty
                            cash += pl
                            record_trade(
                                "SELL",
                                row_data["timestamp"],
                                position_qty,
                                real_close,
                                pred_close,
                                pl,
                            )
                            position_qty = 0
                            avg_entry_price = 0.0

                    elif action == "SHORT":
                        # short entry (only if allowed and flat)
                        if allow_shorting and position_qty == 0:
                            shares_to_short = int(cash // real_close)
                            if shares_to_short > 0:
                                position_qty    = -shares_to_short
                                avg_entry_price = real_close
                                record_trade(
                                    "SHORT",
                                    row_data["timestamp"],
                                    shares_to_short,
                                    real_close,
                                    pred_close,
                                    None,
                                )

                    elif action == "COVER":
                        # close short
                        if position_qty < 0:
                            qty = abs(position_qty)
                            pl = (avg_entry_price - real_close) * qty
                            cash += pl
                            record_trade(
                                "COVER",
                                row_data["timestamp"],
                                qty,
                                real_close,
                                pred_close,
                                pl,
                            )
                            position_qty = 0
                            avg_entry_price = 0.0

                    val = get_portfolio_value(
                        position_qty, cash, real_close, avg_entry_price
                    )
                    portfolio_records.append(
                        {
                            "timestamp": row_data["timestamp"],
                            "portfolio_value": val,
                        }
                    )

                    last_action = action
                    update_backtest_cache(
                        cache_path,
                        cache_meta,
                        mode="ml",
                        total_iterations=total_iterations,
                        next_index=i_ + 1,
                        predictions=predictions,
                        actuals=actuals,
                        timestamps=timestamps,
                        trade_records=trade_records,
                        portfolio_records=portfolio_records,
                        cash=cash,
                        position_qty=position_qty,
                        avg_entry_price=avg_entry_price,
                        last_action=last_action,
                        pending_trades=pending_trades,
                    )
            # ----------------- END inner loop ---------------------------------

        else:
            logging.warning(
                f"[{forest.BACKTEST_TICKER}] Unknown approach={approach}. "
                f"Skipping backtest for {logic_module_name}."
            )
            continue

        # ------------------------------------------------------------------
        #  ▸  RESULTS & METRICS (per logic module)
        # ------------------------------------------------------------------
        collecting_regression = len(predictions) > 0

        if collecting_regression:
            y_pred = np.array(predictions, dtype=float)
            y_test = np.array(actuals,      dtype=float)

            rmse = math.sqrt(mean_squared_error(y_test, y_pred))
            mae  = mean_absolute_error(y_test, y_pred)

            avg_close = y_test.mean() if len(y_test) else 1e-6
            accuracy  = 100.0 - (mae / avg_close * 100.0)

            logging.info(
                f"[{forest.BACKTEST_TICKER}] Backtest ({approach}) — logic={logic_module_name}, "
                f"test_size={test_size}, "
                f"RMSE={rmse:.4f}, MAE={mae:.4f}, "
                f"Accuracy≈{accuracy:.2f}%"
            )
        else:
            logging.info(
                f"[{forest.BACKTEST_TICKER}] Backtest ({approach}) — logic={logic_module_name}, "
                "classification / signal mode "
                "(sub-vote / sub-meta) — RMSE/MAE not computed."
            )

        # ---------------- save artefacts ----------------
        tf_code       = candles.timeframe_to_code(timeframe_for_backtest)
        results_dir   = os.path.join("results", tf_code, forest.BACKTEST_TICKER, logic_num_str)
        os.makedirs(results_dir, exist_ok=True)

        # ▸ prediction CSV & plot — only when we *have* predictions
        if collecting_regression:
            out_pred_csv = os.path.join(
                results_dir,
                f"backtest_predictions_{forest.BACKTEST_TICKER}_{test_size}_{tf_code}_{approach}.csv",
            )
            out_pred_png = os.path.join(
                results_dir,
                f"backtest_predictions_{forest.BACKTEST_TICKER}_{test_size}_{tf_code}_{approach}.png",
            )

            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "actual_close": actuals,
                    "predicted_close": predictions,
                }
            ).to_csv(out_pred_csv, index=False)

            plt.figure(figsize=(10, 6))
            plt.plot(timestamps, actuals,     label="Actual")
            plt.plot(timestamps, predictions, label="Predicted")
            plt.title(
                f"{forest.BACKTEST_TICKER} Backtest ({approach} — last {test_size}) "
                f"[{timeframe_for_backtest}] — {logic_module_name}"
            )
            plt.xlabel("Timestamp")
            plt.ylabel("Close Price")
            plt.legend()
            plt.grid(True)
            if test_size > 30:
                plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(out_pred_png)
            plt.close()

            logging.info(
                f"[{forest.BACKTEST_TICKER}] Saved predictions for {logic_module_name} → {out_pred_csv}"
            )
            logging.info(
                f"[{forest.BACKTEST_TICKER}] Saved prediction plot for {logic_module_name} → {out_pred_png}"
            )

        # ▸ trade log & portfolio series (always saved)
        trade_log_csv = os.path.join(
            results_dir,
            f"backtest_trades_{forest.BACKTEST_TICKER}_{test_size}_{tf_code}_{approach}.csv",
        )
        port_csv = os.path.join(
            results_dir,
            f"backtest_portfolio_{forest.BACKTEST_TICKER}_{test_size}_{tf_code}_{approach}.csv",
        )
        port_png = os.path.join(
            results_dir,
            f"backtest_portfolio_{forest.BACKTEST_TICKER}_{test_size}_{tf_code}_{approach}.png",
        )

        pd.DataFrame(trade_records).to_csv(trade_log_csv, index=False)
        pd.DataFrame(portfolio_records).to_csv(port_csv, index=False)
        logging.info(
            f"[{forest.BACKTEST_TICKER}] Saved trades for {logic_module_name} → {trade_log_csv}"
        )
        logging.info(
            f"[{forest.BACKTEST_TICKER}] Saved portfolio series for {logic_module_name} → {port_csv}"
        )

        # portfolio value plot (only if records exist)
        port_df = pd.DataFrame(portfolio_records)
        if not port_df.empty:
            plt.figure(figsize=(10, 6))
            plt.plot(
                port_df["timestamp"],
                port_df["portfolio_value"],
                label="Portfolio Value",
            )
            plt.title(
                f"{forest.BACKTEST_TICKER} Portfolio Value Backtest ({approach}) "
                f"— {logic_module_name}"
            )
            plt.xlabel("Timestamp")
            plt.ylabel("Portfolio Value (USD)")
            plt.legend()
            plt.grid(True)
            if test_size > 30:
                plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(port_png)
            plt.close()

            logging.info(
                f"[{forest.BACKTEST_TICKER}] Saved portfolio plot for {logic_module_name} → {port_png}"
            )

        if portfolio_records:
            final_value = float(portfolio_records[-1]["portfolio_value"])
        else:
            # fallback (should be rare): if no records, at least return cash
            final_value = float(cash)

        final_values_by_logic[logic_module_name] = final_value
        logging.info(
            f"[{forest.BACKTEST_TICKER}] FINAL portfolio value — {logic_module_name}: {final_value:.2f}"
        )

        clear_backtest_cache(cache_path)

    # ✅ return value(s)
    if run_all:
        return final_values_by_logic
    # single mode: return the only value (or None if nothing ran)
    if logic_modules_to_run:
        return final_values_by_logic.get(logic_modules_to_run[0])
    return None