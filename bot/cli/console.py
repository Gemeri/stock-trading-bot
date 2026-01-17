import bot.selection.ranking as ranking
import bot.trading.logic as logicScript
import bot.selection.loader as loader
import bot.stuffs.candles as candles
import bot.ml.pipelines as pipelines
import bot.trading.orders as orders
import bot.cli.backtest as backtest
import market_timer.timer as timer
import bot.ml.stacking as stacking
import matplotlib.pyplot as plt
import bot.ml.models as models
from tqdm.auto import tqdm
from forest import api
import xgboost as xgb
import pandas as pd
import numpy as np
import schedule
import logging
import forest
import shap
import sys
import os

logger = logging.getLogger(__name__)

N_ESTIMATORS = 100
RANDOM_SEED = 42

def update_env_variable(key: str, value: str):
    env_path = ".env"
    lines = []
    found_key = False

    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = f.readlines()

    new_key_val = f"{key}={value}\n"
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = new_key_val
            found_key = True
            break

    if not found_key:
        lines.append(new_key_val)

    with open(env_path, "w") as f:
        f.writelines(lines)

    os.environ[key] = value
    logging.info(f"Updated .env: {key}={value}")

def console_listener():
    while not forest.SHUTDOWN:
        cmd_line = sys.stdin.readline().strip()
        if not cmd_line:
            continue
        parts = cmd_line.split()
        cmd = parts[0].lower()
        skip_data = ('-r' in parts)

        if cmd == "turnoff":
            logging.info("Received 'turnoff' command. Shutting down gracefully...")
            schedule.clear()
            forest.SHUTDOWN = True

        elif cmd == "api-test":
            logging.info("Testing Alpaca API keys...")
            try:
                account = api.get_account()
                logging.info(f"Account Cash: {account.cash}")
                logging.info("Alpaca API keys are valid (or we are in fake mode).")
            except Exception as e:
                logging.error(f"Alpaca API test failed: {e}")

        elif cmd == "get-data":
            timeframe = forest.BAR_TIMEFRAME
            if len(parts) > 1 and parts[1] != '-r':
                timeframe = parts[1]
            logging.info(f"Received 'get-data' command with timeframe={timeframe}.")
            for ticker in forest.TICKERS:
                df = candles.fetch_candles_plus_features(
                    ticker,
                    bars=forest.N_BARS,
                    timeframe=forest.BAR_TIMEFRAME,
                    rewrite_mode=forest.REWRITE
                )

                tf_code = candles.timeframe_to_code(timeframe)
                ticker_fs = candles.fs_safe_ticker(ticker)
                csv_filename = candles.candle_csv_path(ticker_fs, tf_code)
                df.to_csv(csv_filename, index=False)
                logging.info(f"[{ticker}] Saved candle data + advanced features (minus disabled) to {csv_filename}.")

        elif cmd == "predict-next":
            tf_code = candles.timeframe_to_code(forest.BAR_TIMEFRAME)
            ticker_fs = candles.fs_safe_ticker(forest.BACKTEST_TICKER)
            csv_filename = candles.candle_csv_path(ticker_fs, tf_code)
            if skip_data:
                logging.info(f"[{forest.BACKTEST_TICKER}] predict-next -r: Using existing CSV {csv_filename}")
                if not os.path.exists(csv_filename):
                    logging.error(f"[{forest.BACKTEST_TICKER}] CSV does not exist, skipping.")
                    continue
                df = candles.read_csv_limited(csv_filename)
                if df.empty:
                    logging.error(f"[{forest.BACKTEST_TICKER}] CSV is empty, skipping.")
                    continue
            else:
                df = candles.fetch_candles_plus_features(
                    forest.BACKTEST_TICKER,
                    bars=forest.N_BARS,
                    timeframe=forest.BAR_TIMEFRAME,
                    rewrite_mode=forest.REWRITE
                )
                df.to_csv(csv_filename, index=False)
                logging.info(f"[{forest.BACKTEST_TICKER}] Fetched new data + advanced features (minus disabled), saved to {csv_filename}")

            allowed = set(forest.POSSIBLE_FEATURE_COLS) | {"timestamp"}
            df = df.loc[:, [c for c in df.columns if c in allowed]]

            pred_close = stacking.train_and_predict(df, forest.BACKTEST_TICKER)
            if isinstance(pred_close, tuple):
                pred_close = pred_close[0]
            if pred_close is None:
                logging.error(f"[{forest.BACKTEST_TICKER}] No prediction generated.")
                continue
            current_price = float(df.iloc[-1]['close'])
            try:
                pred_val = float(pred_close)
            except Exception:
                logging.error(f"[{forest.BACKTEST_TICKER}] Prediction not numeric: {pred_close}")
                continue
            logging.info(
                f"[{forest.BACKTEST_TICKER}] Current Price={current_price:.2f}, Predicted Next Close={pred_val:.2f}"
            )

        elif cmd == "force-run":
            logging.info("Force-running job now (ignoring market open check).")
            if not skip_data:
                forest._perform_trading_job(skip_data=False)
            else:
                logging.info("force-run -r: skipping data fetch, using existing CSV + skipping new sentiment fetch.")
                forest._perform_trading_job(skip_data=True)
            logging.info("Force-run job finished.")

        elif cmd == "backtest":
            backtest.run_backtest(parts)

        elif cmd == "get-best-tickers":
            n = 1
            if len(parts) > 1 and parts[1] != "-r":
                try:
                    n = int(parts[1])
                except ValueError:
                    logging.error("Invalid number for get-best-tickers.")
                    continue
            result = ranking.select_best_tickers(top_n=n, skip_data=skip_data)
            logging.info(f"Best tickers: {result}")

        elif cmd == "feature-importance":

            # ──────────────────────────────────────────────────────────────────────────
            # Configs
            # ──────────────────────────────────────────────────────────────────────────
            TRAIN_PCT           = 0.80  # percentage of rows used as the expanding-window start
            MIN_TRAIN_SAMPLES   = 50    # minimum samples required to fit a model at any step
            RANDOM_STATE        = 42
            SAVE_DIR_ROOT       = os.path.join(forest.DATA_DIR, "feature-importance")
            os.makedirs(SAVE_DIR_ROOT, exist_ok=True)

            def _clean_and_select_features(df: pd.DataFrame):
                # If your pipeline normally engineers features from raw OHLCV, you may call them here.
                # Comment these in if needed in your project:
                # df = add_features(df)
                # df = compute_custom_features(df)

                # Select usable features
                feats = [c for c in forest.POSSIBLE_FEATURE_COLS if c in df.columns]
                if not feats:
                    raise RuntimeError("No overlap between POSSIBLE_FEATURE_COLS and CSV columns.")
                X = df[feats].copy()

                # Replace inf with nan, then we'll handle nan at train/test time
                X.replace([np.inf, -np.inf], np.nan, inplace=True)
                return X, feats

            def _make_labels(df: pd.DataFrame):
                # Regression labels
                y_reg_h1  = df['close'].shift(-1)
                y_reg_h50 = df['close'].shift(-50)

                # Classification labels: 1 if next close rises relative to current close, else 0
                y_clf_h1  = (df['close'].shift(-1)  > df['close']).astype(float)
                y_clf_h50 = (df['close'].shift(-50) > df['close']).astype(float)

                # Ensure proper NaN handling for trailing rows
                return y_reg_h1, y_clf_h1, y_reg_h50, y_clf_h50

            def _walkforward_shap_importance(X: pd.DataFrame,
                                             y: pd.Series,
                                             problem: str,
                                             horizon_gap: int) -> np.ndarray:
                """
                Expanding-window walk-forward SHAP importance with XGBoost.
                For horizon_gap=0: train[:i], predict[i]
                For horizon_gap=50: train[:i-50], predict[i]
                Accumulate mean |SHAP| across steps.
                """
                n = len(X)
                start_idx = int(n * TRAIN_PCT)
                start_idx = max(start_idx, horizon_gap + 1)  # ensure train_end will be >= 1
                end_idx   = n - horizon_gap  # last index with a non-NaN label

                if problem == "reg":
                    model_kwargs = dict(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        tree_method="hist",
                        eval_metric="rmse",
                    )
                else:
                    model_kwargs = dict(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        tree_method="hist",
                        eval_metric="logloss",
                        use_label_encoder=False,
                    )

                accum = np.zeros(X.shape[1], dtype=float)
                steps = 0

                # Progress bar
                pbar = tqdm(range(start_idx, end_idx),
                            desc=f"SHAP ({problem}, h={horizon_gap})",
                            leave=False)

                for i in pbar:
                    train_end = i - horizon_gap  # exclusive end for training
                    if train_end < MIN_TRAIN_SAMPLES:
                        continue

                    # Prepare train split; drop rows with NaNs in features or label
                    X_train = X.iloc[:train_end]
                    y_train = y.iloc[:train_end]
                    mask_tr = X_train.notna().all(axis=1) & y_train.notna()
                    X_train = X_train[mask_tr]
                    y_train = y_train[mask_tr]

                    # Ensure enough samples after NaN filtering
                    if len(X_train) < MIN_TRAIN_SAMPLES:
                        continue

                    # For classifiers, labels should be ints (0/1)
                    if problem == "clf":
                        y_train = y_train.astype(int)

                    # Prepare single test row; skip if NaN present
                    X_test = X.iloc[i:i+1]
                    if (not X_test.notna().all(axis=1).iloc[0]) or (not pd.notna(y.iloc[i])):
                        continue

                    # Fit fresh XGBoost model at this step
                    if problem == "reg":
                        model = xgb.XGBRegressor(**model_kwargs)
                    else:
                        model = xgb.XGBClassifier(**model_kwargs)

                    pipelines.fixed_fit(X_train, y_train, model)

                    # SHAP for the single test row
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_vals = explainer.shap_values(X_test)
                    except Exception:
                        # Fallback to model-agnostic Explainer if needed
                        explainer = shap.Explainer(model.predict, X_train, seed=RANDOM_STATE)
                        sv_exp = explainer(X_test)
                        shap_vals = sv_exp.values

                    # Normalize various SHAP return types to a 1D feature vector for this sample
                    if isinstance(shap_vals, list):
                        # list per class -> take positive class (1)
                        sv = np.array(shap_vals[1])[0]
                    elif hasattr(shap_vals, "values"):
                        sv_arr = np.array(shap_vals.values)
                        sv = sv_arr[0] if sv_arr.ndim > 1 else sv_arr
                    else:
                        sv = np.array(shap_vals)
                        sv = sv[0] if sv.ndim > 1 else sv

                    accum += np.abs(sv)
                    steps += 1

                if steps == 0:
                    raise RuntimeError("Walk-forward produced 0 valid steps; check data/labels.")
                return accum / steps  # mean absolute SHAP per feature

            def _save_rankings_and_plot(out_dir: str, label_name: str, features: list, importances: np.ndarray):
                os.makedirs(out_dir, exist_ok=True)
                ranking = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)

                # Save text ranking
                txt_path = os.path.join(out_dir, f"{label_name}.txt")
                with open(txt_path, "w") as f:
                    for idx, (feat, imp) in enumerate(ranking, 1):
                        f.write(f"{idx:>2}. {feat}: {imp:.10f}\n")

                # Save bar chart
                plt.figure(figsize=(12, max(4, len(features) * 0.25)))
                ordered_feats = [r[0] for r in ranking][::-1]
                ordered_imps  = [r[1] for r in ranking][::-1]
                plt.barh(ordered_feats, ordered_imps)
                plt.xlabel("Mean |SHAP value| (walk-forward)")
                plt.title(f"Feature Importance — {label_name}")
                plt.tight_layout()
                png_path = os.path.join(out_dir, f"{label_name}.png")
                plt.savefig(png_path, dpi=200)
                plt.close()

            # ──────────────────────────────────────────────────────────────────────────
            # Main per-ticker loop (expects 'ticker' variable to be defined in your outer loop)
            # ──────────────────────────────────────────────────────────────────────────
            tf_code = candles.timeframe_to_code(forest.BAR_TIMEFRAME)
            csv_file = candles.candle_csv_path(forest.BACKTEST_TICKER, tf_code)
            out_dir  = os.path.join(SAVE_DIR_ROOT, forest.BACKTEST_TICKER)
            os.makedirs(out_dir, exist_ok=True)

            if not os.path.exists(csv_file):
                logging.error(f"[{forest.BACKTEST_TICKER}] CSV {csv_file} not found. Aborting feature-importance for this ticker.")
            else:
                # Load CSV
                try:
                    df = pd.read_csv(csv_file)
                except Exception as e:
                    logging.error(f"[{forest.BACKTEST_TICKER}] Failed to read CSV: {e}")
                    df = pd.DataFrame()

                if df.empty or 'close' not in df.columns:
                    logging.error(f"[{forest.BACKTEST_TICKER}] CSV empty or missing 'close' column. Skipping.")
                else:
                    # Prepare features and labels
                    X, features = _clean_and_select_features(df)
                    y_reg_h1, y_clf_h1, y_reg_h50, y_clf_h50 = _make_labels(df)

                    # Ensure alignment
                    y_reg_h1  = y_reg_h1.reindex(X.index)
                    y_clf_h1  = y_clf_h1.reindex(X.index)
                    y_reg_h50 = y_reg_h50.reindex(X.index)
                    y_clf_h50 = y_clf_h50.reindex(X.index)

                    # 1) Regressor i+1
                    logging.info(f"[{forest.BACKTEST_TICKER}] Running SHAP feature-importance (XGBoost): Regressor next close (i+1).")
                    imp_reg_h1 = _walkforward_shap_importance(X, y_reg_h1, problem="reg", horizon_gap=0)
                    _save_rankings_and_plot(out_dir, f"{tf_code}_reg_next_close_i+1", features, imp_reg_h1)

                    # 2) Classifier i+1 (up/down)
                    logging.info(f"[{forest.BACKTEST_TICKER}] Running SHAP feature-importance (XGBoost): Classifier next close up/down (i+1).")
                    imp_clf_h1 = _walkforward_shap_importance(X, y_clf_h1, problem="clf", horizon_gap=0)
                    _save_rankings_and_plot(out_dir, f"{tf_code}_clf_next_close_updown_i+1", features, imp_clf_h1)

                    # 3) Regressor i+50  (train up to i-50)
                    logging.info(f"[{forest.BACKTEST_TICKER}] Running SHAP feature-importance (XGBoost): Regressor next close (i+50), train up to i-50.")
                    imp_reg_h50 = _walkforward_shap_importance(X, y_reg_h50, problem="reg", horizon_gap=50)
                    _save_rankings_and_plot(out_dir, f"{tf_code}_reg_next_close_i+50", features, imp_reg_h50)

                    # 4) Classifier i+50 (train up to i-50)
                    logging.info(f"[{forest.BACKTEST_TICKER}] Running SHAP feature-importance (XGBoost): Classifier next close up/down (i+50), train up to i-50.")
                    imp_clf_h50 = _walkforward_shap_importance(X, y_clf_h50, problem="clf", horizon_gap=50)
                    _save_rankings_and_plot(out_dir, f"{tf_code}_clf_next_close_updown_i+50", features, imp_clf_h50)

                    # 5) Average across all four tests
                    logging.info(f"[{forest.BACKTEST_TICKER}] Aggregating average importance across all four tests.")
                    all_imps = np.vstack([imp_reg_h1, imp_clf_h1, imp_reg_h50, imp_clf_h50])
                    imp_avg  = all_imps.mean(axis=0)
                    _save_rankings_and_plot(out_dir, f"{tf_code}_AVERAGE_of_all_tests", features, imp_avg)

                    logging.info(f"[{forest.BACKTEST_TICKER}] SHAP feature-importance done. Outputs in: {out_dir}")


        elif cmd == "commands":
            logging.info("Available commands:")
            logging.info("  turnoff")
            logging.info("  api-test")
            logging.info("  get-data [timeframe]")
            logging.info("  predict-next [-r]")
            logging.info("  force-run [-r]")
            logging.info("  backtest <N> [simple|complex] [timeframe?] [-r?]")
            logging.info("  get-best-tickers <N> [-r]")
            logging.info("  feature-importance [-r]")
            logging.info("  set-timeframe (timeframe)")
            logging.info("  set-nbars (Number of candles)")
            logging.info("  trade-logic <logic>")
            logging.info("  set-ntickers (Number)")
            logging.info("  ai-tickers")
            logging.info("  buy (ticker) <amount|$dollars>")
            logging.info("  sell (ticker) <amount|$dollars>")
            logging.info("  commands")

        elif cmd == "backtest-execute":
            if len(parts) < 4:
                logging.info("Usage: backtest-execute <label number> <model name> <direction number (0 SELL, 1 BUY)>")
                continue
            tf_code = candles.timeframe_to_code(forest.BAR_TIMEFRAME)
            ticker_fs = candles.fs_safe_ticker(forest.BACKTEST_TICKER)
            csv_filename = candles.candle_csv_path(ticker_fs, tf_code)
            df = candles.read_csv_limited(csv_filename)
            label = int(parts[1])
            direction = int(parts[3])
            timer.execution_backtest(df, forest.BACKTEST_TICKER, label=label, model=parts[2], direction=direction)
        elif cmd == "get-execution":
            if len(parts) < 4:
                logging.info("Usage: get-execution <label number> <model name> <direction number (0 SELL, 1 BUY)>")
                continue
            tf_code = candles.timeframe_to_code(forest.BAR_TIMEFRAME)
            ticker_fs = candles.fs_safe_ticker(forest.BACKTEST_TICKER)
            csv_filename = candles.candle_csv_path(ticker_fs, tf_code)
            df = candles.read_csv_limited(csv_filename)
            timer.get_execution_decision(df, forest.BACKTEST_TICKER, parts[1], parts[2], parts[3])
        elif cmd == "set-timeframe":
            if len(parts) < 2:
                logging.info("Usage: set-timeframe 4Hour/1Day/etc.")
                continue
            new_tf = parts[1]
            update_env_variable("BAR_TIMEFRAME", new_tf)
            forest.BAR_TIMEFRAME = new_tf
            logging.info(f"Updated BAR_TIMEFRAME in memory to {forest.BAR_TIMEFRAME}")

        elif cmd == "set-nbars":
            if len(parts) < 2:
                logging.info("Usage: set-nbars 5000")
                continue
            new_nbars_str = parts[1]
            try:
                new_nbars = int(new_nbars_str)
                update_env_variable("N_BARS", str(new_nbars))
                forest.N_BARS = new_nbars
                logging.info(f"Updated N_BARS in memory to {forest.N_BARS}")
            except Exception as e:
                logging.error(f"Cannot parse set-nbars: {e}")

        elif cmd == "trade-logic":
            if len(parts) < 2:
                logging.info("Usage: trade-logic <logicValue>  # e.g. 1..15")
                continue
            new_logic = parts[1]
            update_env_variable("TRADE_LOGIC", new_logic)
            forest.TRADE_LOGIC = new_logic
            logging.info(f"Updated TRADE_LOGIC in memory to {forest.TRADE_LOGIC}")
            
        elif cmd == "set-ntickers":
            if len(parts) < 2:
                logging.info("Usage: set-ntickers <intValue>")
                continue
            new_val_str = parts[1]
            try:
                new_val_int = int(new_val_str)
            except ValueError:
                logging.error(f"Invalid integer for set-ntickers: {new_val_str}")
                continue

            update_env_variable("AI_TICKER_COUNT", str(new_val_int))
            forest.AI_TICKER_COUNT = new_val_int
            logging.info(f"Updated AI_TICKER_COUNT in memory to {forest.AI_TICKER_COUNT}")


        elif cmd == "ai-tickers":
            if forest.AI_TICKER_COUNT <= 0:
                logging.info("AI_TICKER_COUNT is 0 or not set. No AI tickers to fetch.")
                continue

            new_ai_list = loader.fetch_new_ai_tickers(forest.AI_TICKER_COUNT, exclude_tickers=[])
            if not new_ai_list:
                logging.info("No AI tickers returned or an error occurred.")
            else:
                logging.info(f"AI recommended tickers: {new_ai_list}")

        elif cmd == "buy":
            if len(parts) < 3:
                logging.info("Usage: buy <ticker> <amount|$dollars>")
                continue

            ticker_use = parts[1].upper()
            amount_str = parts[2]
            live_price = logicScript.get_current_price(ticker_use)

            if amount_str.startswith("$"):
                try:
                    dollars = float(amount_str[1:])
                except ValueError:
                    logging.warning("Dollar amount must be numeric.")
                    continue
                qty = dollars / live_price if live_price > 0 else 0
                if qty <= 0:
                    logging.warning("Dollar amount too small to buy.")
                    continue
                old_full = forest.USE_FULL_SHARES
                forest.USE_FULL_SHARES = False
                orders.buy_shares(ticker_use, qty, live_price, live_price)
                forest.USE_FULL_SHARES = old_full
                logging.info(f"Attempted to buy ${dollars} of {ticker_use} @ {live_price}")
            else:
                try:
                    amount = int(amount_str)
                except ValueError:
                    logging.warning("Amount must be an integer or $dollar value.")
                    continue
                orders.buy_shares(ticker_use, amount, live_price, live_price)
                logging.info(f"Attempted to buy {amount}×{ticker_use} @ {live_price}")

        elif cmd == "sell":
            if len(parts) < 3:
                logging.info("Usage: sell <ticker> <amount|$dollars>")
                continue

            ticker_use = parts[1].upper()
            amount_str = parts[2]
            live_price = logicScript.get_current_price(ticker_use)

            if amount_str.startswith("$"):
                try:
                    dollars = float(amount_str[1:])
                except ValueError:
                    logging.warning("Dollar amount must be numeric.")
                    continue
                qty = dollars / live_price if live_price > 0 else 0
                try:
                    position = api.get_position(ticker_use)
                    owned_qty = float(position.qty)
                except Exception:
                    logging.info(f"You don’t own any shares of {ticker_use}.")
                    continue
                if qty > owned_qty:
                    logging.info(
                        f"Not enough shares to sell: requested ${dollars}, "
                        f"which is {qty:.4f} shares, but you own only {owned_qty} of {ticker_use}."
                    )
                    continue
                old_full = forest.USE_FULL_SHARES
                forest.USE_FULL_SHARES = False
                orders.sell_shares(ticker_use, qty, live_price, live_price)
                forest.USE_FULL_SHARES = old_full
                logging.info(f"Attempted to sell ${dollars} of {ticker_use} @ {live_price}")
            else:
                try:
                    amount = int(amount_str)
                except ValueError:
                    logging.warning("Amount must be an integer or $dollar value.")
                    continue

                try:
                    position = api.get_position(ticker_use)
                    owned_qty = int(float(position.qty))
                except Exception:
                    logging.info(f"You don’t own any shares of {ticker_use}.")
                    continue

                if amount > owned_qty:
                    logging.info(
                        f"Not enough shares to sell: requested {amount}, "
                        f"but you own only {owned_qty} of {ticker_use}."
                    )
                    continue
                orders.sell_shares(ticker_use, amount, live_price, live_price)
                logging.info(f"Attempted to sell {amount}×{ticker_use} @ {live_price}")

        elif cmd == "train-sub":
            if forest.SUB_VERSION == "beta":
                if len(parts) < 2:
                    logging.info("Usage: train-sub <ticker/all>")
                else:
                    ticker_or_all = parts[1].lower()
                    logging.info("Training submodel...")
                    if ticker_or_all == "all":
                        for ticker in loader.load_tickerlist:
                            logging.info(f"Now training {ticker}...")
                            csv_path = candles.candle_csv_path(ticker, candles.timeframe_to_code(forest.BAR_TIMEFRAME))
                            if not os.path.exists(csv_path):
                                logging.info(f"CSV '{csv_path}' does not exist. Skipping {ticker}.")
                                continue
                            df = pd.read_csv(csv_path)
                            models.call_sub_main(ticker, df, -1)
                    else:
                        csv_path = candles.candle_csv_path(ticker_or_all, candles.timeframe_to_code(forest.BAR_TIMEFRAME))
                        if not os.path.exists(csv_path):
                            logging.info(f"CSV '{csv_path}' does not exist")
                        else:
                            df = pd.read_csv(csv_path)
                            models.call_sub_main(ticker_or_all, df, -1)
            else:
                logging.info("This functionality only exists for the beta version")

        else:
            logging.warning(f"Unrecognized command: {cmd_line}")