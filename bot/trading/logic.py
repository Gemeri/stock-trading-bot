from forest import API_KEY, API_SECRET, API_BASE_URL, TRADE_LOGIC
from alpaca_trade_api.rest import REST, QuoteV2, APIError
import bot.ml.registry as registry
import pandas as pd
import importlib
import logging
import json
import sys
import re
import os

logger = logging.getLogger(__name__)

def get_current_price(ticker) -> float:
    key    = API_KEY
    secret = API_SECRET
    link   = API_BASE_URL
    if not key or not secret:
        print("Error: set ALPACA_API_KEY and ALPACA_API_SECRET in your environment.", file=sys.stderr)
        sys.exit(1)

    api = REST(key, secret, link, api_version="v2")

    try:
        quote: QuoteV2 = api.get_latest_quote(ticker)
    except APIError as e:
        print(f"API error fetching quote: {e}", file=sys.stderr)
        sys.exit(2)

    return (quote.bid_price + quote.ask_price) / 2

def get_logic_dir_and_json():
    logic_dir = "logic"
    json_path = os.path.join(logic_dir, "logic_scripts.json")
    return logic_dir, json_path

def _update_logic_json():
    logic_dir, json_path = get_logic_dir_and_json()
    if not os.path.isdir(logic_dir):
        os.makedirs(logic_dir)
    pattern = re.compile(r"^logic_(\d+)_(\w+)\.py$")
    scripts_map = {}
    for fname in os.listdir(logic_dir):
        match = pattern.match(fname)
        if match:
            num_str = match.group(1)
            script_base = fname[:-3]
            try:
                num_int = int(num_str)
                scripts_map[num_int] = script_base
            except ValueError:
                pass
    if not scripts_map:
        with open(json_path, "w") as f:
            json.dump({}, f, indent=2)
        return
    sorted_nums = sorted(scripts_map.keys())
    final_dict = {}
    for i in sorted_nums:
        final_dict[str(i)] = scripts_map[i]
    with open(json_path, "w") as f:
        json.dump(final_dict, f, indent=2)

def _get_logic_script_name(logic_id: str) -> str:
    """Resolve a logic identifier to a module name.

    Supports numeric IDs (mapped via ``logic_scripts.json``) as well as
    direct script filenames such as ``logic_1_forecast_driven.py`` or
    ``logic_1_forecast_driven``.
    """
    logic_dir, json_path = get_logic_dir_and_json()
    # If the user provided a direct filename, strip the extension and use it.
    if logic_id.lower().endswith('.py'):
        return logic_id[:-3]

    # If the user provided a script base (e.g. ``logic_1_forecast_driven``),
    # make sure it exists and return directly.
    candidate_path = os.path.join(logic_dir, f"{logic_id}.py")
    if logic_id.startswith('logic_') and os.path.isfile(candidate_path):
        return logic_id

    if not os.path.isfile(json_path):
        _update_logic_json()
    if not os.path.isfile(json_path):
        return "logic_15_forecast_driven"
    with open(json_path, "r") as f:
        data = json.load(f)
    return data.get(logic_id, "logic_15_forecast_driven")

def trade_logic(current_price: float, predicted_price: float, ticker: str):
    try:
        ml_models = registry.get_ml_models_for_ticker(ticker)
        classifier_stack = {"forest_cls", "xgboost_cls", "lightgbm_cls",
                            "transformer_cls", "sub-vote", "sub_meta",
                            "sub-meta", "sub_vote", "catboost_cls",
                            "catboost_multi", "lightgbm_multi",
                            "xgboost_multi"}
        logic_dir, _ = get_logic_dir_and_json()

        # ─── choose module ───────────────────────────────────────────────────
        if classifier_stack & set(ml_models):
            logic_module_name = "classifier"          # logic/classifier.py
        else:
            logic_module_name = _get_logic_script_name(str(TRADE_LOGIC))

        module_path   = f"{logic_dir}.{logic_module_name}"
        logic_module  = importlib.import_module(module_path)

        real_current  = get_current_price(ticker)
        logic_module.run_logic(real_current, predicted_price, ticker)

    except Exception as e:
        logging.error(f"Error dispatching to trade logic '{logic_module_name}': {e}")

def check_latest_candle_condition(df: pd.DataFrame, timeframe: str, scheduled_time_ny: str) -> bool:
    if df.empty:
        return False

    last_ts = pd.to_datetime(df.iloc[-1]['timestamp'])
    last_ts_str = last_ts.strftime("%H:%M:%S+00:00")
    
    expected = last_ts_str

    if timeframe == "4Hour":
        if scheduled_time_ny == "20:01":
            expected = "20:00:00+00:00"
    elif timeframe == "2Hour":
        if scheduled_time_ny == "10:01":
            expected = "12:00:00+00:00"
        elif scheduled_time_ny == "18:01":
            expected = "20:00:00+00:00"
    elif timeframe == "1Hour":
        if scheduled_time_ny == "11:01":
            expected = "12:00:00+00:00"
        elif scheduled_time_ny == "12:01":
            expected = "13:00:00+00:00"
        elif scheduled_time_ny == "16:01":
            expected = "20:00:00+00:00"
        elif scheduled_time_ny == "17:01":
            expected = "21:00:00+00:00"
    else:
        return True

    if last_ts_str == expected:
        return True
    else:
        logging.info(f"Latest candle time {last_ts_str} does not match expected {expected} for timeframe {timeframe} at scheduled NY time {scheduled_time_ny}.")
        return False