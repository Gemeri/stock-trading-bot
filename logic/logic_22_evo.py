import os
import random
import copy
import logging
from dotenv import load_dotenv

import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- Logging Setup ---- #
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("EvoBot")

# ---- Environment Loading ---- #
load_dotenv()
BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "1Hour")

TIMEFRAME_MAP = {
    "4Hour": "H4",
    "2Hour": "H2",
    "1Hour": "H1",
    "30Min": "M30",
    "15Min": "M15"
}
CONVERTED_TIMEFRAME = TIMEFRAME_MAP.get(BAR_TIMEFRAME, BAR_TIMEFRAME)

def get_csv_filename(ticker: str) -> str:
    return os.path.join("data", f"{ticker}_{CONVERTED_TIMEFRAME}.csv")

# ---- Expression Grammar for Logic Evolution ---- #

FEATURES = [
    "current_price",
    "predicted_price",
    'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions', 'sentiment',
    'price_change', 'high_low_range', 'log_volume', 'macd_line', 'macd_signal', 'macd_histogram',
    'rsi', 'roc', 'atr', 'ema_9', 'ema_21', 'ema_50', 'ema_200', 'adx', 'obv', 'bollinger_upper', 
    'bollinger_lower', 'bollinger_percB', 'returns_1', 'returns_3', 'returns_5', 'std_5', 'std_10',
    'lagged_close_1', 'lagged_close_2', 'lagged_close_3', 'lagged_close_5', 'lagged_close_10',
    'candle_body_ratio', 'wick_dominance', 'gap_vs_prev', 'volume_zscore', 'atr_zscore', 'rsi_zscore',
    'macd_cross', 'macd_hist_flip', 'month', 'hour_sin', 'hour_cos', 'day_of_week_sin', 
    'day_of_week_cos', 'days_since_high', 'days_since_low',
]

OPERATORS = [
    "+", "-", "*", "/", ">", "<", "==", ">=", "<="
]

LOGIC_ACTIONS = [
    '"BUY"', '"SELL"', '"NONE"'
]

MAX_TREE_DEPTH = 4

# ---- Expression Tree Node ---- #

class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value  # Operator, feature, constant, or action
        self.left = left
        self.right = right

    def copy(self):
        if self.left is None and self.right is None:
            return Node(self.value)
        return Node(self.value, self.left.copy() if self.left else None, self.right.copy() if self.right else None)

    def __str__(self):
        if self.value in OPERATORS:
            return f"({self.left} {self.value} {self.right})"
        else:
            return str(self.value)

# ---- Generate Random Expression Tree ---- #

def random_constant():
    return round(random.uniform(-3, 3), 4)

def random_feature():
    return random.choice(FEATURES)

def random_operator():
    return random.choice(OPERATORS)

def random_action():
    return random.choice(LOGIC_ACTIONS)

def generate_condition_tree(depth=0):
    # Only features or constants for expressions, never LOGIC_ACTIONS!
    if depth > MAX_TREE_DEPTH or (depth > 0 and random.random() < 0.5):
        if random.random() < 0.7:
            return Node(random_feature())
        else:
            return Node(random_constant())
    op = random_operator()
    left = generate_condition_tree(depth+1)
    right = generate_condition_tree(depth+1)
    return Node(op, left, right)

def generate_logic_tree(depth=0):
    if depth > MAX_TREE_DEPTH or random.random() < 0.3:
        # Return just an action
        return Node(random_action())
    else:
        cond = generate_condition_tree()
        action = Node(random_action())
        else_branch = generate_logic_tree(depth+1)
        return Node("IF", cond, Node("THEN", action, else_branch))

def node_to_code(node, indent=1):
    space = "    " * indent
    if node is None:
        return f"{space}return \"NONE\""
    if node.value == "IF":
        cond_code = expr_to_code(node.left)
        then_node = node.right
        if then_node.value != "THEN":
            # Defensive: fallback
            return f"{space}if {cond_code}:\n{space}    return \"NONE\""
        action_code = node_to_code(then_node.left, indent + 1)
        else_code = node_to_code(then_node.right, indent + 1)
        s = f"{space}if {cond_code}:\n{action_code}\n"
        if else_code.strip() and "return" in else_code:
            s += f"{space}else:\n{else_code}\n"
        return s
    elif node.value == "THEN":
        # This node only groups two children: action and else branch
        return node_to_code(node.left, indent)
    elif node.value in LOGIC_ACTIONS:
        return f"{space}return {node.value}"
    elif node.value in OPERATORS:
        # fallback: should only occur in expr context
        return f"{space}{expr_to_code(node)}"
    else:
        return f"{space}{str(node.value)}"


def expr_to_code(node):
    if node is None:
        return "0"
    if node.value in OPERATORS:
        left = expr_to_code(node.left)
        right = expr_to_code(node.right)
        # If operator is arithmetic, cast left/right to int if boolean
        if node.value in ["+", "-", "*", "/"]:
            left = f"int({left})" if node.left and node.left.value in OPERATORS and node.left.value in [">", "<", "==", ">=", "<="] else left
            right = f"int({right})" if node.right and node.right.value in OPERATORS and node.right.value in [">", "<", "==", ">=", "<="] else right
        if node.value == "/":
            return f"({left}) / (({right}) if ({right}) != 0 else 1e-6)"
        else:
            return f"({left}) {node.value} ({right})"
    elif node.value in FEATURES:
        return node.value
    elif isinstance(node.value, (int, float)):
        return str(node.value)
    return "0"


def random_logic_tree():
    return generate_logic_tree()

def logic_to_live_code(node, indent=1):
    space = "    " * indent
    if node is None:
        return f"{space}logger.info(f\"No trade action taken.\")"
    if node.value == "IF":
        cond_code = expr_to_code(node.left)
        then_node = node.right
        if then_node.value != "THEN":
            return f"{space}if {cond_code}:\n{space}    logger.info(f\"No trade action taken.\")"
        action_code = logic_to_live_code(then_node.left, indent + 1)
        else_code = logic_to_live_code(then_node.right, indent + 1)
        s = f"{space}if {cond_code}:\n{action_code}\n"
        if else_code.strip():
            s += f"{space}else:\n{else_code}\n"
        return s
    elif node.value == "THEN":
        return logic_to_live_code(node.left, indent)
    elif node.value in LOGIC_ACTIONS:
        action = node.value.replace('"', '')  # remove quotes
        if action == "BUY":
            return (
                f"{space}if position_qty == 0:\n"
                f"{space}    max_shares = int(cash // current_price)\n"
                f"{space}    if max_shares > 0:\n"
                f"{space}        logger.info(f\"[{{ticker}}] Buying {{max_shares}} shares at {{current_price}}.\")\n"
                f"{space}        buy_shares(ticker, max_shares, current_price, predicted_price)\n"
                f"{space}    else:\n"
                f"{space}        logger.info(f\"[{{ticker}}] Insufficient cash to purchase shares.\")"
            )
        elif action == "SELL":
            return (
                f"{space}if position_qty > 0:\n"
                f"{space}    logger.info(f\"[{{ticker}}] Selling {{position_qty}} shares at {{current_price}}.\")\n"
                f"{space}    sell_shares(ticker, position_qty, current_price, predicted_price)\n"
                f"{space}else:\n"
                f"{space}    logger.info(f\"[{{ticker}}] No long position to sell; no action taken.\")"
            )
        else:  # NONE
            return f"{space}logger.info(f\"[{{ticker}}] No trade action taken.\")"
    else:
        return ""

# ---- Candidate (Algorithm) Representation ---- #

class Candidate:
    def __init__(self, logic_tree=None):
        self.logic_tree = logic_tree if logic_tree else random_logic_tree()
        self.fitness = None
        self.code_str = None

    def mutate(self):
        def mutate_node(node, depth=0):
            if node is None:
                return None
            if node.value == "IF" and random.random() < 0.2:
                # Mutate whole if branch
                return generate_logic_tree(depth)
            elif node.value == "THEN" and random.random() < 0.2:
                return Node("THEN", mutate_node(node.left, depth+1), mutate_node(node.right, depth+1))
            elif node.value in OPERATORS and random.random() < 0.3:
                node.value = random_operator()
            elif node.value in FEATURES and random.random() < 0.2:
                node.value = random_feature()
            elif isinstance(node.value, float) and random.random() < 0.4:
                node.value = random_constant()
            elif node.left is None and node.right is None and random.random() < 0.25:
                node.value = random_action()
            if node.left:
                node.left = mutate_node(node.left, depth+1)
            if node.right:
                node.right = mutate_node(node.right, depth+1)
            return node
        self.logic_tree = mutate_node(self.logic_tree)

    def crossover(self, other):
        child = copy.deepcopy(self)
        def get_random_subtree(node):
            nodes = []
            def collect(n):
                if n: nodes.append(n)
                if n.left: collect(n.left)
                if n.right: collect(n.right)
            collect(node)
            return random.choice(nodes) if nodes else node

        node1 = get_random_subtree(child.logic_tree)
        node2 = get_random_subtree(other.logic_tree)
        # Randomly swap
        if random.random() < 0.7:
            node1.value, node2.value = node2.value, node1.value
            node1.left, node2.left = node2.left, node1.left
            node1.right, node2.right = node2.right, node1.right
        return child


    def get_backtest_code(self):
        code = "def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):\n"
        for feat in FEATURES:
            if feat in ["current_price", "predicted_price"]:
                continue
            code += (
                f"    {feat} = candles.get('{feat}', None)\n"
                f"    if {feat} is None:\n"
                f"        {feat} = 0\n"
            )
        code += node_to_code(self.logic_tree, indent=1)
        if "return" not in code:
            code += "\n    return \"NONE\""
        return code


    def get_logic_code(self):
        """
        Generates a fully-self-contained run_logic() function for live trading.
        All features used by GP are fetched from the most-recent row of the
        CSV that matches the ticker and BAR_TIMEFRAME in the .env file.
        """
        # ---------- function header & common setup ----------
        code = '''def run_logic(current_price, predicted_price, ticker):
    """
    Auto-generated live-trading logic.
    Loads the latest feature row from the corresponding CSV each call.
    """
    import os
    import logging
    from functools import lru_cache
    import pandas as pd
    from dotenv import load_dotenv
    from forest import api, buy_shares, sell_shares

    logger = logging.getLogger(__name__)

    # -------------------- environment ---------------------
    load_dotenv()
    BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "1Hour")
    TIMEFRAME_MAP = {
        "4Hour": "H4",
        "2Hour": "H2",
        "1Hour": "H1",
        "30Min": "M30",
        "15Min": "M15"
    }
    CONVERTED_TIMEFRAME = TIMEFRAME_MAP.get(BAR_TIMEFRAME, BAR_TIMEFRAME)

    def _csv_filename(tkr: str) -> str:
        return f"{tkr}_{CONVERTED_TIMEFRAME}.csv"

    # Cache the last row per ticker so repeated calls are cheap
    @lru_cache(maxsize=16)
    def _latest_row(tkr: str):
        fname = _csv_filename(tkr)
        try:
            df = pd.read_csv(fname)
            if df.empty:
                return {}
            return df.iloc[-1].to_dict()
        except FileNotFoundError:
            logger.error(f"[{tkr}] CSV file '{fname}' not found.")
            return {}
        except Exception as exc:
            logger.error(f"[{tkr}] Error reading CSV '{fname}': {exc}")
            return {}

    latest = _latest_row(ticker)

    # ---------------- account / position -----------------
    try:
        account = api.get_account()
        cash = float(account.cash)
    except Exception as e:
        logger.error(f"[{ticker}] Error fetching account details: {e}")
        return

    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception:
        position_qty = 0.0

    logger.info(
        f"[{ticker}] Current Price: {current_price}, "
        f"Predicted Price: {predicted_price}, "
        f"Position: {position_qty}, Cash: {cash}"
    )

    # ---------------- feature assignments ----------------
'''
        # add a concrete assignment line for every feature except price inputs
        for feat in FEATURES:
            if feat in ("current_price", "predicted_price"):
                continue
            code += f"    {feat} = latest.get('{feat}', 0)\n"

        code += "\n    # -------------- decision logic --------------\n"

        # -------------- evolved decision tree ---------------
        code += logic_to_live_code(self.logic_tree, indent=1)
        return code


    def __str__(self):
        return self.get_backtest_code()

class Population:
    def __init__(self, size):
        self.candidates = [Candidate() for _ in range(size)]
        self.generation = 0

    def get_best(self):
        return max(self.candidates, key=lambda c: c.fitness)

    def select(self, k=8):
        # Tournament selection
        return sorted(self.candidates, key=lambda c: c.fitness, reverse=True)[:k]


def load_csv_data(ticker):
    fname = get_csv_filename(ticker)
    if not os.path.isfile(fname):
        raise FileNotFoundError(f"CSV file {fname} not found.")
    df = pd.read_csv(fname)
    # Ensure required columns exist
    needed_cols = {"timestamp", "close", "predicted_close"}
    for col in needed_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing in CSV.")
    # Backward fill features if missing
    for feat in FEATURES:
        if feat not in df.columns and feat not in ("current_price", "predicted_price"):
            df[feat] = np.nan
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)
    return df

def run_candidate_backtest(candidate, df):
    """
    Applies the candidate's logic to the historical data.
    Returns final account value.
    """
    code_str = candidate.get_backtest_code()
    local_scope = {}
    exec(code_str, {}, local_scope)
    run_backtest = local_scope['run_backtest']
    cash = 10000.0
    position_qty = 0
    position_price = 0
    equity_curve = []
    for i in range(2, len(df)):
        row = df.iloc[i]
        # Features from row
        features = {feat: row.get(feat, 0) for feat in FEATURES}
        features["current_price"] = float(row["close"])
        features["predicted_price"] = float(row["predicted_close"])
        current_price = features["current_price"]
        predicted_price = features["predicted_price"]
        ts = row["timestamp"]
        try:
            action = run_backtest(
                current_price=current_price,
                predicted_price=predicted_price,
                position_qty=position_qty,
                current_timestamp=ts,
                candles=features
            )
        except Exception as ex:
            logger.warning(f"Exception in candidate logic: {ex}")
            action = "NONE"
        # Emulate trades
        if action == "BUY" and position_qty == 0:
            qty = int(cash // current_price)
            if qty > 0:
                cash -= qty * current_price
                position_qty += qty
                position_price = current_price
        elif action == "SELL" and position_qty > 0:
            cash += position_qty * current_price
            position_qty = 0
            position_price = 0
        # Mark-to-market
        equity = cash + position_qty * current_price
        equity_curve.append(equity)
    # Fitness: final account value (or other metric)
    total_return = (equity_curve[-1] - 10000) / 10000
    # Optionally add penalties for overtrading or low return
    return total_return, equity_curve

# ---- Evolution Loop ---- #

def evolve(ticker,
    population_size=30,
    generations=30,
    elite_frac=0.3,
    mutation_rate=0.7,
    crossover_rate=0.6,
    patience=7,
):
    logger.info(f"Loading data for {ticker} / {CONVERTED_TIMEFRAME} ...")
    df = load_csv_data(ticker)
    pop = Population(population_size)
    best_fitness_history = []
    best_candidate = None
    best_fitness = -999
    stale = 0

    for gen in range(generations):
        logger.info(f"=== Generation {gen+1} ===")
        # Evaluate fitness
        for c in pop.candidates:
            try:
                c.fitness, _ = run_candidate_backtest(c, df)
            except Exception as ex:
                logger.warning(f"Fitness evaluation failed: {ex}")
                c.fitness = -999
        pop.candidates.sort(key=lambda c: c.fitness, reverse=True)
        best = pop.get_best()
        logger.info(f"Best fitness: {best.fitness:.4f}")
        best_fitness_history.append(best.fitness)
        if best.fitness > best_fitness:
            best_candidate = copy.deepcopy(best)
            best_fitness = best.fitness
            stale = 0
        else:
            stale += 1
        # Early stopping if not improving
        if stale >= patience:
            logger.info(f"Early stopping (no improvement in {patience} generations).")
            break
        # Next generation
        next_candidates = []
        elite_count = max(2, int(population_size * elite_frac))
        # Elitism: retain best
        next_candidates.extend(copy.deepcopy(c) for c in pop.candidates[:elite_count])
        # Generate rest via mutation/crossover
        while len(next_candidates) < population_size:
            if random.random() < crossover_rate:
                p1, p2 = random.sample(pop.candidates[:elite_count+5], 2)
                child = p1.crossover(p2)
                if random.random() < mutation_rate:
                    child.mutate()
                next_candidates.append(child)
            else:
                # Mutate elite
                c = copy.deepcopy(random.choice(pop.candidates[:elite_count]))
                if random.random() < mutation_rate:
                    c.mutate()
                next_candidates.append(c)
        pop.candidates = next_candidates
        pop.generation += 1

    # --- Report best logic and fitness --- #
    logger.info("\n" + "="*36)
    logger.info(" Best Evolved run_backtest:")
    print(best_candidate.get_backtest_code())
    logger.info("\n" + "="*36)
    logger.info(" Best Evolved run_logic (for live trading):")
    print(best_candidate.get_logic_code())
    logger.info(f"Fitness Score: {best_candidate.fitness:.4f}")
    # Plot fitness history
    try:
        plt.plot(best_fitness_history)
        plt.title("Best Fitness per Generation")
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.show()
    except Exception:
        pass
    return best_candidate

# ==============================================================
#  RUNTIME WRAPPERS  –  ALWAYS PRESENT FOR forest.py TO IMPORT
# ==============================================================

CACHE_LIMIT = 100
CACHE_DIR   = Path(__file__).parent / ".evobot_cache" 
LOGIC_FILE  = CACHE_DIR / "logic_src.py"
BT_FILE     = CACHE_DIR / "backtest_src.py"
META_FILE   = CACHE_DIR / "meta.json"

_gp_run_logic    = None
_gp_run_backtest = None
_remaining_calls = None        # loaded from meta or set to CACHE_LIMIT


# -------------------- helper: load cache -------------------- #
def _load_cache():
    global _gp_run_logic, _gp_run_backtest, _remaining_calls

    try:
        meta = json.loads(META_FILE.read_text())
        _remaining_calls = meta["remaining_calls"]
        logic_src  = LOGIC_FILE.read_text()
        bt_src     = BT_FILE.read_text()

        loc_ns = {}
        exec(logic_src, globals(), loc_ns)
        _gp_run_logic = loc_ns["run_logic"]

        loc_ns = {}
        exec(bt_src, globals(), loc_ns)
        _gp_run_backtest = loc_ns["run_backtest"]

        logger.info(f"Loaded GP strategy from disk cache "
                    f"(remaining_calls={_remaining_calls}).")
        return True
    except Exception:
        # cache missing or corrupted
        return False


# -------------------- helper: save cache -------------------- #
def _save_cache(best_cand, remaining):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logic_src = best_cand.get_logic_code()
    bt_src    = best_cand.get_backtest_code()

    LOGIC_FILE.write_text(logic_src)
    BT_FILE.write_text(bt_src)
    META_FILE.write_text(json.dumps({"remaining_calls": remaining}))

    logger.info("Strategy written to disk cache.")


# -------------------- helper: (re)build strategy ------------- #
def _build_gp_functions(ticker):
    global _gp_run_logic, _gp_run_backtest, _remaining_calls

    logger.info("Running GP evolution to (re)build strategy …")
    best = evolve(ticker)            # time-consuming search

    loc_ns = {}
    exec(best.get_logic_code(), globals(), loc_ns)
    _gp_run_logic = loc_ns["run_logic"]

    loc_ns = {}
    exec(best.get_backtest_code(), globals(), loc_ns)
    _gp_run_backtest = loc_ns["run_backtest"]

    _remaining_calls = CACHE_LIMIT if CACHE_LIMIT > 0 else 1
    _save_cache(best, _remaining_calls)


# -------------------- main refresh/initialise logic ---------- #
def _refresh_if_needed(ticker):
    global _remaining_calls

    if CACHE_LIMIT == 0:
        # no caching at all
        _build_gp_functions(ticker)
        return

    # first call in this process
    if _gp_run_logic is None or _gp_run_backtest is None:
        if not _load_cache():
            _build_gp_functions(ticker)

    # decrement counter and refresh if needed
    _remaining_calls -= 1
    if _remaining_calls <= 0:
        logger.info("CACHE_LIMIT reached → rebuilding strategy.")
        _build_gp_functions(ticker)
    else:
        # update meta file to persist new counter
        try:
            META_FILE.write_text(json.dumps({"remaining_calls": _remaining_calls}))
        except Exception:
            pass


# ---------------------- public wrappers ---------------------- #
def run_logic(current_price: float, predicted_price: float, ticker: str):
    _refresh_if_needed(ticker)
    return _gp_run_logic(current_price, predicted_price, ticker)


def run_backtest(
    current_price: float,
    predicted_price: float,
    position_qty: float,
    current_timestamp,
    candles,
    ticker
):
    _refresh_if_needed(ticker)

    if isinstance(candles, pd.DataFrame):
        try:
            row = candles.loc[candles["timestamp"] == current_timestamp].iloc[-1]
        except Exception:
            row = candles.iloc[-1]
        candle_dict = row.to_dict()
    else:
        candle_dict = candles

    return _gp_run_backtest(
        current_price=current_price,
        predicted_price=predicted_price,
        position_qty=position_qty,
        current_timestamp=current_timestamp,
        candles=candle_dict,
    )