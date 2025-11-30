import os
import random
import copy
import logging
from dotenv import load_dotenv
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import re
try:
    import config
except ImportError:
    print("Not running from script")
from typing import List, Tuple, Dict, Optional


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

# ---- Triple-Barrier / Selection Hyperparams (tunable via .env) ---- #
TB_H = 5
TB_THETA_UP = 0.01
TB_THETA_DN = 0.01

TRAIN_FRAC = 0.8
META_POLICY = False
# Composite objective weights (validation-based)
LAMBDA_MDD = 0.10
LAMBDA_TURNOVER = 0.01
LAMBDA_COMPLEXITY = 0.001

# Complexity decomposition
ALPHA_COMPLEXITY = 1.0
BETA_COMPLEXITY = 5.0
GAMMA_FEATURES = 0.25

# Diversity filter among elites
DIVERSITY_TOPK = 10
DIVERSITY_CORR_MAX = 0.92

# Early stopping
EARLY_STOP_PATIENCE = 25

# Repro
random.seed(int(1337))
np.random.seed(int(1337))

def get_csv_filename(ticker: str) -> str:
    fname = f"{ticker}_{CONVERTED_TIMEFRAME}.csv"

    # 1) Running from project root
    p1 = Path("data") / fname
    if p1.is_file():
        return str(p1)

    # 2) Running this file from ./logic (use file location to resolve root)
    try:
        file_dir = Path(__file__).resolve()
        p2 = file_dir.parent.parent / "data" / fname
        if p2.is_file():
            return str(p2)
    except NameError:
        pass

    # 3) Fallback: current working directory’s parent (if CWD is ./logic)
    p3 = Path.cwd().parent / "data" / fname
    if p3.is_file():
        return str(p3)

    # Nothing found — raise with helpful message
    tried = [
        str(p1.absolute()),
        str((Path(__file__).resolve().parent.parent / "data" / fname) if "__file__" in globals() else "(no __file__ available)"),
        str(p3.absolute()),
    ]
    raise FileNotFoundError(
        "CSV file not found for ticker/timeframe.\n"
        "Tried:\n  - " + "\n  - ".join(tried)
    )

FEATURES = [
    "current_price",
    'volume', 'vwap', 'transactions', 'sentiment',
    'greed_index', 'news_count', 'news_volume_z', 'price_change', 'high_low_range', 
    'macd_line', 'macd_signal', 'macd_histogram', 'rsi', 'roc', 'atr', 'ema_9', 'ema_21', 
    'ema_50', 'ema_200', 'adx', 'obv', 'bollinger_percB', 'returns_1', 
    'returns_3', 'returns_5', 'std_5', 'std_10', 'lagged_close_1', 'lagged_close_2', 
    'lagged_close_3', 'lagged_close_5', 'lagged_close_10', 'candle_body_ratio', 
    'wick_dominance', 'gap_vs_prev', 'volume_zscore', 'atr_zscore', 'rsi_zscore',
    'month', 'hour_sin', 'hour_cos', 'day_of_week_sin',  'day_of_week_cos', 
    'days_since_high', 'days_since_low', "d_sentiment"
]

OPERATORS = [
    "+", "-", "*", "/", ">", "<", "==", ">=", "<="
]

LOGIC_ACTIONS = [
    '"BUY"', '"SELL"', '"NONE"'
]

MAX_TREE_DEPTH = 6

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

def _ticker_cache_dir(ticker: str) -> Path:
    safe = "".join(ch for ch in str(ticker) if ch.isalnum() or ch in ("-", "_"))
    return CACHE_DIR / safe


def _logic_file(ticker: str) -> Path:
    return _ticker_cache_dir(ticker) / "logic_src.py"

def _bt_file(ticker: str) -> Path:
    return _ticker_cache_dir(ticker) / "backtest_src.py"

def _meta_file(ticker: str) -> Path:
    return _ticker_cache_dir(ticker) / "meta.json"

def _scorecard_file(ticker: str) -> Path:
    return _ticker_cache_dir(ticker) / "scorecard.json"

def _fitplot_file(ticker: str) -> Path:
    return _ticker_cache_dir(ticker) / "fitness_history.png"

_gp_run_logic_map: dict = {}
_gp_run_backtest_map: dict = {}
_remaining_calls_map: dict = {}

def random_feature():
    return random.choice(FEATURES)

def random_operator():
    return random.choice(OPERATORS)

def random_action():
    # We allow NONE leaves during tree generation, but they fold into final else.
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
        return Node(random_action())
    else:
        cond = generate_condition_tree()
        action = Node(random_action())
        else_branch = generate_logic_tree(depth+1)
        return Node("IF", cond, Node("THEN", action, else_branch))

# --------- DNF Emitter (Flat BUY/SELL/ELSE NONE Chain) ---------- #

def node_to_code(node, indent=1):
    space = "    " * indent

    def _and(a: str, b: str) -> str:
        if a in (None, "", "True"):
            return f"({b})"
        return f"(({a}) and ({b}))"

    def _not(x: str) -> str:
        return f"not ({x})"

    def _collect_action_conditions(n, path_cond="True", acc=None):
        """
        Accumulates a dict: {'BUY': [cond...], 'SELL':[...], 'NONE':[...]}.
        """
        if acc is None:
            acc = {"BUY": [], "SELL": [], "NONE": []}
        if n is None:
            acc["NONE"].append(path_cond)
            return acc

        if getattr(n, "value", None) == "IF":
            cond_code = expr_to_code(n.left) or "False"
            then_node = n.right
            if not then_node or getattr(then_node, "value", None) != "THEN":
                acc["NONE"].append(_and(path_cond, cond_code))
                acc["NONE"].append(_and(path_cond, _not(cond_code)))
                return acc
            _collect_action_conditions(then_node.left,  _and(path_cond, cond_code), acc)
            _collect_action_conditions(then_node.right, _and(path_cond, _not(cond_code)), acc)
            return acc

        if getattr(n, "value", None) == "THEN":
            _collect_action_conditions(n.left, path_cond, acc)
            _collect_action_conditions(n.right, path_cond, acc)
            return acc

        if isinstance(n.value, str) and n.value.replace('"', '') in ("BUY", "SELL", "NONE"):
            acc[n.value.replace('"', '')].append(path_cond)
            return acc

        acc["NONE"].append(path_cond)
        return acc

    def _or_join(conds):
        if not conds:
            return "False"
        seen = set()
        uniq = []
        for c in conds:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        if len(uniq) == 1:
            return uniq[0]
        return " or ".join(f"({c})" for c in uniq)

    acc = _collect_action_conditions(node, "True", None)
    buy_expr = _or_join(acc.get("BUY", []))
    sell_expr = _or_join(acc.get("SELL", []))

    lines = []
    lines.append(f"{space}if {buy_expr}:")
    lines.append(f"{space}    return \"BUY\"")
    lines.append(f"{space}elif {sell_expr}:")
    lines.append(f"{space}    return \"SELL\"")
    lines.append(f"{space}else:")
    lines.append(f"{space}    return \"NONE\"")
    return "\n".join(lines)


def expr_to_code(node):
    if node is None:
        return "0"
    if node.value in OPERATORS:
        left = expr_to_code(node.left)
        right = expr_to_code(node.right)
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

# ----------------- Live Logic Emitter with Meta-Gate ---------------- #

def logic_to_live_code(node, indent=1, meta_cfg: Optional[dict]=None):
    # base indents
    space = " " * 4 * indent
    i1 = " " * 4 * (indent + 1)
    i2 = " " * 4 * (indent + 2)
    i3 = " " * 4 * (indent + 3)
    i4 = " " * 4 * (indent + 4)

    def _and(a: str, b: str) -> str:
        if a in (None, "", "True"):
            return f"({b})"
        return f"(({a}) and ({b}))"

    def _not(x: str) -> str:
        return f"not ({x})"

    def _collect_action_conditions(n, path_cond="True", acc=None):
        if acc is None:
            acc = {"BUY": [], "SELL": [], "NONE": []}
        if n is None:
            acc["NONE"].append(path_cond)
            return acc
        if getattr(n, "value", None) == "IF":
            cond_code = expr_to_code(n.left) or "False"
            then_node = n.right
            if not then_node or getattr(then_node, "value", None) != "THEN":
                acc["NONE"].append(_and(path_cond, cond_code))
                acc["NONE"].append(_and(path_cond, _not(cond_code)))
                return acc
            _collect_action_conditions(then_node.left,  _and(path_cond, cond_code), acc)
            _collect_action_conditions(then_node.right, _and(path_cond, _not(cond_code)), acc)
            return acc
        if getattr(n, "value", None) == "THEN":
            _collect_action_conditions(n.left, path_cond, acc)
            _collect_action_conditions(n.right, path_cond, acc)
            return acc
        if isinstance(n.value, str) and n.value.replace('"', '') in ("BUY", "SELL", "NONE"):
            acc[n.value.replace('"', '')].append(path_cond)
            return acc
        acc["NONE"].append(path_cond)
        return acc

    def _or_join(conds):
        if not conds:
            return "False"
        seen = set()
        uniq = []
        for c in conds:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        if len(uniq) == 1:
            return uniq[0]
        return " or ".join(f"({c})" for c in uniq)

    # Build BUY/SELL boolean expressions from the tree
    acc = _collect_action_conditions(node, "True", None)
    buy_expr = _or_join(acc.get("BUY", []))
    sell_expr = _or_join(acc.get("SELL", []))

    # ---- Optional meta-gate emitter ----
    meta_lines = []
    has_meta = bool(meta_cfg and meta_cfg.get("enabled", False))
    if has_meta:
        feat_names: List[str] = meta_cfg["feature_names"]
        coef = meta_cfg["coef"]
        intercept = meta_cfg["intercept"]
        means = meta_cfg["means"]
        stds = meta_cfg["stds"]

        meta_lines.append(f"{space}def _meta_should_take_trade(direction:int):")
        meta_lines.append(f"{i1}# Logistic meta-gate: features (z-scored by train stats) + direction")
        meta_lines.append(f"{i1}x = []")
        for nm in feat_names:
            if nm == "__direction__":  # last feature is the proposed direction
                meta_lines.append(f"{i1}x.append(direction)")
            else:
                mu = means.get(nm, 0.0)
                sd = stds.get(nm, 1.0) or 1.0
                meta_lines.append(f"{i1}val = latest.get('{nm}', 0)")
                meta_lines.append(f"{i1}x.append((val - ({mu})) / ({sd}))")
        meta_lines.append(f"{i1}lin = ({intercept})")
        for j, _ in enumerate(feat_names):
            cj = coef[j]
            meta_lines.append(f"{i1}lin += ({cj}) * x[{j}]")
        meta_lines.append(f"{i1}p = 1.0 / (1.0 + np.exp(-lin))")
        meta_lines.append(f"{i1}return p >= 0.5")
    else:
        meta_lines.append(f"{space}def _meta_should_take_trade(direction:int):")
        meta_lines.append(f"{i1}# No trained meta-model baked; accept all if asked to use it")
        meta_lines.append(f"{i1}return True")

    # ---- Concrete action blocks (now respect meta_policy at runtime) ----
    def _buy_block():
        if has_meta:
            core = (
                f"{i3}if position_qty == 0:\n"
                f"{i4}max_shares = int(cash // current_price)\n"
                f"{i4}if max_shares > 0:\n"
                f"{i4}    logger.info(f\"[{{ticker}}] Buying {{max_shares}} shares at {{current_price}}.\")\n"
                f"{i4}    buy_shares(ticker, max_shares, current_price, predicted_price)\n"
                f"{i3}else:\n"
                f"{i4}logger.info(f\"[{{ticker}}] Insufficient cash to purchase shares.\")\n"
            )
            return (
                f"{i1}if meta_policy:\n"
                f"{i2}if _meta_should_take_trade(1):\n" + core +
                f"{i2}else:\n"
                f"{i3}logger.info(f\"[{{ticker}}] Meta-gate rejected BUY.\")\n"
                f"{i1}else:\n"
                f"{i2}if position_qty == 0:\n"
                f"{i3}max_shares = int(cash // current_price)\n"
                f"{i3}if max_shares > 0:\n"
                f"{i4}logger.info(f\"[{{ticker}}] Buying {{max_shares}} shares at {{current_price}}.\")\n"
                f"{i4}buy_shares(ticker, max_shares, current_price, predicted_price)\n"
                f"{i2}else:\n"
                f"{i3}logger.info(f\"[{{ticker}}] Insufficient cash to purchase shares.\")"
            )
        else:
            return (
                f"{i1}if position_qty == 0:\n"
                f"{i2}max_shares = int(cash // current_price)\n"
                f"{i2}if max_shares > 0:\n"
                f"{i3}logger.info(f\"[{{ticker}}] Buying {{max_shares}} shares at {{current_price}}.\")\n"
                f"{i3}buy_shares(ticker, max_shares, current_price, predicted_price)\n"
                f"{i1}else:\n"
                f"{i2}logger.info(f\"[{{ticker}}] Insufficient cash to purchase shares.\")"
            )

    def _sell_block():
        if has_meta:
            core = (
                f"{i3}if position_qty > 0:\n"
                f"{i4}logger.info(f\"[{{ticker}}] Selling {{position_qty}} shares at {{current_price}}.\")\n"
                f"{i4}sell_shares(ticker, position_qty, current_price, predicted_price)\n"
                f"{i3}else:\n"
                f"{i4}logger.info(f\"[{{ticker}}] No long position to sell; no action taken.\")\n"
            )
            return (
                f"{i1}if meta_policy:\n"
                f"{i2}if _meta_should_take_trade(-1):\n" + core +
                f"{i2}else:\n"
                f"{i3}logger.info(f\"[{{ticker}}] Meta-gate rejected SELL.\")\n"
                f"{i1}else:\n"
                f"{i2}if position_qty > 0:\n"
                f"{i3}logger.info(f\"[{{ticker}}] Selling {{position_qty}} shares at {{current_price}}.\")\n"
                f"{i3}sell_shares(ticker, position_qty, current_price, predicted_price)\n"
                f"{i2}else:\n"
                f"{i3}logger.info(f\"[{{ticker}}] No long position to sell; no action taken.\")"
            )
        else:
            return (
                f"{i1}if position_qty > 0:\n"
                f"{i2}logger.info(f\"[{{ticker}}] Selling {{position_qty}} shares at {{current_price}}.\")\n"
                f"{i2}sell_shares(ticker, position_qty, current_price, predicted_price)\n"
                f"{i1}else:\n"
                f"{i2}logger.info(f\"[{{ticker}}] No long position to sell; no action taken.\")"
            )

    none_block = f"{i1}logger.info(f\"[{{ticker}}] No trade action taken.\")"

    # ---- Assemble final code ----
    lines = []
    # meta helper
    lines.extend(meta_lines)
    lines.append("")
    # main priority chain
    lines.append(f"{space}if {buy_expr}:")
    lines.append(_buy_block())
    lines.append(f"{space}elif {sell_expr}:")
    lines.append(_sell_block())
    lines.append(f"{space}else:")
    lines.append(none_block)

    # Ensure spaces-only indentation (no tabs)
    return "\n".join(lines).replace("\t", "    ")


# ---- Candidate (Algorithm) Representation ---- #

class Candidate:
    def __init__(self, logic_tree=None):
        self.logic_tree = logic_tree if logic_tree else random_logic_tree()
        self.fitness = None   # used for sorting (validation J)
        self.code_str = None

        # meta-label artifacts (filled after evaluation/selection)
        self.meta_enabled = False
        self.meta_feature_names: List[str] = []
        self.meta_coef: List[float] = []
        self.meta_intercept: float = 0.0
        self.meta_means: Dict[str, float] = {}
        self.meta_stds: Dict[str, float] = {}

        # complexity cache
        self._nodes = None
        self._depth = None
        self._used_feats = None

        # diagnostics
        self._val_metrics = {}
        self._train_metrics = {}
        self._val_signal = None  # for diversity

    def mutate(self):
        def mutate_node(node, depth=0):
            if node is None:
                return None
            if node.value == "IF" and random.random() < 0.2:
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
        if random.random() < 0.7:
            node1.value, node2.value = node2.value, node1.value
            node1.left, node2.left = node2.left, node1.left
            node1.right, node2.right = node2.right, node1.right
        return child

    # --------------- Complexity helpers ---------------- #
    def _traverse(self, node):
        if node is None:
            return 0, 0, set()
        if node.left is None and node.right is None:
            used = {node.value} if node.value in FEATURES else set()
            return 1, 1, used
        ln, ld, lf = self._traverse(node.left)
        rn, rd, rf = self._traverse(node.right)
        used = lf | rf
        if node.value in FEATURES:
            used.add(node.value)
        return 1 + ln + rn, 1 + max(ld, rd), used

    def complexity(self):
        if self._nodes is None or self._depth is None or self._used_feats is None:
            n, d, used = self._traverse(self.logic_tree)
            self._nodes = n
            self._depth = d
            self._used_feats = {u for u in used if isinstance(u, str)}
        return self._nodes, self._depth, len(self._used_feats)

    def get_backtest_code(self):
        # ----- Build BUY/SELL expressions from the GP tree (compile-time) -----
        def _and(a: str, b: str) -> str:
            if a in (None, "", "True"):
                return f"({b})"
            return f"(({a}) and ({b}))"

        def _collect_action_conditions(n, path_cond="True", acc=None):
            if acc is None:
                acc = {"BUY": [], "SELL": [], "NONE": []}
            if n is None:
                acc["NONE"].append(path_cond); return acc
            if getattr(n, "value", None) == "IF":
                cond_code = expr_to_code(n.left) or "False"
                then_node = n.right
                if not then_node or getattr(then_node, "value", None) != "THEN":
                    acc["NONE"].append(_and(path_cond, cond_code))
                    acc["NONE"].append(_and(path_cond, f"not ({cond_code})"))
                    return acc
                _collect_action_conditions(then_node.left,  _and(path_cond, cond_code), acc)
                _collect_action_conditions(then_node.right, _and(path_cond, f"not ({cond_code})"), acc)
                return acc
            if getattr(n, "value", None) == "THEN":
                _collect_action_conditions(n.left, path_cond, acc)
                _collect_action_conditions(n.right, path_cond, acc)
                return acc
            if isinstance(n.value, str) and n.value.replace('"','') in ("BUY","SELL","NONE"):
                acc[n.value.replace('"','')].append(path_cond); return acc
            acc["NONE"].append(path_cond); return acc

        def _or_join(conds):
            if not conds: return "False"
            seen = set(); uniq = []
            for c in conds:
                if c not in seen:
                    seen.add(c); uniq.append(c)
            if len(uniq) == 1: return uniq[0]
            return " or ".join(f"({c})" for c in uniq)

        acc = _collect_action_conditions(self.logic_tree, "True", None)
        buy_expr = _or_join(acc.get("BUY", []))
        sell_expr = _or_join(acc.get("SELL", []))

        # ----- Start emitting function source -----
        lines = []
        lines.append("def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles, meta_policy:")
        lines.append("    import numpy as np")
        # unpack features
        for feat in FEATURES:
            if feat in ["current_price", "predicted_price"]:
                continue
            lines.append(f"    {feat} = candles.get('{feat}', 0)")
        lines.append("")  # spacing

        # meta helper (baked parameters or pass-through)
        if self.meta_enabled and self.meta_feature_names:
            feat_names = self.meta_feature_names
            coef = self.meta_coef
            intercept = self.meta_intercept
            means = self.meta_means
            stds = self.meta_stds

            lines.append("    def _meta_should_take_trade(direction:int):")
            lines.append("        x = []")
            for nm in feat_names:
                if nm == "__direction__":
                    lines.append("        x.append(direction)")
                else:
                    mu = means.get(nm, 0.0)
                    sd = stds.get(nm, 1.0) or 1.0
                    lines.append(f"        x.append(((candles.get('{nm}', 0)) - ({mu})) / ({sd}))")
            lines.append(f"        lin = ({intercept})")
            for j in range(len(feat_names)):
                cj = coef[j]
                lines.append(f"        lin += ({cj}) * x[{j}]")
            lines.append("        p = 1.0 / (1.0 + np.exp(-lin))")
            lines.append("        return p >= 0.5")
        else:
            lines.append("    def _meta_should_take_trade(direction:int):")
            lines.append("        # No trained meta; accept all if asked to use it")
            lines.append("        return True")
        lines.append("")

        # priority chain with runtime meta_policy
        lines.append(f"    if {buy_expr}:")
        lines.append( "        if meta_policy:")
        lines.append( "            if _meta_should_take_trade(+1):")
        lines.append( "                return \"BUY\"")
        lines.append( "            else:")
        lines.append( "                return \"NONE\"")
        lines.append( "        else:")
        lines.append( "            return \"BUY\"")
        lines.append(f"    elif {sell_expr}:")
        lines.append( "        if meta_policy:")
        lines.append( "            if _meta_should_take_trade(-1):")
        lines.append( "                return \"SELL\"")
        lines.append( "            else:")
        lines.append( "                return \"NONE\"")
        lines.append( "        else:")
        lines.append( "            return \"SELL\"")
        lines.append("    else:")
        lines.append("        return \"NONE\"")

        return "\n".join(lines)


    def get_logic_code(self):
        lines = []

        lines.append("def run_logic(current_price, predicted_price, ticker, meta_policy:")
        lines.append("    \"\"\"")
        lines.append("    Auto-generated live-trading logic with optional meta-gate.")
        lines.append("    Loads the latest feature row from the CSV each call.")
        lines.append("    \"\"\"")
        lines.append("    import os")
        lines.append("    import logging")
        lines.append("    from functools import lru_cache")
        lines.append("    import pandas as pd")
        lines.append("    import numpy as np")
        lines.append("    from dotenv import load_dotenv")
        lines.append("    from forest import api, buy_shares, sell_shares")
        lines.append("")
        lines.append("    logger = logging.getLogger(__name__)")
        lines.append("")
        lines.append("    load_dotenv()")
        lines.append("    BAR_TIMEFRAME = os.getenv(\"BAR_TIMEFRAME\", \"1Hour\")")
        lines.append("    TIMEFRAME_MAP = {")
        lines.append("        \"4Hour\": \"H4\",")
        lines.append("        \"2Hour\": \"H2\",")
        lines.append("        \"1Hour\": \"H1\",")
        lines.append("        \"30Min\": \"M30\",")
        lines.append("        \"15Min\": \"M15\"")
        lines.append("    }")
        lines.append("    CONVERTED_TIMEFRAME = TIMEFRAME_MAP.get(BAR_TIMEFRAME, BAR_TIMEFRAME)")
        lines.append("")
        lines.append("    def _csv_filename(tkr: str) -> str:")
        lines.append("        return os.path.join(\"data\", f\"{tkr}_{CONVERTED_TIMEFRAME}.csv\")")
        lines.append("")
        lines.append("    @lru_cache(maxsize=16)")
        lines.append("    def _latest_row(tkr: str):")
        lines.append("        fname = _csv_filename(tkr)")
        lines.append("        try:")
        lines.append("            df = pd.read_csv(fname)")
        lines.append("            if df.empty:")
        lines.append("                return {}")
        lines.append("            return df.iloc[-1].to_dict()")
        lines.append("        except FileNotFoundError:")
        lines.append("            logger.error(f\"[{tkr}] CSV file '{fname}' not found.\")")
        lines.append("            return {}")
        lines.append("        except Exception as exc:")
        lines.append("            logger.error(f\"[{tkr}] Error reading CSV '{fname}': {exc}\")")
        lines.append("            return {}")
        lines.append("")
        lines.append("    latest = _latest_row(ticker)")
        lines.append("")
        lines.append("    try:")
        lines.append("        account = api.get_account()")
        lines.append("        cash = float(account.cash)")
        lines.append("    except Exception as e:")
        lines.append("        logger.error(f\"[{ticker}] Error fetching account details: {e}\")")
        lines.append("        return")
        lines.append("")
        lines.append("    try:")
        lines.append("        pos = api.get_position(ticker)")
        lines.append("        position_qty = float(pos.qty)")
        lines.append("    except Exception:")
        lines.append("        position_qty = 0.0")
        lines.append("")
        lines.append("    logger.info(")
        lines.append("        f\"[{ticker}] Current Price: {current_price}, \"")
        lines.append("        f\"Predicted Price: {predicted_price}, \"")
        lines.append("        f\"Position: {position_qty}, Cash: {cash}\"")
        lines.append("    )")
        lines.append("")
        # feature assignments
        lines.append("    # ---------------- feature assignments ----------------")
        for feat in FEATURES:
            if feat in ("current_price", "predicted_price"):
                continue
            lines.append(f"    {feat} = latest.get('{feat}', 0)")
        lines.append("")
        lines.append("    # -------------- decision logic --------------")

        # meta configuration injection (baked parameters)
        if self.meta_enabled and self.meta_feature_names:
            meta_cfg = {
                "enabled": True,
                "feature_names": self.meta_feature_names,
                "coef": self.meta_coef,
                "intercept": self.meta_intercept,
                "means": self.meta_means,
                "stds": self.meta_stds
            }
        else:
            meta_cfg = {"enabled": False}

        decision_block = logic_to_live_code(self.logic_tree, indent=1, meta_cfg=meta_cfg)
        if not decision_block.strip():
            decision_block = "    logger.info(f\"[{ticker}] No trade action taken.\")"
        decision_block = decision_block.replace('\\t', '    ')
        lines.append(decision_block.rstrip("\n"))
        code = "\n".join(lines) + "\n"
        return code


    def __str__(self):
        return self.get_backtest_code()

# -------------- Data Loading ---------------- #

def load_csv_data(ticker: str, current_timestamp=None) -> pd.DataFrame:
    fname = get_csv_filename(ticker)
    if not os.path.isfile(fname):
        raise FileNotFoundError(f"CSV file {fname} not found.")

    df = pd.read_csv(fname)

    # Required columns
    needed_cols = {"timestamp", "close", "predicted_close"}
    missing = needed_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)} in {fname}")

    # Parse/normalize timestamp
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
    if ts.isna().any():
        try:
            if np.issubdtype(df["timestamp"].dtype, np.number):
                ts_try = pd.to_datetime(df["timestamp"], unit="s", errors="coerce", utc=False)
                if ts_try.dropna().dt.year.max() if not ts_try.dropna().empty else 0 > 2100 or ts_try.isna().all():
                    ts_try = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce", utc=False)
                ts = ts_try
        except Exception:
            pass

    df["timestamp"] = ts
    df = df.dropna(subset=["timestamp"]).copy()

    try:
        if getattr(df["timestamp"].dt, "tz", None) is not None:
            df["timestamp"] = df["timestamp"].dt.tz_convert(None)
    except Exception:
        pass

    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    if current_timestamp is not None:
        ts_cut = pd.to_datetime(current_timestamp, errors="coerce", utc=False)
        if pd.isna(ts_cut):
            raise ValueError(f"current_timestamp '{current_timestamp}' could not be parsed as datetime.")
        try:
            if getattr(ts_cut, "tzinfo", None) is not None:
                try:
                    ts_cut = ts_cut.tz_convert(None)
                except Exception:
                    ts_cut = ts_cut.tz_localize(None)
        except Exception:
            pass
        df = df[df["timestamp"] <= ts_cut].copy()
        if df.empty:
            raise ValueError(f"No data at or before current_timestamp={current_timestamp} in {fname}.")

    # Ensure all GP feature columns exist
    for feat in FEATURES:
        if feat not in df.columns and feat not in ("current_price", "predicted_price"):
            df[feat] = np.nan

    # Fill forward then hard fill (no lag)
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)

    # Convenience columns used by backtests
    df["current_price"] = df["close"]
    df["predicted_price"] = df["predicted_close"]
    return df

# -------------- Labeling: Triple-Barrier ---------------- #

def triple_barrier_labels(close: np.ndarray, up: float, dn: float, h: int) -> np.ndarray:
    n = len(close)
    y = np.zeros(n, dtype=int)
    for i in range(n):
        p0 = close[i]
        up_level = p0 * (1.0 + up)
        dn_level = p0 * (1.0 - dn)
        end = min(n - 1, i + h)
        label = 0
        for j in range(i + 1, end + 1):
            pj = close[j]
            if pj >= up_level:
                label =  +1
                break
            if pj <= dn_level:
                label =  -1
                break
        y[i] = label
    return y

# -------------- Split: Purged + Embargoed (80/20) -------- #

def purged_embargo_split(n: int, train_frac: float, embargo: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Indices [0..n-1]. Train: [0 .. cut-embargo-1], Val: [cut+embargo .. n-1].
    Ensures no overlapping horizon across boundary. Falls back gracefully.
    """
    cut = int(max(1, min(n - 1, np.floor(n * train_frac))))
    train_end = max(0, cut - embargo)
    val_start = min(n, cut + embargo)
    train_idx = np.arange(0, train_end, dtype=int)
    val_idx = np.arange(val_start, n, dtype=int)
    if len(train_idx) == 0 or len(val_idx) == 0:
        # fallback: shrink embargo if dataset small
        train_idx = np.arange(0, max(1, cut), dtype=int)
        val_idx = np.arange(max(1, cut), n, dtype=int)
    return train_idx, val_idx

# -------------- Simple Logistic (Meta-Model) ------------- #

class SimpleLogit:
    def __init__(self, lr=0.1, epochs=200, l2=1e-4):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.w = None
        self.b = 0.0

    def _sig(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            z = X @ self.w + self.b
            p = self._sig(z)
            grad_w = (X.T @ (p - y)) / n + self.l2 * self.w
            grad_b = np.sum(p - y) / n
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

    def predict_proba(self, X):
        z = X @ self.w + self.b
        return self._sig(z)

# --------- Metrics & Simulation (no fees, i execution) ---- #

def simulate_with_meta(df: pd.DataFrame,
                       idx: np.ndarray,
                       s_hat: np.ndarray,
                       meta: Optional[SimpleLogit],
                       feat_names: List[str],
                       means: Dict[str, float],
                       stds: Dict[str, float]) -> Tuple[np.ndarray, int, int, List[float], List[int]]:
    cash = 10000.0
    position_qty = 0
    position_price = 0.0
    equity_curve = []
    trade_pnls = []
    holding_lengths = []
    hold_len = 0

    for i in idx:
        row = df.iloc[i]
        current_price = float(row["close"])
        dir_sig = s_hat[i]
        take = False
        if dir_sig == 0:
            take = False
        else:
            if meta is None:
                take = True
            else:
                # Build meta feature vector: z-score(features) + direction
                x = []
                for nm in feat_names:
                    if nm == "__direction__":
                        x.append(dir_sig)
                    else:
                        v = row.get(nm, 0.0)
                        mu = means.get(nm, 0.0)
                        sd = stds.get(nm, 1.0) or 1.0
                        x.append((v - mu) / sd)
                p = meta.predict_proba(np.array(x)[None, :])[0]
                take = (p >= 0.5)

        # Execute at bar i price (no lag)
        if dir_sig == +1 and take:
            if position_qty == 0:
                qty = int(cash // current_price)
                if qty > 0:
                    cash -= qty * current_price
                    position_qty = qty
                    position_price = current_price
                    hold_len = 0
        elif dir_sig == -1 and take:
            if position_qty > 0:
                pnl = (current_price - position_price) * position_qty
                cash += position_qty * current_price
                position_qty = 0
                position_price = 0.0
                trade_pnls.append(pnl)
                holding_lengths.append(hold_len)
                hold_len = 0
        else:
            # NONE or meta-rejected → do nothing
            pass

        # mark-to-market at bar i
        equity = cash + position_qty * current_price
        equity_curve.append(equity)
        if position_qty > 0:
            hold_len += 1

    # If still holding at end, close to compute final PnL for stats (not for equity since we already marked to market)
    if position_qty > 0:
        last_price = float(df.iloc[idx[-1]]["close"])
        pnl = (last_price - position_price) * position_qty
        trade_pnls.append(pnl)
        holding_lengths.append(hold_len)

    # count trades
    buys = int(len([1 for i in idx if s_hat[i] == +1]))
    sells = int(len([1 for i in idx if s_hat[i] == -1]))
    return np.array(equity_curve), buys, sells, trade_pnls, holding_lengths

def sharpe_ratio(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    mu = returns.mean()
    sd = returns.std(ddof=1)
    if sd == 0:
        return 0.0
    return mu / sd

def max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = -np.inf
    mdd = 0.0
    for v in equity:
        peak = max(peak, v)
        dd = (peak - v) / peak if peak > 0 else 0.0
        mdd = max(mdd, dd)
    return mdd

def turnover(buys: int, sells: int, n_bars: int) -> float:
    if n_bars <= 0:
        return 0.0
    return (buys + sells) / max(1, n_bars)

def deflated_sharpe(sr: float, n_trials: int, T: int) -> float:
    # Lightweight conservative deflation for multiple testing
    if T <= 2 or n_trials <= 1:
        return sr
    return sr / (1.0 + np.log(max(2, n_trials)))

# -------------- Train/Val Evaluation with Meta ------------- #

def get_base_signals(candidate: Candidate, df: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    # Build the run_backtest function for the candidate (action-only logic)
    local_scope = {}
    exec(candidate.get_backtest_code(), {}, local_scope)
    run_backtest = local_scope['run_backtest']

    s = np.zeros(len(df), dtype=int)
    # provide dummy position_qty consistent with interface (not used by logic)
    for i in idx:
        row = df.iloc[i]
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
                position_qty=0,
                current_timestamp=ts,
                candles=features
            )
        except Exception as ex:
            logger.warning(f"Signal gen exception: {ex}")
            action = "NONE"
        if action == "BUY":
            s[i] = +1
        elif action == "SELL":
            s[i] = -1
        else:
            s[i] = 0
    return s

def standardize_matrix(df: pd.DataFrame, idx: np.ndarray, feat_names: List[str]) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
    X = np.zeros((len(idx), len(feat_names)))
    means, stds = {}, {}
    # compute means/stds on the slice
    for j, nm in enumerate(feat_names):
        if nm == "__direction__":
            col = np.array([0.0]*len(idx))  # placeholder; will be filled with direction elsewhere
            mu, sd = 0.0, 1.0
        else:
            col = df.iloc[idx][nm].to_numpy(dtype=float, copy=False)
            mu = float(np.mean(col)) if col.size else 0.0
            sd = float(np.std(col, ddof=1)) if col.size > 1 else 1.0
            if sd == 0.0:
                sd = 1.0
        X[:, j] = (col - mu) / sd if nm != "__direction__" else col
        means[nm] = mu
        stds[nm] = sd
    return X, means, stds

def evaluate_candidate(candidate: Candidate,
                       df: pd.DataFrame,
                       train_idx: np.ndarray,
                       val_idx: np.ndarray,
                       y: np.ndarray,
                       n_trials_so_far: int) -> Candidate:
    # base signals on train/val
    s_train = get_base_signals(candidate, df, train_idx)
    s_val = get_base_signals(candidate, df, val_idx)

    # meta-label targets on train (only where signal != 0)
    # z = 1 if signal matches triple-barrier direction; else 0; treat y==0 as 0.
    mask_train = (s_train[train_idx] != 0)
    if np.sum(mask_train) >= 5:
        feat_names = [f for f in FEATURES if f not in ("current_price", "predicted_price")] + ["__direction__"]
        X_train_raw, means, stds = standardize_matrix(df, train_idx, feat_names)
        # inject direction into last column
        X_train_raw[:, -1] = s_train[train_idx].astype(float)
        z_train = (np.sign(s_train[train_idx]) == np.sign(y[train_idx])).astype(int)
        z_train[y[train_idx] == 0] = 0  # neutral labels → do not take
        z_train = z_train.astype(int)

        # keep only rows where a signal was proposed
        X_train = X_train_raw[mask_train]
        z_train = z_train[mask_train]

        meta = None
        if X_train.shape[0] >= 5:
            meta = SimpleLogit(lr=0.1, epochs=250, l2=1e-4)
            meta.fit(X_train, z_train)
            meta_enabled = True
        else:
            meta_enabled = False
            meta = None

        # simulate on validation with meta gate
        # Build val feature normalization based on TRAIN means/stds
        # (no leakage from val)
        feat_names_use = feat_names
        candidate.meta_enabled = meta_enabled
        candidate.meta_feature_names = feat_names_use if meta_enabled else []
        candidate.meta_means = means if meta_enabled else {}
        candidate.meta_stds = stds if meta_enabled else {}
        if meta_enabled:
            candidate.meta_coef = list(np.asarray(meta.w).ravel())
            candidate.meta_intercept = float(meta.b)
        else:
            candidate.meta_coef = []
            candidate.meta_intercept = 0.0
    else:
        # not enough signals; disable meta
        meta = None
        feat_names = [f for f in FEATURES if f not in ("current_price", "predicted_price")] + ["__direction__"]
        means = {nm: 0.0 for nm in feat_names}
        stds = {nm: 1.0 for nm in feat_names}
        candidate.meta_enabled = False
        candidate.meta_feature_names = []
        candidate.meta_means = {}
        candidate.meta_stds = {}
        candidate.meta_coef = []
        candidate.meta_intercept = 0.0

    equity_val, n_buys, n_sells, trade_pnls, holding_lens = simulate_with_meta(
        df, val_idx, s_val, meta if candidate.meta_enabled else None,
        candidate.meta_feature_names if candidate.meta_enabled else feat_names,
        candidate.meta_means if candidate.meta_enabled else means,
        candidate.meta_stds if candidate.meta_enabled else stds
    )

    # metrics (validation)
    rets_val = np.diff(equity_val, prepend=equity_val[0]) / np.maximum(1e-9, equity_val[:-1].tolist() + [equity_val[-1]])
    sr_val = sharpe_ratio(rets_val)
    mdd_val = max_drawdown(equity_val)
    turn_val = turnover(n_buys, n_sells, len(val_idx))
    # complexity
    nodes, depth, n_feats = candidate.complexity()
    complexity = ALPHA_COMPLEXITY * nodes + BETA_COMPLEXITY * depth + GAMMA_FEATURES * n_feats

    # composite J
    J = sr_val - LAMBDA_MDD * mdd_val - LAMBDA_TURNOVER * turn_val - LAMBDA_COMPLEXITY * complexity
    # deflated Sharpe (informational; not used directly in J)
    dsr = deflated_sharpe(sr_val, max(2, n_trials_so_far), max(5, len(val_idx)))

    candidate.fitness = float(J)
    candidate._val_metrics = {
        "Sharpe": float(sr_val),
        "MDD": float(mdd_val),
        "Turnover": float(turn_val),
        "Nodes": int(nodes),
        "Depth": int(depth),
        "DistinctFeatures": int(n_feats),
        "Complexity": float(complexity),
        "CompositeJ": float(J),
        "DeflatedSharpe": float(dsr),
        "ValTrades": int(n_buys + n_sells)
    }
    candidate._val_signal = s_val[val_idx].astype(int).tolist()

    # Training diagnostics (optional): simulate without meta on train for logging
    equity_tr, nb, ns, _, _ = simulate_with_meta(
        df, train_idx, s_train, None, [], {}, {}
    )
    rets_tr = np.diff(equity_tr, prepend=equity_tr[0]) / np.maximum(1e-9, equity_tr[:-1].tolist() + [equity_tr[-1]])
    candidate._train_metrics = {
        "Sharpe": float(sharpe_ratio(rets_tr)),
        "MDD": float(max_drawdown(equity_tr)),
        "Trades": int(nb + ns)
    }
    return candidate

# ---- Diversity filter for elites (low-correlated signals) ---- #

def select_diverse_elites(cands: List[Candidate], top_k: int, corr_thr: float) -> List[Candidate]:
    selected: List[Candidate] = []
    sigs = []
    for c in sorted(cands, key=lambda x: x.fitness, reverse=True):
        vec = np.array(c._val_signal, dtype=float)
        if vec.size == 0:
            continue
        ok = True
        for s in sigs:
            if len(s) == len(vec):
                if np.std(vec) == 0 or np.std(s) == 0:
                    corr = 1.0 if (np.all(vec == s)) else 0.0
                else:
                    corr = float(np.corrcoef(vec, s)[0,1])
                if abs(corr) > corr_thr:
                    ok = False
                    break
        if ok:
            selected.append(c)
            sigs.append(vec)
        if len(selected) >= top_k:
            break
    if not selected:
        selected = cands[:min(top_k, len(cands))]
    return selected

# ---- Evolution Loop (Train on TRAIN; Select on VALIDATION) ---- #

class Population:
    def __init__(self, size):
        self.candidates = [Candidate() for _ in range(size)]
        self.generation = 0

    def get_best(self):
        return max(self.candidates, key=lambda c: c.fitness if c.fitness is not None else -1e9)

    def select(self, k=8):
        return sorted(self.candidates, key=lambda c: c.fitness, reverse=True)[:k]


def evolve(ticker,
    current_timestamp=None,
    population_size=100,
    generations=100,
    elite_frac=0.3,
    mutation_rate=0.7,
    crossover_rate=0.6,
    patience=EARLY_STOP_PATIENCE,
):
    logger.info(f"Loading data for {ticker} / {CONVERTED_TIMEFRAME} ...")
    df_all = load_csv_data(ticker, current_timestamp)

    # triple-barrier labels on all rows available
    y_all = triple_barrier_labels(df_all["close"].to_numpy(dtype=float), TB_THETA_UP, TB_THETA_DN, TB_H)

    # Split (purged + embargoed) on all data up to current_timestamp
    n = len(df_all)
    train_idx, val_idx = purged_embargo_split(n, TRAIN_FRAC, TB_H)
    logger.info(f"Split -> TRAIN: {len(train_idx)} bars, VAL: {len(val_idx)} bars (embargo={TB_H})")

    pop = Population(population_size)
    best_fitness_history = []
    best_candidate = None
    best_fitness = -999
    stale = 0
    n_trials = 0

    for gen in range(generations):
        logger.info(f"=== Generation {gen+1} ===")
        evaluated = []
        for c in pop.candidates:
            try:
                n_trials += 1
                evaluated.append(evaluate_candidate(c, df_all, train_idx, val_idx, y_all, n_trials))
            except Exception as ex:
                logger.warning(f"Evaluation failed: {ex}")
                c.fitness = -999
                evaluated.append(c)

        # sort by validation composite J
        evaluated.sort(key=lambda c: c.fitness, reverse=True)
        pop.candidates = evaluated

        best = pop.get_best()
        best_fitness_history.append(best.fitness)
        logger.info(f"Best J (val): {best.fitness:.4f} | Sharpe(val)={best._val_metrics.get('Sharpe',0):.3f} MDD(val)={best._val_metrics.get('MDD',0):.3f} Trades(val)={best._val_metrics.get('ValTrades',0)} | Complexity={best._val_metrics.get('Complexity',0):.1f}")

        if best.fitness > best_fitness:
            best_candidate = copy.deepcopy(best)
            best_fitness = best.fitness
            stale = 0
        else:
            stale += 1

        if stale >= patience:
            logger.info(f"Early stopping (no improvement in {patience} generations).")
            break

        # Next generation
        next_candidates: List[Candidate] = []

        # Elitism with diversity filter
        elite_count = max(2, int(population_size * elite_frac))
        elites_diverse = select_diverse_elites(pop.candidates, min(elite_count, DIVERSITY_TOPK), DIVERSITY_CORR_MAX)
        next_candidates.extend(copy.deepcopy(c) for c in elites_diverse)

        # Fill the rest
        while len(next_candidates) < population_size:
            if random.random() < crossover_rate:
                parents = random.sample(pop.candidates[:max(elite_count+5, len(pop.candidates))], 2)
                child = parents[0].crossover(parents[1])
                if random.random() < mutation_rate:
                    child.mutate()
                next_candidates.append(child)
            else:
                base = copy.deepcopy(random.choice(pop.candidates[:max(1, elite_count)]))
                if random.random() < mutation_rate:
                    base.mutate()
                next_candidates.append(base)

        pop.candidates = next_candidates
        pop.generation += 1

    # Logging artifacts
    logger.info("\n" + "="*36)
    logger.info(" Best Evolved run_backtest:")
    print(best_candidate.get_backtest_code())
    logger.info("\n" + "="*36)
    logger.info(" Best Evolved run_logic (for live trading):")
    print(best_candidate.get_logic_code())
    logger.info(f"Validation Composite J: {best_candidate.fitness:.4f}")
    logger.info(f"Validation Metrics: {best_candidate._val_metrics}")
    logger.info(f"Training Metrics: {best_candidate._train_metrics}")

    # Save fitness plot
    try:
        out_png = _fitplot_file(ticker)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.figure()
        plt.plot(best_fitness_history)
        plt.title(f"Best Validation J per Generation – {ticker}")
        plt.xlabel("Generation")
        plt.ylabel("Composite J (val)")
        plt.savefig(out_png)
        plt.close()
        logger.info(f"Saved fitness plot to {out_png}")
    except Exception as e:
        logger.debug(f"Could not render/save fitness plot: {e}")

    # Prepare a simple scorecard (validation-only; test requires future data which is not included up to current_timestamp)
    try:
        scorecard = {
            "ticker": ticker,
            "timeframe": CONVERTED_TIMEFRAME,
            "triple_barrier": {"h": TB_H, "theta_up": TB_THETA_UP, "theta_dn": TB_THETA_DN},
            "split": {"train": len(train_idx), "val": len(val_idx), "embargo": TB_H, "train_frac": TRAIN_FRAC},
            "validation_metrics": best_candidate._val_metrics,
            "training_metrics": best_candidate._train_metrics,
            "meta_enabled": bool(best_candidate.meta_enabled),
            "meta_features": best_candidate.meta_feature_names if best_candidate.meta_enabled else [],
        }
        _scorecard_file(ticker).write_text(json.dumps(scorecard, indent=2))
        logger.info(f"Wrote validation scorecard to {_scorecard_file(ticker)}")
    except Exception as e:
        logger.debug(f"Could not write scorecard: {e}")

    return best_candidate


CACHE_LIMIT = 50
CACHE_DIR = Path(__file__).parent / ".evobot_cache" 
LOGIC_FILE = CACHE_DIR / "logic_src.py"
BT_FILE = CACHE_DIR / "backtest_src.py"
META_FILE = CACHE_DIR / "meta.json"

_gp_run_logic = None
_gp_run_backtest = None
_remaining_calls = None        # loaded from meta or set to CACHE_LIMIT


# -------------------- helper: load cache -------------------- #
def _load_cache(ticker: str) -> bool:
    """Load compiled functions and remaining_calls for this ticker from disk."""
    try:
        meta_path = _meta_file(ticker)
        logic_path = _logic_file(ticker)
        bt_path = _bt_file(ticker)

        meta = json.loads(meta_path.read_text())
        remaining = int(meta.get("remaining_calls", CACHE_LIMIT if CACHE_LIMIT > 0 else 1))

        # Load run_logic
        loc_ns = {}
        exec(logic_path.read_text(), globals(), loc_ns)
        _gp_run_logic_map[ticker] = loc_ns["run_logic"]

        # Load run_backtest
        loc_ns = {}
        exec(bt_path.read_text(), globals(), loc_ns)
        _gp_run_backtest_map[ticker] = loc_ns["run_backtest"]

        _remaining_calls_map[ticker] = remaining

        logger.info(
            f"Loaded GP strategy for {ticker} from {_ticker_cache_dir(ticker)} "
            f"(remaining_calls={remaining})."
        )
        return True
    except Exception as e:
        logger.debug(f"No valid cache for {ticker}: {e}")
        return False


def _save_cache(ticker: str, best_cand, remaining: int) -> None:
    """Persist the compiled sources and remaining_calls for this ticker."""
    tdir = _ticker_cache_dir(ticker)
    tdir.mkdir(parents=True, exist_ok=True)

    logic_src = best_cand.get_logic_code()
    bt_src = best_cand.get_backtest_code()

    _logic_file(ticker).write_text(logic_src)
    _bt_file(ticker).write_text(bt_src)
    _meta_file(ticker).write_text(json.dumps({
        "remaining_calls": int(remaining),
        "val_metrics": best_cand._val_metrics,
        "train_metrics": best_cand._train_metrics,
        "meta_enabled": best_cand.meta_enabled,
        "meta_features": best_cand.meta_feature_names,
        "meta_coef": best_cand.meta_coef,
        "meta_intercept": best_cand.meta_intercept
    }, indent=2))

    logger.info(f"Strategy for {ticker} written to disk cache at {tdir}.")



# -------------------- helper: (re)build strategy ------------- #
def _build_gp_functions(ticker: str, current_timestamp=None) -> None:
    """Run evolution and refresh in-memory funcs and meta for this ticker."""
    logger.info(f"Running GP evolution to (re)build strategy for {ticker} …")
    best = evolve(ticker, current_timestamp)  # time-consuming search

    # Compile and store per-ticker run_logic
    loc_ns = {}
    exec(best.get_logic_code(), globals(), loc_ns)
    _gp_run_logic_map[ticker] = loc_ns["run_logic"]

    # Compile and store per-ticker run_backtest (action-only; live uses run_logic)
    loc_ns = {}
    exec(best.get_backtest_code(), globals(), loc_ns)
    _gp_run_backtest_map[ticker] = loc_ns["run_backtest"]

    # Reset remaining_calls for this ticker
    _remaining_calls_map[ticker] = CACHE_LIMIT if CACHE_LIMIT > 0 else 1

    # Persist to disk
    _save_cache(ticker, best, _remaining_calls_map[ticker])


# -------------------- main refresh/initialise logic ---------- #
def _refresh_if_needed(ticker: str, current_timestamp=None) -> None:
    """Ensure compiled funcs for the ticker exist and rebuild on TTL expiry."""
    # No caching at all → always rebuild
    if CACHE_LIMIT == 0:
        _build_gp_functions(ticker, current_timestamp)
        return

    # First use of this ticker in this process
    if ticker not in _gp_run_logic_map or ticker not in _gp_run_backtest_map:
        if not _load_cache(ticker):
            _build_gp_functions(ticker, current_timestamp)

    # Decrement per-ticker counter and refresh if needed
    _remaining_calls_map[ticker] = int(_remaining_calls_map.get(ticker, 1)) - 1
    if _remaining_calls_map[ticker] <= 0:
        logger.info(f"CACHE_LIMIT reached for {ticker} → rebuilding strategy.")
        _build_gp_functions(ticker, current_timestamp)
    else:
        # Persist updated remaining_calls
        try:
            _meta_file(ticker).write_text(
                json.dumps({"remaining_calls": _remaining_calls_map[ticker]}, indent=2)
            )
        except Exception:
            pass


def run_logic(current_price: float, predicted_price: float, ticker: str):
    _refresh_if_needed(ticker)
    return _gp_run_logic_map[ticker](current_price, predicted_price, ticker, meta_policy=META_POLICY)


def run_backtest(
    current_price: float,
    predicted_price: float,
    position_qty: float,
    current_timestamp,
    candles,
    ticker: str
):
    _refresh_if_needed(ticker, current_timestamp)

    if isinstance(candles, pd.DataFrame):
        try:
            row = candles.loc[candles["timestamp"] == current_timestamp].iloc[-1]
        except Exception:
            row = candles.iloc[-1]
        candle_dict = row.to_dict()
    else:
        candle_dict = candles

    return _gp_run_backtest_map[ticker](
        current_price=current_price,
        predicted_price=predicted_price,
        position_qty=position_qty,
        current_timestamp=current_timestamp,
        candles=candle_dict,
        meta_policy=META_POLICY
    )

def _load_tickerlist_file() -> Path:
    """
    Resolve the path to /data/tickerlist.txt relative to this script living in /logic.
    """
    logic_dir = Path(__file__).resolve().parent
    project_root = logic_dir.parent  # assumes /logic sibling is /data
    tickerlist_path = project_root / "data" / "tickerlist.txt"
    if not tickerlist_path.exists():
        raise FileNotFoundError(f"tickerlist.txt not found at: {tickerlist_path}")
    return tickerlist_path

def _parse_ticker_line(line: str) -> str | None:
    """
    Parse a single line and return the ticker symbol, or None if the line should be ignored.
    """
    line = line.strip()
    if not line:
        return None

    if line.startswith("#"):
        return None

    if re.match(r"^\s*selection_model\s*=", line, flags=re.IGNORECASE):
        return None

    if "," in line:
        ticker = line.split(",", 1)[0].strip()
    else:
        ticker = line.strip()

    if not ticker:
        return None

    return ticker.upper()

def read_tickers_from_file(path: Path) -> List[str]:
    """
    Read the ticker list file and return a de-duplicated, order-preserving list of tickers.
    """
    tickers_ordered: "OrderedDict[str, None]" = OrderedDict()
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            t = _parse_ticker_line(raw)
            if t:
                if not re.match(r"^[A-Z0-9._:-]+$", t):
                    logger.warning(f"Ignoring suspicious ticker '{t}' (contains invalid characters).")
                    continue
                tickers_ordered.setdefault(t, None)
    return list(tickers_ordered.keys())

if __name__ == "__main__":
    tickerlist_path = _load_tickerlist_file()
    tickers = read_tickers_from_file(tickerlist_path)

    if not tickers:
        logger.info("No tickers found in tickerlist.txt.")
    else:
        logger.info(f"Found {len(tickers)} tickers: {', '.join(tickers)}")
        for ticker in tickers:
            logger.info(f"Running _build_gp_functions for {ticker}...")
            _build_gp_functions(ticker)
        logger.info("All tickers processed.")
