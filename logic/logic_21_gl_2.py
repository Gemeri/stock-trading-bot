import os
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from dotenv import load_dotenv

# --- Module Initialization ---
load_dotenv()

BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "1Hour")
TIMEFRAME_MAP = {
    "4Hour": "H4", "2Hour": "H2", "1Hour": "H1", "30Min": "M30", "15Min": "M15"
}
CONVERTED_TIMEFRAME = TIMEFRAME_MAP.get(BAR_TIMEFRAME, BAR_TIMEFRAME)
GA_MODEL_PATH = "_GA_MODEL.pkl"
RL_MODEL_PATH = "_RL_MODEL.pkl"

# --- Utility: CSV Filename ---
def get_csv_filename(ticker: str) -> str:
    return os.path.join("data", f"{ticker}_{CONVERTED_TIMEFRAME}.csv")

# --- Logging ---
LOGGER = logging.getLogger("logic_ga_rl")
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s][%(asctime)s] %(message)s")
    ch.setFormatter(formatter)
    LOGGER.addHandler(ch)

# ------------------------------------------------------------------------
# Genetic Algorithm Section
# ------------------------------------------------------------------------

class RuleSet:
    def __init__(self, feature_ops: List[Tuple[str, str, float]], use_sentiment: bool, logic_mask: str):
        self.feature_ops = feature_ops  # [(feature, op, threshold), ...]
        self.use_sentiment = use_sentiment
        self.logic_mask = logic_mask  # e.g., "((0 and 1) or 2)" on features above

    def decide(self, features: Dict[str, Any], sentiment: float) -> int:
        mask_vals = []
        for feat, op, thresh in self.feature_ops:
            val = features.get(feat, 0.0)
            
            if op == "gt":
                mask_vals.append(val > thresh)
            elif op == "lt":
                mask_vals.append(val < thresh)
            elif op == "cross":
                # For crossing, e.g. macd_line - macd_signal > thresh
                if feat == "macd_cross":
                    v = features.get("macd_line", 0.0) - features.get("macd_signal", 0.0)
                    mask_vals.append(v > thresh)
                elif feat == "ema9_21_cross":
                    v = features.get("ema_9", 0.0) - features.get("ema_21", 0.0)
                    mask_vals.append(v > thresh)
                else:
                    mask_vals.append(False)
            else:
                mask_vals.append(False)

        # Eval mask string to bool
        try:
            logic = self.logic_mask
            # e.g. "((0 and 1) or 2)" -> insert variable strings
            logic = logic.replace("and", "and").replace("or", "or")
            for idx, m in enumerate(mask_vals):
                logic = logic.replace(str(idx), str(bool(m)))
            result = eval(logic)
        except Exception as e:
            LOGGER.error(f"Error evaluating rule logic mask {self.logic_mask} with vals {mask_vals}: {e}")
            result = False

        # Incorporate sentiment if enabled
        if self.use_sentiment and abs(sentiment) > 0.2:
            if sentiment > 0.2:  # positive: supports buy
                if result:
                    return +1
                else:
                    return +1 if sentiment > 0.4 else 0  # amplify buy
            elif sentiment < -0.2:  # negative: supports sell
                return -1
            else:
                return 0
        else:
            return +1 if result else 0  # buy if rule fires

    @staticmethod
    def random(features: List[str]) -> 'RuleSet':
        ops = ["gt", "lt", "cross"]
        # Randomly pick 2-3 features/rules
        n = np.random.choice([2, 3])
        chosen = []
        for _ in range(n):
            f = np.random.choice(features)
            op = np.random.choice(ops)
            thresh = float(np.random.normal(0, 1))  # heuristic: thresholds -1 to 1 range
            # For macd_cross, ema cross use op="cross"
            chosen.append((f, op, thresh))
        mask = " and ".join([str(i) for i in range(n)])
        if n == 3:
            # Randomly OR/AND logic
            if np.random.rand() > 0.5:
                mask = f"(({0} and {1}) or {2})"
            else:
                mask = f"({0} and {1} and {2})"
        use_sentiment = np.random.rand() > 0.5
        return RuleSet(chosen, use_sentiment, mask)

    def mutate(self, features: List[str]):
        which = np.random.choice(len(self.feature_ops))
        f, op, thresh = self.feature_ops[which]
        # Small perturbations
        if np.random.rand() < 0.5:
            # Change threshold
            thresh += np.random.normal(0, 0.2)
        else:
            # Swap feature and op
            f = np.random.choice(features)
            op = np.random.choice(["gt", "lt", "cross"])
        self.feature_ops[which] = (f, op, thresh)
        # Maybe change sentiment use
        if np.random.rand() < 0.2:
            self.use_sentiment = not self.use_sentiment

    def crossover(self, other: 'RuleSet') -> 'RuleSet':
        # Crossover feature rules
        feats = self.feature_ops[:len(self.feature_ops)//2] + other.feature_ops[len(other.feature_ops)//2:]
        # Blend logic mask and sentiment randomly
        mask = np.random.choice([self.logic_mask, other.logic_mask])
        use_sent = np.random.choice([self.use_sentiment, other.use_sentiment])
        return RuleSet(feats, use_sent, mask)

def fitness(rule: RuleSet, df: pd.DataFrame) -> float:
    startcash = 100000.0
    position = 0  # 0 or shares held
    shares = 0
    equity = startcash
    cash = startcash
    prev_peak = startcash
    drawdown = []
    returns = []

    position_val = 0.0
    last_price = None

    for idx, row in df.iterrows():
        feats = row.to_dict()
        # Add macd_cross, ema9_21_cross
        feats["macd_cross"] = feats.get("macd_line", 0.0) - feats.get("macd_signal", 0.0)
        feats["ema9_21_cross"] = feats.get("ema_9", 0.0) - feats.get("ema_21", 0.0)
        sentiment = feats.get("sentiment", 0.0)
        price = feats.get("close", 0.0)

        # Trading decision
        act = rule.decide(feats, sentiment)
        # Order logic
        if act == +1 and position == 0:
            shares = int(cash // price)
            if shares > 0:
                position = shares
                cash -= price * shares
        elif act == -1 and position > 0:
            cash += price * position
            position = 0
            shares = 0
        position_val = price * position
        equity = cash + position_val
        # Calculate drawdown
        prev_peak = max(prev_peak, equity)
        dd = (prev_peak - equity) / prev_peak
        drawdown.append(dd)
        if last_price is not None and position > 0:
            returns.append(np.log(price / last_price))
        last_price = price

    # Final return
    strategy_return = (equity - startcash) / startcash
    cagr = (equity / startcash)**(252*24 / max(len(df), 1)) - 1  # annualized for intraday
    # Sharpe estimate
    if len(returns) > 2:
        sharpe = np.mean(returns) / (np.std(returns)+1e-8) * np.sqrt(252)
    else:
        sharpe = 0.0
    max_dd = max(drawdown) if drawdown else 1.0
    # Composite fitness
    fval = sharpe * (1 - max_dd) * (1 + cagr) + strategy_return
    # Penalize catastrophic DD
    if max_dd > 0.33:
        fval -= abs(max_dd)*0.5
    return fval

def GA_optimize(df: pd.DataFrame, features: List[str], ngen: int = 20, pop: int = 18) -> RuleSet:
    # Init population
    population = [RuleSet.random(features) for _ in range(pop)]
    scores = [0.] * pop
    for gen in range(ngen):
        # Evaluate
        for i in range(pop):
            scores[i] = fitness(population[i], df)
        # Select top 50%
        idx = np.argsort(scores)[-pop//2:]
        survivors = [population[i] for i in idx]
        # Crossover/mutate to refill pop
        nextgen = []
        while len(nextgen) < pop:
            p1, p2 = np.random.choice(survivors, 2, replace=False)
            child = p1.crossover(p2)
            # Mutate
            if np.random.rand() > 0.6:
                child.mutate(features)
            nextgen.append(child)
        population = nextgen
    # Pick best
    best_idx = np.argmax([fitness(r, df) for r in population])
    return population[best_idx]

# -------------------------------------------------------------------------
# RL Section: Q-Learning for Rule Polishing
# -------------------------------------------------------------------------

class RLQAgent:
    def __init__(self, n_states=27, n_actions=3):
        # n_states = 3**3 (3 input rules, trinary spatial code)
        self.n_states = n_states
        self.n_actions = n_actions  # (BUY, SELL, NONE)
        self.q_table = np.zeros((n_states, n_actions))
        self.actions = [1, -1, 0]  # buy, sell, none

    def _state_encode(self, signals: Tuple[int, int, int]) -> int:
        # Map tuple of -1,0,1 into a single state value: trinary code
        s, s2, s3 = [(v+1) for v in signals]
        return s*9 + s2*3 + s3

    def select_action(self, state: int, eps=0.05) -> int:
        if np.random.uniform() < eps:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])

    def update(self, state: int, action: int, reward: float, next_state: int, gamma=0.90, alpha=0.1):
        best_next = np.max(self.q_table[next_state])
        td_target = reward + gamma * best_next
        td_err = td_target - self.q_table[state][action]
        self.q_table[state][action] += alpha * td_err

    def train(self, df: pd.DataFrame, rule: RuleSet, episodes=3):
        for ep in range(episodes):
            position = 0
            cash = 100000.0
            equity = cash
            last_price = None
            prev_peak = 100000.0
            for idx, row in df.iterrows():
                feats = row.to_dict()
                # Construct inputs for RL (3 main signals)
                signal1 = 1 if feats.get("macd_histogram", 0.0) > 0 else -1 if feats.get("macd_histogram", 0.0) < 0 else 0
                rsi = feats.get("rsi", 50.0)
                signal2 = 1 if rsi > 60 else -1 if rsi < 40 else 0
                mom = feats.get("momentum", 0.0)
                signal3 = 1 if mom > 0 else -1 if mom < 0 else 0
                state = self._state_encode((signal1, signal2, signal3))
                # Action
                action = self.select_action(state, eps=0.18)
                price = feats.get("close", 0.0)
                reward = 0.0
                # Simulated trade
                if self.actions[action] == 1 and position == 0:
                    shares = int(cash // price)
                    if shares > 0:
                        position = shares
                        cash -= shares * price
                elif self.actions[action] == -1 and position > 0:
                    cash += position * price
                    position = 0
                position_val = position * price
                equity = cash + position_val
                if last_price is not None:
                    reward = (equity - prev_peak) / (prev_peak + 1e-8)
                # Compute next state
                next_signal1 = 1 if feats.get("macd_histogram", 0.0) > 0 else -1 if feats.get("macd_histogram", 0.0) < 0 else 0
                next_signal2 = 1 if feats.get("rsi", 50.0) > 60 else -1 if feats.get("rsi", 50.0) < 40 else 0
                next_signal3 = 1 if feats.get("momentum", 0.0) > 0 else -1 if feats.get("momentum", 0.0) < 0 else 0
                next_state = self._state_encode((next_signal1, next_signal2, next_signal3))
                self.update(state, action, reward, next_state)
                prev_peak = max(prev_peak, equity)
                last_price = price

    def decide(self, macd_hist, rsi, momentum):
        signal1 = 1 if macd_hist > 0 else -1 if macd_hist < 0 else 0
        signal2 = 1 if rsi > 60 else -1 if rsi < 40 else 0
        signal3 = 1 if momentum > 0 else -1 if momentum < 0 else 0
        state = self._state_encode((signal1, signal2, signal3))
        action_id = np.argmax(self.q_table[state])
        return self.actions[action_id]


# -------------------------------------------------------------------------
# Feature Engineering / Preprocessing
# -------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Fill nan
    df = df.fillna(method="ffill").fillna(0)
    # Ensure all features are float
    numcols = [c for c in df.columns if c != "timestamp"]
    for col in numcols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

# List of features to use for GA
GA_FEATURES = [
    "macd_cross", "ema9_21_cross", "rsi", "adx", "momentum", "atr", "obv", "bollinger_upper", "bollinger_lower", "vwap",
    "lagged_close_1", "lagged_close_2", "lagged_close_3", "sentiment", "predicted_close"
]

# -------------------------------------------------------------------------
# Model Loading/Persistence
# -------------------------------------------------------------------------

def save_rule(rule: RuleSet, filename: str):
    with open(filename, "wb") as f:
        pickle.dump(rule, f)

def load_rule(filename: str) -> RuleSet:
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_rl(agent: RLQAgent, filename: str):
    with open(filename, "wb") as f:
        pickle.dump(agent, f)

def load_rl(filename: str) -> RLQAgent:
    with open(filename, "rb") as f:
        return pickle.load(f)

# -------------------------------------------------------------------------
# Main Live Logic
# -------------------------------------------------------------------------

def run_logic(current_price, predicted_price, ticker):
    import sys
    from forest import api, buy_shares, sell_shares

    # Locate pre-trained models
    ga_path = f"{ticker}_{CONVERTED_TIMEFRAME}{GA_MODEL_PATH}"
    rl_path = f"{ticker}_{CONVERTED_TIMEFRAME}{RL_MODEL_PATH}"

    try:
        rule = load_rule(ga_path)  # should be present from previous backtest
        agent = load_rl(rl_path)
    except Exception as e:
        LOGGER.warning(f"[{ticker}] No trained models found, running baseline logic: {e}")
        # Fallback to naive
        # --- Simple form ---
        if predicted_price > current_price:
            try:
                account = api.get_account()
                cash = float(account.cash)
                pos = api.get_position(ticker)
                position_qty = float(pos.qty)
            except Exception:
                cash = 0.0
                position_qty = 0.0
            if position_qty == 0 and cash > current_price:
                qty = int(cash // current_price)
                buy_shares(ticker, qty, current_price, predicted_price)
                return
        elif predicted_price < current_price:
            try:
                pos = api.get_position(ticker)
                position_qty = float(pos.qty)
            except Exception:
                position_qty = 0.0
            if position_qty > 0:
                sell_shares(ticker, position_qty, current_price, predicted_price)
                return
        return

    # --- In realistic use, fetch last 1 candle of features for ticker (i.e., use forest api's signals or pull CSV record) ---
    csv_fn = get_csv_filename(ticker)
    if not os.path.isfile(csv_fn):
        LOGGER.error(f"CSV file not found for {ticker}: {csv_fn}")
        return

    last_row = pd.read_csv(csv_fn).iloc[-1]
    feats = last_row.to_dict()
    feats["macd_cross"] = feats.get("macd_line", 0.0) - feats.get("macd_signal", 0.0)
    feats["ema9_21_cross"] = feats.get("ema_9", 0.0) - feats.get("ema_21", 0.0)

    try:
        account = api.get_account()
        cash = float(account.cash)
    except Exception:
        cash = 0.0
    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception:
        position_qty = 0.0

    # --- Use GA for regime, RL for tactical timing ---
    ga_signal = rule.decide(feats, feats.get("sentiment", 0.0))  # +1 buy, -1 sell, 0 none

    rl_signal = agent.decide(
        macd_hist=feats.get("macd_histogram", 0.0),
        rsi=feats.get("rsi", 50.0),
        momentum=feats.get("momentum", 0.0)
    )

    action = 0  # 0: None, +1: Buy, -1: Sell

    # Layered decision: must agree or at least "either signals buy while no pos"
    if ga_signal == 1 and rl_signal == 1:
        action = 1
    elif ga_signal == -1 or rl_signal == -1:
        action = -1
    elif ga_signal == 1 or rl_signal == 1:
        if position_qty == 0:
            action = 1
    else:
        action = 0

    if action == 1 and position_qty == 0 and cash > current_price * 0.99:
        max_shares = int(cash // current_price)
        if max_shares > 0:
            buy_shares(ticker, max_shares, current_price, predicted_price)
            LOGGER.info(f"[{ticker}] GA+RL BUY {max_shares} @ {current_price} (pred {predicted_price})")
        return
    elif action == -1 and position_qty > 0:
        sell_shares(ticker, position_qty, current_price, predicted_price)
        LOGGER.info(f"[{ticker}] GA+RL SELL {position_qty} @ {current_price}")
        return
    else:
        return  # NONE

# -------------------------------------------------------------------------
# Step-By-Step (OOS) Training and Backtest Logic
# -------------------------------------------------------------------------

def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles, ticker):
    # Figure out our ticker from env:
    csv_fn = get_csv_filename(ticker)
    if not os.path.isfile(csv_fn):
        LOGGER.error(f"[RUN_BACKTEST] CSV for {ticker} not found: {csv_fn}")
        return "NONE"

    all_df = pd.read_csv(csv_fn)
    all_df = preprocess(all_df)
    # Cut to only data up to timestamp (incl)
    backtest_df = all_df[all_df["timestamp"] <= current_timestamp]
    if len(backtest_df) < 50:  # not enough data to optimize
        return "NONE"
    # --- Cache fit for this window to memory for efficiency
    memory = getattr(run_backtest, "_mem", None)
    if memory and memory["window_max_ts"] == current_timestamp:
        rule, rl_agent = memory["rule"], memory["rl"]
    else:
        # Reoptimize GA for in-sample up to now, every N candles (or on first call)
        rule = GA_optimize(backtest_df, ["macd_cross", "ema9_21_cross", "adx", "rsi", "momentum", "atr", "obv", "vwap", "sentiment", "predicted_close"])
        # Polishing with RL
        rl_agent = RLQAgent()
        rl_agent.train(backtest_df, rule, episodes=2)
        run_backtest._mem = {"window_max_ts": current_timestamp, "rule": rule, "rl": rl_agent}

        # Save best models for later live use
        tickerid = f"{ticker}_{CONVERTED_TIMEFRAME}"
        save_rule(rule, f"{tickerid}{GA_MODEL_PATH}")
        save_rl(rl_agent, f"{tickerid}{RL_MODEL_PATH}")

    # Now, on this candle, get recent features
    row = candles.iloc[-1]
    row_dict = row.to_dict()
    # Add engineered features
    row_dict["macd_cross"] = row_dict.get("macd_line", 0.0) - row_dict.get("macd_signal", 0.0)
    row_dict["ema9_21_cross"] = row_dict.get("ema_9", 0.0) - row_dict.get("ema_21", 0.0)

    ga_signal = rule.decide(row_dict, row_dict.get("sentiment", 0.0))
    rl_signal = rl_agent.decide(
        macd_hist=row_dict.get("macd_histogram", 0.0),
        rsi=row_dict.get("rsi", 50.0),
        momentum=row_dict.get("momentum", 0.0)
    )
    action = 0
    if ga_signal == 1 and rl_signal == 1:
        action = 1
    elif ga_signal == -1 or rl_signal == -1:
        action = -1
    elif ga_signal == 1 or rl_signal == 1:
        if position_qty == 0:
            action = 1
    else:
        action = 0

    if action == 1 and position_qty == 0:
        return "BUY"
    elif action == -1 and position_qty > 0:
        return "SELL"
    else:
        return "NONE"

# -------------------------------------------------------------------------
# END OF FILE
# -------------------------------------------------------------------------