import os
import logging
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from collections import deque
import functools

# For genetic programming
from deap import base, creator, tools, gp

# ==== HYPERPARAMETERS ====
GA_POP_SIZE = 100
GA_N_GEN = 50
GA_CXPB = 0.8     # Crossover rate
GA_MUTPB = 0.1    # Mutation rate
GA_TOUR_SIZE = 3
GA_WALK_FWD_TRAIN_PCT = 0.8

ACTION_BUY = 1
ACTION_NONE = 0
ACTION_SELL = -1

RAND_SEED = 32171

CHAMPION_PICKLE_FILE = "ga_champion_strategy.pkl"

# ========== LOGGING SETUP ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= ENV and DATA ============
load_dotenv()
BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "1Hour")
TIMEFRAME_MAP = {"4Hour":"H4","2Hour":"H2","1Hour":"H1","30Min":"M30","15Min":"M15"}
CONVERTED_TIMEFRAME = TIMEFRAME_MAP.get(BAR_TIMEFRAME, BAR_TIMEFRAME)
TICKERS = os.getenv("TICKERS", "TSLA").split(",")
assert len(TICKERS) == 1
TICKER = TICKERS[0]

FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume', 'vwap', 'sentiment',
    'macd_line', 'macd_signal', 'macd_histogram',
    'rsi', 'momentum', 'roc', 'atr', 'obv',
    'bollinger_upper', 'bollinger_lower',
    'ema_9', 'ema_21', 'ema_50', 'ema_200', 'adx',
    'lagged_close_1', 'lagged_close_2', 'lagged_close_3',
    'lagged_close_5', 'lagged_close_10',
    'candle_body_ratio', "predicted_close"
    'wick_dominance',
    'gap_vs_prev',
    'volume_zscore',
    'atr_zscore',
    'rsi_zscore',
    'adx_trend',
    'macd_cross',
    'macd_hist_flip',
    'day_of_week',
    'days_since_high',
    'days_since_low'
]

# =========== API/EXECUTION STUBS ===========
try:
    from forest import api, buy_shares, sell_shares
except ImportError:
    class DummyAPI:
        cash = 100000
        def get_account(self): return self
        def get_position(self, ticker): raise Exception
    api = DummyAPI()
    def buy_shares(*a, **k): logger.info(f"[STUB] buy_shares{a}{k}")
    def sell_shares(*a, **k): logger.info(f"[STUB] sell_shares{a}{k}")

# ============= RANDOM SEEDING ==============
np.random.seed(RAND_SEED)

# =========== UTILITY: DATA & SERIALIZATION ==========

def get_csv_filename(ticker):
    return f"{ticker}_{CONVERTED_TIMEFRAME}.csv"

def load_all_candles():
    fname = get_csv_filename(TICKER)
    if not os.path.exists(fname):
        raise FileNotFoundError(f"No CSV file found for {fname}")
    candles = pd.read_csv(fname, parse_dates=["timestamp"])
    return candles

def save_champion_strategy(ind, scaler):
    with open(CHAMPION_PICKLE_FILE, "wb") as f:
        pickle.dump({"strategy": ind, "scaler": scaler}, f)

def load_champion_strategy():
    if not os.path.exists(CHAMPION_PICKLE_FILE):
        logger.error("Champion strategy file not found.")
        return None, None
    with open(CHAMPION_PICKLE_FILE, "rb") as f:
        obj = pickle.load(f)
        return obj["strategy"], obj["scaler"]

# ================ GA SYMBOLIC LOGIC ================

def gen_primitive_set():
    pset = gp.PrimitiveSet("MAIN", len(FEATURE_COLUMNS))
    # Arithmetic ops
    pset.addPrimitive(np.add, 2)
    pset.addPrimitive(np.subtract, 2)
    pset.addPrimitive(np.multiply, 2)
    def protected_div(a, b): return a / b if b != 0 else 0
    pset.addPrimitive(protected_div, 2)

    # Boolean logic - combine
    pset.addPrimitive(lambda a, b: float(a > b), 2, name="greater_than")
    pset.addPrimitive(lambda a, b: float(a < b), 2, name="less_than")
    pset.addPrimitive(lambda a: float(a > 0), 1, name="is_positive")
    pset.addPrimitive(lambda a: float(a < 0), 1, name="is_negative")

    # Sigmoid/nonlinear
    pset.addPrimitive(lambda a: np.tanh(a), 1, name="tanh")
    pset.addPrimitive(lambda a: np.abs(a), 1, name="abs")

    # Terminals -- fix with functools.partial instead of lambda:
    import random
    pset.addEphemeralConstant("rand101", functools.partial(random.uniform, -1, 1))

    # Features as arguments x0, x1, ..., xN
    pset.renameArguments(**{f"ARG{i}": FEATURE_COLUMNS[i] for i in range(len(FEATURE_COLUMNS))})
    return pset

def ensure_creator_classes():
    import deap.creator
    if not hasattr(deap.creator, "FitnessMax"):
        deap.creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(deap.creator, "Individual"):
        deap.creator.create("Individual", gp.PrimitiveTree, fitness=deap.creator.FitnessMax)

def tree_to_action(root_output):
    if root_output > 0.5: return ACTION_BUY
    elif root_output < -0.5: return ACTION_SELL
    else: return ACTION_NONE

def action_to_str(a):
    return {ACTION_BUY: "BUY", ACTION_NONE:"NONE", ACTION_SELL:"SELL"}.get(a, "NONE")

# ========== FITNESS: PROFIT METRICS FOR WALK-FORWARD =========

def walk_forward_fitness(individual, toolbox, X, y_price, y_pred, timestamps):
    # STAGE 1: use first 80% for training
    n_total = len(X)
    n_train = int(GA_WALK_FWD_TRAIN_PCT * n_total)
    train_idx = np.arange(0, n_train)
    valid_idx = np.arange(n_train, n_total)

    profit_train, sharpe_train, mdd_train, turnover_train = simulate_trading(
        individual, toolbox, X[train_idx], y_price[train_idx], y_pred[train_idx], timestamps[train_idx]
    )
    profit_val, sharpe_val, mdd_val, turnover_val = simulate_trading(
        individual, toolbox, X[valid_idx], y_price[valid_idx], y_pred[valid_idx], timestamps[valid_idx]
    )
    # Fitness function: maximize final profit + sharpe bonus - drawdown, - turnover
    score_train = profit_train + 0.1*sharpe_train - 5*mdd_train - 0.001*turnover_train
    score_val   = profit_val   + 0.1*sharpe_val   - 5*mdd_val   - 0.001*turnover_val
    # Penalize overfitting by geometric mean weighting
    if np.isnan(score_train) or np.isnan(score_val) or score_train < 0 or score_val < 0:
        score = -1e12  # or another large negative penalty
    else:
        score = (score_train * score_val) ** 0.5
    return (score,)

def simulate_trading(individual, toolbox, X, closes, predicted, timestamps, initial_cash=100000.0):
    func = toolbox.compile(expr=individual)
    position = 0  # shares
    cash = initial_cash
    equity_curve = []
    last_trade_price = 0
    last_action = ACTION_NONE
    turnover = 0  # dollar amount values traded

    for i, feat in enumerate(X):
        price = closes[i]
        pred_p = predicted[i]
        # GP "decision"
        try:
            output = func(*feat)
            action = tree_to_action(output)
        except Exception as e:
            logger.error(f"Tree execution error: {e}")
            action = ACTION_NONE
        # Simple logic, no duplication of positions
        if action == ACTION_BUY and position == 0:
            qty = int(cash // price)
            if qty > 0:
                cash -= qty * price
                position += qty
                turnover += qty * price
                last_trade_price = price
                last_action = ACTION_BUY
        elif action == ACTION_SELL and position > 0:
            cash += position * price
            turnover += position * price
            position = 0
            last_trade_price = price
            last_action = ACTION_SELL
        # else: hold
        total_equity = cash + position * price if position > 0 else cash
        equity_curve.append(total_equity)

    profit = equity_curve[-1] - initial_cash if equity_curve else 0
    returns = np.diff(equity_curve) if len(equity_curve) > 1 else np.zeros(1)
    if len(returns) > 1:
        sharpe = np.mean(returns) / (np.std(returns) + 1e-5)
    else:
        sharpe = 0
    # Max Drawdown
    drawdown = 0
    peak = equity_curve[0] if equity_curve else initial_cash
    for x in equity_curve:
        if x > peak: peak = x
        dd = (peak - x) / peak if peak else 0
        if dd > drawdown: drawdown = dd
    mdd = drawdown
    return profit, sharpe, mdd, turnover

# ========== GA SETUP/EXECUTION CODE ============

def prepare_ga_dataset(up_to_timestamp):
    candles = load_all_candles()
    candles = candles.sort_values("timestamp")
    candles = candles[candles["timestamp"] <= up_to_timestamp]
    candles = candles.dropna(subset=FEATURE_COLUMNS + ['timestamp', 'close', 'predicted_close'])
    X = candles[FEATURE_COLUMNS].values.astype(float)
    y_price = candles['close'].values.astype(float)
    y_pred = candles['predicted_close'].values.astype(float)
    timestamps = candles['timestamp'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y_price, y_pred, timestamps, scaler

def evolve_ga(X, y_price, y_pred, timestamps):
    pset = gen_primitive_set()
    ensure_creator_classes()
    import deap.creator
    toolbox = base.Toolbox()
    toolbox.register("expr_init", gp.genHalfAndHalf, pset=pset, min_=2, max_=5)
    toolbox.register("individual", tools.initIterate, deap.creator.Individual, toolbox.expr_init)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register(
        "evaluate", walk_forward_fitness,
        toolbox=toolbox, X=X, y_price=y_price, y_pred=y_pred, timestamps=timestamps,
    )
    toolbox.register("select", tools.selTournament, tournsize=GA_TOUR_SIZE)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=1, max_=3)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats = tools.MultiStatistics(fitness=stats_fit)
    logbook = tools.Logbook()
    population = toolbox.population(n=GA_POP_SIZE)

    # Main evolution loop
    for g in range(GA_N_GEN):
        # Evaluate fitness
        for ind in population:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)
        # Logging stats
        fits = [ind.fitness.values[0] for ind in population]
        valid_fits = [f for f in fits if f > -1e11]
        if valid_fits:
            max_fit, avg_fit = max(valid_fits), np.mean(valid_fits)
        else:
            max_fit, avg_fit = max(fits), np.mean(fits)
        logger.info(f"[GA] Gen {g+1}/{GA_N_GEN} Max Fit: {max_fit:.2f} Avg Fit: {avg_fit:.2f}")
        # Select
        offspring = toolbox.select(population, len(population))
        # Clone
        offspring = list(map(toolbox.clone, offspring))
        # Crossover/mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.rand() < GA_CXPB:
                gp.cxOnePoint(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for idx in range(len(offspring)):
            if np.random.rand() < GA_MUTPB:
                gp.mutUniform(offspring[idx], expr=toolbox.expr_mut, pset=pset)
                del offspring[idx].fitness.values
        # Replace
        population[:] = offspring
    # Final evaluation and pick best
    for ind in population:
        if not ind.fitness.valid:
            ind.fitness.values = toolbox.evaluate(ind)
    best_ind = tools.selBest(population, 1)[0]
    logger.info(f"[GA] Best individual's fitness: {best_ind.fitness.values[0]:.2f}")
    return best_ind, pset

# ============== PUBLIC INTERFACE ==================

def run_logic(current_price, predicted_price, ticker):
    # API/positions
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

    # Load last champion
    ind, scaler = load_champion_strategy()
    if ind is None or scaler is None:
        logger.error(f"[{ticker}] No GA champion strategy found for trading. Aborting.")
        return

    # Load full candle features for context
    candles = load_all_candles()
    latest = candles.iloc[-1]
    features = []
    for c in FEATURE_COLUMNS:
        val = latest[c] if c in latest else 0.0
        features.append(val)
    try:
        idx_close = FEATURE_COLUMNS.index('close')
        features[idx_close] = current_price
        idx_pred = FEATURE_COLUMNS.index('predicted_close')
        features[idx_pred] = predicted_price
    except Exception as ex:
        logger.warning(f"Could not patch live features: {ex}")
    X_live = np.array([features])
    X_live_scaled = scaler.transform(X_live)

    # GP logic
    try:
        pset = gen_primitive_set()
        func = gp.compile(expr=ind, pset=pset)
        output = func(*X_live_scaled[0])
        action = tree_to_action(output)
    except Exception as ex:
        logger.error(f"GP champion execution failure: {ex}")
        action = ACTION_NONE

    logger.info(
        f"[{ticker}] [LIVE] Current Price: {current_price}, "
        f"Predicted: {predicted_price}, Cash: {cash}, Position: {position_qty}, "
        f"GP Action: {action_to_str(action)}"
    )

    # Execute action
    if action == ACTION_BUY and position_qty == 0:
        max_shares = int(cash // current_price)
        if max_shares > 0:
            logger.info(f"[{ticker}] [LIVE] Buying {max_shares} shares at {current_price}")
            buy_shares(ticker, max_shares, current_price, predicted_price)
        else:
            logger.info(f"[{ticker}] [LIVE] Insufficient cash.")
    elif action == ACTION_SELL and position_qty > 0:
        logger.info(f"[{ticker}] [LIVE] Selling {int(position_qty)} shares at {current_price}")
        sell_shares(ticker, int(position_qty), current_price, predicted_price)
    else:
        logger.info(f"[{ticker}] [LIVE] No action taken ({action_to_str(action)}).")

def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    if not isinstance(current_timestamp, pd.Timestamp):
        ts = pd.Timestamp(current_timestamp)
    else:
        ts = current_timestamp
    # Gather all features/targets for GA up to and including ts
    X, y_price, y_pred, timestamps, scaler = prepare_ga_dataset(ts)

    # -- Evolve on all data up to this timestep (costly, but correct) --
    try:
        best_ind, pset = evolve_ga(X, y_price, y_pred, timestamps)
        save_champion_strategy(best_ind, scaler)
    except Exception as ex:
        logger.error(f"[BACKTEST] GA error at {ts}: {ex}")
        return "NONE"

    this_row = candles.iloc[-1]
    features = []
    for c in FEATURE_COLUMNS:
        features.append(this_row.get(c, 0.0))
    try:
        idx_close = FEATURE_COLUMNS.index('close')
        features[idx_close] = current_price
        idx_pred = FEATURE_COLUMNS.index('predicted_close')
        features[idx_pred] = predicted_price
    except Exception as ex:
        logger.warning(f"Could not patch feature vector: {ex}")
    X_cur = np.array([features])
    X_cur_scaled = scaler.transform(X_cur)

    # GP logic
    try:
        func = gp.compile(expr=best_ind, pset=pset)
        output = func(*X_cur_scaled[0])
        action = tree_to_action(output)
    except Exception as ex:
        logger.error(f"GP backtest execution failure: {ex}")
        action = ACTION_NONE

    logger.info(
        f"[BACKTEST] {ts} Price: {current_price:.2f}, Predicted: {predicted_price:.2f}, "
        f"Position:{position_qty}, GP Action: {action_to_str(action)}"
    )

    # Enforce no margin/shorting
    decision = "NONE"
    if action == ACTION_BUY and position_qty == 0:
        decision = "BUY"
    elif action == ACTION_SELL and position_qty > 0:
        decision = "SELL"
    else:
        decision = "NONE"
    return decision