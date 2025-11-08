# ── Imports ──────────────────────────────────────────────────────────────────
import os
import pickle
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools
from sklearn.preprocessing import StandardScaler
from deap import algorithms

# ── Hyper‑parameters (easy tuning) ───────────────────────────────────────────
POP_SIZE: int = 100
N_GEN: int = 50
CX_RATE: float = 0.8
MUT_RATE: float = 0.1
TOURN_SIZE: int = 3
RISK_FREE: float = 0.0
INITIAL_CASH: float = 100_000.0

MODEL_DIR = Path("./ga_models")
MODEL_DIR.mkdir(exist_ok=True)

# ── Features list (order matters!) ───────────────────────────────────────────
FEATURES: List[str] = [
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

N_FEATURES = len(FEATURES)
GENE_LEN = N_FEATURES + 2           # weights + buy_thresh + sell_thresh
BUY_IDX = N_FEATURES
SELL_IDX = N_FEATURES + 1

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ── Optional trading API stubs ───────────────────────────────────────────────
try:
    from forest import api, buy_shares, sell_shares
except ModuleNotFoundError:
    logger.warning("Trading API not found – using stubs (back‑test mode).")

    class _FakeAccount:
        cash = str(INITIAL_CASH)

    class _FakePosition:
        def __init__(self, qty: float = 0.0): self.qty = str(qty)

    class _FakeAPI:
        def __init__(self): self._pos = {}
        def get_account(self): return _FakeAccount()
        def get_position(self, tic):
            if tic not in self._pos: raise Exception("no position")
            return _FakePosition(self._pos[tic])
        def update(self, tic, qty): self._pos[tic] = qty
    api = _FakeAPI()

    def buy_shares(ticker, qty, price, predicted_price):
        api.update(ticker, qty)
        logger.info(f"[{ticker}] ***STUB BUY {qty} @ {price} "
                    f"– next_pred {predicted_price}")

    def sell_shares(ticker, qty, price, predicted_price):
        api.update(ticker, 0)
        logger.info(f"[{ticker}] ***STUB SELL {qty} @ {price} "
                    f"– next_pred {predicted_price}")

# ── Environment .env helpers ─────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "1Hour")
TIMEFRAME_MAP = {"4Hour": "H4", "2Hour": "H2", "1Hour": "H1",
                 "30Min": "M30", "15Min": "M15"}
CONVERTED_TIMEFRAME = TIMEFRAME_MAP.get(BAR_TIMEFRAME, BAR_TIMEFRAME)

def get_csv_filename(ticker):
    return os.path.join("data", f"{ticker}_{CONVERTED_TIMEFRAME}.csv")

# ── Data cache ───────────────────────────────────────────────────────────────
_DATAFRAME_CACHE = {}
_SCALER_CACHE = {}

toolbox = base.Toolbox()

# ════════════════════════════════════════════════════════════════════════════
# GA – individual representation & evaluation
# ════════════════════════════════════════════════════════════════════════════
def _setup_deap():
    # ── 1) make sure our GA classes exist (idempotent) ────────────────────
    if "GA_Fitness" not in creator.__dict__:
        creator.create("GA_Fitness", base.Fitness, weights=(1.0,))

    if "GA_Individual" not in creator.__dict__:
        creator.create("GA_Individual", list, fitness=creator.GA_Fitness)

    # Provide *alias* so any other code asking for creator.Individual works
    if "Individual" not in creator.__dict__:
        creator.Individual = creator.GA_Individual  # simple alias

    # ── 2) (re‑)bind global toolbox to our safe individual type ────────────
    global toolbox
    toolbox.register("attr_float", np.random.uniform, -1.0, 1.0)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.GA_Individual,
        toolbox.attr_float,
        n=GENE_LEN,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutGaussian,
                     mu=0.0, sigma=0.5, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=TOURN_SIZE)



_setup_deap()

# ── GA operators ─────────────────────────────────────────────────────────────
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutGaussian,
                 mu=0.0, sigma=0.5, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=TOURN_SIZE)


# ── Utility functions ───────────────────────────────────────────────────────
def _load_data(ticker: str) -> pd.DataFrame:
    if ticker not in _DATAFRAME_CACHE:
        csv_path = get_csv_filename(ticker)
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV {csv_path} not found.")
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df.sort_values("timestamp", inplace=True)
        _DATAFRAME_CACHE[ticker] = df
    return _DATAFRAME_CACHE[ticker].copy()

def _standardize(df: pd.DataFrame, ticker: str) -> np.ndarray:
    if ticker not in _SCALER_CACHE:
        scaler = StandardScaler()
        scaler.fit(df[FEATURES])
        _SCALER_CACHE[ticker] = scaler
    else:
        scaler = _SCALER_CACHE[ticker]
    return scaler.transform(df[FEATURES])


def _simulate(
    df: pd.DataFrame,
    features_scaled: np.ndarray,
    individual: List[float],
) -> Tuple[float, float, float, float]:
    cash   = INITIAL_CASH
    shares = 0.0
    equity_curve = []
    trades = 0

    weights = np.array(individual[:N_FEATURES])
    buy_th = individual[BUY_IDX]
    sell_th = individual[SELL_IDX]

    for i, row in enumerate(df.itertuples()):
        feat_vec = features_scaled[i]
        score    = float(np.dot(feat_vec, weights))

        action = "NONE"
        if score > buy_th and shares == 0:
            # BUY
            qty = cash // row.close
            if qty > 0:
                cost = qty * row.close
                cash -= cost
                shares += qty
                trades += 1
                action = "BUY"

        elif score < sell_th and shares > 0:
            # SELL
            cash += shares * row.close
            shares = 0
            trades += 1
            action = "SELL"

        equity = cash + shares * row.close
        equity_curve.append(equity)

    pnl = equity_curve[-1] - INITIAL_CASH
    returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe = 0.0 if returns.std() == 0 else (
        (returns.mean() - RISK_FREE) / returns.std()) * np.sqrt(252)
    max_dd = 0.0
    peak = equity_curve[0]
    for v in equity_curve:
        peak = max(peak, v)
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)
    turnover = trades / len(df)

    return pnl, sharpe, max_dd, turnover


def _fitness_generator(df_train: pd.DataFrame, ticker: str):
    X_scaled = _standardize(df_train, ticker)

    def evaluate(ind: List[float]):
        pnl, sharpe, max_dd, turnover = _simulate(df_train, X_scaled, ind)
        # Fitness: maximize PnL + Sharpe bonus, penalize draw‑down & turnover
        fitness = (
            pnl
            + 1_000 * sharpe
            - 1_000 * max_dd
            - 100  * turnover
        )
        return (fitness,)
    return evaluate


# ── Updated _train_ga  ───────────────────────────────────────────────────────
def _train_ga(df: pd.DataFrame, ticker: str) -> creator.GA_Individual:

    # ── 1) make sure our custom classes exist (idempotent) ────────────────
    if "GA_Fitness" not in creator.__dict__:
        creator.create("GA_Fitness", base.Fitness, weights=(1.0,))
    if "GA_Individual" not in creator.__dict__:
        creator.create("GA_Individual", list, fitness=creator.GA_Fitness)

    # ── 2) build local toolbox so nothing external can collide ────────
    tb = base.Toolbox()
    tb.register("attr_float", np.random.uniform, -1.0, 1.0)
    tb.register(
        "individual",
        tools.initRepeat,
        creator.GA_Individual,
        tb.attr_float,
        n=GENE_LEN,
    )
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("mate", tools.cxUniform, indpb=0.5)
    tb.register("mutate", tools.mutGaussian,
                mu=0.0, sigma=0.5, indpb=0.2)
    tb.register("select", tools.selTournament, tournsize=TOURN_SIZE)

    # bind our fitness evaluator (fresh each call so df can change)
    tb.register("evaluate", _fitness_generator(df, ticker))

    # ── 3) run the evolutionary loop ──────────────────────────────────────
    pop = tb.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pop, _log = algorithms.eaSimple(
            population=pop,
            toolbox=tb,
            cxpb=CX_RATE,
            mutpb=MUT_RATE,
            ngen=N_GEN,
            halloffame=hof,
            verbose=False
        )

    # ── 4) return the champion (hof[0]) ───────────────────────────────────
    return hof[0]


def _champion_path(ticker: str) -> Path:
    return MODEL_DIR / f"ga_champion_{ticker}.pkl"


def _save_champion(individual, ticker: str):
    with open(_champion_path(ticker), "wb") as f:
        pickle.dump(individual, f)


def _load_champion(ticker: str):
    path = _champion_path(ticker)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

# ════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════════════════════
def run_logic(current_price: float, predicted_price: float, ticker: str):
    # ------------------------------------------------------------------ load
    champ = _load_champion(ticker)
    if champ is None:
        logger.info(f"[{ticker}] Champion not found – training from scratch.")
        df_full = _load_data(ticker)
        champ = _train_ga(df_full, ticker)
        _save_champion(champ, ticker)
        logger.info(f"[{ticker}] New champion trained & saved.")

    # ----------------------------------------------------------------‑state
    try:
        account = api.get_account()
        cash = float(account.cash)
    except Exception as exc:
        logger.error(f"[{ticker}] API error getting account: {exc}")
        return "NONE"

    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception:
        position_qty = 0.0

    # ----------------------------------------------------------------‑decide
    feat_vec = np.zeros(N_FEATURES)
    w = np.array(champ[:2])
    buy_th  = champ[BUY_IDX]
    sell_th = champ[SELL_IDX]
    score = np.dot(w, np.array([current_price, predicted_price]))
    action = "NONE"
    if score > buy_th and position_qty == 0:
        max_shares = int(cash // current_price)
        if max_shares > 0:
            buy_shares(ticker, max_shares, current_price, predicted_price)
            action = "BUY"
    elif score < sell_th and position_qty > 0:
        sell_shares(ticker, position_qty, current_price, predicted_price)
        action = "SELL"

    logger.info(
        f"[{ticker}] Action={action} "
        f"score={score:.2f} buy_th={buy_th:.2f} sell_th={sell_th:.2f} "
        f"price={current_price} pred={predicted_price} "
        f"pos={position_qty} cash={cash}"
    )
    return action


def run_backtest(
    current_price: float,
    predicted_price: float,
    position_qty: float,
    current_timestamp: datetime,
    candles: pd.DataFrame,
    ticker
) -> str:
    df_full = _load_data(ticker)
    df_hist = df_full[df_full["timestamp"] <= current_timestamp]

    if len(df_hist) < 100:
        return "NONE"

    champ = _train_ga(df_hist, ticker)

    # ── Decide on *current* row (last in candles)
    weights = np.array(champ[:N_FEATURES])
    buy_th  = champ[BUY_IDX]
    sell_th = champ[SELL_IDX]

    # Build feature vector for the final back‑test candle (row −1)
    cur_row = candles.iloc[-1]
    feat_vec = cur_row[FEATURES].to_numpy(dtype=float)
    # Standardize using scaler fitted on historical data
    scaler = _SCALER_CACHE[ticker]  # already created in training above
    feat_vec_std = scaler.transform(feat_vec.reshape(1, -1))[0]
    score = float(np.dot(feat_vec_std, weights))

    if score > buy_th and position_qty == 0:
        return "BUY"
    elif score < sell_th and position_qty > 0:
        return "SELL"
    return "NONE"
