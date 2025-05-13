# ------------------------------------------------------------
#  symbolic_trade_logic.py
# ------------------------------------------------------------
"""
Auto-symbolic trading logic using genetic programming.

Functions exposed to the main controller
----------------------------------------
run_logic(current_price: float, predicted_price: float, ticker: str) -> None
run_backtest(current_price: float,
             predicted_price: float,
             position_qty: float,
             current_timestamp: pd.Timestamp,
             candles: pd.DataFrame) -> str
"""
from __future__ import annotations

import os
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.utils import check_random_state

# gplearn provides a light-weight symbolic-regression engine
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
import joblib                           # for caching best models
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Environment / constants
# ------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("SymbolicTrade")

from dotenv import load_dotenv

load_dotenv()

BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "1Hour")
TICKERS = os.getenv("TICKERS", "TSLA").split(",")  # always length-1 per spec
TICKER = TICKERS[0]

TIMEFRAME_MAP = {
    "4Hour": "H4",
    "2Hour": "H2",
    "1Hour": "H1",
    "30Min": "M30",
    "15Min": "M15",
}
CONVERTED_TIMEFRAME = TIMEFRAME_MAP.get(BAR_TIMEFRAME, BAR_TIMEFRAME)

DATA_PATH = Path("./")  # same directory; adjust if needed
CACHE_PATH = Path("./model_cache")
CACHE_PATH.mkdir(exist_ok=True)

FEATURES = [
    "open", "high", "low", "close", "volume", "vwap", "sentiment",
    "macd_line", "macd_signal", "macd_histogram",
    "rsi", "momentum", "roc", "atr", "obv",
    "bollinger_upper", "bollinger_lower",
    "ema_9", "ema_21", "ema_50", "ema_200", "adx",
    "lagged_close_1", "lagged_close_2", "lagged_close_3",
    "lagged_close_5", "lagged_close_10",
    "candle_body_ratio", "wick_dominance", "gap_vs_prev",
    "volume_zscore", "atr_zscore", "rsi_zscore",
    "adx_trend", "macd_cross", "macd_hist_flip",
    "day_of_week", "days_since_high", "days_since_low",
    "predicted_close",
]

COMPLEXITY_LAMBDA = 1e-3   # penalty weight; tuned during search as well
INITIAL_CASH = 10_000.0

# ------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------
def get_csv_filename() -> str:
    return f"{TICKER}_{CONVERTED_TIMEFRAME}.csv"


def _load_full_data() -> pd.DataFrame:
    file = DATA_PATH / get_csv_filename()
    if not file.exists():
        raise FileNotFoundError(f"CSV file {file} not found")
    df = pd.read_csv(file)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    # drop rows with missing required features
    df = df.dropna(subset=FEATURES + ["timestamp", "close"])
    # target = next-period close
    df["target"] = df["close"].shift(-1)
    df = df[:-1]  # last row has no target
    return df


def _time_series_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_params: Dict[str, Any],
    n_splits: int = 5,
    complexity_lambda: float = COMPLEXITY_LAMBDA,
    random_state: int = 0,
) -> Tuple[float, float]:
    """
    Returns mean penalised loss and mean raw RMSE across splits.
    Penalised loss = RMSE + λ * program_length
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    losses, rmses = [], []

    for train_idx, test_idx in tscv.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        est = SymbolicRegressor(**model_params, random_state=random_state)
        est.fit(X_train, y_train)
        y_pred = est.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        # program length available after fitting
        complexity = est._program.length
        loss = rmse + complexity_lambda * complexity
        losses.append(loss)
        rmses.append(rmse)

    return float(np.mean(losses)), float(np.mean(rmses))


def _hyperparameter_search(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 0,
) -> Tuple[SymbolicRegressor, Dict[str, Any], float]:
    """
    Brutal but effective grid-plus-shuffle search.
    The best param set minimises penalised loss.
    """
    param_grid = {
        "population_size": [1000, 1500],
        "generations": [20, 30],
        "parsimony_coefficient": [1e-2, 1e-3, 1e-4],
        "max_samples": [0.8, 0.9],
        "max_depth": [4, 5, 6],
        "metric": ["rmse"],
        "stopping_criteria": [0.001],
        "p_crossover": [0.7],
        "p_subtree_mutation": [0.1],
        "p_hoist_mutation": [0.05],
    }

    rng = check_random_state(random_state)
    grid = list(ParameterGrid(param_grid))
    rng.shuffle(grid)  # quick random order

    best_loss = np.inf
    best_params: Dict[str, Any] | None = None
    best_rmse = np.inf

    for params in grid:
        loss, rmse = _time_series_cv(
            X, y, params, random_state=random_state
        )
        logger.debug(f"Params {params} -> penalised {loss:.5f}, rmse {rmse:.5f}")
        if loss < best_loss:
            best_loss, best_params, best_rmse = loss, params, rmse

    assert best_params is not None, "Search failed to find params"

    # Fit final model on full data with best_params
    model = SymbolicRegressor(**best_params, random_state=random_state)
    model.fit(X, y)

    logger.info(
        f"Best GP: penalised {best_loss:.5f} | cv-RMSE {best_rmse:.5f} | length {model._program.length}"
    )
    return model, best_params, best_loss


def _optimize_threshold(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float]:
    """
    Grid-search symmetric thresholds that maximise simple cumulative return
    on *training* data: buy if diff > t_buy, sell if diff < -t_sell.
    Returns (t_buy, t_sell)
    """
    diffs = y_pred - y_true
    pct_diffs = diffs / y_true
    grid = np.linspace(0.0, 0.05, 21)  # 0% … 5% in 0.25% increments
    best_ret, best_pair = -np.inf, (0.0, 0.0)

    for t in grid:
        buy_mask = pct_diffs > t
        sell_mask = pct_diffs < -t
        # simple strategy: +pct_diff if buy, -pct_diff if sell
        returns = np.where(buy_mask, pct_diffs, 0.0) - np.where(sell_mask, pct_diffs, 0.0)
        cum_ret = returns.sum()
        if cum_ret > best_ret:
            best_ret = cum_ret
            best_pair = (t, t)

    logger.debug(f"Optimised thresholds → buy={best_pair[0]:.4%}")
    return best_pair


# Caches to avoid repeating heavy GP search
@dataclass
class _ModelCache:
    model: SymbolicRegressor
    thresholds: Tuple[float, float]
    params: Dict[str, Any]
    loss: float


_model_cache: Dict[str, _ModelCache] = {}  # key = latest_timestamp.iso


def _train_until(timestamp: pd.Timestamp) -> _ModelCache:
    """
    Train (or load cached) GP model using all rows with timestamp <= given.
    """
    key = timestamp.isoformat()
    if key in _model_cache:
        return _model_cache[key]

    df = _load_full_data()
    df = df[df["timestamp"] <= timestamp]
    if len(df) < 200:  # need some minimum history
        raise ValueError("Not enough history for GP training")

    # Assess correlation of predicted_close to target; down-weight if poor
    corr = df["predicted_close"].corr(df["target"])
    if np.abs(corr) < 0.05:
        logger.info("predicted_close shows little predictive power (<0.05); zeroing feature")
        df["predicted_close"] = 0.0

    X = df[FEATURES].values
    y = df["target"].values

    model, params, loss = _hyperparameter_search(X, y, random_state=42)
    y_pred = model.predict(X)
    t_buy, t_sell = _optimize_threshold(df["close"].values, y_pred)

    cache_entry = _ModelCache(model=model,
                              thresholds=(t_buy, t_sell),
                              params=params,
                              loss=loss)
    _model_cache[key] = cache_entry

    # Persist to disk for quicker warm-starts
    fname = CACHE_PATH / f"gp_{key}.joblib"
    joblib.dump(cache_entry, fname)

    return cache_entry


# ------------------------------------------------------------
# Portfolio bookkeeping for back-test
# ------------------------------------------------------------
@dataclass
class _PortfolioState:
    cash: float = INITIAL_CASH
    position_qty: float = 0.0
    position_price: float = 0.0
    equity_history: List[Tuple[pd.Timestamp, float]] = field(default_factory=list)

    def market_value(self, current_price: float) -> float:
        return self.cash + self.position_qty * current_price


_portfolio_state = _PortfolioState()

TRADE_LOG_FILE = Path("trade_log.csv")
if TRADE_LOG_FILE.exists():
    TRADE_LOG_FILE.unlink()  # fresh each run


def _log_trade(
    ts: pd.Timestamp,
    trade_type: str,
    price: float,
    qty: float,
    current_price: float,
    pnl: float,
) -> None:
    row = {
        "timestamp": ts,
        "type": trade_type,
        "price": price,
        "shares": qty,
        "current_price": current_price,
        "pnl": pnl,
    }
    pd.DataFrame([row]).to_csv(
        TRADE_LOG_FILE, mode="a", header=not TRADE_LOG_FILE.exists(), index=False
    )


def _update_equity(ts: pd.Timestamp, current_price: float):
    equity = _portfolio_state.market_value(current_price)
    _portfolio_state.equity_history.append((ts, equity))


def _finalise_backtest():
    # Called automatically at module unload if equity_history non-empty
    if not _portfolio_state.equity_history:
        return

    hist = pd.DataFrame(_portfolio_state.equity_history, columns=["timestamp", "equity"])
    hist.to_csv("equity_curve.csv", index=False)

    # Metrics
    returns = hist["equity"].pct_change().dropna()
    sharpe = returns.mean() / (returns.std() + 1e-12) * np.sqrt(252 * (24 / 4))  # 4-hour bars ≈ 6/day
    max_dd = (hist["equity"].cummax() - hist["equity"]).max()
    total_ret = (hist["equity"].iloc[-1] / INITIAL_CASH) - 1.0

    summary = {
        "final_equity": hist["equity"].iloc[-1],
        "total_return": total_ret,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }
    pd.Series(summary).to_csv("backtest_summary.csv")

    # Plot
    plt.figure()
    plt.plot(hist["timestamp"], hist["equity"])
    plt.title("Portfolio Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity ($)")
    plt.tight_layout()
    plt.savefig("equity_curve.png", dpi=180)
    plt.close()

import atexit
atexit.register(_finalise_backtest)

# ------------------------------------------------------------
#  Public interface functions
# ------------------------------------------------------------
def _generate_signal(
    current_price: float,
    predicted_price: float,
    thresholds: Tuple[float, float],
    position_qty: float,
) -> str:
    diff_pct = (predicted_price - current_price) / current_price
    t_buy, t_sell = thresholds

    if diff_pct > t_buy:
        return "BUY" if position_qty == 0 else "NONE"
    elif diff_pct < -t_sell:
        return "SELL" if position_qty > 0 else "NONE"
    else:
        return "NONE"


def run_logic(current_price: float, predicted_price: float, ticker: str):
    """
    Live-trading entry point.

    *Trains* on the entire CSV available so far, then generates a signal
    using the discovered symbolic equation (ignores the passed-in
    predicted_price from the caller, as we compute our own).
    """
    latest_timestamp = pd.Timestamp.utcnow()
    cache = _train_until(latest_timestamp)

    # Prepare a single-row dataframe with current features
    full_df = _load_full_data()
    latest_row = full_df.iloc[-1]
    X_live = latest_row[FEATURES].values.reshape(1, -1)
    our_pred = cache.model.predict(X_live)[0]

    # trading decision
    from forest import api, buy_shares, sell_shares  # pylint: disable=import-error
    try:
        account = api.get_account()
        cash = float(account.cash)
    except Exception as e:                           # noqa: BLE001
        logger.error("Error fetching account details: %s", e)
        return

    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception:
        position_qty = 0.0

    signal = _generate_signal(current_price, our_pred, cache.thresholds, position_qty)
    logger.info("Live signal %s | current %.2f → pred %.2f", signal, current_price, our_pred)

    if signal == "BUY" and cash >= current_price:
        qty = int(cash // current_price)
        if qty > 0:
            buy_shares(ticker, qty, current_price, our_pred)
    elif signal == "SELL" and position_qty > 0:
        sell_shares(ticker, position_qty, current_price, our_pred)


def run_backtest(
    current_price: float,
    predicted_price: float,
    position_qty: float,
    current_timestamp: pd.Timestamp,
    candles: pd.DataFrame,
) -> str:
    """
    Walk-forward back-test entry point.
    Re-trains *from scratch* on CSV up to current_timestamp.

    Returns "BUY", "SELL", or "NONE".
    Also keeps global portfolio / trade log in sync.
    """
    cache = _train_until(current_timestamp)

    # Use the last row of `candles` (most recent) to compute features
    latest = candles.iloc[-1]
    X_latest = latest[FEATURES].values.reshape(1, -1)
    our_pred = cache.model.predict(X_latest)[0]

    signal = _generate_signal(current_price, our_pred, cache.thresholds, position_qty)

    # Portfolio bookkeeping (no fees, no margin, long-only)
    if signal == "BUY" and _portfolio_state.position_qty == 0:
        qty = int(_portfolio_state.cash // current_price)
        if qty > 0:
            _portfolio_state.cash -= qty * current_price
            _portfolio_state.position_qty = qty
            _portfolio_state.position_price = current_price
            _log_trade(
                current_timestamp,
                "BUY",
                current_price,
                qty,
                current_price,
                pnl=0.0,
            )

    elif signal == "SELL" and _portfolio_state.position_qty > 0:
        qty = _portfolio_state.position_qty
        proceeds = qty * current_price
        pnl = (current_price - _portfolio_state.position_price) * qty
        _portfolio_state.cash += proceeds
        _portfolio_state.position_qty = 0
        _portfolio_state.position_price = 0.0
        _log_trade(
            current_timestamp,
            "SELL",
            current_price,
            qty,
            current_price,
            pnl=pnl,
        )

    # Update equity history for every call
    _update_equity(current_timestamp, current_price)
    return signal