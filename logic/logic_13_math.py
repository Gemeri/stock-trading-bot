from __future__ import annotations

import os
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Union
from inspect import signature
import logic.tools as tools

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.utils import check_random_state
from bot.trading.orders import buy_shares, sell_shares


# gplearn
from gplearn.genetic import SymbolicRegressor
import joblib
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

DATA_PATH = Path("./")
CACHE_PATH = Path("./model_cache")
CACHE_PATH.mkdir(exist_ok=True)

# Model retrain cadence (bucket size)
MODEL_RETRAIN_FREQ = "12H"    # change to '1D', '6H', etc. if desired

FEATURES = tools.FEATURES

COMPLEXITY_LAMBDA = 1e-3
INITIAL_CASH = 10_000.0

# ------------------------------------------------------------
# Caching helpers
# ------------------------------------------------------------
_full_df: pd.DataFrame | None = None      # loaded once, reused
_model_cache: Dict[str, " _ModelCache"] = {}   # key = time-bucket ISO

def get_csv_filename(ticker):
    return tools.get_csv_filename(ticker)

# ------------------------------------------------------------------
# Ensure a DataFrame has all FEATURE columns; create missing ones = 0
# ------------------------------------------------------------------
def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        logger.warning("Missing columns filled with 0.0: %s", missing)
        for c in missing:
            df[c] = 0.0
    return df


def _load_full_data(ticker) -> pd.DataFrame:
    """Load CSV once, then reuse in-memory copy (huge speed-up)."""
    global _full_df
    if _full_df is not None:
        return _full_df

    file = DATA_PATH / get_csv_filename(ticker)
    if not file.exists():
        raise FileNotFoundError(f"CSV file {file} not found")

    df = pd.read_csv(file)
    df = _ensure_features(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.dropna(subset=FEATURES + ["timestamp", "close"])
    df["target"] = df["close"].shift(-1)
    df = df.iloc[:-1]
    _full_df = df
    logger.info("Loaded %d rows from %s", len(df), file.name)
    return _full_df


# ------------------------------------------------------------
# gplearn utilities (version-agnostic)
# ------------------------------------------------------------
def _build_est(model_params: Dict[str, Any], random_state: int):
    allowed = set(signature(SymbolicRegressor).parameters)
    filtered = {k: v for k, v in model_params.items() if k in allowed}
    if "n_jobs" in allowed:
        filtered["n_jobs"] = -1                # full CPU parallelism
    filtered["random_state"] = random_state
    return SymbolicRegressor(**filtered)


def _program_length(prog) -> int:
    if hasattr(prog, "length"):
        attr = prog.length
        return int(attr() if callable(attr) else attr)
    if hasattr(prog, "_length"):
        attr = prog._length
        return int(attr() if callable(attr) else attr)
    try:
        return int(len(prog))
    except Exception:       # noqa: BLE001
        return 0


# ------------------------------------------------------------
# Model selection & training
# ------------------------------------------------------------
def _time_series_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_params: Dict[str, Any],
    random_state: int = 0,
    n_splits: int = 3,                       # ↓ from 5 → 3 (faster)
    complexity_lambda: float = COMPLEXITY_LAMBDA,
) -> float:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    losses = []

    for train_idx, test_idx in tscv.split(X):
        est = _build_est(model_params, random_state)
        est.fit(X[train_idx], y[train_idx])
        y_pred = est.predict(X[test_idx])
        rmse = np.sqrt(mean_squared_error(y[test_idx], y_pred))
        losses.append(rmse + complexity_lambda * _program_length(est._program))

    return float(np.mean(losses))


def _hyperparameter_search(X: np.ndarray, y: np.ndarray) -> SymbolicRegressor:
    """Lightweight grid (<< original) – big speed-up."""
    param_grid = {
        "population_size": [400],         # ↓
        "generations": [15],              # ↓
        "parsimony_coefficient": [1e-3],
        "max_samples": [0.9],
        "max_depth": [4, 5],
        "metric": ["rmse"],
        "p_crossover": [0.7],
        "p_subtree_mutation": [0.1],
    }

    rng = check_random_state(42)
    grid = list(ParameterGrid(param_grid))
    rng.shuffle(grid)

    best_loss = np.inf
    best_params = grid[0]

    for params in grid:
        loss = _time_series_cv(X, y, params, random_state=42)
        if loss < best_loss:
            best_loss, best_params = loss, params

    model = _build_est(best_params, random_state=42)
    model.fit(X, y)
    logger.info(
        "GP selected | loss %.5f | length %d | params %s",
        best_loss,
        _program_length(model._program),
        best_params,
    )
    return model


def _optimize_threshold(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    pct = (y_pred - y_true) / y_true
    grid = np.linspace(0.0, 0.04, 17)       # up to 4 %
    best, best_t = -np.inf, 0.0
    for t in grid:
        ret = np.where(pct > t, pct, 0.0) - np.where(pct < -t, pct, 0.0)
        s = ret.sum()
        if s > best:
            best, best_t = s, t
    return best_t, best_t


# ------------------------------------------------------------
# Model cache entry
# ------------------------------------------------------------
@dataclass
class _ModelCache:
    model: SymbolicRegressor
    thresholds: Tuple[float, float]


def _train_until(ts: Union[pd.Timestamp, str], ticker) -> _ModelCache:
    if isinstance(ts, str):
        ts = pd.to_datetime(ts, utc=True)

    # bucket timestamp → e.g. 12-hour point
    bucket_iso = ts.floor(MODEL_RETRAIN_FREQ).isoformat()
    if bucket_iso in _model_cache:
        return _model_cache[bucket_iso]

    df = _load_full_data(ticker)
    df = df[df["timestamp"] <= ts]
    if len(df) < 150:
        raise ValueError("Insufficient history for GP")

    if abs(df["predicted_close"].corr(df["target"])) < 0.05:
        df["predicted_close"] = 0.0

    X, y = df[FEATURES].values, df["target"].values
    model = _hyperparameter_search(X, y)
    t_buy, t_sell = _optimize_threshold(df["close"].values, model.predict(X))

    cache = _ModelCache(model, (t_buy, t_sell))
    _model_cache[bucket_iso] = cache

    safe_iso = bucket_iso.replace(":", "-")
    joblib.dump(cache, CACHE_PATH / f"gp_{safe_iso}.joblib")
    return cache


# ------------------------------------------------------------
# Portfolio bookkeeping
# ------------------------------------------------------------
@dataclass
class _PortfolioState:
    cash: float = INITIAL_CASH
    position_qty: int = 0
    position_price: float = 0.0
    equity_history: List[Tuple[pd.Timestamp, float]] = field(default_factory=list)

    def market_value(self, price: float) -> float:
        return self.cash + self.position_qty * price


_port = _PortfolioState()
TRADE_LOG = Path("trade_log.csv")
if TRADE_LOG.exists():
    TRADE_LOG.unlink()


def _log_trade(ts, typ, price, qty, pnl):
    pd.DataFrame([{
        "timestamp": ts, "type": typ, "price": price,
        "shares": qty, "pnl": pnl
    }]).to_csv(TRADE_LOG, mode="a", header=not TRADE_LOG.exists(), index=False)


def _update_equity(ts, price):
    _port.equity_history.append((ts, _port.market_value(price)))


def _finalise():
    if not _port.equity_history:
        return
    hist = pd.DataFrame(_port.equity_history, columns=["timestamp", "equity"])
    hist.to_csv("equity_curve.csv", index=False)
    ret = hist["equity"].pct_change().dropna()
    sharpe = ret.mean() / (ret.std() + 1e-12) * np.sqrt(252 * 6)
    mdd = (hist["equity"].cummax() - hist["equity"]).max()
    total = hist["equity"].iloc[-1] / INITIAL_CASH - 1
    pd.Series(
        dict(final_equity=hist.iloc[-1, 1], total_return=total,
             sharpe=sharpe, max_drawdown=mdd)
    ).to_csv("backtest_summary.csv")
    plt.figure()
    plt.plot(hist["timestamp"], hist["equity"])
    plt.title("Equity Curve"); plt.xlabel("Time"); plt.ylabel("Equity ($)")
    plt.tight_layout(); plt.savefig("equity_curve.png", dpi=160); plt.close()


import atexit; atexit.register(_finalise)

# ------------------------------------------------------------
#  Signal helpers
# ------------------------------------------------------------
def _generate_signal(price, pred, th, qty) -> str:
    diff = (pred - price) / price
    b, s = th
    if diff > b:
        return "BUY" if qty == 0 else "NONE"
    if diff < -s:
        return "SELL" if qty > 0 else "NONE"
    return "NONE"


# ------------------------------------------------------------
#  Public API
# ------------------------------------------------------------
def run_logic(current_price: float, _predicted_price: float, ticker: str):
    cache = _train_until(pd.Timestamp.utcnow(), ticker)
    feat_row = _load_full_data(ticker).iloc[-1][FEATURES].values.reshape(1, -1)
    pred = cache.model.predict(feat_row)[0]

    from forest import api
    try:
        cash = float(api.get_account().cash)
        qty = float(api.get_position(ticker).qty)
    except Exception:        # noqa: BLE001
        qty = 0.0; cash = 0.0

    sig = _generate_signal(current_price, pred, cache.thresholds, qty)
    logger.info("LIVE %s | %.2f → %.2f", sig, current_price, pred)

    if sig == "BUY" and cash >= current_price:
        buy_shares(ticker, int(cash // current_price), current_price, pred)
    elif sig == "SELL" and qty > 0:
        sell_shares(ticker, qty, current_price, pred)


def run_backtest(
    current_price: float,
    predicted_price: float,
    position_qty: float,
    current_timestamp: Union[pd.Timestamp, str],
    candles: pd.DataFrame,
    ticker
) -> str:
    if isinstance(current_timestamp, str):
        current_timestamp = pd.to_datetime(current_timestamp, utc=True)

    cache = _train_until(current_timestamp, ticker)
    candles = _ensure_features(candles)
    latest = candles.iloc[-1][FEATURES].values.reshape(1, -1)
    pred = cache.model.predict(latest)[0]
    sig = _generate_signal(current_price, pred, cache.thresholds, position_qty)

    # bookkeeping (long-only, no fees)
    if sig == "BUY" and _port.position_qty == 0:
        qty = int(_port.cash // current_price)
        if qty:
            _port.cash -= qty * current_price
            _port.position_qty = qty
            _port.position_price = current_price
            _log_trade(current_timestamp, "BUY", current_price, qty, 0.0)

    elif sig == "SELL" and _port.position_qty:
        pnl = (_port.position_qty * (current_price - _port.position_price))
        _log_trade(current_timestamp, "SELL", current_price, _port.position_qty, pnl)
        _port.cash += _port.position_qty * current_price
        _port.position_qty = 0
        _port.position_price = 0.0

    _update_equity(current_timestamp, current_price)
    return sig
