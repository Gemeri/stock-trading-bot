from __future__ import annotations

import os
import pickle
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Union
from gplearn.fitness import make_fitness

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 3ʳᵈ-party: gplearn provides fast, pure-Python symbolic regression
# If not installed the user can:  pip install gplearn
# ─────────────────────────────────────────────────────────────────────────────
try:
    from gplearn.genetic import SymbolicRegressor
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "gplearn must be installed for trade_logic.py.\n"
        "Install with:  pip install gplearn"
    ) from e

# Optional: forest trading API ------------------------------------------------
try:
    from forest import api, buy_shares, sell_shares  # type: ignore
except Exception:  # Local or back-test environment – stub out the API.
    api = None  # type: ignore

    def buy_shares(*_a, **_kw):     # pragma: no cover
        logging.info("buy_shares() → stub (not running live)")

    def sell_shares(*_a, **_kw):    # pragma: no cover
        logging.info("sell_shares() → stub (not running live)")


# ─────────────────────────────────────────────────────────────────────────────
# Environment & file handling
# ─────────────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv

load_dotenv()  # .env may contain BAR_TIMEFRAME, TICKERS, …

BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "1Hour")

TIMEFRAME_MAP = {
    "4Hour": "H4",
    "2Hour": "H2",
    "1Hour": "H1",
    "30Min": "M30",
    "15Min": "M15",
}
CONVERTED_TIMEFRAME = TIMEFRAME_MAP.get(BAR_TIMEFRAME, BAR_TIMEFRAME)

def get_csv_file(ticker):
    return os.path.join("data", f"{ticker}_{CONVERTED_TIMEFRAME}.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Constants & globals
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = [
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
TARGET_COL = "future_return"     # created on-the-fly

MODEL_PATH = Path("symbolic_model.pkl")
LOG_PATH = Path("trade_log.csv")
REPORT_PATH = Path("symbolic_report.txt")
PLOT_PATH = Path("portfolio_value.png")

# Globals reused between calls to avoid redundant re-training in live logic
_symbolic_model: Optional[SymbolicRegressor] = None
_symbolic_expr:  Optional[str]               = None
_last_fit_ts:    Optional[pd.Timestamp]      = None

# Back-test portfolio state (globals so that successive run_backtest calls
# share state across candles)
_bt_cash:        float = 100_000.0           # starting test capital
_bt_position:    int   = 0
_bt_portfolio:   list[Tuple[pd.Timestamp, float]] = []  # (ts, equity)

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[
        logging.FileHandler("trade_logic.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═════════════════════════════════════════════════════════════════════════════
def _load_history(ticker) -> pd.DataFrame:
    df = pd.read_csv(get_csv_file(ticker), parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    return df


def _add_future_return(df: pd.DataFrame) -> pd.DataFrame:
    """Add one-step-ahead % return column (future_return)."""
    df = df.copy()
    df[TARGET_COL] = df["close"].shift(-1) / df["close"] - 1.0
    df.dropna(inplace=True)
    return df


def _train_symbolic(
    df_hist: pd.DataFrame,
    force_retrain: bool = False,
) -> Tuple[SymbolicRegressor, str]:
    global _symbolic_model, _symbolic_expr, _last_fit_ts

    last_ts = df_hist["timestamp"].max()

    # Avoid redundant work (live logic): reuse model if already fit on full data
    if (_symbolic_model is not None
            and _last_fit_ts is not None
            and _last_fit_ts == last_ts
            and not force_retrain):
        return _symbolic_model, _symbolic_expr  # type: ignore[return-value]

    # If we have a cached file for identical timestamp we can load it
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            cached_ts, cached_model, cached_expr = pickle.load(f)
        if cached_ts == last_ts and not force_retrain:
            _symbolic_model, _symbolic_expr, _last_fit_ts = (
                cached_model, cached_expr, cached_ts
            )
            return cached_model, cached_expr

    # ─────────────────── actual training ────────────────────────────────────
    df_train = _add_future_return(df_hist)
    X = df_train[FEATURES].values
    y = df_train[TARGET_COL].values

    # Split purely by chronology (no leakage!)
    split_idx = int(len(df_train) * 0.7)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Custom trading-oriented fitness function ------------------------------
# ─── inside _train_symbolic() – replace the metric definition section ───────
    # Custom trading-oriented fitness function ------------------------------
    def trading_metric(y_true, y_pred, sample_weight):
        long_mask      = y_pred > 0.0
        strategy_ret   = np.where(long_mask, y_true, 0.0)
        if len(strategy_ret) == 0:
            return -np.inf                       # no trades → terrible fitness
        total_return   = np.prod(1.0 + strategy_ret) - 1.0
        excess_ret     = strategy_ret - strategy_ret.mean()
        sharpe         = (
            excess_ret.mean() / (excess_ret.std() + 1e-9)
            if excess_ret.std() > 0 else 0.0
        )
        return total_return + sharpe             # bigger = better

    gp_fitness = make_fitness(
        function=trading_metric,     # ✔  keyword argument
        greater_is_better=True,
        wrap=True,                   # (default, keeps the original docstring)
    )


    est = SymbolicRegressor(
        population_size=2000,
        generations=20,
        tournament_size=20,
        stopping_criteria=0.0,
        metric=gp_fitness,        # ← use the wrapped fitness
        parsimony_coefficient=0.01,
        verbose=0,
        random_state=42,
        n_jobs=1,
    )

    est.fit(X_train, y_train)

    expr = est._program.__str__()
    logger.info(f"🔧 New symbolic equation trained on {len(df_train)} samples")
    logger.info(f"    Expression: {expr}")

    # Cache to disk
    with open(MODEL_PATH, "wb") as f:
        pickle.dump((last_ts, est, expr), f)

    _symbolic_model, _symbolic_expr, _last_fit_ts = est, expr, last_ts
    return est, expr


def _signal_from_equation(
    model: SymbolicRegressor,
    row: pd.Series,
    threshold: float = 0.0,
) -> str:
    features = row[FEATURES].values.reshape(1, -1)
    pred = model.predict(features)[0]

    if pred > threshold:
        return "BUY"
    elif pred < -threshold:
        return "SELL"
    else:
        return "NONE"


def _log_trade(ts, action, price, qty, pl=0.0):
    header = not LOG_PATH.exists()

    ts_pd = pd.Timestamp(ts)
    if ts_pd.tzinfo is None:
        ts_pd = ts_pd.tz_localize("UTC")
    else:
        ts_pd = ts_pd.tz_convert("UTC")

    with open(LOG_PATH, "a") as f:
        if header:
            f.write("timestamp,action,price,qty,pl\n")
        f.write(f"{ts_pd.isoformat()},{action},{price:.5f},{qty},{pl:.5f}\n")



def _save_report(expr: str, sharpe: float, max_dd: float, total_ret: float):
    """Persist final summary to text file."""
    with open(REPORT_PATH, "w") as f:
        f.write(f"Symbolic equation: {expr}\n")
        f.write(f"Sharpe ratio     : {sharpe:.4f}\n")
        f.write(f"Max drawdown     : {max_dd:.4%}\n")
        f.write(f"Total return     : {total_ret:.4%}\n")

def run_logic(
    current_price: float,
    predicted_price: float,
    ticker: str,
):
    """
    Live / real-time logic.

    * Loads full history and (re)trains symbolic equation once (cached)
    * Decides BUY / SELL / NONE strictly from symbolic signal
    * Uses forest API buy_shares / sell_shares if available
    * Enforces single long position only
    """
    # ------------------------------------------------------------------ setup
    df_full = _load_history(ticker)
    model, expr = _train_symbolic(df_full)  # reused if already trained
    latest_row = df_full.iloc[-1]           # last candle – features available now

    # ---------------------------------------------------------------- decision
    signal = _signal_from_equation(model, latest_row)

    # forest account ---------------------------------------------------------
    try:
        account = api.get_account() if api else None
        cash    = float(account.cash) if account else 100_000.0
    except Exception:
        cash = 100_000.0

    try:
        position_qty = (
            float(api.get_position(ticker).qty) if api else 0.0
        )
    except Exception:
        position_qty = 0.0

    logger.info(
        f"[{ticker}] Live signal {signal} — "
        f"price={current_price:.2f}, predicted_price={predicted_price:.2f}, "
        f"expr_pred={model.predict(latest_row[FEATURES].values.reshape(1, -1))[0]:.6f}"
    )

    # ---------------------------------------------------------------- action
    if signal == "BUY" and position_qty == 0:
        max_shares = int(cash // current_price)
        if max_shares > 0:
            logger.info(f"[{ticker}] BUY {max_shares} @ {current_price:.2f}")
            buy_shares(ticker, max_shares, current_price, predicted_price)
            _log_trade(datetime.utcnow(), "BUY", current_price, max_shares)
    elif signal == "SELL" and position_qty > 0:
        logger.info(f"[{ticker}] SELL {position_qty} @ {current_price:.2f}")
        sell_shares(ticker, position_qty, current_price, predicted_price)
        _log_trade(datetime.utcnow(), "SELL", current_price, int(position_qty))
    else:
        logger.info(f"[{ticker}] No trade (signal={signal}, pos={position_qty})")


# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(
    current_price: float,
    predicted_price: float,
    position_qty: int,
    current_timestamp: Union[pd.Timestamp, datetime, str],
    candles: pd.DataFrame,
    ticker
) -> str:
    global _bt_cash, _bt_position, _bt_portfolio

    current_timestamp = pd.Timestamp(current_timestamp)

    # --------------------------------------------------- prepare in-sample df
    df_hist = _load_history(ticker)
    df_hist = df_hist[df_hist["timestamp"] <= current_timestamp]
    if len(df_hist) < 100:
        return "NONE"   # not enough data yet

    # (Re)train up-to-now model
    model, expr = _train_symbolic(df_hist, force_retrain=True)

    # Take last row features (the current candle in back-test scope)
    latest_row = df_hist.iloc[-1]
    signal = _signal_from_equation(model, latest_row)

    # ---------------------- execute virtual portfolio rules (single long only)
    action = "NONE"
    if signal == "BUY" and _bt_position == 0:
        shares_to_buy = int(_bt_cash // current_price)
        if shares_to_buy > 0:
            _bt_cash      -= shares_to_buy * current_price
            _bt_position  += shares_to_buy
            _log_trade(current_timestamp, "BUY", current_price, shares_to_buy)
            action = "BUY"

    elif signal == "SELL" and _bt_position > 0:
        _bt_cash     += _bt_position * current_price
        _log_trade(
            current_timestamp,
            "SELL",
            current_price,
            _bt_position,
            pl=_bt_position * (current_price - latest_row["open"]),
        )
        _bt_position = 0
        action = "SELL"

    # Update equity curve
    equity = _bt_cash + _bt_position * current_price
    _bt_portfolio.append((current_timestamp, equity))

    logger.info(
        f"[BT] {current_timestamp}  price={current_price:.2f}  "
        f"sig={signal}  action={action}  pos={_bt_position}  cash={_bt_cash:.2f}  "
        f"equity={equity:.2f}"
    )

    # Optionally, at the end of dataset, write report & plot
    if len(candles) > 0 and current_timestamp == candles["timestamp"].max():
        _finalise_backtest(expr)

    return action


# ═════════════════════════════════════════════════════════════════════════════
# Internal: final back-test report & plot
# ═════════════════════════════════════════════════════════════════════════════
def _finalise_backtest(expr: str):
    if not _bt_portfolio:
        return
    ts, equity_curve = zip(*_bt_portfolio)
    equity_curve = np.array(equity_curve)
    returns = equity_curve[1:] / equity_curve[:-1] - 1.0

    total_return = equity_curve[-1] / equity_curve[0] - 1.0
    sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252 * 24)
    roll_max = np.maximum.accumulate(equity_curve)
    drawdowns = 1.0 - equity_curve / roll_max
    max_dd = drawdowns.max()

    _save_report(expr, sharpe, max_dd, total_return)

    # Save plot --------------------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(ts, equity_curve, linewidth=1.4)
    plt.title(f"Portfolio value")
    plt.xlabel("Time")
    plt.ylabel("Equity ($)")
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()

    logger.info("Back-test complete → report & plot saved")