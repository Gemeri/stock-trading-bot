import os
import logging
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime
from pysr import PySRRegressor
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
import matplotlib.pyplot as plt

# =============================================================================
# Environment & CSV utilities
# =============================================================================
load_dotenv()
BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "1Hour")
TICKERS = os.getenv("TICKERS", "TSLA").split(",")
TIMEFRAME_MAP = {
    "4Hour": "H4",
    "2Hour": "H2",
    "1Hour": "H1",
    "30Min": "M30",
    "15Min": "M15"
}
CONVERTED_TIMEFRAME = TIMEFRAME_MAP.get(BAR_TIMEFRAME, BAR_TIMEFRAME)


def get_csv_filename() -> str:
    """Return the filename for the historical data CSV."""
    return f"{TICKERS[0]}_{CONVERTED_TIMEFRAME}.csv"


def load_historical_data() -> pd.DataFrame:
    """Load full historical CSV into a DataFrame with datetime index."""
    fn = get_csv_filename()
    df = pd.read_csv(fn, parse_dates=["timestamp"] )
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# =============================================================================
# Symbolic Regression Training & Hyperparameter Optimization
# =============================================================================

def optimize_hyperparams(X: pd.DataFrame, y: np.ndarray) -> dict:
    """
    Use Bayesian optimization to tune PySR hyperparameters.
    Returns the best parameters dict.
    """
    # Define search space
    space = [
        Integer(50, 500, name="niterations"),
        Integer(10, 100, name="population_size"),
        Integer(3, 10, name="maxsize"),
        Categorical(["+", "-", "*", "/", "sin", "cos"], name="binary_operators")
    ]

    def objective(params_list):
        niter, pop_size, max_size, operator = params_list
        model = PySRRegressor(
            niterations=niter,
            population_size=pop_size,
            maxsize=max_size,
            binary_operators=[operator],
            unary_operators=["sin", "cos"],
            verbosity=0,
            random_state=0
        )
        # Simple expanding-window validation
        train_size = int(0.7 * len(X))
        X_train, y_train = X.iloc[:train_size], y[:train_size]
        X_val,   y_val   = X.iloc[train_size:], y[train_size:]
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        loss = np.mean((preds - y_val) ** 2)
        # complexity penalty
        complexity = len(model.equations_) if hasattr(model, 'equations_') else max_size
        return loss + 0.01 * complexity

    res = gp_minimize(objective, space, n_calls=20, random_state=0)
    best = {
        "niterations": res.x[0],
        "population_size": res.x[1],
        "maxsize": res.x[2],
        "binary_operators": [res.x[3]],
        "unary_operators": ["sin", "cos"],
        "random_state": 0
    }
    return best


def train_symbolic_model(df: pd.DataFrame) -> PySRRegressor:
    """
    Train a symbolic regression model on input DataFrame to predict 3-candle future return.
    Returns trained PySRRegressor model.
    """
    # Prepare features & target
    features = df.drop(columns=["timestamp"])
    # Create target: 3-candle future return direction
    fut_return = df["close"].shift(-3) / df["close"] - 1
    target = np.where(fut_return > 0.002, 1,
             np.where(fut_return < -0.002, -1, 0))
    mask = ~np.isnan(fut_return)
    X = features.iloc[mask.values]
    y = target[mask.values]

    # Hyperparameter optimization
    best_params = optimize_hyperparams(X, y)

    # Final model training on full data
    model = PySRRegressor(
        **best_params,
        extra_torch=False,
        niterations=best_params["niterations"],
        population_size=best_params["population_size"],
        maxsize=best_params["maxsize"]
    )
    model.fit(X, y)

    # Save equation
    eq_file = "symbolic_equation.txt"
    with open(eq_file, "w") as f:
        f.write(model.get_best())
    return model

# =============================================================================
# Threshold Selection
# =============================================================================

def select_signal_threshold(model: PySRRegressor, X: pd.DataFrame, y: np.ndarray) -> dict:
    """
    Determine optimal signal threshold maximizing profit on training set.
    Returns thresholds dict.
    """
    preds = model.predict(X)
    # Evaluate candidate thresholds based on quantiles
    q_low, q_high = np.percentile(preds, [10, 90])
    return {"low": q_low, "high": q_high}

# =============================================================================
# Run Logic (Live Trading)
# =============================================================================

def run_logic(current_price: float, predicted_price: float, ticker: str):
    """
    Entry-point for live trading logic.
    Trains symbolic model on full CSV and executes BUY/SELL/NONE.
    """
    from forest import api, buy_shares, sell_shares

    logger = logging.getLogger(__name__)
    df = load_historical_data()
    model = train_symbolic_model(df)

    # Prepare latest features
    latest = df.drop(columns=["timestamp"]).iloc[[-1]]
    signal_cont = model.predict(latest)[0]

    # Compute threshold
    features = df.drop(columns=["timestamp"])
    fut_return = df["close"].shift(-3) / df["close"] - 1
    mask = ~np.isnan(fut_return)
    X_train = features.iloc[mask.values]
    y_train = np.where(fut_return[mask.values] > 0, 1,
                       np.where(fut_return[mask.values] < 0, -1, 0))
    thresh = select_signal_threshold(model, X_train, y_train)

    # Fetch cash & position
    try:
        account = api.get_account()
        cash = float(account.cash)
    except Exception as e:
        logger.error(f"[{ticker}] Error fetching account: {e}")
        return
    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception:
        position_qty = 0.0

    logger.info(f"[{ticker}] Signal: {signal_cont}, Thresholds: {thresh}")

    # Decide action
    if signal_cont >= thresh['high'] and position_qty == 0:
        max_shares = int(cash // current_price)
        if max_shares > 0:
            logger.info(f"[{ticker}] BUY {max_shares}")
            buy_shares(ticker, max_shares, current_price, predicted_price)
    elif signal_cont <= thresh['low'] and position_qty > 0:
        logger.info(f"[{ticker}] SELL {position_qty}")
        sell_shares(ticker, int(position_qty), current_price, predicted_price)
    else:
        logger.info(f"[{ticker}] NONE")

# =============================================================================
# Run Backtest Logic
# =============================================================================

def run_backtest(current_price: float,
                 predicted_price: float,
                 position_qty: float,
                 current_timestamp: str,
                 candles: pd.DataFrame) -> str:
    """
    Entry-point for backtesting logic, returns "BUY", "SELL", or "NONE".
    Trains model on historical CSV up to current_timestamp.
    """
    df_full = load_historical_data()
    cutoff = pd.to_datetime(current_timestamp)
    df_hist = df_full[df_full['timestamp'] <= cutoff]

    model = train_symbolic_model(df_hist)

    # Prepare latest row from provided candles
    latest_candle = candles.iloc[[-1]].drop(columns=["timestamp"])
    signal_cont = model.predict(latest_candle)[0]

    # Thresholds from training
    features = df_hist.drop(columns=["timestamp"])
    fut_return = df_hist["close"].shift(-3) / df_hist["close"] - 1
    mask = ~np.isnan(fut_return)
    X_train = features.iloc[mask.values]
    y_train = np.where(fut_return[mask.values] > 0, 1,
                       np.where(fut_return[mask.values] < 0, -1, 0))
    thresh = select_signal_threshold(model, X_train, y_train)

    # Decide action
    if signal_cont >= thresh['high'] and position_qty == 0:
        return "BUY"
    elif signal_cont <= thresh['low'] and position_qty > 0:
        return "SELL"
    else:
        return "NONE"

# =============================================================================
# Backtest Runner & Report
# =============================================================================

def full_backtest_report():
    """
    Execute a full backtest over the last 20% of data and generate metrics.
    Saves trades.csv, portfolio.png, and summary.txt.
    """
    df = load_historical_data()
    split = int(0.8 * len(df))
    backtest_df = df.iloc[split:].copy()

    capital = 10000.0
    position = 0
    cash = capital
    portfolio = []  # track portfolio value
    trades = []

    for idx, row in backtest_df.iterrows():
        ts = row['timestamp']
        cp = row['close']
        pred = row['predicted_close']
        action = run_backtest(cp, pred, position, ts, backtest_df.iloc[max(0, idx-500):idx+1])
        if action == "BUY" and position == 0:
            shares = int(cash // cp)
            cost = shares * cp
            position = shares
            cash -= cost
            trades.append((ts, 'BUY', cp, shares, cash))
        elif action == "SELL" and position > 0:
            proceeds = position * cp
            trades.append((ts, 'SELL', cp, position, proceeds - capital))
            cash += proceeds
            position = 0
        portfolio.append(cash + position * cp)

    # Save trades
    trades_df = pd.DataFrame(trades, columns=["timestamp", "action", "price", "shares", "cash_or_pl"])
    trades_df.to_csv("trades.csv", index=False)

    # Plot portfolio curve
    plt.figure()
    plt.plot(backtest_df['timestamp'], portfolio)
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")
    plt.savefig("portfolio.png")

    # Compute metrics
    returns = pd.Series(portfolio).pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    peak = np.maximum.accumulate(portfolio)
    drawdown = (portfolio - peak) / peak
    max_dd = drawdown.min()
    total_return = (portfolio[-1] / capital - 1) * 100

    with open("summary.txt", "w") as f:
        f.write(f"Sharpe Ratio: {sharpe:.2f}\n")
        f.write(f"Max Drawdown: {max_dd:.2%}\n")
        f.write(f"Total Return: {total_return:.2f}%\n")

    logging.getLogger(__name__).info("Backtest complete. Reports saved.")

# Expose only the two entry-points
__all__ = ["run_logic", "run_backtest"]
