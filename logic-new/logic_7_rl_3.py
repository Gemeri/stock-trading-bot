import os
import gym
import numpy as np
import pandas as pd
from gym import spaces
from stable_baselines3 import PPO
from dotenv import load_dotenv
from forest import api, buy_shares, sell_shares
from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm

# Load environment variables
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


def get_csv_filename(ticker: str) -> str:
    """
    Builds the CSV filename from the ticker and timeframe.
    """
    return f"{ticker}_{CONVERTED_TIMEFRAME}.csv"


def load_data(ticker: str) -> pd.DataFrame:
    """
    Loads and preprocesses historical OHLCV and indicator data for the given ticker.
    Fills and drops NaNs to avoid invalid observations.
    """
    filename = get_csv_filename(ticker)
    df = pd.read_csv(filename, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)

    # Handle missing values
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


class ProgressBarCallback(BaseCallback):
    """
    A SB3 callback that updates a tqdm bar every rollout.
    """
    def __init__(self, total_timesteps: int, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:
        # initialize tqdm bar
        self.pbar = tqdm(total=self.total_timesteps,
                         desc="RL training",
                         unit="step",
                         leave=True)

    def _on_rollout_end(self) -> None:
        # SB3 calls this after each rollout of n_steps
        # self.num_timesteps is cumulative so far
        self.pbar.n = self.num_timesteps
        self.pbar.refresh()

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        # close bar cleanly
        self.pbar.close()

class TradingEnv(gym.Env):
    """
    Custom Gym environment for RL-based trading.
    Observations: all features + cash + position_qty
    Actions: 0=HOLD, 1=BUY, 2=SELL
    Reward: change in total portfolio value.
    """
    def __init__(self, data: pd.DataFrame):
        super().__init__()
        self.data = data
        self.n_steps = len(data)
        self.current_step = 0
        self.initial_cash = 100_000.0
        self.cash = self.initial_cash
        self.position_qty = 0
        self.feature_cols = [c for c in data.columns if c != "timestamp"]
        self.obs_dim = len(self.feature_cols) + 2  # features + cash + position_qty

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.position_qty = 0
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        values = self.data.loc[self.current_step, self.feature_cols].values.astype(np.float32)
        # ensure no NaNs or infs
        assert np.all(np.isfinite(values)), "NaN or inf in observation features"
        return np.concatenate([values, [self.cash, self.position_qty]], axis=0)

    def step(self, action: int):
        current_price = float(self.data.loc[self.current_step, "close"])
        prev_value = self.cash + self.position_qty * current_price

        # Execute action
        if action == 1 and self.position_qty == 0:
            self.position_qty = int(self.cash // current_price)
            self.cash -= self.position_qty * current_price
        elif action == 2 and self.position_qty > 0:
            self.cash += self.position_qty * current_price
            self.position_qty = 0

        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.n_steps

        # Compute reward
        next_price = current_price if done else float(self.data.loc[self.current_step, "close"])
        curr_value = self.cash + self.position_qty * next_price
        reward = curr_value - prev_value

        obs = self._get_observation() if not done else np.zeros(self.obs_dim, dtype=np.float32)
        return obs, reward, done, {}


def train_rl_agent(data: pd.DataFrame) -> PPO:
    """
    Trains an RL agent on the provided historical data and returns the trained model.
    """
    env = TradingEnv(data)
    model = PPO("MlpPolicy", env, verbose=0)
    # Train with enough timesteps for stable performance
    # attach the tqdm-based progress bar
    progress_callback = ProgressBarCallback(total_timesteps=200_000)
    model.learn(total_timesteps=200_000,
               callback=progress_callback)    
    return model


def run_logic(current_price: float, predicted_price: float, ticker: str):
    """
    Main trading entrypoint: trains on full history, then executes an RL-derived action.
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        account = api.get_account(); cash = float(account.cash)
    except Exception as e:
        logger.error(f"[{ticker}] Error fetching account: {e}"); return
    try:
        pos = api.get_position(ticker); position_qty = float(pos.qty)
    except Exception:
        position_qty = 0.0

    data = load_data(ticker)
    model = train_rl_agent(data)

    last = data.iloc[-1]
    feat = last[data.columns.difference(["timestamp"])].values.astype(np.float32)
    obs = np.concatenate([feat, [cash, position_qty]], axis=0)

    action, _ = model.predict(obs, deterministic=True)
    logger.info(f"[{ticker}] RL action: {action}")

    if action == 1:
        qty = int(cash // current_price)
        if qty > 0:
            logger.info(f"[{ticker}] Buying {qty} shares at {current_price}")
            buy_shares(ticker, qty, current_price, predicted_price)
        else:
            logger.info(f"[{ticker}] No cash for BUY")
    elif action == 2:
        if position_qty > 0:
            logger.info(f"[{ticker}] Selling {position_qty} shares at {current_price}")
            sell_shares(ticker, position_qty, current_price, predicted_price)
        else:
            logger.info(f"[{ticker}] No position to SELL")
    else:
        logger.info(f"[{ticker}] HOLD action")


def run_backtest(current_price: float, predicted_price: float, position_qty: float,
                 current_timestamp: str, candles: pd.DataFrame) -> str:
    """
    Backtest entrypoint: trains on history up to current_timestamp and returns BUY/SELL/NONE.
    """
    base_ticker = TICKERS[0]
    full = load_data(base_ticker)
    full["timestamp"] = pd.to_datetime(full["timestamp"])
    cutoff = pd.to_datetime(current_timestamp)
    train_data = full[full["timestamp"] <= cutoff].reset_index(drop=True)

    model = train_rl_agent(train_data)

    row = train_data[train_data["timestamp"] == cutoff].iloc[0]
    feat = row[train_data.columns.difference(["timestamp"])].values.astype(np.float32)
    cash = 0.0 if position_qty > 0 else 100_000.0
    obs = np.concatenate([feat, [cash, position_qty]], axis=0)

    action, _ = model.predict(obs, deterministic=True)
    if action == 1 and position_qty == 0:
        return "BUY"
    if action == 2 and position_qty > 0:
        return "SELL"
    return "NONE"
