"""
TSLA 4-hour PPO trader
──────────────────────
• Loads data/data/TSLA_H4.csv and **only** the approved feature columns  
• Uses a custom Gym⟶SB3 environment with three discrete actions  
      0 = HOLD  (aka “NONE”)
      1 = BUY   (open/extend long up to all available cash)
      2 = SELL  (close entire long)  
• Starts every training cycle with USD 1 000 cash and zero position  
• Implements the reward

        r =
        \begin{cases}
            (P_t - P_{t-1}) \,\times\, \text{trade\_dir}, & \text{if action ≠ HOLD}\\[4pt]
            -0.1, & \text{if action = HOLD}
        \end{cases}

  where trade_dir = +1 for BUY/COVER and –1 for SELL/SHORT  

• Trains on a rolling 1 400-candle window that advances with the live feed  
  – Retrains **from scratch** every 30 environment steps (≈30 new candles)  
  – Performs a 1-step online update after **every** action for fast adaptation  
  – Stores / loads checkpoints automatically  
• Exposes the two integration hooks you requested:
      ▸ run_logic()      – calls the live policy and routes orders
      ▸ run_backtest()   – strictly inspects data ≤ current_timestamp to prevent leakage
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────
from pathlib import Path
import logging
import pickle
import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ──────────────────────────────────────────────────────────────────────────────
# Globals & Configuration
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH          = Path("data") / "TSLA_H4.csv"
CHECKPOINT_DIR     = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "timestamp","open","high","low","close","volume","vwap","transactions","sentiment",
    "price_change","high_low_range","log_volume","macd_line","macd_signal","macd_histogram",
    "rsi","momentum","roc","atr","ema_9","ema_21","ema_50","ema_200","adx","obv",
    "bollinger_upper","bollinger_lower","lagged_close_1","lagged_close_2","lagged_close_3",
    "lagged_close_5","lagged_close_10","candle_body_ratio","wick_dominance","gap_vs_prev",
    "volume_zscore","atr_zscore","rsi_zscore","adx_trend","macd_cross","macd_hist_flip",
    "day_of_week","days_since_high","days_since_low",
]

STARTING_BALANCE    = 1_000.0          # USD
ROLLING_WINDOW      = 1_400           # candles in rolling train set
RETRAIN_EVERY       = 30              # candles before full retrain
PPO_TOTAL_STEPS     = 42_000          # full-cycle train steps
LOG                 = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                    level=logging.INFO)

# Runtime state (persisted across function calls)
state = {
    "df"               : None,   # full dataframe
    "env"              : None,   # live TradingEnv instance
    "model"            : None,   # current SB3-PPO model
    "candle_idx"       : 0,      # index of latest candle we have stepped on
    "steps_until_retrain": RETRAIN_EVERY,
}

# ──────────────────────────────────────────────────────────────────────────────
# Utility: Data loading & filtering
# ──────────────────────────────────────────────────────────────────────────────
def load_csv() -> pd.DataFrame:
    """Load TSLA_H4.csv and keep ONLY the approved columns."""
    df = pd.read_csv(DATA_PATH)
    # Ensure the unwanted column is dropped silently if present
    drop_cols = [c for c in df.columns if c not in FEATURES]
    df = df.drop(columns=drop_cols)
    # Ensure chronological order (older→newer)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

# ──────────────────────────────────────────────────────────────────────────────
# Gym environment
# ──────────────────────────────────────────────────────────────────────────────
class TradingEnv(gym.Env):
    """
    A minimalistic long-only TSLA environment.
    Observation  : the full feature vector (np.float32)
    Actions      : 0=HOLD, 1=BUY (max), 2=SELL (all)
    Reward       : price movement proxy described above
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, start_idx: int):
        super().__init__()
        self.df              = df
        self.start_idx       = start_idx
        self.end_idx         = len(df) - 1
        self.ptr             = start_idx
        self.balance         = STARTING_BALANCE
        self.shares_held     = 0
        self.last_price      = float(df.loc[self.ptr, "close"])

        self.action_space      = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(FEATURES) - 1,), dtype=np.float32
        )

    # ──────────────────────────────────────────────────────────────────────
    def _get_obs(self):
        # Drop 'timestamp' from observation
        obs = self.df.loc[self.ptr, FEATURES[1:]].astype(np.float32).to_numpy()
        return obs

    # ──────────────────────────────────────────────────────────────────────
    def _calculate_reward(self, action: int, current_price: float, prev_close: float):
        if action == 0:  # HOLD
            return -0.1
        trade_dir = 1 if action == 1 else -1
        return (current_price - prev_close) * trade_dir

    # ──────────────────────────────────────────────────────────────────────
    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"
        done = False

        prev_close   = self.last_price
        self.ptr    += 1
        if self.ptr >= self.end_idx:
            done = True
            self.ptr = self.end_idx  # clamp

        current_price = float(self.df.loc[self.ptr, "close"])
        self.last_price = current_price

        # Execute action
        if action == 1:  # BUY
            if self.shares_held == 0:
                max_shares = int(self.balance // current_price)
                if max_shares > 0:
                    self.balance      -= max_shares * current_price
                    self.shares_held  += max_shares
                else:
                    action = 0  # treat as HOLD if insufficient cash
        elif action == 2:  # SELL
            if self.shares_held > 0:
                self.balance     += self.shares_held * current_price
                self.shares_held  = 0
            else:
                action = 0  # treat as HOLD if nothing to sell

        reward = self._calculate_reward(action, current_price, prev_close)
        info   = {"balance": self.balance, "shares": self.shares_held}

        return self._get_obs(), reward, done, False, info

    # ──────────────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ptr          = self.start_idx
        self.balance      = STARTING_BALANCE
        self.shares_held  = 0
        self.last_price   = float(self.df.loc[self.ptr, "close"])
        return self._get_obs(), {}

    # ──────────────────────────────────────────────────────────────────────
    def render(self):
        print(f"T={self.ptr} | Price={self.last_price:.2f} | "
              f"Bal={self.balance:.2f} | Shares={self.shares_held}")

# ──────────────────────────────────────────────────────────────────────────────
# PPO factory
# ──────────────────────────────────────────────────────────────────────────────
def build_ppo(env: gym.Env) -> PPO:
    return PPO(
        policy="MlpPolicy",
        env=env,
        total_timesteps=PPO_TOTAL_STEPS,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        learning_rate=2.5e-4,
        gamma=0.97,
        gae_lambda=0.92,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        verbose=0,
    )

# ──────────────────────────────────────────────────────────────────────────────
# (Re)training & checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────
def retrain_model(current_end_idx: int):
    """
    • Extract the last 1 400 candles ending at current_end_idx (inclusive)  
    • Build fresh environment & model, train, and save checkpoint
    """
    start_idx = max(0, current_end_idx - ROLLING_WINDOW + 1)
    train_df  = state["df"].loc[start_idx:current_end_idx].reset_index(drop=True)

    LOG.info(f"Retraining PPO from scratch on candles [{start_idx}, {current_end_idx}] "
             f"({len(train_df)} rows)")

    env = DummyVecEnv([lambda: TradingEnv(train_df, start_idx=0)])
    model = build_ppo(env)
    model.learn(total_timesteps=PPO_TOTAL_STEPS, progress_bar=False)

    ckpt_file = CHECKPOINT_DIR / f"ppo_step_{current_end_idx}.pkl"
    with ckpt_file.open("wb") as f:  # ➜ lightweight pickle (policy + vec normalize)
        pickle.dump(model, f)
    LOG.info(f"Saved checkpoint ➜ {ckpt_file}")

    state.update({
        "env"  : env,
        "model": model,
    })

# ──────────────────────────────────────────────────────────────────────────────
# Initial bootstrap
# ──────────────────────────────────────────────────────────────────────────────
def _bootstrap():
    if state["df"] is None:
        state["df"] = load_csv()

    # Train initial model on the first rolling window ending at candle 1 399 (0-based)
    retrain_model(current_end_idx=ROLLING_WINDOW - 1)
    state["candle_idx"] = ROLLING_WINDOW - 1
    state["steps_until_retrain"] = RETRAIN_EVERY

_bootstrap()

# ──────────────────────────────────────────────────────────────────────────────
# Public API #1 – run_logic
# ──────────────────────────────────────────────────────────────────────────────
def run_logic(current_price: float, predicted_price: float, ticker: str):
    """
    Live-trading inference hook.  
    • current_price & ticker are forwarded from the outer framework  
    • predicted_price is currently **ignored** (the PPO is the sole signal)  
    • Uses forest.api helpers buy_shares / sell_shares as requested
    """
    from forest import api, buy_shares, sell_shares

    # Make sure dataframe has grown up to (or beyond) current_price timestamp
    global_idx = state["candle_idx"] + 1
    if global_idx >= len(state["df"]):
        LOG.warning("No new candle in CSV yet – skipping.")
        return

    # 1️⃣  Update the env with the new candle & obtain observation
    obs, _ = state["env"].envs[0].reset(options=None)  # vector env wrapper
    # advance to the correct internal ptr
    while state["env"].envs[0].ptr < global_idx:
        state["env"].envs[0].ptr += 1

    # 2️⃣  Predict action
    action, _ = state["model"].predict(obs, deterministic=True)

    # 3️⃣  Fetch account & position
    try:
        account = api.get_account()
        cash    = float(account.cash)
    except Exception as e:
        LOG.error(f"[{ticker}] Error fetching account details: {e}")
        return

    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception:
        position_qty = 0.0

    # 4️⃣  Map PPO action to brokerage
    if action == 1:  # BUY
        if position_qty == 0:
            max_shares = int(cash // current_price)
            if max_shares > 0:
                LOG.info(f"[{ticker}] BUY {max_shares} @ {current_price:.2f}")
                buy_shares(ticker, max_shares, current_price, predicted_price)
        else:
            LOG.info(f"[{ticker}] BUY signal ignored – already long.")
            action = 0  # treat as HOLD for reward
    elif action == 2:  # SELL
        if position_qty > 0:
            LOG.info(f"[{ticker}] SELL {position_qty} @ {current_price:.2f}")
            sell_shares(ticker, position_qty, current_price, predicted_price)
        else:
            LOG.info(f"[{ticker}] SELL signal ignored – no position.")
            action = 0
    else:
        LOG.debug(f"[{ticker}] HOLD action taken.")

    # 5️⃣  Advance env step & apply 1-step online learning
    _obs, reward, done, _, _ = state["env"].step(action)
    state["model"].learn(total_timesteps=1, reset_num_timesteps=False)

    # 6️⃣  Book-keeping & periodic full retrain
    state["candle_idx"]               = global_idx
    state["steps_until_retrain"]     -= 1
    if state["steps_until_retrain"] <= 0:
        retrain_model(current_end_idx=global_idx)
        state["steps_until_retrain"] = RETRAIN_EVERY

    LOG.info(f"[{ticker}] step={global_idx} | reward={reward:.4f} | "
             f"bal={state['env'].envs[0].balance:.2f} | "
             f"shares={state['env'].envs[0].shares_held}")

# ──────────────────────────────────────────────────────────────────────────────
# Public API #2 – run_backtest
# ──────────────────────────────────────────────────────────────────────────────
def run_backtest(current_timestamp: int, position_qty: float, *_, **__):
    """
    Back-testing hook called for each candle by an external harness.  
    Requirements met:  
      • *Only* current_timestamp & position_qty are consumed  
      • No future leaks: model is trained strictly on data ≤ current_timestamp  
      • Returns "BUY", "SELL", or "NONE"
    """
    # Ensure dataframe is loaded
    if state["df"] is None:
        state["df"] = load_csv()

    # Locate the row index for current_timestamp
    try:
        idx = state["df"].index[state["df"]["timestamp"] == current_timestamp][0]
    except IndexError:
        # Timestamp not found (incomplete feed) – abstain
        return "NONE"

    # (Re)train model if our current checkpoint is behind the required idx
    # We subtract one because we predict on candle idx+1
    if idx != state["candle_idx"]:
        retrain_model(current_end_idx=idx)
        state["candle_idx"] = idx

    # Prepare observation at idx
    env_inst = state["env"].envs[0]
    env_inst.ptr = idx
    obs = env_inst._get_obs()

    # Predict action
    action, _ = state["model"].predict(obs, deterministic=True)
    mapping = {0: "NONE", 1: "BUY", 2: "SELL"}

    # Ensure logic regarding already-held position
    if action == 1 and position_qty > 0:
        return "NONE"
    if action == 2 and position_qty == 0:
        return "NONE"
    return mapping[action]

# ──────────────────────────────────────────────────────────────────────────────
# CLI bootstrap (optional)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick smoke-test on the latest candle of the CSV
    ts_latest = state["df"].iloc[-1]["timestamp"]
    signal = run_backtest(ts_latest, position_qty=0.0)
    print(f"Latest back-test signal ➜ {signal}")
