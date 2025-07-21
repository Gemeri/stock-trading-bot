"""
quick_compare_ppo_lstm_net_arch.py
Compare PPO-LSTM with net_arch [128] vs [256] on TSLA 4-hour candles
-------------------------------------------------------------------
Requirements:
    pip install pandas numpy gymnasium sb3-contrib stable-baselines3
"""

import os
import warnings
import numpy as np
import pandas as pd
from typing import List

# stable-baselines3 + contrib
from sb3_contrib.ppo_recurrent import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

warnings.filterwarnings("ignore", category=FutureWarning)

# ────────────────────────────────────────────────────────────────────
# 1. Load & prep data
# ────────────────────────────────────────────────────────────────────
DATA_PATH = "TSLA_H4.csv"           # ← change if needed
TRAIN_TIMESTEPS = 15_000            # quick but useful
ROLLING_WINDOW = 1_400              # last 1400 candles
SEED = 42

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).sort_values("timestamp")
if "predicted_close" in df.columns:
    df = df.drop(columns=["predicted_close"])

FEATURE_COLS: List[str] = [c for c in df.columns if c != "timestamp"]
assert "close" in FEATURE_COLS, "`close` column missing!"

# keep only most-recent rolling window
df = df.iloc[-ROLLING_WINDOW:].reset_index(drop=True)

# ────────────────────────────────────────────────────────────────────
# 2. Minimal trading environment with custom reward
# ────────────────────────────────────────────────────────────────────
import gymnasium as gym
from gymnasium import spaces

class TSLA4HEnv(gym.Env):
    def __init__(self, frame: pd.DataFrame):
        super().__init__()
        self.df = frame.reset_index(drop=True)
        self.n_steps = len(frame)
        self.cur = 0
        self.action_space = spaces.Discrete(3)  # 0 NONE, 1 BUY, 2 SELL
        # obs = features + last action one-hot (optional) – here features only
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(FEATURE_COLS),), dtype=np.float32
        )

    # helpers
    def _get_obs(self):
        obs = self.df.loc[self.cur, FEATURE_COLS].to_numpy(dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0., posinf=0., neginf=0.)
        return obs

    # gym API
    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self.cur = 0
        return self._get_obs(), {}

    def step(self, action_int: int):
        # map int→str for reward formula
        action_map = {0: "NONE", 1: "BUY", 2: "SELL"}
        action_str = action_map[int(action_int)]

        current_price = float(self.df.loc[self.cur, "close"])
        if self.cur >= 1:
            previous_close = float(self.df.loc[self.cur - 1, "close"])
        else:
            previous_close = current_price

        if action_str != "NONE":
            trade_dir = 1 if action_str in ["BUY", "COVER"] else -1
            reward = (current_price - previous_close) * trade_dir
        else:
            reward = -0.1  # penalty for idleness

        self.cur += 1
        done = self.cur >= self.n_steps
        obs = self._get_obs() if not done else np.zeros(len(FEATURE_COLS), dtype=np.float32)
        return obs, reward, done, False, {}

# ────────────────────────────────────────────────────────────────────
# 3. Utility: train & evaluate a single agent
# ────────────────────────────────────────────────────────────────────
def train_and_eval(net_size: int, label: str) -> float:
    env = DummyVecEnv([lambda: TSLA4HEnv(df)])
    model = RecurrentPPO(
        policy=RecurrentActorCriticPolicy,
        env=env,
        verbose=1,
        seed=42,
        n_steps=1_024,
        batch_size=128,
        n_epochs=8,
        learning_rate=2.5e-4,
        gamma=0.97,
        gae_lambda=0.92,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
        policy_kwargs=dict(
            lstm_hidden_size=256,
            net_arch=dict(pi=[128], vf=[128]),
        ),
    )
    model.learn(total_timesteps=42_000, progress_bar=True)
    # quick evaluation: run 10 fresh episodes, average reward
    returns = []
    for _ in range(10):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            ep_ret += reward
        returns.append(ep_ret)
    mean_ret = float(np.mean(returns))
    print(f"[{label}] mean 10-episode return: {mean_ret:8.2f}")
    return mean_ret

# ────────────────────────────────────────────────────────────────────
# 4. Run comparison
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🔎 Comparing PPO-LSTM net_arch sizes on TSLA_H4 (quick run)\n")
    ret128 = train_and_eval(128, "net_arch-128")
    ret256 = train_and_eval(256, "net_arch-256")

    if ret256 > ret128:
        better = "256"
    elif ret128 > ret256:
        better = "128"
    else:
        better = "tie"

    print("\n🏁 Result:")
    if better == "tie":
        print("   Both configurations performed similarly.")
    else:
        print(f"   net_arch-{better} achieved the higher mean return.")
