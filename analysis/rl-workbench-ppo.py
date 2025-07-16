import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import json

# Ignore future warnings from wrapping Gym environments
# warnings.filterwarnings("ignore", category=DeprecationWarning)

TICKER = "AAPL"
INTERVAL = "H4"

optionList = [
    #'128-20000',
    #'128-250000',
    '128-1000000',
    #'256-20000',
    #'256-250000',
    '256-1000000',
    #'512-20000',
    #'512-250000',
    '512-1000000',
]


# Load data
df = pd.read_csv(f"../data/{TICKER}_{INTERVAL}.csv")
df = df.dropna().reset_index(drop=True)

class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df
        self.n_steps = len(df)
        self.current_step = 0
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.position = 0
        self.total_shares = 0
        self.net_worth = self.initial_balance
        self.max_steps = len(df) - 1

        self.features = df[[
            'vwap','high', 'low', 'close', 'open', 
            'macd_line', 'macd_signal', 'macd_histogram', 'rsi', 
            'ema_9', 'ema_21', 'ema_50','ema_200',
            'lagged_close_1','lagged_close_2','lagged_close_3','lagged_close_5','lagged_close_10',
            'momentum', 
            'atr', 'volume_zscore', 'volume', 'days_since_high']].values

        self.action_space = spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.features.shape[1],), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.total_shares = 0
        self.net_worth = self.initial_balance
        self.position = 0
        obs = self._next_observation()
        info = {}
        return obs, info

    def _next_observation(self):
        return self.features[self.current_step].astype(np.float32)

    def step(self, action):
        price = self.df.loc[self.current_step, 'close']

        if action == 1 and self.balance >= price:
            self.total_shares += 1
            self.balance -= price
        elif action == 2 and self.total_shares > 0:
            self.total_shares -= 1
            self.balance += price

        self.net_worth = self.balance + self.total_shares * price
        reward = self.net_worth - self.initial_balance
        self.current_step += 1

        terminated = self.current_step >= self.max_steps
        truncated = False
        obs = self._next_observation()
        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        print(f'Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Shares: {self.total_shares}, Balance: {self.balance:.2f}')

    def get_net_worth(self):
        return self.net_worth

# Wrap the environment
env = DummyVecEnv([lambda: TradingEnv(df)])


for option in optionList:

    n_steps = int(option.split('-')[0])
    total_timesteps = int(option.split('-')[1])

    print(f"Attempting {TICKER} -> n_steps={n_steps} / total_timesteps={total_timesteps}")

    # Train PPO
    model = PPO("MlpPolicy", env, verbose=0, n_steps=n_steps)
    model.learn(total_timesteps=total_timesteps)

    print(f"Completed {TICKER} -> n_steps={n_steps} / total_timesteps={total_timesteps}")

    # Evaluation
    obs = env.reset()
    net_worths = []

    for _ in range(len(df)-1):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        
        net_worth = env.get_attr('get_net_worth')[0]()  
        net_worths.append(net_worth)

        if done[0]:
            break

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(net_worths, label="Net Worth")
    plt.title("PPO Agent Net Worth Over Time (Gymnasium Compatible)")
    plt.xlabel("Steps")
    plt.ylabel("Net Worth ($)")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Save the figure to disk
    plt.savefig(f"output/ppo_net_worth_{TICKER}_{INTERVAL}_{n_steps}_{total_timesteps}.png", dpi=300)  # You can also use .jpg, .pdf, etc.

    print(f"Saved picture for {TICKER} -> n_steps={n_steps} / total_timesteps={total_timesteps}")

print("All completed")