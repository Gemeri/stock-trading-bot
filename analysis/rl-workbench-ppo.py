import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_envs import ShortTradingEnv, NormalTradingEnv
import torch

# Ignore future warnings from wrapping Gym environments
# warnings.filterwarnings("ignore", category=DeprecationWarning)

TICKER = "AAPL"
INTERVAL = "H1"
TRADING_TYPE = 'short'

optionList = [
    '256-100000',
    '256-250000',
    '256-500000',
]

FEATURE_COLUMNS = [
            'vwap','high', 'low', 'close', 'open', 
            'macd_line', 'macd_signal', 'macd_histogram', 'rsi', 
            'ema_9', 'ema_21', 'ema_50','ema_200',
            'lagged_close_1','lagged_close_2','lagged_close_3','lagged_close_5','lagged_close_10',
            'momentum', 
            'atr', 'volume_zscore', 'volume', 'days_since_high']

# seed setting
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# Load data
df = pd.read_csv(f"../data/{TICKER}_{INTERVAL}.csv")
df = df.dropna().reset_index(drop=True)

# Wrap the environment

if TRADING_TYPE == 'short':
    env = DummyVecEnv([lambda: ShortTradingEnv(df, FEATURE_COLUMNS)])
else:
    env = DummyVecEnv([lambda: NormalTradingEnv(df, FEATURE_COLUMNS)])

for option in optionList:

    n_steps = int(option.split('-')[0])
    total_timesteps = int(option.split('-')[1])

    print(f"Attempting {TICKER} -> n_steps={n_steps} / total_timesteps={total_timesteps}")

    # Train PPO
    model = PPO("MlpPolicy", env, 
                verbose=0, 
                n_steps=n_steps,
                batch_size=64,
                gae_lambda=0.95,            # default value
                gamma=0.95,
                n_epochs=10,
                ent_coef=0.005,             # Small entropy → some exploration
                learning_rate=2.5e-4,       # default value
                clip_range=0.2,             # default value
                max_grad_norm=0.5,
                vf_coef=0.5, 
                )
    model.learn(total_timesteps=total_timesteps)

    print(f"Completed {TICKER} -> n_steps={n_steps} / total_timesteps={total_timesteps}")

    # Evaluation
    obs = env.reset()
    net_worths = []
    stock_prices = []

    for step in range(len(df)-1):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        
        net_worth = env.get_attr('get_net_worth')[0]()  
        net_worths.append(net_worth)

        current_step = env.get_attr('current_step')[0]
        stock_price = df['close'].iloc[current_step]
        stock_prices.append(stock_price)

        if done[0]:
            break

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(net_worths, label="Net Worth")
    plt.plot(stock_prices, label="Stock Price ($)", linestyle="--")
    plt.title("PPO Agent Net Worth Over Time (Gymnasium Compatible)")
    plt.xlabel("Steps")
    plt.ylabel("Net Worth ($)")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Save the figure to disk
    plt.savefig(f"output/ppo_net_worth_{TICKER}_{INTERVAL}_{n_steps}_{total_timesteps}_{TRADING_TYPE}.png", dpi=300)  # You can also use .jpg, .pdf, etc.

    print(f"Saved picture for {TICKER} -> n_steps={n_steps} / total_timesteps={total_timesteps}")

print("All completed")