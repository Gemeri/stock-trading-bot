import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gymnasium import spaces
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from trading_envs import ShortTradingEnv, NormalTradingEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, EvalCallback
import torch

# Ignore future warnings from wrapping Gym environments
# warnings.filterwarnings("ignore", category=DeprecationWarning)

TICKER = "AMZN"
INTERVAL = "H1"
TRADING_TYPE = 'normal'

optionList = [
    #'256-180000',
    #'256-200000',
    #'256-220000',
    #'256-250000',
    '256-500000',
    '256-1000000',
    '256-2000000',
]

FEATURE_COLUMNS = [
    "price_change", "high_low_range", "gap_vs_prev",
    "macd_line", "macd_signal", "macd_histogram", "macd_cross", "macd_hist_flip",
    "rsi", "momentum", "roc", "atr",
    "ema_9", "ema_21", "ema_50", "ema_200",
    "volume_zscore", "atr_zscore", "rsi_zscore",
    "adx", "adx_trend", "obv",
    "candle_body_ratio", "wick_dominance",
    "day_of_week", "days_since_high", "days_since_low"
]
# seed setting
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# max 8 threads
torch.set_num_threads(8)

# Load data
df = pd.read_csv(f"../data/{TICKER}_{INTERVAL}.csv")
df = df.dropna().reset_index(drop=True)


env = DummyVecEnv([lambda: Monitor(NormalTradingEnv(df, FEATURE_COLUMNS))])

for option in optionList:

    n_steps = int(option.split('-')[0])
    total_timesteps = int(option.split('-')[1])

    print(f"Attempting {TICKER} -> n_steps={n_steps} / total_timesteps={total_timesteps}")

    # Train PPO
    model = PPO("MlpPolicy", env, 
                verbose=0,
                n_steps=n_steps,
                batch_size=64,
                policy_kwargs=dict(net_arch={"pi": [128], "vf": [128, 128]}),
                gae_lambda=0.95,            # default value
                gamma=0.95,
                n_epochs=4,
                ent_coef=0.05,              # default 0.005
                learning_rate=1e-4,       # 2.5e-4 is the default value
                clip_range=0.2,             # default value
                max_grad_norm=0.5,
                vf_coef=0.5,
                tensorboard_log="./ppo_logs/",
                )
    
    model.learn(total_timesteps=total_timesteps, tb_log_name=f"PPO_{TICKER}")

    print(f"Completed {TICKER} -> n_steps={n_steps} / total_timesteps={total_timesteps}")

    # Evaluation
    obs = env.reset()
    net_worths = []
    actions = []
    stock_prices = []
    balances = []

    for step in range(int(len(df)/2)):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        
        net_worth = env.get_attr('get_net_worth')[0]()  
        net_worths.append(net_worth)

        last_action = env.get_attr('get_last_action')[0]()
        actions.append(last_action*1000)

        balance = env.get_attr('get_balance')[0]()
        balances.append(balance)

        stock_price = df['close'].iloc[step]
        stock_prices.append(float(stock_price))

        if done[0]:
            break

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 📈 Plot Net Worth (left Y-axis)
    ax1.plot(net_worths, color='blue', label="Net Worth ($)")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Net Worth ($)", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)

    # 📊 Plot action bars on the same axis (behind the net worth)
    bar_colors = ['green' if a > 0 else 'red' if a < 0 else 'gray' for a in actions]
    ax1.bar(range(len(actions)), actions, color=bar_colors, alpha=0.5, label='Trades')

    # 📉 Create a second y-axis for Stock Price (right Y-axis)
    ax2 = ax1.twinx()
    ax2.plot(stock_prices, color='orange', label="Stock Price ($)")
    ax2.set_ylabel("Stock Price ($)", color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # 💰 Balance (third Y-axis on the far right)
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # Move third axis 60 pts outward
    ax3.plot(balances, color='purple', label='Balance ($)', linestyle='--')
    ax3.set_ylabel("Balance ($)", color='purple')
    ax3.tick_params(axis='y', labelcolor='purple')

    # 🏷️ Title and combined legend
    plt.title("PPO Agent: Net Worth, Stock Price, Balance & Trade Actions")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc="upper left")

    
    fig.tight_layout()
    

    # Save the figure to disk
    plt.savefig(f"output/ppo_net_worth_{TICKER}_{INTERVAL}_{n_steps}_{total_timesteps}_{TRADING_TYPE}.png", dpi=300)  # You can also use .jpg, .pdf, etc.

    print(f"Saved picture for {TICKER} -> n_steps={n_steps} / total_timesteps={total_timesteps}")

print("All completed")