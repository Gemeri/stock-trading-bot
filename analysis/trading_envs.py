import gymnasium as gym
import numpy as np
import math
from gymnasium import spaces

class ShortTradingEnv(gym.Env):
    def __init__(self, df, featureCols:list, buy_sell_rate:float = 0.01, short_rate:float = 0.01):
        super().__init__()
        self.df = df
        self.features = df[featureCols].values
        self.prices = df['close'].values
        self.max_steps = len(self.df) - 1

        self.initial_balance = 10000

        self.buy_sell_rate = buy_sell_rate
        self.short_rate = short_rate

        if self.buy_sell_rate > 1 or self.buy_sell_rate <= 0:
            raise ValueError(f"buy/sell rate cannot be > 1 or <= 0: {self.buy_sell_rate} provided");
    
        self.reset()

        self.action_space = spaces.Discrete(5)  # 0 = Hold, 1 = Buy, 2 = Sell, 3 = Short Sell, 4 = Short Cover
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(featureCols),), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.shares_held = 0
        self.short_position = 0
        self.current_step = 0
        return self._get_observation(), {}

    def _get_observation(self):
        return self.features[self.current_step].astype(np.float32)

    def step(self, action):
        price = self.prices[self.current_step]
        reward = 0

        # Calculate how much we can buy/sell
        shares_affordable = math.floor(self.balance / price)

        shares_to_buy = max(1, math.floor(shares_affordable * self.buy_sell_rate))
        shares_to_sell = max(1, math.floor(self.shares_held * self.buy_sell_rate))
        shares_to_short = max(1, math.floor(shares_affordable * self.short_rate))
        shares_to_cover = max(1, math.floor(self.short_position * 0.25))

        # Actions
        if action == 1 and self.balance >= price:  # Buy
            self.shares_held += shares_to_buy
            self.balance -= shares_to_buy * price

        elif action == 2 and self.shares_held > 0:  # Sell
            self.shares_held -= shares_to_sell
            self.balance += shares_to_sell * price

        elif action == 3:  # Short Sell (open short position)
            self.short_position += shares_to_short
            self.balance += shares_to_short * price

        elif action == 4 and self.short_position > 0:  # Cover Short (buy back shares)
            self.short_position -= shares_to_cover
            self.balance -= shares_to_cover * price

        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.max_steps

        prev_worth = self.net_worth

        # Net worth = cash + value of long positions - cost to cover shorts
        long_value = self.shares_held * price
        short_liability = self.short_position * price
        self.net_worth = self.balance + long_value - short_liability

        # reward as delta between
        reward = self.net_worth - prev_worth

        return self._get_observation(), reward, done, False, {}

    def get_net_worth(self):
        return self.net_worth
    
    def render(self):
        print(f"Step: {self.current_step} | Net Worth: {self.net_worth:.2f} | Balance: {self.balance:.2f} | Shares: {self.shares_held} | Shorts: {self.short_position}")

class NormalTradingEnv(gym.Env):
    def __init__(self, df, featureList:list, buy_sell_rate:float = 0.1):
        super(NormalTradingEnv, self).__init__()
        self.df = df
        self.n_steps = len(df)
        self.current_step = 0
        self.initial_balance = 100000
        self.balance = self.initial_balance
        self.position = 0
        self.total_shares = 0
        self.net_worth = self.initial_balance
        self.max_steps = len(df) - 1
        self.buy_sell_rate = buy_sell_rate
        self.last_action = 0 # nothing
        self.returns_window = []
        
        if self.buy_sell_rate > 1 or self.buy_sell_rate <= 0:
            raise ValueError(f"buy/sell rate cannot be > 1 or <= 0: {self.buy_sell_rate} provided");
    
        self.features = df[featureList].values

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
        self.last_action = 0
        self.returns_window = []
        
        obs = self._next_observation()
        info = {}
        return obs, info

    def _next_observation(self):
        return self.features[self.current_step].astype(np.float32)

    def step(self, action):
        price = self.df.loc[self.current_step, 'close']

        shares_traded = 0

        if action == 1 and self.balance >= price:

            # max shares to buy
            shares_affordable = math.floor(self.balance/price)

            # we apply a percentage 
            shares_traded = math.floor(self.buy_sell_rate*shares_affordable)

            self.total_shares += shares_traded
            self.balance -= price * shares_traded

            # we save for plotting
            self.last_action = shares_traded

        elif action == 2 and self.total_shares > 0:

            # we sell only a % of the shares we own + cap
            shares_traded = math.floor(self.buy_sell_rate*self.total_shares)

            self.balance += price * shares_traded
            self.total_shares -= shares_traded

            # we save for plotting
            self.last_action = -shares_traded

        else:
            # just hold
            self.last_action = 0

        prev_worth = self.net_worth

        self.net_worth = self.balance + self.total_shares * price

        self.returns_window.append(self.net_worth / prev_worth)
        reward = np.mean(self.returns_window[-5:])

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
    
    def get_last_action(self):
        return self.last_action
    
    def get_balance(self):
        return self.balance
