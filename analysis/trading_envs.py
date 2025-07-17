import gymnasium as gym
import numpy as np
import math
from gymnasium import spaces

class ShortTradingEnv(gym.Env):
    def __init__(self, df, featureCols:list, buy_sell_rate:float = 0.01, short_rate:float = 0.001, buy_sell_cap=150000):
        super().__init__()
        self.df = df
        self.features = df[featureCols].values
        self.prices = df['close'].values
        self.max_steps = len(self.df) - 1

        self.initial_balance = 10000

        self.buy_sell_rate = buy_sell_rate
        self.buy_sell_cap = buy_sell_cap
        self.short_rate = short_rate

        if self.buy_sell_rate > 1 or self.buy_sell_rate <= 0:
            raise ValueError(f"buy/sell rate cannot be > 1 or <= 0: {self.buy_sell_rate} provided");
    
        self.reset()

        self.action_space = spaces.Discrete(4)  # 0 = Hold, 1 = Buy, 2 = Sell, 3 = Short Sell
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

        # min between the cap and what we can buy X the rate
        shares_to_buy = min(self.buy_sell_cap, math.floor(shares_affordable * self.buy_sell_rate))

        # min between the cap and what we hold X the rate
        shares_to_sell = min(self.buy_sell_cap, math.floor(self.shares_held * self.buy_sell_rate))
        
        # shorting: we buy a % of your net worth
        shares_to_short = min(self.buy_sell_cap, math.floor(self.net_worth * self.short_rate))

        # short covering: we sell 25% of our short position
        shares_to_cover = min(self.buy_sell_cap, math.floor(self.short_position * 0.25))

        # Actions
        if action == 1 and self.balance >= price:  # Buy
            self.shares_held += shares_to_buy
            self.balance -= shares_to_buy * price

        elif action == 2 and self.shares_held > 0:  # Sell
            self.shares_held -= shares_to_sell
            self.balance += shares_to_sell * price

        elif action == 3 and self.balance >= price:  # Short Sell (open short position)
            self.short_position += shares_to_short
            self.balance += shares_to_short * price

        elif action == 4 and self.short_position > 0:  # Cover Short (buy back shares)
            self.short_position -= shares_to_cover
            self.balance -= shares_to_cover * price

        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Net worth calculation
        current_value = self.shares_held * price - self.short_position * price
        self.net_worth = self.balance + current_value
        reward = self.net_worth - self.initial_balance

        return self._get_observation(), reward, done, False, {}

    def get_net_worth(self):
        return self.net_worth
    
    def render(self):
        print(f"Step: {self.current_step} | Net Worth: {self.net_worth:.2f} | Balance: {self.balance:.2f} | Shares: {self.shares_held} | Shorts: {self.short_position}")

class NormalTradingEnv(gym.Env):
    def __init__(self, df, featureList:list, buy_sell_rate:float = 0.2, buy_sell_cap=5):
        super(NormalTradingEnv, self).__init__()
        self.df = df
        self.n_steps = len(df)
        self.current_step = 0
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.position = 0
        self.total_shares = 0
        self.net_worth = self.initial_balance
        self.max_steps = len(df) - 1
        self.buy_sell_rate = buy_sell_rate
        self.buy_sell_cap = buy_sell_cap
        
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
        obs = self._next_observation()
        info = {}
        return obs, info

    def _next_observation(self):
        return self.features[self.current_step].astype(np.float32)

    def step(self, action):
        price = self.df.loc[self.current_step, 'close']

        if action == 1 and self.balance >= price:

            # max shares to buy
            sharesToBuy = math.floor(self.balance/price)

            # we apply a percentage + cap
            sharesToBuy = min(math.floor(self.buy_sell_rate*sharesToBuy), self.buy_sell_cap)

            self.total_shares += sharesToBuy
            self.balance -= price * sharesToBuy
        elif action == 2 and self.total_shares > 0:

            # we sell only a % of the shares we own + cap
            sharesToSell = min(math.floor(self.buy_sell_rate*self.total_shares), self.buy_sell_cap)

            self.balance += price * sharesToSell
            self.total_shares -= sharesToSell

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
