import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import spaces
import gym
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor

# Load environment variables from .env
load_dotenv()
BAR_TIMEFRAME = os.getenv('BAR_TIMEFRAME')
TICKERS = os.getenv('TICKERS').split(',')
DISABLED_FEATURES = os.getenv('DISABLED_FEATURES').split(',')

# Convert BAR_TIMEFRAME to CSV filename suffix
timeframe_map = {
    "4Hour": "H4",
    "2Hour": "H2",
    "1Hour": "H1",
    "30Min": "M30",
    "15Min": "M15"
}
converted_timeframe = timeframe_map.get(BAR_TIMEFRAME, "H1")  # Default to "H1" if not found

# Define all possible CSV features (all numerical)
all_features = [
    'open', 'high', 'low', 'close', 'volume', 'vwap', 'price_change', 'high_low_range',
    'log_volume', 'sentiment', 'price_return', 'candle_rise', 'body_size', 'wick_to_body',
    'macd_line', 'rsi', 'momentum', 'roc', 'atr', 'hist_vol', 'obv', 'volume_change',
    'stoch_k', 'bollinger_upper', 'bollinger_lower', 'lagged_close_1', 'lagged_close_2',
    'lagged_close_3', 'lagged_close_5', 'lagged_close_10'
]
enabled_features = [f for f in all_features if f not in DISABLED_FEATURES]

# Custom Trading Environment for RL
class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        # Ensure no NaNs in features or close
        self.df = df.dropna(subset=enabled_features + ['close'])
        self.current_step = 0
        self.max_steps = len(self.df) - 1
        self.initial_cash = 10000
        self.cash = self.initial_cash
        self.position_qty = 0
        self.portfolio_value = self.initial_cash
        self.trade_size = 100
        self.prev_price = self.df.iloc[0]['close']

        # Train Random Forest model for price prediction
        rf_features = [col for col in self.df.columns if col not in ['timestamp', 'close']]
        X = self.df[rf_features].iloc[:-1]
        y = self.df['close'].iloc[1:]
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(X, y)

        # Define state and action spaces
        self.state_size = len(enabled_features) + 2  # enabled_features + predicted_price + position_qty
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)  # 0=BUY, 1=SELL, 2=SHORT, 3=COVER, 4=NONE

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.position_qty = 0
        self.portfolio_value = self.initial_cash
        self.prev_price = self.df.iloc[0]['close']
        return self._get_state()

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        prev_portfolio_value = self.cash + self.position_qty * current_price

        # Execute action based on current position
        if action == 0 and self.position_qty < 1:  # BUY
            cost = self.trade_size * current_price
            if self.cash >= cost:
                self.cash -= cost
                self.position_qty = self.trade_size
        elif action == 1 and self.position_qty > 0:  # SELL
            proceeds = self.position_qty * current_price
            self.cash += proceeds
            self.position_qty = 0
        elif action == 2 and self.position_qty > -1:  # SHORT
            proceeds = self.trade_size * current_price
            self.cash += proceeds
            self.position_qty = -self.trade_size
        elif action == 3 and self.position_qty < 0:  # COVER
            cost = abs(self.position_qty) * current_price
            self.cash -= cost
            self.position_qty = 0
        # Action 4 is NONE, do nothing

        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        if not done:
            next_price = self.df.iloc[self.current_step]['close']
            self.portfolio_value = self.cash + self.position_qty * next_price
            reward = self.portfolio_value - prev_portfolio_value
            # Penalize NONE action and encourage trading
            if action == 4:
                reward -= 0.1  # Adjusted penalty to balance with typical reward scale
            # Incorporate stop-loss/take-profit logic
            price_change = (next_price - self.prev_price) / self.prev_price
            if self.position_qty > 0 and price_change <= -0.05:  # 5% stop-loss for long
                reward -= 50  # Penalty for hitting stop-loss
            elif self.position_qty < 0 and price_change >= 0.05:  # 5% stop-loss for short
                reward -= 50
            self.prev_price = next_price
        else:
            reward = 0

        state = self._get_state() if not done else np.zeros(self.state_size, dtype=np.float32)
        return state, reward, done, {}

    def _get_state(self):
        row = self.df.iloc[self.current_step]
        # Ensure features are numerical
        features = row[enabled_features].astype(float).values
        pred_input = row[[col for col in self.df.columns if col not in ['timestamp', 'close']]].astype(float).values.reshape(1, -1)
        predicted_price = self.rf_model.predict(pred_input)[0]
        state = np.append(features, [predicted_price, self.position_qty])
        # Explicitly convert state to float32 to ensure compatibility with PyTorch
        return np.array(state, dtype=np.float32)

# Initialize and train the PPO model
first_ticker = TICKERS[0]
csv_file = f"{first_ticker}_{converted_timeframe}.csv"
df = pd.read_csv(csv_file)
env = DummyVecEnv([lambda: TradingEnv(df)])
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    gamma=0.99,
    n_steps=2048,
    verbose=1
)
model.learn(total_timesteps=10000)
model.save("ppo_trading_model")
model = PPO.load("ppo_trading_model")

# Global CSV data for backtesting
csv_data = None

def run_logic(current_price, predicted_price, ticker):
    """
    Execute live trading decisions based on RL model.
    
    Args:
        current_price (float): Current market price.
        predicted_price (float): Predicted next price from RF model.
        ticker (str): Stock ticker symbol.
    """
    from forest import api, buy_shares, sell_shares, short_shares, close_short
    csv_file = f"{ticker}_{converted_timeframe}.csv"
    df = pd.read_csv(csv_file)
    # Drop rows with NaNs in enabled features to ensure clean data
    df = df.dropna(subset=enabled_features)
    last_row = df.iloc[-1]
    # Ensure features are numerical
    features = last_row[enabled_features].astype(float).values
    pos = api.get_position(ticker)
    position_qty = float(pos.qty)
    state = np.append(features, [predicted_price, position_qty])
    # Explicitly convert state to float32 to ensure compatibility with PyTorch
    state = np.array(state, dtype=np.float32)
    
    action, _ = model.predict(state, deterministic=True)
    action_str = ["BUY", "SELL", "SHORT", "COVER", "NONE"][action]
    
    # Execute trade with no duplicates
    if action_str == "BUY" and position_qty < 1:
        buy_shares(ticker, 100)
    elif action_str == "SELL" and position_qty > 0:
        sell_shares(ticker, position_qty)
    elif action_str == "SHORT" and position_qty > -1:
        short_shares(ticker, 100)
    elif action_str == "COVER" and position_qty < 0:
        close_short(ticker, abs(position_qty))
    # NONE action is implicitly handled by doing nothing

def run_backtest(current_price, predicted_price, position_qty):
    """
    Perform backtesting by returning trading decisions based on RL model.
    
    Args:
        current_price (float): Current price from historical data.
        predicted_price (float): Predicted next price from RF model.
        position_qty (float): Current position quantity.
    
    Returns:
        str: Trading action ("BUY", "SELL", "SHORT", "COVER", "NONE").
    """
    global csv_data
    if csv_data is None:
        first_ticker = TICKERS[0]
        csv_file = f"{first_ticker}_{converted_timeframe}.csv"
        csv_data = pd.read_csv(csv_file)
        # Drop rows with NaNs in enabled features to ensure clean data
        csv_data = csv_data.dropna(subset=enabled_features)
    
    # Find row matching current_price (assuming exact match for simplicity)
    matching_rows = csv_data[csv_data['close'] == current_price]
    if matching_rows.empty:
        return "NONE"  # Default to NONE if no matching row
    row = matching_rows.iloc[-1]
    # Ensure features are numerical
    features = row[enabled_features].astype(float).values
    state = np.append(features, [predicted_price, position_qty])
    # Explicitly convert state to float32 to ensure compatibility with PyTorch
    state = np.array(state, dtype=np.float32)
    
    action, _ = model.predict(state, deterministic=True)
    action_str = ["BUY", "SELL", "SHORT", "COVER", "NONE"][action]
    
    # Apply no-duplicate-trade logic
    if action_str == "BUY" and position_qty >= 1:
        return "NONE"
    elif action_str == "SELL" and position_qty <= 0:
        return "NONE"
    elif action_str == "SHORT" and position_qty <= -1:
        return "NONE"
    elif action_str == "COVER" and position_qty >= 0:
        return "NONE"
    return action_str

# Online learning update function (called externally or periodically)
def update_model(new_data_df):
    global model
    env = DummyVecEnv([lambda: TradingEnv(new_data_df)])
    model.set_env(env)
    model.learn(total_timesteps=1000, reset_num_timesteps=False)
    model.save("ppo_trading_model")
    model = PPO.load("ppo_trading_model")