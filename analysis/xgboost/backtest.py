import pandas as pd
import numpy as np
import xgboost as xgb
import platform
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from linear import stock_price_direction

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils import load_and_engineer_features


# Load training data
df = load_and_engineer_features('../../data/AMZN_H1.csv')

# some time engineering
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Feature Engineering on Timestamp
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek  # Monday=0, Sunday=6
df['day_of_month'] = df.index.day
df['month'] = df.index.month
df['quarter'] = df.index.quarter
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_month_start'] = (df.index.is_month_start).astype(int)
df['is_month_end'] = (df.index.is_month_end).astype(int)

# Create cyclical features for hour and day of week
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Price-based features
df['price_change_pct'] = df['price_change'] / df['open']
df['high_low_range_pct'] = df['high_low_range'] / df['open']
df['body_size'] = abs(df['close'] - df['open'])
df['body_to_range_ratio'] = df['body_size'] / df['high_low_range']

# Volume-based features
df['volume_change_pct'] = df['volume_change'] / df['volume'].shift(1).fillna(1)
df['volume_zscore'] = (df['volume'] - df['volume'].mean()) / df['volume'].std()
df['vwap_diff'] = df['vwap'] - df['open']
df['volume_price_interaction'] = df['volume'] * df['close']

# Technical indicators
# Moving averages
df['ma_3'] = df['close'].rolling(window=3).mean()
df['ma_5'] = df['close'].rolling(window=5).mean()
df['ma_10'] = df['close'].rolling(window=10).mean()

# RSI smoothing
df['rsi_smoothed'] = df['rsi'].rolling(window=3).mean()

# MACD features
df['macd_signal'] = df['macd_signal'].fillna(method='bfill')
df['macd_hist_smoothed'] = df['macd_histogram'].rolling(window=3).mean()

# ATR features
df['atr_smoothed'] = df['atr'].rolling(window=3).mean()

# Bollinger Bands
df['bb_mid'] = df['ma_20']  # Using ma_20 from the data
df['bb_upper'] = df['bollinger_upper']
df['bb_lower'] = df['bollinger_lower']
df['bb_width'] = df['bb_upper'] - df['bb_lower']
df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

# Momentum and ROC
df['momentum_1'] = df['close'].diff(1)
df['roc_1'] = df['close'].pct_change(1)

# ADX trend strength
df['adx_trend_strength'] = df['adx'] / 100

# Lagged features
for lag in [1, 2, 3, 5, 10]:
    df[f'close_lag_{lag}'] = df['close'].shift(lag)
    df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
    df[f'returns_1_lag_{lag}'] = df['returns_1'].shift(lag)

LOOKAHEAD_LIST = [3, 5, 8]

# Shift 'close' to predict the next candle's close

for lookahead in LOOKAHEAD_LIST:
    df[f"target_{lookahead}"] = df['close'].shift(-lookahead)

df.dropna(inplace=True)  # Remove last row, which now has a NaN target

# Portfolio and tracking
initial_cash = 100000
cash = initial_cash
position = 0
portfolio_values = []
last_predicted_price = 0
last_trade_date = False

# for plotting
timestamps = []
actions = []
stock_prices = []
balances = []

# Backtest from candle end-1200 to now -> 18mo circa
for i in range(len(df)-800, len(df)-1):

    print(f"Loading candles from {i-800} to {i}")

    # Train model on previous 12 months (approx. 800 candles)
    train_data = df.iloc[i-800:i]
    test_data = df.iloc[i:i+1]
    
    # Use simple features (you can expand)
    feature_columns = [
        'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions',
        'price_change', 'high_low_range', 'log_volume', 'returns_1', 'returns_3', 
        'returns_5', 'ma_3', 'ma_5', 'ma_10', 'std_5', 'std_10', 'open_close_diff',
        'volume_change', 'macd_line', 'macd_signal', 'macd_histogram', 'rsi',
        'momentum', 'roc', 'atr', 'ema_9', 'ema_21', 'ema_50', 'ema_200', 'adx',
        'obv', 'bollinger_upper', 'bollinger_lower', 'hour', 'day_of_week',
        'day_of_month', 'month', 'quarter', 'is_weekend', 'is_month_start',
        'is_month_end', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'price_change_pct', 'high_low_range_pct', 'body_size', 'body_to_range_ratio',
        'volume_change_pct', 'volume_zscore', 'vwap_diff', 'volume_price_interaction',
        'ma_3', 'ma_5', 'ma_10', 'rsi_smoothed', 'macd_hist_smoothed', 'atr_smoothed',
        'bb_width', 'bb_position', 'momentum_1', 'roc_1', 'adx_trend_strength'
        ]

    # Add lagged features to feature list
    for lag in [1, 2, 3, 5, 10]:
        feature_columns.extend([f'close_lag_{lag}', f'volume_lag_{lag}', f'returns_1_lag_{lag}'])

    predicted_prices = []

    for lookahead in LOOKAHEAD_LIST:
        
        X_train = train_data[feature_columns]
        y_train = train_data[f"target_{lookahead}"]

        X_test = test_data[feature_columns]
        
        model = xgb.XGBRegressor(
            n_estimators=100, 
            max_depth=5
        )

        model.fit(X_train, y_train)
        
        # we append to the list
        predicted_prices.append(model.predict(X_test)[0])

    print("models trained")
    
    current_price = test_data['close'].values[0]

    direction = stock_price_direction([0] + LOOKAHEAD_LIST, [current_price] + predicted_prices)

    # direction is between +90 and - 90
    # let's consider 0 to 30 and 30+
    # also 0 to -30 and less then -30
    print(f"projected direction: {direction:.2f}")

    last_action = 0
    
    # extract the date
    date = pd.to_datetime(test_data["timestamp"].values[0]).date()

    # Skip if a trade was already made today (simulate T+1 rule)
    # if date == last_trade_date:
    #     print(f"Skipping trade because last_trade_date = {last_trade_date}")
    #     portfolio_value = cash + position * current_price
    #     continue

    # moderate LONG ENTRY
    if direction > 0 and direction < 30:

        # exit half our shorts first
        if position < 0:
            shorts_to_close = abs(position // 2)
            cash -= shorts_to_close * current_price
            position += shorts_to_close

        # buy shares moderately if we can
        to_invest = 0.1 * cash
        num_shares = to_invest // current_price
        if num_shares > 0:

            print(f"** BOUGHT {num_shares} at {current_price} = {num_shares * current_price}")
            cash -= num_shares * current_price
            position += num_shares
            entry_price = current_price
            last_trade_date = date

    # strong LONG entry
    elif direction > 30:

        # exit ALL short first
        if position < 0:
            cash -= abs(position) * current_price
            position = 0

        # buy shares more aggressively if we can
        to_invest = 0.2 * cash
        num_shares = to_invest // current_price
        if num_shares > 0:

            print(f"** BOUGHT {num_shares} at {current_price} = {num_shares * current_price}")
            cash -= num_shares * current_price
            position += num_shares
            entry_price = current_price
            last_trade_date = date

    # moderate short
    elif direction < 0 and direction > -30:
        
        # we sell 50% of our shares
        if position > 0:

            to_sell = position // 2 
            
            print(f"** SOLD {to_sell} at {current_price} = {to_sell * current_price}")
            cash += to_sell * current_price
            position -= to_sell
            last_trade_date = date

    
    # strong short
    elif direction < -30:
        
        # we sell 100% of our shares
        if position > 0:

            to_sell = position

            print(f"** SOLD {to_sell} at {current_price} = {to_sell * current_price}")
            cash += to_sell * current_price
            position -= to_sell
            last_trade_date = date

        # then we go short (20% of cash)
        to_short = 0.2 * cash
        num_shorts = to_short // current_price
        if num_shorts > 0:
            cash += num_shorts * current_price
            position -= num_shorts  # Negative value = short position
            entry_price = current_price
            last_trade_date = date

    # Track portfolio value
    portfolio_value = cash + position * current_price

    print(f"-> current position: {position} / portfolio_value {portfolio_value}")
    
    # we record info for plotting
    actions.append(last_action*1000)
    stock_prices.append(current_price)
    portfolio_values.append(portfolio_value)
    balances.append(cash)
    timestamps.append(test_data['timestamp'].values[0])

    # we stop if we made a trade
    # if last_trade_date == date:
    #     input()


print("plotting")

# ⏱ Prepare timestamp index
timestamps = pd.to_datetime(df['timestamp'].iloc[:len(balances)])  # Adjust column name if needed

# 📈 Create plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# dates formatting
ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
fig.autofmt_xdate()

# 📉 Plot Net Worth (left Y-axis)
ax1.plot(timestamps, balances, color='blue', label="Net Worth ($)")
ax1.set_xlabel("Time")
ax1.set_ylabel("Net Worth ($)", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True)

# 📊 Plot action bars (on same axis)
bar_colors = ['green' if a > 0 else 'red' if a < 0 else 'gray' for a in actions]
ax1.bar(timestamps, actions, color=bar_colors, alpha=0.5, label='Trades')

# 📈 Plot Stock Price (right Y-axis)
ax2 = ax1.twinx()
ax2.plot(timestamps, stock_prices, color='orange', label="Stock Price ($)")
ax2.set_ylabel("Stock Price ($)", color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# 💰 Plot Balance (third Y-axis)
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))  # Offset third axis
ax3.plot(timestamps, balances, color='purple', label='Balance ($)', linestyle='--')
ax3.set_ylabel("Balance ($)", color='purple')
ax3.tick_params(axis='y', labelcolor='purple')

# 🏷️ Format X-axis as dates
import matplotlib.dates as mdates
ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
fig.autofmt_xdate()

# 🧾 Title and combined legend
plt.title("XGboost prediction: Net Worth, Stock Price, Balance & Trade Actions")
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
ax1.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc="upper left")

# Save figure
fig.tight_layout()    


# Only use TkAgg if NOT on macOS
if platform.system() != "Darwin":
    plt.savefig("backtest-plot.png")
else:
    plt.show()


