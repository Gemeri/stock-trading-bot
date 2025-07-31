import pandas as pd
import numpy as np
import xgboost as xgb
import platform
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from linear import stock_price_direction

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils import load_and_engineer_features

STOCK_TICKER = "AAPL"
LOOKAHEAD_LIST = [3, 5, 8]
ENABLE_MODERATE = False
ENABLE_INTRADAY = False
WEAK_STRONG_THRESHOLD = 20
SHORT_RATE = 0.2
BUY_RATE = 0.2

# Load training data
df = load_and_engineer_features(f"../fetch/data/{STOCK_TICKER}_H1.csv")

# ----------------------------
# 2. Feature Engineering on Timestamp
# ----------------------------
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['month'] = df['timestamp'].dt.month
df['quarter'] = df['timestamp'].dt.quarter

# Optional: Cyclical encoding for hour and day_of_week
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Drop original timestamp and non-numeric columns if any
# df = df.drop(['hour', 'day_of_week', 'month', 'quarter'], axis=1)

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
position_values = []

# Backtest from candle end-1200 to now -> 18mo circa
for i in range(len(df)-800, len(df)-1):

    print(f"Loading candles for {STOCK_TICKER} from {i-800} to {i}")

    # Train model on previous 12 months (approx. 800 candles)
    train_data = df.iloc[i-800:i]
    test_data = df.iloc[i:i+1]
    
    # Use simple features (you can expand)
    feature_columns = [
        'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions',
        'price_change', 'high_low_range', 'log_volume', 'returns_1',
        'returns_3', 'returns_5', 'ma_3', 'ma_5', 'ma_10', 'std_5', 'std_10',
        'open_close_diff', 'volume_change', 'macd_line', 'macd_signal',
        'macd_histogram', 'rsi', 'momentum', 'roc', 'atr', 'ema_9', 'ema_21',
        'ema_50', 'ema_200', 'adx', 'obv', 'bollinger_upper', 'bollinger_lower',
        'lagged_close_1', 'lagged_close_2', 'lagged_close_3', 'lagged_close_5',
        'lagged_close_10', 'candle_body_ratio', 'wick_dominance', 'gap_vs_prev',
        'volume_zscore', 'atr_zscore', 'rsi_zscore', 'adx_trend', 'macd_cross',
        'macd_hist_flip', 'days_since_high', 'days_since_low',
        # Time-based engineered features
        'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'is_weekend'
        ]

    # Ensure all feature columns exist
    feature_columns = [col for col in feature_columns if col in df.columns]

    predicted_prices = []

    for lookahead in LOOKAHEAD_LIST:
        
        X_train = train_data[feature_columns]
        y_train = train_data[f"target_{lookahead}"]

        X_test = test_data[feature_columns]
        
        model = xgb.XGBRegressor(
            n_estimators=100, 
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
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
    if not ENABLE_INTRADAY and date == last_trade_date:
        print(f"Skipping trade because last_trade_date = {last_trade_date}")
        portfolio_value = cash + position * current_price
        continue

    # moderate LONG ENTRY
    if direction > 0 and direction < WEAK_STRONG_THRESHOLD and ENABLE_MODERATE:

        # exit half our shorts first
        if position < 0:
            shorts_to_close = abs(position // 2)
            cash -= shorts_to_close * current_price
            position += shorts_to_close
            last_trade_date = date

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
    elif direction > WEAK_STRONG_THRESHOLD:

        # exit ALL short first
        if position < 0:
            cash -= abs(position) * current_price
            position = 0
            last_trade_date = date

        # buy shares more aggressively if we can
        to_invest = BUY_RATE * cash
        num_shares = to_invest // current_price
        if num_shares > 0:

            print(f"** BOUGHT {num_shares} at {current_price} = {num_shares * current_price}")
            cash -= num_shares * current_price
            position += num_shares
            entry_price = current_price
            last_trade_date = date

    # moderate short
    elif direction < 0 and direction > -WEAK_STRONG_THRESHOLD and ENABLE_MODERATE:
        
        # we sell 50% of our shares
        if position > 0:

            to_sell = position // 2 
            
            print(f"** SOLD {to_sell} at {current_price} = {to_sell * current_price}")
            cash += to_sell * current_price
            position -= to_sell
            last_trade_date = date

    
    # strong short
    elif direction < -WEAK_STRONG_THRESHOLD:
        
        # we sell 100% of our shares
        if position > 0:

            to_sell = position

            print(f"** SOLD {to_sell} at {current_price} = {to_sell * current_price}")
            cash += to_sell * current_price
            position -= to_sell
            last_trade_date = date

        # then we go short (SHORT_RATE% of cash)
        to_short = SHORT_RATE * cash
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
    position_values.append(position)
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
ax1.plot(timestamps, portfolio_values, color='blue', label="Net Worth ($)")
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

# 💰 Plot Cacsh (third Y-axis)
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))  # Offset third axis
ax3.plot(timestamps, balances, color='red', label='Cash ($)', linestyle=':')
ax3.set_ylabel("Cash ($)", color='red')
ax3.tick_params(axis='y', labelcolor='red')

# 🏷️ Format X-axis as dates
import matplotlib.dates as mdates
ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
fig.autofmt_xdate()

# 🧾 Title and combined legend
plt.title(f"XGboost prediction {STOCK_TICKER}: Net Worth, Stock Price, Balance & Trade Actions")
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
ax1.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc="upper left")

# Save figure
fig.tight_layout()    

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

plt.savefig(f"pred/backtest-graph-{STOCK_TICKER}-{timestamp}.png")

plot_df = pd.DataFrame({
    'timestamp': timestamps,
    'cash': balances,
    'portfolio_value': portfolio_values,
    'position': position_values,
    'stock_price': stock_prices,
    'action': actions
})

# Save as CSV
plot_df.to_csv(f"pred/backtest-data-{STOCK_TICKER}-{timestamp}.csv", index=True)

# Only use TkAgg if NOT on macOS
if platform.system() == "Darwin":
    plt.show()


