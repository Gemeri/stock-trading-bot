import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from utils import load_and_engineer_features


# Load training data
df = load_and_engineer_features('../../data/AMZN_H1.csv')

LOOKAHEAD_LIST = [1, 2, 3, 5]

# Shift 'close' to predict the next candle's close

for lookahead in LOOKAHEAD_LIST:
    df[f"target_{lookahead}"] = df['close'].shift(-lookahead)

df.dropna(inplace=True)  # Remove last row, which now has a NaN target

# Portfolio and tracking
initial_cash = 100000
cash = initial_cash
shares = 0
portfolio_values = []
last_predicted_price = 0

# for plotting
timestamps = []
actions = []
stock_prices = []
balances = []

# Backtest from candle end-1200 to now
for i in range(len(df)-1200, len(df)-1):

    print(f"Loading candles from {i-800} to {i}")

    # Train model on previous 6 months (approx. 400 candles)
    train_data = df.iloc[i-800:i]
    test_data = df.iloc[i:i+1]
    
    # Use simple features (you can expand)
    features = [
        'returns_1', 'returns_3', 'returns_5',
        'ma_3', 'ma_5', 'ma_10',
        'std_5', 'std_10',
        'high_low_range', 'open_close_diff',
        'volume_change'
    ]


    predicted_prices = []

    for lookahead in LOOKAHEAD_LIST:
        
        X_train = train_data[features]
        y_train = train_data[f"target_{lookahead}"]

        X_test = test_data[features]
        
        model = xgb.XGBRegressor(
            n_estimators=100, 
            max_depth=5
        )

        model.fit(X_train, y_train)
        
        # we append to the list
        predicted_prices.append(model.predict(X_test)[0])

    print("models trained")
    
    current_price = test_data['close'].values[0]

    max_predicted = max(predicted_prices)
    min_predicted = min(predicted_prices)

    print(f"min price: {min_predicted} - max price {max_predicted} - current price {current_price}")

    # we chose the price with the highest swing
    chosen_price = max_predicted if abs(current_price - max_predicted) > abs(current_price - min_predicted) else min_predicted
    
    projected_change = (chosen_price - current_price) / current_price

    print(f"projected change: {projected_change}")

    last_action = 0

    # Strategy
    if projected_change > 0.01:
        # Go long: invest 30% of current cash
        to_invest = 0.3 * cash
        num_shares = to_invest // current_price
        if num_shares > 0:
            cash -= num_shares * current_price
            shares += num_shares
            last_action = +num_shares
    elif projected_change < -0.01:
        # Go short (sell all)
        cash += shares * current_price
        shares = 0
        last_action = -shares
    
    # Track portfolio value
    portfolio_value = cash + shares * current_price
    
    # we record info for plotting
    actions.append(last_action*1000)
    stock_prices.append(current_price)
    portfolio_values.append(portfolio_value)
    balances.append(cash)
    timestamps.append(test_data['timestamp'].values[0])

    print(f"portfolio updated: {portfolio_value}")


print("plotting")

# ⏱ Prepare timestamp index
timestamps = pd.to_datetime(df['timestamp'].iloc[:len(net_worths)])  # Adjust column name if needed

# 📈 Create plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# dates formatting
ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
fig.autofmt_xdate()

# 📉 Plot Net Worth (left Y-axis)
ax1.plot(timestamps, net_worths, color='blue', label="Net Worth ($)")
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
plt.title("PPO Agent: Net Worth, Stock Price, Balance & Trade Actions")
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
ax1.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc="upper left")

# Save figure
fig.tight_layout()    

# Step 6: Plot results
plt.figure(figsize=(12, 6))
plt.plot(timestamps, portfolio_values, label='Portfolio Value')
plt.title('Backtest Portfolio Value over Time')
plt.xlabel('Candle Index')
plt.ylabel('Portfolio Value ($)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


