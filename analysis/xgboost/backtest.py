import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from utils import load_and_engineer_features


# Load training data
df = load_and_engineer_features('../../data/AMZN_H1.csv')

# Features selected for modeling
features = [
    'returns_1', 'returns_3', 'returns_5',
    'ma_3', 'ma_5', 'ma_10',
    'std_5', 'std_10',
    'high_low_range', 'open_close_diff',
    'volume_change'
]

# Train-test split
X = df[features]
y = df['close']

X_train, X_test, y_train, y_test, time_train, time_test = train_test_split(
    X, y, df['timestamp'], test_size=0.2, shuffle=False
)

# Train XGBoost regressor
model = xgb.XGBRegressor(
    n_estimators=114, 
    max_depth=9, 
    learning_rate=0.14264252588219034,
    subsample=0.5524803023252148,
    colsample_bytree=0.7687841723045249,
    gamma=0.5856035407199236,
    reg_alpha=0.5063880221467401,
    reg_lambda=0.0728996118523866,
)
model.fit(X_train, y_train)

# now we run a backtest

# Shift signal to simulate entering at today's close, exiting tomorrow's close
df['next_price'] = df['actual_price'].shift(-1)

# Realized return from position
df['realized_return'] = (df['next_price'] - df['actual_price']) / df['actual_price']
df['strategy_return'] = df['signal'] * df['realized_return']

# Drop rows where we can't get next day's price
df = df.dropna(subset=['strategy_return'])

# Aggregate by date (equal-weight average return per day)
daily_returns = df.groupby('date')['strategy_return'].mean()

# Calculate cumulative return
cumulative_returns = (1 + daily_returns).cumprod()

return daily_returns, cumulative_returns, df

# === Simulate predictions with noise ===
# In real case, replace this with your model's predicted prices
df['predicted_price'] = df['price'].shift(-1) * (1 + np.random.uniform(-PREDICTION_ERROR, PREDICTION_ERROR, size=n))




# Plot actual vs predicted
plt.figure(figsize=(14, 6))
plt.plot(backtest_df['timestamp'], backtest_df['actual_close'], label='Actual Close', color='blue')
plt.plot(backtest_df['timestamp'], backtest_df['predicted_close'], label='Predicted Close', color='orange')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.title('Predicted vs Actual Close Prices')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()