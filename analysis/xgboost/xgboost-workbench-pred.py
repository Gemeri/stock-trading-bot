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

# now we check on another stock

# load the test data
dt = load_and_engineer_features('../../data/AAPL_H1.csv')

X = df[features]
y = df['close']

# Predict on test
y_pred = model.predict(X)

# Backtest results
backtest_df = pd.DataFrame({
    'timestamp': df['timestamp'],
    'predicted_close': y_pred,
    'actual_close': y.values
})
backtest_df['pct_diff'] = 100 * (backtest_df['actual_close'] - backtest_df['predicted_close']) / backtest_df['actual_close']

# Save to CSV
# backtest_df.to_csv('pred/backtest_results.csv', index=False)

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