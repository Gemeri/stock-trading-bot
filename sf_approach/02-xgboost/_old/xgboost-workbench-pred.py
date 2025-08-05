import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
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
    'volume_change', 'close'
]

# Shift 'close' to predict the next candle's close
df['target'] = df['close'].shift(-1)
df.dropna(inplace=True)  # Remove last row, which now has a NaN target

# Train-test split
X = df[features]
y = df['target']

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

# === Make predictions ===
y_pred = model.predict(X)

rmse = root_mean_squared_error(y, y_pred)
print(f"Test RMSE: {rmse:.4f}")

# === Calculate percentage difference ===
percentage_diff = 100 * (y_pred - y.values) / y.values

# === Create DataFrame for export ===
results_df = pd.DataFrame({
    'timestamp': df['timestamp'].values,
    'predictedClose': y_pred,
    'realClose': y.values,
    'percentageDifference': percentage_diff
})


# Save to CSV
results_df.to_csv('predicted_vs_real_test.csv', index=False)
print("✅ CSV saved: predicted_vs_real_test.csv")

# === Plot predicted vs. real prices ===
plt.figure(figsize=(14, 6))
plt.plot(df['timestamp'].values, y.values, label='Actual Price', color='black', linewidth=2)
plt.plot(df['timestamp'].values, y_pred, label='Predicted Price', color='blue', linestyle='--')
plt.title("Predicted vs Actual Close Prices (Test Set)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()