import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("../data/ORCL_H4.csv")

# Drop rows with NaNs (you can improve this with imputation)
df = df.dropna()

# Define feature columns (take all to start)
feature_cols = df.columns.tolist()
feature_cols.remove('timestamp')
if 'predicted_close' in feature_cols:
    feature_cols.remove('predicted_close')

print(f"Using cols: {feature_cols}")

# Define the label: future return (shifted close-return)
df['target'] = df['close'].shift(-1)
df = df.dropna()

X = df[feature_cols]
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # time series split
)

# Train model
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    objective='reg:squarederror',
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print(f"Test RMSE: {rmse:.6f}")

# Plot feature importance
xgb.plot_importance(model, importance_type='gain', height=0.5, max_num_features=25)
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()