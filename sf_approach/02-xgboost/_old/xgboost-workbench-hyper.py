import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
import numpy as np

# Load the data
df = pd.read_csv("../fetch/data/AMZN_H1.csv")

# Drop rows with any missing values
df.dropna(inplace=True)

# Sort by timestamp (important for time series)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.sort_values('timestamp', inplace=True)

# Define features and target
target = 'close'
features = df.columns.difference(['timestamp', target])
X = df[features]
y = df[target]

# Use TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# Define Optuna objective
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 1.0),
    }

    model = xgb.XGBRegressor(**params)
    scores = cross_val_score(
        model, X, y,
        cv=tscv,
        scoring=make_scorer(mean_squared_error, greater_is_better=False),
        n_jobs=-1
    )
    return np.mean(scores)

# Optimize
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Output best hyperparameters
print("Best hyperparameters:")
print(study.best_params)