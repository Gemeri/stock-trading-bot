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
    
    # Strategy
    if projected_change > 0.01:
        # Go long: invest 30% of current cash
        to_invest = 0.3 * cash
        num_shares = to_invest // current_price
        if num_shares > 0:
            cash -= num_shares * current_price
            shares += num_shares
    elif projected_change < -0.01:
        # Go short (sell all)
        cash += shares * current_price
        shares = 0
    
    # Track portfolio value
    portfolio_value = cash + shares * current_price
    portfolio_values.append(portfolio_value)

    print(f"portfolio updated: {portfolio_value}")


print("plotting")

# Step 6: Plot results
plt.figure(figsize=(12, 6))
plt.plot(range(len(df)-1200, len(df)-1), portfolio_values, label='Portfolio Value')
plt.title('Backtest Portfolio Value over Time')
plt.xlabel('Candle Index')
plt.ylabel('Portfolio Value ($)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()