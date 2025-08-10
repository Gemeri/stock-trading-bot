import pandas as pd
import numpy as np
import xgboost as xgb
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from linear import stock_price_direction
from data import PredictItem
from utils import to_df

from utils import load_and_engineer_features

LOOKAHEAD_LIST = [3, 5, 8]
ENABLE_INTRADAY = False
SHORT_RATE = 0.2
BUY_RATE = 0.2
TRAIN_BACK_WINDOW = 800

if len(sys.argv) > 1:
    stock_ticker = sys.argv[1].upper()
    print(f"Stock ticker: {stock_ticker}")
else:
    raise ValueError("No stock ticker specified")

# Load training data
df = load_and_engineer_features(f"../01_fetch/data/{stock_ticker}_H1.csv")

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

# we need a dataframe to include
# timestamp, close, direction
predict_item_list:list[PredictItem] = []

# Backtest from candle end-1200 to now -> 18mo circa
for i in range(len(df)-TRAIN_BACK_WINDOW, len(df)-1):

    print(f"Loading candles for {stock_ticker} from {i-800} to {i}")

    # Train model on previous 12 months (approx. 800 candles)
    train_data = df.iloc[i-TRAIN_BACK_WINDOW:i]
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

    current_price = test_data['close'].values[0]

    direction = stock_price_direction([0] + LOOKAHEAD_LIST, [current_price] + predicted_prices)

    # extract the date
    timestamp = pd.to_datetime(test_data["timestamp"].values[0])

    item = PredictItem(timestamp, current_price, direction)

    print(f"Item -> {item} / still to go: {len(df)-1-i}")
    
    # populate the working dataset
    predict_item_list.append(item)
                           
print("#### Prediction phase completed")

df = to_df(predict_item_list)

df.to_csv(f"pred/predict-data-{stock_ticker}.csv", index=True)

print("#### Prediction file saved")