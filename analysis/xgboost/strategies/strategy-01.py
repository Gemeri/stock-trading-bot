import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Parameters ===
INITIAL_CASH = 10_000
THRESHOLD = 0.015  # 1.5% threshold
PREDICTION_ERROR = 0.01  # +/-1% error in prediction

def generate_signals(df, upper_threshold=0.02, lower_threshold=-0.02):
    df = df.copy()
    
    # Calculate expected return from prediction
    df['expected_return'] = (df['predicted_price'] - df['actual_price']) / df['actual_price']
    
    # Generate trading signal
    # 1 = Long, -1 = Short, 0 = No trade
    df['signal'] = 0
    df.loc[df['expected_return'] > upper_threshold, 'signal'] = 1
    df.loc[df['expected_return'] < lower_threshold, 'signal'] = -1
    
    return df

def backtest(df):
    df = df.sort_values(['timestamp'])
    
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

# Sample simulation data (replace with real model predictions)
def simulate_sample_data():
    dates = pd.date_range(start='2023-01-01', periods=100)
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
    
    data = []
    np.random.seed(42)
    for ticker in tickers:
        prices = 100 + np.cumsum(np.random.normal(0, 1, size=len(dates)))
        for i in range(len(dates)):
            actual_price = prices[i]
            # Predicted price with some noise around actual (+/- 1%)
            predicted_price = actual_price * (1 + np.random.uniform(-0.01, 0.01))
            data.append([dates[i], ticker, actual_price, predicted_price])
    
    df = pd.DataFrame(data, columns=['date', 'ticker', 'actual_price', 'predicted_price'])
    return df

# Run the strategy
df = simulate_sample_data()
df_signals = generate_signals(df)
daily_returns, cumulative_returns, df_with_trades = backtest(df_signals)

# Plot performance
plt.figure(figsize=(10, 5))
plt.plot(cumulative_returns.index, cumulative_returns.values)
plt.title("Cumulative Return of Strategy")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.grid(True)
plt.show()

