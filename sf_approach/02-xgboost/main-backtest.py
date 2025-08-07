from alpaca_backtest import run_backtest
from plot import plot_and_save
from utils import load_predictions

import numpy as np

STOCK_TICKER = "INTC"

MIN_ACTION_THRESHOLD = 1
MAX_ACTION_THRESHOLD = 20

# we load the CSV as dataframe
predict_item_list = load_predictions(f"pred/predict-data-{STOCK_TICKER}.csv")

print(f"Loaded {len(predict_item_list)} items for ticker {STOCK_TICKER}")

top_balance = 0
best_threshold = False

best_backtest_list = []

for action_threshold in np.arange(MIN_ACTION_THRESHOLD, MAX_ACTION_THRESHOLD + 0.1, 0.1):

    print(f"Attempting backtest with threshold: {action_threshold}")

    backtest_list = run_backtest(predict_item_list, action_threshold, initial_cash=1000, buy_rate=1, short_rate=0.5)
    
    latest_balance = backtest_list[-1].portfolio_value

    print(f"---> Balance: f{latest_balance}")

    if latest_balance > top_balance:
        top_balance = latest_balance
        best_threshold = action_threshold
        best_backtest_list = backtest_list

#

print(f"ALL DONE")

print(f"=> top_balance: {top_balance}")
print(f"=> best_threshold: {best_threshold}")

plot_and_save(STOCK_TICKER, best_backtest_list, best_threshold)