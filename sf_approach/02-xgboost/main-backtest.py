from backtest import run_backtest
from plot import plot_and_save
from utils import load_predictions


STOCK_TICKER = "AAPL"

MIN_ACTION_THRESHOLD = 5
MAX_ACTION_THRESHOLD = 20

MIN_BUY_RATE = 0.1
MAX_BUY_RATE = 0.3

MIN_SHORT_RATE = 0.1
MAX_SHORT_RATE = 0.3

# we load the CSV as dataframe
predict_item_list = load_predictions(f"pred/predict-data-{STOCK_TICKER}.csv")

print(f"Loaded {len(predict_item_list)} items for ticker {STOCK_TICKER}")

top_balance = 0
best_threshold = False
best_buy_rate = False
best_short_rate = False

for action_threshold in range(MIN_ACTION_THRESHOLD, MAX_ACTION_THRESHOLD + 1):

    for buy_rate in range(MIN_BUY_RATE, MAX_BUY_RATE, 0.01):
            
        for short_rate in range(MIN_SHORT_RATE, MAX_SHORT_RATE, 0.01):
    
            print(f"Attempting backtest with threshold: {action_threshold} / buy_rate={buy_rate}, short_rate={short_rate}")

            backtest_list = run_backtest(predict_item_list, action_threshold, buy_rate=buy_rate, short_rate=short_rate)
            
            latest_balance = backtest_list[-1].portfolio_value

            print(f"---> Balance: f{latest_balance}")

            if latest_balance > top_balance:
                top_balance = latest_balance
                best_threshold = best_threshold
                best_buy_rate = best_buy_rate
                best_short_rate = best_short_rate

        #plot_and_save(STOCK_TICKER, backtest_list, action_threshold)

print(f"ALL DONE")

print(f"=> top_balance: {top_balance}")
print(f"=> best_threshold: {best_threshold}")
print(f"=> best_short_rate: {best_short_rate}")
print(f"=> best_buy_rate: {best_buy_rate}")