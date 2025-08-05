from backtest import run_backtest
from plot import plot_and_save
from utils import load_predictions


STOCK_TICKER = "AAPL"

MIN_ACTION_THRESHOLD = 5
MAX_ACTION_THRESHOLD = 20

# we load the CSV as dataframe
predict_item_list = load_predictions(f"pred/predict-data-{STOCK_TICKER}.csv")

print(f"Loaded {len(predict_item_list)} items for ticker {STOCK_TICKER}")

for action_threshold in range(MIN_ACTION_THRESHOLD, MAX_ACTION_THRESHOLD + 1):
    
    print(f"Attempting backtest with threshold: {action_threshold}")
    backtest_list = run_backtest(predict_item_list, action_threshold)
    
    latest_balance = backtest_list[-1].portfolio_value

    print(f"---> Balance: f{latest_balance}")

    plot_and_save(STOCK_TICKER, backtest_list, action_threshold)

print(f"ALL DONE")