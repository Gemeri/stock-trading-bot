# Logic for the back test

- we load 1200 candles (1 year) using the alpaca apis
- we run a backtest from candles 400 to 1200
-- we start with 10k dollars
-- we train a xgboost model on the earlierst 6mo worth of data using n_estimators=114 and max_depth=9, 
-- we predict the next candle
-- depending on the prediciton we apply a simple trading strategy:
--- if the stock is projected to grow more than 1% we go long (buy max 10% of available funds of stocks)
--- if the stock is predicted to drop more than 1% we go short (sell all stocks)
--- if the stock is predicted to stay within +/- 1% we hold (do nothing)
-- we track the value of the portfolio over time
- we collect the result of the backtest and show it on a graph


F1, F2, F3, close, target (close+1)

F1, F2, F3, close, predicted_target, target (close+1)



# Logic for the bot

- wake up
- get the latest 3mo worth of candles ~ 400 candles (378)

- review previous predicted vs real one and see error rate
-- report to telegram
- hyperparameter tuning
- train the model
- review root_mean_squared_error and see if within the margin
-- report to telegram
- predict the next close price
- apply trading strategy
-- report to telegram


