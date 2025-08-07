import logging
from fetch_data import fetch_candles_plus_features

logging.basicConfig(
    level=logging.INFO,  # or DEBUG, WARNING, etc.
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("Getting data... ")

TICKER_LIST = ['CPG', 'ADT']

for ticker in TICKER_LIST:
    
    logging.info(f"Collecting data for {ticker}")

    # let's download 1 ticket
    data = fetch_candles_plus_features(ticker, 5000, '1Hour') # ~ at 1hr / 7 candles/day = ~130 days / half a year

    print(data.head())

    logging.info(f"Data for {ticker} collected")

logging.info("closing down")