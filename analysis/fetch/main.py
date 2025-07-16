import logging
from fetch_data import fetch_candles_plus_features

logging.basicConfig(
    level=logging.INFO,  # or DEBUG, WARNING, etc.
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("Getting data... ")

# let's download 1 ticket
data = fetch_candles_plus_features('AAPL', 3000, '2Hour') # ~ at 1hr / 7 candles/day = ~130 days / half a year

logging.info("... done")

print(data.head())

logging.info("closing down")