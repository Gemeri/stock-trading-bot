# Stock Trading Bot

A comprehensive automated stock trading bot that uses advanced machine learning models, technical indicators, sentiment analysis from financial news, and multiple trading logic strategies to generate trading signals and execute orders via the Alpaca API. The bot also supports Discord notifications, backtesting, and AI-powered ticker recommendations.

---

## Overview

This project is designed to automate stock trading by:
- **Fetching Market Data:** Uses the Alpaca API to retrieve historical candle data.
- **Feature Engineering:** Computes technical indicators (e.g., RSI, MACD, Bollinger Bands), momentum, and custom features.
- **Sentiment Analysis:** Leverages a pre-trained financial news sentiment model to integrate market sentiment into trading decisions.
- **Machine Learning Models:** Supports both Random Forest and XGBoost models for next-close price prediction.
- **Modular Trading Logic:** Implements a flexible, pluggable architecture where trading strategies (including reinforcement learning, genetic algorithms, and more) reside in separate logic scripts.
- **Scheduling & Interactive Console:** Schedules trading jobs based on New York market times and provides a console-based command interface for live interactions.
- **Discord Integration:** Sends order notifications via Discord when enabled.
- **Backtesting & Feature Importance:** Provides tools to backtest strategies and analyze feature contributions.

---

## Features

- **Market Data Fetching:** Retrieves up to 10000 bars of data with configurable timeframes (15Min, 30Min, 1Hour, 2Hour, 4Hour, 1Day, etc.).
- **Advanced Feature Engineering:** Computes various technical indicators such as RSI, MACD, ATR, Bollinger Bands, momentum, and many lagged features.
- **Sentiment Analysis:** Processes financial news headlines and summaries using a Transformer model to generate sentiment scores.
- **Model Prediction:** Predicts the next close price using either a Random Forest or XGBoost regressor.
- **Flexible Trading Logic:** Switch between 30+ logic modules (found in the `logic/` directory) for different trading strategies.
- **Interactive Commands:** Commands include testing API keys, data fetching, predictions, running sentiment jobs, backtesting, and more.
- **Discord Notifications:** Optionally sends trading alerts via Discord direct messages.
- **AI Ticker Recommendations:** Uses OpenAI and Bing to suggest additional stock tickers dynamically.
- **Backtesting:** Offers both “simple” and “complex” backtesting approaches with trade logs and portfolio tracking.

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/gemeri/stock-trading-bot
   cd stock-trading-bot
   ```

2. **Create a Virtual Environment (Recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   Ensure you have Python 3.7+ installed. Then, install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

rename the `env.env` to `.env` to customize your environment variables. Below is the configuration:

```ini
# .env
OPENAI_API_KEY=
ALPACA_API_KEY=
ALPACA_API_SECRET=
ALPACA_BASE_URL=https://paper-api.alpaca.markets
BING_API_KEY=
TICKERS=TSLA
BAR_TIMEFRAME=4Hour      # Options: 15Min, 30Min, 1Hour, 2Hour, 4Hour, 1Day, etc.
N_BARS=5000
NEWS_MODE=on            # Options: on or off
DISABLED_FEATURES=body_size,candle_rise,high_low_range,hist_vol,log_volume,macd_line,price_change,price_return,roc,rsi,stoch_k,transactions,volume,volume_change,wick_to_body
TRADE_LOGIC=30          # Select which trading logic script to use (found in the logic/ directory)
AI_TICKER_COUNT=0 
DISCORD_MODE=off        # Options: on or off
DISCORD_TOKEN=
DISCORD_USER_ID=
ML_MODEL=forest         # Options: xgboost or forest
```

**Customization Options:**
- **TICKERS:** Set your primary stock tickers (comma-separated).
- **BAR_TIMEFRAME & N_BARS:** Control the time granularity and number of bars to fetch.
- **NEWS_MODE:** Toggle sentiment analysis on or off.
- **DISABLED_FEATURES:** Disable specific feature calculations if not needed.
- **TRADE_LOGIC:** Select the trading strategy module.
- **AI_TICKER_COUNT:** Enable AI-powered ticker recommendations.
- **DISCORD_MODE:** Enable or disable Discord notifications.
- **ML_MODEL:** Choose between using a Random Forest or XGBoost model.

---

## Usage

### Running the Bot

Start the bot by running the main script:

```bash
python forest.py
```

The bot will:
- Load the configuration from the `.env` file.
- Schedule trading and sentiment update jobs based on New York time.
- Start an interactive console listener to accept live commands.

### Console Commands

The interactive console allows you to execute various commands:
- **turnoff:** Shutdown the bot gracefully.
- **api-test:** Test Alpaca API connectivity.
- **get-data [timeframe]:** Fetch and save candle data.
- **predict-next [-r]:** Predict the next closing price.
- **run-sentiment [-r]:** Run the sentiment update job.
- **force-run [-r]:** Force execution of the trading job, bypassing market open checks.
- **backtest <N> [simple|complex] [timeframe?] [-r]:** Run backtesting with specified parameters.
- **feature-importance [-r]:** Compute and display feature importances.
- **set-tickers / set-timeframe / set-nbars / set-news:** Update environment variables on the fly.
- **trade-logic:** Switch the trading strategy.
- **disable-feature / auto-feature:** Manage feature calculations.
- **set-ntickers:** Set the number of AI tickers.
- **ai-tickers:** Fetch AI-based ticker recommendations.
- **create-script (name):** Create a new trading logic script template in the `logic/` directory.

Use the command `commands` to see a full list of available commands.

---

## Trading Logic Modules

The `logic/` subdirectory contains over 30 trading logic scripts that implement various strategies including:
- Reinforcement Learning (RL) models
- Traditional machine learning algorithms
- Genetic algorithms and other advanced trading techniques

The active module is selected via the `TRADE_LOGIC` environment variable. You can switch strategies without modifying the core code.

---

## Backtesting

The bot includes an extensive backtesting mode. Two approaches are provided:
- **Simple Backtest:** Trains on a fixed split of historical data.
- **Complex Backtest:** Re-trains the model iteratively over historical data to simulate evolving market conditions.

Backtesting commands generate CSV files with predictions, trade logs, and portfolio values as well as plots to help you visualize performance.

---

## Logging & Notifications

- **Logging:** Detailed logs are generated for every step (data fetching, feature engineering, model training, order execution, etc.). Check the console or log files for more information.
- **Discord Notifications:** If enabled (`DISCORD_MODE=on`), trading orders and alerts are sent via Discord direct messages using the provided bot token and user ID.

---

## Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to check [issues page](https://github.com/yourusername/stock-trading-bot/issues) if you want to contribute.

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push to your fork.
4. Open a pull request describing your changes.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- **Alpaca API:** For providing market data and trading functionalities.
- **Transformers Library:** For the pre-trained sentiment analysis model.
- **OpenAI & Bing APIs:** For powering the AI ticker recommendation system.
- **Community Contributions:** Thanks to all contributors who helped improve this project.

---
