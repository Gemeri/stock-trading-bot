# Stock Trading Bot

> **Note:** Some parts of the script may currently contain errors related to the ML_MODEL options being transformer and the "all" meta model. These issues are known and will be fixed in an upcoming update

A highly configurable and extensible stock trading bot that fetches market data, computes advanced features, trains and evaluates machine learning models, and executes trades via the Alpaca API. It supports multiple strategies (supervised, reinforcement learning, genetic algorithms, mathematical heuristics), ensemble submodels (sub-vote, sub-meta), news sentiment integration, interactive console commands, scheduling, backtesting, and optional Discord notifications.

---

## Features

* **Data Collection & Feature Engineering**: Fetches historical OHLCV data from Alpaca and computes technical and sentiment-based features.
* **Multiple Strategy Support**: Includes supervised models (Random Forest, XGBoost, LSTM, Transformers), reinforcement learning agents (DQN, PPO, SAC, DDQN), genetic algorithms, and mathematical heuristics.
* **Ensemble Submodels**: Combines five submodels via voting (`sub-vote`) or meta-learning (`sub-meta`) strategies.
* **News Sentiment Analysis**: Integrates Bing News API for sentiment scoring when `NEWS_MODE` is enabled.
* **Interactive Console**: Real-time commands to manage data fetching, predictions, sentiment runs, backtests, and bot configuration.
* **Scheduling**: Automatic job scheduling based on configurable timeframes (15 Min, 30 Min, 1 Hour, 2 Hour, 4 Hour) for U.S. market hours.
* **Backtesting**: Powerful backtester that generates trade logs, portfolio P\&L, and prediction/importance reports with CSV and PNG outputs.
* **Discord Integration**: Optional Discord mode for real-time alerts and summaries.

---

## Repository Structure

```
├── LICENSE
├── TSLA_H4.csv           # Sample 4‑hour TSLA data
├── env.env               # Environment variable template
├── forest.py             # Main entry point and orchestrator
├── timeframe.py          # Defines NY session trigger times for each bar length
├── logic/                # Active trading logic scripts (numbered strategies)
├── logic-old/            # Legacy/deprecated logic scripts
├── sub/                  # Ensemble submodel implementation
├── requirements.txt      # Python dependencies
├── results/              # Backtest output (CSV & PNG) organized by strategy
└── README.md             # This file
```

---

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Gemeri/stock-trading-bot.git
   cd stock-trading-bot
   ```

2. **Create & activate a virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:

   * Copy `env.env` to `.env` in the project root.
   * Fill in your API keys and desired settings (see next section).

---

## Configuration (`.env`)

```dotenv
# OpenAI for advanced models & scripts (optional)
OPENAI_API_KEY=your_openai_api_key

# Alpaca Trading API (paper or live)
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_API_SECRET=your_alpaca_api_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Bing News for sentiment analysis (optional)
BING_API_KEY=your_bing_api_key

# Trading & Data Settings
TICKERS=TSLA, AAPL             # Comma‑separated list of tickers
BAR_TIMEFRAME=30Min            # 15Min, 30Min, 1Hour, 2Hour, 4Hour, etc.
N_BARS=5000                    # Number of historical bars to fetch
NEWS_MODE=on                   # on/off to enable news sentiment integration
TRADE_LOGIC=1                  # Select logic script ID from `logic/` folder
AI_TICKER_COUNT=0              # Number of tickers to generate via AI models (0=disabled)

# Machine Learning Settings
ML_MODEL=forest                # forest, xgboost, lstm, transformer, sub-vote, sub-meta
META_MODEL_TYPE=logreg         # logreg, lgbm, xgb, nn (for sub-meta)
USE_META_LABEL=false           # true/false to use meta labels in ensemble

# Scheduling
RUN_SCHEDULE=on                # on/off to enable scheduled jobs
SENTIMENT_OFFSET_MINUTES=5     # Minutes before bar close to fetch sentiment
REWRITE=off                    # on/off to force re-computation of features

# Discord Integration (optional)
DISCORD_MODE=off               # on/off to enable Discord notifications
DISCORD_TOKEN=your_discord_bot_token
DISCORD_USER_ID=your_discord_user_id
```

---

## Usage

### Running the Bot

Start the trading bot with:

```bash
python forest.py
```

The bot will:

1. Load your configuration from `.env`.
2. Schedule data-fetching, model retraining, and trading jobs according to `BAR_TIMEFRAME` and `timeframe.py` settings.
3. Spawn an interactive console listener for on‑the‑fly commands.

### Interactive Console Commands

At any time, type commands in the terminal where the bot is running:

* `turnoff`                           – Gracefully shuts down the bot.
* `api-test`                         – Validates Alpaca API credentials.
* `get-data [TIMEFRAME] [-r]`        – Fetches OHLCV bars + features; `-r` uses existing CSV.
* `predict-next [TIMEFRAME] [-r]`    – Computes next-bar prediction; skips data fetch if `-r`.
* `feature-importance [-r]`          – Calculates and plots feature importances.
* `run-sentiment [-r]`               – Updates news sentiment scores.
* `force-run [-r]`                   – Immediately runs sentiment + trading job, ignoring market hours.
* `backtest <TEST_SIZE> [simple|complex]`  – Runs backtest on last `TEST_SIZE` bars.
* `trade-logic <ID>`                 – Switches to logic script `logic_<ID>`.py.
* `set-tickers <T1,T2,...>`         – Updates `TICKERS` list at runtime.
* `set-timeframe <TF>`               – Updates `BAR_TIMEFRAME` at runtime.
* `set-nbars <N>`                   – Updates `N_BARS` at runtime.
* `set-news <on|off>`               – Toggles `NEWS_MODE`.
* `set-logic <ID>`                  – Updates `TRADE_LOGIC`.
* `ai-tickers`                      – Displays AI‑generated tickers (if `AI_TICKER_COUNT`>0).
* `create-script <NAME>`            – Scaffolds a new logic script template in `logic/`.
* `commands`                        – Prints this command list.

### Backtesting & Results

Backtest outputs are saved under:

```
results/<logic_id>/
├── backtest_trades_<TICKER>_<N>_<TF>_<approach>.csv
├── backtest_predictions_<TICKER>_<N>_<TF>_<approach>.csv
├── backtest_portfolio_<TICKER>_<N>_<TF>_<approach>.csv
├── backtest_predictions_<...>.png
└── backtest_portfolio_<...>.png
```

Use these files for performance review and analysis.

---

## Models & Logic Scripts

* **Supervised**: `forest.py` trains Random Forest, XGBoost, LightGBM, Transformer, LSTM models.
* **Reinforcement Learning**: `logic_5_rl_1.py` & up implement DQN, PPO, SAC, DDQN, etc.
* **Genetic Algorithms**: `logic_2_ga_1.py`, `logic_3_ga_2.py`, `logic_4_ga_3.py`.
* **Mathematical Heuristics**: `logic_16_math.py` & up.
* **Neural Networks**: `logic_12_nn_1.py`, `logic_15_nn_2.py`, `logic_19_nn.py`.
* **Ensemble Submodels**: See `sub/` directory for the `sub-vote` and `sub-meta` implementations.

Legacy scripts are available in `logic-old/` but not actively maintained.

---

## Scheduling

Schedule triggers for each `BAR_TIMEFRAME` are defined in `timeframe.py`. By default, times are aligned to U.S. market hours in Eastern Time. Modify `TIMEFRAME_SCHEDULE` as needed.

---

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add new logic script"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

Please follow PEP8 style and write clear docstrings for new logic scripts.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
