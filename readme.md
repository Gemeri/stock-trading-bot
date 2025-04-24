# Algorithmic Trading Platform – Documentation

---

> **Note:**  
> You must **rename the file `env.env` to `.env`** before running anything.  
> The `.env` file contains all configuration keys/credentials and is **essential** for the app to work.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Configuration (`.env`)](#configuration-env)
- [How the System Works](#how-the-system-works)
  - [1. Data Fetch & Feature Engineering](#1-data-fetch--feature-engineering)
  - [2. ML Model Layer](#2-ml-model-layer)
  - [3. News & Sentiment Analysis](#3-news--sentiment-analysis)
  - [4. Customizable Trade Logic](#4-customizable-trade-logic)
  - [5. Trading, Orders & Logging](#5-trading-orders--logging)
  - [6. Scheduling & Automation](#6-scheduling--automation)
  - [7. Interactive Console](#7-interactive-console)
  - [8. Backtesting](#8-backtesting)
- [Trade Logic Scripts](#trade-logic-scripts)
- [Timeframe Customization](#timeframe-customization)
- [Commands Cheat Sheet](#commands-cheat-sheet)
- [Adding New Logic Scripts](#adding-new-logic-scripts)
- [Troubleshooting](#troubleshooting)
- [License](#license)


---

## Overview

This is an automated, multi-strategy algorithmic trading framework written in Python. It is designed for
- **Research**
- **Automated live equities trading (via [Alpaca](https://alpaca.markets))**
- **Feature engineering, ensemble predictions, RL, and AI-enhanced ticker selection**

The platform fetches historical and real-time stock data, engineers features (e.g. technical indicators, sentiment), applies various ML and deep learning models, and then executes trading/position management based on customizable strategy scripts. It includes robust support for **backtesting**, **feature importance**, **real-time execution**, and **strategy prototyping**.

The trading decision process is **modular and swappable** via the `/logic` directory, allowing for 30+ different (and user-extendable) trading logic implementations.

---

## Features

- **Alpaca API** integration for live trading, portfolio management, and fetching news.
- **Automated data fetch & engineering** (candles, technical indicators, rolling features, etc.).
- **Sentiment Analysis**: Integrates news headlines and AI models for market sentiment.
- **Configurable ML stack**: 
  - Random Forests, XGBoost, LSTMs, Transformers (regression/classification), etc.
  - Meta-model stacking (ensemble).
- **Plug & play trade logic scripts** (in `/logic`).
- **Reinforcement Learning** strategies (PPO, DDQN, etc.)
- **Feature selection/importance & auto-disable**
- **Advanced backtesting** (out-of-sample, walk-forward, export to CSV, plots)
- **Discord bot** notifications (optional)
- **AI-augmented ticker selection (ChatGPT + Bing)**
- **Console command interface** for on-the-fly management, model selection, and debugging.
- **.env-based configuration** for flexible deployments and rapid testing.


---

## Project Structure

```
├── forest.py               # Main script; brings everything together
├── logic/
│   ├── logic_15_forecast_driven.py
│   ├── logic_17_gl_short.py
│   ├── logic_20_ddqn_2.py
│   ├── logic_23_ga.py
│   ├── logic_29_ppo_comp.py
│   ├── logic_30_main.py
│   └── ... (30 total, numbered)
├── logic/logic_scripts.json    # mapping: TRADE_LOGIC number → script
├── timeframe.py               # Trading schedule time customizations
├── .env                       # Must be renamed from env.env
└── ...
```

---

## Setup & Installation

1. **Clone the repo**

    ```bash
    git clone https://github.com/gemeri/stock-trading-bot
    cd stock-trading-bot
    ```

2. **Rename the environment file (MANDATORY)**

    ```bash
    mv env.env .env
    ```

3. **Install Python requirements**

    > It is strongly recommended to use a virtual environment.
    ```bash
    # Create and activate a venv (optional but recommended)
    python3 -m venv .venv
    source .venv/bin/activate

    # Install all dependencies
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

4. **Add your API keys**

    Open `.env` in a text editor and fill in:
    - **ALPACA_API_KEY** / **ALPACA_API_SECRET**
    - **OPENAI_API_KEY**
    - **BING_API_KEY**
    - (Optionally) Discord keys/tokens

    Adjust **TICKERS**, **ML_MODEL**, **BAR_TIMEFRAME**, etc., as needed.

5. **(Optional) Setup Discord bot**  
    Set `DISCORD_MODE=on`, and fill in the `DISCORD_TOKEN` and `DISCORD_USER_ID`.

---

## Configuration (`.env`)

The `.env` file supplies all credentials, tickers, trading settings, schedules, etc.

Example:

```ini
OPENAI_API_KEY=sk-...
ALPACA_API_KEY=PK...
ALPACA_API_SECRET=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets

BING_API_KEY=...      # Needed for AI ticker selection

TICKERS=TSLA,AAPL,MSFT
BAR_TIMEFRAME=4Hour   # 15Min, 30Min, 1Hour, 2Hour, 4Hour, 1Day
N_BARS=5000

NEWS_MODE=on          # Sentiment/news integration: on/off
DISABLED_FEATURES=stoch_k,macd_line # Features to remove (comma-separated). 'main' or 'base' for presets.
TRADE_LOGIC=15        # Selects which /logic/ script runs (see below)
AI_TICKER_COUNT=1     # How many AI-proposed tickers to add

DISCORD_MODE=off      # on/off
DISCORD_TOKEN=
DISCORD_USER_ID=

ML_MODEL=all          # forest, xgboost, lstm, transformer, transformer_cls, or all
RUN_SCHEDULE=on       # auto trading on schedule
SENTIMENT_OFFSET_MINUTES=20 # when to trigger news model relative to trading
```

---

## How the System Works

### 1. Data Fetch & Feature Engineering

- Candlesticks and historical data for your stocks are pulled from Alpaca.
- Technical indicators computed: e.g. EMA, MACD, RSI, Bollinger Bands, momentum, lagged prices, OBV, volume features, etc.
- Feature set is customizable via `.env`.

### 2. ML Model Layer

- Selects and fits models (RandomForest, XGBoost, LSTM, Transformer, etc.) based on `.env:ML_MODEL`.
- Final prediction is meta-ensembled if multiple models are enabled.
- Supports both regression (price forecast) and classification (direction, via TransformerCL).

### 3. News & Sentiment Analysis

- If `NEWS_MODE=on`, the system collects the latest market news per ticker and applies an AI model that outputs a **sentiment score**.
- The sentiment is "attached" to each candle/timestamp for training/prediction.
- Sentiment feature can be removed by turning `NEWS_MODE=off`.

### 4. Customizable Trade Logic

- **The heart of the strategy is the "Trade Logic Script" selected by `TRADE_LOGIC` in the `.env` file.**
- Each script in `/logic/` implements:
  - `run_logic(current_price, predicted_price, ticker)`  
    (run for live trades with latest data)
  - `run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles)`  
    (used during backtests)
- Common designs include forecast-driven, meta-ML, RL-based (PPO, DDQN), genetic algorithms, and multi-model hybrids.

### 5. Trading, Orders & Logging

- When trade logic triggers a buy/sell/short/cover, the **Alpaca API** is used for order execution.
- Full trade logs are saved in `trade_log.csv`, including: timestamp, action, price, prediction, P/L, and associated logic.
- Risk allocation is handled (capital-per-ticker), prevents duplicate positions, covers prior shorts/longs as needed.

### 6. Scheduling & Automation

- All trading (and news) jobs are queued via the [schedule](https://schedule.readthedocs.io/) module.
- The times are controlled per timeframe (see [timeframe customization](#timeframe-customization)).
- Jobs can be run on schedule or forced interactively.

### 7. Interactive Console

- The main script reads standard input for commands, e.g.:
    - `get-data`
    - `predict-next`
    - `force-run`
    - `set-tickers MSFT,NVDA`
    - `trade-logic 30`
    - `backtest 100 simple 4Hour`
    - `feature-importance`
    - and much more (see the [cheat sheet](#commands-cheat-sheet))

### 8. Backtesting

- The `backtest` command enables walk-forward or rolling backtests.
- All trades, predictions & portfolio values are exported to CSV and plots.
- Uses each trade logic script’s `run_backtest`, so strategies can be reliably evaluated on historical data.

---

## Trade Logic Scripts

Each file in `/logic/` is a separate **strategy module**. You can select which to use in `.env`  
(e.g. `TRADE_LOGIC=30` → `logic_30_main.py`).

**Examples (included):**
- `logic_15_forecast_driven.py` – Simple: Long when forecast says up, short when down.
- `logic_17_gl_short.py` – Hybrid: Genetic algorithms, XGBoost, PPO RL (supports both long and short).
- `logic_20_ddqn_2.py` – RL: Double DQN agent and RandomForest ensemble.
- `logic_23_ga.py` – Walkforward GA optimizer: Evolve pure-momentum+sentiment parameters.
- `logic_29_ppo_comp.py` – RL vs RL competition using Stable-Baselines PPO.
- `logic_30_main.py` – RL hybrid with ML-driven feature (predicted price), full PPO plus RandomForest/XGBoost for edge.
- *(and more... total 30 scripts: use, mix, or create your own!)*

Every script must implement:
```python
def run_logic(current_price, predicted_price, ticker):
    # executes trade live

def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    # returns: "BUY", "SELL", "SHORT", "COVER", or "NONE"
```
See `/logic/logic_15_forecast_driven.py` for a well-documented example of the interface.

---

## Timeframe Customization

All NY-time market triggers (when jobs run) are housed in:

- `timeframe.py` and
- `forest.py` (`TIMEFRAME_SCHEDULE`)

Example (in `timeframe.py`):
```python
TIMEFRAME_SCHEDULE = {
    "4Hour": ["12:01", "16:01", "20:01"],
    "2Hour": ["10:01", ...],
    ...
}
```
Modify these times to adjust trade/news run-windows for your preferred bar-length.

---

## Commands Cheat Sheet

All commands can be typed **at the console** after starting `forest.py`:

| Command                           | Description                                                   |
|------------------------------------|---------------------------------------------------------------|
| turnoff                           | Graceful shutdown                                            |
| api-test                          | Test Alpaca API keys                                         |
| get-data [timeframe]              | Fetch & save current candles; build features                  |
| predict-next [-r]                 | Predict next close for all tickers (use -r for cached data)   |
| run-sentiment [-r]                | Analyze news/sentiment and merge to CSV                       |
| force-run [-r]                    | Execute trade logic immediately (bypassing schedule)          |
| backtest <N> [simple|complex] ... | Run backtest for N candles/rows (outputs CSV/plots)           |
| feature-importance [-r]           | Report feature importances on latest data                     |
| set-tickers T1,T2,...             | Change ticker list in .env + memory                           |
| set-timeframe [frame]             | E.g. 4Hour, 15Min, 1Hour, etc.                               |
| set-nbars [num]                   | How many candles to fetch                                    |
| set-news [on/off]                 | Toggle sentiment/news features                                |
| trade-logic [number]              | Switch core trading script (see `/logic/`)                    |
| disable-feature <list/main/base>  | Drop features (comma list, or main/base presets)              |
| auto-feature [-r]                 | Auto-disable low-importance features                          |
| set-ntickers <int>                | Set number of AI tickers to propose via GPT/Bing              |
| ai-tickers                        | Show AI-curated tickers                                       |
| create-script <name>              | Create a new trade logic script template                      |
| commands                          | Print all commands                                            |

---

## Adding New Logic Scripts

1. **Create a new script:**
    - Use the command `create-script my_strategy` at the console OR
    - Manually create `logic/logic_31_my_strategy.py`

2. **Implement at least:**
    ```python
    def run_logic(current_price, predicted_price, ticker):
        # (see provided templates for structure)

    def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
        # Return action as string: "BUY", "SELL", "SHORT", "COVER", or "NONE"
    ```

3. **Select via:**  
    - Update `.env: TRADE_LOGIC=<number>` (e.g. `TRADE_LOGIC=31`)
    - The system will reload the mapping automatically.

---

## Troubleshooting

- **Nothing happens / No trades / No data:**  
  - Ensure `.env` is named correctly and filled out with your API keys and TICKERS.
  - Check console logs/errors (output window).
  - Use `get-data` or `predict-next` to ensure data is flowing.

- **Dependencies errors:**  
  - Ensure all pip packages are installed (see [Setup](#setup--installation)).

- **Strategy "does nothing":**
  - Some strategies are highly restrictive or may require longer datasets or certain market conditions.

- **Alpaca issues:**
  - Use `api-test` to check API keys.
  - If trading on paper, ensure `ALPACA_BASE_URL` points to Alpaca's paper endpoint.

- **Discord:**  
    - If you want Discord notifications, set `DISCORD_MODE=on` and supply bot credentials.

---

## License

MIT License
You are welcome to modify, fork, and contribute new ideas, logic scripts, and enhancements!

---

Happy Trading!  
