# 🌳 Quantitative Trading Framework (forest.py)

> **Note:**  
> You must **rename the file `env.env` to `.env`** before running anything.  
> The `.env` file contains all configuration keys/credentials and is **essential** for the app to work.

## Overview

This is a **modular, multi-strategy algorithmic trading and research platform** with:

- **Feature-rich data pipelines** (Alpaca, news, Discord, sentiment, technicals, etc.)
- **Extensible ML & AI model ecosystem** (Random Forest, XGBoost, LightGBM, LSTM, Transformer, PPO, DDQN, GP, RL, Symbolic Regression, Voting/Meta-model stacking, etc.)
- **Plug-and-Play trade logic scripts** (rule-based, RL, Genetic Programming, meta-ensemble, etc.)
- **Sub-model voting/meta-ensembling** infrastructure
- **Backtesting & research automation**
- **Live trading** via Alpaca (paper and live)
- **News sentiment/natural language** integration (Bing, OpenAI, huggingface transformers)
- **Console/CLI + Discord bot remote control**
- **Easy configuration via `.env` and hot-reloading**

> FULLY CUSTOMIZABLE & RESEARCHER-FRIENDLY  
> NEW trade logic? Drop it in the `/logic` folder as a script, swap via env, and backtest instantly!

---

## Table of Contents

- [Key Features](#key-features)
- [Quickstart](#quickstart)
- [Trade Logic Scripts](#trade-logic-scripts)
- [Sub-model Ensemble / Voting](#sub-model-ensemble--voting)
- [Live Trading & Discord](#live-trading--discord)
- [Backtesting & Research](#backtesting--research)
- [Folder Structure](#folder-structure)
- [Configuration (.env)](#configuration-env)
- [Command Reference](#command-reference)
- [Adding Your Own ML Models or Logic](#adding-your-own-ml-models-or-logic)
- [Contributing](#contributing)
- [Requirements](#requirements)
- [License](#license)

---

## Key Features

- **ML Model Zoo**: Choose from Random Forest, XGBoost, LightGBM, LSTM, Transformer, Meta-model stacking, RL (PPO, DDQN), Symbolic Regression (PySR), and Genetic Programming.
- **Feature Engineering**: Automated technical, statistical, and sentiment feature creation (over 35 features).
- **Sentiment/NLP**: Real-time news from Alpaca News + Bing API, scored with RoBERTa sentiment classifier.
- **Plug-n-Play Trade Logic**: 30+ modular scripts fielding everything from simple rules to RL/GA/voting/meta.
- **Backtest Engine**: Walk-forward, expanding-window, portfolio-level, with CSV/plot output for research.
- **Sub-model Ensemble**: Research-grade submodel meta/vote stacking (see `/sub`).
- **Tight Alpaca Integration**: Real trading support (market/order sending, position management).
- **Discord Bot**: Optional trade/alert integration over Discord.
- **AI/LLM Ticker Discovery**: Automated Bing+GPT picking of new tickers for portfolio/trading.
- **Research-Speed CLI**: Hot-swap commands: feature importance, data fetch, prediction, backtests, etc.
- **Pro Researcher Controls**: Toggle features, timeframes, news, trade logic, tickers, etc at runtime.
- **Quick Scripting**: Auto-create templates for new logic scripts and curiosity-driven research.
- **Code Modularity**: Just drop new logic in `/logic/`, or plug in a new `/sub/` submodel.

---

## Quickstart

### Requirements

- Python 3.10+ recommended
- OS: Works on Mac, Windows, Linux
- **API keys for [Alpaca](https://alpaca.markets/), [OpenAI](https://openai.com/), [Bing](https://azure.microsoft.com/services/cognitive-services/bing-web-search-api/)**

#### Installation

```bash
git clone https://github.com/yourusername/forest-quant-trading.git
cd forest-quant-trading

# Install Python deps
pip install -r requirements.txt
# (See below for requirements)

# Copy/modify .env template
cp .env.example .env
vim .env  # Fill in API keys and settings!
```

#### Run the bot

```bash
python forest.py
```

#### Sample .env

```ini
OPENAI_API_KEY=YOUR_OPENAI_KEY_HERE
ALPACA_API_KEY=YOUR_ALPACA_KEY
ALPACA_API_SECRET=YOUR_SECRET
ALPACA_BASE_URL=https://paper-api.alpaca.markets
BING_API_KEY=YOUR_BING_API_KEY
TICKERS=TSLA,AAPL
BAR_TIMEFRAME=30Min       # 15Min, 30Min, 1Hour, 2Hour, 4Hour, 1Day, ...
N_BARS=5000
NEWS_MODE=on
TRADE_LOGIC=30
AI_TICKER_COUNT=0
DISCORD_MODE=off
DISCORD_TOKEN=
DISCORD_USER_ID=
ML_MODEL=forest           # forest, xgboost, lstm, transformer, all, classifier, sub-meta, sub-vote
RUN_SCHEDULE=on
SENTIMENT_OFFSET_MINUTES=5
REWRITE=off
```

---

## Trade Logic Scripts

Trade execution is **decoupled from prediction**!  
Each trading "brain" is its own .py script in `/logic/`, hot-swappable via `.env:TRADE_LOGIC`:

Example (from `/logic/logic_1_forecast_driven.py`):
```python
def run_logic(current_price, predicted_price, ticker):
    if predicted_price > current_price:
        # Buy logic
    elif predicted_price < current_price:
        # Sell logic
    else:
        # No trade

def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    # Return "BUY", "SELL", or "NONE"
```

**Included scripts:**  
- `logic_1_forecast_driven.py` - Basic forecast buy/sell (no short)
- `logic_2_ga_1.py` - Genetic Programming evolved trade logic (GA via deap)
- `logic_14_rl_10.py` - PPO RL agent (Stable-Baselines3, custom Gym env)
- `logic_15_best.py` - Improved RL w/ memory (FAISS + SBERT trade memory, deep PPO)
- `logic_16_math.py` - Symbolic Regression via PySR + skopt for math-form trading rules
- `logic_20_ddqn_2.py` - DDQN RL agent w/ Random Forest predictor hybrid
- `logic_26_ppo_2.py` - PPO agent, integrates Random Forest & XGBoost for price
- `logic_30_main.py` - Pure PPO + XGBoost/RandomForest, plug-and-play

#### Add your own:
```bash
python forest.py
> create-script my_logic
# then open logic/logic_31_my_logic.py and hack away!
```

---

## Sub-model Ensemble / Voting

**Why "forest"?** Because we orchestrate an ensemble (subforest) of smaller trees:
- `/sub/main.py` + `/sub/*.py` (momentum, mean reversion, sentiment/volume, lagged returns, trend, etc)
- **Voting ("sub-vote") and meta-model Logistic Regression ("sub-meta")** stack sub-models for robust signal
- Integrates seamlessly with the rest of the scheduling/backtesting engine
- Full walk-forward backtest and live "ensemble signal" possible

#### To use voting/meta ensemble:
Set in `.env`:
```
ML_MODEL=sub-vote       # or sub-meta
TRADE_LOGIC=<logic_id>  # e.g. 1, 14, etc (will use classifier logic)
```

---

## Live Trading & Discord

- **Runs on Alpaca**: fully supports paper and live trading
- **All API keys loaded from .env**
- **Position management, real PnL**
- **Discord bot supported:** (set `DISCORD_MODE=on`) for DMs/alerts

---

## Backtesting & Research

- **Command line research mode** (auto-backtests, rolling feature importance, get-data, etc)
- **Walk-forward expanding-window tests**
- **Portfolio/Trade log CSVs and all plots auto-saved**
- **Research commands:**

    ```bash
    > get-data                   # Fetch latest candles & features
    > predict-next [-r]          # ML pred for next close
    > backtest 200 [simple|complex] [timeframe] [-r]     # Rolling backtest of last 200 candles
    > feature-importance [-r]    # Rolling window RF feature importances
    > ai-tickers                 # Bing+GPT-generated trending symbols (for AI_TICKER_COUNT)
    > create-script my_rl_logic  # New logic script ready to go!
    > set-tickers AAPL,MSFT,QCOM
    > set-timeframe 4Hour
    ...
    ```

---

## Folder Structure

```
/
├─ forest.py              # Main controller: data fetch, ML, trading, CLI, scheduling
├─ .env                   # Configuration (API keys, tickers, logic, etc)
├─ logic/                 # User trade logic scripts (logic_N_strategy.py)
│   ├─ ...
│   └─ logic_14_rl_10.py  # RL PPO-based strategy, e.g.
├─ sub/                   # Sub-model voting/meta model folder (mean reversion, momentum, ...)
│   ├─ main.py            # Orchestrates the sub-model ensemble
│   │
│   └─ mean_reversion.py  # Example: sub-model as re-usable module
├─ results/               # Backtest results, plots, logs
│   └─ ...
└─ requirements.txt       # (See below: alpine, RL, pytorch, etc)
```

---

## Configuration (.env)

The system is **100% controlled via the `.env` file**:

- **TICKERS:** List of tickers (comma separated)
- **BAR_TIMEFRAME:** Data granularity (15Min, 30Min, 1Hour, ...)
- **N_BARS:** Number of bars in local data/history
- **NEWS_MODE:** on/off — fetch and attach Alpaca news sentiment
- **TRADE_LOGIC:** Which numbered logic script to use (`logic_N_*.py`)
- **ML_MODEL:** ML backbone(s): forest, xgboost, lstm, transformer, all, sub-meta, sub-vote
- **DISABLED_FEATURES:** (optional) comma list of features to drop from feature set
- **AI_TICKER_COUNT:** Use Bing+GPT to automatically propose trending tickers
- **RUN_SCHEDULE:** on/off (enables/disables scheduling/trading); else command line only
- **REWRITE:** on/off (force overwrite local CSVs and features)

---

## Command Reference

The main system runs an **interactive console (CLI) for research operations**:

| Command                        | Purpose                                         |
| ------------------------------ | ----------------------------------------------- |
| `turnoff`                      | Graceful shutdown                               |
| `api-test`                     | Verify Alpaca API key/connection                |
| `get-data [timeframe]`         | Download latest candles+features for all tickers|
| `predict-next [-r]`            | Forecast next bar for each ticker               |
| `run-sentiment [-r]`           | Update news sentiment scores                    |
| `force-run [-r]`               | Force a full run (data, sentiment, trading)     |
| `backtest <N> [...args]`       | Rolling-window backtest (full trading simulation)|
| `feature-importance [-r]`      | Rolling feature importance via RF               |
| `commands`                     | Show all commands                               |
| `set-tickers T1,T2,T3`         | Update tickers in session/env                   |
| `set-timeframe 1Hour`          | Change global bar timeframe                     |
| `set-nbars 7000`               | Set number of bars                              |
| `set-news on/off`              | Toggle news mode                                |
| `trade-logic 15`               | Change trade script number                      |
| `set-ntickers 7`               | AI_TICKER_COUNT update (auto AI tickers)        |
| `ai-tickers`                   | Display current AI-generated tickers            |
| `create-script my_logic`       | Create a new, ready-to-edit logic script        |

---

## Adding Your Own ML Models or Logic

**To add a new ML model type (e.g. TabNet, TCN, new RL):**
1. Add a handler in `forest.py`'s ML model zoo (see `get_single_model()`)
2. Add necessary import and requirements.
3. If needed, add "mode" in train_and_predict pipeline.

**To add/experiment with new trade logic or agent:**
1. Run: `create-script my_experiment`
2. Edit your new file in `/logic/`
3. Select it via `.env:TRADE_LOGIC`, then use all the main research commands.

**Sub-model voting/meta:** Edit/add in `/sub/`, register in `sub/main.py`.

---

## Sample Workflows

- **Regular live trading (scheduled):**
    - Fill in credentials in `.env`
    - Enable RUN_SCHEDULE=on
    - Start with `python forest.py` — will fetch, fit, schedule, and execute trades at NY session times

- **Research / Backtest:**
    - Any time: `python forest.py`
    - Console: use `backtest`, change models/logics, run `feature-importance`, etc.

- **Meta-model research:**
    - Set `ML_MODEL=all` / `ML_MODEL=sub-meta` and run walk-forward backtest.

- **Fully custom RL experiment:**
    - Drop your logic script in `/logic/`, register environments if needed, and swap TRADE_LOGIC.

---

## Example: Sub-model mean_reversion.py

```python
FEATURES = [...]
def compute_labels(df): ...
def fit(X, y): ...
def predict(model, X): ...
```

---

## Requirements

Key libraries:
- pandas, numpy, scikit-learn, matplotlib
- requests, openai, python-dotenv
- xgboost, lightgbm
- pytorch, torch, tensorflow, keras
- gym, stable-baselines3, deap, pysr, skopt
- sentence-transformers, faiss-cpu (for trade memory/embeddings)
- alpaca-trade-api
- discord.py (for Discord integration)
- schedule, tqdm, ast, etc.

See `requirements.txt` for a full list.

---

## License

MIT License

---

## Contributing

Contributions welcome!  
Please open an issue or PR for feature requests, bugfixes, or ideas.  
Major new models/logic scripts go in `/logic/`.  
Sub-models or enhancements to the ensemble go in `/sub/`.

---

**Happy Quant Researching and Systematic Trading!**

Questions?  
Raise a GitHub Issue or open a Discussion.

---
