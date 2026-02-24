import forest
from forest import api
from openai import OpenAI
import requests
import logging
import forest
import os

logger = logging.getLogger(__name__)

forest.TICKERS, forest.TICKER_ML_OVERRIDES, forest.SELECTION_MODEL, forest.TRADE_LOGIC

def load_tickerlist() -> list[str]:
    """Load tickers and per-ticker ML overrides from tickerlist.txt."""
    forest.TICKER_ML_OVERRIDES = {}
    forest.SELECTION_MODEL = None
    tickers: list[str] = []

    if not os.path.exists(forest.TICKERLIST_PATH):
        logging.error(f"Ticker list file not found: {forest.TICKERLIST_PATH}")
        forest.TICKERS = []
        return []

    with open(forest.TICKERLIST_PATH) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            if "=" in ln:
                key, val = ln.split("=", 1)
                if key.strip().lower() == "selection_model":
                    val = val.strip()
                    # Allow direct trade logic scripts via selection_model
                    if val.lower().endswith('.py'):
                        forest.SELECTION_MODEL = f"logic{os.sep}{val}"
                        forest.TRADE_LOGIC = val
                    else:
                        forest.SELECTION_MODEL = val
                continue
            parts = [p.strip() for p in ln.split(",")]
            ticker = parts[0].upper()
            tickers.append(ticker)
            if len(parts) > 1 and parts[1]:
                forest.TICKER_ML_OVERRIDES[ticker] = parts[1].strip()

    forest.TICKERS = tickers
    return tickers

# load tickers at startup
load_tickerlist()

def fetch_new_ai_tickers(num_needed, exclude_tickers):
    openai_client = OpenAI(api_key=forest.OPENAI_API_KEY)

    exclude_list_str = ", ".join(sorted(exclude_tickers)) if exclude_tickers else "None"
    prompt_text = (
        f"You are an AI that proposes exactly {num_needed} unique US stock tickers (one per line)\n"
        f"that are likely to rise soon, based on fundamental/technical analysis.\n"
        f"Use your search tool to find promising tickers with good fundemental, techincal and sentiment factors\n"
        f"Do NOT include these tickers: {exclude_list_str}\n"
        f"Output only {num_needed} lines, each line is a ticker symbol only, no extra text."
    )

    try:
        system_prompt = "You are a financial assistant that suggests promising tickers."
        completion = openai_client.responses.create(
            model="gpt-5.2",
            tools=[{"type": "web_search"}],
            reasoning={"effort": "medium"},
            instructions=system_prompt,
            input=prompt_text
        )
        content = completion.output_text

    except Exception as e:
        logging.error(f"Error calling OpenAI ChatCompletion: {e}")
        return []

    lines = content.split('\n')
    candidate_tickers = []
    for ln in lines:
        tck = ln.strip().upper()
        tck = tck.replace('.', '').replace('-', '').replace(' ', '')
        if tck and tck not in exclude_tickers:
            candidate_tickers.append(tck)

    candidate_tickers = candidate_tickers[:num_needed]
    return candidate_tickers


def _ensure_ai_tickers():

    if forest.AI_TICKER_COUNT <= 0:
        return

    still_open = set()
    try:
        positions = api.list_positions()
        for p in positions:
            if p.symbol in forest.AI_TICKERS and abs(float(p.qty)) > 0:
                still_open.add(p.symbol)
    except Exception as e:
        logging.error(f"Error retrieving positions in _ensure_ai_tickers: {e}")

    forest.AI_TICKERS = [t for t in forest.AI_TICKERS if t in still_open]

    needed = forest.AI_TICKER_COUNT - len(forest.AI_TICKERS)
    if needed > 0:
        exclude = set(forest.TICKERS) | set(forest.AI_TICKERS)
        new_ai_tickers = fetch_new_ai_tickers(needed, exclude)
        for new_tck in new_ai_tickers:
            if new_tck not in forest.AI_TICKERS:
                forest.AI_TICKERS.append(new_tck)
        logging.info(f"Fetched {len(new_ai_tickers)} new AI tickers: {new_ai_tickers}")
    else:
        logging.info("No need to fetch new AI tickers; current AI Tickers are sufficient.")