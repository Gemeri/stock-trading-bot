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

    def bing_web_search(query):
        """Simple Bing Web Search to get some snippet info for each query."""
        search_url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": forest.BING_API_KEY}
        params = {"q": query, "textDecorations": True, "textFormat": "HTML", "count": 2}
        try:
            r = requests.get(search_url, headers=headers, params=params)
            r.raise_for_status()
            data = r.json()
            found = []
            if 'webPages' in data:
                for v in data['webPages']['value']:
                    found.append({
                        'name': v['name'],
                        'url': v['url'],
                        'snippet': v['snippet'],
                        'displayUrl': v['displayUrl']
                    })
            return found
        except Exception as e:
            logging.error(f"Bing search failed: {e}")
            return []

    search_queries = [
        "best US stocks expected to rise soon",
        "top bullish stocks to watch in the US market"
    ]

    snippet_contexts = []
    for one_query in search_queries:
        results = bing_web_search(one_query)
        for item in results:
            snippet_contexts.append(f"{item['name']}: {item['snippet']}")

    joined_snippets = "\n".join(snippet_contexts)

    exclude_list_str = ", ".join(sorted(exclude_tickers)) if exclude_tickers else "None"
    prompt_text = (
        f"You are an AI that proposes exactly {num_needed} unique US stock tickers (one per line)\n"
        f"that are likely to rise soon, based on fundamental/technical analysis.\n"
        f"Use the following context from Bing if helpful:\n{joined_snippets}\n\n"
        f"Do NOT include these tickers: {exclude_list_str}\n"
        f"Output only {num_needed} lines, each line is a ticker symbol only, no extra text."
    )

    try:
        messages = [
            {"role": "system", "content": "You are a financial assistant that suggests promising tickers."},
            {"role": "user", "content": prompt_text}
        ]

        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            store=True
        )

        content = completion.choices[0].message.content.strip()

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