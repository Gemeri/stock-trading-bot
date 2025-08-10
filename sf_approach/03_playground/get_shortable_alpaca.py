import alpaca_trade_api as tradeapi
import concurrent.futures
from config import API_KEY, API_SECRET, API_BASE_URL

# Minimum market cap in dollars (e.g. 2 billion)
MIN_MARKET_CAP = 2_000_000_000
MAX_PRICE = 60

api = tradeapi.REST(API_KEY, API_SECRET, API_BASE_URL)

def fetch_asset_data(asset):
    symbol = asset.symbol
    try:
        # Get the latest bar (1 day) using get_bars (limit=1)
        bars = api.get_bars(symbol, tradeapi.TimeFrame.Day, limit=1).df

        if bars.empty:
            return None
        price = bars['close'][-1]

        asset_info = api.get_asset(symbol)

        print(f"{{symbol}}: Price ${price:.2f}, Market Cap ${asset_info.market_cap:,}")

        if price <= MAX_PRICE and asset_info.market_cap is not None and asset_info.market_cap >= MIN_MARKET_CAP:
            print(f"---> {{symbol}}: Price ${price:.2f}, Market Cap ${asset_info.market_cap:,}")
            return {
                'symbol': symbol,
                'price': price,
                'market_cap': asset_info.market_cap
            }
    except Exception as e:
        print(f"error: {e}")
        return None
    return None

def main():
    active_assets = api.list_assets(status='active')

    # Filter assets shortable and marginable first
    candidates = [a for a in active_assets if a.shortable and a.marginable]

    print(f"candidates: {len(candidates)}")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fetch_asset_data, asset) for asset in candidates]
        for future in concurrent.futures.as_completed(futures):
            data = future.result()
            if data:
                results.append(data)

    # Sort by market cap descending
    results.sort(key=lambda x: x['market_cap'], reverse=True)

    for r in results:
        print(f"{r['symbol']}: Price ${r['price']:.2f}, Market Cap ${r['market_cap']:,}")

if __name__ == '__main__':
    main()