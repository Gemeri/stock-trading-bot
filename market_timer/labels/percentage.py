import pandas as pd
import numpy as np

window=9
threshold_pct=0.5

def build_labels(df, action_type):
    # Validate inputs
    if action_type not in [0, 1]:
        raise ValueError("action_type must be 0 (SELL) or 1 (BUY)")
    
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")
    
    # Create a copy to avoid modifying original
    df_labeled = df.copy().reset_index(drop=True)
    
    # Initialize new columns
    df_labeled['label'] = None
    df_labeled['optimal_price'] = np.nan
    df_labeled['price_diff_pct'] = np.nan
    df_labeled['candles_to_optimal'] = np.nan
    
    # Process each candle (except last 'window' candles which don't have enough forward data)
    for i in range(len(df_labeled) - window):
        current_price = df_labeled.loc[i, 'close']
        
        # Define the forward-looking window
        future_start = i + 1
        future_end = i + window
        future_window = df_labeled.loc[future_start:future_end]
        
        if action_type == 1:  # BUY action - look for lowest price
            # Find the lowest low in the forward window
            optimal_price = future_window['low'].min()
            optimal_idx = future_window['low'].idxmin()
            
            # Calculate percentage difference (positive if current is higher than optimal)
            price_diff_pct = ((current_price - optimal_price) / optimal_price) * 100
            
            # Label as EXECUTE if current price is within threshold of the lowest price
            if price_diff_pct <= threshold_pct:
                label = 'EXECUTE'
            else:
                label = 'WAIT'
        
        else:  # action_type == 0: SELL action - look for highest price
            # Find the highest high in the forward window
            optimal_price = future_window['high'].max()
            optimal_idx = future_window['high'].idxmax()
            
            # Calculate percentage difference (positive if optimal is higher than current)
            price_diff_pct = ((optimal_price - current_price) / current_price) * 100
            
            # Label as EXECUTE if current price is within threshold of the highest price
            if price_diff_pct <= threshold_pct:
                label = 'EXECUTE'
            else:
                label = 'WAIT'
        
        # Calculate how many candles ahead the optimal price occurs
        candles_to_optimal = optimal_idx - i
        
        # Assign values
        df_labeled.loc[i, 'label'] = label
        df_labeled.loc[i, 'optimal_price'] = optimal_price
        df_labeled.loc[i, 'price_diff_pct'] = price_diff_pct
        df_labeled.loc[i, 'candles_to_optimal'] = candles_to_optimal
    
    # Handle the last 'window' candles (no future data available)
    # Mark them as NaN or 'UNKNOWN'
    last_candles_start = len(df_labeled) - window
    df_labeled.loc[last_candles_start:, 'label'] = 'UNKNOWN'
    
    return df_labeled


# Example usage and testing
if __name__ == "__main__":
    # Create sample TSLA-like data for demonstration
    np.random.seed(42)
    n_candles = 100
    
    # Generate synthetic price data
    base_price = 250
    price_changes = np.random.randn(n_candles).cumsum() * 2
    
    sample_df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_candles, freq='4H'),
        'open': base_price + price_changes + np.random.randn(n_candles) * 0.5,
        'high': base_price + price_changes + np.random.randn(n_candles) * 0.5 + 2,
        'low': base_price + price_changes + np.random.randn(n_candles) * 0.5 - 2,
        'close': base_price + price_changes,
        'volume': np.random.randint(1000000, 5000000, n_candles)
    })
    
    # Ensure high is highest and low is lowest
    sample_df['high'] = sample_df[['open', 'high', 'low', 'close']].max(axis=1)
    sample_df['low'] = sample_df[['open', 'high', 'low', 'close']].min(axis=1)
    
    print("=" * 80)
    print("EXAMPLE 1: BUY ACTION (action_type=1)")
    print("=" * 80)
    df_buy = build_labels(sample_df, action_type=1, window=9, threshold_pct=0.5)
    print("\nFirst 20 rows with labels:")
    print(df_buy[['close', 'optimal_price', 'price_diff_pct', 'candles_to_optimal', 'label']].head(20))
    
    print("\n" + "=" * 80)
    print("EXAMPLE 2: SELL ACTION (action_type=0)")
    print("=" * 80)
    df_sell = build_labels(sample_df, action_type=0, window=9, threshold_pct=0.5)
    print("\nFirst 20 rows with labels:")
    print(df_sell[['close', 'optimal_price', 'price_diff_pct', 'candles_to_optimal', 'label']].head(20))
    
    print("\n" + "=" * 80)
    print("LABEL DISTRIBUTION")
    print("=" * 80)
    print("\nBUY labels:")
    print(df_buy['label'].value_counts())
    print("\nSELL labels:")
    print(df_sell['label'].value_counts())
    
    print("\n" + "=" * 80)
    print("STATISTICS FOR 'EXECUTE' LABELS")
    print("=" * 80)
    print("\nBUY - Average candles to optimal when EXECUTE:")
    execute_buy = df_buy[df_buy['label'] == 'EXECUTE']
    if len(execute_buy) > 0:
        print(f"  Mean: {execute_buy['candles_to_optimal'].mean():.2f} candles")
        print(f"  Median: {execute_buy['candles_to_optimal'].median():.2f} candles")
    
    print("\nSELL - Average candles to optimal when EXECUTE:")
    execute_sell = df_sell[df_sell['label'] == 'EXECUTE']
    if len(execute_sell) > 0:
        print(f"  Mean: {execute_sell['candles_to_optimal'].mean():.2f} candles")
        print(f"  Median: {execute_sell['candles_to_optimal'].median():.2f} candles")