import pandas as pd
import numpy as np

tp_percent=0.03
sl_percent=0.015
horizon=9

def build_labels(df, action_type):
    labels = []
    prices = df['close'].values
    
    for i in range(len(prices)):
        # If we don't have enough candles left for a full horizon, label as 0 (Wait)
        if i + horizon >= len(prices):
            labels.append(0)
            continue
            
        entry_price = prices[i]
        
        # Define boundaries based on action type
        if action_type == 1: # BUY logic
            upper_barrier = entry_price * (1 + tp_percent)
            lower_barrier = entry_price * (1 - sl_percent)
        else: # SELL logic
            upper_barrier = entry_price * (1 + sl_percent) # Risk (Price goes up)
            lower_barrier = entry_price * (1 - tp_percent) # Profit (Price goes down)
            
        triggered = 0 # Default to 'Wait'
        
        # Look ahead up to 9 candles
        for j in range(1, horizon + 1):
            future_price = prices[i + j]
            
            if action_type == 1: # For BUY
                if future_price >= upper_barrier:
                    triggered = 1 # Profit hit! Execute now.
                    break
                elif future_price <= lower_barrier:
                    triggered = 0 # Loss hit or neutral. Wait.
                    break
            else: # For SELL
                if future_price <= lower_barrier:
                    triggered = 1 # Profit hit (price dropped)! Execute now.
                    break
                elif future_price >= upper_barrier:
                    triggered = 0 # Loss hit. Wait.
                    break
                    
        labels.append(triggered)
        
    df['label'] = labels
    return df