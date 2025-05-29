#!/usr/bin/env python3
"""
GA Trading Strategy Optimization and Backtesting

This script:
• Reads a CSV file with 2809 lines having features such as timestamp, open, high, low, close, vwap,
  momentum, atr, obv, bollinger_upper, bollinger_lower, lagged_close_1, lagged_close_2, lagged_close_3,
  lagged_close_5, lagged_close_10, sentiment.
• Splits the data into a training set (first 80%) and a test set (last 20%).
• Uses a genetic algorithm (GA) to optimize three strategy parameters:
    - weight_momentum: weight for the momentum indicator.
    - weight_sentiment: weight for the sentiment indicator.
    - threshold: a threshold value for entering a trade.
• The trading simulation uses these rules:
    - When flat (no position):
         • If signal = weight_momentum * momentum + weight_sentiment * sentiment > threshold, then BUY (go long).
         • If signal < -threshold, then SHORT.
    - When long:
         • If signal falls below 0, then SELL.
         • Immediately after SELL, if signal < -threshold, then open a SHORT.
    - When short:
         • If signal rises above 0, then COVER.
         • Immediately after COVER, if signal > threshold, then open a BUY.
• The simulation uses an “all‐in” approach with an initial capital of $10,000.
• The script outputs two CSV files:
    - "trade_log.csv" – with details for each trade (portfolio total worth, action, shares, price, timestamp, profit/loss).
    - "portfolio_history.csv" – showing the portfolio value over time.
• It also creates a plot (“portfolio_value.png”) of the portfolio value over time.

Required packages:
    pip install numpy pandas matplotlib

Run the script with:
    python ga_trading.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from datetime import datetime
import logging
from dotenv import load_dotenv
import os

# ----------------------------
# Trading simulation functions
# ----------------------------

def simulate_trading(data, params, initial_capital=10000):
    """
    Simulate trading on given data using strategy parameters.
    The strategy:
      - Computes signal = weight_momentum * momentum + weight_sentiment * sentiment.
      - Enters long if signal > threshold, short if signal < -threshold.
      - Exits long if signal < 0; exits short if signal > 0.
      - Checks for immediate reversals.
    
    Returns:
      final_portfolio: final portfolio value.
      trade_log: list of trades (with portfolio_value, action, shares, price, timestamp, profit).
      portfolio_history: list of portfolio values over time.
    """
    weight_momentum = params['weight_momentum']
    weight_sentiment = params['weight_sentiment']
    threshold = params['threshold']
    
    capital = initial_capital
    position = 0    # 1 for long, -1 for short, 0 for none
    shares = 0
    entry_price = None
    entry_capital = None
    
    trade_log = []
    portfolio_history = []
    
    for idx, row in data.iterrows():
        current_time = row['timestamp']
        current_price = row['close']
        # Compute signal
        signal = weight_momentum * row['momentum'] + weight_sentiment * row['sentiment']
        
        # Update portfolio value based on current position
        if position == 1:
            portfolio_value = shares * current_price
        elif position == -1:
            portfolio_value = entry_capital + (entry_price - current_price) * shares
        else:
            portfolio_value = capital
        
        portfolio_history.append({'timestamp': current_time, 'portfolio_value': portfolio_value})
        
        # Trading rules:
        if position == 0:
            if signal > threshold:
                # Enter long
                shares = capital / current_price
                entry_price = current_price
                entry_capital = capital
                position = 1
                capital = 0
                trade_log.append({
                    'portfolio_value': portfolio_value,
                    'action': 'BUY',
                    'shares': shares,
                    'price': current_price,
                    'timestamp': current_time,
                    'profit': 0
                })
            elif signal < -threshold:
                # Enter short
                shares = capital / current_price
                entry_price = current_price
                entry_capital = capital
                position = -1
                capital = 0
                trade_log.append({
                    'portfolio_value': portfolio_value,
                    'action': 'SHORT',
                    'shares': shares,
                    'price': current_price,
                    'timestamp': current_time,
                    'profit': 0
                })
        elif position == 1:
            if signal < 0:
                # Exit long
                sell_price = current_price
                profit = (sell_price - entry_price) * shares
                capital = entry_capital + profit
                trade_log.append({
                    'portfolio_value': portfolio_value,
                    'action': 'SELL',
                    'shares': shares,
                    'price': current_price,
                    'timestamp': current_time,
                    'profit': profit
                })
                position = 0
                shares = 0
                entry_price = None
                entry_capital = None
                # Immediate reversal check: if signal is strongly negative
                if signal < -threshold:
                    shares = capital / current_price
                    entry_price = current_price
                    entry_capital = capital
                    position = -1
                    capital = 0
                    trade_log.append({
                        'portfolio_value': portfolio_value,
                        'action': 'SHORT',
                        'shares': shares,
                        'price': current_price,
                        'timestamp': current_time,
                        'profit': 0
                    })
        elif position == -1:
            if signal > 0:
                # Exit short (COVER)
                cover_price = current_price
                profit = (entry_price - cover_price) * shares
                capital = entry_capital + profit
                trade_log.append({
                    'portfolio_value': portfolio_value,
                    'action': 'COVER',
                    'shares': shares,
                    'price': current_price,
                    'timestamp': current_time,
                    'profit': profit
                })
                position = 0
                shares = 0
                entry_price = None
                entry_capital = None
                # Immediate reversal check: if signal is strongly positive
                if signal > threshold:
                    shares = capital / current_price
                    entry_price = current_price
                    entry_capital = capital
                    position = 1
                    capital = 0
                    trade_log.append({
                        'portfolio_value': portfolio_value,
                        'action': 'BUY',
                        'shares': shares,
                        'price': current_price,
                        'timestamp': current_time,
                        'profit': 0
                    })
    
    # Close any open position at end of data
    final_price = data.iloc[-1]['close']
    final_time = data.iloc[-1]['timestamp']
    if position == 1:
        sell_price = final_price
        profit = (sell_price - entry_price) * shares
        capital = entry_capital + profit
        portfolio_value = shares * final_price
        trade_log.append({
            'portfolio_value': portfolio_value,
            'action': 'SELL',
            'shares': shares,
            'price': final_price,
            'timestamp': final_time,
            'profit': profit
        })
    elif position == -1:
        cover_price = final_price
        profit = (entry_price - cover_price) * shares
        capital = entry_capital + profit
        portfolio_value = capital
        trade_log.append({
            'portfolio_value': portfolio_value,
            'action': 'COVER',
            'shares': shares,
            'price': final_price,
            'timestamp': final_time,
            'profit': profit
        })
    
    final_portfolio = capital if position == 0 else portfolio_value
    return final_portfolio, trade_log, portfolio_history

# ----------------------------
# Cross-Validation Evaluation
# ----------------------------

def evaluate_strategy_cv(params, data, cv=4, initial_capital=10000):
    """
    Evaluate a strategy using cross-validation (walk-forward analysis).
    Splits the data into 'cv' segments, runs the simulation on each,
    and returns the average final portfolio value.
    """
    n = len(data)
    fold_size = n // cv
    fitness_values = []
    for i in range(cv):
        start = i * fold_size
        end = n if i == cv - 1 else (i + 1) * fold_size
        segment = data.iloc[start:end].reset_index(drop=True)
        final_value, _, _ = simulate_trading(segment, params, initial_capital)
        fitness_values.append(final_value)
    return np.mean(fitness_values)

# ----------------------------
# Genetic Algorithm Utilities
# ----------------------------

def create_individual():
    """Generate a random individual (set of strategy parameters)."""
    return {
        'weight_momentum': random.uniform(-10, 10),
        'weight_sentiment': random.uniform(-10, 10),
        'threshold': random.uniform(0.1, 5)
    }

def crossover(parent1, parent2):
    """Simple crossover: average the parameters from both parents."""
    return {
        'weight_momentum': (parent1['weight_momentum'] + parent2['weight_momentum']) / 2,
        'weight_sentiment': (parent1['weight_sentiment'] + parent2['weight_sentiment']) / 2,
        'threshold': (parent1['threshold'] + parent2['threshold']) / 2
    }

def mutate(individual, mutation_rate):
    """Mutate an individual by adding Gaussian noise to each parameter."""
    individual['weight_momentum'] += random.gauss(0, mutation_rate)
    individual['weight_sentiment'] += random.gauss(0, mutation_rate)
    individual['threshold'] += random.gauss(0, mutation_rate/2)
    individual['threshold'] = max(0.1, individual['threshold'])
    return individual

def rank_selection(population, fitnesses):
    """
    Rank-based selection.
    Sort the population by fitness (highest first) and assign probabilities
    proportional to rank. Higher-ranked individuals are more likely to be chosen.
    """
    N = len(population)
    sorted_indices = np.argsort(fitnesses)[::-1]  # indices in descending order
    sorted_population = [population[i] for i in sorted_indices]
    weights = np.array([N - i for i in range(N)], dtype=float)
    probs = weights / weights.sum()
    selected = np.random.choice(sorted_population, p=probs)
    return selected

def hill_climb(individual, data, cv=4, initial_capital=10000, iterations=20, step_size=0.1):
    """
    Apply hill climbing on the individual.
    For a fixed number of iterations, make small adjustments (perturbations)
    and accept the change if it improves fitness.
    """
    best = individual.copy()
    best_fitness = evaluate_strategy_cv(best, data, cv, initial_capital)
    for _ in range(iterations):
        candidate = best.copy()
        candidate['weight_momentum'] += random.gauss(0, step_size)
        candidate['weight_sentiment'] += random.gauss(0, step_size)
        candidate['threshold'] += random.gauss(0, step_size/2)
        candidate['threshold'] = max(0.1, candidate['threshold'])
        candidate_fitness = evaluate_strategy_cv(candidate, data, cv, initial_capital)
        if candidate_fitness > best_fitness:
            best = candidate
            best_fitness = candidate_fitness
    return best

def genetic_algorithm_cv(train_data, population_size=50, generations=30,
                         initial_mutation_rate=0.2, min_mutation_rate=0.05,
                         cv=4, random_injection_rate=0.2, initial_capital=10000):
    """
    Enhanced GA that uses cross-validation for fitness evaluation,
    dynamic mutation rate, rank-based selection, and a final hill climbing step.
    """
    population = [create_individual() for _ in range(population_size)]
    
    for gen in range(generations):
        # Dynamic mutation rate: linearly decrease from initial_mutation_rate to min_mutation_rate
        current_mutation_rate = initial_mutation_rate * (1 - gen / (generations - 1)) + min_mutation_rate
        fitnesses = [evaluate_strategy_cv(ind, train_data, cv, initial_capital) for ind in population]
        
        # Sort population by fitness descending
        sorted_population = [x for _, x in sorted(zip(fitnesses, population), key=lambda pair: pair[0], reverse=True)]
        best_fitness = max(fitnesses)
        print(f"Generation {gen+1}: Best CV Fitness = {best_fitness:.2f}, Mutation Rate = {current_mutation_rate:.3f}")
        
        # Elitism: retain top 10%
        n_elite = max(1, int(0.1 * population_size))
        elite = sorted_population[:n_elite]
        
        # Determine how many random new individuals to inject for exploration
        num_random = int(random_injection_rate * population_size)
        num_children = population_size - n_elite - num_random
        
        children = []
        for _ in range(num_children):
            parent1 = rank_selection(population, fitnesses)
            parent2 = rank_selection(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutate(child, current_mutation_rate)
            children.append(child)
        
        random_individuals = [create_individual() for _ in range(num_random)]
        population = elite + children + random_individuals
    
    # Select the best individual after GA
    best_individual = max(population, key=lambda ind: evaluate_strategy_cv(ind, train_data, cv, initial_capital))
    # Refine best individual using hill climbing (local search)
    best_individual = hill_climb(best_individual, train_data, cv, initial_capital, iterations=20, step_size=0.1)
    best_fitness = evaluate_strategy_cv(best_individual, train_data, cv, initial_capital)
    print(f"\nBest parameters after hill climbing: {best_individual}")
    print(f"Best cross-validated training portfolio value: {best_fitness:.2f}")
    return best_individual

# ----------------------------
# New Functions for External Trade Logic and Backtesting
# ----------------------------

def run_logic(current_price, predicted_price, ticker):
    """
    Runs the Genetic Algorithm on the entire CSV for the given ticker and timeframe,
    then decides the trading action using the best parameters and executes the trade
    by calling functions from forest: buy_shares, sell_shares, short_shares, close_short.
    
    Parameters:
      current_price: The current market price.
      predicted_price: The predicted future price.
      ticker: The stock ticker symbol.
    """
    from forest import api, buy_shares, sell_shares, short_shares, close_short
    load_dotenv()
    
    BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "1Hour")
    TICKERS = os.getenv("TICKERS", "TSLA,AAPL")
    DISABLED_FEATURES = os.getenv("DISABLED_FEATURES", "")
    NEWS_MODE = os.getenv("NEWS_MODE", "on").lower()
    
    timeframe_mapping = {
         "4Hour": "H4",
         "2Hour": "H2",
         "1Hour": "H1",
         "30Min": "M30",
         "15Min": "M15"
    }
    suffix = timeframe_mapping.get(BAR_TIMEFRAME, "H1")
    first_ticker = TICKERS.split(",")[0].strip()
    csv_filename = f"{first_ticker}_{suffix}.csv"
    
    try:
        df = pd.read_csv(csv_filename)
    except Exception as e:
        logging.error(f"Error loading CSV {csv_filename}: {e}")
        return
    
    # Filter out disabled features
    disabled_list = [feat.strip() for feat in DISABLED_FEATURES.split(",") if feat.strip()]
    for col in disabled_list:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Handle NEWS_MODE: if off, set sentiment to 0
    if NEWS_MODE == "off":
        df['sentiment'] = 0.0
    
    # Ensure required columns exist
    required_columns = ['timestamp', 'close', 'momentum']
    if NEWS_MODE != "off":
        required_columns.append('sentiment')
    for col in required_columns:
        if col not in df.columns:
            logging.error(f"Required column {col} missing in CSV")
            return
    
    # Run GA on the entire CSV data
    best_params = genetic_algorithm_cv(df, population_size=50, generations=30,
                                       initial_mutation_rate=0.2, min_mutation_rate=0.05,
                                       cv=4, random_injection_rate=0.2, initial_capital=10000)
    
    # Determine action using the latest candle from the CSV
    latest = df.iloc[-1]
    current_momentum = latest['momentum']
    current_sentiment = latest['sentiment'] if 'sentiment' in latest else 0.0
    signal = best_params['weight_momentum'] * current_momentum + best_params['weight_sentiment'] * current_sentiment
    threshold = best_params['threshold']
    
    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except Exception as e:
        logging.info(f"No open position for {ticker}: {e}")
        position_qty = 0.0
    
    account = api.get_account()
    cash = float(account.cash)
    
    # Decide action based on position and signal
    action = None
    if position_qty == 0:
        if signal > threshold:
            action = 0  # BUY
        elif signal < -threshold:
            action = 2  # SHORT
        else:
            action = 4  # HOLD
    elif position_qty > 0:
        if signal < 0:
            action = 1  # SELL
        else:
            action = 4  # HOLD
    elif position_qty < 0:
        if signal > 0:
            action = 3  # COVER
        else:
            action = 4  # HOLD
    
    # Execute trade based on action (prevent duplicate trades)
    if action == 0 and position_qty <= 0:
        max_shares = int(cash // current_price)
        logging.info("buy")
        buy_shares(ticker, max_shares, current_price, predicted_price)
    elif action == 1 and position_qty > 0:
        logging.info("sell")
        sell_shares(ticker, position_qty, current_price, predicted_price)
    elif action == 2 and position_qty >= 0:
        max_shares = int(cash // current_price)
        logging.info("short")
        short_shares(ticker, max_shares, current_price, predicted_price)
    elif action == 3 and position_qty < 0:
        qty_to_close = abs(position_qty)
        logging.info("cover")
        close_short(ticker, qty_to_close, current_price)
    # Action 4 is HOLD; do nothing

def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    """
    Runs the Genetic Algorithm using training data from the CSV (filtered up to current_timestamp)
    and then decides the trading action using the best parameters.
    
    Parameters:
      current_price: The current market price.
      predicted_price: The predicted future price.
      position_qty: The current position quantity.
      current_timestamp: The timestamp for the current candle.
      candles: A DataFrame of all the candles that will be backtested (not used for training).
      
    Returns:
      A string representing the trade action: "BUY", "SELL", "SHORT", "COVER", or "NONE".
    """
    load_dotenv()
    
    BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "1Hour")
    TICKERS = os.getenv("TICKERS", "TSLA,AAPL")
    DISABLED_FEATURES = os.getenv("DISABLED_FEATURES", "")
    NEWS_MODE = os.getenv("NEWS_MODE", "on").lower()
    
    timeframe_mapping = {
         "4Hour": "H4",
         "2Hour": "H2",
         "1Hour": "H1",
         "30Min": "M30",
         "15Min": "M15"
    }
    suffix = timeframe_mapping.get(BAR_TIMEFRAME, "H1")
    first_ticker = TICKERS.split(",")[0].strip()
    csv_filename = f"{first_ticker}_{suffix}.csv"
    
    try:
        df = pd.read_csv(csv_filename)
    except Exception as e:
        logging.error(f"Error loading CSV {csv_filename}: {e}")
        return "NONE"
    
    # Filter out disabled features
    disabled_list = [feat.strip() for feat in DISABLED_FEATURES.split(",") if feat.strip()]
    for col in disabled_list:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Handle NEWS_MODE: if off, set sentiment to 0
    if NEWS_MODE == "off":
        df['sentiment'] = 0.0
    
    # Ensure required columns exist
    required_columns = ['timestamp', 'close', 'momentum']
    if NEWS_MODE != "off":
        required_columns.append('sentiment')
    for col in required_columns:
        if col not in df.columns:
            logging.error(f"Required column {col} missing in CSV")
            return "NONE"
    
    # Convert timestamp column to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    current_timestamp_dt = pd.to_datetime(current_timestamp)
    # Filter training data: all candles up to the current timestamp
    train_data = df[df['timestamp'] <= current_timestamp_dt]
    if train_data.empty:
        logging.error("No training data available up to the current timestamp")
        return "NONE"
    
    # Run GA on the training data from the CSV
    best_params = genetic_algorithm_cv(train_data, population_size=50, generations=30,
                                       initial_mutation_rate=0.2, min_mutation_rate=0.05,
                                       cv=4, random_injection_rate=0.2, initial_capital=10000)
    
    # Determine action using the latest candle in the training data
    latest = train_data.iloc[-1]
    current_momentum = latest['momentum']
    current_sentiment = latest['sentiment'] if 'sentiment' in latest else 0.0
    signal = best_params['weight_momentum'] * current_momentum + best_params['weight_sentiment'] * current_sentiment
    threshold = best_params['threshold']
    
    action = None
    if position_qty == 0:
        if signal > threshold:
            action = 0  # BUY
        elif signal < -threshold:
            action = 2  # SHORT
        else:
            action = 4  # HOLD
    elif position_qty > 0:
        if signal < 0:
            action = 1  # SELL
        else:
            action = 4  # HOLD
    elif position_qty < 0:
        if signal > 0:
            action = 3  # COVER
        else:
            action = 4  # HOLD
    
    if action == 0 and position_qty <= 0:
        return "BUY"
    elif action == 1 and position_qty > 0:
        return "SELL"
    elif action == 2 and position_qty >= 0:
        return "SHORT"
    elif action == 3 and position_qty < 0:
        return "COVER"
    else:
        return "NONE"