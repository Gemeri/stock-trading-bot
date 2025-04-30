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
    - When long:
         • If signal falls below 0, then SELL.
    - No short selling or covering.
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
      - Enters long if signal > threshold, exits long if signal < 0.
      - No short selling.
    
    Returns:
      final_portfolio: final portfolio value.
      trade_log: list of trades (with portfolio_value, action, shares, price, timestamp, profit).
      portfolio_history: list of portfolio values over time.
    """
    weight_momentum = params['weight_momentum']
    weight_sentiment = params['weight_sentiment']
    threshold = params['threshold']
    
    capital = initial_capital
    position = 0    # 1 for long, 0 for none
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
    
    final_portfolio = capital
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
    """Rank-based selection."""
    N = len(population)
    sorted_indices = np.argsort(fitnesses)[::-1]
    sorted_population = [population[i] for i in sorted_indices]
    weights = np.array([N - i for i in range(N)], dtype=float)
    probs = weights / weights.sum()
    selected = np.random.choice(sorted_population, p=probs)
    return selected

def hill_climb(individual, data, cv=4, initial_capital=10000, iterations=20, step_size=0.1):
    """Apply hill climbing on the individual."""
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
    """Enhanced GA that uses cross-validation for fitness evaluation..."""  

    population = [create_individual() for _ in range(population_size)]
    for gen in range(generations):
        current_mutation_rate = initial_mutation_rate * (1 - gen / (generations - 1)) + min_mutation_rate
        fitnesses = [evaluate_strategy_cv(ind, train_data, cv, initial_capital) for ind in population]
        sorted_population = [x for _, x in sorted(zip(fitnesses, population), key=lambda pair: pair[0], reverse=True)]
        best_fitness = max(fitnesses)
        print(f"Generation {gen+1}: Best CV Fitness = {best_fitness:.2f}, Mutation Rate = {current_mutation_rate:.3f}")
        n_elite = max(1, int(0.1 * population_size))
        elite = sorted_population[:n_elite]
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
    best_individual = max(population, key=lambda ind: evaluate_strategy_cv(ind, train_data, cv, initial_capital))
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
    Runs the Genetic Algorithm... then decides trading action without short selling.
    """
    from forest import api, buy_shares, sell_shares
    load_dotenv()
    BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "1Hour")
    TICKERS = os.getenv("TICKERS", "TSLA,AAPL")
    DISABLED_FEATURES = os.getenv("DISABLED_FEATURES", "")
    NEWS_MODE = os.getenv("NEWS_MODE", "on").lower()
    timeframe_mapping = {"4Hour": "H4", "2Hour": "H2", "1Hour": "H1", "30Min": "M30", "15Min": "M15"}
    suffix = timeframe_mapping.get(BAR_TIMEFRAME, "H1")
    first_ticker = TICKERS.split(",")[0].strip()
    csv_filename = f"{first_ticker}_{suffix}.csv"
    try:
        df = pd.read_csv(csv_filename)
    except Exception as e:
        logging.error(f"Error loading CSV {csv_filename}: {e}")
        return
    disabled_list = [feat.strip() for feat in DISABLED_FEATURES.split(",") if feat.strip()]
    for col in disabled_list:
        if col in df.columns:
            df = df.drop(columns=[col])
    if NEWS_MODE == "off":
        df['sentiment'] = 0.0
    required_columns = ['timestamp', 'close', 'momentum']
    if NEWS_MODE != "off":
        required_columns.append('sentiment')
    for col in required_columns:
        if col not in df.columns:
            logging.error(f"Required column {col} missing in CSV")
            return
    best_params = genetic_algorithm_cv(df, population_size=50, generations=30,
                                       initial_mutation_rate=0.2, min_mutation_rate=0.05,
                                       cv=4, random_injection_rate=0.2, initial_capital=10000)
    latest = df.iloc[-1]
    current_momentum = latest['momentum']
    current_sentiment = latest.get('sentiment', 0.0)
    signal = best_params['weight_momentum'] * current_momentum + best_params['weight_sentiment'] * current_sentiment
    threshold = best_params['threshold']
    from forest.api import get_position, get_account
    try:
        pos = api.get_position(ticker)
        position_qty = float(pos.qty)
    except:
        position_qty = 0.0
    account = api.get_account()
    cash = float(account.cash)
    action = None
    if position_qty == 0:
        if signal > threshold:
            action = 'BUY'
        else:
            action = 'HOLD'
    elif position_qty > 0:
        if signal < 0:
            action = 'SELL'
        else:
            action = 'HOLD'
    if action == 'BUY' and position_qty <= 0:
        max_shares = int(cash // current_price)
        buy_shares(ticker, max_shares, current_price, predicted_price)
    elif action == 'SELL' and position_qty > 0:
        sell_shares(ticker, position_qty, current_price, predicted_price)


def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    """
    Runs the GA using training data... then decides action without short selling.
    """
    load_dotenv()
    BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "1Hour")
    TICKERS = os.getenv("TICKERS", "TSLA,AAPL")
    DISABLED_FEATURES = os.getenv("DISABLED_FEATURES", "")
    NEWS_MODE = os.getenv("NEWS_MODE", "on").lower()
    timeframe_mapping = {"4Hour": "H4", "2Hour": "H2", "1Hour": "H1", "30Min": "M30", "15Min": "M15"}
    suffix = timeframe_mapping.get(BAR_TIMEFRAME, "H1")
    first_ticker = TICKERS.split(",")[0].strip()
    csv_filename = f"{first_ticker}_{suffix}.csv"
    try:
        df = pd.read_csv(csv_filename)
    except Exception as e:
        logging.error(f"Error loading CSV {csv_filename}: {e}")
        return "NONE"
    disabled_list = [feat.strip() for feat in DISABLED_FEATURES.split(",") if feat.strip()]
    for col in disabled_list:
        if col in df.columns:
            df = df.drop(columns=[col])
    if NEWS_MODE == "off":
        df['sentiment'] = 0.0
    required_columns = ['timestamp', 'close', 'momentum']
    if NEWS_MODE != "off":
        required_columns.append('sentiment')
    for col in required_columns:
        if col not in df.columns:
            logging.error(f"Required column {col} missing in CSV")
            return "NONE"
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    current_timestamp_dt = pd.to_datetime(current_timestamp)
    train_data = df[df['timestamp'] <= current_timestamp_dt]
    if train_data.empty:
        logging.error("No training data available up to the current timestamp")
        return "NONE"
    best_params = genetic_algorithm_cv(train_data, population_size=50, generations=30,
                                       initial_mutation_rate=0.2, min_mutation_rate=0.05,
                                       cv=4, random_injection_rate=0.2, initial_capital=10000)
    latest = train_data.iloc[-1]
    current_momentum = latest['momentum']
    current_sentiment = latest.get('sentiment', 0.0)
    signal = best_params['weight_momentum'] * current_momentum + best_params['weight_sentiment'] * current_sentiment
    threshold = best_params['threshold']
    action = None
    if position_qty == 0:
        if signal > threshold:
            action = "BUY"
        else:
            action = "NONE"
    elif position_qty > 0:
        if signal < 0:
            action = "SELL"
        else:
            action = "NONE"
    return action
