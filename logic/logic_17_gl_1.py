import os
import sys
import time
import logging
import random
import copy
import pickle
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
from bot.trading.orders import buy_shares, sell_shares


import numpy as np
import pandas as pd
import logic.tools as tools

from dotenv import load_dotenv

# === 1. Logging Setup ===
logger = logging.getLogger("ga_rl_trade_logic")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# === 2. Environment Variables/Csv Handling ===

load_dotenv()
def get_csv_filename(ticker):
    return tools.get_csv_filename(ticker)


def load_csv_df(
    ticker: str,
    upto_timestamp: Optional[datetime] = None
) -> pd.DataFrame:
    fname = get_csv_filename(ticker)
    if not os.path.exists(fname):
        logger.error(f"Historical CSV not found: {fname}")
        raise FileNotFoundError(f"No historical csv for {ticker}")
    df = pd.read_csv(fname)
    # Enforce correct timestamp parsing
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    if upto_timestamp is not None:
        mask = df["timestamp"] <= pd.to_datetime(upto_timestamp)
        df = df.loc[mask]
    return df.reset_index(drop=True)


# === 3. GA Optimizer ===

class GAOptimizer:

    def __init__(self,
                 feature_cols: List[str],
                 population_size: int = 60,
                 n_generations: int = 25,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.02,
                 random_seed: int = 42):
        """
        population_size: >= 50
        n_generations: >= 20
        """
        self.feature_cols = feature_cols
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)
        self.best_params = None

    def _init_population(self) -> List[np.ndarray]:
        # E.g. W shape: (n_features, )
        pop = [np.random.uniform(-2, 2, size=len(self.feature_cols))
               for _ in range(self.population_size)]
        return pop

    def _evaluate_individual(self,
                            individual: np.ndarray,
                            df: pd.DataFrame) -> Tuple[float, Dict]:
        cash = 100000
        position = 0
        entry_price = 0
        equity_curve = []
        n_trades = 0
        max_drawdown = 0
        peak = cash
        last_action = 0  # 0: NONE, 1: BUY, 2: SELL

        for i, row in df.iterrows():
            state = row[self.feature_cols].values.astype(np.float32)
            signal = np.dot(individual, state)
            # Map to discrete action. You can use thresholds here or sign(signal)
            if signal > 1.0:
                action = 1  # BUY
            elif signal < -1.0:
                action = 2  # SELL
            else:
                action = 0  # NONE

            price = row["close"]

            # Simple position policy: 1 unit max
            if action == 1 and position == 0:
                entry_price = price
                position = 1
                n_trades += 1
            elif action == 2 and position == 1:
                # Close long
                cash += (price - entry_price)
                position = 0
                n_trades += 1
            # else, do nothing

            net_equity = cash
            if position == 1:
                net_equity += price - entry_price
            equity_curve.append(net_equity)
            if net_equity > peak:
                peak = net_equity
            drawdown_cur = (peak - net_equity)
            if drawdown_cur > max_drawdown:
                max_drawdown = drawdown_cur

        profit = equity_curve[-1] - 100000
        fitness = profit - 0.5*max_drawdown - 0.1*n_trades  # Drawdown penalty
        metrics = {
            "net_profit": profit,
            "n_trades": n_trades,
            "max_drawdown": max_drawdown,
        }
        return fitness, metrics

    def _select_parents(self, population: List[np.ndarray], fitnesses: List[float]) -> List[np.ndarray]:
        # Tournament selection
        selected = []
        for _ in range(self.population_size):
            i, j = random.sample(range(len(population)), 2)
            winner = population[i] if fitnesses[i] > fitnesses[j] else population[j]
            selected.append(winner)
        return selected

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, len(parent1)-1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1.copy(), parent2.copy()

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_rate:
                individual[i] += np.random.normal(0, 0.5)
        return individual

    def optimize(self, df: pd.DataFrame) -> Dict:
        logger.info("GAOptimizer: Starting evolution...")
        population = self._init_population()
        best_fitness = float('-inf')
        best_individual = None
        best_metrics = None

        for gen in range(self.n_generations):
            fitnesses = []
            infos = []
            for ind in population:
                fit, metrics = self._evaluate_individual(ind, df)
                fitnesses.append(fit)
                infos.append(metrics)
            # Track best
            idx = int(np.argmax(fitnesses))
            if fitnesses[idx] > best_fitness:
                best_fitness = fitnesses[idx]
                best_individual = population[idx].copy()
                best_metrics = infos[idx]
            logger.info(f"GA Gen {gen+1}/{self.n_generations} | Best Fit: {best_fitness:.2f} "
                        f"| Net: {best_metrics['net_profit']:.2f} Dd: {best_metrics['max_drawdown']:.2f} Trades: {best_metrics['n_trades']}")

            # Selection
            parents = self._select_parents(population, fitnesses)
            # Crossover & Mutation
            next_gen = []
            for i in range(0, self.population_size, 2):
                p1, p2 = parents[i], parents[i+1]
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                next_gen.extend([c1, c2])
            population = next_gen[:self.population_size]

        self.best_params = best_individual
        logger.info(f"GAOptimizer: Done. Best NetProfit={best_metrics['net_profit']:.2f} "
                    f"Drawdown={best_metrics['max_drawdown']:.2f}")
        return {"params": best_individual, "metrics": best_metrics}


# === 4. RL Environment (Gym-Style) ===

class RLEnvironment:
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], random_seed: int = 42, window_size:int = 1):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.window_size = window_size  # for possible stacking
        self.random_seed = random_seed

        self.action_space = {0: "NONE", 1: "BUY", 2: "SELL"}
        self.reset()
    
    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.position = 0    # 0: flat, 1: long
        self.cash = 100000
        self.entry_price = 0
        self.done = False
        self.n_trades = 0
        self.equity_curve = []
        # Initial state
        state = self._get_state()
        return state

    def _get_state(self) -> np.ndarray:
        idx = self.current_step
        if idx >= len(self.df):
            idx = len(self.df)-1
        state = self.df.loc[idx, self.feature_cols].values.astype(np.float32)
        # Optionally, extend with position/cash info
        state = np.concatenate([state, [self.position]])
        return state

    def step(self, action:int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.done:
            return self._get_state(), 0.0, True, {}
        cur_row = self.df.iloc[self.current_step]
        price = cur_row["close"]
        reward = 0.0
        info = {}

        if action == 1 and self.position == 0:
            # Buy
            self.entry_price = price
            self.position = 1
            self.n_trades += 1
            reward -= 0.02  # penalty for trading activity
        elif action == 2 and self.position == 1:
            # Sell/Close
            step_reward = price - self.entry_price
            reward += step_reward
            self.cash += step_reward
            self.position = 0
            self.n_trades += 1
            self.entry_price = 0
        elif action == 0:
            # NONE
            reward -= 0.001  # Minor penalty to avoid inactivity if desired

        # optional: penalize drawdown or losing streaks
        net_equity = self.cash
        if self.position == 1:
            net_equity += price - self.entry_price
        self.equity_curve.append(net_equity)
        if len(self.equity_curve) > 10:
            dd = max(self.equity_curve[-10:]) - net_equity
            reward -= 0.01 * dd  # penalize drawdown

        self.current_step += 1
        if self.current_step >= len(self.df):  # End of episode
            self.done = True
        return self._get_state(), reward, self.done, info


# === 5. RL Agent ===

class RLAgent:
    def __init__(self, feature_cols: List[str], seed_params: np.ndarray, n_actions: int = 3, use_mlp: bool = False):
        self.feature_cols = feature_cols
        self.n_actions = n_actions
        self.use_mlp = use_mlp
        self.seed_params = seed_params
        self.model = None  # placeholder; for MLP, use torch.nn.Module

        if not use_mlp:
            # Simple linear Q(state, action) = w_action^T * state
            # We'll create (n_actions, n_features) weight matrix, seed action=1 with GA weights
            self.weights = np.random.uniform(-1,1, (n_actions, len(feature_cols) + 1))  # +1 for position info
            self.weights[1, :] = np.concatenate([seed_params, [1]])  # action 1=BUY gets GA weights
        # For real RL, TO-DO: set up Q-network, optim, etc.
        # Optionally, add epsilon-greedy, replay buffer, etc.

    def train(self, df: pd.DataFrame, n_episodes:int = 100, batch_size:int = 32, gamma:float = 0.99) -> "RLAgent":
        # For full implementation, plug in RLlib, stable_baselines, or torch.
        logger.info("RLAgent: Training (toy algo)...")
        env = RLEnvironment(df, self.feature_cols)
        total_rewards = []
        lr = 0.001  # Learning rate for linear Q (toy)

        for episode in range(n_episodes):
            obs = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                q_values = np.dot(self.weights, obs)
                action = int(np.argmax(q_values))  # Greedy
                if np.random.rand() < max(0.1, 0.9*(1 - episode / n_episodes)):  # Epsilon
                    action = np.random.choice([0,1,2])
                next_obs, reward, done, _ = env.step(action)
                # Update Q: Q(s,a) += lr * (r + gamma * max_a' Q(s',a') - Q(s,a))
                target = reward + gamma * np.max(np.dot(self.weights, next_obs))
                td_error = target - np.dot(self.weights[action,:], obs)
                self.weights[action,:] += lr * td_error * obs
                obs = next_obs
                total_reward += reward
            total_rewards.append(total_reward)

            if (episode+1) % 10 == 0:
                logger.info(f"RLAgent: Episode {episode+1}/{n_episodes}, Reward: {total_reward:.2f}")
        # Save best policy if reward improves
        self.model = self.weights.copy()
        logger.info(f"RLAgent: Training done. Mean reward (last 10): {np.mean(total_rewards[-10:]):.2f}")
        return self  # For chaining

    def act(self, state: np.ndarray) -> str:
        if self.model is not None:
            q_values = np.dot(self.model, state)
            action = int(np.argmax(q_values))
        else:
            # Default to simple threshold
            signal = np.dot(self.seed_params, state[:-1])
            if signal > 1.0:
                action = 1
            elif signal < -1.0:
                action = 2
            else:
                action = 0
        action_map = {0: "NONE", 1: "BUY", 2: "SELL"}
        return action_map[action]


# === 6. Core Entrypoints ===

def _get_feature_cols(df: pd.DataFrame) -> List[str]:
    # As per column order, remove ['timestamp','close','predicted_close'] only
    skip = {"timestamp", "close"}
    feature_cols = [c for c in df.columns if c not in skip]
    # Make sure predicted_close is retained, as it's a feature not a target
    return feature_cols

def run_backtest(current_price: float,
                 predicted_price: float,
                 position_qty: float,
                 current_timestamp: datetime,
                 candles: pd.DataFrame,
                 ticker) -> str:
    try:
        logger.info(f"Backtest: {current_timestamp} cur_price={current_price:.2f} pred={predicted_price:.2f} pos={position_qty}")

        # 1) Load all historical csv data UP TO current_timestamp (no lookahead!)
        hist_df = load_csv_df(ticker, upto_timestamp=current_timestamp)
        # features
        feature_cols = _get_feature_cols(hist_df)

        # 2) GA: Optimize feature weights over historical walkforward data
        ga = GAOptimizer(feature_cols, population_size=60, n_generations=25)
        ga_result = ga.optimize(hist_df)
        ga_best_params = ga_result["params"]
        # 3) RLAgent: train seeded on best GA params
        rl_agent = RLAgent(feature_cols, seed_params=ga_best_params)
        rl_agent.train(hist_df, n_episodes=100)
        # 4) Infer: current state for agent (build full vector)
        state_values = []
        # Map features using latest row (candles.iloc[-1]), but ensure same order
        cur_idx = candles["timestamp"].idxmax()
        cur_row = candles.iloc[cur_idx] if cur_idx is not None else candles.iloc[-1]
        for col in feature_cols:
            if col == "position_qty":
                state_values.append(position_qty)
            elif col == "predicted_close":
                state_values.append(predicted_price)
            else:
                val = cur_row.get(col, np.nan)
                if pd.isnull(val):
                    # fallback
                    state_values.append(0.0)
                else:
                    state_values.append(val)
        state_ext = np.array(state_values + [position_qty])
        # 5) Output agent action
        action = rl_agent.act(state_ext)
        logger.info(f"Backtest: RL decision={action}")
        return action
    except Exception as e:
        logger.error(f"run_backtest error: {str(e)}", exc_info=True)
        return "NONE"


def run_logic(current_price: float,
              predicted_price: float,
              ticker: str) -> None:
    try:
        logger.info(f"Live Logic: {ticker} cur={current_price:.2f}, pred={predicted_price:.2f}")
        # 1) Load ALL history
        hist_df = load_csv_df(ticker)
        feature_cols = _get_feature_cols(hist_df)

        # 2) GA
        ga = GAOptimizer(feature_cols, population_size=60, n_generations=25)
        ga_result = ga.optimize(hist_df)
        ga_best_params = ga_result["params"]
        # 3) RLAgent
        rl_agent = RLAgent(feature_cols, seed_params=ga_best_params)
        rl_agent.train(hist_df, n_episodes=120)

        # 4) Build state (assume live: get last row)
        last_row = hist_df.iloc[-1]
        state_values = []
        for col in feature_cols:
            if col == "predicted_close":
                state_values.append(float(predicted_price))
            else:
                val = last_row.get(col, np.nan)
                if pd.isnull(val):
                    state_values.append(0.0)
                else:
                    state_values.append(val)
        # For position, try to get from API if possible:
        try:
            from forest import api
            position_qty = float(api.get_positions(ticker)[0]["qty"])
        except Exception:
            position_qty = 0.0
        state_ext = np.array(state_values + [position_qty])
        # 5) Action
        action = rl_agent.act(state_ext)
        logger.info(f"Live Logic: Model action = {action}")
        # 6) Trade execution
        try:
            from forest import api
            cash = api.get_cash()
            logger.info(f"Live: Current cash {cash}, pos {position_qty}")
            if action == "BUY":
                buy_shares(ticker, size=1)
                logger.info(f"BUY executed: {ticker}")
            elif action == "SELL":
                sell_shares(ticker, size=1)
                logger.info(f"SELL executed: {ticker}")
            else:
                logger.info("No trade executed.")
        except Exception as api_exc:
            logger.error(f"Error executing trade: {api_exc}", exc_info=True)
        # Optionally, persist best model
        try:
            with open(f"{ticker}_rl_agent.pkl", "wb") as f:
                pickle.dump(rl_agent, f)
        except Exception as persist_exc:
            logger.error(f"Error saving RL agent: {persist_exc}", exc_info=True)
    except Exception as e:
        logger.error(f"run_logic error: {str(e)}", exc_info=True)