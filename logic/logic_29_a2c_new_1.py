# logic.py
import os
import math
import numpy as np
import pandas as pd

# If using python-dotenv, uncomment:
# from dotenv import load_dotenv
# load_dotenv()

from sklearn.ensemble import RandomForestRegressor


###############################################################################
#                       A2C-STYLE RL IMPLEMENTATION (MINIMAL)                 #
###############################################################################
class A2CAgent:
    """
    A minimal A2C-like implementation (not production-ready).
    In real usage, consider stable-baselines or a more robust framework.
    """

    def __init__(
        self,
        state_size,
        action_size=5,
        gamma=0.99,
        actor_lr=0.001,
        critic_lr=0.005,
        none_penalty=0.001,
    ):
        """
        :param state_size: dimensionality of input (features + position, etc.)
        :param action_size: 5 actions: BUY(0), SELL(1), SHORT(2), COVER(3), NONE(4)
        :param gamma: discount factor
        :param actor_lr: learning rate for policy (actor)
        :param critic_lr: learning rate for value function (critic)
        :param none_penalty: penalty for taking "NONE" action to encourage trading
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.none_penalty = none_penalty

        # Initialize random weights for actor/critic (simple linear for demo).
        self.actor_w = np.random.randn(state_size, action_size) * 0.01
        self.critic_w = np.random.randn(state_size, 1) * 0.01

    def _softmax(self, logits):
        """
        Safely convert logits to float array, then apply softmax.
        Adds an epsilon to avoid dividing by zero, and ensures no NaNs left.
        """
        logits = np.array(logits, dtype=float, ndmin=1)
        # Subtract max for numerical stability
        shifted = logits - np.max(logits)
        ex = np.exp(shifted)
        sum_ex = np.sum(ex)

        # Avoid divide-by-zero: if sum_ex is extremely small, fallback to uniform
        if sum_ex < 1e-12:
            probs = np.ones_like(ex) / len(ex)
        else:
            probs = ex / sum_ex

        # Convert any leftover NaNs or inf to safe numbers
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

        # Finally, if due to rounding probs don't sum to 1, re-normalize
        total = np.sum(probs)
        if total < 1e-12:
            # fallback to uniform again
            probs = np.ones_like(probs) / len(probs)
        else:
            probs /= total

        return probs

    def predict_value(self, state):
        """
        Critic: estimate value of current state V(s).
        """
        state = np.array(state, dtype=float)
        return state.dot(self.critic_w)[0]

    def predict_policy(self, state):
        """
        Actor: compute probabilities over actions (softmax of linear logits).
        """
        state = np.array(state, dtype=float)
        logits = state.dot(self.actor_w)  # shape = (action_size,)
        return self._softmax(logits)

    def choose_action(self, state):
        """
        Sample an action from the policy distribution.
        """
        state = np.array(state, dtype=float)
        probs = self.predict_policy(state)  # ensures no NaN
        action = np.random.choice(len(probs), p=probs)  # can raise ValueError if probs has NaN
        return action, probs

    def update(self, trajectory):
        """
        A2C update: For each step in the trajectory, do one-step advantage update.
        trajectory is a list of (state, action, reward, next_state, done).
        """
        for (state, action, reward, next_state, done) in trajectory:
            state = np.array(state, dtype=float)
            next_state = np.array(next_state, dtype=float)

            state_value = self.predict_value(state)
            if done:
                target = reward
            else:
                next_value = self.predict_value(next_state)
                target = reward + self.gamma * next_value

            advantage = target - state_value

            # Actor update (policy gradient)
            probs = self.predict_policy(state)
            dlog_policy = -probs
            dlog_policy[action] += 1.0  # derivative of log(pi(a|s)) wrt action

            grad_actor = np.outer(state, dlog_policy * advantage)
            self.actor_w += self.actor_lr * grad_actor

            # Critic update (value function)
            grad_critic = np.outer(state, advantage)
            self.critic_w += self.critic_lr * grad_critic


###############################################################################
#                       HELPER / UTILITY FUNCTIONS                            #
###############################################################################
def convert_timeframe(tf_str):
    """
    Convert timeframe from e.g. '4Hour' to 'H4', '2Hour' to 'H2', etc.
    """
    mapping = {
        "4Hour": "H4",
        "2Hour": "H2",
        "1Hour": "H1",
        "30Min": "M30",
        "15Min": "M15",
    }
    return mapping.get(tf_str, tf_str)

def load_env_vars():
    """
    Load environment variables (BAR_TIMEFRAME, TICKERS, DISABLED_FEATURES).
    """
    bar_timeframe = os.environ.get("BAR_TIMEFRAME", "4Hour")
    tickers = os.environ.get("TICKERS", "TSLA,AAPL")
    disabled_features = os.environ.get("DISABLED_FEATURES", "")
    tickers_list = [t.strip() for t in tickers.split(",") if t.strip()]
    disabled_list = [f.strip() for f in disabled_features.split(",") if f.strip()]
    return bar_timeframe, tickers_list, disabled_list

def load_csv_data(ticker, timeframe_suffix):
    """
    Load CSV (ticker_timeframe.csv), sort by timestamp if present.
    """
    filename = f"{ticker}_{timeframe_suffix}.csv"
    df = pd.read_csv(filename)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
    df.reset_index(drop=True, inplace=True)
    return df

def filter_features(df, disabled_features, predicted_col="predicted_price"):
    """
    Remove disabled features. Ensure 'predicted_price' column exists.
    """
    all_cols = set(df.columns)
    to_drop = [c for c in all_cols if c in disabled_features]
    df_filtered = df.drop(columns=to_drop, errors="ignore")

    if predicted_col not in df_filtered.columns:
        df_filtered[predicted_col] = np.nan

    return df_filtered

def compute_reward(trade_pnl_series, risk_free=0.0):
    """
    Example reward: Sharpe-like ratio: (mean - rf) / stdev
    """
    if len(trade_pnl_series) < 2:
        return -0.1
    rets = np.array(trade_pnl_series)
    mean_r = rets.mean()
    std_r = rets.std() if rets.std() != 0 else 1e-9
    sharpe = (mean_r - risk_free) / std_r
    return sharpe


###############################################################################
#               Global RL Agent & RandomForest Model                          #
###############################################################################
GLOBAL_RL_AGENT = None
GLOBAL_RF_MODEL = None
TRAINED_DATA_CACHE = None

def train_rl_model(df, none_penalty=0.001):
    """
    Train (or update) the A2C RL model over the entire DataFrame once.
    """
    global GLOBAL_RL_AGENT

    drop_cols = ["timestamp"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    state_size = len(feature_cols) + 1  # +1 for position
    action_size = 5

    if (GLOBAL_RL_AGENT is None) or (GLOBAL_RL_AGENT.state_size != state_size):
        GLOBAL_RL_AGENT = A2CAgent(
            state_size=state_size,
            action_size=action_size,
            gamma=0.99,
            actor_lr=0.001,
            critic_lr=0.005,
            none_penalty=none_penalty,
        )

    trajectory = []
    position_qty = 0
    avg_price = 0
    trade_pnls = []

    for i in range(len(df) - 1):
        current_row = df.iloc[i].fillna(0.0)
        next_row = df.iloc[i + 1].fillna(0.0)

        feats = current_row[feature_cols].values
        state = np.concatenate([feats, [position_qty]], axis=0)

        action, _ = GLOBAL_RL_AGENT.choose_action(state)

        current_price = current_row.get("close", current_row.get("open", 0.0))
        next_price = next_row.get("close", next_row.get("open", 0.0))

        reward = 0.0
        done = False

        # Naive position logic
        if action == 0:  # BUY
            if position_qty <= 0:
                if position_qty < 0:
                    realized_pnl = (avg_price - current_price) * abs(position_qty)
                    trade_pnls.append(realized_pnl)
                position_qty = 1
                avg_price = current_price

        elif action == 1:  # SELL
            if position_qty > 0:
                realized_pnl = (current_price - avg_price) * position_qty
                trade_pnls.append(realized_pnl)
                position_qty = 0
                avg_price = 0

        elif action == 2:  # SHORT
            if position_qty >= 0:
                if position_qty > 0:
                    realized_pnl = (current_price - avg_price) * position_qty
                    trade_pnls.append(realized_pnl)
                position_qty = -1
                avg_price = current_price

        elif action == 3:  # COVER
            if position_qty < 0:
                realized_pnl = (avg_price - current_price) * abs(position_qty)
                trade_pnls.append(realized_pnl)
                position_qty = 0
                avg_price = 0

        elif action == 4:  # NONE
            reward -= GLOBAL_RL_AGENT.none_penalty

        # Unrealized PnL shaping
        if position_qty > 0:
            reward += (next_price - current_price) * position_qty
        elif position_qty < 0:
            reward += (current_price - next_price) * abs(position_qty)

        feats_next = next_row[feature_cols].values
        next_state = np.concatenate([feats_next, [position_qty]], axis=0)

        if i + 1 == len(df) - 1:
            done = True

        trajectory.append((state, action, reward, next_state, done))

    # Single update
    GLOBAL_RL_AGENT.update(trajectory)

    final_reward = compute_reward(trade_pnls)
    # print("train_rl_model => final reward:", final_reward)
    return GLOBAL_RL_AGENT

def maybe_train_random_forest(df):
    """
    Optionally train a RandomForest to predict next close.
    """
    global GLOBAL_RF_MODEL
    if "close" not in df.columns:
        return

    df["target_next_close"] = df["close"].shift(-1)
    df.dropna(subset=["target_next_close"], inplace=True)

    exclude_cols = ["timestamp", "predicted_price", "target_next_close"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].fillna(0.0).values
    y = df["target_next_close"].values

    if GLOBAL_RF_MODEL is None:
        GLOBAL_RF_MODEL = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)

    GLOBAL_RF_MODEL.fit(X, y)

###############################################################################
#                run_logic: LIVE / REAL-TIME TRADING LOGIC                    #
###############################################################################
def run_logic(current_price, predicted_price, ticker):
    """
    Called in real-time. 
    1) Load CSV for (ticker + timeframe).
    2) Filter disabled features.
    3) Possibly train RL if data changed.
    4) Decide among buy_shares, sell_shares, short_shares, close_short, or none.
    """
    from forest import api, buy_shares, sell_shares, short_shares, close_short

    global TRAINED_DATA_CACHE

    bar_timeframe, tickers_list, disabled_list = load_env_vars()
    tf_suffix = convert_timeframe(bar_timeframe)

    df = load_csv_data(ticker, tf_suffix)
    df_filtered = filter_features(df, disabled_list, predicted_col="predicted_price")

    maybe_train_random_forest(df_filtered)

    # Overwrite predicted_price w/ RF if available
    if GLOBAL_RF_MODEL is not None:
        feature_cols = [c for c in df_filtered.columns if c not in ["timestamp","predicted_price","target_next_close"]]
        X_all = df_filtered[feature_cols].fillna(0.0).values
        pred_all = GLOBAL_RF_MODEL.predict(X_all)
        df_filtered["predicted_price"] = pred_all

    # Then override last row's predicted_price with user-supplied
    df_filtered.at[len(df_filtered) - 1, "predicted_price"] = predicted_price

    if TRAINED_DATA_CACHE is None or not TRAINED_DATA_CACHE.equals(df_filtered):
        train_rl_model(df_filtered)
        TRAINED_DATA_CACHE = df_filtered.copy()

    last_row = df_filtered.iloc[-1].fillna(0.0)
    drop_cols = ["timestamp"]
    feature_cols = [c for c in df_filtered.columns if c not in drop_cols]

    pos = api.get_position(ticker)
    position_qty = float(pos.qty)

    state_vec = last_row[feature_cols].values
    state = np.concatenate([state_vec, [position_qty]], axis=0)

    action, _ = GLOBAL_RL_AGENT.choose_action(state)

    # Convert to actual trade function
    action_name = "NONE"
    if action == 0:  # BUY
        if position_qty <= 0:
            action_name = "BUY"
            buy_shares(ticker, 1)
    elif action == 1:  # SELL
        if position_qty > 0:
            action_name = "SELL"
            sell_shares(ticker, abs(int(position_qty)))
    elif action == 2:  # SHORT
        if position_qty >= 0:
            action_name = "SHORT"
            short_shares(ticker, 1)
    elif action == 3:  # COVER
        if position_qty < 0:
            action_name = "COVER"
            close_short(ticker, abs(int(position_qty)))
    # else: NONE

    return


###############################################################################
#                run_backtest: OFFLINE / BACKTEST LOGIC                       #
###############################################################################
def run_backtest(current_price, predicted_price, position_qty, current_timestamp, candles):
    """
    Returns "BUY", "SELL", "SHORT", "COVER", or "NONE" for backtesting.
    - Use first TICKER from .env
    - Use same RL + RF logic
    - 'candles' can be an int or a DataFrame (we use len if DF).
    """
    global TRAINED_DATA_CACHE

    # If candles is a DataFrame, interpret its length
    if isinstance(candles, pd.DataFrame):
        candles = len(candles)
    else:
        candles = int(candles)

    bar_timeframe, tickers_list, disabled_list = load_env_vars()
    tf_suffix = convert_timeframe(bar_timeframe)

    if len(tickers_list) < 1:
        return "NONE"
    first_ticker = tickers_list[0]

    df = load_csv_data(first_ticker, tf_suffix)
    df_filtered = filter_features(df, disabled_list, predicted_col="predicted_price")

    maybe_train_random_forest(df_filtered)
    if GLOBAL_RF_MODEL is not None:
        feature_cols = [c for c in df_filtered.columns if c not in ["timestamp","predicted_price","target_next_close"]]
        X_all = df_filtered[feature_cols].fillna(0.0).values
        pred_all = GLOBAL_RF_MODEL.predict(X_all)
        df_filtered["predicted_price"] = pred_all

    # Overwrite row's predicted_price for matching timestamp
    if "timestamp" in df_filtered.columns:
        mask = df_filtered["timestamp"] == pd.to_datetime(current_timestamp)
        if mask.any():
            df_filtered.loc[mask, "predicted_price"] = predicted_price

    # Possibly train RL
    if TRAINED_DATA_CACHE is None or not TRAINED_DATA_CACHE.equals(df_filtered):
        train_rl_model(df_filtered)
        TRAINED_DATA_CACHE = df_filtered.copy()

    # Restrict to last `candles` bars
    if candles < len(df_filtered):
        df_backtest = df_filtered.iloc[-candles:]
    else:
        df_backtest = df_filtered.copy()

    # Identify the row for current_timestamp or last row
    if "timestamp" in df_backtest.columns:
        mask2 = df_backtest["timestamp"] == pd.to_datetime(current_timestamp)
        if mask2.any():
            action_row = df_backtest.loc[mask2]
        else:
            action_row = df_backtest.iloc[[-1]]
    else:
        action_row = df_backtest.iloc[[-1]]

    drop_cols = ["timestamp"]
    feature_cols = [c for c in df_filtered.columns if c not in drop_cols]

    row_for_state = action_row.iloc[0].fillna(0.0)
    state_vec = row_for_state[feature_cols].values
    state = np.concatenate([state_vec, [position_qty]], axis=0)

    action, probs = GLOBAL_RL_AGENT.choose_action(state)

    # If any leftover numeric instability, check for NaN in `probs` (debug):
    # if np.isnan(probs).any():
    #     print("DEBUG: probs contained NaN!", probs)

    action_name = "NONE"
    if action == 0:  # BUY
        if position_qty <= 0:
            action_name = "BUY"
    elif action == 1:  # SELL
        if position_qty > 0:
            action_name = "SELL"
    elif action == 2:  # SHORT
        if position_qty >= 0:
            action_name = "SHORT"
    elif action == 3:  # COVER
        if position_qty < 0:
            action_name = "COVER"
    else:
        action_name = "NONE"

    return action_name
