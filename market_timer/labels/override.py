from __future__ import annotations

import numpy as np
import pandas as pd

horizon = 9
epsilon = 0.002
price_col = "close"
use_next_candle_price = True
next_price_col = "open"
require_full_horizon = True
out_prefix = ""
opportunity_horizon = None
min_opportunity = 0.0

def add_execute_until_deterioration_labels(
    df: pd.DataFrame,
    action_flag: int,              # 1 = BUY, 0 = SELL
) -> pd.DataFrame:
    
    # --------- validation ----------
    if action_flag not in (0, 1):
        raise ValueError("action_flag must be 0 (SELL) or 1 (BUY).")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if epsilon < 0:
        raise ValueError("epsilon must be >= 0")
    if price_col not in df.columns:
        raise KeyError(f"df is missing required column: {price_col!r}")
    if use_next_candle_price and next_price_col not in df.columns:
        raise KeyError(f"df is missing required column: {next_price_col!r}")
    if opportunity_horizon is not None and opportunity_horizon < 1:
        raise ValueError("opportunity_horizon must be >= 1 when provided.")
    if min_opportunity < 0:
        raise ValueError("min_opportunity must be >= 0")

    out = df.copy()

    # --------- execution price proxy p_t ----------
    exec_price = out[next_price_col].shift(-1) if use_next_candle_price else out[price_col]
    out[f"{out_prefix}exec_price"] = exec_price

    # Rolling window length is H+1 for inclusive [t..t+H]
    window = horizon + 1
    min_periods = window if require_full_horizon else 1

    # Forward-looking rolling best via reverse-rolling trick
    rev = exec_price.iloc[::-1]

    if action_flag == 1:  # BUY
        fwd_best = rev.rolling(window=window, min_periods=min_periods).min().iloc[::-1]
        wait_improvement = (exec_price - fwd_best) / exec_price
    else:  # SELL
        fwd_best = rev.rolling(window=window, min_periods=min_periods).max().iloc[::-1]
        wait_improvement = (fwd_best - exec_price) / exec_price

    out[f"{out_prefix}fwd_best_{horizon}"] = fwd_best
    out[f"{out_prefix}wait_improvement_{horizon}"] = wait_improvement

    # Base label: execute if waiting can't improve enough
    label_bool = wait_improvement <= epsilon

    if require_full_horizon:
        valid = exec_price.notna() & fwd_best.notna() & wait_improvement.notna()
        out[f"{out_prefix}label_execute"] = np.where(valid, label_bool.astype(int), np.nan)
    else:
        out[f"{out_prefix}label_execute"] = label_bool.astype(int)

    # --------- optional: opportunity filter ----------
    if opportunity_horizon is not None:
        M = opportunity_horizon
        win2 = M + 1
        min_periods2 = win2 if require_full_horizon else 1

        rev2 = exec_price.iloc[::-1]

        if action_flag == 1:  # BUY: need upside opportunity after execution
            fwd_max_M = rev2.rolling(window=win2, min_periods=min_periods2).max().iloc[::-1]
            opportunity = (fwd_max_M - exec_price) / exec_price
        else:  # SELL: need downside opportunity after execution
            fwd_min_M = rev2.rolling(window=win2, min_periods=min_periods2).min().iloc[::-1]
            opportunity = (exec_price - fwd_min_M) / exec_price

        out[f"{out_prefix}opportunity_{M}"] = opportunity

        opp_ok = opportunity >= min_opportunity

        if require_full_horizon:
            valid2 = out[f"{out_prefix}label_execute"].notna() & opportunity.notna()
            out[f"{out_prefix}label_execute_final"] = np.where(
                valid2,
                (out[f"{out_prefix}label_execute"].astype(float).astype(int).values == 1) & opp_ok.values,
                np.nan,
            ).astype("float")
            # Convert True/False to 1/0 while preserving NaN
            out[f"{out_prefix}label_execute_final"] = out[f"{out_prefix}label_execute_final"].where(
                out[f"{out_prefix}label_execute_final"].isna(),
                out[f"{out_prefix}label_execute_final"].astype(int),
            )
        else:
            out[f"{out_prefix}label_execute_final"] = (
                (out[f"{out_prefix}label_execute"] == 1) & opp_ok
            ).astype(int)

    return out