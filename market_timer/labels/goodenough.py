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

def add_good_enough_band_labels(
    df: pd.DataFrame,
    action_type: int,
) -> pd.DataFrame:

    if action_type not in (0, 1):
        raise ValueError("action_type must be 0 (SELL) or 1 (BUY)")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if epsilon < 0:
        raise ValueError("epsilon must be >= 0")

    if price_col not in df.columns:
        raise KeyError(f"df is missing required column: {price_col!r}")
    if use_next_candle_price and next_price_col not in df.columns:
        raise KeyError(f"df is missing required column: {next_price_col!r}")

    out = df.copy()

    # Execution price proxy p_t
    exec_price = out[next_price_col].shift(-1) if use_next_candle_price else out[price_col]
    out[f"{out_prefix}exec_price"] = exec_price

    window = horizon + 1
    min_periods = window if require_full_horizon else 1

    # Forward-looking rolling best via reverse-rolling trick
    rev = exec_price.iloc[::-1]

    if action_type == 1:  # BUY
        fwd_best = rev.rolling(window=window, min_periods=min_periods).min().iloc[::-1]
        label_bool = exec_price <= (fwd_best * (1.0 + epsilon))
    else:  # SELL
        fwd_best = rev.rolling(window=window, min_periods=min_periods).max().iloc[::-1]
        label_bool = exec_price >= (fwd_best * (1.0 - epsilon))

    out[f"{out_prefix}fwd_best_{horizon}"] = fwd_best

    if require_full_horizon:
        valid = exec_price.notna() & fwd_best.notna()
        out[f"{out_prefix}label_execute"] = np.where(valid, label_bool.astype(int), np.nan)
    else:
        out[f"{out_prefix}label_execute"] = label_bool.astype(int)

    return out