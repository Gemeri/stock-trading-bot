from forest import HORIZON, POSSIBLE_FEATURE_COLS
import pandas as pd
import numpy as np

def _ensure_multi_horizon_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create a ternary target column for the configured horizon if missing."""
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain 'close' column to build targets.")

    close = df["close"].astype(float)

    col = f"target_{HORIZON}"
    if col in df.columns:
        return df  # already present, don't overwrite

    fwd_ret = close.shift(-HORIZON) / close - 1.0

    # Rolling quantiles over the past 32 values of fwd_ret
    rolling_low = (
        fwd_ret.rolling(32, min_periods=32)
                .quantile(0.33)
    )
    rolling_high = (
        fwd_ret.rolling(32, min_periods=32)
                .quantile(0.67)
    )

    target = np.zeros(len(df), dtype=int)

    valid = (~fwd_ret.isna()) & (~rolling_low.isna()) & (~rolling_high.isna())
    target[(fwd_ret > rolling_high) & valid] = 1
    target[(fwd_ret < rolling_low) & valid] = -1

    df[col] = target

    return df

def _get_feature_and_target_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return available feature columns and ensured target column names."""
    # make sure target_* columns are present (created in-memory if needed)
    df = _ensure_multi_horizon_targets(df)

    feature_cols = [c for c in POSSIBLE_FEATURE_COLS if c in df.columns]

    target_cols: list[str] = []
    col = f"target_{HORIZON}"
    if col in df.columns:
        target_cols.append(col)

    return feature_cols, target_cols