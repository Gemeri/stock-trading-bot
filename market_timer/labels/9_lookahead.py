import numpy as np
import pandas as pd

lookahead = 9
close_col = "close"
label_col = "label"

def build_labels(
    df: pd.DataFrame,
    action_type: int,
) -> pd.DataFrame:

    if action_type not in (0, 1):
        raise ValueError("action_type must be 0 (highest close) or 1 (lowest close)")
    if close_col not in df.columns:
        raise KeyError(f"'{close_col}' column not found in df")

    n = len(df)
    closes = df[close_col].to_numpy(dtype=float, copy=False)
    labels = np.zeros(n, dtype=np.int8)

    for i in range(n):
        start = i + 1
        if start >= n:
            break

        end = min(n - 1, i + lookahead)  # inclusive
        window = closes[start:end + 1]

        # If window is all-NaN, skip
        if window.size == 0 or np.all(np.isnan(window)):
            continue

        # Pick index within the window (ties -> first occurrence)
        j = np.nanargmax(window) if action_type == 0 else np.nanargmin(window)
        labels[start + j] = 1

    out = df.copy()
    out[label_col] = labels
    return out
