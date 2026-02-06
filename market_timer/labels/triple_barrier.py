"""
triple_barrier.py

Drop-in replacement for the old script:
- Keeps the SAME public function signature and behavior:
    build_labels(df, action_type) -> df (with df["label"] added)
- Internally uses the EXACT quantile-based triple-barrier label implementation you provided.
"""

import numpy as np
import pandas as pd

# Match old default horizon (old script used horizon=9)
HORIZON = 9

# === EXACT LABEL SETTINGS (from your provided code) ===
QUANTILE_WINDOW = 32  # number of past samples used to estimate tp/sl quantiles
Q_TP = 0.70           # take-profit quantile of "future max return" distribution
Q_SL = 0.20           # stop-loss quantile of "future min return" distribution (negative)

# Reasonable bounds to prevent crazy barriers (expressed as fractions, e.g. 0.03 = 3%)
MIN_TP = 0.005
MAX_TP = 0.08
MIN_SL = 0.003
MAX_SL = 0.05


def _safe_prices_close(df: pd.DataFrame) -> np.ndarray:
    if "close" not in df.columns:
        raise ValueError("DataFrame missing required column: 'close'")
    prices = pd.to_numeric(df["close"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    prices = prices.ffill().bfill()
    if prices.isna().any():
        raise ValueError("close contains NaNs even after ffill/bfill; cannot label safely.")
    return prices.to_numpy(dtype=float)


def _build_future_extreme_returns(prices: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """
    For each i, compute:
      max_ret[i] = max_{j=1..h} (price[i+j]/price[i] - 1)
      min_ret[i] = min_{j=1..h} (price[i+j]/price[i] - 1)

    Last `horizon` rows are NaN.
    """
    n = len(prices)
    max_ret = np.full(n, np.nan, dtype=float)
    min_ret = np.full(n, np.nan, dtype=float)

    for i in range(n - horizon):
        entry = prices[i]
        fw = prices[i + 1 : i + horizon + 1]
        rets = fw / entry - 1.0
        max_ret[i] = float(np.max(rets))
        min_ret[i] = float(np.min(rets))

    return max_ret, min_ret


def _rolling_quantile_past_only(arr: np.ndarray, window: int, q: float) -> np.ndarray:
    """
    Rolling quantile using ONLY past values:
      out[i] = quantile(arr[i-window : i], q)
    So index i uses data strictly before i.
    """
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)

    for i in range(n):
        start = max(0, i - window)
        # strictly past:
        past = arr[start:i]
        past = past[np.isfinite(past)]
        if past.size < max(30, int(0.25 * min(window, i))):  # require some minimum history
            continue
        out[i] = float(np.quantile(past, q))
    return out


def build_labels_quantile_triple_barrier(
    df: pd.DataFrame,
    action_type: int,
    horizon: int = HORIZON,
    q_window: int = QUANTILE_WINDOW,
    q_tp: float = Q_TP,
    q_sl: float = Q_SL,
    min_tp: float = MIN_TP,
    max_tp: float = MAX_TP,
    min_sl: float = MIN_SL,
    max_sl: float = MAX_SL,
) -> pd.DataFrame:
    """
    Builds a dynamic triple-barrier label using rolling quantiles of forward-return extremes.

    label = 1 if (profit barrier hit first within horizon) else 0.
    (same style as your earlier triple-barrier label)
    """
    if action_type not in (0, 1):
        raise ValueError("action_type must be 0 (SELL) or 1 (BUY)")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if q_window < 30:
        raise ValueError("q_window should be >= 30 for stable quantiles")
    if not (0.0 < q_tp < 1.0) or not (0.0 < q_sl < 1.0):
        raise ValueError("q_tp and q_sl must be in (0, 1)")

    out = df.copy()
    prices = _safe_prices_close(out)

    # Precompute future extreme returns (uses future, but ONLY to define the label)
    max_ret, min_ret = _build_future_extreme_returns(prices, horizon=horizon)

    # Rolling quantiles of those extremes using PAST data only (no leakage)
    tp_dyn = _rolling_quantile_past_only(max_ret, window=q_window, q=q_tp)
    sl_dyn_raw = _rolling_quantile_past_only(min_ret, window=q_window, q=q_sl)

    # sl_dyn_raw is typically negative; convert to positive magnitude
    sl_dyn = np.abs(sl_dyn_raw)

    # Clamp to sane bounds
    tp_dyn = np.clip(tp_dyn, min_tp, max_tp)
    sl_dyn = np.clip(sl_dyn, min_sl, max_sl)

    n = len(prices)
    labels = np.zeros(n, dtype=int)

    # We can only label when:
    # - enough future horizon exists
    # - enough history exists to define tp/sl (tp_dyn & sl_dyn finite)
    for i in range(n):
        if i + horizon >= n:
            labels[i] = 0
            continue
        if not (np.isfinite(tp_dyn[i]) and np.isfinite(sl_dyn[i])):
            # not enough history -> no label signal; default 0
            labels[i] = 0
            continue

        entry = prices[i]
        tp = float(tp_dyn[i])
        sl = float(sl_dyn[i])

        if action_type == 1:  # BUY
            upper = entry * (1.0 + tp)
            lower = entry * (1.0 - sl)
        else:  # SELL
            # SELL profits when price drops by tp, risks when price rises by sl
            upper = entry * (1.0 + sl)  # risk
            lower = entry * (1.0 - tp)  # profit

        triggered = 0
        for j in range(1, horizon + 1):
            fp = prices[i + j]
            if action_type == 1:  # BUY
                if fp >= upper:
                    triggered = 1
                    break
                if fp <= lower:
                    triggered = 0
                    break
            else:  # SELL
                if fp <= lower:
                    triggered = 1
                    break
                if fp >= upper:
                    triggered = 0
                    break

        labels[i] = triggered

    out["label"] = labels
    out["tp_dyn"] = tp_dyn
    out["sl_dyn"] = sl_dyn
    return out


# === Drop-in wrapper: SAME inputs/outputs as the OLD script ===
def build_labels(df: pd.DataFrame, action_type: int) -> pd.DataFrame:
    """
    Old signature + old output behavior:
      - inputs: (df, action_type)
      - output: returns df with ONLY df['label'] added (no extra columns)

    action_type:
      1 = BUY, 0 = SELL
    """
    labeled = build_labels_quantile_triple_barrier(df, action_type, horizon=HORIZON)

    # Mutate/return like the old script: only attach 'label'
    df["label"] = labeled["label"].to_numpy(dtype=int)
    return df
