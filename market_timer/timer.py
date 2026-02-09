import importlib
import json
from pathlib import Path
from typing import Callable, Tuple, Optional
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# -------------------- CONFIG --------------------

USE_STATIC_THRESHOLDS = False

# Fix #2: rolling window
USE_ROLLING_WINDOW = True
TRAIN_WINDOW = 2000          # tune 1500–3000
MIN_TRAIN_BARS = 240         # skip training if not enough history

# Fix #3: threshold smoothing
USE_THRESHOLD_SMOOTHING = True
THRESH_SMOOTH_ALPHA = 0.15   # EWMA smoothing factor (0.10–0.25 typical)

# Threshold tuning objective:
#   "trade"   -> your current weighted Fbeta - lambda*FPR
#   "accuracy"-> pure accuracy on calibration set
THRESH_OBJECTIVE = "recall_fpr"   # <-- new option

TARGET_FPR = 0.20        # start 0.15–0.25
MIN_PRECISION = 0.35     # prevents degenerate spam
MAX_POS_RATE_MULT = 1.15 # predicted-1 rate <= 1.15 * base_rate
MIN_POS_RATE_FLOOR = 0.02
# Hybrid settings
HYBRID_ALPHA_TPR = 0.70       # weight on catching 1s (0.6–0.8 good range)
HYBRID_MIN_PRECISION = 0.35   # prevent spam-1 behavior
HYBRID_MAX_FPR = 0.18         # keep 0-label performance meaningful
HYBRID_MIN_RECALL = 0.18      # prevent all-0 collapse
HYBRID_MAX_POS_RATE_MULT = 1.25  # cap predicted positives to ~1.1–1.4x base rate
HYBRID_MIN_POS_RATE_FLOOR = 0.02  # at least 2% positives predicted (or 1 sample)

# Threshold tuning defaults (for "trade")
THRESH_BETA = 2.0
THRESH_LAMBDA_FPR = 0.9
MIN_PRECISION = 0.0

# Fix #4: episode-state features (computed causally inside rolling window)
EP_FEATURES = [
    "bars_since_episode_start",
    "pct_from_episode_start",
    "min_since_episode_start",
    "pct_from_min_since_start",
]

# IMPORTANT: prevent label leakage / feature mismatches when label builders mutate df
EXCLUDE_PREFIXES = ("label",)
EXCLUDE_EXACT = {
    "optimal_price",
    "price_diff_pct",
    "candles_to_optimal",
    "label_execute",
    "label_execute_final",
}

TCN_RETRAIN = 10

# ------------------------------------------------


def _is_rl_model(model_name: str) -> bool:
    return model_name.lower() in {"rl"}


def _direction_to_str(direction: int) -> str:
    return "BUY" if direction == 1 else "SELL"


def _get_label_builder(label: int) -> Tuple[Callable[[pd.DataFrame, int], pd.DataFrame], str]:
    if label == 1:
        module = importlib.import_module("market_timer.labels.goodenough")
        return module.build_labels, "label_execute"
    if label == 2:
        module = importlib.import_module("market_timer.labels.percentage")
        return module.build_labels, "label"
    if label == 3:
        module = importlib.import_module("market_timer.labels.triple_barrier")
        return module.build_labels, "label"
    raise ValueError("label must be an int between 1 and 5")


def _normalize_labels(y: pd.Series) -> pd.Series:
    if y.dtype == object:
        y = y.map({"EXECUTE": 1, "WAIT": 0, "UNKNOWN": np.nan})
    y = pd.to_numeric(y, errors="coerce")
    return y

def _best_threshold_recall_at_fpr(
    probas: np.ndarray,
    y_true: pd.Series,
    target_fpr: float = TARGET_FPR,
    min_precision: float = MIN_PRECISION,
    max_pos_rate_mult: float = MAX_POS_RATE_MULT,
    min_pos_rate_floor: float = MIN_POS_RATE_FLOOR,
) -> float:
    if probas.size == 0:
        return 0.5

    y = np.asarray(y_true, dtype=int)
    base_rate = float(np.mean(y == 1))
    n = len(y)

    if base_rate <= 0.0:
        return 0.99

    max_pos_rate = min(0.85, max_pos_rate_mult * base_rate)
    min_pos_rate = max(1.0 / n, min_pos_rate_floor)

    qs = np.linspace(0.01, 0.99, 250)
    grid = np.unique(np.quantile(probas, qs))
    grid = np.clip(grid, 1e-6, 1 - 1e-6)

    best_t = float(np.quantile(probas, 1.0 - base_rate))  # fallback ~match base rate
    best_recall = -1.0
    best_tnr = -1.0

    for t in grid:
        pred = (probas >= t).astype(int)
        pos_rate = float(pred.mean())
        if pos_rate < min_pos_rate or pos_rate > max_pos_rate:
            continue

        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        tnr = tn / max(tn + fp, 1)

        if precision < min_precision:
            continue
        if fpr > target_fpr:
            continue

        # primary objective: maximize recall
        # tie-breaker: maximize TNR (keeps 0s important)
        if (recall > best_recall) or (recall == best_recall and tnr > best_tnr):
            best_recall = float(recall)
            best_tnr = float(tnr)
            best_t = float(t)

    return best_t


def _best_threshold_hybrid(
    probas: np.ndarray,
    y_true: pd.Series,
    alpha_tpr: float = HYBRID_ALPHA_TPR,
    min_precision: float = HYBRID_MIN_PRECISION,
    max_fpr: float = HYBRID_MAX_FPR,
    min_recall: float = HYBRID_MIN_RECALL,
    max_pos_rate_mult: float = HYBRID_MAX_POS_RATE_MULT,
    min_pos_rate_floor: float = HYBRID_MIN_POS_RATE_FLOOR,
) -> float:
    """
    Hybrid threshold: prioritize recall (TPR) but preserve 0-label quality via FPR/precision caps.
    Maximizes: score = alpha*TPR + (1-alpha)*TNR
    subject to constraints.
    """
    if probas.size == 0:
        return 0.5

    y = np.asarray(y_true, dtype=int)
    base_rate = float(np.mean(y == 1))
    n = len(y)

    if base_rate <= 0.0:
        return 0.99

    # Bound predicted positive rate to avoid "spam 1s" or "all 0s"
    max_pos_rate = min(0.85, max_pos_rate_mult * base_rate)
    min_pos_rate = max(1.0 / n, min_pos_rate_floor, 0.25 * base_rate)

    qs = np.linspace(0.01, 0.99, 250)
    grid = np.unique(np.quantile(probas, qs))
    grid = np.clip(grid, 1e-6, 1 - 1e-6)

    best_t = 0.5
    best_score = -1e18

    for t in grid:
        pred = (probas >= t).astype(int)
        pos_rate = float(pred.mean())
        if pos_rate < min_pos_rate or pos_rate > max_pos_rate:
            continue

        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)   # TPR
        tnr = tn / max(tn + fp, 1)      # TNR
        fpr = fp / max(fp + tn, 1)

        if precision < min_precision:
            continue
        if recall < min_recall:
            continue
        if fpr > max_fpr:
            continue

        score = alpha_tpr * recall + (1.0 - alpha_tpr) * tnr

        if score > best_score:
            best_score = score
            best_t = float(t)

    # Fallback: choose threshold roughly matching base rate, but ensure at least 1 positive prediction
    if best_score == -1e18:
        t = float(np.quantile(probas, 1.0 - base_rate))
        t = float(np.clip(t, 1e-6, 1 - 1e-6))
        if float((probas >= t).mean()) < (1.0 / len(probas)):
            t = float(np.max(probas) - 1e-6)
        return t

    return best_t


def _infer_feature_cols(df: pd.DataFrame) -> list[str]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    out: list[str] = []
    for c in numeric_cols:
        cl = str(c).lower()
        if cl.startswith(EXCLUDE_PREFIXES):
            continue
        if c in EXCLUDE_EXACT:
            continue
        out.append(c)
    if not out:
        raise ValueError("No numeric feature columns available in df after exclusions.")
    return out


def _prepare_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing required feature cols: {missing}")
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.ffill().fillna(0.0)
    return X


def _infer_price_col(df: pd.DataFrame) -> str:
    if "open" in df.columns:
        return "open"
    if "close" in df.columns:
        return "close"
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No numeric columns found to infer a price column.")
    return numeric_cols[0]


def _exec_prices_next_bar(df: pd.DataFrame, price_col: str) -> pd.Series:
    p = pd.to_numeric(df[price_col], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().bfill()
    exec_prices = p.shift(-1)
    if len(exec_prices) > 0:
        exec_prices.iloc[-1] = float(p.iloc[-1])
    return exec_prices


def _add_episode_state_features(X: pd.DataFrame, exec_prices: pd.Series) -> pd.DataFrame:
    """
    Adds Fix #4 features causally relative to the START of the current rolling window.
    This is leakage-safe and stable:
      - baseline is the first price in the slice
      - running min uses cummin (past-only)
    """
    if len(X) != len(exec_prices):
        raise ValueError(f"Episode feature length mismatch: X={len(X)} vs exec_prices={len(exec_prices)}")

    ep = exec_prices.astype(float).to_numpy()
    ep = np.where(np.isfinite(ep), ep, np.nan)

    if len(ep) == 0:
        for c in EP_FEATURES:
            X[c] = 0.0
        return X

    baseline = ep[0]
    if not np.isfinite(baseline) or baseline == 0.0:
        baseline = float(np.nanmean(ep[np.isfinite(ep)])) if np.any(np.isfinite(ep)) else 1.0
        if baseline == 0.0:
            baseline = 1.0

    bars_since = np.arange(len(ep), dtype=float)

    pct_from_baseline = (ep - baseline) / baseline
    # replace NaNs produced by missing ep
    pct_from_baseline = np.where(np.isfinite(pct_from_baseline), pct_from_baseline, 0.0)

    # running min (causal)
    running_min = np.minimum.accumulate(np.where(np.isfinite(ep), ep, np.inf))
    running_min = np.where(np.isfinite(running_min) & (running_min != np.inf), running_min, baseline)

    pct_from_min = (ep - running_min) / np.where(running_min == 0.0, 1.0, running_min)
    pct_from_min = np.where(np.isfinite(pct_from_min), pct_from_min, 0.0)

    X = X.copy()
    X["bars_since_episode_start"] = bars_since
    X["pct_from_episode_start"] = pct_from_baseline
    X["min_since_episode_start"] = running_min
    X["pct_from_min_since_start"] = pct_from_min

    return X


def _compute_pos_weight(y_train: pd.Series) -> float:
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    ratio = neg / max(pos, 1)
    pw = float(np.sqrt(ratio))
    return float(np.clip(pw, 1.0, 6.0))


def _fit_model(
    model_name: str,
    x_fit: pd.DataFrame,
    y_fit: pd.Series,
    eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
):
    model_name = model_name.lower()

    if model_name in {"cat", "catboost", "cb"}:
        from market_timer.models import cat

        if y_fit.nunique(dropna=True) < 2:
            raise ValueError("Training labels contain only one class in this window.")

        pos_weight = _compute_pos_weight(y_fit)

        if eval_set is not None:
            X_val, y_val = eval_set
            if y_val.nunique(dropna=True) < 2:
                eval_set = None

        # Fix #2: USE half-life weights (recent > old)
        hl = max(50, len(x_fit) // 3)

        return cat.fit(
            x_fit,
            y_fit,
            half_life=hl,
        )

    if model_name in {"cat-multi", "cat_multi", "catmulti"}:
        from market_timer.models import cat

        if y_fit.nunique(dropna=True) < 2:
            raise ValueError("Training labels contain only one class in this window.")

        pos_weight = _compute_pos_weight(y_fit)
        y_flipped = 1 - y_fit

        if eval_set is not None:
            X_val, y_val = eval_set
            if y_val.nunique(dropna=True) < 2:
                eval_set = None

        hl = max(50, len(x_fit) // 3)

        m1 = cat.fit(x_fit, y_fit, half_life=hl)
        m2 = cat.fit(
            x_fit,
            y_flipped,
            half_life=hl,
        )
        return (m1, m2)

    if model_name in {"lstm"}:
        from market_timer.models import lstm
        return lstm.fit(x_fit, y_fit)

    if model_name in {"tcn"}:
        from market_timer.models import tcn
        hl = max(50, len(x_fit) // 3)
        return tcn.fit(x_fit, y_fit, half_life=hl)

    raise ValueError(f"Unsupported model: {model_name!r}")


def _predict_proba(model_name: str, model, x_pred: pd.DataFrame) -> float:
    model_name = model_name.lower()

    if model_name in {"cat", "catboost", "cb"}:
        from market_timer.models import cat
        proba = cat.predict(model, x_pred)
        return float(proba[-1])

    if model_name in {"cat-multi", "cat_multi", "catmulti"}:
        from market_timer.models import cat
        execute_model, wait_model = model
        execute_proba = float(cat.predict(execute_model, x_pred)[-1])
        wait_proba = float(cat.predict(wait_model, x_pred)[-1])
        inverted_wait_proba = 1.0 - wait_proba
        return (execute_proba + inverted_wait_proba) / 2.0

    if model_name in {"lstm"}:
        from market_timer.models import lstm
        proba = lstm.predict(model, x_pred)
        return float(proba[-1])

    if model_name in {"tcn"}:
        from market_timer.models import tcn
        proba = tcn.predict(model, x_pred)
        return float(proba[-1])

    raise ValueError(f"Unsupported model: {model_name!r}")


def _predict_proba_series(model_name: str, model, x_pred: pd.DataFrame) -> np.ndarray:
    model_name = model_name.lower()

    if model_name in {"cat", "catboost", "cb"}:
        from market_timer.models import cat
        return np.asarray(cat.predict(model, x_pred), dtype=float)

    if model_name in {"cat-multi", "cat_multi", "catmulti"}:
        from market_timer.models import cat
        execute_model, wait_model = model
        execute_proba = np.asarray(cat.predict(execute_model, x_pred), dtype=float)
        wait_proba = np.asarray(cat.predict(wait_model, x_pred), dtype=float)
        inverted_wait_proba = 1.0 - wait_proba
        return (execute_proba + inverted_wait_proba) / 2.0

    if model_name in {"lstm"}:
        from market_timer.models import lstm
        return np.asarray(lstm.predict(model, x_pred), dtype=float)

    if model_name in {"tcn"}:
        from market_timer.models import tcn
        return np.asarray(tcn.predict(model, x_pred), dtype=float)

    raise ValueError(f"Unsupported model: {model_name!r}")


def _align_tcn_calibration(
    probas: np.ndarray,
    y_cal: pd.Series,
) -> Tuple[np.ndarray, pd.Series]:
    if len(probas) == len(y_cal):
        return probas, y_cal
    if len(probas) == 0:
        return probas, y_cal.iloc[0:0]
    if len(probas) > len(y_cal):
        return probas[-len(y_cal) :], y_cal
    return probas, y_cal.iloc[-len(probas) :]


def _tcn_paths() -> Tuple[Path, Path, Path]:
    try:
        base_dir = Path(__file__).resolve().parent
    except Exception:
        base_dir = Path.cwd()
    tcn_dir = base_dir / "TCN"
    tcn_dir.mkdir(parents=True, exist_ok=True)
    model_path = tcn_dir / "model.pt"
    cache_path = tcn_dir / "cache.json"
    return tcn_dir, model_path, cache_path


def _load_tcn_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {"remaining": 0}
    try:
        payload = json.loads(cache_path.read_text())
    except json.JSONDecodeError:
        return {"remaining": 0}
    remaining = payload.get("remaining")
    if not isinstance(remaining, int):
        remaining = 0
    return {"remaining": max(0, remaining)}


def _save_tcn_cache(cache_path: Path, remaining: int) -> None:
    cache_path.write_text(json.dumps({"remaining": max(0, int(remaining))}))


def _best_threshold_weighted(
    probas: np.ndarray,
    y_true: pd.Series,
    beta: float = THRESH_BETA,
    lambda_fpr: float = THRESH_LAMBDA_FPR,
    max_pos_rate_mult: float = MAX_POS_RATE_MULT,
    min_precision: float = MIN_PRECISION,
) -> float:
    if probas.size == 0:
        return 0.5

    y = np.asarray(y_true, dtype=int)
    base_rate = float(np.mean(y == 1))

    if base_rate <= 0.0:
        return 0.99

    max_pos_rate = min(0.85, max_pos_rate_mult * base_rate)
    min_pos_rate = max(1.0 / len(y), min(0.02, 0.5 * base_rate))

    qs = np.linspace(0.01, 0.99, 200)
    grid = np.unique(np.quantile(probas, qs))
    grid = np.clip(grid, 1e-6, 1 - 1e-6)

    best_t = 0.5
    best_score = -1e18
    beta2 = beta * beta

    for t in grid:
        pred = (probas >= t).astype(int)
        pos_rate = float(pred.mean())

        if pos_rate > max_pos_rate or pos_rate < min_pos_rate:
            continue

        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)

        if prec < min_precision:
            continue

        denom = (beta2 * prec + rec)
        fbeta = (1.0 + beta2) * (prec * rec) / denom if denom > 0 else 0.0
        score = fbeta - lambda_fpr * fpr

        if score > best_score:
            best_score = score
            best_t = float(t)

    if best_score == -1e18:
        t = float(np.quantile(probas, 1.0 - base_rate))
        t = float(np.clip(t, 1e-6, 1 - 1e-6))
        if float((probas >= t).mean()) < (1.0 / len(probas)):
            t = float(np.max(probas) - 1e-6)
        return t

    return best_t


def _best_threshold_accuracy(probas: np.ndarray, y_true: pd.Series) -> float:
    """
    Optional: pick threshold that maximizes calibration-set accuracy:
      t* = argmax_t (TP+TN)/N
    """
    if probas.size == 0:
        return 0.5
    y = np.asarray(y_true, dtype=int)

    qs = np.linspace(0.01, 0.99, 200)
    grid = np.unique(np.quantile(probas, qs))
    grid = np.clip(grid, 1e-6, 1 - 1e-6)

    best_t = 0.5
    best_acc = -1.0

    for t in grid:
        pred = (probas >= t).astype(int)
        acc = float((pred == y).mean())
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)

    return best_t


def _time_split_fit_eval_cal(
    X: pd.DataFrame,
    y: pd.Series,
    min_chunk: int = 60,
    eval_frac: float = 0.15,
    cal_frac: float = 0.20,
):
    n = len(X)
    eval_n = max(min_chunk, int(n * eval_frac))
    cal_n = max(min_chunk, int(n * cal_frac))

    if n < (eval_n + cal_n + 50):
        cal_n = max(min_chunk, min(cal_n, n // 3))
        eval_n = cal_n

    cut_fit = n - (eval_n + cal_n)
    cut_eval = n - cal_n

    if cut_fit < 50:
        cut_fit = max(1, n - cal_n)
        cut_eval = cut_fit

    X_fit = X.iloc[:cut_fit]
    y_fit = y.iloc[:cut_fit]

    X_eval = X.iloc[cut_fit:cut_eval]
    y_eval = y.iloc[cut_fit:cut_eval]

    X_cal = X.iloc[cut_eval:]
    y_cal = y.iloc[cut_eval:]

    return X_fit, y_fit, X_eval, y_eval, X_cal, y_cal


def get_execution_decision(
    df: pd.DataFrame,
    ticker: str,
    label: int,
    model: str,
    direction: int,
):
    if direction not in (0, 1):
        raise ValueError("direction must be 0 (SELL) or 1 (BUY)")

    if _is_rl_model(model):
        from market_timer.models import rl
        model_direction = _direction_to_str(direction)
        fitted = rl.fit(df.copy(), direction=model_direction)
        return rl.predict(fitted, df.copy())

    # rolling window for live decision too (optional but recommended)
    df_work = df.copy()
    if USE_ROLLING_WINDOW and len(df_work) > TRAIN_WINDOW:
        df_work = df_work.iloc[-TRAIN_WINDOW:].copy()

    # infer base features BEFORE labeling
    base_feature_cols = _infer_feature_cols(df_work)

    # price series for Fix #4 episode-state features
    price_col = _infer_price_col(df_work)
    exec_prices = _exec_prices_next_bar(df_work, price_col).reset_index(drop=True)

    label_builder, label_col = _get_label_builder(label)
    labeled = label_builder(df_work.copy(), direction)
    if label == 2 and "label_execute_final" in labeled.columns:
        label_col = "label_execute_final"

    y = _normalize_labels(labeled[label_col]).reset_index(drop=True)

    X_base = _prepare_features(df_work, base_feature_cols).reset_index(drop=True)
    X_full = _add_episode_state_features(X_base, exec_prices)

    mask = y.notna().to_numpy()
    X_train = X_full.iloc[mask]
    y_train = y.iloc[mask].astype(int)

    if X_train.empty:
        raise ValueError("No labeled rows available for training.")

    X_fit, y_fit, X_eval, y_eval, X_cal, y_cal = _time_split_fit_eval_cal(X_train, y_train)
    model_key = model.lower()
    if model_key == "tcn":
        _, model_path, cache_path = _tcn_paths()
        cache = _load_tcn_cache(cache_path)
        remaining = cache["remaining"]
        if remaining <= 0 or not model_path.exists():
            fitted = _fit_model(model, X_fit, y_fit, eval_set=(X_eval, y_eval) if len(X_eval) >= 50 else None)
            from market_timer.models import tcn
            tcn.save(fitted, model_path)
            remaining = TCN_RETRAIN
        else:
            from market_timer.models import tcn
            fitted = tcn.load(model_path)
    else:
        fitted = _fit_model(model, X_fit, y_fit, eval_set=(X_eval, y_eval) if len(X_eval) >= 50 else None)

    if USE_STATIC_THRESHOLDS:
        threshold = 0.5
    else:
        probas_cal = _predict_proba_series(model, fitted, X_cal)
        if model_key == "tcn":
            probas_cal, y_cal = _align_tcn_calibration(probas_cal, y_cal)
        if THRESH_OBJECTIVE == "accuracy":
            threshold = _best_threshold_accuracy(probas_cal, y_cal)
        elif THRESH_OBJECTIVE == "recall_fpr":
            threshold = _best_threshold_recall_at_fpr(probas_cal, y_cal)
        elif THRESH_OBJECTIVE == "hybrid":
            threshold = _best_threshold_hybrid(probas_cal, y_cal)
        else:
            threshold = _best_threshold_weighted(probas_cal, y_cal)



    x_pred = X_full.tail(1)
    prob_execute = _predict_proba(model, fitted, x_pred)
    if model_key == "tcn":
        remaining = max(remaining - 1, 0)
        _save_tcn_cache(cache_path, remaining)
    return "EXECUTE" if prob_execute >= threshold else "WAIT"


def execution_backtest(
    df: pd.DataFrame,
    ticker: str,
    label: int,
    model: str,
    direction: int,
):
    import os
    from pathlib import Path

    if direction not in (0, 1):
        raise ValueError("direction must be 0 (SELL) or 1 (BUY)")

    model_key = model.lower()

    try:
        base_dir = Path(__file__).resolve().parent  # type: ignore[name-defined]
    except Exception:
        base_dir = Path.cwd()

    graphs_dir = base_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    def _safe_name(s: str) -> str:
        return "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in str(s))

    direction_str = "BUY" if direction == 1 else "SELL"
    prefix = _safe_name(f"{ticker}_{direction_str}_label{label}_{model}")

    # price series (used by BOTH branches; safe to compute once)
    price_col = _infer_price_col(df)
    exec_prices_all = _exec_prices_next_bar(df, price_col)

    try:
        from tqdm.auto import tqdm
    except Exception:
        def tqdm(iterable, **kwargs):
            return iterable

    def _compute_episode_savings(decisions: pd.DataFrame) -> dict:
        if decisions.empty:
            return {
                "episodes": 0,
                "money_saved": 0.0,
                "avg_savings_per_episode": 0.0,
                "episode_savings": np.array([], dtype=float),
                "episode_start_times": [],
                "episode_exec_times": [],
                "forced_last_episode": False,
            }

        episode_savings = []
        episode_start_times = []
        episode_exec_times = []
        forced_last_episode = False

        in_episode = False
        baseline_price = None
        baseline_time = None

        for i in range(len(decisions)):
            pred = int(decisions["pred"].iloc[i])
            price = float(decisions["exec_price"].iloc[i])
            t = decisions["time"].iloc[i]

            if not in_episode:
                in_episode = True
                baseline_price = price
                baseline_time = t

            if pred == 1:
                if direction == 1:   # BUY
                    savings = float(baseline_price - price)
                else:                # SELL
                    savings = float(price - baseline_price)

                episode_savings.append(savings)
                episode_start_times.append(baseline_time)
                episode_exec_times.append(t)

                in_episode = False
                baseline_price = None
                baseline_time = None

        if in_episode and baseline_price is not None:
            forced_last_episode = True
            last_price = float(decisions["exec_price"].iloc[-1])
            last_time = decisions["time"].iloc[-1]
            if direction == 1:
                savings = float(baseline_price - last_price)
            else:
                savings = float(last_price - baseline_price)

            episode_savings.append(savings)
            episode_start_times.append(baseline_time)
            episode_exec_times.append(last_time)

        arr = np.asarray(episode_savings, dtype=float)
        money_saved = float(arr.sum()) if arr.size else 0.0
        avg_savings = float(arr.mean()) if arr.size else 0.0

        return {
            "episodes": int(arr.size),
            "money_saved": money_saved,
            "avg_savings_per_episode": avg_savings,
            "episode_savings": arr,
            "episode_start_times": episode_start_times,
            "episode_exec_times": episode_exec_times,
            "forced_last_episode": forced_last_episode,
        }

    # ============================================================
    # RL BACKTEST (ONLY RL GRAPHS; NO LSTM/CAT GRAPHS CREATED HERE)
    # ============================================================
    if _is_rl_model(model):
        from market_timer.models import rl

        model_direction = _direction_to_str(direction)

        n = len(df)
        split = int(n * 0.8)
        train_df = df.iloc[:split].copy()

        fitted = rl.fit(train_df, direction=model_direction)

        exec_price_col = fitted.exec_price_col
        if exec_price_col not in df.columns:
            raise ValueError(f"df missing exec_price_col={exec_price_col!r} for RL evaluation")

        exec_prices = df[exec_price_col].shift(-1).to_numpy(dtype=np.float32)
        exec_prices[-1] = np.float32(df[exec_price_col].iloc[-1])

        tp_dyn, sl_dyn = rl._tb_precompute_dynamic_barriers(
            prices=exec_prices.astype(np.float64),
            horizon=fitted.tb_horizon,
            q_window=fitted.tb_q_window,
            q_tp=fitted.tb_q_tp,
            q_sl=fitted.tb_q_sl,
            min_tp=fitted.tb_min_tp,
            max_tp=fitted.tb_max_tp,
            min_sl=fitted.tb_min_sl,
            max_sl=fitted.tb_max_sl,
        )
        _, tb_score = rl._tb_precompute_scores(
            prices=exec_prices.astype(np.float64),
            direction=model_direction,
            horizon=fitted.tb_horizon,
            tp_dyn=tp_dyn,
            sl_dyn=sl_dyn,
        )

        start_idx = max(split, fitted.window - 1)
        end_idx = n - 2
        if start_idx > end_idx:
            raise ValueError("Not enough rows in test segment for RL backtest.")

        wait_steps = 0
        t0 = start_idx

        episodes = 0
        exec_scores: list[float] = []
        exec_positive = 0

        execute_count = 0
        wait_count = 0

        # For graphs + savings
        episode_start_idxs: list[int] = []
        episode_exec_idxs: list[int] = []
        episode_savings: list[float] = []

        it = tqdm(
            range(start_idx, end_idx + 1),
            total=(end_idx - start_idx + 1),
            desc=f"RL Backtest {ticker} ({model_direction})",
            unit="step",
        )

        for idx in it:
            if wait_steps == 0:
                t0 = idx

            if wait_steps >= fitted.horizon:
                action = "EXECUTE"
            else:
                action = rl.predict(fitted, df.iloc[: idx + 1], wait_steps=wait_steps)

            if action == "WAIT":
                wait_count += 1
                wait_steps = min(wait_steps + 1, fitted.horizon)
            else:
                execute_count += 1
                te = idx

                exec_price = float(exec_prices[te])

                score = float(tb_score[te]) if np.isfinite(tb_score[te]) else float(-fitted.tb_max_sl)
                exec_scores.append(score)
                if score > 0.0:
                    exec_positive += 1

                # Savings vs "execute immediately at t0" (per 1 unit)
                baseline_price = float(exec_prices[t0])
                if model_direction == "BUY":
                    savings = float(baseline_price - exec_price)
                else:
                    savings = float(exec_price - baseline_price)

                episode_start_idxs.append(int(t0))
                episode_exec_idxs.append(int(te))
                episode_savings.append(savings)

                episodes += 1
                wait_steps = 0

            if (idx - start_idx) % 25 == 0:
                avg_score_running = float(np.mean(exec_scores)) if exec_scores else 0.0
                positive_rate = (exec_positive / episodes) if episodes else 0.0
                it.set_postfix(
                    episodes=episodes,
                    hit_pos=f"{positive_rate:.4f}",
                    avg_score=f"{avg_score_running:.6f}",
                    exec=execute_count,
                    wait=wait_count,
                )

        avg_score = float(np.mean(exec_scores)) if exec_scores else None
        med_score = float(np.median(exec_scores)) if exec_scores else None
        hit_pos_rate = (exec_positive / episodes) if episodes else None

        money_saved = float(np.sum(episode_savings)) if episode_savings else 0.0
        avg_saved = float(np.mean(episode_savings)) if episode_savings else 0.0

        logging.info(
            "RL eval done. episodes=%s avg_score=%.6f hit_pos=%.4f execute=%s wait=%s money_saved=%.6f",
            episodes,
            avg_score if avg_score is not None else -1.0,
            hit_pos_rate if hit_pos_rate is not None else 0.0,
            execute_count,
            wait_count,
            money_saved,
        )

        print(
            f"[RL] episodes={episodes} hit_pos={hit_pos_rate} execute={execute_count} wait={wait_count} money_saved={money_saved:.6f}"
        )

        # --------- RL graphs (ONLY) ----------
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Base price series over the *test* segment
            x = df.index[start_idx : end_idx + 1]
            y = (
                pd.to_numeric(df[exec_price_col], errors="coerce")
                .iloc[start_idx : end_idx + 1]
                .ffill()
                .bfill()
                .to_numpy()
            )

            # (Existing) Price w/ episode markers (combined)
            fig = plt.figure()
            plt.plot(x, y)

            if episode_start_idxs:
                xs = df.index[episode_start_idxs]
                ys = pd.to_numeric(df[exec_price_col], errors="coerce").iloc[episode_start_idxs].to_numpy()
                plt.scatter(xs, ys, marker="o", label="episode_start")

            if episode_exec_idxs:
                xs = df.index[episode_exec_idxs]
                ys = pd.to_numeric(df[exec_price_col], errors="coerce").iloc[episode_exec_idxs].to_numpy()
                plt.scatter(xs, ys, marker="^", label="execute")

            plt.title(f"{ticker} RL Price + Episode Markers ({model_direction})")
            plt.legend()
            plt.tight_layout()
            fig.savefig(graphs_dir / f"{prefix}_rl_price_markers.png", dpi=160)
            plt.close(fig)

            # (NEW #1) Price + EXECUTE markers only
            fig = plt.figure()
            plt.plot(x, y)

            if episode_start_idxs:
                xs = df.index[episode_start_idxs]
                ys = pd.to_numeric(df[exec_price_col], errors="coerce").iloc[episode_start_idxs].to_numpy()
                plt.scatter(xs, ys, marker="o", label="episode_start")

            if episode_exec_idxs:
                xs = df.index[episode_exec_idxs]
                ys = pd.to_numeric(df[exec_price_col], errors="coerce").iloc[episode_exec_idxs].to_numpy()
                plt.scatter(xs, ys, marker="^", label="model_execute")

            plt.title(f"{ticker} RL Price + Model Execute Markers ({model_direction})")
            plt.legend()
            plt.tight_layout()
            fig.savefig(graphs_dir / f"{prefix}_rl_price_best_execute_markers.png", dpi=160)
            plt.close(fig)

            # Execute score histogram
            if exec_scores:
                fig = plt.figure()
                plt.hist(np.asarray(exec_scores, dtype=float), bins=40)
                plt.title(f"{ticker} RL Execute Score Histogram ({model_direction})")
                plt.xlabel("tb_score")
                plt.ylabel("count")
                plt.tight_layout()
                fig.savefig(graphs_dir / f"{prefix}_rl_regret_hist.png", dpi=160)
                plt.close(fig)

            # Episode savings curve
            if episode_savings:
                fig = plt.figure()
                cs = np.cumsum(np.asarray(episode_savings, dtype=float))
                plt.plot(cs)
                plt.title(f"{ticker} RL Cumulative Savings ({model_direction})")
                plt.xlabel("episode")
                plt.ylabel("cumulative_savings (per 1 unit)")
                plt.tight_layout()
                fig.savefig(graphs_dir / f"{prefix}_rl_cum_savings.png", dpi=160)
                plt.close(fig)

        except Exception as e:
            logging.warning("RL graph generation failed: %s", e)

        return {
            "ticker": ticker,
            "samples": episodes,              # episodes, not candles
            "correct": None,
            "accuracy": hit_pos_rate,         # percent of execute actions with positive TB score
            "avg_percent_regret": None,
            "median_percent_regret": None,
            "hit@0": None,
            "hit@1": None,
            "hit@2": None,
            "execute": execute_count,
            "wait": wait_count,
            "true_1_count": None,
            "true_0_count": None,
            "pred_1_count": None,
            "pred_0_count": None,
            "false_positives": None,
            "false_positive_pct": None,
            "avg_execute_score": avg_score,
            "median_execute_score": med_score,
            "money_saved": money_saved,                     # NEW
            "avg_savings_per_episode": avg_saved,           # NEW
            "graphs_dir": str(graphs_dir),                  # handy
        }

    # ============================================================
    # NON-RL BACKTEST (LSTM/CAT GRAPHS LIVE HERE ONLY)
    # ============================================================

    # FIX: infer fixed base feature columns BEFORE labeling
    base_feature_cols = _infer_feature_cols(df)

    label_builder, label_col = _get_label_builder(label)
    full_labeled = label_builder(df.copy(), direction)
    if label == 2 and "label_execute_final" in full_labeled.columns:
        label_col = "label_execute_final"

    y_full = _normalize_labels(full_labeled[label_col]).reset_index(drop=True)

    n = len(df)
    start_idx = max(0, n - 500)

    total = 0
    correct = 0

    true_1_count = 0
    true_0_count = 0
    pred_1_count = 0
    pred_0_count = 0
    false_positives = 0

    tp = tn = fp = fn = 0

    times = []
    y_true_list = []
    y_pred_list = []
    proba_list = []
    thr_list = []
    exec_price_list = []

    train_failures = 0
    last_train_error: Optional[str] = None

    # Fix #3: threshold smoothing state
    prev_thr: Optional[float] = None

    it = tqdm(range(start_idx, n), total=(n - start_idx), desc=f"Backtest {ticker} ({direction_str})", unit="step")

    for idx in it:
        y_true = y_full.iloc[idx]
        if pd.isna(y_true):
            continue

        # Fix #2: rolling train window
        if USE_ROLLING_WINDOW:
            start = max(0, idx - TRAIN_WINDOW)
        else:
            start = 0

        train_raw = df.iloc[start:idx].copy()
        if len(train_raw) < MIN_TRAIN_BARS:
            continue

        # Build X_train with base features + Fix #4 episode-state features
        X_train_base = _prepare_features(train_raw, base_feature_cols).reset_index(drop=True)
        exec_train = exec_prices_all.iloc[start:idx].reset_index(drop=True)
        X_train = _add_episode_state_features(X_train_base, exec_train)

        # y_train from labeled copy
        train_labeled = label_builder(train_raw.copy(), direction)
        train_label_col = "label_execute_final" if (label == 2 and "label_execute_final" in train_labeled.columns) else label_col
        y_train = _normalize_labels(train_labeled[train_label_col]).reset_index(drop=True)

        # Align
        m = min(len(X_train), len(y_train))
        X_train = X_train.iloc[:m]
        y_train = y_train.iloc[:m]

        mask = y_train.notna().to_numpy()
        X_train = X_train.iloc[mask]
        y_train = y_train.iloc[mask].astype(int)

        if X_train.empty or y_train.nunique() < 2:
            continue

        prob_execute = float("nan")
        threshold = 0.5

        try:
            X_fit, y_fit, X_eval, y_eval, X_cal, y_cal = _time_split_fit_eval_cal(X_train, y_train)
            fitted = _fit_model(model, X_fit, y_fit, eval_set=(X_eval, y_eval) if len(X_eval) >= 50 else None)

            if USE_STATIC_THRESHOLDS:
                thr_new = 0.5
            else:
                probas_cal = _predict_proba_series(model, fitted, X_cal)
                if model_key == "tcn":
                    probas_cal, y_cal = _align_tcn_calibration(probas_cal, y_cal)
                if THRESH_OBJECTIVE == "accuracy":
                    thr_new = _best_threshold_accuracy(probas_cal, y_cal)
                elif THRESH_OBJECTIVE == "recall_fpr":
                    thr_new = _best_threshold_recall_at_fpr(probas_cal, y_cal)
                elif THRESH_OBJECTIVE == "hybrid":
                    thr_new = _best_threshold_hybrid(probas_cal, y_cal)
                else:
                    thr_new = _best_threshold_weighted(probas_cal, y_cal)

            # Fix #3: EWMA smoothing
            if USE_THRESHOLD_SMOOTHING:
                if prev_thr is None:
                    prev_thr = float(thr_new)   # initialize to first real threshold
                threshold = THRESH_SMOOTH_ALPHA * float(thr_new) + (1.0 - THRESH_SMOOTH_ALPHA) * float(prev_thr)
                prev_thr = threshold
            else:
                threshold = float(thr_new)

            # Build x_pred from the SAME rolling slice start..idx (inclusive)
            pred_raw = df.iloc[start:idx + 1].copy()
            X_pred_base = _prepare_features(pred_raw, base_feature_cols).reset_index(drop=True)
            exec_pred = exec_prices_all.iloc[start:idx + 1].reset_index(drop=True)
            X_pred_full = _add_episode_state_features(X_pred_base, exec_pred)
            x_pred = X_pred_full.tail(1)

            prob_execute = _predict_proba(model, fitted, x_pred)

            if not np.isfinite(prob_execute):
                raise ValueError("prob_execute is not finite")

            predicted = 1 if prob_execute >= threshold else 0

        except Exception as e:
            train_failures += 1
            last_train_error = str(e)
            predicted = int(y_train.mean() >= 0.5)

        y_true_int = int(y_true)

        total += 1
        if y_true_int == 1:
            true_1_count += 1
        else:
            true_0_count += 1

        if predicted == 1:
            pred_1_count += 1
        else:
            pred_0_count += 1

        if y_true_int == predicted:
            correct += 1

        if predicted == 1 and y_true_int == 0:
            false_positives += 1

        if y_true_int == 1 and predicted == 1:
            tp += 1
        elif y_true_int == 0 and predicted == 0:
            tn += 1
        elif y_true_int == 0 and predicted == 1:
            fp += 1
        else:
            fn += 1

        times.append(df.index[idx])
        y_true_list.append(y_true_int)
        y_pred_list.append(int(predicted))
        proba_list.append(float(prob_execute) if np.isfinite(prob_execute) else float("nan"))
        thr_list.append(float(threshold))
        exec_price_list.append(float(exec_prices_all.iloc[idx]))

        if total and (total % 25 == 0):
            fp_pct_running = (false_positives / true_0_count * 100.0) if true_0_count else 0.0
            it.set_postfix(samples=total, acc=f"{(correct / total):.4f}", fp_pct=f"{fp_pct_running:.2f}%", t1=true_1_count, t0=true_0_count)

    accuracy = (correct / total) if total else 0.0
    false_positive_pct = (false_positives / true_0_count * 100.0) if true_0_count else 0.0
    label_1_accuracy = (tp / true_1_count) if true_1_count else 0.0
    label_0_accuracy = (tn / true_0_count) if true_0_count else 0.0

    print(f"Accuracy: {accuracy:.4f}")
    print(f"{correct}/{total} Correct")
    print(f"True label counts: 1={true_1_count}, 0={true_0_count}")
    print(f"Pred label counts: 1={pred_1_count}, 0={pred_0_count}")
    print(f"False positives: {false_positives} (FPR={false_positive_pct:.2f}%)")
    print(f"Label-1 accuracy (recall@1): {label_1_accuracy:.4f}")
    print(f"Label-0 accuracy (recall@0): {label_0_accuracy:.4f}")
    if train_failures:
        print(f"Training/pred failures: {train_failures} | last_error: {last_train_error}")

    decisions_df = pd.DataFrame(
        {
            "time": times,
            "true": y_true_list,
            "pred": y_pred_list,
            "proba": proba_list,
            "threshold": thr_list,
            "exec_price": exec_price_list,
        }
    )

    savings_info = _compute_episode_savings(decisions_df)
    money_saved = float(savings_info["money_saved"])
    avg_savings_per_episode = float(savings_info["avg_savings_per_episode"])
    episodes = int(savings_info["episodes"])

    print(f"Money saved: {money_saved:.6f} | episodes={episodes} | avg/episode={avg_savings_per_episode:.6f}")

    # --- graphs (NON-RL only; unchanged) ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not decisions_df.empty:
            x = decisions_df["time"].to_numpy()

            fig = plt.figure()
            plt.step(x, decisions_df["true"].to_numpy(dtype=float), where="post", label="true")
            plt.step(x, decisions_df["pred"].to_numpy(dtype=float), where="post", label="pred")
            plt.title(f"{ticker} Real vs Predicted Labels ({direction_str})")
            plt.ylim(-0.1, 1.1)
            plt.legend()
            plt.tight_layout()
            fig.savefig(graphs_dir / f"{prefix}_real_vs_pred_labels.png", dpi=160)
            plt.close(fig)

            fig = plt.figure()
            plt.plot(x, decisions_df["proba"].to_numpy(dtype=float), label="p(EXECUTE)")
            plt.plot(x, decisions_df["threshold"].to_numpy(dtype=float), label="threshold")
            plt.scatter(x, decisions_df["true"].to_numpy(dtype=float), marker="o", label="true_label")
            plt.title(f"{ticker} Probabilities vs Threshold ({direction_str})")
            plt.ylim(-0.1, 1.1)
            plt.legend()
            plt.tight_layout()
            fig.savefig(graphs_dir / f"{prefix}_proba_vs_threshold.png", dpi=160)
            plt.close(fig)

            fig = plt.figure()
            plt.plot(x, decisions_df["exec_price"].to_numpy(dtype=float), label=f"exec_price (next {price_col})")

            start_times = savings_info["episode_start_times"]
            exec_times = savings_info["episode_exec_times"]

            if start_times:
                start_mask = decisions_df["time"].isin(start_times)
                plt.scatter(
                    decisions_df.loc[start_mask, "time"].to_numpy(),
                    decisions_df.loc[start_mask, "exec_price"].to_numpy(dtype=float),
                    marker="o",
                    label="episode_start (baseline)",
                )

            if exec_times:
                exec_mask = decisions_df["time"].isin(exec_times)
                plt.scatter(
                    decisions_df.loc[exec_mask, "time"].to_numpy(),
                    decisions_df.loc[exec_mask, "exec_price"].to_numpy(dtype=float),
                    marker="^",
                    label="model_execute (pred=1)",
                )

            plt.title(f"{ticker} Price + Predicted Episode Markers ({direction_str})")
            plt.legend()
            plt.tight_layout()
            fig.savefig(graphs_dir / f"{prefix}_price_episode_markers.png", dpi=160)
            plt.close(fig)

            fig = plt.figure()
            plt.plot(x, decisions_df["exec_price"].to_numpy(dtype=float), label=f"exec_price (next {price_col})")

            if start_times:
                start_mask = decisions_df["time"].isin(start_times)
                plt.scatter(
                    decisions_df.loc[start_mask, "time"].to_numpy(),
                    decisions_df.loc[start_mask, "exec_price"].to_numpy(dtype=float),
                    marker="o",
                    label="episode_start (baseline)",
                )

            true_exec_mask = (decisions_df["true"].to_numpy(dtype=int) == 1)
            if np.any(true_exec_mask):
                plt.scatter(
                    decisions_df.loc[true_exec_mask, "time"].to_numpy(),
                    decisions_df.loc[true_exec_mask, "exec_price"].to_numpy(dtype=float),
                    marker="x",
                    label="true_execute (label=1)",
                )

            plt.title(f"{ticker} Price + TRUE Execute Markers ({direction_str})")
            plt.legend()
            plt.tight_layout()
            fig.savefig(graphs_dir / f"{prefix}_price_true_execute_markers.png", dpi=160)
            plt.close(fig)

            fig = plt.figure()
            plt.plot(x, decisions_df["exec_price"].to_numpy(dtype=float), label=f"exec_price (next {price_col})")

            if start_times:
                start_mask = decisions_df["time"].isin(start_times)
                plt.scatter(
                    decisions_df.loc[start_mask, "time"].to_numpy(),
                    decisions_df.loc[start_mask, "exec_price"].to_numpy(dtype=float),
                    marker="o",
                    label="episode_start (baseline)",
                )

            pred_exec_mask = (decisions_df["pred"].to_numpy(dtype=int) == 1)
            if np.any(pred_exec_mask):
                plt.scatter(
                    decisions_df.loc[pred_exec_mask, "time"].to_numpy(),
                    decisions_df.loc[pred_exec_mask, "exec_price"].to_numpy(dtype=float),
                    marker="^",
                    label="pred_execute (pred=1)",
                )

            if np.any(true_exec_mask):
                plt.scatter(
                    decisions_df.loc[true_exec_mask, "time"].to_numpy(),
                    decisions_df.loc[true_exec_mask, "exec_price"].to_numpy(dtype=float),
                    marker="x",
                    label="true_execute (label=1)",
                )

            plt.title(f"{ticker} Price + TRUE vs PRED Execute Markers ({direction_str})")
            plt.legend()
            plt.tight_layout()
            fig.savefig(graphs_dir / f"{prefix}_price_true_vs_pred_execute_markers.png", dpi=160)
            plt.close(fig)

            ep_s = savings_info["episode_savings"]
            if isinstance(ep_s, np.ndarray) and ep_s.size:
                fig = plt.figure()
                plt.plot(np.cumsum(ep_s))
                plt.title(f"{ticker} Cumulative Savings ({direction_str})")
                plt.xlabel("episode")
                plt.ylabel("cumulative_savings (per 1 unit)")
                plt.tight_layout()
                fig.savefig(graphs_dir / f"{prefix}_cumulative_savings.png", dpi=160)
                plt.close(fig)

                fig = plt.figure()
                plt.hist(ep_s, bins=40)
                plt.title(f"{ticker} Episode Savings Histogram ({direction_str})")
                plt.xlabel("episode_savings (per 1 unit)")
                plt.ylabel("count")
                plt.tight_layout()
                fig.savefig(graphs_dir / f"{prefix}_savings_hist.png", dpi=160)
                plt.close(fig)

    except Exception as e:
        logging.warning("Graph generation failed: %s", e)

    return {
        "ticker": ticker,
        "samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "label_1_accuracy": float(label_1_accuracy),
        "label_0_accuracy": float(label_0_accuracy),
        "true_1_count": true_1_count,
        "true_0_count": true_0_count,
        "pred_1_count": pred_1_count,
        "pred_0_count": pred_0_count,
        "false_positives": false_positives,
        "false_positive_pct": false_positive_pct,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "money_saved": money_saved,
        "avg_savings_per_episode": avg_savings_per_episode,
        "episodes": episodes,
        "price_col_for_savings": price_col,
        "graphs_dir": str(graphs_dir),
        "forced_last_episode": bool(savings_info["forced_last_episode"]),
        "train_failures": int(train_failures),
        "last_train_error": last_train_error,
        "train_window": int(TRAIN_WINDOW) if USE_ROLLING_WINDOW else None,
        "threshold_objective": str(THRESH_OBJECTIVE),
        "threshold_smoothing_alpha": float(THRESH_SMOOTH_ALPHA) if USE_THRESHOLD_SMOOTHING else None,
    }
