import importlib
from typing import Callable, Tuple
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

USE_STATIC_THRESHOLDS = True

def _get_label_builder(label: int) -> Tuple[Callable[[pd.DataFrame, int], pd.DataFrame], str]:
    if label == 1:
        module = importlib.import_module("market_timer.labels.goodenough")
        return module.build_labels, "label_execute"
    if label == 2:
        module = importlib.import_module("market_timer.labels.override")
        return module.build_labels, "label_execute"
    if label == 3:
        module = importlib.import_module("market_timer.labels.percentage")
        return module.build_labels, "label"
    if label == 4:
        module = importlib.import_module("market_timer.labels.triple_barrier")
        return module.build_labels, "label"
    if label == 5:
        module = importlib.import_module("market_timer.labels.9_lookahead")
        return module.build_labels, "label"
    raise ValueError("label must be an int between 1 and 5")


def _normalize_labels(y: pd.Series) -> pd.Series:
    if y.dtype == object:
        y = y.map({"EXECUTE": 1, "WAIT": 0})
    y = pd.to_numeric(y, errors="coerce")
    return y


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No numeric feature columns available in df.")
    X = df[numeric_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.ffill().bfill().fillna(0.0)
    return X


def _fit_model(model_name: str, x_train: pd.DataFrame, y_train: pd.Series):
    model_name = model_name.lower()
    if model_name in {"cat", "catboost", "cb"}:
        from market_timer.models import cat

        return cat.fit(x_train, y_train)
    if model_name in {"lstm"}:
        from market_timer.models import lstm

        return lstm.fit(x_train, y_train)
    raise ValueError(f"Unsupported model: {model_name!r}")


def _predict_proba(model_name: str, model, x_pred: pd.DataFrame) -> float:
    model_name = model_name.lower()
    if model_name in {"cat", "catboost", "cb"}:
        from market_timer.models import cat

        proba = cat.predict(model, x_pred)
        return float(proba[-1])
    if model_name in {"lstm"}:
        from market_timer.models import lstm

        proba = lstm.predict(model, x_pred)
        return float(proba[-1])
    raise ValueError(f"Unsupported model: {model_name!r}")

def _predict_proba_series(model_name: str, model, x_pred: pd.DataFrame) -> np.ndarray:
    model_name = model_name.lower()
    if model_name in {"cat", "catboost", "cb"}:
        from market_timer.models import cat

        return np.asarray(cat.predict(model, x_pred), dtype=float)
    if model_name in {"lstm"}:
        from market_timer.models import lstm

        return np.asarray(lstm.predict(model, x_pred), dtype=float)
    raise ValueError(f"Unsupported model: {model_name!r}")


def _best_threshold_from_grid(probas: np.ndarray, y_true: pd.Series) -> float:
    if probas.size == 0:
        return 0.5
    y_arr = np.asarray(y_true, dtype=int)
    grid = np.linspace(0.5, 0.8, num=31)
    best_threshold = 0.5
    best_accuracy = -1.0
    for threshold in grid:
        preds = (probas >= threshold).astype(int)
        accuracy = float((preds == y_arr).mean())
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)
    return best_threshold



def get_execution_decision(
    df: pd.DataFrame,
    ticker: str,
    label: int,
    model: str,
    direction: int,
):
    if direction not in (0, 1):
        raise ValueError("direction must be 0 (SELL) or 1 (BUY)")

    label_builder, label_col = _get_label_builder(label)
    labeled = label_builder(df.copy(), direction)
    if label == 2 and "label_execute_final" in labeled.columns:
        label_col = "label_execute_final"

    y = _normalize_labels(labeled[label_col])
    X = _prepare_features(df)

    mask = y.notna()
    X_train = X.loc[mask]
    y_train = y.loc[mask].astype(int)

    if X_train.empty:
        raise ValueError("No labeled rows available for training.")

    fitted = _fit_model(model, X_train, y_train)
    if USE_STATIC_THRESHOLDS:
        threshold = 0.5
    else:
        probas_train = _predict_proba_series(model, fitted, X_train)
        threshold = _best_threshold_from_grid(probas_train, y_train)
    x_pred = X.tail(1)
    prob_execute = _predict_proba(model, fitted, x_pred)

    return "EXECUTE" if prob_execute >= threshold else "WAIT"


def execution_backtest(
    df: pd.DataFrame,
    ticker: str,
    label: int,
    model: str,
    direction: int,
):
    if direction not in (0, 1):
        raise ValueError("direction must be 0 (SELL) or 1 (BUY)")

    # tqdm progress bar (safe fallback if tqdm isn't installed)
    try:
        from tqdm.auto import tqdm
    except Exception:
        def tqdm(iterable, **kwargs):
            return iterable

    label_builder, label_col = _get_label_builder(label)
    full_labeled = label_builder(df.copy(), direction)
    if label == 2 and "label_execute_final" in full_labeled.columns:
        label_col = "label_execute_final"

    y_full = _normalize_labels(full_labeled[label_col])
    X_full = _prepare_features(df)

    n = len(df)
    start_idx = max(0, n - 500)

    total = 0
    correct = 0

    it = tqdm(
        range(start_idx, n),
        total=(n - start_idx),
        desc=f"Backtest {ticker} ({'BUY' if direction == 1 else 'SELL'})",
        unit="step",
    )

    for idx in it:
        y_true = y_full.iloc[idx]
        if pd.isna(y_true):
            continue

        train_slice = df.iloc[:idx].copy()
        if len(train_slice) < 2:
            continue

        train_labeled = label_builder(train_slice, direction)
        if label == 2 and "label_execute_final" in train_labeled.columns:
            train_label_col = "label_execute_final"
        else:
            train_label_col = label_col

        y_train = _normalize_labels(train_labeled[train_label_col])
        X_train = _prepare_features(train_slice)

        mask = y_train.notna()
        X_train = X_train.loc[mask]
        y_train = y_train.loc[mask].astype(int)

        if X_train.empty:
            continue

        try:
            fitted = _fit_model(model, X_train, y_train)
            if USE_STATIC_THRESHOLDS:
                threshold = 0.5
            else:
                probas_train = _predict_proba_series(model, fitted, X_train)
                threshold = _best_threshold_from_grid(probas_train, y_train)

            x_pred = X_full.iloc[[idx]]
            prob_execute = _predict_proba(model, fitted, x_pred)
            predicted = 1 if prob_execute >= threshold else 0
        except Exception:
            fallback_threshold = 0.5
            predicted = int(y_train.mean() >= fallback_threshold)

        total += 1
        if int(y_true) == predicted:
            correct += 1

        # Update bar info occasionally to avoid slowing the loop
        if total and (total % 25 == 0):
            it.set_postfix(
                samples=total,
                acc=f"{(correct / total):.4f}",
            )

    accuracy = (correct / total) if total else 0.0
    logging.info("Accuracy: " + accuracy)
    logging.info(correct + "/" + total + " Correct")
    return {
        "ticker": ticker,
        "samples": total,
        "correct": correct,
        "accuracy": accuracy,
    }
