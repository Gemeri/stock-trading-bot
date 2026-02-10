from __future__ import annotations

from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


def _validate_X(X: Union[pd.DataFrame, np.ndarray], name: str) -> None:
    if isinstance(X, np.ndarray):
        if X.ndim != 2:
            raise ValueError(f"{name} must be 2D, got shape {X.shape}")
    elif not isinstance(X, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame or a 2D numpy array")


def _as_1d_int_labels(y: Union[pd.Series, np.ndarray, list], name: str) -> np.ndarray:
    y_arr = np.asarray(y).reshape(-1)
    if y_arr.size == 0:
        raise ValueError(f"{name} is empty")
    return y_arr.astype(int)


def _half_life_weights(n: int, half_life: Optional[int]) -> Optional[np.ndarray]:
    if half_life is None:
        return None
    hl = max(1, int(half_life))
    ages = np.arange(n - 1, -1, -1, dtype=float)
    return np.exp(-np.log(2.0) * ages / float(hl))


def _compute_pos_weight(y: np.ndarray, cap: float = 20.0) -> float:
    y = np.asarray(y).reshape(-1).astype(int)
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    ratio = neg / max(pos, 1.0)
    return float(np.clip(ratio, 1.0, cap))


def fit(
    x_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray, list],
    half_life: Optional[int] = None,
    eval_set: Optional[
        Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray, list]]
    ] = None,
    eval_fraction: float = 0.2,  # used only if eval_set is None
    min_eval_size: int = 50,  # used only if eval_set is None
    early_stopping_rounds: int = 100,
):
    """
    Binary CatBoost fit with:
      - optional half-life sample weights (more weight to recent samples)
      - imbalance-aware class_weights (like test.py)
      - proper early stopping via eval_set
      - time-based validation split by default (last eval_fraction chunk)

    Returns a fitted CatBoostClassifier.
    """
    # ---- Input validation ----
    _validate_X(x_train, "x_train")
    y_arr = _as_1d_int_labels(y_train, "y_train")

    n = int(y_arr.size)
    if hasattr(x_train, "__len__") and len(x_train) != n:
        raise ValueError(f"x_train and y_train length mismatch: {len(x_train)} vs {n}")

    unique_vals = set(np.unique(y_arr).tolist())
    if not unique_vals.issubset({0, 1}):
        raise ValueError(f"Unsupported label set for this model: {sorted(unique_vals)}")

    # ---- Optional half-life weights (recency weighting) ----
    weights_all = _half_life_weights(n, half_life)

    # ---- Build train/val split (time-based) unless user supplies eval_set ----
    X_fit = x_train
    y_fit = y_arr
    w_fit = weights_all

    X_val = None
    y_val = None
    w_val = None

    if eval_set is not None:
        X_val, y_val_in = eval_set
        _validate_X(X_val, "X_val")
        y_val = _as_1d_int_labels(y_val_in, "y_val")

        if hasattr(X_val, "__len__") and len(X_val) != len(y_val):
            raise ValueError(
                f"X_val and y_val length mismatch: {len(X_val)} vs {len(y_val)}"
            )

        # If user provides arbitrary eval_set, we cannot safely align half-life weights -> don't weight eval
        w_val = None

    else:
        # default time-based holdout: last eval_fraction chunk
        if eval_fraction > 0.0 and n >= (min_eval_size * 2 + 20):
            eval_n = max(min_eval_size, int(n * float(eval_fraction)))
            if n - eval_n >= 20:
                if isinstance(x_train, pd.DataFrame):
                    X_fit = x_train.iloc[:-eval_n]
                    X_val = x_train.iloc[-eval_n:]
                else:
                    X_fit = x_train[:-eval_n]
                    X_val = x_train[-eval_n:]

                y_fit = y_arr[:-eval_n]
                y_val = y_arr[-eval_n:]

                if weights_all is not None:
                    w_fit = weights_all[:-eval_n]
                    w_val = weights_all[-eval_n:]

    # ---- Class weights (imbalance-aware like test.py) ----
    pos = float(np.sum(y_fit == 1))
    neg = float(np.sum(y_fit == 0))

    if pos >= neg:
        # 1 is majority, 0 is minority -> upweight class 0
        neg_weight = float(np.clip(pos / max(neg, 1.0), 1.0, 20.0))
        class_weights = [neg_weight, 1.0]
    else:
        # 0 is majority, 1 is minority -> upweight class 
        pos_weight = _compute_pos_weight(y_fit, cap=20.0)
        class_weights = [1.0, pos_weight]

    # ---- Pools ----
    train_pool = Pool(X_fit, y_fit, weight=w_fit)
    eval_pool = None
    if X_val is not None and y_val is not None:
        eval_pool = Pool(X_val, y_val, weight=w_val)

    # ---- Model params (CatBoost structure matching test.py) ----
    params = dict(
        iterations=3000,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=20.0,
        min_data_in_leaf=80,
        loss_function="Logloss",
        eval_metric="PRAUC",
        custom_metric=["AUC", "Logloss"],
        bootstrap_type="Bernoulli",
        subsample=0.8,
        rsm=0.85,
        random_strength=2.0,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        class_weights=class_weights,
    )

    # Only enable early stopping if we actually have an eval set
    if eval_pool is not None and early_stopping_rounds and early_stopping_rounds > 0:
        params["early_stopping_rounds"] = int(early_stopping_rounds)

    model = CatBoostClassifier(**params)

    if eval_pool is not None:
        model.fit(train_pool, eval_set=eval_pool, use_best_model=True)
    else:
        model.fit(train_pool)

    return model


def predict(model, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    _validate_X(X, "X")
    proba = model.predict_proba(X)[:, 1]  # P(class=1)
    return np.asarray(proba, dtype=float)
