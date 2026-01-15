from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

def fit(
    x_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray, list],
    half_life: Optional[int] = None,
):
    """
    Train a CatBoost multiclass classifier.
    - x_train, y_train are passed from the main script
    - half_life controls exponential decay sample weights (newest weighted most)
    """

    # ---- Normalize inputs ----
    if isinstance(x_train, np.ndarray):
        if x_train.ndim != 2:
            raise ValueError(f"x_train must be 2D, got shape {x_train.shape}")
    elif not isinstance(x_train, pd.DataFrame):
        raise TypeError("x_train must be a pandas DataFrame or a 2D numpy array")

    y_arr = np.asarray(y_train).reshape(-1)
    n = len(y_arr)
    if n == 0:
        raise ValueError("y_train is empty")
    if hasattr(x_train, "__len__") and len(x_train) != n:
        raise ValueError(f"x_train and y_train length mismatch: {len(x_train)} vs {n}")

    # ---- Label mapping (-1/0/1 -> 0/1/2) ----
    # If your y is already 0/1/2, this will leave it unchanged.
    label_mapping = {-1: 0, 0: 1, 1: 2}
    reverse_mapping = {v: k for k, v in label_mapping.items()}

    # Map only if values look like -1/0/1; otherwise assume already encoded.
    unique_vals = set(np.unique(y_arr).tolist())
    if unique_vals.issubset({-1, 0, 1}):
        y_encoded = np.vectorize(label_mapping.get)(y_arr).astype(int)
    else:
        y_encoded = y_arr.astype(int)

    # ---- Optional half-life weights ----
    weights = None
    if half_life is not None:
        hl = max(1, int(half_life))
        # oldest gets largest age, newest gets age 0
        ages = np.arange(n - 1, -1, -1, dtype=float)
        weights = np.exp(-np.log(2.0) * ages / float(hl))

    train_pool = Pool(x_train, y_encoded, weight=weights)

    # ---- Model ----
    model = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=5,
        l2_leaf_reg=5,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=42,
        early_stopping_rounds=20,
        verbose=False,
    )

    # Note: early_stopping_rounds only triggers if you also pass eval_set.
    # If you don't plan to pass eval_set here, you can remove early_stopping_rounds.
    model.fit(train_pool)
    return model


def predict(model, X):
    proba = model.predict_proba(X)[:, 1]
    return np.asarray(proba, dtype=float)