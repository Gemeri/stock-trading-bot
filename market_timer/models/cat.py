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


def _confidence_from_p(
    p: np.ndarray,
    model: Optional[CatBoostClassifier] = None,
    X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p_c = np.clip(p, eps, 1.0 - eps)

    # 1) Class confidence
    class_score = p

    if model is not None and X is not None:
        # Get cumulative per-tree predictions via staged_predict_proba
        # Use eval_period to sample ~100 checkpoints evenly across all trees
        n_trees = model.tree_count_
        eval_period = max(1, n_trees // 100)
        tree_probas = np.array(
            [s[:, 1] for s in model.staged_predict_proba(X, eval_period=eval_period)]
        )  # shape: (n_checkpoints, n_samples)
        tree_probas = np.clip(tree_probas, eps, 1.0 - eps)

        # 2) Gini: variance of tree predictions, normalized to [0,1]
        #    Var=0 (all trees agree) -> cert=1 | Var=0.25 (max) -> cert=0
        tree_var = np.var(tree_probas, axis=0)           # (n_samples,)
        gini_cert = np.clip(1.0 - 4.0 * tree_var, 0.0, 1.0)

        # 3) Entropy: mean binary entropy of individual tree predictions
        #    If trees are individually uncertain (p_i ≈ 0.5), mean H is high -> cert low
        #    If trees are individually decisive (p_i near 0 or 1), mean H is low -> cert high
        tree_H = -(
            tree_probas * np.log2(tree_probas)
            + (1.0 - tree_probas) * np.log2(1.0 - tree_probas)
        )  # (n_checkpoints, n_samples), each value in [0,1]
        mean_tree_H = np.mean(tree_H, axis=0)            # (n_samples,)
        entropy_cert = np.clip(1.0 - mean_tree_H, 0.0, 1.0)

    else:
        # Fallback: p-only path (no model available)
        gini_cert = np.abs(2.0 * p - 1.0)
        H = -(p_c * np.log2(p_c) + (1.0 - p_c) * np.log2(1.0 - p_c))
        entropy_cert = np.clip(1.0 - H, 0.0, 1.0)

    # Make both directional: 0 -> 0, 0.5 -> 0.5, 1 -> 1
    gini_score    = 0.5 + (p - 0.5) * gini_cert
    entropy_score = 0.5 + (p - 0.5) * entropy_cert

    score = (class_score + gini_score + entropy_score) / 3.0
    return np.clip(score, 0.0, 1.0)


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
        eval_metric = "AUC"
        custom_metric = ["AUC", "Logloss", "PRAUC"]
        # 1 is majority, 0 is minority -> upweight class 0
        neg_weight = float(np.clip(pos / max(neg, 1.0), 1.0, 20.0))
        class_weights = [neg_weight, 1.0]
    else:
        eval_metric = "PRAUC"
        custom_metric = ["AUC", "Logloss"]
        # 0 is majority, 1 is minority -> upweight class 1
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
        eval_metric=eval_metric,
        custom_metric=custom_metric,
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


def predict(
    model: CatBoostClassifier,
    X: Union[pd.DataFrame, np.ndarray],
    confidence_score: Optional[bool] = False,
    shap_reasoning: Optional[bool] = False,
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    _validate_X(X, "X")

    # Usual probability path (P(class=1))
    proba_1 = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    out = proba_1 if not confidence_score else _confidence_from_p(proba_1, model, X)
    print(f"Proba: {out}")

    if not shap_reasoning:
        return out

    # ---- SHAP feature contributions for label 1 only ----
    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        pool = Pool(X)
        n_samples = int(X.shape[0])
    else:
        X_arr = np.asarray(X)
        feature_names = [f"f{i}" for i in range(int(X_arr.shape[1]))]
        pool = Pool(X_arr, feature_names=feature_names)
        n_samples = int(X_arr.shape[0])

    shap_vals = np.asarray(model.get_feature_importance(pool, type="ShapValues"))

    # CatBoost shapes:
    # - Binary: (n_samples, n_features + 1)
    # - Multiclass: (n_samples, n_classes, n_features + 1)
    if shap_vals.ndim == 2:
        contrib = shap_vals[:, :-1]  # drop base_value column
    elif shap_vals.ndim == 3:
        class_idx = 1 if shap_vals.shape[1] > 1 else 0
        contrib = shap_vals[:, class_idx, :-1]  # drop base_value column
    else:
        raise ValueError(f"Unexpected SHAP values shape: {shap_vals.shape}")

    if contrib.shape[1] != len(feature_names):
        raise ValueError(
            f"SHAP feature mismatch: contrib has {contrib.shape[1]} features, "
            f"but X has {len(feature_names)}"
        )

    if n_samples == 1:
        feature_contribs = {
            name: float(contrib[0, i]) for i, name in enumerate(feature_names)
        }
    else:
        feature_contribs = {
            name: np.asarray(contrib[:, i], dtype=float)
            for i, name in enumerate(feature_names)
        }

    return out, feature_contribs