import importlib
from typing import Callable, Tuple
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

USE_STATIC_THRESHOLDS = True


def _is_rl_model(model_name: str) -> bool:
    return model_name.lower() in {"rl"}


def _direction_to_str(direction: int) -> str:
    return "BUY" if direction == 1 else "SELL"


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
    if model_name in {"cat-multi", "cat_multi", "catmulti"}:
        from market_timer.models import cat

        y_flipped = 1 - y_train
        return cat.fit(x_train, y_train), cat.fit(x_train, y_flipped)
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



def execution_backtest(
    df: pd.DataFrame,
    ticker: str,
    label: int,
    model: str,
    direction: int,
):
    if direction not in (0, 1):
        raise ValueError("direction must be 0 (SELL) or 1 (BUY)")

    # ---------------- RL PATH (unchanged behavior; label-count metrics not applicable) ----------------
    if _is_rl_model(model):
        from market_timer.models import rl

        # tqdm progress bar (safe fallback if tqdm isn't installed)
        try:
            from tqdm.auto import tqdm
        except Exception:
            def tqdm(iterable, **kwargs):
                return iterable

        model_direction = _direction_to_str(direction)

        # ---- Holdout split to avoid leakage (train on early data, eval on later data)
        n = len(df)
        split = int(n * 0.8)
        train_df = df.iloc[:split].copy()

        fitted = rl.fit(train_df, direction=model_direction)

        # Execution prices must match RL's training assumption: "next candle open"
        exec_price_col = fitted.exec_price_col
        if exec_price_col not in df.columns:
            raise ValueError(f"df missing exec_price_col={exec_price_col!r} for RL evaluation")

        exec_prices = df[exec_price_col].shift(-1).to_numpy(dtype=np.float32)
        exec_prices[-1] = np.float32(df[exec_price_col].iloc[-1])

        # We evaluate only on the test segment, but the state can use earlier history.
        start_idx = max(split, fitted.window - 1)
        end_idx = n - 2  # need idx+1 to exist for "next open" assumption
        if start_idx > end_idx:
            raise ValueError("Not enough rows in test segment for RL backtest.")

        # ---- Episode-based evaluation
        wait_steps = 0
        t0 = start_idx

        episodes = 0
        regrets: list[float] = []

        hit0 = 0
        hit1 = 0
        hit2 = 0

        execute_count = 0
        wait_count = 0

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

                end = min(t0 + fitted.horizon, len(exec_prices) - 1)
                window = exec_prices[t0 : end + 1]
                exec_price = float(exec_prices[te])

                if model_direction == "BUY":
                    best_pos = int(np.argmin(window))
                    best_price = float(window[best_pos])
                    best_t = t0 + best_pos
                    regret = max(exec_price - best_price, 0.0)
                else:
                    best_pos = int(np.argmax(window))
                    best_price = float(window[best_pos])
                    best_t = t0 + best_pos
                    regret = max(best_price - exec_price, 0.0)

                percent_regret = regret / max(best_price, 1e-12)
                regrets.append(float(percent_regret))

                dist = abs(te - best_t)
                if dist == 0:
                    hit0 += 1
                if dist <= 1:
                    hit1 += 1
                if dist <= 2:
                    hit2 += 1

                episodes += 1
                wait_steps = 0

            if (idx - start_idx) % 25 == 0:
                avg_regret_running = float(np.mean(regrets)) if regrets else 0.0
                hit1_running = (hit1 / episodes) if episodes else 0.0
                it.set_postfix(
                    episodes=episodes,
                    hit1=f"{hit1_running:.4f}",
                    avg_regret=f"{avg_regret_running:.6f}",
                    exec=execute_count,
                    wait=wait_count,
                )

        avg_regret = float(np.mean(regrets)) if regrets else None
        med_regret = float(np.median(regrets)) if regrets else None

        hit0_rate = (hit0 / episodes) if episodes else None
        hit1_rate = (hit1 / episodes) if episodes else None
        hit2_rate = (hit2 / episodes) if episodes else None

        logging.info(
            "RL eval done. episodes=%s avg_regret=%.6f hit@1=%.4f execute=%s wait=%s",
            episodes,
            avg_regret if avg_regret is not None else -1.0,
            hit1_rate if hit1_rate is not None else 0.0,
            execute_count,
            wait_count,
        )

        # Label counts / FP% aren't meaningful for this RL metric; return None for those.
        print(f"[RL] episodes={episodes} hit@1={hit1_rate} execute={execute_count} wait={wait_count}")

        return {
            "ticker": ticker,
            "samples": episodes,              # episodes, not candles
            "correct": None,
            "accuracy": hit1_rate,            # treat hit@1 as "accuracy-like"
            "avg_percent_regret": avg_regret,
            "median_percent_regret": med_regret,
            "hit@0": hit0_rate,
            "hit@1": hit1_rate,
            "hit@2": hit2_rate,
            "execute": execute_count,
            "wait": wait_count,
            "true_1_count": None,
            "true_0_count": None,
            "pred_1_count": None,
            "pred_0_count": None,
            "false_positives": None,
            "false_positive_pct": None,
        }

    # ---------------- NON-RL PATH (UPDATED: counts + FP%) ----------------

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

    # New counters
    true_1_count = 0
    true_0_count = 0
    pred_1_count = 0
    pred_0_count = 0
    false_positives = 0  # predicted 1 when true is 0

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
            # fallback: predict majority class in training set (with 0.5 rule)
            fallback_threshold = 0.5
            predicted = int(y_train.mean() >= fallback_threshold)

        y_true_int = int(y_true)

        # Update totals
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

        # Update bar info occasionally
        if total and (total % 25 == 0):
            # False positive % as False Positive Rate (FPR) over actual negatives: FP / (FP + TN) = FP / true_0_count
            fp_pct_running = (false_positives / true_0_count * 100.0) if true_0_count else 0.0
            it.set_postfix(
                samples=total,
                acc=f"{(correct / total):.4f}",
                fp_pct=f"{fp_pct_running:.2f}%",
                t1=true_1_count,
                t0=true_0_count,
            )

    accuracy = (correct / total) if total else 0.0

    # False positive % as False Positive Rate (FPR) over actual negatives
    false_positive_pct = (false_positives / true_0_count * 100.0) if true_0_count else 0.0

    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"{correct}/{total} Correct")
    logging.info(f"True label counts: 1={true_1_count}, 0={true_0_count}")
    logging.info(f"Pred label counts: 1={pred_1_count}, 0={pred_0_count}")
    logging.info(f"False positives: {false_positives} (FPR={false_positive_pct:.2f}%)")

    # Also print (as requested)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"{correct}/{total} Correct")
    print(f"True label counts: 1={true_1_count}, 0={true_0_count}")
    print(f"Pred label counts: 1={pred_1_count}, 0={pred_0_count}")
    print(f"False positives: {false_positives} (false positive % (FPR)={false_positive_pct:.2f}%)")

    return {
        "ticker": ticker,
        "samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "true_1_count": true_1_count,
        "true_0_count": true_0_count,
        "pred_1_count": pred_1_count,
        "pred_0_count": pred_0_count,
        "false_positives": false_positives,
        "false_positive_pct": false_positive_pct,  # FPR = FP / actual_0 * 100
    }
