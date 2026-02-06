"""
test.py — Evaluate feature signal vs QUANTILE-based TRIPLE-BARRIER label
(using a rolling quantile window to set TP/SL dynamically).

Idea:
- Instead of fixed tp_percent/sl_percent, compute rolling return quantiles
  from past data (no future leakage):
    tp_pct_i = quantile( future_max_return_distribution , q_tp )
    sl_pct_i = abs( quantile( future_min_return_distribution , q_sl ) )

- Then apply the same "first barrier hit within horizon" labeling:
    BUY:  hit +tp first => label=1, hit -sl first => label=0, none => 0
    SELL: profit when price drops by tp, risk when price rises by sl
          hit profit first => 1 else 0

Notes:
- Quantile window is *backward-looking* (uses only past candles) so it’s safe.
- You can drop the last HORIZON rows (recommended) to avoid forced tail labels.
- Still evaluates with PR-AUC / ROC-AUC + decile lift + top-X% threshold analysis.

Run:
  python test.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


# ---------------------------
# Label config (QUANTILE triple barrier)
# ---------------------------
HORIZON = 9
ACTION_TYPE = 1  # 1=BUY, 0=SELL

# Rolling quantile window (past-only) for barrier estimation
QUANTILE_WINDOW = 32  # number of past samples used to estimate tp/sl quantiles
Q_TP = 0.70           # take-profit quantile of "future max return" distribution
Q_SL = 0.20           # stop-loss quantile of "future min return" distribution (negative)

# Reasonable bounds to prevent crazy barriers (expressed as fractions, e.g. 0.03 = 3%)
MIN_TP = 0.005
MAX_TP = 0.08
MIN_SL = 0.003
MAX_SL = 0.05

# If True: drop last `HORIZON` rows (since they don't have full lookahead)
DROP_LAST_HORIZON = True

# ---------------------------
# Data / eval config
# ---------------------------
CSV_PATH = "PYPL_H4.csv"
TRAIN_FRAC = 0.8

# ---------------------------
# CatBoost setup (regularized + imbalance-aware)
# ---------------------------
CB_PARAMS = dict(
    iterations=3000,
    learning_rate=0.03,
    depth=5,
    l2_leaf_reg=20.0,
    min_data_in_leaf=80,
    loss_function="Logloss",
    eval_metric="PRAUC",               # optimize for PR-AUC (better for imbalanced-ish)
    custom_metric=["AUC", "Logloss"],
    bootstrap_type="Bernoulli",
    subsample=0.8,
    rsm=0.85,
    random_strength=2.0,
    random_seed=42,
    verbose=False,
    early_stopping_rounds=150,
    allow_writing_files=False,
)

# Optional: recency weighting (half-life in samples). Set None to disable.
HALF_LIFE = None  # e.g. 400, or None

# Optional: "execute only top X%" analysis (sanity check)
TOP_PCT_EXECUTE = 0.10  # 10% most confident predictions


# ============================================================
# QUANTILE TRIPLE-BARRIER LABEL BUILDER
# ============================================================
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


# ============================================================
# Feature prep
# ============================================================
def prepare_X(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.ffill().fillna(0.0)
    return X


# ============================================================
# Metrics: PR-AUC + ROC-AUC (sklearn optional)
# ============================================================
def pr_auc_fallback(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / np.maximum(tp[-1], 1)
    return float(np.trapz(prec, rec))


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return pr_auc_fallback(y_true, y_score)


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        y_true = y_true.astype(int)
        pos = y_score[y_true == 1]
        neg = y_score[y_true == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        cnt = 0.0
        tot = 0.0
        for sp in pos:
            tot += len(neg)
            cnt += float(np.sum(sp > neg)) + 0.5 * float(np.sum(sp == neg))
        return float(cnt / max(tot, 1.0))


# ============================================================
# Lift + permutation importance
# ============================================================
def decile_lift(y_true: np.ndarray, y_score: np.ndarray, bins: int = 10) -> pd.DataFrame:
    tmp = pd.DataFrame({"y": y_true, "p": y_score}).sort_values("p", ascending=False).reset_index(drop=True)
    tmp["bin"] = pd.cut(tmp.index, bins=bins, labels=False, include_lowest=True)
    base = float(tmp["y"].mean())
    out = tmp.groupby("bin")["y"].mean().to_frame("pos_rate")
    out["lift_vs_base"] = out["pos_rate"] / (base if base > 0 else 1e-12)
    out["base_rate"] = base
    return out


def permutation_importance_pr_auc(
    model: CatBoostClassifier,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_repeats: int = 1,
    seed: int = 42,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    base_p = model.predict_proba(X_val)[:, 1]
    base = compute_pr_auc(y_val, base_p)

    drops: dict[str, float] = {}
    for col in X_val.columns:
        drop_vals = []
        for _ in range(n_repeats):
            Xp = X_val.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            pp = model.predict_proba(Xp)[:, 1]
            drop_vals.append(base - compute_pr_auc(y_val, pp))
        drops[col] = float(np.mean(drop_vals))
    return pd.Series(drops).sort_values(ascending=False)


# ============================================================
# Training helpers
# ============================================================
def _half_life_weights(n: int, half_life: int | None) -> np.ndarray | None:
    if half_life is None:
        return None
    hl = max(1, int(half_life))
    ages = np.arange(n - 1, -1, -1, dtype=float)
    return np.exp(-np.log(2.0) * ages / float(hl))


def _compute_pos_weight(y: np.ndarray, cap: float = 20.0) -> float:
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    ratio = neg / max(pos, 1.0)
    return float(np.clip(ratio, 1.0, cap))


# ============================================================
# Main evaluation
# ============================================================
def evaluate_signal(df: pd.DataFrame, feature_cols: list[str], train_frac: float = TRAIN_FRAC):
    labeled = build_labels_quantile_triple_barrier(
        df,
        action_type=ACTION_TYPE,
        horizon=HORIZON,
        q_window=QUANTILE_WINDOW,
        q_tp=Q_TP,
        q_sl=Q_SL,
        min_tp=MIN_TP,
        max_tp=MAX_TP,
        min_sl=MIN_SL,
        max_sl=MAX_SL,
    )

    # Optional: drop tail to avoid forced 0 labels from lack of horizon
    if DROP_LAST_HORIZON and len(labeled) > HORIZON:
        labeled = labeled.iloc[: -HORIZON].copy()

    y = pd.to_numeric(labeled["label"], errors="coerce").astype(int)
    X = prepare_X(labeled, feature_cols)

    n = len(y)
    split = int(n * train_frac)
    X_tr, y_tr = X.iloc[:split], y.iloc[:split]
    X_va, y_va = X.iloc[split:], y.iloc[split:]

    base_rate = float(y_va.mean())
    print(f"Validation base rate P(y=1) = {base_rate:.4f}")

    pos_weight = _compute_pos_weight(y_tr.to_numpy(), cap=20.0)

    params = dict(CB_PARAMS)
    params["class_weights"] = [1.0, pos_weight]

    w_tr = _half_life_weights(len(y_tr), HALF_LIFE)
    train_pool = Pool(X_tr, y_tr, weight=w_tr)

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=(X_va, y_va))

    p_va = model.predict_proba(X_va)[:, 1]
    pr = compute_pr_auc(y_va.to_numpy(), p_va)
    roc = compute_roc_auc(y_va.to_numpy(), p_va)

    print(f"Validation PR-AUC  = {pr:.4f} (baseline ~ {base_rate:.4f})")
    print(f"Validation ROC-AUC = {roc:.4f}")
    print(f"Best iteration (early stopped) = {model.get_best_iteration()}")
    print(f"Train pos_weight used = {pos_weight:.3f}")

    lift = decile_lift(y_va.to_numpy(), p_va, bins=10)
    print("\nDecile lift (bin 0 is TOP probabilities):")
    print(lift)

    # Top-X% execute behavior (sanity check)
    k = int(max(1, round(len(p_va) * float(TOP_PCT_EXECUTE))))
    thr_topk = float(np.sort(p_va)[-k]) if k < len(p_va) else float(np.min(p_va))
    pred_topk = (p_va >= thr_topk).astype(int)

    tp = int(np.sum((pred_topk == 1) & (y_va.to_numpy() == 1)))
    fp = int(np.sum((pred_topk == 1) & (y_va.to_numpy() == 0)))
    fn = int(np.sum((pred_topk == 0) & (y_va.to_numpy() == 1)))
    tn = int(np.sum((pred_topk == 0) & (y_va.to_numpy() == 0)))

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)

    print(f"\nTop-{int(TOP_PCT_EXECUTE*100)}% threshold analysis:")
    print(f"  threshold = {thr_topk:.6f}")
    print(f"  predicted_1_rate = {pred_topk.mean():.4f}")
    print(f"  precision = {prec:.4f} | recall = {rec:.4f} | FPR = {fpr:.4f}")

    imp = permutation_importance_pr_auc(model, X_va, y_va.to_numpy(), n_repeats=1)
    print("\nTop permutation importances (PR-AUC drop):")
    print(imp.head(15))

    # Quick diagnostics on dynamic barriers (helpful sanity check)
    tp_dyn = labeled.get("tp_dyn")
    sl_dyn = labeled.get("sl_dyn")
    if tp_dyn is not None and sl_dyn is not None:
        tp_dyn = pd.to_numeric(tp_dyn, errors="coerce")
        sl_dyn = pd.to_numeric(sl_dyn, errors="coerce")
        print("\nDynamic barrier diagnostics (tp_dyn/sl_dyn):")
        print(f"  tp_dyn mean={tp_dyn.mean():.4f} median={tp_dyn.median():.4f} min={tp_dyn.min():.4f} max={tp_dyn.max():.4f}")
        print(f"  sl_dyn mean={sl_dyn.mean():.4f} median={sl_dyn.median():.4f} min={sl_dyn.min():.4f} max={sl_dyn.max():.4f}")

    return model, lift, imp


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)

    feature_cols = [
        "open","high","low","close","volume","vwap","transactions","greed_index",
        "sentiment","news_count","price_change","high_low_range","macd_line",
        "macd_signal","macd_histogram","rsi","roc","atr","ema_9","ema_21","ema_50",
        "ema_200","adx","obv","bollinger_percB","returns_1","returns_3","returns_5",
        "std_5","std_10","lagged_close_1","lagged_close_2","lagged_close_3",
        "lagged_close_5","lagged_close_10","candle_body_ratio","wick_dominance",
        "news_volume_z","d_sentiment","gap_vs_prev","volume_zscore","atr_zscore",
        "rsi_zscore","month","hour_sin","hour_cos","day_of_week_sin",
        "day_of_week_cos","days_since_high","days_since_low"
    ]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in df: {missing}")

    if "close" not in df.columns:
        raise ValueError("df must contain 'close' for this label")

    side = "BUY" if ACTION_TYPE == 1 else "SELL"
    print(
        f"Using QUANTILE TRIPLE-BARRIER label: side={side}, horizon={HORIZON}, drop_tail={DROP_LAST_HORIZON}\n"
        f"Quantile window={QUANTILE_WINDOW}, q_tp={Q_TP:.2f}, q_sl={Q_SL:.2f}, "
        f"tp_bounds=[{MIN_TP:.3f},{MAX_TP:.3f}], sl_bounds=[{MIN_SL:.3f},{MAX_SL:.3f}]"
    )

    evaluate_signal(df, feature_cols)
