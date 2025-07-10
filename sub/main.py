import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

from typing import Tuple

import sub.momentum            as mod1
import sub.trend_continuation  as mod2
import sub.mean_reversion      as mod3
import sub.lagged_return       as mod4
import sub.sentiment_volume    as mod5


USE_CLASSIFIER = True
if USE_CLASSIFIER:
    import sub.classifier as mod6
    SUBMODS = [mod1, mod2, mod3, mod4, mod5, mod6]
else:
    SUBMODS = [mod1, mod2, mod3, mod4, mod5]
N_SUBMODS = len(SUBMODS)

# ─── Base Directories ─────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
ROOT_DIR    = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
RESULTS_DIR = os.path.join(BASE_DIR, "sub-results")

# ─── Configuration ───────────────────────────────────────────────────────────
load_dotenv()
BAR_TIMEFRAME       = os.getenv("BAR_TIMEFRAME", "4Hour")
TICKERS             = os.getenv("TICKERS", "TSLA").split(",")
ML_MODEL            = os.getenv("ML_MODEL",  "sub-vote")
TIMEFRAME_MAP       = {"4Hour":"H4","2Hour":"H2","1Hour":"H1","30Min":"M30","15Min":"M15"}
CONVERTED_TIMEFRAME = TIMEFRAME_MAP.get(BAR_TIMEFRAME, BAR_TIMEFRAME)
HISTORY_PATH = os.path.join(RESULTS_DIR, f"submeta_history_{CONVERTED_TIMEFRAME}.csv")
USE_META_LABEL = os.getenv("USE_META_LABEL", "false") 

CSV_PATH  = os.path.join(ROOT_DIR, f"{TICKERS}_{CONVERTED_TIMEFRAME}.csv")
MODE       = ML_MODEL  # "sub-vote" or "sub-meta"
EXECUTION  = globals().get("EXECUTION", "backtest")

# Default return thresholds (used only for submodels returning regression)
REG_UP, REG_DOWN = 0.003, -0.003

# Set to True to force thresholds to 0.55/0.45 everywhere
# Disabled by default so thresholds can adapt to data
FORCE_STATIC_THRESHOLDS = False
STATIC_UP = 0.55
STATIC_DOWN = 0.45

META_MODEL_TYPE = os.getenv("META_MODEL_TYPE", "logreg")
VALID_META_MODELS = {"logreg", "lgbm", "xgb", "nn", "cat"}
if META_MODEL_TYPE not in VALID_META_MODELS:
    raise ValueError(f"META_MODEL_TYPE must be one of {VALID_META_MODELS}")


# ------------------------------------------------------------------------------


def vote_from_prob(p: float, up: float, down: float) -> float:
    """Voting logic for classification-style submodels."""
    return 1 if p > up else (0 if p < down else 0.5)

def vote_from_ret(r: float) -> float:
    """Voting logic for regression-style submodels."""
    return 1 if r > REG_UP else (0 if r < REG_DOWN else 0.5)

def actual_direction(df: pd.DataFrame, t: int) -> int:
    """
    Returns 1 if next close > current close, else 0.
    (Close-to-close direction, as required.)
    """
    return int(df['close'].iat[t+1] > df['close'].iat[t])

# ─── Helper to construct the chosen meta learner ─────────────────────────────
def _make_meta_model():
    """
    Return an *un-fitted* probability-calibrated classifier.
    All models expose .fit(X,y) and .predict_proba(X) with a calibrated output
    that is much better aligned across different meta-model options.
    """
    if META_MODEL_TYPE == "logreg":
        base = LogisticRegression(max_iter=2000, class_weight="balanced")
    elif META_MODEL_TYPE == "lgbm":
        from lightgbm import LGBMClassifier
        base = LGBMClassifier(
            n_estimators=400, learning_rate=0.03,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=-1
        )
    elif META_MODEL_TYPE == "xgb":
        from xgboost import XGBClassifier
        base = XGBClassifier(
            n_estimators=500, learning_rate=0.03, max_depth=4,
            subsample=0.8, colsample_bytree=0.9,
            tree_method="hist", objective="binary:logistic",
            eval_metric="logloss", random_state=42,
            use_label_encoder=False
        )
    elif META_MODEL_TYPE == "cat":
        from catboost import CatBoostClassifier
        base = CatBoostClassifier(
            iterations=600, depth=6, learning_rate=0.05,
            l2_leaf_reg=3, bagging_temperature=0.5,
            loss_function="Logloss", eval_metric="AUC",
            random_seed=42, verbose=False
        )
    else:                                  # 'nn'
        from sklearn.neural_network import MLPClassifier
        base = MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu",
            solver="adam", alpha=1e-4, batch_size="auto",
            learning_rate_init=5e-4, max_iter=800, random_state=42
        )

    # --- temperature-scale the probability output -----------------
    return CalibratedClassifierCV(
        base, method="isotonic", cv=3, n_jobs=-1
    )

def update_submeta_history(CSV_PATH, HISTORY_PATH,
                           submods=SUBMODS, window=50, verbose=True):
    """
    Keeps submeta_history_*.csv in sync with the raw price CSV.
    Handles an arbitrary number of sub-models (N_SUBMODS).
    """
    df = pd.read_csv(CSV_PATH)
    if verbose:
        print("── DEBUG: CSV_PATH =", CSV_PATH)
        print("── DEBUG: columns in CSV:", df.columns.tolist())
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    col_m  = [f"m{i+1}"   for i in range(len(submods))]
    col_acc= [f"acc{i+1}" for i in range(len(submods))]
    if verbose:
        print(HISTORY_PATH)
    if os.path.exists(HISTORY_PATH):
        if verbose:
            print("exists")
        hist = pd.read_csv(HISTORY_PATH)
        hist["timestamp"] = pd.to_datetime(hist["timestamp"], utc=True)
        last_ts   = hist["timestamp"].iloc[-1]
        start_idx = df[df["timestamp"] > last_ts].index.min()
        if pd.isna(start_idx):
            if verbose: print("[HISTORY] Already up-to-date.")
            return hist
    else:
        hist = pd.DataFrame(columns=["timestamp", *col_m, *col_acc, "meta_label"])
        start_idx = int(len(df) * 0.6)

    # pre-compute full labels/targets for ground truth accuracy
    label_dfs = {}
    for mod in submods:
        if hasattr(mod, "compute_labels"):
            label_dfs[mod] = mod.compute_labels(df).reset_index(drop=True)
        else:
            label_dfs[mod] = mod.compute_target(df).reset_index(drop=True)

    # rolling storage for accuracy -----------------------------------------
    preds_hist  = [[] for _ in submods]
    labels_hist = [[] for _ in submods]
    recs = []

    for i in range(start_idx, len(df) - 1):
        slc = df.iloc[: i + 1].reset_index(drop=True)
        preds, labels = [], []

        for mod in submods:
            if hasattr(mod, "compute_labels"):
                d   = mod.compute_labels(slc)
                tr  = d.iloc[:-1].dropna(subset=mod.FEATURES + ["label"])
                last_feats = d[mod.FEATURES].iloc[[-1]]
                model = mod.fit(tr[mod.FEATURES].values, tr["label"].values)
                p = mod.predict(model, last_feats)[0]
                lbl = label_dfs[mod]["label"].iloc[i]
            else:                                    # regression style
                d   = mod.compute_target(slc)
                tr  = d.iloc[:-1].dropna(subset=mod.FEATURES + ["target"])
                last_feats = d[mod.FEATURES].iloc[[-1]]
                model = mod.fit(tr[mod.FEATURES], tr["target"])
                p = mod.predict(model, last_feats)[0]
                lbl = int(label_dfs[mod]["target"].iloc[i] > 0)
            preds.append(p); labels.append(lbl)

        # rolling accuracies ------------------------------------------------
        for k in range(len(submods)):
            preds_hist[k].append(preds[k]); labels_hist[k].append(labels[k])

        accs = []
        for k in range(len(submods)):
            ph = np.array(preds_hist[k][-window-1:-1])
            lh = np.array(labels_hist[k][-window-1:-1])
            accs.append(np.mean(np.round(ph) == lh) if len(ph) else 0.5)

        rec = {"timestamp": slc["timestamp"].iat[-1],
               **{col_m[k]:   preds[k] for k in range(len(submods))},
               **{col_acc[k]: accs[k]  for k in range(len(submods))},
               "meta_label":  int(df["close"].iat[i+1] > df["close"].iat[i])}
        recs.append(rec)

        if verbose and (i - start_idx) % 10 == 0:
            print(f"[HISTORY] appended idx={i}")

    if recs:
        hist = pd.concat([hist, pd.DataFrame(recs)], ignore_index=True)
        hist.to_csv(HISTORY_PATH, index=False)
        if verbose: print(f"[HISTORY] Updated ⇒ {len(hist)} rows.")
    return hist



def optimize_asymmetric_thresholds(
    probs: np.ndarray, labels: np.ndarray, metric="accuracy",
    manual_thresholds=None
) -> Tuple[float, float]:
    if FORCE_STATIC_THRESHOLDS:
        return STATIC_UP, STATIC_DOWN
    if manual_thresholds is not None:
        return manual_thresholds

    # --- coarse → fine grid search (5× faster) ----------------------------
    up_coarse   = np.linspace(0.60, 0.80, 11)
    down_coarse = np.linspace(0.20, 0.40, 11)

    best_up, best_dn, best_sc = 0.55, 0.45, -np.inf
    for up in up_coarse:
        dn_candidates = down_coarse[down_coarse < up - 0.05]
        for dn in dn_candidates:
            mask  = (probs > up) | (probs < dn)
            if mask.sum() < 30:                       # skip tiny support
                continue

            if metric == "accuracy":
                score = (labels[mask] == (probs[mask] > up)).mean()
            elif metric == "f1":
                from sklearn.metrics import f1_score
                score = f1_score(labels[mask], probs[mask] > up)
            elif metric == "profit":                  # NEW ▼
                # assume +1 / −1 payoff per correct direction
                pnl = np.where(
                    probs[mask] > up,  (labels[mask] == 1).astype(int),
                    - (labels[mask] == 0).astype(int)
                )
                score = pnl.mean()
            else:                                     # roc_auc, etc.
                from sklearn.metrics import roc_auc_score
                score = roc_auc_score(labels[mask], probs[mask])

            if score > best_sc:
                best_sc, best_up, best_dn = score, up, dn

    return best_up, best_dn



def train_and_save_meta(
    cache_csv: str,
    model_pkl: str,
    threshold_pkl: str,
    metric: str = "accuracy",
    n_mods: int = N_SUBMODS,
):
    """
    Train the chosen meta-model using sub-model history (m1..mN + acc1..accN).
    Works for either 5 or 6 sub-models.
    """
    hist = pd.read_csv(cache_csv)
    X = hist[
        [f"m{i+1}"   for i in range(n_mods)] +
        [f"acc{i+1}" for i in range(n_mods)]
    ].values
    mean_prob = hist[[f"m{i+1}" for i in range(n_mods)]].mean(axis=1).values.reshape(-1, 1)
    X = np.hstack([X, mean_prob])
    y = hist["meta_label"].values

    split = int(len(y) * 0.6)
    X_train, y_train = X[:split], y[:split]
    X_val,   y_val   = X[split:], y[split:]

    meta = _make_meta_model()
    meta.fit(X_train, y_train)

    probs_val = meta.predict_proba(X_val)[:, 1]
    up, down  = optimize_asymmetric_thresholds(probs_val, y_val, metric=metric)

    with open(model_pkl, "wb") as f:
        pickle.dump(meta, f)
    with open(threshold_pkl, "wb") as f:
        pickle.dump({"up": up, "down": down, "model_type": META_MODEL_TYPE}, f)

    return meta, up, down


def backtest_submodels(
    df: pd.DataFrame,
    initial_frac: float = 0.6,
    window: int = 50,
) -> pd.DataFrame:
    """
    Walk-forward back-test of *all active* sub-models.
    Outputs only predictions m1..mN, rolling accuracies acc1..accN,
    and meta_label.  When USE_CLASSIFIER is false N=5 (legacy path).
    """
    # pick sub-models --------------------------------------------------------
    submods = [mod1, mod2, mod3, mod4, mod5] + ([mod6] if USE_CLASSIFIER else [])
    n_mods  = len(submods)

    n   = len(df)
    cut = int(n * initial_frac)
    rec = []

    # pre-compute full labels/targets once for accuracy reference ----------
    label_dfs = {}
    for mod in submods:
        if hasattr(mod, "compute_labels"):
            label_dfs[mod] = mod.compute_labels(df).reset_index(drop=True)
        else:
            label_dfs[mod] = mod.compute_target(df).reset_index(drop=True)

    # rolling containers for accuracy computation ---------------------------
    preds_history  = [[] for _ in range(n_mods)]
    labels_history = [[] for _ in range(n_mods)]

    for t in tqdm(range(cut, n - 1), desc="Submodel BT"):
        slice_df = df.iloc[: t + 1].reset_index(drop=True)

        preds, labels = [], []
        # ---------- single-bar prediction for every sub-model --------------
        for mod in submods:
            if hasattr(mod, "compute_labels"):          # classifier sub-model
                d   = mod.compute_labels(slice_df)
                tr  = d.iloc[:-1].dropna(subset=mod.FEATURES + ["label"])
                last_feats = d[mod.FEATURES].iloc[[-1]]
                model = mod.fit(tr[mod.FEATURES].values, tr["label"].values)
                p   = mod.predict(model, last_feats)[0]
                lbl = label_dfs[mod]["label"].iloc[t]
            else:                                       # regression sub-model
                d   = mod.compute_target(slice_df)
                tr  = d.iloc[:-1].dropna(subset=mod.FEATURES + ["target"])
                last_feats = d[mod.FEATURES].iloc[[-1]]
                model = mod.fit(tr[mod.FEATURES], tr["target"])
                p   = mod.predict(model, last_feats)[0]
                lbl = int(label_dfs[mod]["target"].iloc[t] > 0)
            preds.append(p); labels.append(lbl)

        # ---------- rolling accuracy histories ----------------------------
        for k in range(n_mods):
            preds_history[k].append(preds[k])
            labels_history[k].append(labels[k])

        accs = []
        for k in range(n_mods):
            ph = np.array(preds_history[k][-window - 1 : -1])
            lh = np.array(labels_history[k][-window - 1 : -1])
            accs.append(np.mean(np.round(ph) == lh) if len(ph) else 0.5)

        rec_dict = {
            "t": t,
            **{f"m{k+1}":   preds[k] for k in range(n_mods)},
            **{f"acc{k+1}": accs[k]  for k in range(n_mods)},
            "meta_label": int(df["close"].iat[t + 1] > df["close"].iat[t]),
        }
        rec.append(rec_dict)

    return pd.DataFrame(rec).set_index("t")

# ─── NEW: Walk-forward meta back-test ────────────────────────────────────────
def walkforward_meta_backtest(
    cache_csv: str,
    n_mods: int = N_SUBMODS,
    initial_frac: float = 0.6,
    metric: str = "accuracy",
):
    """
    Incremental (i+1) walk-forward evaluation:

    • Train on the first ⌊initial_frac·N⌋ rows
    • Predict row i
    • Append row i to the training set
    • Repeat until the end of the history file
    """
    hist = pd.read_csv(cache_csv)
    feature_cols = (
        [f"m{k}"   for k in range(1, n_mods + 1)] +
        [f"acc{k}" for k in range(1, n_mods + 1)]
    )
    N = len(hist)
    start = int(N * initial_frac)
    if start < 10:
        raise ValueError("History too short for walk-forward meta test.")

    records = []

    for i in tqdm(range(start, N), desc="Walk-forward (meta)"):
        # ─── ❶ train on rows [: i] ─────────────────────────────────────────
        X_train = hist.loc[:i - 1, feature_cols].values
        # add consensus (mean of all sub-model probabilities) as an extra column
        mean_tr = (
            hist.loc[:i - 1, [f"m{k}" for k in range(1, n_mods + 1)]]
                .mean(axis=1)
                .values.reshape(-1, 1)
        )
        X_train = np.hstack([X_train, mean_tr])

        y_train = hist.loc[:i - 1, "meta_label"].values

        meta = _make_meta_model()
        meta.fit(X_train, y_train)

        # ─── ❷ choose asymmetric thresholds on an *internal* validation split
        #      (20 % of the current training slice)
        split_val = max(1, int(len(y_train) * 0.2))
        X_val, y_val = X_train[-split_val:], y_train[-split_val:]
        probs_val = meta.predict_proba(X_val)[:, 1]
        up, down = optimize_asymmetric_thresholds(probs_val, y_val, metric=metric)

        # ─── ❸ predict the held-out row i ──────────────────────────────────
        x_row = hist.loc[i, feature_cols].values.reshape(1, -1)
        x_row = np.hstack(
            [x_row,
            np.array([[hist.loc[i, [f"m{k}" for k in range(1, n_mods + 1)]].mean()]])]
        )
        prob_up = float(meta.predict_proba(x_row)[0, 1])

        action = "BUY" if prob_up > up else ("SELL" if prob_up < down else "HOLD")

        # ─── ❹ collect results ────────────────────────────────────────────
        rec = {
            "timestamp": hist["timestamp"].iloc[i],
            "meta_prob": prob_up,
            "action": action,
            "up_thresh": up,
            "down_thresh": down,
        }
        for k in range(1, n_mods + 1):
            rec[f"m{k}"]   = hist[f"m{k}"].iloc[i]
            rec[f"acc{k}"] = hist[f"acc{k}"].iloc[i]
        records.append(rec)

    return pd.DataFrame(records)


# ───────────────────────── run_backtest ──────────────────────────────────────
def _enforce_position_rules(actions, start_open=False):
    """Adjust BUY/SELL signals based on existing position state."""
    out = []
    open_pos = start_open
    for act in actions:
        a = str(act).upper()
        if a == "BUY":
            if open_pos:
                out.append("NONE")
            else:
                out.append("BUY")
                open_pos = True
        elif a == "SELL":
            if not open_pos:
                out.append("NONE")
            else:
                out.append("SELL")
                open_pos = False
        else:
            out.append(a)
    return out


def run_backtest(return_df: bool = False, start_open: bool = False):
    """
    Back-test either classic sub-vote or the walk-forward meta pipeline.
    """
    df_orig = pd.read_csv(CSV_PATH)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ─── Assemble active sub-model list ────────────────────────────────────
    submods = [mod1, mod2, mod3, mod4, mod5] + ([mod6] if USE_CLASSIFIER else [])
    n_mods  = len(submods)

    # ----------------------------------------------------------------------
    # ---------- Classic majority-vote pipeline ----------------------------
    # ----------------------------------------------------------------------
    if MODE == "sub-vote":
        back = backtest_submodels(df_orig, initial_frac=0.6)
        VOTE_UP, VOTE_DOWN = 0.55, 0.45

        # one vote column per active sub-model -----------------------------
        for k, mod in enumerate(submods, start=1):
            pred_col, vote_col = f"m{k}", f"v{k}"
            if hasattr(mod, "compute_labels"):          # classifier style
                back[vote_col] = back[pred_col].map(
                    lambda p: vote_from_prob(p, VOTE_UP, VOTE_DOWN)
                )
            else:                                       # regression style
                back[vote_col] = back[pred_col].map(vote_from_ret)

        vote_cols    = [f"v{k}" for k in range(1, n_mods + 1)]
        majority_req = (n_mods // 2) + 1

        back["buy_votes"]  = (back[vote_cols] == 1).sum(axis=1)
        back["sell_votes"] = (back[vote_cols] == 0).sum(axis=1)
        back["action"] = back.apply(
            lambda r: 1 if r.buy_votes  >= majority_req else
                      (0 if r.sell_votes >= majority_req else 0.5),
            axis=1,
        )

        back = back.reset_index().rename(columns={"index": "t"})
        back["timestamp"] = pd.to_datetime(
            df_orig["timestamp"].iloc[back["t"] + 1]
        )
        df_out = back[
            ["timestamp", *[f"m{k}" for k in range(1, n_mods + 1)], "action"]
        ]
        df_out["action"] = df_out["action"].map({1: "BUY", 0: "SELL", 0.5: "HOLD"})

    # ----------------------------------------------------------------------
    # ---------- NEW: Walk-forward meta pipeline ---------------------------
    # ----------------------------------------------------------------------
    else:
        cache_csv = os.path.join(
            RESULTS_DIR, f"submeta_history_{CONVERTED_TIMEFRAME}.csv"
        )
        # ensure history is up-to-date
        update_submeta_history(CSV_PATH, cache_csv, submods=submods, verbose=True)

        df_out = walkforward_meta_backtest(
            cache_csv=cache_csv,
            n_mods=n_mods,
            initial_frac=0.6,
            metric="accuracy",
        )

        out_fn = os.path.join(RESULTS_DIR, f"meta_results_walkforward.csv")
        df_out.to_csv(out_fn, index=False, float_format="%.8f")
        print(f"[RESULT] Walk-forward results saved to {out_fn}")

    # ----------------------------------------------------------------------
    if return_df:
        df_out["action"] = _enforce_position_rules(df_out["action"].tolist(), start_open)
        return df_out

    

# ─── tiny helpers (unchanged) ────────────────────────────────────────────────
def _prepare_train(df_slice: pd.DataFrame, module):
    """Return (X, y) for the given sub-module on df_slice[:-1]."""
    if hasattr(module, "compute_labels"):
        d  = module.compute_labels(df_slice)
        tr = d.iloc[:-1].dropna(subset=module.FEATURES + ["label"])
        return tr[module.FEATURES].values, tr["label"].values
    d  = module.compute_target(df_slice)
    tr = d.iloc[:-1].dropna(subset=module.FEATURES + ["target"])
    return tr[module.FEATURES], tr["target"]


def _row(df_slice: pd.DataFrame, module, idx: int | None = None):
    d = (
        module.compute_labels(df_slice)
        if hasattr(module, "compute_labels")
        else module.compute_target(df_slice)
    )

    if idx is not None and idx in d.index:
        return d.loc[[idx], module.FEATURES]

    # graceful fall-back when the requested index is missing
    return d[module.FEATURES].iloc[[-1]]



# ────────────────────────── run_live (verbose always on) ────────────────────
def run_live(return_result: bool = True, position_open: bool = False):
    """
    Produce one live prediction.

    • MODE == "sub-vote" → majority-vote result (0 / 0.5 / 1)
    • MODE == "sub-meta" → tuple (probability, "BUY"/"HOLD"/"SELL")

    Console output is always enabled.
    """
    # ─── Choose active sub-models -----------------------------------------
    submods = [mod1, mod2, mod3, mod4, mod5]
    if "USE_CLASSIFIER" in globals() and USE_CLASSIFIER:
        submods.append(mod6)                # mod6 == sub.classifier
    n_mods = len(submods)

    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    t = len(df) - 1                         # last bar index

    # ─── Majority-vote path ───────────────────────────────────────────────
    if MODE == "sub-vote":
        VOTE_UP, VOTE_DOWN = 0.55, 0.45

        # current-bar predictions for each sub-model -----------------------
        preds = []
        for mod in submods:
            preds.append(
                mod.predict(
                    mod.fit(*_prepare_train(df, mod)),
                    _row(df, mod, t)
                )[0]
            )

        # translate predictions → votes -----------------------------------
        votes = []
        for k, mod in enumerate(submods):
            if hasattr(mod, "compute_labels"):           # classifier style
                votes.append(vote_from_prob(preds[k], VOTE_UP, VOTE_DOWN))
            else:                                        # regression style
                votes.append(vote_from_ret(preds[k]))

        buy, sell   = votes.count(1), votes.count(0)
        majority_req = (n_mods // 2) + 1                # 3 of 5, 4 of 6, …
        majority     = 1 if buy  >= majority_req else \
                       (0 if sell >= majority_req else 0.5)
        action = "BUY" if majority == 1 else ("SELL" if majority == 0 else "HOLD")
        action = _enforce_position_rules([action], position_open)[0]
        print(f"[LIVE-VOTE] result = {action}")
        return action if return_result else None

    # ─── Meta-model path ──────────────────────────────────────────────────
    hist = update_submeta_history(
        CSV_PATH, HISTORY_PATH, submods=submods, verbose=True
    )

    meta_pkl   = os.path.join(RESULTS_DIR, "meta_model.pkl")
    thresh_pkl = os.path.join(RESULTS_DIR, "meta_thresholds.pkl")

    must_retrain = True
    if os.path.exists(meta_pkl) and os.path.exists(thresh_pkl):
        with open(thresh_pkl, "rb") as f:
            thresh_dict = pickle.load(f)
        if thresh_dict.get("model_type") == META_MODEL_TYPE:
            with open(meta_pkl, "rb") as f:
                meta = pickle.load(f)
            up, down   = thresh_dict["up"], thresh_dict["down"]
            must_retrain = False
            print(f"[META] Loaded cached {META_MODEL_TYPE} model")

    if must_retrain:
        meta, up, down = train_and_save_meta(HISTORY_PATH, meta_pkl, thresh_pkl)
        print(f"[META] Re-trained {META_MODEL_TYPE}  BUY>{up:.2f} / SELL<{down:.2f}")

    # current sub-model preds + rolling accuracies -------------------------
    window = 50
    p, accs = [], []

    for mod in submods:
        if hasattr(mod, "compute_labels"):               # classifier sub-model
            d  = mod.compute_labels(df)
            pi = mod.predict(mod.fit(*_prepare_train(df, mod)), _row(df, mod, t))[0]

            past_labels = d["label"].iloc[-(window + 1):-1].values
            past_preds  = d["label"].iloc[-(window + 1):-1].index.map(
                lambda j: mod.predict(
                    mod.fit(*_prepare_train(d.iloc[: j + 1], mod)),
                    _row(d, mod, j)
                )[0]
            ).values if len(d) > window else np.array([])

            acc = (np.round(past_preds) == past_labels).mean() if len(past_preds) else 0.5
        else:                                            # regression sub-model
            d  = mod.compute_target(df)
            pi = mod.predict(mod.fit(*_prepare_train(df, mod)), _row(df, mod, t))[0]

            past_labels = (d["target"].iloc[-(window + 1):-1] > 0).astype(int).values
            past_preds  = d["target"].iloc[-(window + 1):-1].index.map(
                lambda j: int(
                    mod.predict(
                        mod.fit(*_prepare_train(d.iloc[: j + 1], mod)),
                        _row(d, mod, j)
                    )[0] > 0
                )
            ).values if len(d) > window else np.array([])

            acc = (np.round(past_preds) == past_labels).mean() if len(past_preds) else 0.5

        p.append(pi)
        accs.append(acc)

    feat_vec = np.array(p + accs + [np.mean(p)]).reshape(1, -1)  # add consensus feature
    prob     = float(meta.predict_proba(feat_vec)[0, 1])
    action = "BUY" if prob > up else ("SELL" if prob < down else "HOLD")
    action = _enforce_position_rules([action], position_open)[0]
    print(f"[LIVE] P(up)={prob:.3f}  →  {action}  (BUY>{up:.2f} / SELL<{down:.2f})")

    return (prob, action) if return_result else None



# ─── Entrypoint ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Run live prediction mode")
    parser.add_argument("--return_df", action="store_true", help="Return DataFrame instead of writing CSV")
    args = parser.parse_args()

    if args.live:
        run_live()
    else:
        run_backtest(return_df=args.return_df)

