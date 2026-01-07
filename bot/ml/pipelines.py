from forest import USE_HALF_LIFE, POSSIBLE_FEATURE_COLS, DISABLE_PRED_CLOSE
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import bot.ml.registry as registry
import lightgbm as lgb
import xgboost as xgb
import logging
import tqdm

N_ESTIMATORS = 100
RANDOM_SEED = 42

logger = logging.getLogger(__name__)

def roll_predicted_close(
    ticker: str,
    df: pd.DataFrame,
    first_missing_pred_idx: int | None = None,
    first_new_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Regenerates df['predicted_close'] in-place (rolling ML), starting from the earliest:
      - first missing predicted_close, OR
      - first newly-added candle timestamp (more robust than len diff)

    Uses the same gating logic as before:
      - only runs when has_regressor and DISABLE_PRED_CLOSE == "False"
    """

    def get_features(local_df: pd.DataFrame) -> list[str]:
        exclude = {"predicted_close"}
        return [c for c in POSSIBLE_FEATURE_COLS if c in local_df.columns and c not in exclude]

    if "predicted_close" not in df.columns:
        df["predicted_close"] = np.nan

    ml_models = registry.get_ml_models_for_ticker(ticker)
    has_regressor = any(
        m in ["xgboost", "forest", "rf", "randomforest", "lstm", "transformer", "catboost"]
        for m in ml_models
    )

    # keep same gating condition
    if not (has_regressor and DISABLE_PRED_CLOSE == "False"):
        return df

    # choose start of "new rows" by timestamp, if provided
    if first_new_ts is not None and "timestamp" in df.columns and not df.empty:
        n_start_new = int(df["timestamp"].searchsorted(first_new_ts, side="left"))
    else:
        n_start_new = len(df)

    if first_missing_pred_idx is not None:
        n_start = min(first_missing_pred_idx, n_start_new)
    else:
        n_start = n_start_new

    start = max(1, n_start)
    end = len(df)
    total = max(0, end - start)
    iterator = (
        tqdm(
            range(start, end),
            desc=f"[{ticker}] predicted_close",
            unit="row",
            dynamic_ncols=True,
            leave=False,
        )
        if tqdm and total > 0
        else range(start, end)
    )

    logging.info(f"[{ticker}] Rolling ML predicted_close from index {start} over {end - start} rows…")

    if len(df) - n_start > 0:
        logging.info(f"[{ticker}] Rolling ML predicted_close for {len(df) - n_start} new row(s)…")

        feature_cols = get_features(df)

        for i in iterator:
            try:
                X_train = df.iloc[:i][feature_cols]
                y_train = df.iloc[:i]["close"]

                m = ml_models[0]
                if m in ["forest", "rf", "randomforest"]:
                    mdl = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
                elif m == "xgboost":
                    mdl = xgb.XGBRegressor(
                        n_estimators=114,
                        max_depth=9,
                        learning_rate=0.14264252588219034,
                        subsample=0.5524803023252148,
                        colsample_bytree=0.7687841723045249,
                        gamma=0.5856035407199236,
                        reg_alpha=0.5063880221467401,
                        reg_lambda=0.0728996118523866,
                    )
                elif m == "catboost":
                    mdl = CatBoostRegressor(
                        iterations=600,
                        depth=6,
                        learning_rate=0.05,
                        l2_leaf_reg=3,
                        bagging_temperature=0.5,
                        loss_function="RMSE",
                        eval_metric="RMSE",
                        random_seed=RANDOM_SEED,
                        verbose=False,
                    )
                elif m in ["lightgbm", "lgbm"]:
                    mdl = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
                elif m == "lstm":
                    mdl = registry.get_single_model(
                        "lstm",
                        input_shape=(X_train.shape[0], X_train.shape[1]),
                        num_features=X_train.shape[1],
                        lstm_seq=X_train.shape[0],
                    )
                elif m == "transformer":
                    Transformer = registry.get_single_model(
                        "transformer",
                        num_features=X_train.shape[1],
                        lstm_seq=X_train.shape[0],
                    )
                    mdl = Transformer(num_features=X_train.shape[1], seq_len=X_train.shape[0])
                else:
                    mdl = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)

                weights = add_weights(len(X_train))  # kept as-is (even if unused)
                fixed_fit(X_train, y_train, mdl)

                X_pred = df.iloc[[i - 1]][feature_cols]
                df.at[i, "predicted_close"] = mdl.predict(X_pred)[0]
            except Exception as e:
                logging.error(f"[{ticker}] predicted_close error idx={i}: {e}")
                df.at[i, "predicted_close"] = np.nan

    return df

def make_pipeline(base_est):
    return Pipeline([
        ("sc",  StandardScaler()),
        ("cal", CalibratedClassifierCV(base_est, method="isotonic", cv=3)),
    ])

def fixed_fit(X_train, Y_train, model):
    if USE_HALF_LIFE == False:
        return model.fit(X_train, Y_train)
    else:
        return model.fit(X_train, Y_train, sample_weight=add_weights(len(X_train)))
    
HALF_LIFE = 750
def add_weights(n_samples: int) -> np.ndarray:
    ages = np.arange(n_samples - 1, -1, -1)
    return np.exp(-np.log(2) * ages / HALF_LIFE)