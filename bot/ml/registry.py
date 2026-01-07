import os
import forest
import logging
import xgboost as xgb
import torch.nn as nn
import lightgbm as lgb
from tensorflow import keras
from tensorflow.keras import layers
from forest import ML_MODEL, SELECTION_MODEL
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


logger = logging.getLogger(__name__)

N_ESTIMATORS = 100
RANDOM_SEED = 42

def parse_ml_models(override: str | None = None) -> list[str]:
    ml_raw = override if override is not None else os.getenv("ML_MODEL", ML_MODEL)
    raw = [m.strip().lower() for m in ml_raw.split(",") if m.strip()]

    if "all" in raw:
        raw += [
            "forest",
            "xgboost",
            "lightgbm",
            "catboost",
            "lstm",
            "transformer",
        ]
    if "all_cls" in raw:
        raw += [
            "forest_cls",
            "xgboost_cls",
            "catboost_cls",
            "lightgbm_cls",
            "transformer_cls",
        ]

    canonical: list[str] = []
    for m in raw:
        match m:
            case "lgbm" | "lightgbm":                 canonical.append("lightgbm")
            case "lgbm_cls" | "lightgbm_cls":         canonical.append("lightgbm_cls")
            case "lgbm_multi" | "lightgbm_multi":     canonical.append("lightgbm_multi")
            case "rf" | "randomforest":               canonical.append("forest")
            case "rf_cls" | "forest_cls":             canonical.append("forest_cls")
            case "xgb_cls" | "xgboost_cls":           canonical.append("xgboost_cls")
            case "xgb_multi" | "xgboost_multi":       canonical.append("xgboost_multi")
            case "cb" | "catboost":                   canonical.append("catboost")
            case "cb_cls" | "catboost_cls":           canonical.append("catboost_cls")
            case "cb_multi" | "catboost_multi":       canonical.append("catboost_multi")
            case "classifier" | "transformer_cls":    canonical.append("transformer_cls")
            case "sub_vote" | "sub-vote":             canonical.append("sub-vote")
            case "sub_meta" | "sub-meta":             canonical.append("sub-meta")
            case _:                                   canonical.append(m)

    # de-dupe while preserving order
    seen, out = set(), []
    for x in canonical:
        if x not in seen:
            seen.add(x)
            out.append(x)

    # strip meta tokens if they slipped through
    out = [m for m in out if m not in ("all", "all_cls")]

    logging.debug("[parse_ml_models] ML_MODEL=%s  ➜  %s", ML_MODEL, out)
    return out

def get_ml_models_for_ticker(ticker: str | None = None, for_selection: bool = False) -> list[str]:
    """Return ML models considering per-ticker and selection overrides."""
    if for_selection and SELECTION_MODEL:
        return parse_ml_models(SELECTION_MODEL)
    if ticker and ticker in forest.TICKER_ML_OVERRIDES:
        return parse_ml_models(forest.ICKER_ML_OVERRIDES[ticker])
    return parse_ml_models()

def get_single_model(model_name, input_shape=None, num_features=None, lstm_seq=60):
    if model_name in ["forest", "rf", "randomforest"]:
        return RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
    elif model_name == "xgboost":
        return xgb.XGBRegressor(
            n_estimators=114, 
            max_depth=9, 
            learning_rate=0.14264252588219034,
            subsample=0.5524803023252148,
            colsample_bytree=0.7687841723045249,
            gamma=0.5856035407199236,
            reg_alpha=0.5063880221467401,
            reg_lambda=0.0728996118523866,
        )
    elif model_name == "catboost":
        return CatBoostRegressor(iterations=600, depth=6, learning_rate=0.05,
            l2_leaf_reg=3, bagging_temperature=0.5,
            loss_function="RMSE", eval_metric="RMSE",
            random_seed=42, verbose=False)
    elif model_name == "lightgbm":
        return lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
    elif model_name == "forest_cls":
        return RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                      random_state=RANDOM_SEED)

    elif model_name == "xgboost_cls":
        return xgb.XGBClassifier(n_estimators=N_ESTIMATORS,
                             random_state=RANDOM_SEED,
                             use_label_encoder=False, eval_metric="logloss")
    elif model_name == "catboost_cls":
        return CatBoostClassifier(iterations=600, depth=6, learning_rate=0.05,
            l2_leaf_reg=3, bagging_temperature=0.5,
            loss_function="Logloss", eval_metric="AUC",
            random_seed=42, verbose=False)
    elif model_name == "lightgbm_cls":
        return lgb.LGBMClassifier(n_estimators=N_ESTIMATORS,
                              random_state=RANDOM_SEED)
    elif model_name == "lstm":
        if input_shape is None and num_features is not None:
            input_shape = (lstm_seq, num_features)
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Masking(mask_value=0.),
            layers.LSTM(128, return_sequences=True, dropout=0.25, recurrent_dropout=0.25),
            layers.BatchNormalization(),
            layers.LSTM(64, return_sequences=True, dropout=0.20, recurrent_dropout=0.20),
            layers.BatchNormalization(),
            layers.LSTM(32, return_sequences=False, dropout=0.10, recurrent_dropout=0.10),
            layers.BatchNormalization(),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.15),
            layers.Dense(1)
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mae")
        return model
    elif model_name == "transformer":
        class LargeTransformerRegressor(nn.Module):
            def __init__(self, num_features, seq_len=60):
                super().__init__()
                d_model = 64
                self.embedding = nn.Linear(num_features, d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=8,
                    dim_feedforward=256,
                    dropout=0.20,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
                self.dropout = nn.Dropout(0.20)
                self.fc1 = nn.Linear(d_model, 32)
                self.relu = nn.ReLU()
                self.out = nn.Linear(32, 1)
            def forward(self, x):
                em = self.embedding(x)
                out = self.transformer(em)
                out = out.mean(dim=1)
                out = self.dropout(out)
                out = self.fc1(out)
                out = self.relu(out)
                return self.out(out)
        return LargeTransformerRegressor
    elif model_name == "transformer_cls":
        class LargeTransformerClassifier(nn.Module):
            def __init__(self, num_features, seq_len=60):
                super().__init__()
                d_model = 64
                self.embedding = nn.Linear(num_features, d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=8,
                    dim_feedforward=256,
                    dropout=0.20,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
                self.dropout = nn.Dropout(0.20)
                self.fc1 = nn.Linear(d_model, 32)
                self.relu = nn.ReLU()
                self.out = nn.Linear(32, 2)
            def forward(self, x):
                em = self.embedding(x)
                out = self.transformer(em)
                out = out.mean(dim=1)
                out = self.dropout(out)
                out = self.fc1(out)
                out = self.relu(out)
                return self.out(out)
        return LargeTransformerClassifier
    else:
        logging.warning(f"Unknown model type {model_name}. Using RandomForest.")
        return RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)