
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier 
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier, LGBMRegressor
from forest import HORIZON, POSSIBLE_FEATURE_COLS
from xgboost import XGBClassifier, XGBClassifier
from sklearn.preprocessing import StandardScaler
import bot.features.engineering as engineering
from sklearn.linear_model import RidgeCV
import bot.features.targets as targets
import bot.stuffs.candles as candles
import bot.ml.pipelines as pipelines
import bot.ml.registry as registry
import bot.ml.models as models
from tensorflow import keras
import forest as forest
import torch.nn as nn
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import numpy as np
import logging
import torch

api = forest.api

N_ESTIMATORS = 100
RANDOM_SEED = 42
SUBMETA_USE_ACTION = True

logger = logging.getLogger(__name__)

def _fit_base_classifiers(
    X_scaled:  np.ndarray,
    y_bin:     np.ndarray,
    seq_len:   int,
    model_names: set[str] | None = None
) -> dict[str, object]:
    """Fit only the requested base classifiers."""
    if model_names is None:
        model_names = {"forest_cls", "xgboost_cls", "lightgbm_cls", "transformer_cls", "catboost_cls"}

    models: dict[str, object] = {}
    pos_weight = ((len(y_bin) - y_bin.sum()) / y_bin.sum()) if y_bin.sum() else 1.0

    # ── tree models ────────────────────────────────────────────────────────
    if "forest_cls" in model_names:
        rf_raw = RandomForestClassifier(
            n_estimators=400,          # a bit deeper
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED,
        )
        pipe = pipelines.make_pipeline(rf_raw)
        pipelines.fixed_fit(X_scaled, y_bin, pipe)
        models["forest_cls"] = pipe


    # ▸ xgboost
    if "xgboost_cls" in model_names:
        xgb_raw = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            eval_metric="logloss",
            scale_pos_weight=pos_weight,
            tree_method="hist",
        )
        pipe = pipelines.make_pipeline(xgb_raw)
        pipelines.fixed_fit(X_scaled, y_bin, pipe)
        models["xgboost_cls"] = pipe

    if "catboost_cls" in model_names:
        cb_raw = CatBoostClassifier(iterations=600, depth=6, learning_rate=0.05,
                                    l2_leaf_reg=3, bagging_temperature=0.5,
                                    loss_function="Logloss", eval_metric="AUC",
                                    random_seed=RANDOM_SEED, verbose=False)
        pipe = pipelines.make_pipeline(cb_raw)
        pipelines.fixed_fit(X_scaled, y_bin, pipe)
        models["catboost_cls"] = pipe


    # ▸ lightgbm
    if "lightgbm_cls" in model_names:
        lgb_raw = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            is_unbalance=True,
        )
        pipe = pipelines.make_pipeline(lgb_raw)
        pipelines.fixed_fit(X_scaled, y_bin, pipe)
        models["lightgbm_cls"] = pipe

    # ── transformer classifier ────────────────────────────────────────────
    if "transformer_cls" in model_names:
        Xs, ys = engineering.series_to_supervised(X_scaled, y_bin, seq_len)
        if len(Xs):
            TCls = registry.get_single_model("transformer_cls",
                                    num_features=X_scaled.shape[1])
            net = TCls(num_features=X_scaled.shape[1], seq_len=seq_len)
            opt = torch.optim.Adam(net.parameters(), lr=5e-4)
            loss = nn.CrossEntropyLoss()

            X_t = torch.tensor(Xs, dtype=torch.float32)
            y_t = torch.tensor(ys, dtype=torch.long)

            net.train()
            for _ in range(10):  # very small budget
                opt.zero_grad()
                loss(net(X_t), y_t).backward()
                opt.step()

            net.eval()
            models["transformer_cls"] = net

    return models

def _train_meta_classifier(models: dict[str, object],
                           X_scaled: np.ndarray,
                           y_bin:    np.ndarray,
                           seq_len:  int) -> tuple[CalibratedClassifierCV, list[str]]:
    """Train a simple meta classifier using out-of-fold probabilities."""
    from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
    from sklearn.base import clone

    preds: list[np.ndarray] = []
    names_used: list[str] = []
    cv = TimeSeriesSplit(n_splits=4, test_size=seq_len)

    for name in ("forest_cls", "xgboost_cls", "lightgbm_cls", "catboost_cls"):
        if name not in models:
            continue
        pipe = models[name]   
        p = cross_val_predict(
                pipe, X_scaled, y_bin,
                cv=cv, method="predict_proba"
        )[:, 1]
        preds.append(p)
        names_used.append(name)

    if not preds:
        raise ValueError("No base classifier predictions available for meta model")

    meta_X = np.column_stack(preds)

    # ① fit ridge on the stacked OOF probabilities
    ridge = RidgeCV().fit(meta_X, y_bin)

    # ② calibrate that ridge so its output is a true probability
    meta_cal = CalibratedClassifierCV(ridge, method="isotonic", cv=3)
    meta_cal.fit(meta_X, y_bin)

    return meta_cal, names_used


def _predict_meta(models: dict[str, object],
                  meta_model: CalibratedClassifierCV,
                  names_used: list[str],
                  X_last_scaled: np.ndarray,
                  X_scaled:      np.ndarray,
                  seq_len:       int) -> float:
    base_now: list[float] = []
    for name in names_used:
        if name == "transformer_cls":
            seq = X_scaled[-seq_len:].reshape(1, seq_len, -1)
            prob = torch.softmax(
                models[name](torch.tensor(seq, dtype=torch.float32)),
                dim=1
            )[0, 1].item()
        else:
            prob = models[name].predict_proba(X_last_scaled)[0, 1]
        base_now.append(prob)

    meta_prob = meta_model.predict_proba([base_now])[0, 1]
    return float(meta_prob)

def train_and_predict(df: pd.DataFrame, return_model_stack=False, ticker: str | None = None, for_selection: bool = False):
    df = candles.limit_df_rows(df)
    if 'sentiment' in df.columns and df["sentiment"].dtype == object:
        try:
            df["sentiment"] = df["sentiment"].astype(float)
        except Exception as e:
            logging.error(f"Cannot convert sentiment column to float: {e}")
            return None

    df_feat = engineering.add_features(df)
    df_feat = engineering.compute_custom_features(df_feat)

    available_cols = [c for c in POSSIBLE_FEATURE_COLS if c in df_feat.columns]

    if 'close' not in df_feat.columns:
        logging.error("No 'close' in DataFrame, cannot create target.")
        return None

    df_full = targets._ensure_multi_horizon_targets(df_feat.copy())
    horizon_gap = HORIZON
    df_full['ret1']    = df_full['close'].pct_change()
    df_full['direction'] = (df_full['close'].shift(-horizon_gap) > df_full['close']).astype(int)
    df_full['target']  = df_full['close'].shift(-horizon_gap)

    df_train = df_full.dropna(subset=['target']).copy()
    if len(df_train) < 70:
        logging.error("Not enough rows after shift to train. Need more candles.")
        return None

    X = df_train[available_cols]
    y = df_train['target']
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    nan_counts    = X.isna().sum()
    inf_counts    = X.applymap(np.isinf).sum()
    target_nan    = y.isna().sum()
    target_inf    = np.isinf(y).sum()
    logging.info(f"Features NaN per column:\n{nan_counts[nan_counts>0].to_dict()}")
    logging.info(f"Features Inf per column:\n{inf_counts[inf_counts>0].to_dict()}")
    logging.info(f"Target NaNs: {target_nan}, Target INFs: {target_inf}")

    if nan_counts.any() or inf_counts.any() or target_nan > 0 or target_inf > 0:
        logging.error("→ Skipping LSTM/Transformer due to NaN/Inf above.")
        use_transformer = False
    else:
        use_transformer = True

    last_row_features = df_full.iloc[-1][available_cols]
    last_row_df       = pd.DataFrame([last_row_features], columns=available_cols)
    last_X_np         = np.array(last_row_df)

    ml_models = registry.get_ml_models_for_ticker(ticker if not for_selection else None, for_selection=for_selection)

    if "sub-vote" in ml_models or "sub-meta" in ml_models:
        mode    = "sub-vote" if "sub-vote" in ml_models else "sub-meta"
        position_open = False
        if ticker is not None:
            try:
                pos_qty = float(api.get_position(ticker).qty)
                position_open = abs(pos_qty) > 0
            except Exception:
                position_open = False
        result = models.call_sub_main(ticker, df_full, 0, position_open=position_open)
        logging.info(f"Sub-{mode} run_live returned: {result!r}")

        # ── decide what we pass back upstream ───────────────────────────────
        if mode == "sub-meta":
            if SUBMETA_USE_ACTION:
                action = result[1] if isinstance(result, tuple) else result
                return action.upper()
            else:
                prob = result[0] if isinstance(result, tuple) else result
                return prob
        else:
            return result
    out_preds = []
    out_names = []

    # Sequence models
    seq_len = 60

    use_meta_model = sorted(ml_models) == sorted(["forest", "xgboost", "lightgbm", "lstm", "transformer", "catboost"])

    multi_classifier_set = {"xgboost_multi", "catboost_multi", "lightgbm_multi"}
    classifier_set = {"forest_cls", "xgboost_cls",
                    "lightgbm_cls", "transformer_cls", "catboost_cls"}

    if any(m in multi_classifier_set for m in ml_models):
        feature_cols, target_cols = targets._get_feature_and_target_columns(df_full)

        if not feature_cols:
            logging.error("No feature columns available for multi-class classification.")
            return None
        if not target_cols:
            logging.error("No target columns available for multi-class classification.")
            return None

        target_col = target_cols[0]
        X_multi = df_full[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        y_multi = df_full[target_col].astype(int)

        valid_mask = df_full["close"].shift(-HORIZON).notna()
        if valid_mask.sum() < 2:
            logging.error("Not enough data to train multi-class models.")
            return None

        scaler_multi = StandardScaler().fit(X_multi[valid_mask])
        X_multi_scaled = scaler_multi.transform(X_multi[valid_mask])
        X_last_multi = scaler_multi.transform(X_multi.iloc[[-1]])

        label_to_index = {-1: 0, 0: 1, 1: 2}
        y_encoded = y_multi[valid_mask].map(label_to_index).to_numpy()

        proba_preds: list[np.ndarray] = []

        if "xgboost_multi" in ml_models:
            xgb_multi = XGBClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                num_class=3,
                random_state=RANDOM_SEED,
                tree_method="hist",
            )
            pipelines.fixed_fit(X_multi_scaled, y_encoded, xgb_multi)
            proba_preds.append(xgb_multi.predict_proba(X_last_multi)[0])

        if "catboost_multi" in ml_models:
            cb_multi = CatBoostClassifier(
                iterations=600,
                depth=6,
                learning_rate=0.05,
                l2_leaf_reg=3,
                bagging_temperature=0.5,
                loss_function="MultiClass",
                eval_metric="MultiClass",
                random_seed=RANDOM_SEED,
                verbose=False,
            )
            pipelines.fixed_fit(X_multi_scaled, y_encoded, cb_multi)
            proba_preds.append(cb_multi.predict_proba(X_last_multi)[0])

        if "lightgbm_multi" in ml_models:
            lgb_multi = LGBMClassifier(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=64,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multiclass",
                num_class=3,
                random_state=RANDOM_SEED,
            )
            pipelines.fixed_fit(X_multi_scaled, y_encoded, lgb_multi)
            proba_preds.append(lgb_multi.predict_proba(X_last_multi)[0])

        if not proba_preds:
            logging.error(f"Unknown multi-classifier in request: {ml_models}")
            return None

        avg_proba = np.mean(np.vstack(proba_preds), axis=0)
        idx_to_label = {0: -1, 1: 0, 2: 1}
        pred_idx = int(np.argmax(avg_proba))
        return idx_to_label.get(pred_idx, 0)

    if any(m in classifier_set for m in ml_models):

        price_like = {"open", "high", "low", "close", "vwap"}
        cls_cols = [c for c in available_cols if c not in price_like]
        X_cls = df_train[cls_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        y_bin = df_train["direction"].values

        # Fit the scaler on all available training rows so that the feature
        # matrix used for model fitting aligns with the label vector.  The
        # final row is reserved for prediction only and removed when
        # training the classifiers below.
        scaler = StandardScaler().fit(X_cls)
        X_scaled = scaler.transform(X_cls)
        X_last_s = X_scaled[[-1]]

        # --------------------------------------------------
        #  a)   SIMPLE single-classifier request
        # --------------------------------------------------
        if len(ml_models) == 1 and ml_models[0] != "all_cls":
            name = ml_models[0]
            base_models = _fit_base_classifiers(
                X_scaled[:-1], y_bin[:-1], seq_len, set(ml_models)
            )
            if name not in base_models:
                logging.error(f"Unknown classifier {name}.")
                return 0.5
            if name == "transformer_cls":
                seq = torch.tensor(
                    X_scaled[-seq_len:].reshape(1, seq_len, -1), dtype=torch.float32
                )
                p = torch.softmax(base_models[name](seq), dim=1)[0, 1].item()
            else:
                p = base_models[name].predict_proba(X_last_s)[0, 1]
            return float(p)

        # --------------------------------------------------
        #  b)   STACK  … all_cls / explicit 4-list
        # --------------------------------------------------
        wanted = {"forest_cls", "xgboost_cls", "lightgbm_cls", "transformer_cls", "catboost_cls"}
        if (ml_models == ["all_cls"]) or set(ml_models) == wanted:

            base_models = _fit_base_classifiers(
                X_scaled[:-1], y_bin[:-1], seq_len, wanted
            )
            meta_model, names_used = _train_meta_classifier(
                base_models, X_scaled[:-1], y_bin[:-1], seq_len
            )
            return _predict_meta(
                base_models, meta_model, names_used, X_last_s, X_scaled, seq_len
            )

        # --------------------------------------------------
        #  c)   anything else → error
        # --------------------------------------------------
        logging.error(f"Classifier request {ml_models} not supported.")
        return 0.5


    # --- If meta model (all 5 regressors selected) ---
    if use_meta_model:
        model_names_for_meta = ["forest", "xgboost", "lightgbm", "lstm", "transformer", "catboost"]
        N = len(df)
        start_idx = max(seq_len+1, 70)
        preds_dict = {name:[] for name in model_names_for_meta}
        y_true = []
        logging.info(f"Meta model expanding-window walk-forward: {start_idx} ... {N-2}")
        for i in range(start_idx, N-1):
            train_idx = range(i)
            local_X = X.iloc[train_idx]
            local_y = y.iloc[train_idx]
            X_pred_row = X.iloc[[i]]
            # FOREST
            try:
                rf_model_meta = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
                pipelines.fixed_fit(local_X, local_y, rf_model_meta)
                rf_pred = rf_model_meta.predict(X_pred_row)[0]
            except:
                rf_pred = np.nan
            preds_dict["forest"].append(rf_pred)
            # XGBOOST
            try:
                xgb_model_meta = xgb.XGBRegressor(
                    n_estimators=114, 
                    max_depth=9, 
                    learning_rate=0.14264252588219034,
                    subsample=0.5524803023252148,
                    colsample_bytree=0.7687841723045249,
                    gamma=0.5856035407199236,
                    reg_alpha=0.5063880221467401,
                    reg_lambda=0.0728996118523866,
                )
                pipelines.fixed_fit(local_X, local_y, xgb_model_meta)
                xgb_pred = xgb_model_meta.predict(X_pred_row)[0]
            except:
                xgb_pred = np.nan
            preds_dict["xgboost"].append(xgb_pred)
            try:
                cb_model_meta = CatBoostRegressor(iterations=600, depth=6, learning_rate=0.05,
                                                l2_leaf_reg=3, bagging_temperature=0.5,
                                                loss_function="RMSE", eval_metric="RMSE",
                                                random_seed=RANDOM_SEED, verbose=False)
                pipelines.fixed_fit(local_X, local_y, cb_model_meta)
                cb_pred = cb_model_meta.predict(X_pred_row)[0]
            except:
                cb_pred = np.nan
            preds_dict["catboost"].append(cb_pred)
            # LIGHTGBM
            try:
                lgbm_model_meta = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
                pipelines.fixed_fit(local_X, local_y, lgbm_model_meta)
                lgbm_pred = lgbm_model_meta.predict(X_pred_row)[0]
            except:
                lgbm_pred = np.nan
            preds_dict["lightgbm"].append(lgbm_pred)
            # LSTM
            try:
                if use_transformer and i > seq_len+1:
                    X_lstm_np = local_X.values
                    y_lstm_np = local_y.values
                    X_lstm_win, y_lstm_win = engineering.series_to_supervised(X_lstm_np, y_lstm_np, seq_len)
                    lstm_model_meta = registry.get_single_model("lstm", input_shape=(seq_len, X_lstm_np.shape[1]), num_features=X_lstm_np.shape[1], lstm_seq=seq_len)
                    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, verbose=0, restore_best_weights=True)
                    cp = keras.callbacks.ModelCheckpoint("lstm_meta_model.keras", save_best_only=True, monitor="val_loss", mode="min", verbose=0)
                    lstm_model_meta.fit(
                        X_lstm_win, y_lstm_win,
                        epochs=10,
                        batch_size=32,
                        verbose=0,
                        validation_split=0.18,
                        callbacks=[es, cp], 
                        sample_weight=pipelines.add_weights(len(X_lstm_win))
                    )
                    last_seq = local_X.iloc[-seq_len:].values.reshape(1,seq_len,-1)
                    lstm_pred = lstm_model_meta.predict(last_seq, verbose=0)[0][0]
                else:
                    lstm_pred = np.nan
            except:
                lstm_pred = np.nan
            preds_dict["lstm"].append(lstm_pred)
            # TRANSFORMER
            try:
                if use_transformer and i > seq_len+1:
                    scaler = StandardScaler()
                    tr_X = scaler.fit_transform(local_X.values)
                    tr_y = local_y.values
                    Xtr_win, ytr_win = engineering.series_to_supervised(tr_X, tr_y, seq_len)
                    TransformerRegMeta = registry.get_single_model("transformer", num_features=tr_X.shape[1])
                    tr_reg_model_meta = TransformerRegMeta(num_features=tr_X.shape[1], seq_len=seq_len)
                    opt = torch.optim.Adam(tr_reg_model_meta.parameters(), lr=0.0008)
                    loss_fn = torch.nn.L1Loss()
                    tr_reg_model_meta.train()
                    Xtr_win_torch = torch.tensor(Xtr_win, dtype=torch.float32)
                    ytr_win_torch = torch.tensor(ytr_win, dtype=torch.float32)
                    for epoch in range(5):
                        opt.zero_grad()
                        loss = loss_fn(tr_reg_model_meta(Xtr_win_torch).squeeze(), ytr_win_torch)
                        loss.backward()
                        opt.step()
                    last_seq = scaler.transform(local_X.iloc[-seq_len:].values).reshape(1,seq_len,-1)
                    last_seq_torch = torch.tensor(last_seq, dtype=torch.float32)
                    tr_pred = tr_reg_model_meta(last_seq_torch).detach().numpy()[0,0]
                else:
                    tr_pred = np.nan
            except:
                tr_pred = np.nan
            preds_dict["transformer"].append(tr_pred)
            # Save true y
            y_true.append(y.iloc[i])
        preds_arr = np.vstack([preds_dict["forest"],
                               preds_dict["xgboost"],
                               preds_dict["catboost"],
                               preds_dict["lightgbm"],
                               preds_dict["lstm"],
                               preds_dict["transformer"]]).T
        y_true = np.array(y_true)
        valid_mask = ~np.isnan(preds_arr).any(axis=1) & ~np.isnan(y_true)
        preds_arr_valid = preds_arr[valid_mask]
        y_true_valid = y_true[valid_mask]
        if len(y_true_valid) < 5:
            logging.error("Meta model: Not enough valid rows for fitting.")
            pred = np.nanmean([preds_arr_valid[-1,i] for i in range(preds_arr_valid.shape[1]) if not np.isnan(preds_arr_valid[-1,i])])
            if return_model_stack:
                return pred, {k: preds_arr_valid[-1, idx] for idx, k in enumerate(model_names_for_meta)}
            else:
                return pred
        meta_model = RidgeCV().fit(preds_arr_valid[:-1], y_true_valid[:-1])  # last one = test
        pred_stack = preds_arr_valid[-1].reshape(1,-1)
        final_pred = float(meta_model.predict(pred_stack)[0])
        if return_model_stack:
            out = {k: preds_arr_valid[-1, idx] for idx, k in enumerate(model_names_for_meta)}
            out['meta_pred'] = final_pred
            return final_pred, out
        else:
            return final_pred

    # If not meta: just pick single selected regressor
    if "lstm" in ml_models and use_transformer:
        X_lstm_np = np.array(X)
        y_lstm_np = np.array(y)
        X_lstm_win, y_lstm_win = engineering.series_to_supervised(X_lstm_np, y_lstm_np, seq_len)
        lstm_model = registry.get_single_model("lstm", input_shape=(seq_len, X_lstm_np.shape[1]), num_features=X_lstm_np.shape[1], lstm_seq=seq_len)
        es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, verbose=0, restore_best_weights=True)
        cp = keras.callbacks.ModelCheckpoint("lstm_best_model.keras", save_best_only=True, monitor="val_loss", mode="min", verbose=0)
        lstm_model.fit(
            X_lstm_win, y_lstm_win,
            epochs=80,
            batch_size=32,
            verbose=0,
            validation_split=0.18,
            callbacks=[es, cp],
            sample_weight=pipelines.add_weights(len(X_lstm_win))
        )
        lstm_model.load_weights("lstm_best_model.keras")
        last_seq = X_lstm_np[-seq_len:,:].reshape(1,seq_len,-1)
        lstm_pred = lstm_model.predict(last_seq, verbose=0)[0][0]
        out_preds.append(lstm_pred)
        out_names.append("lstm_pred")

    if "xgboost" in ml_models:
        xgb_model = xgb.XGBRegressor(
            n_estimators=114, 
            max_depth=9, 
            learning_rate=0.14264252588219034,
            subsample=0.5524803023252148,
            colsample_bytree=0.7687841723045249,
            gamma=0.5856035407199236,
            reg_alpha=0.5063880221467401,
            reg_lambda=0.0728996118523866,
        )
        pipelines.fixed_fit(X, y, xgb_model)
        xgb_pred = xgb_model.predict(last_row_df)[0]
        out_preds.append(xgb_pred)
        out_names.append("xgb_pred")

    if "catboost" or "cb" in ml_models:
        cb_model = CatBoostRegressor(iterations=600, depth=6, learning_rate=0.05,
                                                l2_leaf_reg=3, bagging_temperature=0.5,
                                                loss_function="RMSE", eval_metric="RMSE",
                                                random_seed=RANDOM_SEED, verbose=False)
        pipelines.fixed_fit(X, y, cb_model)
        cb_pred = cb_model.predict(last_row_df)[0]
        out_preds.append(cb_pred)
        out_names.append("cb_pred")

    if "lightgbm" in ml_models or "lgbm" in ml_models:
        lgbm_model = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
        pipelines.fixed_fit(X, y, lgbm_model)
        lgbm_pred = lgbm_model.predict(last_row_df)[0]
        out_preds.append(lgbm_pred)
        out_names.append("lgbm_pred")

    if "forest" in ml_models or "rf" in ml_models or "randomforest" in ml_models:
        rf_model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
        pipelines.fixed_fit(X, y, rf_model)
        rf_pred = rf_model.predict(last_row_df)[0]
        out_preds.append(rf_pred)
        out_names.append("rf_pred")

    if "transformer" in ml_models and use_transformer:
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X.values)
            y_scale_mean = y.mean()
            y_scale_std = y.std() if y.std() > 1e-5 else 1.0
            Xtr_np = X_scaled
            ytr_np = ((y.values - y_scale_mean) / y_scale_std)
            if np.isnan(Xtr_np).any() or np.isnan(ytr_np).any():
                use_this_transformer = False
            else:
                use_this_transformer = True
            if use_this_transformer:
                Xtr_win, ytr_win = engineering.series_to_supervised(Xtr_np, ytr_np, seq_len)
                Xtr_win_torch = torch.tensor(Xtr_win, dtype=torch.float32)
                ytr_win_torch = torch.tensor(ytr_win, dtype=torch.float32)
                TransformerReg = registry.get_single_model("transformer", num_features=Xtr_np.shape[1])
                tr_reg_model = TransformerReg(num_features=Xtr_np.shape[1], seq_len=seq_len)
                opt = torch.optim.Adam(tr_reg_model.parameters(), lr=0.0005)
                loss_fn = torch.nn.L1Loss()
                tr_reg_model.train()
                for epoch in range(50):
                    opt.zero_grad()
                    y_pred = tr_reg_model(Xtr_win_torch).squeeze()
                    if torch.isnan(y_pred).any():
                        break
                    loss = loss_fn(y_pred, ytr_win_torch)
                    if torch.isnan(loss):
                        break
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(tr_reg_model.parameters(), max_norm=2.0)
                    opt.step()
                tr_reg_model.eval()
                last_x_seq = X_scaled[-seq_len:,:].reshape(1,seq_len,-1)
                last_seq_torch = torch.tensor(last_x_seq, dtype=torch.float32)
                with torch.no_grad():
                    y_pred_scaled = tr_reg_model(last_seq_torch).cpu().numpy()[0,0]
                tr_reg_pred = (y_pred_scaled * y_scale_std) + y_scale_mean
                if not np.isnan(tr_reg_pred) and not np.isinf(tr_reg_pred):
                    out_preds.append(tr_reg_pred)
                    out_names.append("tr_reg_pred")
        except Exception as e:
            logging.error(f"TransformerReg failed to train: {e}")

    # Output
    if len(out_preds)==1:
        if return_model_stack:
            return out_preds[0], {out_names[0]:out_preds[0]}
        else:
            return out_preds[0]
    elif len(out_preds) > 1:
        pred = np.mean(out_preds)
        if return_model_stack:
            return pred, dict(zip(out_names,out_preds))
        else:
            return pred
    else:
        return None