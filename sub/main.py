#!/usr/bin/env python3
"""
main.py

Orchestrates the five sub-models in either “sub-vote” or “sub-meta” mode,
and in either “live” or “backtest” execution—configured by in-script variables.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from dotenv import load_dotenv

import sub.momentum            as mod1
import sub.trend_continuation  as mod2
import sub.mean_reversion      as mod3
import sub.lagged_return       as mod4
import sub.sentiment_volume    as mod5

# ─── Base directories ─────────────────────────────────────────────────────────
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

CSV_PATH  = os.path.join(ROOT_DIR, f"{TICKERS}_{CONVERTED_TIMEFRAME}.csv")
MODE       = ML_MODEL  # "sub-vote" or "sub-meta"
EXECUTION  = globals().get("EXECUTION", "backtest")
CLS_UP, CLS_DOWN = 0.55, 0.45
REG_UP, REG_DOWN = 0.003, -0.003


def vote_from_prob(p):
    return 1 if p > CLS_UP else (0 if p < CLS_DOWN else 0.5)

def vote_from_ret(r):
    return 1 if r > REG_UP else (0 if r < REG_DOWN else 0.5)

def actual_direction(df, t):
    return int(df['open'].iat[t+1] > df['open'].iat[t])


def backtest_submodels(df, initial_frac=0.8):
    """
    Walk-forward backtest sub-models over df[:cut…n-2]:
      - train on df[:t+1]
      - predict on bar t
    Returns DataFrame indexed by t with columns m1…m5 and 'label'.
    """
    n   = len(df)
    cut = int(n * initial_frac)
    rec = []
    for t in tqdm(range(cut, n-1), desc="Submodel BT"):
        slice_df = df.iloc[:t+1].reset_index(drop=True)
        label    = actual_direction(df, t)

        d1  = mod1.compute_labels(slice_df)
        tr1 = d1.iloc[:-1].dropna(subset=mod1.FEATURES + ['label'])
        p1  = mod1.predict(mod1.fit(tr1[mod1.FEATURES].values, tr1['label'].values),
                           d1.loc[[t], mod1.FEATURES])[0]

        d2  = mod2.compute_labels(slice_df)
        tr2 = d2.iloc[:-1].dropna(subset=mod2.FEATURES + ['label'])
        p2  = mod2.predict(mod2.fit(tr2[mod2.FEATURES], tr2['label']),
                           d2.loc[[t], mod2.FEATURES])[0]

        d3  = mod3.compute_labels(slice_df)
        tr3 = d3.iloc[:-1].dropna(subset=mod3.FEATURES + ['label'])
        p3  = mod3.predict(mod3.fit(tr3[mod3.FEATURES].values, tr3['label'].values),
                           d3.loc[[t], mod3.FEATURES])[0]

        d4  = mod4.compute_target(slice_df)
        tr4 = d4.iloc[:-1].dropna(subset=mod4.FEATURES + ['target'])
        p4  = mod4.predict(mod4.fit(tr4[mod4.FEATURES], tr4['target']),
                           d4.loc[[t], mod4.FEATURES])[0]

        d5  = mod5.compute_labels(slice_df, min_return=0.002)
        tr5 = d5.iloc[:-1].dropna(subset=mod5.FEATURES + ['label'])
        p5  = mod5.predict(mod5.fit(tr5[mod5.FEATURES], tr5['label']),
                           d5.loc[[t], mod5.FEATURES])[0]

        rec.append({'t':t, 'm1':p1, 'm2':p2, 'm3':p3, 'm4':p4, 'm5':p5, 'label':label})

    return pd.DataFrame(rec).set_index('t')


def run_backtest(return_df=False):
    df_orig = pd.read_csv(CSV_PATH)
    back    = backtest_submodels(df_orig, initial_frac=0.8)

    if MODE == 'sub-vote':
        back['v1'] = back['m1'].map(vote_from_prob)
        back['v2'] = back['m2'].map(vote_from_prob)
        back['v3'] = back['m3'].map(vote_from_prob)
        back['v4'] = back['m4'].map(vote_from_ret)
        back['v5'] = back['m5'].map(vote_from_prob)
        back['buy_votes']  = (back[['v1','v2','v3','v4','v5']] == 1).sum(axis=1)
        back['sell_votes'] = (back[['v1','v2','v3','v4','v5']] == 0).sum(axis=1)
        back['action']     = back.apply(
            lambda r: 1 if r.buy_votes>=3 else (0 if r.sell_votes>=3 else 0.5),
            axis=1
        )
    else:
        back['meta_pred'] = np.nan
        cutoff = int(len(back)*0.8)
        for i in range(cutoff, len(back)):
            train = back.iloc[:i]
            Xm    = train[['m1','m2','m3','m4','m5']].values
            ym    = train['label'].values
            meta  = LogisticRegression().fit(Xm, ym)
            xi    = back[['m1','m2','m3','m4','m5']].iloc[i].values.reshape(1,-1)
            back.at[i,'meta_pred'] = meta.predict_proba(xi)[0,1]
        back['action'] = back['meta_pred']

    # attach timestamp
    n   = len(df_orig)
    back = back.reset_index().rename(columns={'index':'t'})
    back['timestamp'] = df_orig['timestamp'].iloc[ back['t'] + 1 ].values

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_fn = os.path.join(RESULTS_DIR, f"meta_results_{MODE}.csv")
    back[['timestamp','m1','m2','m3','m4','m5','action']].to_csv(out_fn, index=False)
    print(f"Results saved to {out_fn}")

    if return_df:
        return back[['timestamp','m1','m2','m3','m4','m5','action']]


def run_live(return_result=False):
    """
    For sub-vote: train each sub-model on full history, vote on last bar.
    For sub-meta: first run does realistic i+1 backtest on final 20%, cache it; subsequent runs incrementally backtest new bars only.
    """
    # Load price+features and parse timestamps
    df = pd.read_csv(CSV_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    n = len(df)

    if MODE == 'sub-vote':
        # train each sub-model on all history and vote on last bar
        t = n - 1
        d1 = mod1.compute_labels(df)
        tr1 = d1.iloc[:-1].dropna(subset=mod1.FEATURES+['label'])
        p1 = mod1.predict(mod1.fit(tr1[mod1.FEATURES].values, tr1['label'].values), d1.loc[[t], mod1.FEATURES])[0]
        d2 = mod2.compute_labels(df)
        tr2 = d2.iloc[:-1].dropna(subset=mod2.FEATURES+['label'])
        p2 = mod2.predict(mod2.fit(tr2[mod2.FEATURES], tr2['label']), d2.loc[[t], mod2.FEATURES])[0]
        d3 = mod3.compute_labels(df)
        tr3 = d3.iloc[:-1].dropna(subset=mod3.FEATURES+['label'])
        p3 = mod3.predict(mod3.fit(tr3[mod3.FEATURES].values, tr3['label'].values), d3.loc[[t], mod3.FEATURES])[0]
        d4 = mod4.compute_target(df)
        tr4 = d4.iloc[:-1].dropna(subset=mod4.FEATURES+['target'])
        p4 = mod4.predict(mod4.fit(tr4[mod4.FEATURES], tr4['target']), d4.loc[[t], mod4.FEATURES])[0]
        d5 = mod5.compute_labels(df, min_return=0.002)
        tr5 = d5.iloc[:-1].dropna(subset=mod5.FEATURES+['label'])
        p5 = mod5.predict(mod5.fit(tr5[mod5.FEATURES], tr5['label']), d5.loc[[t], mod5.FEATURES])[0]
        votes = [vote_from_prob(x) for x in (p1, p2, p3, p4, p5)]
        buy, sell = votes.count(1), votes.count(0)
        result = 1 if buy >= 3 else (0 if sell >= 3 else 0.5)
        if return_result:
            return result
        else:
            print(result)
    else:
        # sub-meta: cached backtest history
        cache_fn = os.path.join(RESULTS_DIR, f"submeta_history_{CONVERTED_TIMEFRAME}.csv")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        # load or initialize cache
        if os.path.exists(cache_fn):
            rec = pd.read_csv(cache_fn)
            rec['timestamp'] = pd.to_datetime(rec['timestamp'], utc=True)
        else:
            # initial realistic backtest on final 20%
            rec = backtest_submodels(df, initial_frac=0.8)
            rec = rec.reset_index().rename(columns={'index':'t'})
            rec['timestamp'] = df['timestamp'].iloc[rec['t'] + 1].values
            rec = rec[['timestamp','m1','m2','m3','m4','m5','label']]
            rec.to_csv(cache_fn, index=False)
        # incremental backtest on new bars
        last_ts = rec['timestamp'].max()
        new_df = df[df['timestamp'] > last_ts]
        for idx in new_df.index:
            t = idx - 1
            if t < 0:
                continue
            slice_df = df.iloc[:t+1].reset_index(drop=True)
            label = actual_direction(df, t)
            d1 = mod1.compute_labels(slice_df)
            tr1 = d1.iloc[:-1].dropna(subset=mod1.FEATURES+['label'])
            p1 = mod1.predict(mod1.fit(tr1[mod1.FEATURES].values, tr1['label'].values), d1.loc[[t], mod1.FEATURES])[0]
            d2 = mod2.compute_labels(slice_df)
            tr2 = d2.iloc[:-1].dropna(subset=mod2.FEATURES+['label'])
            p2 = mod2.predict(mod2.fit(tr2[mod2.FEATURES], tr2['label']), d2.loc[[t], mod2.FEATURES])[0]
            d3 = mod3.compute_labels(slice_df)
            tr3 = d3.iloc[:-1].dropna(subset=mod3.FEATURES+['label'])
            p3 = mod3.predict(mod3.fit(tr3[mod3.FEATURES].values, tr3['label'].values), d3.loc[[t], mod3.FEATURES])[0]
            d4 = mod4.compute_target(slice_df)
            tr4 = d4.iloc[:-1].dropna(subset=mod4.FEATURES+['target'])
            p4 = mod4.predict(mod4.fit(tr4[mod4.FEATURES], tr4['target']), d4.loc[[t], mod4.FEATURES])[0]
            d5 = mod5.compute_labels(slice_df, min_return=0.002)
            tr5 = d5.iloc[:-1].dropna(subset=mod5.FEATURES+['label'])
            p5 = mod5.predict(mod5.fit(tr5[mod5.FEATURES], tr5['label']), d5.loc[[t], mod5.FEATURES])[0]
            ts = df['timestamp'].iloc[t+1]
            rec = rec.append({'timestamp':ts,'m1':p1,'m2':p2,'m3':p3,'m4':p4,'m5':p5,'label':label}, ignore_index=True)
        rec.to_csv(cache_fn, index=False)
        # retrain meta-model on full cache
        Xm = rec[['m1','m2','m3','m4','m5']].values
        ym = rec['label'].values
        meta = LogisticRegression().fit(Xm, ym)
        # predict next-bar probability
        t = n - 1
        d1 = mod1.compute_labels(df); tr1 = d1.iloc[:-1].dropna(subset=mod1.FEATURES+['label'])
        p1 = mod1.predict(mod1.fit(tr1[mod1.FEATURES].values, tr1['label'].values), d1.loc[[t], mod1.FEATURES])[0]
        d2 = mod2.compute_labels(df); tr2 = d2.iloc[:-1].dropna(subset=mod2.FEATURES+['label'])
        p2 = mod2.predict(mod2.fit(tr2[mod2.FEATURES], tr2['label']), d2.loc[[t], mod2.FEATURES])[0]
        d3 = mod3.compute_labels(df); tr3 = d3.iloc[:-1].dropna(subset=mod3.FEATURES+['label'])
        p3 = mod3.predict(mod3.fit(tr3[mod3.FEATURES].values, tr3['label'].values), d3.loc[[t], mod3.FEATURES])[0]
        d4 = mod4.compute_target(df); tr4 = d4.iloc[:-1].dropna(subset=mod4.FEATURES+['target'])
        p4 = mod4.predict(mod4.fit(tr4[mod4.FEATURES], tr4['target']), d4.loc[[t], mod4.FEATURES])[0]
        d5 = mod5.compute_labels(df, min_return=0.002); tr5 = d5.iloc[:-1].dropna(subset=mod5.FEATURES+['label'])
        p5 = mod5.predict(mod5.fit(tr5[mod5.FEATURES], tr5['label']), d5.loc[[t], mod5.FEATURES])[0]
        prob = meta.predict_proba([[p1,p2,p3,p4,p5]])[0,1]
        if return_result:
            return prob
        else:
            print(f"Next-bar up probability: {prob:.4f}")
