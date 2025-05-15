#!/usr/bin/env python3
"""
main.py

Orchestrates the five sub-models in either “sub-vote” or “sub-meta” mode,
and in either “live” or “backtest” execution—configured by in-script variables.

Just edit CSV_PATH, MODE and EXECUTION below, then run:

    python main.py
"""

import os
import numpy as np
import pandas as pd
from types import SimpleNamespace
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv

import sub.momentum            as mod1
import sub.trend_continuation  as mod2
import sub.mean_reversion      as mod3
import sub.lagged_return       as mod4
import sub.sentiment_volume    as mod5

# ─── Configuration ──────────────────────────────────────────────────────────────

load_dotenv()
BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "4Hour")
TICKERS = os.getenv("TICKERS", "TSLA").split(",")
ML_MODEL  = os.getenv("ML_MODEL", "sub-vote")
TIMEFRAME_MAP = {
    "4Hour": "H4",
    "2Hour": "H2",
    "1Hour": "H1",
    "30Min": "M30",
    "15Min": "M15"
}
CONVERTED_TIMEFRAME = TIMEFRAME_MAP.get(BAR_TIMEFRAME, BAR_TIMEFRAME)

# Path to your OHLC+features CSV
CSV_PATH = f"{TICKERS}_{CONVERTED_TIMEFRAME}.csv"

# "sub-vote" or "sub-meta"
MODE = ML_MODEL

# "live" or "backtest"
EXECUTION = globals().get("EXECUTION", "backtest")

# Voting thresholds
CLS_UP   = 0.55
CLS_DOWN = 0.45
REG_UP   = 0.003
REG_DOWN = -0.003

# ─── Voting Helpers ────────────────────────────────────────────────────────────

def vote_from_prob(p):
    if p > CLS_UP:      return 1   # BUY
    if p < CLS_DOWN:    return 0   # SELL
    return 0.5                  # NONE

def vote_from_ret(r):
    if r > REG_UP:      return 1
    if r < REG_DOWN:    return 0
    return 0.5

def actual_direction(df, t):
    """1 if next-bar open > this-bar open, else 0"""
    return int(df['open'].iat[t+1] > df['open'].iat[t])

# ─── Backtest Sub-Models ───────────────────────────────────────────────────────

def backtest_submodels(df, initial_frac=0.8):
    """
    Walk-forward backtest sub-models:
      - train each on df[:t+1]
      - predict numeric output on bar t
      - record actual next-bar direction
    """
    n = len(df)
    cut = int(n * initial_frac)
    rec = []
    for t in tqdm(range(cut, n-1), desc="Submodel BT"):
        slice_df = df.iloc[:t+1].reset_index(drop=True)
        label    = actual_direction(df, t)

        # 1) Momentum
        d1 = mod1.compute_labels(slice_df)
        tr1 = d1.iloc[:-1].dropna(subset=mod1.FEATURES + ['label'])
        m1 = mod1.fit(tr1[mod1.FEATURES].values, tr1['label'].values)
        p1 = mod1.predict(m1, d1.loc[[t], mod1.FEATURES])[0]

        # 2) Trend continuation
        d2 = mod2.compute_labels(slice_df)
        tr2 = d2.iloc[:-1].dropna(subset=mod2.FEATURES + ['label'])
        m2 = mod2.fit(tr2[mod2.FEATURES], tr2['label'])
        p2 = mod2.predict(m2, d2.loc[[t], mod2.FEATURES])[0]

        # 3) Mean reversion
        d3 = mod3.compute_labels(slice_df)
        tr3 = d3.iloc[:-1].dropna(subset=mod3.FEATURES + ['label'])
        m3 = mod3.fit(tr3[mod3.FEATURES].values, tr3['label'].values)
        p3 = mod3.predict(m3, d3.loc[[t], mod3.FEATURES])[0]

        # 4) Lagged return
        d4 = mod4.compute_target(slice_df)
        tr4 = d4.iloc[:-1].dropna(subset=mod4.FEATURES + ['target'])
        m4 = mod4.fit(tr4[mod4.FEATURES], tr4['target'])
        p4 = mod4.predict(m4, d4.loc[[t], mod4.FEATURES])[0]

        # 5) Sentiment‐Volume anomaly
        d5 = mod5.compute_labels(slice_df, min_return=0.002)
        tr5 = d5.iloc[:-1].dropna(subset=mod5.FEATURES + ['label'])
        m5 = mod5.fit(tr5[mod5.FEATURES], tr5['label'])
        p5 = mod5.predict(m5, d5.loc[[t], mod5.FEATURES])[0]

        rec.append({'t':t, 'm1':p1, 'm2':p2, 'm3':p3, 'm4':p4, 'm5':p5, 'label':label})

    return pd.DataFrame(rec).set_index('t')

# ─── Run Backtest ──────────────────────────────────────────────────────────────

def run_backtest(return_df=False):
    df   = pd.read_csv(CSV_PATH)
    back = backtest_submodels(df, initial_frac=0.8)

    if MODE == 'sub-vote':
        back['v1'] = back['m1'].map(vote_from_prob)
        back['v2'] = back['m2'].map(vote_from_prob)
        back['v3'] = back['m3'].map(vote_from_prob)
        back['v4'] = back['m4'].map(vote_from_ret)
        back['v5'] = back['m5'].map(vote_from_prob)
        back['buy_votes']  = (back[['v1','v2','v3','v4','v5']]==1).sum(axis=1)
        back['sell_votes'] = (back[['v1','v2','v3','v4','v5']]==0).sum(axis=1)
        back['action'] = back.apply(
            lambda r: 1 if r.buy_votes >= 3 else (0 if r.sell_votes >= 3 else 0.5),
            axis=1
        )
        acc = accuracy_score(back['label'], (back['action']==1).astype(int))

    else:  # sub-meta
        back['meta_pred'] = np.nan
        for i in range(len(back)):
            if i < int(len(back)*0.8): 
                continue
            train = back.iloc[:i]
            Xm    = train[['m1','m2','m3','m4','m5']].values
            ym    = train['label'].values
            meta  = LogisticRegression()
            meta.fit(Xm, ym)
            xi    = back[['m1','m2','m3','m4','m5']].iloc[i].values.reshape(1,-1)
            back.iloc[i, back.columns.get_loc('meta_pred')] = meta.predict_proba(xi)[0,1]

        valid = back['meta_pred'].notna()
        back.loc[valid,'action'] = back.loc[valid,'meta_pred'].map(vote_from_prob)
        acc = accuracy_score(back.loc[valid,'label'], (back.loc[valid,'action']==1).astype(int))

    # Now output timestamp, action (as "BUY"/"SELL"/"NONE")
    # Map t index back to DataFrame timestamp
    df_orig = pd.read_csv(CSV_PATH)
    # The walk-forward test window starts at cut = int(n*0.8)
    n = len(df_orig)
    cut = int(n * 0.8)
    timestamps = df_orig['timestamp'].iloc[cut+1:].values  # aligning with backtest
    # The action column in back aligns with t= cut ... n-2
    actions = back['action'].values
    act_map = {1: "BUY", 0: "SELL", 0.5: "NONE"}
    action_labels = [act_map.get(a, "NONE") for a in actions]
    out_df = pd.DataFrame({
        "timestamp": timestamps,
        "action": action_labels
    })
    results_dir = os.path.join("sub", "sub-results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    out_fn = os.path.join(results_dir, f"meta_results_{MODE}.csv")
    out_df.to_csv(out_fn, index=False)
    print(f"Results saved to {out_fn}")

    if return_df:
        return out_df
    else:
        return

def run_live(return_result=False):
    df = pd.read_csv(CSV_PATH)
    t  = len(df) - 1

    d1 = mod1.compute_labels(df)
    tr1 = d1.iloc[:-1].dropna(subset=mod1.FEATURES+['label'])
    m1 = mod1.fit(tr1[mod1.FEATURES].values, tr1['label'].values)
    p1 = mod1.predict(m1, d1.loc[[t], mod1.FEATURES])[0]

    d2 = mod2.compute_labels(df)
    tr2 = d2.iloc[:-1].dropna(subset=mod2.FEATURES+['label'])
    m2 = mod2.fit(tr2[mod2.FEATURES], tr2['label'])
    p2 = mod2.predict(m2, d2.loc[[t], mod2.FEATURES])[0]

    d3 = mod3.compute_labels(df)
    tr3 = d3.iloc[:-1].dropna(subset=mod3.FEATURES+['label'])
    m3 = mod3.fit(tr3[mod3.FEATURES].values, tr3['label'].values)
    p3 = mod3.predict(m3, d3.loc[[t], mod3.FEATURES])[0]

    d4 = mod4.compute_target(df)
    tr4 = d4.iloc[:-1].dropna(subset=mod4.FEATURES+['target'])
    m4 = mod4.fit(tr4[mod4.FEATURES], tr4['target'])
    p4 = mod4.predict(m4, d4.loc[[t], mod4.FEATURES])[0]

    d5 = mod5.compute_labels(df, min_return=0.002)
    tr5 = d5.iloc[:-1].dropna(subset=mod5.FEATURES+['label'])
    m5 = mod5.fit(tr5[mod5.FEATURES], tr5['label'])
    p5 = mod5.predict(m5, d5.loc[[t], mod5.FEATURES])[0]

    if MODE == 'sub-vote':
        votes = [
            vote_from_prob(p1),
            vote_from_prob(p2),
            vote_from_prob(p3),
            vote_from_ret(p4),
            vote_from_prob(p5),
        ]
        buy   = votes.count(1)
        sell  = votes.count(0)
        act   = 1 if buy>=3 else (0 if sell>=3 else 0.5)
        result = {1:"BUY",0.5:"NONE",0:"SELL"}[act]
    else:  # sub-meta
        try:
            # Use latest meta fit if available
            hist = pd.read_csv(os.path.join("sub", "sub-results", f"meta_results_{MODE}.csv")).dropna(subset=['action'])
            Xm   = hist[['m1','m2','m3','m4','m5']].values
            ym   = (hist['action'] == "BUY").astype(int)
            meta = LogisticRegression().fit(Xm, ym)
            prob = meta.predict_proba([[p1,p2,p3,p4,p5]])[0,1]
            act  = vote_from_prob(prob)
            result = {1:"BUY",0.5:"NONE",0:"SELL"}[act]
        except Exception:
            result = "NONE"

    if return_result:
        return result
    else:
        print("LIVE {} → {}".format(MODE, result))

# ─── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if EXECUTION == "backtest":
        run_backtest()
    else:
        run_live()
