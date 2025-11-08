# sub/main.py
# ============================================================
# Meta Orchestrator for Multi-Model Trading Stack
# - Live: returns BUY/SELL/NONE (with on-the-fly, leakage-safe training)
# - Backtest: walk-forward training at every step (fit up to i-h; predict at i)
# - Pretrain/Analysis: train_models() -> returns per-model importance + AUC/RMSE
#
# PUBLIC ENTRYPOINTS (used by caller):
#   - run_live(return_result=True, position_open=False)
#   - run_backtest(return_df=True)                # respects global `backtest_amount`
#   - train_models() -> pd.DataFrame              # SHAP-like importances + metrics
#
# Meta model:
#   - If a trained meta state exists, it's used; otherwise we degrade to a
#     robust prior-weighted average combiner.
# ============================================================

from __future__ import annotations

import os
import sys
import json
import math
import time
import pickle
import importlib
import requests
import importlib.util
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple, Callable

import numpy as np
import pandas as pd

# Optional env loader
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from polygon import RESTClient  # type: ignore

# -------------------------------
# Globals set by forest.py
# -------------------------------
MODE: str = "single"        # set by forest.py
EXECUTION: str = "live"     # set by forest.py
CSV_PATH: str = ""          # set by forest.py


# ============================================================
# Data Contracts
# ============================================================

@dataclass
class SubmodelResult:
    # Core
    y_pred: pd.Series
    signal: pd.Series
    # Optional
    proba_up: Optional[pd.Series] = None
    pred_std: Optional[pd.Series] = None
    confidence: Optional[pd.Series] = None
    costs: Optional[pd.Series] = None
    trade_mask: Optional[pd.Series] = None
    live_metrics: Optional[Dict[str, float]] = None
    feature_importance: Optional[pd.Series] = None
    used_features: Optional[List[str]] = None
    warmup_bars: int = 0
    model_name: str = ""
    params: Optional[Dict[str, Any]] = None
    state: Optional[Any] = None

@dataclass
class MetaResult:
    y_pred_net_bps: pd.Series
    proba_profit: pd.Series
    pred_std: Optional[pd.Series]
    signal: pd.Series
    position_suggested: pd.Series
    trade_mask: pd.Series
    submodel_weights: Optional[Dict[str, pd.Series]]
    costs_est: Optional[pd.Series]
    diagnostics: Optional[Dict[str, Any]]


# ============================================================
# Paths & Persistence
# ============================================================

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SUB_DIR = _THIS_DIR
_MODELS_DIR = os.path.join(_SUB_DIR, "models")
_META_DIR = os.path.join(_SUB_DIR, "meta")
_ARTIFACTS_DIR = os.path.join(_SUB_DIR, "artifacts")

os.makedirs(_MODELS_DIR, exist_ok=True)
os.makedirs(_META_DIR, exist_ok=True)
os.makedirs(_ARTIFACTS_DIR, exist_ok=True)

_DEFAULT_THRESHOLDS_PATH = os.path.join(_META_DIR, "thresholds.json")
_DEFAULT_META_STATE_PATH = os.path.join(_META_DIR, "meta_state.pkl")


# ============================================================
# Utilities
# ============================================================

def _merge_asof_to_index(left_index: pd.DatetimeIndex, right_df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust, leakage-safe as-of join:
      - Always builds a clean '_ts' key on both sides.
      - Tolerates right_df having any existing column named 'index'/'level_0', etc.
      - Tolerates right_df without a DatetimeIndex by detecting a datetime-like column.
    """
    # ---- Left side: canonical _ts
    li = pd.DataFrame(index=pd.DatetimeIndex(left_index))
    li["_ts"] = pd.to_datetime(li.index, utc=True, errors="coerce")
    li = li[["_ts"]].sort_values("_ts").reset_index(drop=True)

    # ---- Right side: build/derive _ts
    rj = right_df.copy()

    # If DatetimeIndex, use it directly
    if isinstance(rj.index, pd.DatetimeIndex):
        rj["_ts"] = pd.to_datetime(rj.index, utc=True, errors="coerce")
    else:
        # If a datetime-like column exists, use it
        candidates = ["_ts", "timestamp", "t", "ts", "date", "datetime", "time", "DATE", "Date"]
        found = None
        for c in candidates:
            if c in rj.columns:
                found = c
                break
        if found is not None:
            rj["_ts"] = pd.to_datetime(rj[found], utc=True, errors="coerce")
        else:
            # Last resort: reset index and use the first column produced
            tmp = rj.reset_index()
            first_col = tmp.columns[0]
            rj = tmp
            rj["_ts"] = pd.to_datetime(rj[first_col], utc=True, errors="coerce")

    # Keep only rows with a valid timestamp, sort for merge_asof
    rj = rj.dropna(subset=["_ts"]).sort_values("_ts").reset_index(drop=True)

    # ---- As-of merge (backward)
    merged = pd.merge_asof(li, rj, on="_ts", direction="backward")

    # Set index to the left index timestamps, preserving left tz if any
    merged = merged.set_index("_ts")
    if left_index.tz is not None:
        merged.index = merged.index.tz_convert(left_index.tz)

    # Align exactly to left_index order
    return merged.loc[left_index]


def _rolling_zscore(s: pd.Series, win: int, minp: Optional[int] = None) -> pd.Series:
    minp = minp or max(5, win // 5)
    m = s.rolling(win, min_periods=minp).mean()
    sd = s.rolling(win, min_periods=minp).std(ddof=0)
    return (s - m) / sd.replace(0, np.nan)

def _compute_rv20_from_close(close: pd.Series, minp: int = 5) -> pd.Series:
    # realized *variance* over 20 bars (sum of squared 1-bar log returns)
    r1 = np.log(close).diff()
    return (r1 ** 2).rolling(20, min_periods=minp).sum().rename("rv_20")

def _ensure_price_col(d: pd.DataFrame, price_col: str = "close") -> None:
    if price_col not in d.columns:
        raise ValueError(f"Column '{price_col}' missing; required to compute features.")

def _load_leader_csv(leader_csv_path: str) -> Optional[pd.DataFrame]:
    try:
        if leader_csv_path and os.path.exists(leader_csv_path):
            ldf = pd.read_csv(leader_csv_path)
            ldf = _ensure_dt_index(ldf)
            return ldf
    except Exception as e:
        _log(f"WARNING: failed to load leader CSV '{leader_csv_path}': {e}")
    return None

def compute_features(
    df_in: pd.DataFrame,
    leader_csv_path: Optional[str] = None,
    rv_basket_csvs: Optional[List[str]] = None,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Build all required features for submodels (leakage-safe).
    - Base features from the tmp CSV (OHLCV + already-computed tech columns)
    - Leader-based features from QQQ_h4.csv (or a path you pass in)
    - Optional cross-asset RV median from a basket (falls back to asset+leader median)
    """
    df = _ensure_dt_index(df_in.copy())
    _ensure_price_col(df, price_col=price_col)

    # ---------- BASE FEATURES (from tmp CSV only) ----------
    # EMA slopes (diff). If EMA_n not present, compute a quick EMA first.
    for n in (9, 21, 50):
        ema_col = f"ema_{n}"
        slope_col = f"ema_slope_{n}"
        if slope_col not in df.columns:
            if ema_col not in df.columns:
                df[ema_col] = df[price_col].ewm(span=n, adjust=False).mean()
            df[slope_col] = df[ema_col].diff()

    # EMA200 distance
    if "ema200_dist" not in df.columns:
        if "ema_200" not in df.columns:
            df["ema_200"] = df[price_col].ewm(span=200, adjust=False).mean()
        df["ema200_dist"] = (df[price_col] - df["ema_200"]) / df["ema_200"].replace(0, np.nan)

    # Bollinger bandwidth (prefer provided upper/lower; else build 20,SMA+STD)
    if "bb_bw" not in df.columns:
        sma20 = df[price_col].rolling(20, min_periods=5).mean()
        std20 = df[price_col].rolling(20, min_periods=5).std(ddof=0)
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        df["bb_bw"] = ((upper - lower) / sma20.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    # Realized variance rv_20 (from 1-bar returns)
    if "rv_20" not in df.columns:
        r1 = df["returns_1"] if "returns_1" in df.columns else np.log(df[price_col]).diff()
        df["rv_20"] = (r1 ** 2).rolling(20, min_periods=5).sum()

    # Drawdown over 63 bars
    if "dd_63" not in df.columns:
        roll_max = df[price_col].rolling(63, min_periods=5).max()
        df["dd_63"] = (df[price_col] / roll_max - 1.0)

    # ADV ratio (volume vs rolling mean)
    if "adv_ratio" not in df.columns and "volume" in df.columns:
        df["adv_ratio"] = df["volume"] / df["volume"].rolling(63, min_periods=5).mean()

    # Donchian channels (20)
    if "donchian20_up" not in df.columns and "high" in df.columns:
        df["donchian20_up"] = df["high"].rolling(20, min_periods=5).max()
    if "donchian20_dn" not in df.columns and "low" in df.columns:
        df["donchian20_dn"] = df["low"].rolling(20, min_periods=5).min()

    # VWAP deviation (running approximation using typical price)
    if "vwap_dev" not in df.columns and {"high","low","volume"}.issubset(df.columns):
        tp = (df["high"] + df["low"] + df[price_col]) / 3.0
        cum_pv = (tp * df["volume"]).cumsum()
        cum_v  = df["volume"].replace(0, np.nan).cumsum()
        vwap_run = (cum_pv / cum_v).rename("vwap_run")
        df["vwap_dev"] = (df[price_col] - vwap_run) / vwap_run

    # Month sin/cos (from index)
    if "month_sin" not in df.columns or "month_cos" not in df.columns:
        month = df.index.month
        df["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
        df["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)

    # Vol-of-vol (std of log(rv_20))
    if "vov_20" not in df.columns:
        rv = df["rv_20"].replace(0, np.nan)
        df["vov_20"] = np.log(rv).replace([np.inf, -np.inf], np.nan).rolling(20, min_periods=5).std(ddof=0)

    # Downside semivariance over 20 bars
    if "semivar_dn_20" not in df.columns:
        r1 = df["returns_1"] if "returns_1" in df.columns else np.log(df[price_col]).diff()
        dn = np.minimum(r1, 0.0)
        df["semivar_dn_20"] = (dn ** 2).rolling(20, min_periods=5).mean()

    # ---------- LEADER-BASED FEATURES (QQQ_h4.csv or custom) ----------
    # Default path if not provided: sub/data/QQQ_h4.csv
    if leader_csv_path is None:
        leader_csv_path = os.path.join("data", "QQQ_h4.csv")

    leader_df = _load_leader_csv(leader_csv_path)
    if leader_df is None:
        _log(f"WARNING: leader CSV not found -> {leader_csv_path}. Leader features will be NaN.")
        # Create empty placeholders so downstream code doesn't fail
        for col in ["leader_ret_1","leader_ret_2","leader_ret_3","leader_momentum_5","leader_rv_20","spread_z","rv_median_xasset"]:
            if col not in df.columns:
                df[col] = np.nan
        return df

    # Compute leader primitives on leader's OWN timeline (no peeking)
    leader_close = leader_df["close"]
    leader_logp  = np.log(leader_close)
    leader_ret1  = leader_logp.diff()
    leader_ret2  = leader_logp.diff(2)
    leader_ret3  = leader_logp.diff(3)
    leader_mom5  = leader_logp.diff(5)             # 5-bar momentum (log-return)
    leader_rv20  = (leader_ret1 ** 2).rolling(20, min_periods=5).sum()

    leader_feat = pd.DataFrame({
        "leader_ret_1": leader_ret1,
        "leader_ret_2": leader_ret2,
        "leader_ret_3": leader_ret3,
        "leader_momentum_5": leader_mom5,
        "leader_rv_20": leader_rv20,
        "leader_logp": leader_logp,
    }, index=leader_df.index).dropna(how="all")

    # As-of join to df's index (safe: only uses <= current ts)
    leader_aligned = _merge_asof_to_index(df.index, leader_feat)

    # Attach aligned series
    for c in ["leader_ret_1","leader_ret_2","leader_ret_3","leader_momentum_5","leader_rv_20","leader_logp"]:
        if c not in df.columns:
            df[c] = leader_aligned[c]

    # Hedge/spread z-score vs leader (rolling beta on aligned returns)
    # Use aligned 1-bar returns for both series
    asset_ret1  = df["returns_1"] if "returns_1" in df.columns else np.log(df[price_col]).diff()
    lead_ret1_a = df["leader_ret_1"]
    win_beta = 200
    cov_al   = asset_ret1.rolling(win_beta, min_periods=25).cov(lead_ret1_a)
    var_lead = lead_ret1_a.rolling(win_beta, min_periods=25).var()
    beta     = (cov_al / var_lead.replace(0, np.nan)).clip(-10, 10)

    # Compute spread (log-price spread with rolling beta), then z-score it
    asset_logp = np.log(df[price_col])
    spread = asset_logp - (beta * df["leader_logp"])
    df["spread_z"] = _rolling_zscore(spread, win=200)

    # ---------- Cross-asset RV median ----------
    # If basket provided, compute each RV20 and as-of merge; else fallback to median(asset_rv20, leader_rv20)
    rv_candidates = []
    # current asset
    rv_candidates.append(df["rv_20"])
    # leader
    rv_candidates.append(df["leader_rv_20"])

    if rv_basket_csvs:
        for p in rv_basket_csvs:
            try:
                bdf = pd.read_csv(p)
                bdf = _ensure_dt_index(bdf)
                if "close" not in bdf.columns:
                    continue
                b_rv = _compute_rv20_from_close(bdf["close"])
                b_aligned = _merge_asof_to_index(df.index, b_rv.to_frame("rv_20_b"))
                rv_candidates.append(b_aligned["rv_20_b"])
            except Exception as e:
                _log(f"WARNING: skipping basket file '{p}': {e}")

    # Median across available columns, row-wise
    rv_stack = pd.concat(rv_candidates, axis=1)
    df["rv_median_xasset"] = rv_stack.median(axis=1)

    # Housekeeping: drop temp leader_logp if not needed downstream
    if "leader_logp" in df.columns:
        # Keep if you also want it for debugging; otherwise:
        pass

    return df

def _log(s: str) -> None:
    print(f"[sub.main] {s}", flush=True)

def _safe_import(module_name: str, file_path: Optional[str] = None):
    try:
        if file_path and os.path.exists(file_path):
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = mod
                spec.loader.exec_module(mod)
                return mod
        return importlib.import_module(module_name)
    except Exception as e:
        _log(f"WARNING: Could not import '{module_name}' ({file_path}). Error: {e}")
        return None

def _load_json(path: str, default: Optional[dict] = None) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}

def _save_json(path: str, obj: dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def _load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)

def _save_pickle(path: str, obj: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically-stable logistic function that avoids overflow in exp().
    Works with numpy arrays or pandas Series/Index (converted to ndarray).
    """
    z = np.asarray(x, dtype=float)
    out = np.empty_like(z, dtype=float)

    # For z >= 0:  1 / (1 + exp(-z))
    pos = z >= 0
    if np.any(pos):
        zp = np.clip(z[pos], None, 709.0)  # guard huge positives (exp(>709) overflows in float64)
        out[pos] = 1.0 / (1.0 + np.exp(-zp))

    # For z < 0: exp(z) / (1 + exp(z))  (avoids exp(-z) on big |z|)
    neg = ~pos
    if np.any(neg):
        zn = np.clip(z[neg], -709.0, None)  # guard huge negatives
        ez = np.exp(zn)
        out[neg] = ez / (1.0 + ez)

    # Handle any accidental NaNs/Infs robustly
    out = np.where(np.isfinite(out), out, 0.5)
    return out


def _entropy_std_from_proba(p: pd.Series) -> pd.Series:
    return np.sqrt(np.clip(p, 1e-9, 1-1e-9) * np.clip(1-p, 1e-9, 1-1e-9))


def _as_date_str(ts: pd.Timestamp) -> str:
    ts = pd.Timestamp(ts).tz_convert("UTC") if ts.tzinfo else pd.Timestamp(ts, tz="UTC")
    return ts.strftime("%Y-%m-%d")


def _infer_asset_ticker() -> Optional[str]:
    """
    Priority:
      1) TARGET_TICKER / ASSET_TICKER / TICKER env
      2) Parse from CSV_PATH filename like 'TSLA_h4.csv' -> 'TSLA'
    """
    for k in ("TARGET_TICKER", "ASSET_TICKER", "TICKER"):
        v = os.getenv(k, "").strip()
        if v:
            return v.upper()
    try:
        fname = os.path.basename(CSV_PATH)
        root = os.path.splitext(fname)[0]
        # take the leading token before first '_' and strip non-alnum
        tok = root.split("_")[0]
        tok = "".join(ch for ch in tok if ch.isalnum() or ch == ":" or ch == ".")
        return tok.upper() if tok else None
    except Exception:
        return None


def _aggs_to_df(aggs_obj) -> pd.DataFrame:
    """
    Robust conversion for Polygon aggregates objects -> DataFrame with UTC DatetimeIndex and 'close' column.
    Handles both attributes style (c, t) and dict style.
    """
    if aggs_obj is None:
        return pd.DataFrame()
    results = getattr(aggs_obj, "results", None)
    if results is None:
        # sometimes the SDK returns a plain list
        results = aggs_obj
    rows = []
    for a in (results or []):
        # attribute or dict access
        ad = a.__dict__ if hasattr(a, "__dict__") else dict(a)
        # field variations across SDKs
        t = ad.get("timestamp", ad.get("t", None))
        c = ad.get("close", ad.get("c", None))
        if t is None or c is None:
            continue
        ts = pd.to_datetime(int(t), unit="ms", utc=True, errors="coerce")
        rows.append({"timestamp": ts, "close": float(c)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).dropna()
    df = df.set_index("timestamp").sort_index()
    return df


def _treasury_yields_to_df(ty_obj) -> pd.DataFrame:
    """
    Normalize Polygon 'treasury yields' payload into a wide frame with columns:
      ['yield_2y','yield_10y'] and DatetimeIndex
    Supports both wide records (yield_10_year, yield_2_year) and long (tenor/value).
    """
    res = getattr(ty_obj, "results", None)
    if res is None:
        res = ty_obj
    if not res:
        return pd.DataFrame()

    raw = pd.DataFrame([getattr(r, "__dict__", dict(r)) for r in res])
    # unify timestamp
    # typical keys: 'date' (ISO), or 'timestamp'
    if "date" in raw.columns:
        idx = pd.to_datetime(raw["date"], utc=True, errors="coerce")
    else:
        idx = pd.to_datetime(raw.get("timestamp", pd.NaT), unit="ms", utc=True, errors="coerce")
    raw.index = idx
    raw = raw.sort_index()

    # WIDE form: columns like 'yield_10_year', 'yield_2_year'
    cand_10y = next((c for c in raw.columns if "yield_10" in c and "year" in c), None)
    cand_2y  = next((c for c in raw.columns if "yield_2" in c  and "year" in c), None)
    if cand_10y and cand_2y:
        out = pd.DataFrame({
            "yield_10y": pd.to_numeric(raw[cand_10y], errors="coerce"),
            "yield_2y":  pd.to_numeric(raw[cand_2y],  errors="coerce"),
        }, index=raw.index)
        return out

    # LONG form: rows with tenor/value -> pivot
    if {"tenor", "value"}.issubset(raw.columns):
        sub = raw[["tenor", "value"]].copy()
        sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
        # keep only desired tenors
        mask = sub["tenor"].astype(str).str.contains("2", case=False) | sub["tenor"].astype(str).str.contains("10", case=False)
        sub = sub[mask]
        wide = sub.pivot_table(values="value", index=raw.index, columns="tenor", aggfunc="last")
        # try matching common labels
        y10 = None
        for k in wide.columns:
            if "10" in str(k):
                y10 = wide[k]
        y2 = None
        for k in wide.columns:
            if "2" in str(k):
                y2 = wide[k]
        out = pd.DataFrame({"yield_10y": y10, "yield_2y": y2}, index=wide.index)
        return out.dropna(how="all")

    return pd.DataFrame()

# --- network helper ---
try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore

def _http_get(url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Optional[str]:
    try:
        if requests is not None:
            r = requests.get(url, params=params or {}, timeout=timeout)
            return r.text if r.ok else None
        import urllib.parse, urllib.request  # type: ignore
        q = urllib.parse.urlencode(params or {})
        full = f"{url}?{q}" if q else url
        with urllib.request.urlopen(full, timeout=timeout) as resp:  # type: ignore
            return resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return None


def _fred_series_to_df(series_id: str, start: str, end: str, out_col: str) -> pd.DataFrame:
    """
    Parse FRED CSV (free, no key). Tolerant to column casing and BOMs.
    Ensures UTC DatetimeIndex and clips to [start, end].
    """
    txt = _http_get("https://fred.stlouisfed.org/graph/fredgraph.csv", {"id": series_id})
    if not txt:
        return pd.DataFrame()
    import io
    df = pd.read_csv(io.StringIO(txt))
    # Normalize headers
    df.columns = [str(c).strip().upper() for c in df.columns]
    # DATE column tolerance
    date_col = "DATE" if "DATE" in df.columns else df.columns[0]
    # Value column tolerance
    val_col = series_id.upper() if series_id.upper() in df.columns else [c for c in df.columns if c != date_col][0]
    # Build frame
    out = pd.DataFrame(
        {out_col: pd.to_numeric(df[val_col], errors="coerce")},
        index=pd.to_datetime(df[date_col], utc=True, errors="coerce"),
    ).sort_index()
    # Clip to window
    start_ts, end_ts = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    return out.loc[(out.index >= start_ts) & (out.index <= end_ts)]


def _fetch_polygon_aggs(client: Any, ticker: str, mult: int, timespan: str,
                        start_s: str, end_s: str, limit: int = 5000) -> pd.DataFrame:
    try:
        aggs = client.get_aggs(ticker, mult, timespan, start_s, end_s, limit=limit)
        df = _aggs_to_df(aggs)
        return df
    except Exception:
        return pd.DataFrame()


def _fetch_vix_daily_polygon_or_fred(client: Any, start_s: str, end_s: str) -> pd.DataFrame:
    # Polygon I:VIX (requires Indices entitlement)
    if client is not None:
        vix_df = _fetch_polygon_aggs(client, "I:VIX", 1, "day", start_s, end_s, limit=5000)
        if not vix_df.empty:
            return vix_df.rename(columns={"close": "vix_level"})[["vix_level"]]
    # FRED fallback (VIXCLS)
    return _fred_series_to_df("VIXCLS", start_s, end_s, out_col="vix_level")


def _fetch_treasury_10y2y_polygon_or_fred(client: Any, start_s: str, end_s: str) -> pd.DataFrame:
    # Polygon attempt
    if client is not None:
        try:
            ty = None
            for kwargs in (
                {"limit": 5000},
                {"reference_date_gte": start_s, "reference_date_lte": end_s, "limit": 5000},
                {"date_gte": start_s, "date_lte": end_s, "limit": 5000},
            ):
                try:
                    ty = client.get_treasury_yields(**kwargs)
                    if ty:
                        break
                except Exception:
                    continue
            df_ty = _treasury_yields_to_df(ty)
            if not df_ty.empty and {"yield_10y", "yield_2y"}.issubset(df_ty.columns):
                out = df_ty.copy()
                out["term_spread"] = out["yield_10y"] - out["yield_2y"]
                out["dUST10y"] = out["yield_10y"].diff()
                return out[["yield_10y", "yield_2y", "term_spread", "dUST10y"]]
        except Exception:
            pass
    # FRED fallback
    fred10 = _fred_series_to_df("DGS10", start_s, end_s, out_col="yield_10y")
    fred02 = _fred_series_to_df("DGS2",  start_s, end_s, out_col="yield_2y")
    if fred10.empty and fred02.empty:
        return pd.DataFrame()
    merged = fred10.join(fred02, how="outer").sort_index()
    merged["term_spread"] = merged["yield_10y"] - merged["yield_2y"]
    merged["dUST10y"] = merged["yield_10y"].diff()
    return merged[["yield_10y", "yield_2y", "term_spread", "dUST10y"]]


def compute_macro_features(
    df_in: pd.DataFrame,
    asset_ticker: Optional[str] = None,
    dxy_proxy: Optional[str] = None,
) -> pd.DataFrame:
    """
    Free-first macro features:
      - term_spread, dUST10y: Polygon -> FRED (DGS10/DGS2)
      - vix_level:           Polygon I:VIX -> FRED VIXCLS
      - dxy_mom:             Polygon aggregates for UUP (free)
      - carry:               Polygon FUT_FRONT/NEXT; else Polygon equity dividends TTM/price
      - curve_slope:         Polygon FUT_STRIP linear slope across maturities

    merge_asof is used everywhere for leakage safety.
    """
    df = df_in.copy()
    for col in ["carry", "curve_slope", "term_spread", "dUST10y", "vix_level", "dxy_mom"]:
        if col not in df.columns:
            df[col] = np.nan
    if len(df.index) == 0:
        return df

    api_key = os.getenv("POLYGON_API_KEY", "").strip()
    client = None
    if api_key:
        try:
            client = RESTClient(api_key)  # type: ignore
        except Exception as e:
            _log(f"WARNING: polygon RESTClient init failed: {e}")

    # Window
    start = df.index.min() - pd.Timedelta(days=7)
    end   = df.index.max() + pd.Timedelta(days=7)
    start_s, end_s = _as_date_str(start), _as_date_str(end)

    # 1) UST term structure (Polygon -> FRED)
    try:
        df_ty = _fetch_treasury_10y2y_polygon_or_fred(client, start_s, end_s)
        if not df_ty.empty:
            add = _merge_asof_to_index(df.index, df_ty[["term_spread", "dUST10y"]])
            df[["term_spread", "dUST10y"]] = add[["term_spread", "dUST10y"]]
        else:
            _log("WARNING: treasury yields unavailable from both Polygon and FRED; leaving NaN.")
    except Exception as e:
        _log(f"WARNING: treasury yields fetch failed: {e}")

    # 2) VIX (Polygon -> FRED)
    try:
        df_vix = _fetch_vix_daily_polygon_or_fred(client, start_s, end_s)
        if not df_vix.empty:
            add = _merge_asof_to_index(df.index, df_vix[["vix_level"]])
            df["vix_level"] = add["vix_level"]
        else:
            _log("WARNING: VIX unavailable from both Polygon and FRED; leaving NaN.")
    except Exception as e:
        _log(f"WARNING: VIX fetch failed: {e}")

    # 3) DXY momentum via UUP (Polygon, free)
    try:
        proxy = (dxy_proxy or os.getenv("DXY_PROXY_TICKER", "UUP")).strip() or "UUP"
        dxy_aggs = _fetch_polygon_aggs(client, proxy, 1, "day", start_s, end_s, limit=5000) if client else pd.DataFrame()
        if not dxy_aggs.empty:
            # ensure UTC index for merge
            dxy_aggs.index = pd.to_datetime(dxy_aggs.index, utc=True)
            logp = np.log(dxy_aggs["close"])
            dxy_aggs["dxy_mom"] = (logp - logp.shift(20))
            add = _merge_asof_to_index(df.index, dxy_aggs[["dxy_mom"]])
            df["dxy_mom"] = add["dxy_mom"]
        else:
            _log(f"WARNING: DXY proxy '{proxy}' aggregates unavailable; leaving dxy_mom NaN.")
    except Exception as e:
        _log(f"WARNING: DXY proxy fetch failed: {e}")

    # 4) Carry
    carry_done = False
    try:
        fut_front = os.getenv("FUT_FRONT", "").strip()
        fut_next  = os.getenv("FUT_NEXT", "").strip()
        if client and fut_front and fut_next:
            df_f = _fetch_polygon_aggs(client, fut_front, 1, "day", start_s, end_s, limit=5000).rename(columns={"close": "front"})
            df_n = _fetch_polygon_aggs(client, fut_next,  1, "day", start_s, end_s, limit=5000).rename(columns={"close": "next"})
            fut = df_f.join(df_n, how="outer").sort_index()
            if not fut.empty and {"front","next"}.issubset(fut.columns):
                fut["carry"] = fut["next"] / fut["front"] - 1.0
                add = _merge_asof_to_index(df.index, fut[["carry"]])
                df["carry"] = add["carry"]
                carry_done = True
    except Exception as e:
        _log(f"WARNING: futures carry fetch failed: {e}")

    if not carry_done and client:
        try:
            ticker = (asset_ticker or _infer_asset_ticker())
            if ticker:
                # dividends (sparse)
                divs = None
                for kwargs in (
                    {"ticker": ticker, "limit": 1000, "order": "asc"},
                    {"ticker": ticker, "limit": 1000},
                ):
                    try:
                        divs = client.list_dividends(**kwargs)
                        if divs:
                            break
                    except Exception:
                        continue
                rows = [getattr(d, "__dict__", dict(d)) for d in (getattr(divs, "results", divs) or [])]
                df_div = pd.DataFrame(rows)
                px = _fetch_polygon_aggs(client, ticker, 1, "day", start_s, end_s, limit=5000)
                if not df_div.empty and not px.empty:
                    # choose the best date field
                    for dc in ("EX_DIVIDEND_DATE", "ex_dividend_date", "pay_date", "declaration_date", "record_date"):
                        if dc in df_div.columns:
                            df_div["ts"] = pd.to_datetime(df_div[dc], utc=True, errors="coerce")
                            break
                    if "ts" not in df_div.columns:
                        df_div["ts"] = pd.NaT
                    df_div = df_div[["ts", "cash_amount"]].dropna()
                    df_div["cash_amount"] = pd.to_numeric(df_div["cash_amount"], errors="coerce")
                    if not df_div.empty:
                        div_series = df_div.set_index("ts").sort_index()["cash_amount"]
                        daily = px[["close"]].copy()
                        # daily grid + 365D rolling sum
                        cf = div_series.reindex(daily.index, method=None).fillna(0.0)
                        cf = cf.asfreq("D", method=None).fillna(0.0)
                        ttm = cf.rolling(window="365D").sum()
                        ttm_on_mkt = ttm.reindex(daily.index, method="pad").fillna(method="bfill")
                        dy = (ttm_on_mkt / daily["close"].replace(0, np.nan)).rename("carry")
                        add = _merge_asof_to_index(df.index, dy.to_frame("carry"))
                        df["carry"] = add["carry"]
                        carry_done = True
            if not carry_done:
                _log("INFO: carry not computed (no FUT_* or dividends available).")
        except Exception as e:
            _log(f"WARNING: equity carry build failed: {e}")

    # 5) curve_slope from a futures strip
    try:
        fut_strip_env = os.getenv("FUT_STRIP", "").strip()
        if client and fut_strip_env:
            tickers = [t.strip() for t in fut_strip_env.split(",") if t.strip()]
            cols = []
            for tkr in tickers:
                df_k = _fetch_polygon_aggs(client, tkr, 1, "day", start_s, end_s, limit=5000).rename(columns={"close": tkr})
                if not df_k.empty:
                    cols.append(df_k)
            if cols:
                panel = pd.concat(cols, axis=1).sort_index()
                n = len(tickers)
                x = np.arange(n)
                x_center = x - x.mean()
                def _row_slope(row: pd.Series) -> float:
                    if n <= 1:
                        return np.nan
                    y = row.values.astype(float)
                    msk = np.isfinite(y)
                    if msk.sum() < 2:
                        return np.nan
                    yc = y[msk] - np.nanmean(y[msk])
                    xc = x_center[msk]
                    den = float((xc ** 2).sum())
                    if den == 0:
                        return np.nan
                    return float(np.nansum(yc * xc) / den)
                slope = panel.apply(_row_slope, axis=1).rename("curve_slope")
                add = _merge_asof_to_index(df.index, slope.to_frame("curve_slope"))
                df["curve_slope"] = add["curve_slope"]
    except Exception as e:
        _log(f"WARNING: curve_slope build failed: {e}")

    return df

# ============================================================
# Costs & Vol
# ============================================================

def default_cost_model(signals: pd.Series) -> pd.Series:
    base_bps = 2.0
    if len(signals) < 2:
        return pd.Series([base_bps], index=signals.index)
    d_sig = signals.diff().abs().fillna(0.0)
    impact_bps = 12.0 * d_sig  # tune for your venue
    return (base_bps + impact_bps).clip(lower=0.0)

def estimate_sigma(df: pd.DataFrame, rv_col: str = "rv_20") -> pd.Series:
    if rv_col in df.columns:
        s = df[rv_col].copy()
        if s.median(skipna=True) > 0.05:
            s = np.sqrt(np.clip(s, 0, None))
        return s.fillna(method="ffill").fillna(s.median())
    for cand in ["returns_1", "returns"]:
        if cand in df.columns:
            return df[cand].rolling(20, min_periods=5).std().fillna(method="ffill")
    return pd.Series(1.0, index=df.index)


# ============================================================
# Targets / Labels Builders (shared)
# ============================================================

def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    # try common columns
    for col in ["timestamp", "ts", "date", "datetime", "time"]:
        if col in df.columns:
            d = df.copy()
            d[col] = pd.to_datetime(d[col], utc=True, errors="coerce")
            d = d.set_index(col).sort_index()
            return d
    # fallback - best effort
    d = df.copy()
    d.index = pd.to_datetime(d.index, errors="coerce", utc=True)
    return d.sort_index()

def _forward_return_bps(close: pd.Series, h: int) -> pd.Series:
    fwd = np.log(close.shift(-h) / close)
    return (10_000.0 * fwd).rename(f"fwd_ret_h{h}_bps")

def _returns_1_log(close: pd.Series) -> pd.Series:
    return np.log(close).diff().fillna(0.0)

def _vol_ref_bps(ret1: pd.Series, window: int = 200) -> pd.Series:
    # rolling std of 1-bar log returns, mapped to bps
    return (10_000.0 * ret1.rolling(window, min_periods=max(20, window//5)).std()).rename("vol_ref_bps")

def _label_after_cost(fwd_bps: pd.Series, tau_bps: float) -> pd.Series:
    return (fwd_bps - float(tau_bps) > 0).astype(int).rename(f"label_gt_{tau_bps}bps")

def _rv_next_from_returns(ret1: pd.Series) -> pd.Series:
    # simple 1-step ahead realized variance proxy
    return (ret1.shift(-1) ** 2).rename("rv_next_1")

def _crash_label_fallback(fwd_bps: pd.Series, vol_bps: pd.Series, k_sigma: float = 2.5, abs_floor_bps: Optional[float] = 40.0) -> pd.Series:
    thr = k_sigma * vol_bps.abs().fillna(vol_bps.median())
    if abs_floor_bps is not None:
        thr = np.maximum(thr, abs_floor_bps)
    return (fwd_bps < -thr).astype(int).rename("label_crash")


# ============================================================
# Submodel Registry
# ============================================================

SUBMODELS_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Trend – Short (regression)
    "ts_trend_short": {
        "module": "ts_trend_short",
        "file": os.path.join(_SUB_DIR, "ts_trend_short.py"),
        "kind": "regression",
        "horizon": 1,
        "features": [
            "returns_3","returns_5","macd_line","macd_signal","macd_histogram",
            "adx","rsi","ema_9","ema_21","ema_slope_9","ema_slope_21","ema200_dist",
            "bollinger_percB","bb_bw","rv_20","dd_63","volume_zscore","adv_ratio",
            "days_since_high","days_since_low",
        ],
        "fit_fn": "fit_ts_trend_short_state",
        "params": {"clip_k_bps": 50.0, "rv20_max": None},
        "enabled": True,
    },
    # Trend – Medium (regression, h=6)
    "ts_trend_medium": {
        "module": "ts_trend_medium",
        "file": os.path.join(_SUB_DIR, "ts_trend_medium.py"),
        "kind": "regression",
        "horizon": 6,
        "features": [
            "ema_21","ema_50","ema_200","ema_slope_21","ema_slope_50","ema200_dist",
            "donchian20_up","donchian20_dn","macd_line","macd_signal","adx",
            "rv_20","bb_bw","dd_63","days_since_high","days_since_low",
        ],
        "fit_fn": "fit_ts_trend_medium_state",
        "params": {"clip_k_bps": 60.0, "alpha": 0.5},
        "enabled": True,
    },
    # Adaptive trend (unsupervised/online state)
    "ts_trend_adaptive": {
        "module": "ts_trend_adaptive",
        "file": os.path.join(_SUB_DIR, "ts_trend_adaptive.py"),
        "kind": "unsupervised",
        "horizon": 1,
        "features": ["returns_1","returns_3","rv_20","ema_slope_21","ema200_dist"],
        "fit_fn": "fit_ts_trend_adaptive_state",
        "params": {"clip_k_bps": 50.0, "p_trend_thresh": 0.6, "rls_lambda": 0.985, "rls_delta": 1e4, "bps_scale": 10_000.0},
        "enabled": True,
    },
    # K-Day Reversal (regression)
    "kday_reversal": {
        "module": "kday_reversal_catboost",
        "file": os.path.join(_SUB_DIR, "kday_reversal_catboost.py"),
        "kind": "regression",
        "horizon": 1,
        "features": [
            "returns_1","returns_3","bollinger_percB","rsi","vwap_dev","volume_zscore",
            "obv","candle_body_ratio","wick_dominance","adv_ratio","rv_20","hour_sin","hour_cos",
        ],
        "fit_fn": "fit_kday_reversal_state",
        "params": {"clip_k_bps": 45.0, "n_estimators": 600, "n_ensembles": 5, "random_state": 42},
        "enabled": True,
    },
    # Gap Reversal (classification)
    "gap_reversal": {
        "module": "gap_reversal_xgb",
        "file": os.path.join(_SUB_DIR, "gap_reversal_xgb.py"),
        "kind": "classification",
        "horizon": 1,
        "features": [
            "gap_vs_prev","returns_1","rsi","bollinger_percB","vwap_dev","volume_zscore",
            "transactions","candle_body_ratio","wick_dominance","rv_20",
            "hour_sin","hour_cos","day_of_week_sin","day_of_week_cos",
        ],
        "fit_fn": "fit_gap_reversal_state",
        "params": {"clip_k_bps": 50.0, "rv20_z_max": 3.5, "min_abs_edge_bps": 10.0, "calibrate_platt": True, "n_estimators": 500},
        "enabled": True,
    },
    # Seasonality (classification)
    "seasonality": {
        "module": "seasonality_logit",
        "file": os.path.join(_SUB_DIR, "seasonality_logit.py"),
        "kind": "classification",
        "horizon": 1,
        "features": [
            "hour_sin","hour_cos","day_of_week_sin","day_of_week_cos","month_sin","month_cos","rv_20","adv_ratio",
        ],
        "fit_fn": "fit_seasonality_state",
        "params": {"clip_k_bps": 40.0, "p_floor": 0.02, "p_cap": 0.98, "C": 1.0, "class_weight": "balanced", "move_scale_bps": 40.0},
        "enabled": True,
    },
    # Carry / Term (regression, h=6 by default)
    "carry_term": {
        "module": "carry_term_linear",
        "file": os.path.join(_SUB_DIR, "carry_term_linear.py"),
        "kind": "regression",
        "horizon": 6,
        "features": ["carry","curve_slope","adv_ratio","rv_20","bb_bw","dd_63"],
        "fit_fn": "fit_carry_term_state",
        "params": {"clip_k_bps": 60.0, "model_type": "ridge", "alpha": 0.25, "scaler_mode": "standard"},
        "enabled": True,
    },
    # Vol Timing (rv target)
    "vol_timing": {
        "module": "vol_timing_elasticnet",
        "file": os.path.join(_SUB_DIR, "vol_timing_elasticnet.py"),
        "kind": "rv",
        "horizon": 1,
        "features": ["rv_20","std_10","vov_20","semivar_dn_20","bb_bw","dd_63","adv_ratio","volume_zscore"],
        "fit_fn": "fit_vol_timing_state",
        "params": {
            "target_mode": "rv_log",
            "k_bps_per_logrv": 35.0,
            "sign_when_high_vol": -1,
            # --- ElasticNet/scaler knobs (helps convergence) ---
            "alpha": 0.05,          # ↑ regularisation
            "l1_ratio": 0.1,        # mostly ridge, a touch of lasso
            "max_iter": 20000,      # plenty of iterations
            "tol": 1e-4,            # slightly looser tolerance
            "scaler_mode": "robust" # robust scaling
        },
        "enabled": True,
        },
    # Macro Regime (unsupervised/supervised)
    "macro_regime": {
        "module": "macro_regime_gbt_hmm",
        "file": os.path.join(_SUB_DIR, "macro_regime_gbt_hmm.py"),
        "kind": "unsupervised",
        "horizon": 1,
        "features": ["term_spread","dUST10y","vix_level","dxy_mom","rv_median_xasset"],
        "fit_fn": "fit_macro_regime_state",
        "params": {"emit_signal": False, "clip_k_bps": 40.0, "p_stay_off": 0.96, "p_stay_on": 0.96},
        "enabled": True,
    },
    # Lead-Lag (regression)
    "leadlag": {
        "module": "leadlag_lgbm",
        "file": os.path.join(_SUB_DIR, "leadlag_lgbm.py"),
        "kind": "regression",
        "horizon": 1,
        "features": ["leader_ret_1","leader_ret_2","leader_ret_3","leader_momentum_5","leader_rv_20","spread_z","rv_20"],
        "fit_fn": "fit_leadlag_state",
        "params": {"clip_k_bps": 60.0, "min_abs_edge_bps": None},
        "enabled": True,
    },
    # TCN Small (regression)
    "tcn_small": {
        "module": "tcn_small",
        "file": os.path.join(_SUB_DIR, "tcn_small.py"),
        "kind": "regression",
        "horizon": 1,
        "features": [
            "returns_1","returns_3","returns_5","rsi","macd_line","macd_signal","macd_histogram",
            "ema_slope_9","ema_slope_21","ema200_dist","bollinger_percB","rv_20","volume_zscore",
            "vwap_dev","candle_body_ratio","wick_dominance",
        ],
        "fit_fn": "fit_tcn_state",
        "params": {"clip_k_bps": 60.0, "mc_passes": 16, "window": 128, "hidden_ch": 32, "n_blocks": 3, "kernel_size": 5, "dropout": 0.1, "epochs": 20, "batch_size": 128, "lr": 1e-3},
        "enabled": True,
    },
    # News / Sentiment (classification)
    "news_sentiment": {
        "module": "news_sentiment_lgbm",
        "file": os.path.join(_SUB_DIR, "news_sentiment_lgbm.py"),
        "kind": "classification",
        "horizon": 1,
        "features": ["sentiment","d_sentiment","news_count","news_volume_z","greed_index","volume","volume_zscore","hour_sin","hour_cos","rv_20"],
        "fit_fn": "fit_news_senti_state",
        "params": {"clip_k_bps": 50.0, "min_proba_edge": 0.05, "n_estimators": 700, "eval_fraction": 0.2, "early_stopping_rounds": 60, "random_state": 42},
        "enabled": True,
    },
    # --- NEW: Crash Sentinel (classification) ---
    "crash_sentinel": {
        "module": "crash_sentinel_lgbm",
        "file": os.path.join(_SUB_DIR, "crash_sentinel_lgbm.py"),
        "kind": "crash",
        "horizon": 1,  # set 1..3; labels must match chosen h
        "features": [
            "rv_20","semivar_dn_20","dd_63","bb_bw","adv_ratio","volume_zscore",
            "transactions","macd_line","macd_signal","macd_histogram"
        ],
        "fit_fn": "fit_crash_sentinel_state",
        "params": {"clip_k_bps": 50.0, "p_crash_gate": 0.35, "n_estimators": 600, "eval_fraction": 0.2, "early_stopping_rounds": 50},
        "enabled": True,
    },
}

CONTEXT_MODELS = {"macro_regime", "vol_timing"}  # used as gates/priors


# ============================================================
# Meta Learner (as in your original file)
# ============================================================

class MetaModel:
    def __init__(self, meta_type: Optional[str] = None):
        # normalize aliases
        mt = (meta_type or os.getenv("META_MODEL_TYPE", "ridge")).strip().lower()
        alias_map = {
            "cat": "catboost",
            "cb": "catboost",
            "xgb": "xgboost",
            "lgb": "lightgbm",
            "lgbm": "lightgbm",
        }
        self.meta_type = alias_map.get(mt, mt)

        self._init_models()
        self.resid_span = 60
        self.reg_scaler = None
        self.cls_calibrator = None
        self.feature_list_reg: List[str] = []
        self.feature_list_cls: List[str] = []

        # learned fill values for inference-time imputation
        self.reg_fill_values: Optional[pd.Series] = None
        self.cls_fill_values: Optional[pd.Series] = None

    def _init_models(self):
        self.reg_model = None
        self.cls_model = None

        mt = self.meta_type
        _log(f"META_MODEL_TYPE={mt}")

        lgb = _safe_import("lightgbm")
        xgb = _safe_import("xgboost")
        cat = _safe_import("catboost")

        from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression

        self.LogisticRegression = LogisticRegression
        self.Ridge = Ridge
        self.ElasticNet = ElasticNet

        if mt == "lightgbm" and lgb is not None:
            self.reg_model = lgb.LGBMRegressor(
                objective="regression", learning_rate=0.03, num_leaves=31, n_estimators=600,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, random_state=42,
            )
            self.cls_model = lgb.LGBMClassifier(
                objective="binary", learning_rate=0.03, num_leaves=31, n_estimators=600,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, random_state=42,
            )
        elif mt == "xgboost" and xgb is not None:
            self.reg_model = xgb.XGBRegressor(
                max_depth=4, n_estimators=700, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, random_state=42,
                objective="reg:squarederror",
            )
            self.cls_model = xgb.XGBClassifier(
                max_depth=4, n_estimators=700, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, random_state=42,
                objective="binary:logistic", eval_metric="logloss",
            )
        elif mt == "catboost" and cat is not None:
            self.reg_model = cat.CatBoostRegressor(
                depth=6, iterations=700, learning_rate=0.05, loss_function="RMSE",
                verbose=False, random_seed=42,
            )
            self.cls_model = cat.CatBoostClassifier(
                depth=6, iterations=700, learning_rate=0.05, loss_function="Logloss",
                verbose=False, random_seed=42,
            )
        elif mt == "elasticnet":
            self.reg_model = self.ElasticNet(alpha=0.02, l1_ratio=0.3, random_state=42, max_iter=10000)
            self.cls_model = self.LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, class_weight="balanced")
        else:
            # default: linear models
            self.reg_model = self.Ridge(alpha=1.0)
            self.cls_model = self.LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, class_weight="balanced")

        # If a requested library wasn't available, we already fell back above.
        if (mt in {"lightgbm", "xgboost", "catboost"} and self.reg_model is not None and self.cls_model is not None):
            return
        if mt in {"lightgbm", "xgboost", "catboost"}:
            _log(f"WARNING: '{mt}' not available; falling back to Ridge/Logit.")

    @staticmethod
    def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
        """Replace ±inf with NaN to prepare for imputation."""
        return df.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _align_and_impute(
        X: pd.DataFrame,
        cols: List[str],
        fill_values: Optional[pd.Series],
        default_fill: float = 0.0,
    ) -> pd.DataFrame:
        """Ensure all required columns exist (in order) and impute NaNs."""
        X2 = X.copy()
        # add missing columns
        miss = [c for c in cols if c not in X2.columns]
        for m in miss:
            X2[m] = np.nan
        # drop extras
        extra = [c for c in X2.columns if c not in cols]
        if extra:
            X2.drop(columns=extra, inplace=True)
        # column order
        X2 = X2.loc[:, cols]
        # sanitize and impute
        X2 = MetaModel._sanitize(X2)
        if fill_values is not None and not fill_values.empty:
            # use learned medians where available
            for c in cols:
                fv = fill_values.get(c, default_fill)
                if not np.isfinite(fv):
                    fv = default_fill
                X2[c] = X2[c].fillna(fv)
        else:
            X2 = X2.fillna(default_fill)
        return X2

    def fit(self, X_reg: pd.DataFrame, y_reg_bps: pd.Series, X_cls: pd.DataFrame, y_cls_bin: pd.Series):
        # sanitize
        X_reg = self._sanitize(X_reg)
        X_cls = self._sanitize(X_cls)

        # keep rows fully observed for training
        reg_mask = X_reg.notna().all(axis=1) & y_reg_bps.notna()
        cls_mask = X_cls.notna().all(axis=1) & y_cls_bin.notna()
        Xr, yr = X_reg.loc[reg_mask], y_reg_bps.loc[reg_mask]
        Xc, yc = X_cls.loc[cls_mask], y_cls_bin.loc[cls_mask]

        # remember feature sets
        self.feature_list_reg = list(Xr.columns)
        self.feature_list_cls = list(Xc.columns)

        # learn fill values (medians) from the training frame for inference-time imputation
        self.reg_fill_values = Xr.median(numeric_only=True)
        self.cls_fill_values = Xc.median(numeric_only=True)

        # fit models
        self.reg_model.fit(Xr.values, yr.values)
        self.cls_model.fit(Xc.values, yc.values)

        # residual volatility proxy for pred std
        yhat = pd.Series(self.reg_model.predict(Xr.values), index=Xr.index)
        resid = (yr - yhat)
        self.resid_ewm_std = resid.ewm(span=self.resid_span, adjust=False, min_periods=10).std()

    def predict(self, X_reg: pd.DataFrame, X_cls: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        # Align and impute using training-time medians
        Xr = self._align_and_impute(X_reg, self.feature_list_reg, self.reg_fill_values, default_fill=0.0)
        Xc = self._align_and_impute(X_cls, self.feature_list_cls, self.cls_fill_values, default_fill=0.0)

        # sklearn / most libs expect ndarray
        y_pred_net_bps = pd.Series(self.reg_model.predict(Xr.values), index=Xr.index)

        # classification probas
        if hasattr(self.cls_model, "predict_proba"):
            proba_profit = pd.Series(self.cls_model.predict_proba(Xc.values)[:, 1], index=Xc.index)
        else:
            # fallback for libs without predict_proba (unlikely here)
            logits = pd.Series(self.cls_model.predict(Xc.values), index=Xc.index)
            proba_profit = pd.Series(_sigmoid(logits.values), index=Xc.index)

        # std blend
        pred_std_reg = pd.Series(np.nan, index=y_pred_net_bps.index)
        try:
            resid_std_last = float(self.resid_ewm_std.iloc[-1])
            if math.isfinite(resid_std_last) and resid_std_last > 0:
                pred_std_reg = pd.Series(resid_std_last, index=y_pred_net_bps.index)
        except Exception:
            pass

        entropy_std = _entropy_std_from_proba(proba_profit)
        denom = (np.abs(y_pred_net_bps).rolling(50, min_periods=5).median() + 1e-6)
        pred_std = 0.6 * entropy_std + 0.4 * (pred_std_reg / denom)
        pred_std = pred_std.replace([np.inf, -np.inf], np.nan).fillna(entropy_std)

        return y_pred_net_bps, proba_profit, pred_std

    def dump_state(self) -> Dict[str, Any]:
        return {
            "meta_type": self.meta_type,
            "reg_model": self.reg_model,
            "cls_model": self.cls_model,
            "resid_span": self.resid_span,
            "resid_ewm_std": getattr(self, "resid_ewm_std", None),
            "feature_list_reg": self.feature_list_reg,
            "feature_list_cls": self.feature_list_cls,
            # persist fill values to keep inference deterministic after reload
            "reg_fill_values": self.reg_fill_values,
            "cls_fill_values": self.cls_fill_values,
        }

    @staticmethod
    def load_from_state(state: Dict[str, Any]) -> "MetaModel":
        mm = MetaModel(state.get("meta_type", None))
        mm.reg_model = state["reg_model"]
        mm.cls_model = state["cls_model"]
        mm.resid_span = state.get("resid_span", 60)
        mm.resid_ewm_std = state.get("resid_ewm_std", None)
        mm.feature_list_reg = state.get("feature_list_reg", [])
        mm.feature_list_cls = state.get("feature_list_cls", [])
        mm.reg_fill_values = state.get("reg_fill_values", None)
        mm.cls_fill_values = state.get("cls_fill_values", None)
        return mm


# ============================================================
# Thresholds & Meta State
# ============================================================

def load_thresholds(path: str = _DEFAULT_THRESHOLDS_PATH) -> Dict[str, Any]:
    defaults = {
        "k_bps_signal": 50.0,
        "sigma_target_ann": 0.15,
        "Q_max": 1.0,
        "enter_thresh": 0.35,
        "exit_thresh": 0.20,
        "cost_buffer_bps": 5.0,
        "kill_sharpe_lookback": 60,
        "kill_sharpe_floor": -0.3,
        "drawdown_floor_pct": -12.0,
        # Training/labels knobs
        "label_tau_bps": 5.0,     # after-cost threshold for 'up' labels
        "crash_k_sigma": 2.5,
        "crash_abs_floor_bps": 40.0,
        "val_fraction": 0.2,      # for train_models() split
        "embargo_bars": 5,        # gap between train/val
    }
    user = _load_json(path, default={})
    defaults.update(user or {})
    return defaults

def load_meta_model() -> Optional[MetaModel]:
    if os.path.exists(_DEFAULT_META_STATE_PATH):
        try:
            state = _load_pickle(_DEFAULT_META_STATE_PATH)
            return MetaModel.load_from_state(state)
        except Exception as e:
            _log(f"WARNING: failed to load meta_state: {e}")
    _log("Meta state not found; will use prior-weighted average combiner.")
    return None


# ============================================================
# Meta Features / Weights
# ============================================================

def compute_prior_weight(res: SubmodelResult) -> pd.Series:
    idx = res.y_pred.index
    conf = res.confidence if (res.confidence is not None) else pd.Series(0.5, index=idx)
    ps = res.pred_std if (res.pred_std is not None) else pd.Series(conf.median(), index=idx)
    m = ps.rolling(200, min_periods=20).median()
    mad = (ps - m).abs().rolling(200, min_periods=20).median()
    z = (ps - m) / (mad + 1e-9)
    w = (conf / (1.0 + z.clip(lower=0))).clip(0.0, 1.0)
    return w

def apply_context_gates(
    w: pd.Series,
    macro_res: Optional[SubmodelResult] = None,
    vol_res: Optional[SubmodelResult] = None,
    sleeve: str = "trend"
) -> pd.Series:
    out = w.copy()
    if macro_res and macro_res.proba_up is not None:
        p_on = macro_res.proba_up.reindex(out.index).fillna(method="ffill")
        if sleeve == "trend":
            out *= (0.8 + 0.4 * p_on)
        elif sleeve == "meanrev":
            out *= (1.2 - 0.4 * p_on)
    if vol_res and vol_res.signal is not None:
        v_sig = vol_res.signal.reindex(out.index).fillna(method="ffill")
        if sleeve == "trend":
            out *= (1.0 + (-0.5) * v_sig.clip(lower=0))
        elif sleeve == "meanrev":
            out *= (1.0 + 0.5 * v_sig.clip(lower=0))
    return out.clip(0.0, 1.5)

def meta_feature_builder(
    sub_results: Dict[str, SubmodelResult],
    df_slice: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.Series]]:
    idx = df_slice.index
    col_reg, col_cls = {}, {}
    per_sub_prior: Dict[str, pd.Series] = {}

    macro_res = sub_results.get("macro_regime", None)
    vol_res = sub_results.get("vol_timing", None)

    for name, res in sub_results.items():
        if name in CONTEXT_MODELS:
            continue
        y = res.y_pred.reindex(idx)
        s = res.signal.reindex(idx)
        ps = res.pred_std.reindex(idx) if res.pred_std is not None else pd.Series(np.nan, index=idx)
        cf = res.confidence.reindex(idx) if res.confidence is not None else pd.Series(np.nan, index=idx)
        tm = res.trade_mask.reindex(idx) if res.trade_mask is not None else pd.Series(True, index=idx)

        w0 = compute_prior_weight(res).reindex(idx).fillna(0.5)
        sleeve = "meanrev" if ("reversal" in name) else "trend"
        w_adj = apply_context_gates(w0, macro_res, vol_res, sleeve=sleeve)
        per_sub_prior[name] = w_adj.clip(0.0, 1.5)

        col_reg[f"{name}_y_pred"]  = y
        col_reg[f"{name}_y_conf"]  = y * cf.fillna(0.5)
        col_reg[f"{name}_w_prior"] = w_adj
        col_reg[f"{name}_tm"]      = tm.astype(float)
        col_reg[f"{name}_std"]     = ps

        col_cls[f"{name}_y_pred"]  = y
        col_cls[f"{name}_y_conf"]  = y * cf.fillna(0.5)
        col_cls[f"{name}_w_prior"] = w_adj
        col_cls[f"{name}_tm"]      = tm.astype(float)
        col_cls[f"{name}_std"]     = ps

    if macro_res is not None:
        if macro_res.proba_up is not None:
            col_reg["macro_proba_on"] = macro_res.proba_up.reindex(idx)
            col_cls["macro_proba_on"] = macro_res.proba_up.reindex(idx)
        if macro_res.pred_std is not None:
            col_reg["macro_std"] = macro_res.pred_std.reindex(idx)
            col_cls["macro_std"] = macro_res.pred_std.reindex(idx)

    if vol_res is not None:
        if vol_res.y_pred is not None:
            col_reg["vol_edge_bps"] = vol_res.y_pred.reindex(idx)
            col_cls["vol_edge_bps"] = vol_res.y_pred.reindex(idx)
        if vol_res.pred_std is not None:
            col_reg["vol_std"] = vol_res.pred_std.reindex(idx)
            col_cls["vol_std"] = vol_res.pred_std.reindex(idx)

    for raw in ["rv_20", "bb_bw", "dd_63"]:
        if raw in df_slice.columns:
            col_reg[raw] = df_slice[raw]
            col_cls[raw] = df_slice[raw]

    X_reg = pd.DataFrame(col_reg, index=idx).replace([np.inf, -np.inf], np.nan)
    X_cls = pd.DataFrame(col_cls, index=idx).replace([np.inf, -np.inf], np.nan)
    return X_reg, X_cls, per_sub_prior


# ============================================================
# On-the-fly Training Helpers (core of leakage-safe logic)
# ============================================================

def _import_submodule(name: str, cfg: Dict[str, Any]):
    return _safe_import(cfg["module"], cfg.get("file"))

def _common_targets(df: pd.DataFrame, horizons_needed: List[int]) -> Dict[str, pd.Series]:
    if "close" not in df.columns:
        raise ValueError("Column 'close' is required to build forward-return targets.")
    close = df["close"]
    ret1 = _returns_1_log(close)
    targets: Dict[str, pd.Series] = {
        "ret1": ret1,
        "rv_next": _rv_next_from_returns(ret1),
        "vol_ref_bps": _vol_ref_bps(ret1, window=200),
    }
    for h in sorted(set(horizons_needed)):
        targets[f"fwd_h{h}_bps"] = _forward_return_bps(close, h)
    return targets

def _fit_state_for_model(
    name: str,
    cfg: Dict[str, Any],
    mod: Any,
    df_train: pd.DataFrame,
    targets: Dict[str, pd.Series],
    thresholds: Dict[str, Any],
) -> Optional[Any]:
    """Fit a submodel state on df_train (already leakage-safe)."""
    if df_train is None or len(df_train) < 50:
        return None

    features = cfg["features"]
    missing = [f for f in features if f not in df_train.columns]
    if missing:
        _log(f"'{name}' skipped (missing features): {missing}")
        return None

    fit_fn = getattr(mod, cfg["fit_fn"], None)
    if fit_fn is None:
        _log(f"'{name}' missing fit function '{cfg['fit_fn']}'. Skipping.")
        return None

    kind = cfg["kind"]
    h = int(cfg["horizon"])

    try:
        if kind == "regression":
            y = targets[f"fwd_h{h}_bps"].loc[df_train.index]
            if name == "kday_reversal":
                state = fit_fn(df_train, y, features=features,
                               n_estimators=cfg["params"].get("n_estimators", 600),
                               n_ensembles=cfg["params"].get("n_ensembles", 5),
                               random_state=cfg["params"].get("random_state", 42))
            elif name == "carry_term":
                state = fit_fn(df_train, y, features=features,
                               model_type=cfg["params"].get("model_type","ridge"),
                               alpha=cfg["params"].get("alpha",0.25),
                               scaler_mode=cfg["params"].get("scaler_mode","standard"))
            elif name == "ts_trend_medium":
                state = fit_fn(df_train, y, features=features, alpha=cfg["params"].get("alpha",0.5))
            elif name == "tcn_small":
                p = cfg["params"]
                state = fit_fn(
                    df_train, y, features=features,
                    horizon=h, window=p.get("window",128), hidden_ch=p.get("hidden_ch",32),
                    n_blocks=p.get("n_blocks",3), kernel_size=p.get("kernel_size",5),
                    dropout=p.get("dropout",0.1), epochs=p.get("epochs",20),
                    batch_size=p.get("batch_size",128), lr=p.get("lr",1e-3)
                )
            else:
                state = fit_fn(df_train, y, features=features)
            return state

        elif kind == "classification":
            tau = float(thresholds["label_tau_bps"])
            y = _label_after_cost(targets[f"fwd_h{h}_bps"], tau).loc[df_train.index]
            if name == "gap_reversal":
                p = cfg["params"]
                state = fit_fn(
                    df_train, y, features=features,
                    move_scale_bps=p.get("move_scale_bps", None),
                    calibrate_platt=p.get("calibrate_platt", True),
                    xgb_params={"max_depth": 4, "learning_rate": 0.05},
                    n_estimators=p.get("n_estimators", 500),
                )
            elif name == "seasonality":
                p = cfg["params"]
                state = fit_fn(
                    df_train, y, features=features, C=p.get("C",1.0),
                    class_weight=p.get("class_weight","balanced"),
                    move_scale_bps=p.get("move_scale_bps",40.0),
                )
            elif name == "news_sentiment":
                p = cfg["params"]
                state = fit_fn(
                    df_train, y, features=features, returns_bps=targets[f"fwd_h{h}_bps"].loc[df_train.index],
                    lgb_params=None, n_estimators=p.get("n_estimators",700),
                    eval_fraction=p.get("eval_fraction",0.2),
                    early_stopping_rounds=p.get("early_stopping_rounds",60),
                    random_state=p.get("random_state",42), move_scale_bps=None
                )
            else:
                # default classification fit signature
                state = fit_fn(df_train, y, features=features)
            return state

        elif kind == "rv":
            p = cfg["params"]
            rv_next = targets["rv_next"].loc[df_train.index]
            state = fit_fn(
                df_train, target=rv_next, features=features,
                target_mode=p.get("target_mode", "rv_log"),
                k_bps_per_logrv=p.get("k_bps_per_logrv", 35.0),
                sign_when_high_vol=p.get("sign_when_high_vol", -1),
                # ---- pass-through ElasticNet/scaler knobs ----
                alpha=p.get("alpha", 0.02),
                l1_ratio=p.get("l1_ratio", 0.2),
                scaler_mode=p.get("scaler_mode", "standard"),
                max_iter=p.get("max_iter", 4000),
                tol=p.get("tol", 1e-5),
                random_state=p.get("random_state", 42),
            )
            return state

        elif kind == "unsupervised":
            if name == "ts_trend_adaptive":
                p = cfg["params"]
                hmm_feats = [c for c in ["returns_1","rv_20","rsi","adx"] if c in df_train.columns]
                state = getattr(mod, cfg["fit_fn"])(
                    df_train, features=features, hmm_features=(hmm_feats or None),
                    rls_lambda=p.get("rls_lambda",0.985), rls_delta=p.get("rls_delta",1e4),
                    bps_scale=p.get("bps_scale",10_000.0)
                )
            else:
                p = cfg["params"]
                state = getattr(mod, cfg["fit_fn"])(
                    df_train, y_risk_on=None, features=features,
                    returns_bps=targets["fwd_h1_bps"].loc[df_train.index] if "fwd_h1_bps" in targets else None,
                    p_stay_off=p.get("p_stay_off",0.96), p_stay_on=p.get("p_stay_on",0.96),
                )
            return state

        elif kind == "crash":
            # Use module helper if available
            build_lbl = getattr(mod, "build_crash_label_from_returns", None)
            fwd = targets[f"fwd_h{h}_bps"]
            vol = targets["vol_ref_bps"]
            if build_lbl is not None:
                y = build_lbl(
                    fwd_ret_bps=fwd.loc[df_train.index],
                    vol_ref=vol.loc[df_train.index],
                    k_sigma=float(thresholds["crash_k_sigma"]),
                    absolute_bps_floor=float(thresholds["crash_abs_floor_bps"]),
                )
            else:
                y = _crash_label_fallback(
                    fwd.loc[df_train.index], vol.loc[df_train.index],
                    k_sigma=float(thresholds["crash_k_sigma"]),
                    abs_floor_bps=float(thresholds["crash_abs_floor_bps"]),
                )
            p = cfg["params"]
            state = getattr(mod, cfg["fit_fn"])(
                df_train, y, features=features,
                lgb_params={"num_leaves": 31, "learning_rate": 0.03},
                n_estimators=p.get("n_estimators",600),
                eval_fraction=p.get("eval_fraction",0.2),
                early_stopping_rounds=p.get("early_stopping_rounds",50)
            )
            return state

    except Exception as e:
        _log(f"ERROR: fit failed for '{name}': {e}")

    return None

def _predict_one(
    name: str,
    cfg: Dict[str, Any],
    mod: Any,
    df_slice: pd.DataFrame,
    state: Any
) -> Optional[SubmodelResult]:
    """Call submodel.predict_submodel on df_slice with fitted state."""
    if state is None:
        return None
    if not hasattr(mod, "predict_submodel"):
        _log(f"'{name}' missing predict_submodel(). Skipping.")
        return None
    try:
        res: SubmodelResult = mod.predict_submodel(
            df_slice,
            horizon=int(cfg["horizon"]),
            features=cfg["features"],
            as_of=df_slice.index[-1],
            state=state,
            params=cfg.get("params", {}),
            cost_model=default_cost_model,
        )
        return res
    except Exception as e:
        _log(f"ERROR: predict failed for '{name}': {e}")
        return None


# ============================================================
# Submodel Runner (walk-forward: fit up to i-h, then predict at i)
# ============================================================

def run_one_submodel_walkforward(
    name: str,
    df_slice: pd.DataFrame,
    thresholds: Dict[str, Any],
) -> Optional[SubmodelResult]:
    cfg = SUBMODELS_REGISTRY[name]
    if not cfg.get("enabled", True):
        return None

    mod = _import_submodule(name, cfg)
    if mod is None:
        return None

    h = int(cfg["horizon"])
    if len(df_slice) <= h + 25:
        return None

    # Train set ends at i-h (exclude last h bars)
    df_train = df_slice.iloc[: -h]
    # Fit
    targets = _common_targets(df_slice, horizons_needed=[h, 1])
    state = _fit_state_for_model(name, cfg, mod, df_train, targets, thresholds)
    # Predict on slice ≤ as_of
    return _predict_one(name, cfg, mod, df_slice, state)


def run_all_submodels(df_slice: pd.DataFrame, thresholds: Dict[str, Any]) -> Dict[str, SubmodelResult]:
    results: Dict[str, SubmodelResult] = {}
    for name in SUBMODELS_REGISTRY.keys():
        res = run_one_submodel_walkforward(name, df_slice, thresholds)
        if res is not None:
            results[name] = res
    return results


# ============================================================
# Meta Inference
# ============================================================

def prior_weighted_blend(
    subs: Dict[str, SubmodelResult],
    w_priors: Dict[str, pd.Series],
    idx: pd.Index
) -> Tuple[pd.Series, pd.Series, pd.Series, Dict[str, pd.Series]]:
    num = pd.Series(0.0, index=idx)
    den = pd.Series(0.0, index=idx)
    weights_detail: Dict[str, pd.Series] = {}
    pred_std_acc = pd.Series(0.0, index=idx)

    for name, res in subs.items():
        w = w_priors.get(name, pd.Series(0.5, index=idx)).reindex(idx).fillna(0.5).clip(0.0, 1.5)
        y = res.y_pred.reindex(idx).fillna(0.0)
        s = res.pred_std.reindex(idx).fillna(y.abs().rolling(50, min_periods=5).median() if res.pred_std is None else 0.0)
        num = num.add(w * y, fill_value=0.0)
        den = den.add(w, fill_value=0.0)
        weights_detail[name] = w
        pred_std_acc = pred_std_acc.add(w * s, fill_value=0.0)

    den = den.replace(0.0, np.nan)
    y_blend = (num / den).fillna(0.0)
    proba_profit = pd.Series(_sigmoid(y_blend / (y_blend.abs().rolling(200, min_periods=20).median() + 1e-6)), index=idx)
    pred_std = (pred_std_acc / den.replace(0.0, np.nan)).fillna(pred_std_acc)
    return y_blend, proba_profit, pred_std, weights_detail

def meta_infer_on_slice(
    df_slice: pd.DataFrame,
    thresholds: Dict[str, Any],
    meta_model: Optional[MetaModel],
) -> MetaResult:
    idx = df_slice.index
    sub_results = run_all_submodels(df_slice, thresholds)
    subs_effective = {k: v for k, v in sub_results.items() if k not in CONTEXT_MODELS}
    if not subs_effective:
        zero = pd.Series(0.0, index=idx)
        return MetaResult(
            y_pred_net_bps=zero, proba_profit=zero, pred_std=zero,
            signal=zero, position_suggested=zero, trade_mask=pd.Series(True, index=idx),
            submodel_weights={}, costs_est=zero, diagnostics={"note": "no_submodels_available"}
        )

    X_reg, X_cls, w_priors = meta_feature_builder(sub_results, df_slice)

    blended_signal = pd.Series(0.0, index=idx)
    for name, res in subs_effective.items():
        w = w_priors.get(name, pd.Series(0.5, index=idx)).reindex(idx).fillna(0.5)
        s = res.signal.reindex(idx).fillna(0.0)
        blended_signal = blended_signal.add(w * s, fill_value=0.0)
    costs_est = default_cost_model(blended_signal)

    if meta_model is not None and (meta_model.feature_list_reg and meta_model.feature_list_cls):
        y_pred_net_bps, proba_profit, pred_std = meta_model.predict(X_reg, X_cls)
        weights_detail = None
    elif meta_model is not None:
        _log("Meta model has no feature lists; using prior-weighted average.")
        y_pred_net_bps, proba_profit, pred_std, weights_detail = prior_weighted_blend(subs_effective, w_priors, idx)
    else:
        y_pred_net_bps, proba_profit, pred_std, weights_detail = prior_weighted_blend(subs_effective, w_priors, idx)

    k_bps = float(thresholds["k_bps_signal"])
    signal = pd.Series(np.tanh(y_pred_net_bps / max(k_bps, 1e-6)), index=idx, name="meta_signal")

    sigma = estimate_sigma(df_slice)
    sigma_target_bar = float(thresholds["sigma_target_ann"]) / math.sqrt(252.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pos_suggested = (sigma_target_bar / (sigma.replace(0, np.nan))) * signal
    pos_suggested = pos_suggested.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-float(thresholds["Q_max"]), float(thresholds["Q_max"]))

    trade_mask_series = pd.Series(True, index=idx)
    # for name, res in subs_effective.items():
    #     if res.trade_mask is not None:
    #         trade_mask_series &= res.trade_mask.reindex(idx).fillna(True)
    # trade_mask_series &= (np.abs(y_pred_net_bps) > (costs_est + float(thresholds["cost_buffer_bps"])))

    diagnostics = {"note": "ok", "sigma_target_bar": sigma_target_bar, "k_bps_signal": k_bps}
    return MetaResult(
        y_pred_net_bps=y_pred_net_bps,
        proba_profit=proba_profit,
        pred_std=pred_std,
        signal=signal,
        position_suggested=pos_suggested,
        trade_mask=trade_mask_series,
        submodel_weights=weights_detail,
        costs_est=costs_est,
        diagnostics=diagnostics,
    )


# ============================================================
# Decisioning
# ============================================================

def decide_action(
    signal: float,
    position_open: bool,
    enter_thresh: float,
    exit_thresh: float,
    trade_mask: bool
) -> str:
    if not trade_mask:
        return "SELL" if position_open else "NONE"
    abs_s = abs(signal)
    if not position_open:
        if abs_s >= enter_thresh:
            return "BUY" if signal > 0 else "SELL"
        return "NONE"
    else:
        if abs_s <= exit_thresh:
            return "SELL" if position_open else "NONE"
        return "NONE"


# ============================================================
# I/O (UPDATED to compute features + macro features)
# ============================================================

def _load_df_from_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV_PATH not found: {path}")
    base = pd.read_csv(path)
    base = _ensure_dt_index(base)

    # Build technical & leader features (uses default leader at sub/data/QQQ_h4.csv; override via env)
    leader_path_env = os.getenv("LEADER_CSV_PATH", os.path.join("data", "QQQ_H4.csv"))
    # Optional extra basket CSVs (comma-separated)
    basket_env = os.getenv("XASSET_BASKET_CSVS", "").strip()
    basket_list = [p for p in (s.strip() for s in basket_env.split(",")) if p] if basket_env else None

    df = compute_features(base, leader_csv_path=leader_path_env, rv_basket_csvs=basket_list, price_col="close")

    # Append macro features from Polygon
    asset_ticker = _infer_asset_ticker()
    dxy_proxy = os.getenv("DXY_PROXY_TICKER", "UUP")
    print("running compute_macro_features")
    df = compute_macro_features(df, asset_ticker=asset_ticker, dxy_proxy=dxy_proxy)

    return df

# ============================================================
# Meta training helpers (NEW)
# ============================================================

def _build_meta_training_frame(
    df_full: pd.DataFrame,
    thresholds: Dict[str, Any],
    start_i: Optional[int] = None,
    end_i: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Walk-forward, leakage-safe construction of meta features/labels.
    At each i in [start_i, end_i], we:
      - Fit each submodel on df[:i-h] and predict on df[:i] (via run_all_submodels)
      - Build meta features on the *last* row only (df[:i].tail(1))
      - Record y_reg (fwd_h1_bps at i) and y_cls (after-cost label at i)
    Returns X_reg, y_reg_bps, X_cls, y_cls_bin all indexed by the as-of timestamps.
    """
    assert len(df_full) > 0, "empty df_full"
    # horizons used by targets (meta learns h=1 by design)
    targets = _common_targets(df_full, horizons_needed=[1])

    # choose warmup/start defaults (enough bars for rolling features and submodel fitting)
    max_h = max(int(cfg["horizon"]) for cfg in SUBMODELS_REGISTRY.values() if cfg.get("enabled", True))
    default_start = max(200, max_h + 30)
    start_i = default_start if start_i is None else max(default_start, int(start_i))
    end_i = (len(df_full) - 1) if end_i is None else int(end_i)
    end_i = max(start_i, min(end_i, len(df_full) - 1))

    tau = float(thresholds["label_tau_bps"])

    Xr_rows: List[pd.DataFrame] = []
    Xc_rows: List[pd.DataFrame] = []
    y_reg_vals: List[float] = []
    y_cls_vals: List[int] = []
    idx_out: List[pd.Timestamp] = []

    for i in range(start_i, end_i + 1):
        df_slice = df_full.iloc[: i + 1]
        subs = run_all_submodels(df_slice, thresholds)
        if not subs:
            continue
        # Features built on the current as-of row only
        Xi_reg, Xi_cls, _ = meta_feature_builder(subs, df_slice.tail(1))
        # Labels at the same as-of index i (next move from i, not using any future submodel data)
        y_bps = float(targets["fwd_h1_bps"].iloc[i])
        y_bin = int((y_bps - tau) > 0)

        Xr_rows.append(Xi_reg)
        Xc_rows.append(Xi_cls)
        y_reg_vals.append(y_bps)
        y_cls_vals.append(y_bin)
        idx_out.append(df_slice.index[-1])

    if not Xr_rows:
        _log("WARNING: meta training frame has 0 rows.")
        return (pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), pd.Series(dtype=int))

    Xr = pd.concat(Xr_rows, axis=0)
    Xc = pd.concat(Xc_rows, axis=0)
    y_reg = pd.Series(y_reg_vals, index=idx_out, name="y_reg_bps")
    y_cls = pd.Series(y_cls_vals, index=idx_out, name="y_cls_bin")

    # Drop rows with any NaNs across the features we ended up with
    mask = Xr.notna().all(axis=1) & Xc.notna().all(axis=1)
    Xr, Xc, y_reg, y_cls = Xr.loc[mask], Xc.loc[mask], y_reg.loc[mask], y_cls.loc[mask]

    return Xr, y_reg, Xc, y_cls

def _history_csv_path_for_ticker(ticker: str) -> str:
    """Where we persist walk-forward submodel outputs."""
    fname = f"{ticker.upper()}_HISTORY.csv"
    return os.path.join(_ARTIFACTS_DIR, fname)

def _load_df_for_ticker(ticker: str) -> pd.DataFrame:
    """Wrapper that resolves the csv path for ticker, then builds all features."""
    path = CSV_PATH
    return _load_df_from_csv(path)

def _load_history_for_ticker(ticker: str) -> pd.DataFrame:
    """Load history CSV if present; returns empty DataFrame if missing.
    - Ensures UTC DatetimeIndex named 'ts'
    - Sorts, drops duplicate timestamps (keeps last)
    """
    p = _history_csv_path_for_ticker(ticker)
    if not os.path.exists(p):
        return pd.DataFrame()
    try:
        df = pd.read_csv(p, parse_dates=["ts"])
        if "ts" not in df.columns:
            # tolerate old files with 'timestamp' or unnamed first col
            if "timestamp" in df.columns:
                df.rename(columns={"timestamp": "ts"}, inplace=True)
            elif df.shape[1] > 0:
                df.rename(columns={df.columns[0]: "ts"}, inplace=True)
        idx = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        out = df.drop(columns=["ts"], errors="ignore")
        out.index = idx
        out.index.name = "ts"
        out = out[~out.index.duplicated(keep="last")].sort_index()
        return out
    except Exception as e:
        _log(f"WARNING: failed to read history '{p}': {e}")
        return pd.DataFrame()

def _history_column_order() -> List[str]:
    """
    Deterministic superset of possible HISTORY columns, in stable order.
    We always prefer this order when creating or upgrading the file.
    """
    base_fields = ["y_pred", "signal", "proba_up", "pred_std", "confidence", "trade_mask"]
    cols: List[str] = ["ts"]
    for name in SUBMODELS_REGISTRY.keys():
        for f in base_fields:
            cols.append(f"{name}__{f}")
    return cols

def _upgrade_history_header_if_needed(path: str, desired_cols: List[str]) -> List[str]:
    """If the on-disk CSV header is missing new columns, widen it in-place (no recompute)."""
    if not os.path.exists(path):
        return desired_cols
    try:
        existing_cols = list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        # If header cannot be read, leave it to append logic to recreate
        return desired_cols

    if not existing_cols:
        return desired_cols

    need_upgrade = any(c not in existing_cols for c in desired_cols)
    if not need_upgrade:
        return existing_cols

    try:
        # Read full CSV, widen to union, rewrite
        df = pd.read_csv(path)
        cols_union = _stable_union_columns(existing_cols, desired_cols)
        for c in cols_union:
            if c not in df.columns:
                df[c] = np.nan
        df = df.reindex(columns=cols_union)
        df.to_csv(path, index=False)
        _log(f"[history] header upgraded with {len(cols_union)} columns.")
        return cols_union
    except Exception as e:
        _log(f"WARNING: failed to upgrade HISTORY header: {e}")
        return existing_cols

def _stable_union_columns(existing: List[str], desired: List[str]) -> List[str]:
    """Return a stable union with 'ts' first, then desired order, then any extras."""
    if not existing:
        return desired
    out = ["ts"]
    desired_body = [c for c in desired if c != "ts"]
    exist_body = [c for c in existing if c != "ts"]
    # Desired first
    out.extend(desired_body)
    # Then any existing columns not yet included
    out.extend([c for c in exist_body if c not in desired_body])
    # De-dup while preserving order
    seen = set()
    uniq = []
    for c in out:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq

def _append_history_row(ticker: str, row: Dict[str, Any]) -> None:
    """
    Append a single candle row to the HISTORY CSV for `ticker`.
    - Creates the file with a full header if it doesn't exist.
    - If the file exists with a different header, we **upgrade** it in-place
      to the superset (append-only; no recompute).
    """
    path = _history_csv_path_for_ticker(ticker)
    desired_cols = _history_column_order()

    # Ensure header exists / upgraded before writing
    write_header = not os.path.exists(path)
    cols_to_use = desired_cols
    if not write_header:
        cols_to_use = _upgrade_history_header_if_needed(path, desired_cols)

    # Build a 1-row DataFrame aligned to chosen header
    rec = {c: row.get(c, np.nan) for c in cols_to_use}
    # Ensure proper types
    ts_val = row.get("ts", None)
    rec["ts"] = pd.to_datetime(ts_val, utc=True, errors="coerce")
    for c in cols_to_use:
        if c.endswith("__trade_mask") and c in row and row[c] is not None:
            try:
                rec[c] = int(bool(row[c]))
            except Exception:
                rec[c] = np.nan

    df1 = pd.DataFrame([rec]).reindex(columns=cols_to_use)
    df1.to_csv(path, mode="a", header=write_header, index=False)

def _flatten_subresults_last_row(subs: Dict[str, SubmodelResult], ts: pd.Timestamp) -> Dict[str, Any]:
    """
    Take the last point of each SubmodelResult in `subs` and flatten into a row dict
    with columns like '{name}__y_pred', '{name}__signal', '{name}__proba_up', '{name}__pred_std',
    '{name}__confidence', '{name}__trade_mask', plus 'ts'.
    """
    row: Dict[str, Any] = {}
    for name, res in subs.items():
        pfx = f"{name}__"

        def last(series: Optional[pd.Series]):
            if series is None or len(series) == 0:
                return np.nan
            try:
                return float(series.iloc[-1])
            except Exception:
                return np.nan

        row[pfx + "y_pred"]     = last(res.y_pred)
        row[pfx + "signal"]     = last(res.signal)
        row[pfx + "proba_up"]   = last(res.proba_up)
        row[pfx + "pred_std"]   = last(res.pred_std)
        row[pfx + "confidence"] = last(res.confidence)

        # trade_mask as 0/1
        tmv = res.trade_mask.iloc[-1] if (res.trade_mask is not None and len(res.trade_mask) > 0) else True
        row[pfx + "trade_mask"] = int(bool(tmv))

    row["ts"] = ts
    return row

def _default_history_start_index(df_full: pd.DataFrame) -> int:
    """Choose a conservative warmup start index for history generation."""
    max_h = max(int(cfg["horizon"]) for cfg in SUBMODELS_REGISTRY.values() if cfg.get("enabled", True))
    warmup_env = os.getenv("HISTORY_WARMUP_BARS", "").strip()
    if warmup_env.isdigit():
        return max(int(warmup_env), max_h + 30, 200)
    return max(200, max_h + 30)

def _generate_history_rows(
    df_full: pd.DataFrame,
    thresholds: Dict[str, Any],
    start_i: int,
    end_i: int,
    ticker: Optional[str] = None,
    stream_to_csv: bool = True,
) -> pd.DataFrame:
    """
    Walk-forward compute submodel outputs for i in [start_i..end_i].
    If `stream_to_csv` and `ticker` are provided, append each row to HISTORY.csv
    immediately after it is computed (line-by-line), allowing safe interruption.
    Returns a DataFrame with the rows computed in this call (for in-memory use).
    """
    recs: List[Dict[str, Any]] = []
    end_i = min(end_i, len(df_full) - 1)

    for i in range(start_i, end_i + 1):
        df_slice = df_full.iloc[: i + 1]
        subs = run_all_submodels(df_slice, thresholds)
        if not subs:
            continue

        row = _flatten_subresults_last_row(subs, df_slice.index[-1])

        if stream_to_csv and ticker is not None:
            try:
                _append_history_row(ticker, row)
            except Exception as e:
                _log(f"[history] append failed at i={i}: {e}")

        recs.append(row)

    if not recs:
        return pd.DataFrame()

    out = pd.DataFrame.from_records(recs).set_index("ts").sort_index()
    out.index = pd.to_datetime(out.index, utc=True)
    out.index.name = "ts"
    return out

def _ensure_history_up_to_date(
    ticker: str,
    df_full: pd.DataFrame,
    thresholds: Dict[str, Any],
    min_coverage: float = 0.95  # kept for backwards compat; no longer used to trigger rebuilds
) -> pd.DataFrame:
    """
    Ensure (ticker)_HISTORY.csv exists and is append-only up to date.
    - If missing/empty: create file and stream from warmup.
    - Otherwise: **always append** from the last written timestamp forward.
    - Never delete/rebuild due to low coverage; we only rebuild if the file is unreadable.
    Returns a DataFrame of the up-to-date history aligned to df_full.index.
    """
    path = _history_csv_path_for_ticker(ticker)
    idx_all = pd.DatetimeIndex(df_full.index)

    # 1) Try to load what we have; if unreadable, fall back to full build
    hist = _load_history_for_ticker(ticker)

    if hist.empty:
        _log(f"[history] {ticker}: initializing history (streaming) ...")
        si = _default_history_start_index(df_full)
        _generate_history_rows(df_full, thresholds, si, len(df_full) - 1, ticker=ticker, stream_to_csv=True)
        hist = _load_history_for_ticker(ticker)
        return hist.loc[hist.index.intersection(idx_all)].sort_index()

    # 2) Sanitize loaded history to our known index
    hist = hist.loc[hist.index.intersection(idx_all)].sort_index()
    if hist.empty:
        # File had timestamps outside our df_full (e.g., TZ mismatch). Start fresh append.
        _log(f"[history] {ticker}: header ok, but no overlapping timestamps; will append from warmup.")
        si = _default_history_start_index(df_full)
        _generate_history_rows(df_full, thresholds, si, len(df_full) - 1, ticker=ticker, stream_to_csv=True)
        hist = _load_history_for_ticker(ticker)
        return hist.loc[hist.index.intersection(idx_all)].sort_index()

    # 3) Append strictly from the last written timestamp forward
    last_hist_ts = hist.index.max()
    last_data_ts = idx_all.max()
    if pd.isna(last_hist_ts) or (last_hist_ts >= last_data_ts):
        # Already up-to-date
        return hist.loc[hist.index.intersection(idx_all)].sort_index()

    # Find first df_full index strictly greater than last_hist_ts
    # (np.searchsorted handles exact matches; 'right' ensures we start after the last written candle)
    try:
        start_i = int(np.searchsorted(idx_all.values, last_hist_ts.to_datetime64(), side="right"))
    except Exception:
        # Fallback if to_datetime64 fails
        start_i = int(np.searchsorted(idx_all.view(np.int64), last_hist_ts.value, side="right"))

    # Obey warmup floor so we never ask submodels to predict pre-warmup
    start_i = max(start_i, _default_history_start_index(df_full))

    if start_i <= len(df_full) - 1:
        _log(f"[history] {ticker}: appending rows from i={start_i} (streaming) ...")
        _generate_history_rows(df_full, thresholds, start_i, len(df_full) - 1, ticker=ticker, stream_to_csv=True)
        hist = _load_history_for_ticker(ticker)

    # 4) Return trimmed & ordered history
    hist = hist.loc[hist.index.intersection(idx_all)].sort_index()
    return hist

# ========================================================
# NEW: Build SubmodelResult dict directly from HISTORY CSV
# ========================================================
def _subresults_from_history_on_slice(history_df: pd.DataFrame, df_slice: pd.DataFrame) -> Dict[str, SubmodelResult]:
    """
    Recreate per-sub SubmodelResult using timeseries from history on the slice index.
    Only fields needed by meta_feature_builder/prior weighting are reconstructed.
    """
    idx = df_slice.index
    hist = history_df.reindex(idx)  # align to slice
    out: Dict[str, SubmodelResult] = {}

    for name in SUBMODELS_REGISTRY.keys():
        pfx = f"{name}__"
        cols = {
            "y_pred":      pfx + "y_pred",
            "signal":      pfx + "signal",
            "proba_up":    pfx + "proba_up",
            "pred_std":    pfx + "pred_std",
            "confidence":  pfx + "confidence",
            "trade_mask":  pfx + "trade_mask",
        }

        # If this submodel wasn't persisted (e.g., disabled), skip it
        if not any(c in hist.columns for c in cols.values()):
            continue

        def get_series(key: str) -> Optional[pd.Series]:
            col = cols[key]
            if col not in hist.columns:
                return None
            s = pd.to_numeric(hist[col], errors="coerce")
            s.index = idx
            if key == "trade_mask":
                # trade_mask stored as 0/1; coerce to bool with NaNs treated as True (tradable)
                return s.fillna(1.0).astype(bool)
            return s

        # Explicit None checks (avoid using `or` with Series)
        y_pred_s = get_series("y_pred")
        if y_pred_s is None:
            y_pred_s = pd.Series(np.nan, index=idx)

        signal_s = get_series("signal")
        if signal_s is None:
            signal_s = pd.Series(0.0, index=idx)

        proba_up_s = get_series("proba_up")  # can remain None
        pred_std_s = get_series("pred_std")  # can remain None
        confidence_s = get_series("confidence")  # can remain None

        trade_mask_s = get_series("trade_mask")
        if trade_mask_s is None:
            trade_mask_s = pd.Series(True, index=idx)

        res = SubmodelResult(
            y_pred=y_pred_s,
            signal=signal_s,
            proba_up=proba_up_s,
            pred_std=pred_std_s,
            confidence=confidence_s,
            trade_mask=trade_mask_s,
            model_name=name,
            warmup_bars=0,
        )
        out[name] = res

    return out

def meta_infer_on_slice_from_history(
    df_slice: pd.DataFrame,
    thresholds: Dict[str, Any],
    meta_model: Optional[MetaModel],
    history_df: pd.DataFrame,
) -> MetaResult:
    """Same contract as meta_infer_on_slice, but uses precomputed submodel outputs from HISTORY instead of refitting."""
    idx = df_slice.index
    sub_results = _subresults_from_history_on_slice(history_df, df_slice)
    subs_effective = {k: v for k, v in sub_results.items() if k not in CONTEXT_MODELS}
    if not subs_effective:
        zero = pd.Series(0.0, index=idx)
        return MetaResult(
            y_pred_net_bps=zero, proba_profit=zero, pred_std=zero,
            signal=zero, position_suggested=zero, trade_mask=pd.Series(True, index=idx),
            submodel_weights={}, costs_est=zero, diagnostics={"note": "no_submodels_available_from_history"}
        )

    X_reg, X_cls, w_priors = meta_feature_builder(sub_results, df_slice)

    blended_signal = pd.Series(0.0, index=idx)
    for name, res in subs_effective.items():
        w = w_priors.get(name, pd.Series(0.5, index=idx)).reindex(idx).fillna(0.5)
        s = res.signal.reindex(idx).fillna(0.0)
        blended_signal = blended_signal.add(w * s, fill_value=0.0)
    costs_est = default_cost_model(blended_signal)

    if meta_model is not None and (meta_model.feature_list_reg and meta_model.feature_list_cls):
        y_pred_net_bps, proba_profit, pred_std = meta_model.predict(X_reg, X_cls)
        weights_detail = None
    elif meta_model is not None:
        _log("Meta model has no feature lists; using prior-weighted blend (history).")
        y_pred_net_bps, proba_profit, pred_std, weights_detail = prior_weighted_blend(subs_effective, w_priors, idx)
    else:
        y_pred_net_bps, proba_profit, pred_std, weights_detail = prior_weighted_blend(subs_effective, w_priors, idx)

    k_bps = float(thresholds["k_bps_signal"])
    signal = pd.Series(np.tanh(y_pred_net_bps / max(k_bps, 1e-6)), index=idx, name="meta_signal")

    sigma = estimate_sigma(df_slice)
    sigma_target_bar = float(thresholds["sigma_target_ann"]) / math.sqrt(252.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pos_suggested = (sigma_target_bar / (sigma.replace(0, np.nan))) * signal
    pos_suggested = pos_suggested.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-float(thresholds["Q_max"]), float(thresholds["Q_max"]))

    trade_mask_series = pd.Series(True, index=idx)
    # for name, res in subs_effective.items():
    #     if res.trade_mask is not None:
    #        trade_mask_series &= res.trade_mask.reindex(idx).fillna(True)
    # trade_mask_series &= (np.abs(y_pred_net_bps) > (costs_est + float(thresholds["cost_buffer_bps"])))

    diagnostics = {"note": "ok_from_history", "sigma_target_bar": sigma_target_bar, "k_bps_signal": k_bps}
    return MetaResult(
        y_pred_net_bps=y_pred_net_bps,
        proba_profit=proba_profit,
        pred_std=pred_std,
        signal=signal,
        position_suggested=pos_suggested,
        trade_mask=trade_mask_series,
        submodel_weights=weights_detail,
        costs_est=costs_est,
        diagnostics=diagnostics,
    )

# =======================================================
# NEW: Meta training frame built from HISTORY (fast path)
# =======================================================

def _build_meta_training_frame_from_history(
    df_full: pd.DataFrame,
    history_df: pd.DataFrame,
    thresholds: Dict[str, Any],
    start_i: Optional[int] = None,
    end_i: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Same output as _build_meta_training_frame, but uses saved submodel history.
    At each i, we build meta features from history on df[:i] (leakage-safe) and
    take labels at i (fwd_h1_bps).
    """
    assert len(df_full) > 0, "empty df_full"
    targets = _common_targets(df_full, horizons_needed=[1])

    default_start = _default_history_start_index(df_full)
    start_i = default_start if start_i is None else max(default_start, int(start_i))
    end_i = (len(df_full) - 1) if end_i is None else int(end_i)
    end_i = max(start_i, min(end_i, len(df_full) - 1))

    tau = float(thresholds["label_tau_bps"])

    Xr_rows: List[pd.DataFrame] = []
    Xc_rows: List[pd.DataFrame] = []
    y_reg_vals: List[float] = []
    y_cls_vals: List[int] = []
    idx_out: List[pd.Timestamp] = []

    for i in range(start_i, end_i + 1):
        df_slice = df_full.iloc[: i + 1]
        subs = _subresults_from_history_on_slice(history_df, df_slice)
        if not subs:
            continue
        Xi_reg, Xi_cls, _ = meta_feature_builder(subs, df_slice.tail(1))

        y_bps = float(targets["fwd_h1_bps"].iloc[i])
        y_bin = int((y_bps - tau) > 0)

        Xr_rows.append(Xi_reg)
        Xc_rows.append(Xi_cls)
        y_reg_vals.append(y_bps)
        y_cls_vals.append(y_bin)
        idx_out.append(df_slice.index[-1])

    if not Xr_rows:
        _log("WARNING: meta history training frame has 0 rows.")
        return (pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), pd.Series(dtype=int))

    Xr = pd.concat(Xr_rows, axis=0)
    Xc = pd.concat(Xc_rows, axis=0)
    y_reg = pd.Series(y_reg_vals, index=idx_out, name="y_reg_bps")
    y_cls = pd.Series(y_cls_vals, index=idx_out, name="y_cls_bin")

    mask = Xr.notna().all(axis=1) & Xc.notna().all(axis=1)
    Xr, Xc, y_reg, y_cls = Xr.loc[mask], Xc.loc[mask], y_reg.loc[mask], y_cls.loc[mask]
    return Xr, y_reg, Xc, y_cls


def train_meta_model_from_history(
    df_full: pd.DataFrame,
    thresholds: Dict[str, Any],
    train_end_i: Optional[int] = None,
    save_path: str = _DEFAULT_META_STATE_PATH,
    history_df: Optional[pd.DataFrame] = None,
) -> Optional[MetaModel]:
    """
    Trains the meta model on a *historical* window only (≤ train_end_i),
    using precomputed submodel predictions from HISTORY if provided
    (falls back to slow path if not).
    """
    if len(df_full) < 300:
        _log("Meta: not enough rows to train (need ≥300).")
        return None

    if train_end_i is None:
        train_end_i = len(df_full) - 1

    if history_df is not None and not history_df.empty:
        Xr, y_reg, Xc, y_cls = _build_meta_training_frame_from_history(
            df_full=df_full,
            history_df=history_df,
            thresholds=thresholds,
            start_i=None,
            end_i=train_end_i,
        )
    else:
        # Fallback (slow) to original builder that refits submodels
        Xr, y_reg, Xc, y_cls = _build_meta_training_frame(
            df_full=df_full,
            thresholds=thresholds,
            start_i=None,
            end_i=train_end_i,
        )

    if Xr.empty or Xc.empty:
        _log("Meta: training frame empty after NA filtering.")
        return None

    mm = MetaModel(os.getenv("META_MODEL_TYPE", "ridge"))
    mm.fit(Xr, y_reg, Xc, y_cls)
    _save_pickle(save_path, mm.dump_state())
    _log(f"Meta: trained on {len(Xr)} rows, {Xr.shape[1]} reg feats / {Xc.shape[1]} cls feats. Saved -> {save_path}")
    return mm


def train_meta_state_offline(exclude_last_n: int = 0) -> Optional[MetaModel]:
    """
    Convenience entrypoint:
      - Loads CSV
      - Excludes the last `exclude_last_n` bars from meta training (to avoid peeking at a holdout/backtest)
      - Trains meta model from history and persists state.
    """
    thresholds = load_thresholds()
    df_full = _load_df_from_csv(CSV_PATH)
    if len(df_full) == 0:
        _log("Meta: no data loaded.")
        return None

    train_end_i = len(df_full) - 1 - int(max(0, exclude_last_n))
    if train_end_i < 250:
        _log("Meta: window too small after exclusion.")
        return None

    return train_meta_model_from_history(df_full, thresholds, train_end_i=train_end_i)



# ============================================================
# Public API
# ============================================================

def run_live(return_result: bool = True, position_open: bool = False):
    """
    Live = train up to (T - h) for each sub, then predict at T.
    """
    thresholds = load_thresholds()
    df = _load_df_from_csv(CSV_PATH)
    if len(df) == 0:
        return "NONE" if return_result else None

    meta_model = load_meta_model()
    meta_res = meta_infer_on_slice(df, thresholds, meta_model)

    ts = df.index[-1]
    sig = float(meta_res.signal.iloc[-1])
    tm = bool(meta_res.trade_mask.iloc[-1])
    action = decide_action(
        signal=sig,
        position_open=position_open,
        enter_thresh=float(thresholds["enter_thresh"]),
        exit_thresh=float(thresholds["exit_thresh"]),
        trade_mask=tm,
    )

    if return_result:
        return action
    return None


def run_backtest(
    ticker: str,
    backtest_amount: int,
    return_df: bool = True,
    pretrain_meta: Optional[bool] = True
) -> Optional[pd.DataFrame]:
    """
    Backtest = iterate i over the selected window.
      - Ensures (ticker)_HISTORY.csv is up-to-date for df_full.
      - Optionally pretrains meta on the **pre-backtest** window (history-only),
        then runs inference using HISTORY at each step (no submodel refits).
      - For backtest, the meta model trains strictly on predictions/outputs
        *before* the ones it is asked to predict (no leakage).

    UPDATED:
      - Adds a 'timestamp' column (UTC) mirroring the DatetimeIndex ('ts') so external
        listeners that call .sort_values("timestamp") won't raise KeyError.
    """
    thresholds = load_thresholds()
    df_full = _load_df_for_ticker(ticker)
    if len(df_full) < 200:
        return pd.DataFrame() if return_df else None

    # Ensure we have current history for this ticker
    hist = _ensure_history_up_to_date(
        ticker,
        df_full,
        thresholds,
        min_coverage=float(os.getenv("HISTORY_MIN_COVERAGE", "0.95")),
    )

    # Determine evaluation window
    start_i = 1
    if backtest_amount and backtest_amount > 0 and backtest_amount < len(df_full):
        start_i = len(df_full) - backtest_amount
    start_i = max(start_i, _default_history_start_index(df_full))  # obey warmup

    # Should we pretrain the meta model on the *pre-backtest* segment?
    if pretrain_meta is None:
        pretrain_meta = os.getenv("META_PRETRAIN_IN_BACKTEST", "1") == "1"

    meta_model = None
    if pretrain_meta and start_i > 50:
        # Train meta using only data up to start_i-1, from HISTORY
        meta_model = train_meta_model_from_history(
            df_full=df_full,
            thresholds=thresholds,
            train_end_i=start_i - 1,
            history_df=hist,
        )
    if meta_model is None:
        meta_model = load_meta_model()  # may be None => prior-weighted blend

    # Walk forward and infer using history at each step
    records: List[Dict[str, Any]] = []
    position_open = False

    for i in range(start_i, len(df_full)):
        df_slice = df_full.iloc[: i + 1]
        meta_res = meta_infer_on_slice_from_history(df_slice, thresholds, meta_model, hist)
        # right after meta_res = meta_infer_on_slice_from_history(...)
        ts = pd.to_datetime(df_slice.index[-1], utc=True)
        print(
            f"[META] {ts}  y_pred_bps={float(meta_res.y_pred_net_bps.iloc[-1]):+.2f}  "
            f"proba={float(meta_res.proba_profit.iloc[-1]):.3f}  "
            f"signal={float(meta_res.signal.iloc[-1]):+.3f}  "
            f"pred_std={float(meta_res.pred_std.iloc[-1]) if meta_res.pred_std is not None else float('nan'):.3f}  "
            f"trade_mask={bool(meta_res.trade_mask.iloc[-1])}"
        )
        sig = float(meta_res.signal.iloc[-1])
        tm = bool(meta_res.trade_mask.iloc[-1])
        action = decide_action(
            signal=sig,
            position_open=position_open,
            enter_thresh=float(thresholds["enter_thresh"]),
            exit_thresh=float(thresholds["exit_thresh"]),
            trade_mask=tm,
        )
        if action == "BUY":
            position_open = True
        elif action == "SELL":
            position_open = False

        ts = pd.to_datetime(df_slice.index[-1], utc=True)
        rec = {
            "ts": ts,
            "action": action,
            "signal": sig,
            "position_suggested": float(meta_res.position_suggested.iloc[-1]),
            "y_pred_net_bps": float(meta_res.y_pred_net_bps.iloc[-1]),
            "proba_profit": float(meta_res.proba_profit.iloc[-1]),
            "pred_std": float(meta_res.pred_std.iloc[-1]) if meta_res.pred_std is not None else np.nan,
            "trade_mask": tm,
            "costs_est": float(meta_res.costs_est.iloc[-1]) if meta_res.costs_est is not None else np.nan,
        }
        if meta_res.diagnostics:
            for k in ["sigma_target_bar", "k_bps_signal"]:
                if k in meta_res.diagnostics:
                    rec[k] = meta_res.diagnostics[k]
        records.append(rec)

    out_df = pd.DataFrame.from_records(records)

    # Ensure DatetimeIndex named 'ts' (UTC), keep a 'timestamp' column for external sorters
    if not out_df.empty:
        out_df["ts"] = pd.to_datetime(out_df["ts"], utc=True, errors="coerce")
        out_df = out_df.set_index("ts").sort_index()
        out_df.index.name = "ts"
        # Provide a real column that mirrors the index for consumers expecting 'timestamp'
        out_df["timestamp"] = out_df.index  # this is UTC
    return out_df if return_df else None

def _extract_feature_importance_from_state(state: Any, feature_names: List[str]) -> Optional[pd.Series]:
    """
    Best-effort feature importance extraction from a submodel 'state'.
    Priority:
      1) state['feature_importance'] if present
      2) state['model'].feature_importances_ if present
      3) state['model'].coef_ mapped to features (abs value)
    """
    try:
        if isinstance(state, dict):
            if "feature_importance" in state and isinstance(state["feature_importance"], pd.Series):
                s = state["feature_importance"]
                return s.reindex(feature_names).fillna(0.0)
            mdl = state.get("model", None)
            if mdl is not None:
                if hasattr(mdl, "feature_importances_"):
                    vals = np.array(getattr(mdl, "feature_importances_"))
                    vals = (vals / (vals.sum() + 1e-12))
                    return pd.Series(vals, index=feature_names).fillna(0.0)
                if hasattr(mdl, "coef_"):
                    coef = getattr(mdl, "coef_")
                    if coef is not None:
                        coef = np.ravel(coef)
                        vals = np.abs(coef)
                        vals = (vals / (vals.sum() + 1e-12))
                        return pd.Series(vals, index=feature_names).fillna(0.0)
        # If state is a model itself:
        if hasattr(state, "feature_importances_"):
            vals = np.array(getattr(state, "feature_importances_"))
            vals = (vals / (vals.sum() + 1e-12))
            return pd.Series(vals, index=feature_names).fillna(0.0)
        if hasattr(state, "coef_"):
            coef = getattr(state, "coef_")
            coef = np.ravel(coef)
            vals = np.abs(coef)
            vals = (vals / (vals.sum() + 1e-12))
            return pd.Series(vals, index=feature_names).fillna(0.0)
    except Exception:
        pass
    return None


def train_models(ticker: str) -> pd.DataFrame:
    """
    Analysis / pretraining for live usage.
      - Ensures (ticker)_HISTORY.csv exists & is up-to-date.
      - Builds leakage-safe META training/validation frames directly from HISTORY.
      - Fits & persists the meta model (no submodel refits).
      - Reports per-submodel metrics using HISTORY vs targets and meta metrics.

    Returns a DataFrame similar in spirit to the old report. Feature importances
    for submodels are omitted (since we don’t refit them); meta importances are
    still reported when available.
    """
    thresholds = load_thresholds()
    df_full = _load_df_for_ticker(ticker)
    if len(df_full) < 300:
        _log("Not enough rows to train analysis. Provide >= 300 bars.")
        return pd.DataFrame(columns=["model","feature","importance","metric","value"])

    # Ensure current history
    hist = _ensure_history_up_to_date(ticker, df_full, thresholds, min_coverage=float(os.getenv("HISTORY_MIN_COVERAGE", "0.95")))

    # Targets & split
    horizons_needed = [cfg["horizon"] for cfg in SUBMODELS_REGISTRY.values() if cfg.get("enabled", True)]
    targets = _common_targets(df_full, horizons_needed=horizons_needed + [1])

    val_frac = float(thresholds["val_fraction"])
    embargo = int(thresholds["embargo_bars"])
    n = len(df_full)
    val_start = max(50, int(n * (1.0 - val_frac)))
    train_end = max(20, val_start - embargo)
    df_tr = df_full.iloc[:train_end]
    df_val = df_full.iloc[val_start:]

    # ======= META: train from HISTORY on the train window, evaluate on val window
    Xr_tr, y_reg_tr, Xc_tr, y_cls_tr = _build_meta_training_frame_from_history(
        df_full=df_full,
        history_df=hist,
        thresholds=thresholds,
        start_i=None,
        end_i=train_end,
    )
    rows: List[Dict[str, Any]] = []

    from sklearn.metrics import roc_auc_score, mean_squared_error

    mm = None
    meta_auc, meta_rmse = np.nan, np.nan
    if not Xr_tr.empty and not Xc_tr.empty:
        mm = MetaModel(os.getenv("META_MODEL_TYPE", "ridge"))
        mm.fit(Xr_tr, y_reg_tr, Xc_tr, y_cls_tr)
        try:
            _save_pickle(_DEFAULT_META_STATE_PATH, mm.dump_state())
            _log(f"[meta] trained on {len(Xr_tr)} rows (history); saved state -> {_DEFAULT_META_STATE_PATH}")
        except Exception as e:
            _log(f"[meta] failed to save state: {e}")

        # Validation frame (history)
        Xr_val, y_reg_val, Xc_val, y_cls_val = _build_meta_training_frame_from_history(
            df_full=df_full,
            history_df=hist,
            thresholds=thresholds,
            start_i=val_start,
            end_i=len(df_full) - 1,
        )
        if not Xr_val.empty and not Xc_val.empty:
            yhat_bps, p_up, _ = mm.predict(Xr_val, Xc_val)
            try:
                meta_auc = roc_auc_score(y_cls_val.reindex(p_up.index).fillna(0), p_up.fillna(0.5))
            except Exception:
                meta_auc = np.nan
            try:
                meta_rmse = math.sqrt(mean_squared_error(y_reg_val.reindex(yhat_bps.index).fillna(0.0), yhat_bps.fillna(0.0)))
            except Exception:
                meta_rmse = np.nan

            rows.append({"model": "meta_cls", "feature": "OVERALL", "importance": 1.0, "metric": "AUC",  "value": float(meta_auc) if np.isfinite(meta_auc) else np.nan})
            rows.append({"model": "meta_reg", "feature": "OVERALL", "importance": 1.0, "metric": "RMSE", "value": float(meta_rmse) if np.isfinite(meta_rmse) else np.nan})

        # Meta importances (best-effort)
        if hasattr(mm, "feature_list_reg") and mm.feature_list_reg:
            imp_reg = _extract_feature_importance_from_state(mm.reg_model, mm.feature_list_reg) or pd.Series(0.0, index=mm.feature_list_reg)
            imp_reg = (imp_reg / (imp_reg.abs().sum() + 1e-12)).fillna(0.0)
            for feat, val in imp_reg.items():
                rows.append({"model": "meta_reg", "feature": feat, "importance": float(val), "metric": "RMSE" if np.isfinite(meta_rmse) else None, "value": float(meta_rmse) if np.isfinite(meta_rmse) else np.nan})

        if hasattr(mm, "feature_list_cls") and mm.feature_list_cls:
            imp_cls = _extract_feature_importance_from_state(mm.cls_model, mm.feature_list_cls) or pd.Series(0.0, index=mm.feature_list_cls)
            imp_cls = (imp_cls / (imp_cls.abs().sum() + 1e-12)).fillna(0.0)
            for feat, val in imp_cls.items():
                rows.append({"model": "meta_cls", "feature": feat, "importance": float(val), "metric": "AUC" if np.isfinite(meta_auc) else None, "value": float(meta_auc) if np.isfinite(meta_auc) else np.nan})
    else:
        _log("[meta] skipped training from history (no rows). Using prior-weighted blend at inference time.")

    # ======= Per-submodel metrics from HISTORY (no refit)
    # We compute AUC/RMSE by comparing history predictions vs true targets on validation window.
    for name, cfg in SUBMODELS_REGISTRY.items():
        if not cfg.get("enabled", True):
            continue

        pfx = f"{name}__"
        kind = cfg["kind"]
        h = int(cfg["horizon"])

        # Use the validation period
        idx_val = df_val.index
        y_true_bps = targets.get(f"fwd_h{h}_bps", pd.Series(index=idx_val)).reindex(idx_val)

        # Pull history predictions aligned to val
        y_pred_hist = pd.to_numeric(hist.get(pfx + "y_pred", pd.Series(index=df_full.index)), errors="coerce").reindex(idx_val)
        proba_up_hist = pd.to_numeric(hist.get(pfx + "proba_up", pd.Series(index=df_full.index)), errors="coerce").reindex(idx_val)

        auc_val, rmse_val = np.nan, np.nan
        try:
            if kind in ("classification", "crash"):
                if proba_up_hist.isna().all():
                    # Fallback: map y_pred to pseudo-prob via sigmoid
                    proba_up_hist = pd.Series(_sigmoid((y_pred_hist / (y_pred_hist.abs().rolling(200, min_periods=20).median() + 1e-6)).fillna(0.0)), index=idx_val)
                # Build labels for classification
                if kind == "crash":
                    vol = targets["vol_ref_bps"].reindex(idx_val)
                    y_lbl = _crash_label_fallback(y_true_bps, vol, float(thresholds["crash_k_sigma"]), float(thresholds["crash_abs_floor_bps"]))
                else:
                    tau = float(thresholds["label_tau_bps"])
                    y_lbl = _label_after_cost(y_true_bps, tau)
                from sklearn.metrics import roc_auc_score
                auc_val = roc_auc_score(y_lbl.fillna(0), proba_up_hist.fillna(0.5))
                rows.append({"model": name, "feature": "OVERALL", "importance": 1.0, "metric": "AUC", "value": float(auc_val) if np.isfinite(auc_val) else np.nan})
            else:
                # Regression/rv/unsupervised: RMSE on y_pred vs fwd return
                from sklearn.metrics import mean_squared_error
                rmse_val = math.sqrt(mean_squared_error(y_true_bps.fillna(0.0), y_pred_hist.fillna(0.0)))
                rows.append({"model": name, "feature": "OVERALL", "importance": 1.0, "metric": "RMSE", "value": float(rmse_val) if np.isfinite(rmse_val) else np.nan})
        except Exception as e:
            _log(f"[analysis] '{name}' metric calc from history failed: {e}")

    out = pd.DataFrame(rows)
    if not out.empty:
        out.sort_values(["model","feature"], inplace=True)
    return out
