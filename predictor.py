# predictor.py
from __future__ import annotations

import io
import os
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ===== Streamlit / Plotly =====
import streamlit as st
import plotly.graph_objects as go

# ===== ML / Metrics =====
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error

# ===== Hijri (optional) =====
try:
    from hijri_converter import convert as hj
    HIJRI_OK = True
except Exception:
    HIJRI_OK = False

# =============================================================================
# Config & paths (portable for Streamlit Cloud)
# =============================================================================
APP_DIR = Path(__file__).parent
DEFAULT_PATH = str(APP_DIR / "default.csv")      # bundled demo data (safe/public)
STORE_PATH = APP_DIR / ".data" / "stored_weekly.csv"  # local fallback store
STORE_PATH.parent.mkdir(exist_ok=True)

OCCS = ["New Year", "Valentine", "Mother’s Day", "Ramadan", "Eid al-Fitr", "Eid al-Adha"]
LAGS = [1, 2, 3, 4, 8, 12, 52]
ROLLS = [4, 8, 12]
FOURIER_K = 3
NEAR_WINDOW_WEEKS = 1
AUTO_CALIBRATE = True
MANUAL_UPLIFT = 1.05
OCCASION_BOOST = 0.10
CLIP_MIN = 0.0
MAX_WEEKS = 130

# =============================================================================
# Secrets helper (works locally & on Streamlit Cloud)
# =============================================================================
def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return st.secrets[key]  # Streamlit Cloud
    except Exception:
        return os.getenv(key, default)  # local dev / env

# Toggle S3 persistence with secrets
USE_S3 = bool(get_secret("STORE_S3_BUCKET"))
S3_BUCKET = get_secret("STORE_S3_BUCKET")
S3_KEY    = get_secret("STORE_S3_KEY", "stored_weekly.csv")
AWS_REGION = get_secret("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = get_secret("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = get_secret("AWS_SECRET_ACCESS_KEY")

# =============================================================================
# Flexible loader
# =============================================================================
def load_dataset(file_or_path: Any, prefer_utf16_tsv: bool = True, sheet_name: Optional[Any] = 0
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if file_or_path is None:
        raise ValueError("load_dataset: No file provided (got None).")

    name = getattr(file_or_path, "name", None)
    ext = (Path(name).suffix.lower() if name
           else (Path(str(file_or_path)).suffix.lower() if isinstance(file_or_path, str) else "")).strip()
    info: Dict[str, Any] = {
        "source_name": name or str(file_or_path),
        "ext": ext or "unknown",
        "engine": None, "encoding": None, "sep": None, "reader": None,
    }

    def fresh_buf():
        if hasattr(file_or_path, "read"):
            raw = file_or_path.getvalue() if hasattr(file_or_path, "getvalue") else file_or_path.read()
            return io.BytesIO(raw)
        return None

    if ext == ".parquet":
        df = pd.read_parquet(file_or_path)
        info.update(reader="read_parquet", engine="pyarrow/fastparquet")
        return df, info

    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(file_or_path, sheet_name=sheet_name)
        info.update(reader="read_excel", sheet_name=sheet_name)
        return df, info

    if ext in {".json", ".jsonl"}:
        lines = (ext == ".jsonl")
        df = pd.read_json(file_or_path, lines=lines)
        info.update(reader="read_json", lines=lines)
        return df, info

    if prefer_utf16_tsv:
        try:
            df = pd.read_csv(fresh_buf() or file_or_path, sep="\t", encoding="utf-16", engine="python")
            info.update(reader="read_csv", sep="\\t", encoding="utf-16", engine="python", tried="preferred utf-16 tsv")
            return df, info
        except Exception:
            pass

    encodings = ["utf-8-sig", "utf-8", "utf-16", "utf-16le", "utf-16be", "cp1252"]
    for enc in encodings:
        try:
            df = pd.read_csv(fresh_buf() or file_or_path, sep=None, engine="python", encoding=enc)
            info.update(reader="read_csv", sep="auto", encoding=enc, engine="python")
            return df, info
        except Exception:
            continue

    delims = ["\t", ",", ";", "|"]
    last_err = None
    for enc in encodings:
        for sep in delims:
            try:
                df = pd.read_csv(fresh_buf() or file_or_path, sep=sep, encoding=enc, engine="python")
                info.update(reader="read_csv", sep=repr(sep), encoding=enc, engine="python")
                return df, info
            except Exception as e:
                last_err = e

    raise RuntimeError(f"Could not load file {name or file_or_path!r} with common strategies.") from last_err

# =============================================================================
# Helpers: calendar, features, auto-detect
# =============================================================================
def week_start_aligned(d: pd.Timestamp, anchor_weekday: int) -> pd.Timestamp:
    offset = (d.weekday() - anchor_weekday) % 7
    return (d - pd.Timedelta(days=offset)).normalize()

def detect_occasions_for_week(week_start: pd.Timestamp, anchor_weekday: int) -> set:
    occ = set()
    for i in range(7):
        date = week_start + pd.Timedelta(days=i)
        # Gregorian
        if date.month == 1 and date.day == 1:  occ.add("New Year")
        if date.month == 2 and date.day == 14: occ.add("Valentine")
        if date.month == 3 and date.day == 21: occ.add("Mother’s Day")
        # Hijri
        if HIJRI_OK:
            h = hj.Gregorian(date.year, date.month, date.day).to_hijri()
            hm, hd = h.month, h.day
            if hm == 9:                        occ.add("Ramadan")
            if hm == 10 and 1 <= hd <= 3:      occ.add("Eid al-Fitr")
            if hm == 12 and hd == 10:          occ.add("Eid al-Adha")
    return occ

def add_lags_and_rolls(dfw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=dfw.index)
    out["orders"] = dfw["Delivered Orders"].astype(float)
    out["revenue"] = dfw["Delivered Revenue"].astype(float)
    for lag in LAGS:
        out[f"orders_lag_{lag}"] = out["orders"].shift(lag)
        out[f"revenue_lag_{lag}"] = out["revenue"].shift(lag)
    for r in ROLLS:
        out[f"orders_rollmean_{r}"] = out["orders"].shift(1).rolling(r).mean()
        out[f"revenue_rollmean_{r}"] = out["revenue"].shift(1).rolling(r).mean()
    return out

def _to_float_no_commas(series) -> pd.Series:
    return pd.to_numeric(pd.Series(series).astype(str).str.replace(",", "", regex=False), errors="coerce")

def autodetect_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    cols_lower = {c.lower(): c for c in df.columns}
    # Date
    date_keys = ["date", "order date", "transaction date", "month, day, year of date"]
    date_col = None
    for k in date_keys:
        if k in cols_lower: date_col = cols_lower[k]; break
    if date_col is None:
        for c in df.columns:
            try:
                if pd.to_datetime(df[c], errors="coerce").notna().mean() > 0.7:
                    date_col = c; break
            except Exception:
                continue
    # Orders
    order_candidates = ["delivered orders", "orders", "qty", "quantity", "units", "order_count"]
    orders_col = next((cols_lower[k] for k in order_candidates if k in cols_lower), None)
    # Revenue/Sales
    revenue_candidates = ["delivered revenue", "revenue", "sales", "gmv", "amount", "turnover"]
    revenue_col = next((cols_lower[k] for k in revenue_candidates if k in cols_lower), None)

    if not date_col or not orders_col or not revenue_col:
        raise ValueError("Auto-detection failed: need a Date, Orders, and Revenue column.")
    return date_col, orders_col, revenue_col

# ---------- Historical occasions from rules + optional data merge ----------
def infer_occasion_names_from_rules(index: pd.DatetimeIndex, anchor_weekday: int) -> pd.Series:
    names = []
    for d in index:
        ws = week_start_aligned(pd.Timestamp(d), anchor_weekday)
        occs = detect_occasions_for_week(ws, anchor_weekday)
        names.append(", ".join(sorted(occs)) if occs else "None")
    return pd.Series(names, index=index, name="occasion_name")

def weekly_occ_names_from_raw(df_raw: pd.DataFrame, date_col: str) -> Optional[pd.Series]:
    if "occasion" not in df_raw.columns:
        return None
    tmp = df_raw[["occasion"]].copy()
    tmp["Date"] = pd.to_datetime(df_raw[date_col], errors="coerce")
    tmp = tmp.dropna(subset=["Date"])
    tmp["week_start"] = tmp["Date"] - pd.to_timedelta((tmp["Date"].dt.dayofweek + 1) % 7, unit="D")
    weekly_names = (
        tmp.groupby(tmp["week_start"].dt.normalize())["occasion"]
           .apply(lambda s: ", ".join(sorted({str(x).strip() for x in s if str(x).strip() and str(x).strip() != "None"})) or "None")
    )
    weekly_names.name = "occasion_name"
    return weekly_names

def _combine_name_series(pref_rules: pd.Series, from_data: Optional[pd.Series]) -> pd.Series:
    if from_data is None:
        return pref_rules
    idx = pref_rules.index.union(from_data.index)
    def splitset(v):
        if pd.isna(v): return set()
        return {t.strip() for t in str(v).split(",") if t.strip() and t.strip() != "None"}
    out = []
    for d in idx:
        a = splitset(pref_rules.get(d, "None"))
        b = splitset(from_data.get(d, "None"))
        s = ", ".join(sorted(a | b)) if (a | b) else "None"
        out.append(s)
    return pd.Series(out, index=idx, name="occasion_name").reindex(pref_rules.index).fillna("None")

def build_flags_from_names(index: pd.DatetimeIndex, names: pd.Series) -> pd.DataFrame:
    flags = pd.DataFrame(index=index)
    for k in OCCS:
        flags[f"is_{k}"] = names.str.contains(rf"\b{k}\b", regex=True).astype(int)
    flags["is_occasion"] = (flags.filter(like="is_").sum(axis=1) > 0).astype(int)

    near_any = flags["is_occasion"].copy()
    for s in range(1, NEAR_WINDOW_WEEKS + 1):
        near_any = (
            near_any
            | flags["is_occasion"].shift(s).fillna(0).astype(int)
            | flags["is_occasion"].shift(-s).fillna(0).astype(int)
        )
    flags["near_occasion"] = near_any.astype(int)

    flags["month"] = index.month.astype(int)
    flags["weekofyear"] = index.isocalendar().week.astype(int)
    flags["year"] = index.year.astype(int)

    t = np.arange(len(index))
    flags["t"] = t
    period = 52
    for k in range(1, FOURIER_K + 1):
        flags[f"sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        flags[f"cos_{k}"] = np.cos(2 * np.pi * k * t / period)

    return flags

# ---------- Future calendar (rule-based) ----------
def build_calendar_future(index: pd.DatetimeIndex, anchor_weekday: int, t0: int):
    flags = pd.DataFrame(index=index)
    names = []
    for k in OCCS:
        flags[f"is_{k}"] = 0

    for d in index:
        ws = week_start_aligned(pd.Timestamp(d), anchor_weekday)
        occs = detect_occasions_for_week(ws, anchor_weekday)
        for k in occs:
            if k in OCCS:
                flags.at[d, f"is_{k}"] = 1
        names.append(", ".join(sorted(occs)) if occs else "None")

    flags["is_occasion"] = (flags.filter(like="is_").sum(axis=1) > 0).astype(int)

    near_any = flags["is_occasion"].copy()
    for s in range(1, NEAR_WINDOW_WEEKS + 1):
        near_any = (
            near_any
            | flags["is_occasion"].shift(s).fillna(0).astype(int)
            | flags["is_occasion"].shift(-s).fillna(0).astype(int)
        )
    flags["near_occasion"] = near_any.astype(int)

    flags["month"] = index.month.astype(int)
    flags["weekofyear"] = index.isocalendar().week.astype(int)
    flags["year"] = index.year.astype(int)

    t = np.arange(len(index)) + int(t0)
    flags["t"] = t

    period = 52
    for k in range(1, FOURIER_K + 1):
        flags[f"sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        flags[f"cos_{k}"] = np.cos(2 * np.pi * k * t / period)

    names = pd.Series(names, index=index, name="occasion_name")
    return flags, names

# =============================================================================
# Persistence: local CSV (fallback) or S3 (private)
# =============================================================================
def save_weekly_store_local(weekly: pd.DataFrame):
    df = weekly.reset_index().rename(columns={"index": "week_start"})
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df[["week_start", "Delivered Orders", "Delivered Revenue"]]
    df.to_csv(STORE_PATH, index=False)

def load_weekly_store_local(default_weekly: pd.DataFrame) -> pd.DataFrame:
    if STORE_PATH.exists():
        df = pd.read_csv(STORE_PATH)
        df["week_start"] = pd.to_datetime(df["week_start"])
        return df.set_index("week_start").sort_index()
    save_weekly_store_local(default_weekly)
    return default_weekly.copy()

def save_weekly_store_s3(weekly: pd.DataFrame):
    import boto3
    if not S3_BUCKET:
        raise RuntimeError("STORE_S3_BUCKET not set in secrets.")
    buf = io.BytesIO()
    weekly.reset_index().rename(columns={"index":"week_start"})[
        ["week_start", "Delivered Orders", "Delivered Revenue"]
    ].to_csv(buf, index=False)
    buf.seek(0)
    client = boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    client.upload_fileobj(buf, S3_BUCKET, S3_KEY)

def load_weekly_store_s3(fallback: pd.DataFrame) -> pd.DataFrame:
    import boto3
    client = boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    try:
        buf = io.BytesIO()
        client.download_fileobj(S3_BUCKET, S3_KEY, buf)
        buf.seek(0)
        df = pd.read_csv(buf)
        df["week_start"] = pd.to_datetime(df["week_start"])
        return df.set_index("week_start").sort_index()
    except Exception:
        # Seed on first run
        save_weekly_store_s3(fallback)
        return fallback.copy()

# =============================================================================
# Streamlit UI
# =============================================================================
st.set_page_config(page_title="Weekly Sales Predictor", layout="wide")
st.title("📈 Weekly Sales Predictor")

# Upload (top) + mode + horizon
up = st.file_uploader("Upload your dataset (CSV / Excel / Parquet / JSON)",
                      type=["csv", "xlsx", "xls", "parquet", "json", "jsonl"])

mode = st.radio(
    "Upload mode",
    ["Train on new file", "Update existing history"],
    help="• Train on new file: replace working history with this file.\n• Update existing history: append/overwrite new weeks into stored history."
)

horizon = st.number_input("Prediction limit (weeks)", min_value=1, max_value=156, value=12, step=1)

# --- Load DEFAULT to seed store and for fallback ---
try:
    df_default_raw, _ = load_dataset(DEFAULT_PATH)
    date_col_def, ord_col_def, rev_col_def = autodetect_columns(df_default_raw)
    df_def = df_default_raw.copy()
    df_def["Date"] = pd.to_datetime(df_def[date_col_def], errors="coerce")
    df_def = df_def.dropna(subset=["Date"]).sort_values("Date")
    df_def[ord_col_def] = _to_float_no_commas(df_def[ord_col_def])
    df_def[rev_col_def] = _to_float_no_commas(df_def[rev_col_def])
    df_def["week_start"] = df_def["Date"] - pd.to_timedelta((df_def["Date"].dt.dayofweek + 1) % 7, unit="D")
    weekly_default = (
        df_def.groupby(df_def["week_start"].dt.normalize())
             .agg({ord_col_def: "sum", rev_col_def: "sum"})
             .rename(columns={ord_col_def: "Delivered Orders", rev_col_def: "Delivered Revenue"})
             .sort_index()
    )
except Exception as e:
    st.error(f"Failed to read default.csv: {e}")
    st.stop()

# Load store: S3 if configured; else local
try:
    if USE_S3:
        weekly_store = load_weekly_store_s3(weekly_default)
    else:
        weekly_store = load_weekly_store_local(weekly_default)
except Exception as e:
    st.error(f"Failed to load store: {e}")
    st.stop()

# --- If user uploaded a file, preprocess it to weekly ---
weekly_new = None
df_raw = None
date_col = ord_col = rev_col = None
if up:
    try:
        df_raw, meta = load_dataset(up)
        date_col, ord_col, rev_col = autodetect_columns(df_raw)
        st.success(f"Loaded: **{meta['source_name']}** — shape {df_raw.shape[0]}×{df_raw.shape[1]}")
        df_u = df_raw.copy()
        df_u["Date"] = pd.to_datetime(df_u[date_col], errors="coerce")
        df_u = df_u.dropna(subset=["Date"]).sort_values("Date")
        df_u[ord_col] = _to_float_no_commas(df_u[ord_col])
        df_u[rev_col] = _to_float_no_commas(df_u[rev_col])
        df_u["week_start"] = df_u["Date"] - pd.to_timedelta((df_u["Date"].dt.dayofweek + 1) % 7, unit="D")
        weekly_new = (
            df_u.groupby(df_u["week_start"].dt.normalize())
               .agg({ord_col: "sum", rev_col: "sum"})
               .rename(columns={ord_col: "Delivered Orders", rev_col: "Delivered Revenue"})
               .sort_index()
        )
    except Exception as e:
        st.error(f"Failed to process uploaded file: {e}")
        st.stop()

# --- Decide training weekly series based on mode ---
if up and mode == "Train on new file":
    weekly_train = weekly_new.copy()
    try:
        if USE_S3:
            save_weekly_store_s3(weekly_train)
        else:
            save_weekly_store_local(weekly_train)
    except Exception as e:
        st.warning(f"Could not persist new store: {e}")
    st.info("Training on the uploaded file (replaced stored history).")
elif up and mode == "Update existing history":
    merged = weekly_store.copy()
    if weekly_new is not None and not weekly_new.empty:
        merged = merged.combine_first(weekly_new)
        merged.update(weekly_new)
        merged = merged.sort_index()
        try:
            if USE_S3:
                save_weekly_store_s3(merged)
            else:
                save_weekly_store_local(merged)
        except Exception as e:
            st.warning(f"Could not persist updated store: {e}")
        st.info(f"Stored history updated: {len(weekly_new)} weekly rows merged.")
    weekly_train = merged
else:
    weekly_train = weekly_store.copy()
    st.warning(f"No file uploaded. Using stored history ({len(weekly_train)} rows).")

# === Align weekly frequency, fill, clip ===
if weekly_train.empty:
    st.error("No rows to train after preprocessing.")
    st.stop()

anchor_weekday = int(pd.Series(weekly_train.index.weekday).mode()[0]) if len(weekly_train) else 6
freq_map = {0: "W-MON", 1: "W-TUE", 2: "W-WED", 3: "W-THU", 4: "W-FRI", 5: "W-SAT", 6: "W-SUN"}
WEEK_FREQ = freq_map.get(anchor_weekday, "W-SUN")

weekly = weekly_train.asfreq(WEEK_FREQ)
for c in ["Delivered Orders", "Delivered Revenue"]:
    if weekly[c].isna().any():
        weekly[c] = weekly[c].interpolate(method="time")
weekly[["Delivered Orders", "Delivered Revenue"]] = weekly[["Delivered Orders", "Delivered Revenue"]].clip(lower=0)

# =============================================================================
# Historical occasions (rules + optional 'occasion' from raw data)
# =============================================================================
occ_names_hist_rules = infer_occasion_names_from_rules(weekly.index, anchor_weekday)
occ_from_default = weekly_occ_names_from_raw(df_default_raw, date_col_def)
occ_names_hist = _combine_name_series(occ_names_hist_rules, occ_from_default)

if df_raw is not None and date_col is not None:
    occ_from_uploaded = weekly_occ_names_from_raw(df_raw, date_col)
    occ_names_hist = _combine_name_series(occ_names_hist, occ_from_uploaded)

cal_flags_hist = build_flags_from_names(weekly.index, occ_names_hist)

# =============================================================================
# Features & model
# =============================================================================
lags_rolls = add_lags_and_rolls(weekly)
supervised = pd.concat([lags_rolls, cal_flags_hist], axis=1)
supervised["occasion_name"] = occ_names_hist
supervised = supervised.dropna().copy()

if supervised.empty:
    st.error("Not enough data after feature generation.")
    st.stop()

TARGETS = ["orders", "revenue"]
feature_cols = [c for c in supervised.columns if c not in TARGETS + ["occasion_name"]]
X_all = supervised[feature_cols]
y_all = supervised[TARGETS]

rf = RandomForestRegressor(
    n_estimators=900, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1
)

# OOF calibration
oof_pred = np.zeros_like(y_all.values)
tscv = TimeSeriesSplit(n_splits=4)
m_os, m_rev = [], []
for (tr, te) in tscv.split(X_all):
    rf.fit(X_all.iloc[tr], y_all.iloc[tr])
    pred = rf.predict(X_all.iloc[te])
    oof_pred[te, :] = pred
    m_os.append(mean_absolute_percentage_error(y_all.iloc[te, 0], pred[:, 0]))
    m_rev.append(mean_absolute_percentage_error(y_all.iloc[te, 1], pred[:, 1]))

cal_orders  = float(np.clip(np.median(np.where(oof_pred[:, 0] > 0, y_all.values[:, 0] / oof_pred[:, 0], 1.0)), 0.8, 1.3))
cal_revenue = float(np.clip(np.median(np.where(oof_pred[:, 1] > 0, y_all.values[:, 1] / oof_pred[:, 1], 1.0)), 0.8, 1.3))

# Fit final model
rf.fit(X_all, y_all)

# =============================================================================
# Forecast (rule-based future calendar)
# =============================================================================
last_date = weekly.index[-1]
future_index = pd.date_range(
    start=last_date + pd.offsets.Week(weekday=anchor_weekday),
    periods=MAX_WEEKS, freq=WEEK_FREQ
)
last_t = int(supervised["t"].iloc[-1])
cal_flags_future, occ_names_future = build_calendar_future(future_index, anchor_weekday, t0=last_t + 1)

H = int(min(horizon, MAX_WEEKS))
future_is_occ = cal_flags_future["is_occasion"].values

# Recursive forecast
hist_orders = weekly["Delivered Orders"].astype(float).copy()
hist_revenue = weekly["Delivered Revenue"].astype(float).copy()
pred_orders, pred_revenue = [], []

for i in range(H):
    d = future_index[i]
    row = {"orders": np.nan, "revenue": np.nan}
    for lag in LAGS:
        row[f"orders_lag_{lag}"]  = hist_orders.iloc[-lag]  if len(hist_orders)  >= lag else hist_orders.mean()
        row[f"revenue_lag_{lag}"] = hist_revenue.iloc[-lag] if len(hist_revenue) >= lag else hist_revenue.mean()
    for rwin in ROLLS:
        row[f"orders_rollmean_{rwin}"]  = hist_orders.iloc[-(rwin+1):-1].mean()  if len(hist_orders)  >= rwin+1 else hist_orders.mean()
        row[f"revenue_rollmean_{rwin}"] = hist_revenue.iloc[-(rwin+1):-1].mean() if len(hist_revenue) >= rwin+1 else hist_revenue.mean()
    for c in cal_flags_future.columns:
        row[c] = cal_flags_future.at[d, c]
    x = pd.DataFrame([row], columns=feature_cols, index=[d])
    yhat = rf.predict(x)[0]
    o_pred, r_pred = float(yhat[0]), float(yhat[1])
    pred_orders.append(o_pred); pred_revenue.append(r_pred)
    hist_orders  = pd.concat([hist_orders,  pd.Series([o_pred], index=[d])])
    hist_revenue = pd.concat([hist_revenue, pd.Series([r_pred], index=[d])])

# Calibrate + boosts
pred_orders  = np.maximum(np.array(pred_orders)  * (cal_orders  if AUTO_CALIBRATE else 1.0) * MANUAL_UPLIFT, CLIP_MIN)
pred_revenue = np.maximum(np.array(pred_revenue) * (cal_revenue if AUTO_CALIBRATE else 1.0) * MANUAL_UPLIFT, CLIP_MIN)
pred_orders  = pred_orders  * (1.0 + OCCASION_BOOST * future_is_occ[:H])
pred_revenue = pred_revenue * (1.0 + OCCASION_BOOST * future_is_occ[:H])

out = pd.DataFrame({
    "date": future_index[:H],
    "occasion": occ_names_future.iloc[:H].values,
    "delivered_revenue": pred_revenue,
    "delivered_orders":  pred_orders,
})
out["AOV"] = out["delivered_revenue"] / np.where(out["delivered_orders"] > 0, out["delivered_orders"], np.nan)

# =============================================================================
# Single-plot visualization (Orders, Revenue, AOV) + markers (like screenshot)
# =============================================================================
hist = weekly.copy()
hist["AOV"] = hist["Delivered Revenue"] / np.where(hist["Delivered Orders"] > 0, hist["Delivered Orders"], np.nan)
fut = out.set_index("date").sort_index()

hist_occ_dates = hist.index[cal_flags_hist["is_occasion"].astype(bool)]
hist_occ_texts = supervised["occasion_name"].reindex(hist_occ_dates).fillna("Occasion").astype(str)
fut_occ_dates  = fut.index[cal_flags_future.loc[fut.index, "is_occasion"].astype(bool)]
fut_occ_texts  = fut.loc[fut_occ_dates, "occasion"].fillna("Occasion").astype(str)

fig = go.Figure()

# Orders
fig.add_trace(go.Scatter(
    x=hist.index, y=hist["Delivered Orders"], mode="lines",
    name="Historical Orders", yaxis="y",
    customdata=supervised["occasion_name"].reindex(hist.index).fillna("None"),
    hovertemplate="Week=%{x|%Y-%m-%d}<br>Orders=%{y:.0f}<br>Occasion=%{customdata}<extra></extra>"
))
fig.add_trace(go.Scatter(
    x=fut.index, y=fut["delivered_orders"], mode="lines+markers",
    name="Forecast Orders", yaxis="y",
    customdata=fut["occasion"].astype(str),
    hovertemplate="Week=%{x|%Y-%m-%d}<br>Orders=%{y:.0f}<br>Occasion=%{customdata}<extra></extra>"
))

# Revenue
fig.add_trace(go.Scatter(
    x=hist.index, y=hist["Delivered Revenue"], mode="lines",
    name="Historical Revenue", yaxis="y2",
    customdata=supervised["occasion_name"].reindex(hist.index).fillna("None"),
    hovertemplate="Week=%{x|%Y-%m-%d}<br>Revenue=%{y:.2f}<br>Occasion=%{customdata}<extra></extra>"
))
fig.add_trace(go.Scatter(
    x=fut.index, y=fut["delivered_revenue"], mode="lines+markers",
    name="Forecast Revenue", yaxis="y2",
    customdata=fut["occasion"].astype(str),
    hovertemplate="Week=%{x|%Y-%m-%d}<br>Revenue=%{y:.2f}<br>Occasion=%{customdata}<extra></extra>"
))

# AOV
fig.add_trace(go.Scatter(
    x=hist.index, y=hist["AOV"], mode="lines",
    name="Historical AOV", yaxis="y3",
    hovertemplate="Week=%{x|%Y-%m-%d}<br>AOV=%{y:.2f}<extra></extra>"
))
fig.add_trace(go.Scatter(
    x=fut.index, y=fut["AOV"], mode="lines+markers",
    name="Forecast AOV", yaxis="y3",
    hovertemplate="Week=%{x|%Y-%m-%d}<br>AOV=%{y:.2f}<extra></extra>"
))

# Occasion markers lane
fig.add_trace(go.Scatter(
    x=hist_occ_dates, y=[0.05]*len(hist_occ_dates), yaxis="y4",
    mode="markers+text", name="Occasion (hist)",
    text=hist_occ_texts, textposition="top center",
    marker=dict(symbol="diamond", size=9, opacity=0.85),
    hovertemplate="Week=%{x|%Y-%m-%d}<br>%{text}<extra></extra>"
))
fig.add_trace(go.Scatter(
    x=fut_occ_dates, y=[0.05]*len(fut_occ_dates), yaxis="y4",
    mode="markers+text", name="Occasion (future)",
    text=fut_occ_texts, textposition="top center",
    marker=dict(symbol="triangle-up", size=10, opacity=0.9),
    hovertemplate="Week=%{x|%Y-%m-%d}<br>%{text}<extra></extra>"
))

# Forecast start marker
last_hist = hist.index.max()
fig.add_shape(type="line", x0=last_hist, x1=last_hist, y0=0, y1=1,
              xref="x", yref="paper", line=dict(dash="dot"))
fig.add_annotation(x=last_hist, y=1, xref="x", yref="paper",
                   text="Forecast start", showarrow=False, yshift=10)

# Layout (no 'week' step; use month/year)
fig.update_layout(
    title="Single-Plot — Orders, Revenue, AOV (+ Occasion markers, no shading)",
    xaxis=dict(
        title="Week start",
        rangeslider=dict(visible=True),
        rangeselector=dict(
            buttons=[
                dict(count=3,  label="3M",  step="month", stepmode="backward"),
                dict(count=6,  label="6M",  step="month", stepmode="backward"),
                dict(count=12, label="1Y",  step="month", stepmode="backward"),
                dict(label="YTD", step="year", stepmode="todate"),
                dict(step="all")
            ]
        )
    ),
    yaxis=dict(title="Orders", side="left"),
    yaxis2=dict(title="Revenue", overlaying="y", side="right", showgrid=False),
    yaxis3=dict(title="AOV", overlaying="y", side="right",
                anchor="free", position=0.93, showgrid=False, zeroline=False),
    yaxis4=dict(overlaying="y", side="left", range=[0,1],
                showticklabels=False, showgrid=False, zeroline=False),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    autosize=False, width=1700, height=900,
    margin=dict(l=80, r=120, t=70, b=60),
    font=dict(size=15)
)
fig.update_traces(selector=dict(mode="lines"), line=dict(width=3))
fig.update_traces(selector=dict(mode="lines+markers"), line=dict(width=3), marker=dict(size=8))

st.plotly_chart(fig, use_container_width=True)

# Preview & Download
with st.expander("🔎 Last 10 rows (historical)"):
    hist_tail = hist.tail(10)
    hist_tail_show = pd.DataFrame({
        "date": hist_tail.index.strftime("%Y-%m-%d"),
        "occasion": supervised["occasion_name"].reindex(hist_tail.index).astype(str).values,
        "delivered_revenue": hist["Delivered Revenue"].tail(10).values,
        "delivered_orders":  hist["Delivered Orders"].tail(10).values,
    })
    hist_tail_show["AOV"] = hist_tail_show["delivered_revenue"] / np.where(
        hist_tail_show["delivered_orders"] > 0, hist_tail_show["delivered_orders"], np.nan
    )
    st.dataframe(hist_tail_show)

with st.expander("📄 Full Forecast"):
    out_show = out.copy()
    out_show["date"] = out_show["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(out_show, use_container_width=True)

@st.cache_data
def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇️ Download forecast CSV",
    data=_csv_bytes(out.assign(date=out["date"].dt.strftime("%Y-%m-%d"))),
    file_name="weekly_forecast.csv",
    mime="text/csv",
)

# Training summary
st.caption(
    f"Calibration (OOF MAPE): Orders ~ {np.mean(m_os)*100:,.2f}% | Revenue ~ {np.mean(m_rev)*100:,.2f}%  ·  "
    f"Cal Factors → Orders: {cal_orders:.3f} · Revenue: {cal_revenue:.3f}  ·  "
    f"Fixed uplift={MANUAL_UPLIFT:.2f}, occasion boost={OCCASION_BOOST*100:.0f}%"
)
