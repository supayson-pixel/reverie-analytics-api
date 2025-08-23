import io
import re
import uuid
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------
# Service setup
# -----------------------------
app = FastAPI(title="Reverie Analytics API")

# Allow your frontends to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.reveriesun.com",
        "https://reveriesun.com",
        "https://reveriesun.netlify.app",
        "https://inspiring-tarsier-97b2c7.netlify.app",
        "http://localhost:5173",  # local dev
        "http://localhost:8080",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (stateless hosting keeps this lightweight)
_STORAGE: Dict[str, pd.DataFrame] = {}

# Regex helpers for numeric cleaning
_CURRENCY_CHARS = re.compile(r"[€£$]")
_GROUPING_CHARS = re.compile(r"[,_ ]")

# -----------------------------
# Small utilities
# -----------------------------
def sniff_delimiter(text: str) -> str:
    if "\t" in text:
        return "\t"
    # choose the most frequent of ; or ,
    comma = text.count(",")
    semi = text.count(";")
    return ";" if semi > comma else ","

def load_table(upload: UploadFile) -> pd.DataFrame:
    name = (upload.filename or "").lower()
    raw = upload.file.read()

    if name.endswith((".xlsx", ".xls")):
        # Excel
        return pd.read_excel(io.BytesIO(raw))
    # CSV / TXT
    text = raw.decode("utf-8", errors="ignore")
    delim = sniff_delimiter(text)
    # for odd delimiters, prefer python engine
    engine = "python" if len(delim) > 1 else "c"
    return pd.read_csv(io.StringIO(text), sep=delim, engine=engine)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = (
        pd.Series(df.columns)
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w]+", "_", regex=True)
    )
    # ensure uniqueness
    seen = {}
    out = []
    for c in cols:
        if c not in seen:
            seen[c] = 1
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}_{seen[c]}")
    df.columns = out
    return df

def to_datetime_guess(s: pd.Series) -> pd.Series:
    # robust parse, no deprecated infer flag
    parsed = pd.to_datetime(s, errors="coerce", utc=False)
    return parsed

def maybe_numeric_series(s: pd.Series) -> pd.Series:
    """Try converting an object-like column to numeric in a safe, vectorized way.
    Only adopt conversion if it succeeds for a healthy fraction of rows."""
    if pd.api.types.is_numeric_dtype(s):
        return s

    if not pd.api.types.is_object_dtype(s) and not pd.api.types.is_string_dtype(s):
        return s

    st = s.astype(str).str.strip()
    st = st.replace(r"^\s*$", pd.NA, regex=True)        # empty/whitespace -> NaN
    st = st.str.replace(_CURRENCY_CHARS, "", regex=True)
    st = st.str.replace(_GROUPING_CHARS, "", regex=True)
    st = st.str.replace(r"^\(([^)]*)\)$", r"-\1", regex=True)  # (123) -> -123

    nums = pd.to_numeric(st, errors="coerce")
    ok_ratio = nums.notna().mean()
    # adopt conversion if >60% of non-null rows convert cleanly OR there were few non-numeric strings
    if ok_ratio >= 0.60:
        return nums
    return s

def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        df[c] = maybe_numeric_series(df[c])
    return df

def detect_types(df: pd.DataFrame) -> Dict[str, str]:
    types = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            types[c] = "date"
        elif pd.api.types.is_numeric_dtype(s):
            types[c] = "numeric"
        else:
            # treat as date if converting yields many valid values
            candidate = to_datetime_guess(s)
            if candidate.notna().mean() >= 0.70:
                df[c] = candidate
                types[c] = "date"
                continue
            # categorical heuristic: small unique set relative to rows
            nunq = s.dropna().nunique()
            if nunq <= max(20, int(len(s) * 0.2)):
                types[c] = "category"
            else:
                types[c] = "text"
    return types

def round_num(x: Any, ndigits: int = 2):
    if pd.isna(x):
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return round(float(x), ndigits)
    return x

def null_counts(df: pd.DataFrame) -> Dict[str, int]:
    out = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            out[c] = int(s.astype(str).str.strip().replace(r"^\s*$", pd.NA, regex=True).isna().sum())
        else:
            out[c] = int(s.isna().sum())
    return out

def numeric_summary(df: pd.DataFrame, schema: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    numeric_cols = [c for c, t in schema.items() if t == "numeric"]
    for c in numeric_cols:
        desc = df[c].describe(percentiles=[0.25, 0.5, 0.75])
        result[c] = {
            "count": round_num(desc.get("count"), 0),
            "mean": round_num(desc.get("mean")),
            "std": round_num(desc.get("std")),
            "min": round_num(desc.get("min")),
            "25%": round_num(desc.get("25%")),
            "50%": round_num(desc.get("50%")),
            "75%": round_num(desc.get("75%")),
            "max": round_num(desc.get("max")),
        }
    return result

def date_summaries(df: pd.DataFrame, schema: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for c, t in schema.items():
        if t != "date":
            continue
        s = pd.to_datetime(df[c], errors="coerce")
        if s.notna().sum() == 0:
            continue
        s_valid = s.dropna()
        by_month = (
            s_valid.dt.to_period("M").astype(str).value_counts().sort_index().to_dict()
        )
        by_weekday = (
            s_valid.dt.day_name().value_counts().reindex(
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                fill_value=0,
            ).to_dict()
        )
        out[c] = {
            "min": s_valid.min().strftime("%Y-%m-%d"),
            "max": s_valid.max().strftime("%Y-%m-%d"),
            "span_days": int((s_valid.max() - s_valid.min()).days),
            "by_month": by_month,
            "by_weekday": by_weekday,
        }
    return out

def top_categories(df: pd.DataFrame, schema: Dict[str, str]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for c, t in schema.items():
        if t == "category":
            vc = df[c].astype(str).replace("nan", np.nan).dropna().value_counts().head(5)
            out[c] = {k: int(v) for k, v in vc.items()}
    return out

def preview_payload(df: pd.DataFrame, max_rows: int = 10) -> Dict[str, Any]:
    rows = df.head(max_rows).copy()
    # convert datetimes to ISO strings for JSON
    for c in rows.columns:
        if pd.api.types.is_datetime64_any_dtype(rows[c]):
            rows[c] = rows[c].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
    return {
        "columns": list(map(str, rows.columns)),
        "rows": rows.replace({np.nan: None}).to_dict(orient="records"),
        "total_rows": int(len(df)),
        "total_cols": int(df.shape[1]),
    }

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Reverie Analytics API. Try /health or /analytics/*"}

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "ts": datetime.utcnow().isoformat() + "Z"}

@app.post("/analytics/upload")
async def upload(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        df = load_table(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {e}")

    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No data found in file.")

    df = normalize_columns(df)
    # Try to parse obvious dates first (vectorized)
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            # Quickly tag columns that look like ISO date-like strings
            hint = df[c].astype(str).str.contains(r"\d{4}-\d{1,2}-\d{1,2}", regex=True, na=False).mean()
            if hint >= 0.6:
                parsed = to_datetime_guess(df[c])
                if parsed.notna().mean() >= 0.6:
                    df[c] = parsed

    # Numeric coercion
    df = coerce_numeric_columns(df)

    dsid = str(uuid.uuid4())
    _ STORAGE_LIMIT = 30  # simple cap to avoid memory creep
    _STORAGE[dsid] = df
    if len(_STORAGE) > _ STORAGE_LIMIT:
        # drop the oldest key
        oldest_key = next(iter(_STORAGE))
        _STORAGE.pop(oldest_key, None)

    return {"dataset_id": dsid, "columns": list(df.columns), "rows": int(len(df))}

@app.get("/analytics/profile")
def profile(dataset_id: str) -> Dict[str, Any]:
    if dataset_id not in _STORAGE:
        raise HTTPException(status_code=404, detail="dataset_id not found")

    df = _STORAGE[dataset_id].copy()

    # Re-detect/normalize (cheap insurance if upload route changes later)
    df = normalize_columns(df)
    df = coerce_numeric_columns(df)
    schema = detect_types(df)

    payload: Dict[str, Any] = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": list(df.columns),
        "schema": schema,
        "nulls": null_counts(df),
        "numeric_summary": numeric_summary(df, schema),
        "date_summary": date_summaries(df, schema),
        "top5_categories": top_categories(df, schema),
        "preview": preview_payload(df),
        "notes": {
            "nulls_definition": "Counts of empty, whitespace-only, or NaN values per column.",
        },
    }
    return payload
