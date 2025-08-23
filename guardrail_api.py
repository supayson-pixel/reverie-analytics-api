import io
import os
import re
import uuid
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------------------------------------------------------
# App & CORS
# -----------------------------------------------------------------------------
app = FastAPI(title="Reverie Analytics API", version="0.2.0")

ALLOWED_ORIGINS = [
    "https://www.reveriesun.com",
    "https://reveriesun.com",
    # keep your temporary Netlify subdomain(s) while testing:
    "https://inspiring-tarsier-97b2c7.netlify.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# In-memory storage (ephemeral)
# -----------------------------------------------------------------------------
_DATASETS: Dict[str, pd.DataFrame] = {}

# -----------------------------------------------------------------------------
# Helpers: missing values, numeric & date coercion
# -----------------------------------------------------------------------------
_MISSING_PATTERNS = [
    r"^\s*$",     # empty/whitespace
    r"^na$", r"^n/a$", r"^none$", r"^null$", r"^nil$",
    r"^-$", r"^--$",
]
_MISSING_REGEX = re.compile("|".join(_MISSING_PATTERNS), re.IGNORECASE)

def standardize_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Convert empty/placeholder strings to NaN across all object/string columns."""
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    if len(obj_cols):
        df[obj_cols] = df[obj_cols].applymap(
            lambda x: np.nan if isinstance(x, str) and _MISSING_REGEX.match(x) else x
        )
    return df

# currency/commas/% cleanup
def _clean_numeric_str(x: Any) -> Any:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    if not isinstance(x, str):
        return np.nan

    s = x.strip()
    if not s:
        return np.nan

    # parentheses negatives e.g. (1,234.56)
    neg = s.startswith("(") and s.endswith(")")
    s = s.replace("−", "-")  # U+2212 to ASCII hyphen

    # keep digits, decimal, minus
    s_num = re.sub(r"[^0-9.\-]", "", s)
    try:
        val = float(s_num) if s_num not in ("", "-", ".") else np.nan
    except Exception:
        val = np.nan
    if neg and not (isinstance(val, float) and np.isnan(val)):
        val = -val
    return val

def to_numeric_clean(series: pd.Series) -> pd.Series:
    """Return a numeric series by cleaning strings like '$1,234' or '93.2%'.
       Works for object/string/numeric series. Always returns float dtype."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype(float)
    if pd.api.types.is_string_dtype(series) or series.dtype == "object":
        return series.map(_clean_numeric_str).astype(float)
    # anything else -> try numeric
    return pd.to_numeric(series, errors="coerce").astype(float)

def coerce_numeric_columns(df: pd.DataFrame, min_convertible: float = 0.6) -> pd.DataFrame:
    """Convert object-like columns to numeric if ≥ min_convertible fraction parse OK."""
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            continue
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            converted = to_numeric_clean(s)
            frac = converted.notna().mean()
            if frac >= min_convertible and converted.notna().sum() > 0:
                df[col] = converted
    return df

def coerce_datetime_columns(df: pd.DataFrame, min_convertible: float = 0.6) -> pd.DataFrame:
    """Convert object-like columns to datetime if ≥ min_convertible parse OK."""
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_numeric_dtype(s):
            continue
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            parsed = pd.to_datetime(s, errors="coerce", utc=False)
            frac = parsed.notna().mean()
            if frac >= min_convertible and parsed.notna().sum() > 0:
                df[col] = parsed.dt.tz_localize(None)
    return df

# -----------------------------------------------------------------------------
# File readers
# -----------------------------------------------------------------------------
def _read_text(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin1", errors="ignore")

def read_upload_to_df(upload: UploadFile) -> pd.DataFrame:
    name = (upload.filename or "").lower()
    ext = os.path.splitext(name)[1]

    # Excel
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(upload.file)

    # CSV/TXT/TSV
    file_bytes = upload.file.read()
    text = _read_text(file_bytes)

    if ext == ".tsv":
        return pd.read_csv(io.StringIO(text), sep="\t")

    # sep=None lets pandas sniff; engine='python' for flexibility
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")

# -----------------------------------------------------------------------------
# Profiling logic
# -----------------------------------------------------------------------------
def numeric_summary(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    num = df.select_dtypes(include="number")
    for c in num.columns:
        s = num[c].dropna()
        if s.empty:
            continue
        stats = {
            "count": float(len(s)),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if len(s) > 1 else float("nan"),
            "min": float(s.min()),
            "25%": float(s.quantile(0.25)),
            "50%": float(s.quantile(0.50)),
            "75%": float(s.quantile(0.75)),
            "max": float(s.max()),
        }
        # round nicely
        for k in list(stats.keys()):
            if pd.isna(stats[k]):
                continue
            stats[k] = float(np.round(stats[k], 2))
        out[c] = stats
    return out

_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

def date_summary(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    dt = df.select_dtypes(include="datetime")
    for c in dt.columns:
        s = dt[c].dropna()
        if s.empty:
            continue
        mn, mx = s.min(), s.max()
        span = int((mx - mn).days)
        by_month = s.dt.to_period("M").astype(str).value_counts().sort_index()
        by_wd = s.dt.day_name().value_counts()
        # include zeroes for missing weekdays
        by_wd_full = {wd: int(by_wd.get(wd, 0)) for wd in _WEEKDAYS}
        out[c] = {
            "min": str(mn.date()),
            "max": str(mx.date()),
            "span_days": span,
            "by_month": {k: int(v) for k, v in by_month.items()},
            "by_weekday": by_wd_full,
        }
    return out

def top5_categories(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for c in df.columns:
        s = df[c]
        # skip numeric and datetime columns
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        # only meaningful for small cardinality
        vc = s.dropna().astype(str).str.strip()
        if vc.empty:
            continue
        freq = vc.value_counts().head(5)
        if not freq.empty:
            out[c] = {k: int(v) for k, v in freq.items()}
    return out

def df_preview(df: pd.DataFrame, limit: int = 10) -> Dict[str, Any]:
    rows = df.head(limit).to_dict(orient="records")
    cols = list(df.columns)
    return {"preview": rows, "preview_columns": cols}

def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    # standardize empties/NA placeholders
    df = standardize_missing(df.copy())

    # try date & numeric coercions (order helps avoid bad numeric over strings that are dates)
    df = coerce_datetime_columns(df)
    df = coerce_numeric_columns(df)

    shape = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
    cols = list(df.columns)

    # null counts after cleaning
    nulls = df.isna().sum().astype(int).to_dict()

    resp: Dict[str, Any] = {
        "shape": shape,
        "columns": cols,
        "nulls": nulls,
        "numeric_summary": numeric_summary(df),
        "date_summary": date_summary(df),
        "top5_categories": top5_categories(df),
    }
    resp.update(df_preview(df))
    return resp

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Reverie Analytics API. Try /health or /analytics/*"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/analytics/upload")
async def upload(file: UploadFile = File(...)) -> Dict[str, str]:
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    try:
        df = read_upload_to_df(file)
        if df is None or df.empty:
            raise ValueError("No rows parsed.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")

    dsid = str(uuid.uuid4())
    _DATASETS[dsid] = df
    return {"dataset_id": dsid}

@app.get("/analytics/profile")
def profile(dataset_id: str) -> Dict[str, Any]:
    if not dataset_id or dataset_id not in _DATASETS:
        raise HTTPException(status_code=404, detail="dataset_id not found.")
    df = _DATASETS[dataset_id]
    try:
        return profile_dataframe(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"profiling failed: {e}")
