# guardrail_api.py
# FastAPI service for quick file uploads + profiling
# Endpoints:
#   GET  /health
#   POST /analytics/upload        (returns dataset_id)
#   GET  /analytics/profile?dataset_id=...

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Any
from uuid import uuid4
import pandas as pd
import numpy as np
import io, csv, re

app = FastAPI(title="Reverie Analytics API", version="1.0.0")

# -----------------------------------------------------------------------------
# CORS – allow your production + preview domains
# -----------------------------------------------------------------------------
ALLOWED_ORIGINS = [
    "https://www.reveriesun.com",
    "https://reveriesun.com",
    "https://reveriesun.netlify.app",
    "https://inspiring-tarsier-97b2c7.netlify.app",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5500",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # use ["*"] if you prefer
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Simple in-memory object store for uploaded files (dataset_id -> bytes, name)
# -----------------------------------------------------------------------------
_DATA_STORE: Dict[str, Dict[str, Any]] = {}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
_CURRENCY_CHARS = re.compile(r"[\$,]")
_PERCENT_CHAR = re.compile(r"%")
_WHITESPACE = re.compile(r"\s+")


def sniff_delimiter(sample: bytes) -> str:
    """Try to guess delimiter for CSV-like files."""
    try:
        text = sample.decode("utf-8", errors="ignore")
        dialect = csv.Sniffer().sniff(text.splitlines()[0])
        return dialect.delimiter
    except Exception:
        # fallback to comma
        return ","


def read_table(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load CSV/TXT/XLS/XLSX into a DataFrame with best-effort parsing."""
    name = filename.lower()
    stream = io.BytesIO(file_bytes)

    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(stream)
        return df

    # CSV / TXT
    # Try UTF-8 first; if it fails in a way that affects structure, fallback
    text = file_bytes.decode("utf-8-sig", errors="ignore")
    delim = sniff_delimiter(file_bytes)
    try:
        df = pd.read_csv(io.StringIO(text), sep=delim)
    except Exception:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")

    # If we somehow got a single merged column but commas exist, re-read
    if df.shape[1] == 1 and "," in text.splitlines()[0]:
        df = pd.read_csv(io.StringIO(text), sep=",")
    return df


def try_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to convert obvious date-like text columns to datetime."""
    df = df.copy()
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        if not pd.api.types.is_object_dtype(s) and not pd.api.types.is_string_dtype(s):
            continue

        # Try parse; consider it a date if many non-nulls parse successfully
        parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True, utc=False)
        non_null = s.notna().sum()
        parsed_ok = parsed.notna().sum()
        if non_null > 0 and parsed_ok / max(1, non_null) >= 0.7:
            df[col] = parsed.dt.tz_localize(None)  # drop tz if present
    return df


def to_numeric_clean(series: pd.Series) -> pd.Series:
    """
    Convert strings with currency symbols, commas, percent signs to numeric.
    Keeps original if too few values convert.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series

    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return series

    txt = series.astype(str)
    txt = _CURRENCY_CHARS.sub("", txt)          # $ 1,234.56 -> 1234.56
    txt = _WHITESPACE.sub("", txt)              # remove stray spaces
    pct_mask = txt.str.contains("%", na=False)
    txt = _PERCENT_CHAR.sub("", txt)
    num = pd.to_numeric(txt, errors="coerce")

    # If we cleaned % values, interpret 12.3% as 0.123 *unless* most look like whole dollars
    if pct_mask.any():
        num.loc[pct_mask] = num.loc[pct_mask] / 100.0

    # adopt conversion if at least 60% of non-null strings converted
    orig_non_null = series.notna().sum()
    converted = num.notna().sum()
    if orig_non_null > 0 and converted / orig_non_null >= 0.6:
        return num
    return series


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply numeric coercion (with currency/percent support) to object columns."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        df[col] = to_numeric_clean(df[col])
    return df


def round_floats(obj, ndigits: int = 2):
    """Recursively round floats in dict/list structures."""
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, ndigits) for v in obj]
    if isinstance(obj, float) and np.isfinite(obj):
        return round(obj, ndigits)
    if isinstance(obj, (np.floating,)):
        try:
            return round(float(obj), ndigits)
        except Exception:
            return float(obj)
    return obj


def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Build the profiling summary JSON."""
    # Basic info
    shape = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
    columns = [str(c) for c in df.columns]

    # Null counts
    nulls = {str(c): int(df[c].isna().sum()) for c in df.columns}

    # Numeric summary (rounded)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_summary: Dict[str, Dict[str, Any]] = {}
    for c in numeric_cols:
        s = df[c].dropna()
        if s.empty:
            continue
        q25, q50, q75 = s.quantile([0.25, 0.5, 0.75]).tolist()
        numeric_summary[str(c)] = {
            "count": int(s.count()),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if s.count() > 1 else 0.0,
            "min": float(s.min()),
            "25%": float(q25),
            "50%": float(q50),
            "75%": float(q75),
            "max": float(s.max()),
        }

    # Top categories (strings / non-numeric, non-datetime)
    cat_cols = [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    top5_categories: Dict[str, Dict[str, int]] = {}
    for c in cat_cols:
        vc = df[c].dropna().astype(str).value_counts().head(5)
        if not vc.empty:
            top5_categories[str(c)] = {str(k): int(v) for k, v in vc.items()}

    # dtypes
    dtypes = {str(c): str(dt) for c, dt in df.dtypes.items()}

    # date range summary
    date_summary: Dict[str, Dict[str, str]] = {}
    for c in df.select_dtypes(include="datetime64[ns]").columns:
        s = df[c].dropna()
        if not s.empty:
            date_summary[str(c)] = {
                "min": s.min().isoformat(),
                "max": s.max().isoformat(),
            }

    out = {
        "shape": shape,
        "columns": columns,
        "nulls": nulls,
        "numeric_summary": numeric_summary,
        "top5_categories": top5_categories,
        "dtypes": dtypes,
        "date_summary": date_summary,
    }
    return round_floats(out, 2)

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return JSONResponse(
        {"message": "Reverie Analytics API. Try /health or /analytics/*"},
        status_code=200,
    )

@app.post("/analytics/upload")
async def upload(file: UploadFile = File(...)):
    """Upload a CSV/XLSX file and receive a dataset_id for profiling."""
    name = (file.filename or "upload").strip()
    ext = (name.split(".")[-1] or "").lower()

    if ext not in ("csv", "txt", "xlsx", "xls"):
        raise HTTPException(status_code=415, detail="Unsupported file type")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")

    dsid = str(uuid4())
    _DATA_STORE[dsid] = {"name": name, "bytes": data}
    return {"dataset_id": dsid, "filename": name, "size": len(data)}

@app.get("/analytics/profile")
def profile(dataset_id: str):
    """Return profiling summary for a previously uploaded dataset."""
    item = _DATA_STORE.get(dataset_id)
    if not item:
        raise HTTPException(status_code=404, detail="Unknown dataset_id")

    # Load DataFrame
    df = read_table(item["bytes"], item["name"])

    # Best-effort cleanups
    df.columns = [str(c).strip() for c in df.columns]
    df = try_parse_dates(df)
    df = coerce_numeric_columns(df)

    # Build and return summary
    return profile_dataframe(df)
