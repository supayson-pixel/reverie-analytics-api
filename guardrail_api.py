# guardrail_api.py
from __future__ import annotations

import io
import uuid
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ------------------------------------------------------------------------------
# FastAPI app + CORS
# ------------------------------------------------------------------------------

ALLOWED_ORIGINS = [
    "https://www.reveriesun.com",
    "https://reveriesun.com",
    # Netlify (prod/preview)
    "https://inspiring-tarsier-97b2c7.netlify.app",
    "https://reveriesun.netlify.app",
    # Local dev (optional)
    "http://localhost:3000",
    "http://localhost:5173",
]

app = FastAPI(title="Reverie Analytics API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# In-memory dataset cache
# ------------------------------------------------------------------------------

_DATASETS: Dict[str, pd.DataFrame] = {}

# ------------------------------------------------------------------------------
# Helpers: robust file reading
# ------------------------------------------------------------------------------

def _read_csv_safely(text: str) -> pd.DataFrame:
    """
    Try reading CSV text with pandas. First attempt default, then python engine
    with sep=None (sniffer) for odd delimiters.
    """
    try:
        return pd.read_csv(io.StringIO(text))
    except Exception:
        # Odd delimiters or multi-char separators -> sniff with python engine
        return pd.read_csv(io.StringIO(text), sep=None, engine="python")


def _read_bytes_as_df(name: str, data: bytes) -> pd.DataFrame:
    nlower = (name or "").lower()
    # Excel
    if nlower.endswith(".xlsx") or nlower.endswith(".xls"):
        # openpyxl is needed for .xlsx; if not installed this will raise
        try:
            return pd.read_excel(io.BytesIO(data))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read Excel: {e}")
    # CSV / TXT
    try:
        text = data.decode("utf-8", errors="ignore")
    except Exception:
        # last resort
        text = data.decode("latin1", errors="ignore")
    return _read_csv_safely(text)


# ------------------------------------------------------------------------------
# Type conversion utilities
# ------------------------------------------------------------------------------

def to_numeric_clean(series: pd.Series) -> pd.Series:
    """
    Convert strings with currency symbols, commas, spaces, or percent signs to numeric.
    Uses vectorized string ops; falls back to original if too few values convert.
    """
    # Already numeric? nothing to do.
    if pd.api.types.is_numeric_dtype(series):
        return series

    # Only try for object/string-like columns.
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return series

    s = series.astype("string")

    # Detect percents before removing symbol
    pct_mask = s.str.contains("%", na=False)

    # Vectorized cleanup
    s = s.str.replace(r"[\$,]", "", regex=True)   # $ 1,234.56 -> 1234.56
    s = s.str.replace(r"\s+", "", regex=True)     # remove stray spaces
    s = s.str.replace(r"%", "", regex=True)       # drop percent sign

    num = pd.to_numeric(s, errors="coerce")

    # Interpret percent values as decimal if there was a percent sign
    if pct_mask.any():
        num.loc[pct_mask] = num.loc[pct_mask] / 100.0

    # Adopt conversion only if at least 60% of non-null values parsed
    orig_non_null = series.notna().sum()
    converted = num.notna().sum()
    if orig_non_null > 0 and converted / orig_non_null >= 0.6:
        return num

    return series


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt numeric coercion on non-datetime columns."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            continue
        out[col] = to_numeric_clean(out[col])
    return out


def maybe_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    For object/string columns, try to parse to datetime. Convert a column if
    >= 60% of non-null values parse successfully.
    """
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            continue
        if not (pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col])):
            continue

        s = out[col].astype("string")
        parsed = pd.to_datetime(s, errors="coerce", utc=False)  # no infer_datetime_format (deprecated)
        non_null = s.notna().sum()
        parsed_ok = parsed.notna().sum()
        if non_null > 0 and parsed_ok / non_null >= 0.6:
            out[col] = parsed.dt.tz_localize(None)
    return out


# ------------------------------------------------------------------------------
# Summaries
# ------------------------------------------------------------------------------

def numeric_summary(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    num = df.select_dtypes(include=["number"])
    if num.empty:
        return {}
    desc = num.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
    # Transpose-like: {col: {metric: value}}
    out: Dict[str, Dict[str, Any]] = {}
    metrics = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    for col in num.columns:
        out[col] = {}
        for m in metrics:
            val = desc.get(m, {}).get(col, None)
            if isinstance(val, (np.floating, float)):
                val = round(float(val), 2)
            elif isinstance(val, (np.integer, int)):
                val = int(val)
            out[col][m] = val
    return out


def top5_categories(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    # Only for non-numeric columns (including datetimes? skip)
    non_num = df.select_dtypes(exclude=["number", "datetime64[ns]"])
    for col in non_num.columns:
        vc = (
            non_num[col]
            .astype("string")
            .value_counts(dropna=True)
            .head(5)
        )
        if not vc.empty:
            out[col] = {str(k): int(v) for k, v in vc.items()}
    return out


def date_summary(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    For each datetime column, report: min, max, span_days, counts by month & weekday.
    """
    out: Dict[str, Dict[str, Any]] = {}
    date_cols = df.select_dtypes(include=["datetime64[ns]"]).columns
    for col in date_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        dmin = s.min()
        dmax = s.max()
        by_month = (
            s.dt.to_period("M").astype(str).value_counts().sort_index().to_dict()
        )
        by_weekday = (
            s.dt.day_name().value_counts().reindex(
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                fill_value=0
            ).to_dict()
        )
        out[col] = {
            "min": dmin.date().isoformat(),
            "max": dmax.date().isoformat(),
            "span_days": int((dmax - dmin).days),
            "by_month": by_month,
            "by_weekday": by_weekday,
        }
    return out


def nulls_per_column(df: pd.DataFrame) -> Dict[str, int]:
    return {str(c): int(df[c].isna().sum()) for c in df.columns}


# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------

@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Reverie Analytics API. Try /health or /analytics/*"}

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/analytics/upload")
async def upload(file: UploadFile = File(...)) -> JSONResponse:
    try:
        data = await file.read()
        df = _read_bytes_as_df(file.filename or "upload", data)

        # Normalize: drop fully empty columns & rows
        df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

        # Keep original column order/names
        ds_id = str(uuid.uuid4())
        _DATASETS[ds_id] = df

        return JSONResponse(
            {
                "dataset_id": ds_id,
                "rows": int(len(df)),
                "columns": [str(c) for c in df.columns],
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Upload error: {e}")

@app.get("/analytics/profile")
def profile(dataset_id: str = Query(..., min_length=3)) -> JSONResponse:
    if dataset_id not in _DATASETS:
        raise HTTPException(status_code=404, detail="dataset_id not found")

    df = _DATASETS[dataset_id].copy()

    # Type coercions
    df = maybe_parse_dates(df)
    df = coerce_numeric_columns(df)

    # Build response
    resp: Dict[str, Any] = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": [str(c) for c in df.columns],
        "nulls": nulls_per_column(df),
        "numeric_summary": numeric_summary(df),   # rounded inside
        "top5_categories": top5_categories(df),
        "date_summary": date_summary(df),
    }

    return JSONResponse(resp)
