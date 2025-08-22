# guardrail_api.py
# Reverie Analytics API – clean + profile uploaded CSV/XLSX
# v0.3.3

from __future__ import annotations

import io
import csv
import re
import uuid
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------
# FastAPI app + CORS
# -----------------------------
app = FastAPI(
    title="Reverie Analytics API",
    description="Upload a dataset, then profile it.",
    version="0.3.3",
)

# Keep CORS permissive during setup. Lock down later for prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# In-memory dataset store
# -----------------------------
_DATASETS: Dict[str, pd.DataFrame] = {}

# -----------------------------
# Robust CSV/Excel reading
# -----------------------------
def _looks_broken(df: pd.DataFrame) -> bool:
    if df is None or df.shape[0] == 0:
        return True
    if df.shape[1] == 1:
        return True
    nn = df.notna().mean()
    return (nn.max() > 0.8) and ((nn < 0.2).sum() >= max(1, df.shape[1] - 1))


def _strip_quotes_everywhere(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().strip('"').strip("'") for c in df.columns]
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            s = df[col].astype("string")
            s = s.str.replace(r'^\s*("|\')', "", regex=True)
            s = s.str.replace(r'("|\')\s*$', "", regex=True)
            df[col] = s
    return df


def _read_csv_text(text: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
        if not _looks_broken(df):
            return _strip_quotes_everywhere(df)
    except Exception:
        pass

    for sep in [",", "\t", ";", "|"]:
        try:
            df2 = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
            if not _looks_broken(df2):
                return _strip_quotes_everywhere(df2)
        except Exception:
            continue

    try:
        df3 = pd.read_csv(
            io.StringIO(text),
            sep=",",
            engine="python",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
        )
        return _strip_quotes_everywhere(df3)
    except Exception:
        pass

    df_fallback = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    return _strip_quotes_everywhere(df_fallback)


def _read_dataframe_from_upload(file: UploadFile) -> pd.DataFrame:
    name = (file.filename or "").lower()

    if name.endswith((".xlsx", ".xls")):
        data = file.file.read()
        df = pd.read_excel(io.BytesIO(data))
        return _strip_quotes_everywhere(df)

    raw = file.file.read()
    try:
        text = raw.decode("utf-8-sig", errors="replace")
    except Exception:
        text = raw.decode("utf-8", errors="replace")

    return _read_csv_text(text)

# -----------------------------
# Cleaning / coercions
# -----------------------------
def to_numeric_clean(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    t = s.astype("string")

    has_percent = t.str.contains("%", regex=False, na=False)

    t = t.str.replace(r"[^0-9\-\.,%()]", "", regex=True)

    neg = t.str.contains(r"\(", regex=True, na=False) & t.str.contains(r"\)", regex=True, na=False)
    t = t.str.replace(r"[()]", "", regex=True)

    both = t.str.contains(r"\.", regex=True, na=False) & t.str.contains(r",", regex=True, na=False)
    t = t.mask(both, t.str.replace(",", "", regex=False))

    comma_only = ~t.str.contains(r"\.", regex=True, na=False) & t.str.contains(r",", regex=True, na=False)
    t = t.mask(comma_only, t.str.replace(",", ".", regex=False))

    t = t.str.replace("%", "", regex=False)

    out = pd.to_numeric(t, errors="coerce")
    out = out.where(~neg, -out)
    out = out.where(~has_percent, out / 100.0)
    return out


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        s = df[col]
        s_num = to_numeric_clean(s)
        if pd.api.types.is_numeric_dtype(s):
            df[col] = pd.to_numeric(s, errors="coerce")
        else:
            valid_ratio = s_num.notna().mean()
            if valid_ratio >= 0.6:
                df[col] = s_num
    return df


def detect_date_columns(df: pd.DataFrame) -> List[str]:
    candidates: List[str] = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            candidates.append(col)
            continue
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            parsed = pd.to_datetime(s, errors="coerce", utc=False)
            if parsed.notna().mean() >= 0.6:
                candidates.append(col)
    return candidates


def summarize_dates(df: pd.DataFrame, date_cols: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    for col in date_cols:
        dt = pd.to_datetime(df[col], errors="coerce", utc=False)
        if dt.notna().sum() == 0:
            continue

        dmin = dt.min()
        dmax = dt.max()
        by_month = dt.dt.to_period("M").astype(str).value_counts().sort_index()
        by_weekday = dt.dt.day_name().value_counts()
        by_weekday = by_weekday.reindex(weekday_order, fill_value=0).astype(int)

        out[col] = {
            "min": dmin.strftime("%Y-%m-%d"),
            "max": dmax.strftime("%Y-%m-%d"),
            "span_days": int((dmax.normalize() - dmin.normalize()).days),
            "by_month": by_month.to_dict(),
            "by_weekday": by_weekday.to_dict(),
        }
    return out

# -----------------------------
# Profiling
# -----------------------------
def round2(x):
    if pd.isna(x):
        return None
    if isinstance(x, (int, float)):
        return int(x) if float(x).is_integer() else round(float(x), 2)
    return x


def numeric_summary(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in num_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() == 0:
            continue
        q = s.quantile([0.25, 0.5, 0.75])
        desc = {
            "count": int(s.notna().sum()),
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "25%": q.get(0.25, None),
            "50%": q.get(0.5, None),
            "75%": q.get(0.75, None),
            "max": s.max(),
        }
        out[col] = {k: round2(v) for k, v in desc.items()}
    return out


def top5_categories(df: pd.DataFrame, exclude_cols: List[str] | None = None) -> Dict[str, Dict[str, int]]:
    exclude = set(exclude_cols or [])
    out: Dict[str, Dict[str, int]] = {}
    non_num_cols = df.select_dtypes(exclude=["number", "datetime"]).columns.tolist()
    for col in non_num_cols:
        if col in exclude:
            continue
        counts = df[col].astype("string").value_counts(dropna=True).head(5)
        if len(counts) > 0:
            out[col] = counts.to_dict()
    return out


def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    date_cols = detect_date_columns(df)
    for c in date_cols:
        if not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=False)
    df = coerce_numeric_columns(df)

    result: Dict[str, Any] = {}
    result["shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
    result["columns"] = df.columns.astype(str).tolist()
    result["nulls"] = {c: int(df[c].isna().sum()) for c in df.columns}
    result["numeric_summary"] = numeric_summary(df)
    result["date_summary"] = summarize_dates(df, date_cols)
    result["top5_categories"] = top5_categories(df, exclude_cols=date_cols)
    return result

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root() -> Dict[str, Any]:
    return {"message": "Reverie Analytics API. Try /health or /analytics/*", "version": app.version}

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/analytics/upload")
async def upload(file: UploadFile = File(...)) -> Dict[str, str]:
    try:
        df = _read_dataframe_from_upload(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if df is None or df.shape[0] == 0:
        raise HTTPException(status_code=400, detail="Uploaded file appears empty.")

    dataset_id = str(uuid.uuid4())
    _DATASETS[dataset_id] = df
    return {"dataset_id": dataset_id}

@app.get("/analytics/profile")
def profile(dataset_id: str = Query(...)) -> Dict[str, Any]:
    df = _DATASETS.get(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="dataset_id not found or expired")
    try:
        return profile_dataframe(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"profiling failed: {e}")

@app.get("/analytics/preview")
def preview(
    dataset_id: str = Query(...),
    limit: int = Query(10, ge=1, le=100),
) -> Dict[str, Any]:
    """
    Return first N rows for a dataset_id for quick preview.
    """
    df = _DATASETS.get(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="dataset_id not found or expired")

    sample = df.head(limit).copy()

    # Format datetimes as ISO strings for the UI
    for c in sample.columns:
        if pd.api.types.is_datetime64_any_dtype(sample[c]):
            sample[c] = pd.to_datetime(sample[c], errors="coerce", utc=False)
            sample[c] = sample[c].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Replace NaN with None for JSON
    sample = sample.where(pd.notna(sample), None)

    return {
        "columns": [str(c) for c in sample.columns],
        "rows": sample.to_dict(orient="records"),
        "limit": limit,
        "total_rows": int(df.shape[0]),
    }

# -----------------------------
# Local dev (Render uses Start Command)
# -----------------------------
if __name__ == "__main__":
    import uvicorn, os
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("guardrail_api:app", host="0.0.0.0", port=port, reload=True)
