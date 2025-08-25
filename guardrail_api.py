# guardrail_api.py
# FastAPI service for upload + dataframe profiling
# Dependencies: fastapi, uvicorn[standard], python-multipart, pandas, pydantic

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import io
import re
import uuid
from typing import Dict, Any

app = FastAPI(title="Reverie Analytics API")

# --- CORS: allow your domains + any Netlify preview site ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],  # keep empty when using regex below
    allow_origin_regex=r"https://([a-z0-9-]+\.)?reveriesun\.com$|https://[a-z0-9-]+\.netlify\.app$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Defensive: sometimes proxies send stray OPTIONS preflights – respond OK.
@app.options("/{rest_of_path:path}")
def preflight_ok(rest_of_path: str, request: Request):
    return JSONResponse({"ok": True})

# ------------------------------------------------------------------
# Simple health + root
# ------------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/")
def root():
    return {"message": "DAFE API. Try /health or /analytics/*"}

# ------------------------------------------------------------------
# Upload handling and in-memory dataset store
# ------------------------------------------------------------------
_DATASETS: Dict[str, pd.DataFrame] = {}

class ProfileResponse(BaseModel):
    shape: Dict[str, int]
    columns: list
    nulls: Dict[str, int]
    numeric_summary: Dict[str, Dict[str, float]]
    date_summary: Dict[str, Dict[str, Any]] = {}
    top5_categories: Dict[str, Dict[str, int]] = {}
    preview: Dict[str, Any] = {}

# --- utilities -----------------------------------------------------

def _read_any_table(upload: UploadFile) -> pd.DataFrame:
    """
    Reads CSV/TXT/XLSX/XLS. For CSV, auto-detects delimiter via pandas engine='python'.
    """
    name = (upload.filename or "").lower()
    raw = upload.file.read()  # bytes
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    if name.endswith(".xlsx") or name.endswith(".xls"):
        # Excel requires openpyxl for .xlsx; if not installed this will fail.
        # You can add `openpyxl` to requirements.txt if you want xlsx uploads.
        try:
            return pd.read_excel(io.BytesIO(raw))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Excel read failed: {e}")

    # default: CSV/TXT
    # Let pandas sniff the delimiter (engine='python' allows automatic sep=None)
    # Decode with utf-8, fall back to latin-1 for odd encodings.
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV read failed: {e}")

    return df


_CURRENCY_CHARS = re.compile(r"[\$,()%\s]")
_COMMA = re.compile(r",")

def to_numeric_clean_series(s: pd.Series) -> pd.Series:
    """
    Vectorized numeric coercion:
    - strips currency symbols, commas, parens
    - handles negatives in () style
    Returns float series (NaN where not parseable).
    """
    if s.dtype.kind in "biufc":  # already numeric
        return s.astype(float)

    # Work on string view
    st = s.astype("string", copy=False)

    # Detect "(123.45)" -> "-123.45"
    neg_mask = st.str.contains(r"^\s*\(.*\)\s*$", regex=True, na=False)
    cleaned = st.str.replace(r"^\s*\((.*)\)\s*$", r"-\1", regex=True)

    # Remove currency and commas
    cleaned = cleaned.str.replace(r"[\$,%\s]", "", regex=True)
    cleaned = cleaned.str.replace(",", "", regex=False)

    out = pd.to_numeric(cleaned, errors="coerce")
    # If we removed trailing % signs, user may expect it as actual numeric (already handled above)
    return out.astype(float)

def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to convert 'object'-like columns that are actually numbers into numeric dtype.
    We only adopt the conversion if it increases the number of non-null numeric values,
    to avoid destroying text columns.
    """
    out = df.copy()
    for col in out.columns:
        if out[col].dtype.kind in "biufc":
            continue
        s = out[col]
        try:
            converted = to_numeric_clean_series(s)
            # If conversion yields more valid numbers than before, keep it
            if converted.notna().sum() >= s.notna().sum() * 0.4:  # fairly permissive
                out[col] = converted
        except Exception:
            # leave as-is
            pass
    return out

def is_date_like(s: pd.Series) -> bool:
    """
    Heuristic: can the majority of non-null values parse as dates?
    """
    if s.isna().all():
        return False
    try:
        parsed = pd.to_datetime(s, errors="coerce", utc=False)
        valid = parsed.notna().sum()
        n = s.notna().sum()
        return bool(n) and valid / n >= 0.6  # 60% parses as date
    except Exception:
        return False

def numeric_summary(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    res: Dict[str, Dict[str, float]] = {}
    for col in df.columns:
        if df[col].dtype.kind in "biufc":
            s = df[col].dropna().astype(float)
            if s.empty:
                continue
            q = s.quantile([0.25, 0.5, 0.75])
            res[col] = {
                "count": float(len(s)),
                "mean": round(float(s.mean()), 2),
                "std": round(float(s.std(ddof=1)), 2) if len(s) > 1 else 0.0,
                "min": round(float(s.min()), 2),
                "25%": round(float(q.loc[0.25]), 2),
                "50%": round(float(q.loc[0.5]), 2),
                "75%": round(float(q.loc[0.75]), 2),
                "max": round(float(s.max()), 2),
            }
    return res

def date_summary(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    res: Dict[str, Dict[str, Any]] = {}
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for col in df.columns:
        s = df[col]
        if not is_date_like(s):
            continue
        parsed = pd.to_datetime(s, errors="coerce", utc=False)
        parsed = parsed.dropna()
        if parsed.empty:
            continue
        by_month = parsed.dt.to_period("M").astype(str).value_counts().sort_index()
        by_wd = parsed.dt.day_name().value_counts()
        # include zeros for missing weekdays
        by_wd = {wd: int(by_wd.get(wd, 0)) for wd in weekdays}
        res[col] = {
            "min": str(parsed.min().date()),
            "max": str(parsed.max().date()),
            "span_days": int((parsed.max() - parsed.min()).days),
            "by_month": {k: int(v) for k, v in by_month.items()},
            "by_weekday": by_wd,
        }
    return res

def top5_categories(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    res: Dict[str, Dict[str, int]] = {}
    for col in df.columns:
        s = df[col]
        # skip numeric columns and strongly date-like columns
        if s.dtype.kind in "biufc" or is_date_like(s):
            continue
        vc = s.astype("string", copy=False).value_counts().head(5)
        if not vc.empty:
            res[col] = {str(k): int(v) for k, v in vc.items()}
    return res

def preview_rows(df: pd.DataFrame, rows: int = 10) -> Dict[str, Any]:
    head = df.head(rows)
    return {
        "rows": rows,
        "columns": list(head.columns),
        "data": head.replace({np.nan: None}).to_dict(orient="records"),
    }

# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.post("/analytics/upload")
async def upload(file: UploadFile = File(...)):
    try:
        df = _read_any_table(file)
    finally:
        try:
            await file.close()
        except Exception:
            pass

    # Clean up columns (strip BOM, whitespace)
    df.columns = [str(c).encode("utf-8", "ignore").decode("utf-8").strip().strip('"').strip("'") for c in df.columns]
    # Normalize numerics
    df = coerce_numeric_columns(df)

    dataset_id = str(uuid.uuid4())
    _DATASETS[dataset_id] = df

    return {"dataset_id": dataset_id, "rows": int(df.shape[0]), "columns": int(df.shape[1])}

@app.get("/analytics/profile", response_model=ProfileResponse)
def profile(dataset_id: str = Query(..., description="dataset id returned by /analytics/upload")):
    if dataset_id not in _DATASETS:
        raise HTTPException(status_code=404, detail="dataset_id not found")

    df = _DATASETS[dataset_id]

    out: Dict[str, Any] = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": list(df.columns),
        "nulls": {c: int(df[c].isna().sum()) for c in df.columns},
        "numeric_summary": numeric_summary(df),
        "date_summary": date_summary(df),
        "top5_categories": top5_categories(df),
        "preview": preview_rows(df, rows=10),
    }
    return out


# For local debug:
#   uvicorn guardrail_api:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("guardrail_api:app", host="0.0.0.0", port=8000, reload=True)
