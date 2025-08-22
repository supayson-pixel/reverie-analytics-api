# guardrail_api.py
# Reverie Analytics API – clean + profile uploaded CSV/XLSX
# v0.3.0

from __future__ import annotations

import io
import uuid
from typing import Dict, Any, List

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------
# FastAPI app + CORS
# -----------------------------
app = FastAPI(
    title="Reverie Analytics API",
    description="Upload a dataset, then profile it.",
    version="0.3.0",
)

# Allow your production domains and any Netlify preview
_CORS_ORIGINS = [
    "https://www.reveriesun.com",
    "https://reveriesun.com",
    "https://reveriesun.netlify.app",
    "https://inspiring-tarsier-97b2c7.netlify.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_origin_regex=r"https://.*\.netlify\.app",   # enable deploy previews
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# In-memory dataset store
# -----------------------------
_DATASETS: Dict[str, pd.DataFrame] = {}

# -----------------------------
# Helpers: input loading
# -----------------------------
def _read_dataframe_from_upload(file: UploadFile) -> pd.DataFrame:
    name = (file.filename or "").lower()

    if name.endswith((".xlsx", ".xls")):
        # Excel
        data = file.file.read()
        return pd.read_excel(io.BytesIO(data))
    else:
        # CSV/TXT — let pandas sniff delimiter. Use python engine to avoid c-engine warnings.
        text = file.file.read().decode("utf-8", errors="replace")
        # sep=None triggers sniffing; python engine required for separator sniff
        return pd.read_csv(io.StringIO(text), sep=None, engine="python")

# -----------------------------
# Helpers: coercions
# -----------------------------
def to_numeric_clean(s: pd.Series) -> pd.Series:
    """
    Convert strings like '$1,234.56', '45%', '1 234', '(123)' to floats.
    Keeps NaN where conversion fails.
    Vectorized, no per-row regex.
    """
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    # Work on string view
    t = s.astype("string")

    # Detect percent anywhere
    has_percent = t.str.contains("%", regex=False, na=False)

    # Remove everything except digits, signs, decimal sep and percent
    # Keep . and , then normalize comma thousands.
    # 1) Strip currency/letters/spaces
    t = t.str.replace(r"[^0-9\-\.,%()]", "", regex=True)

    # 2) Handle parentheses negative e.g. (123.45)
    neg = t.str.contains(r"\(", regex=True, na=False) & t.str.contains(r"\)", regex=True, na=False)
    t = t.str.replace(r"[()]", "", regex=True)

    # 3) If both '.' and ',' exist, assume ',' are thousands -> drop commas
    both = t.str.contains(r"\.", regex=True, na=False) & t.str.contains(r",", regex=True, na=False)
    t = t.mask(both, t.str.replace(",", "", regex=False))

    # 4) Else, if only comma and no dot, treat comma as decimal
    comma_only = ~t.str.contains(r"\.", regex=True, na=False) & t.str.contains(r",", regex=True, na=False)
    t = t.mask(comma_only, t.str.replace(",", ".", regex=False))

    # 5) Remove stray percent sign for numeric parse
    t = t.str.replace("%", "", regex=False)

    out = pd.to_numeric(t, errors="coerce")
    out = out.where(~neg, -out)          # apply negative where parentheses were present
    out = out.where(~has_percent, out / 100.0)  # 45% -> 0.45
    return out


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns that look numeric (after cleaning) to floats.
    Uses a simple validity ratio threshold.
    """
    df = df.copy()
    for col in df.columns:
        s = df[col]
        s_num = to_numeric_clean(s)
        if pd.api.types.is_numeric_dtype(s):
            df[col] = pd.to_numeric(s, errors="coerce")
        else:
            valid_ratio = s_num.notna().mean()
            if valid_ratio >= 0.6:  # at least 60% cleanly parsed -> treat as numeric
                df[col] = s_num
    return df


def detect_date_columns(df: pd.DataFrame) -> List[str]:
    """
    Heuristically find columns that are dates.
    We parse without infer_datetime_format to avoid warnings/noise.
    """
    candidates: List[str] = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            candidates.append(col)
            continue
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            # Try tolerant parse; treat as date if most non-nulls parse
            parsed = pd.to_datetime(s, errors="coerce", utc=False)
            if parsed.notna().mean() >= 0.6:
                candidates.append(col)
    return candidates


def summarize_dates(df: pd.DataFrame, date_cols: List[str]) -> Dict[str, Any]:
    """
    For each date column, return min/max/span_days, counts by month & weekday.
    """
    out: Dict[str, Any] = {}
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    for col in date_cols:
        dt = pd.to_datetime(df[col], errors="coerce", utc=False)
        if dt.notna().sum() == 0:
            continue

        dmin = dt.min()
        dmax = dt.max()
        by_month = (
            dt.dt.to_period("M").astype(str).value_counts().sort_index()
        )
        by_weekday = dt.dt.day_name().value_counts()
        # order weekdays nicely
        by_weekday = by_weekday.reindex(weekday_order).dropna().astype(int)

        out[col] = {
            "min": dmin.strftime("%Y-%m-%d"),
            "max": dmax.strftime("%Y-%m-%d"),
            "span_days": int((dmax.normalize() - dmin.normalize()).days),
            "by_month": by_month.to_dict(),
            "by_weekday": by_weekday.to_dict(),
        }
    return out

# -----------------------------
# Helpers: profiling
# -----------------------------
def round2(x):
    if pd.isna(x):
        return None
    if isinstance(x, (int, float)):
        # keep integers as ints, floats rounded
        return int(x) if float(x).is_integer() else round(float(x), 2)
    return x


def numeric_summary(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Describe numeric columns, rounded to 2 decimals.
    """
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


def top5_categories(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """
    For non-numeric columns, return top 5 categories.
    """
    out: Dict[str, Dict[str, int]] = {}
    non_num_cols = df.select_dtypes(exclude=["number", "datetime"]).columns.tolist()
    for col in non_num_cols:
        # Treat as strings for value_counts
        counts = df[col].astype("string").value_counts(dropna=True).head(5)
        if len(counts) > 0:
            out[col] = counts.to_dict()
    return out


def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute shape, columns, nulls, numeric summary, date summary, and top categories.
    """
    # Infer & coerce numeric and date columns first
    df = coerce_numeric_columns(df)
    date_cols = detect_date_columns(df)

    result: Dict[str, Any] = {}
    result["shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
    result["columns"] = df.columns.astype(str).tolist()

    # null counts
    result["nulls"] = {c: int(df[c].isna().sum()) for c in df.columns}

    # numeric summary (rounded)
    result["numeric_summary"] = numeric_summary(df)

    # date summary
    result["date_summary"] = summarize_dates(df, date_cols)

    # top categories for non-numeric/string columns
    result["top5_categories"] = top5_categories(df)

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

    # Basic empty check
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

# -----------------------------
# Local dev (Render uses Start Command)
# -----------------------------
if __name__ == "__main__":
    import uvicorn, os
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("guardrail_api:app", host="0.0.0.0", port=port, reload=True)
