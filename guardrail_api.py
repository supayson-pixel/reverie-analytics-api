from __future__ import annotations

import io
import re
import uuid
from typing import Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------
# FastAPI app + CORS
# -----------------------------
app = FastAPI(title="DAFE API")

# For the demo, keep CORS permissive to avoid surprises between Netlify & Render.
# If you want to lock down later, replace ["*"] with your exact origins.
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
DATASETS: Dict[str, pd.DataFrame] = {}

# -----------------------------
# Helpers (read/clean/coerce)
# -----------------------------

# Currency symbols & spaces; keep digits, minus, and decimal.
_CURRENCY_CHARS = re.compile(r"[^\d\-\.\%\,\(\)\s]")

def _sniff_delimiter(text: str) -> str:
    # Gentle heuristic; prefer comma, then tab, then semicolon/pipe
    candidates = [",", "\t", ";", "|"]
    counts = {c: text.count(c) for c in candidates}
    delim = max(counts, key=counts.get)
    # if no delimiter found, default to comma
    return delim if counts[delim] > 0 else ","

def _read_upload_to_df(file: UploadFile) -> pd.DataFrame:
    name = (file.filename or "").lower()

    if name.endswith((".xlsx", ".xls")):
        # Excel path
        df = pd.read_excel(file.file)
    else:
        # CSV/TXT path
        raw = file.file.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            # Fallback commonly helps with Windows-1252 exports
            text = raw.decode("latin-1")

        delim = _sniff_delimiter(text)
        df = pd.read_csv(io.StringIO(text), sep=delim, engine="python")

    # Normalize column names: strip whitespace & wrapping quotes; drop fully-empty cols
    df.columns = (
        pd.Series(df.columns)
        .astype("string")
        .str.strip()
        .str.replace(r'^["\']|["\']$', "", regex=True)
        .str.replace(r"^Unnamed:.*", "", regex=True)
    )
    df = df.loc[:, df.columns.ne("")]            # remove columns that became empty after cleaning
    df = df.dropna(axis=1, how="all")            # drop all-NaN columns
    # trim whitespace for string-like columns
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype("string").str.strip()

    return df


def to_numeric_clean_series(s: pd.Series) -> pd.Series:
    """
    Clean a series into numeric:
      - remove currency symbols & letters
      - turn '(123.45)' into '-123.45'
      - remove commas
      - handle '%' as fraction (e.g., '35%' -> 0.35)
    """
    s_str = s.astype("string")

    # mark if any % present (apply once for the whole series)
    has_percent = s_str.str.contains("%", na=False).any()

    def _clean_one(txt: pd._libs.missing.NAType | str) -> str:
        if txt is pd.NA or txt is None:
            return ""
        txt = str(txt).strip()
        if not txt:
            return ""
        # turn '(123)' into '-123'
        txt = re.sub(r"^\(([^)]+)\)$", r"-\1", txt)
        # drop currency/letters but keep digits, minus, dot, comma, percent, parentheses already handled
        txt = _CURRENCY_CHARS.sub("", txt)
        # remove commas
        txt = txt.replace(",", "")
        # strip spaces
        txt = txt.replace(" ", "")
        # remove any stray chars except digits, minus, dot
        txt = re.sub(r"[^0-9\.\-]", "", txt)
        return txt

    cleaned = s_str.map(_clean_one)
    out = pd.to_numeric(cleaned, errors="coerce")
    if has_percent:
        # if this column had any percentages, interpret values as percents
        out = out / 100.0
    return out


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to coerce non-numeric, non-datetime columns into numeric if most values are parseable.
    """
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue

        numeric = to_numeric_clean_series(df[col])
        valid_ratio = numeric.notna().mean() if len(numeric) else 0.0
        # convert if at least 60% became numbers
        if valid_ratio >= 0.60:
            df[col] = numeric
    return df


def detect_date_columns(df: pd.DataFrame) -> List[str]:
    date_cols: List[str] = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        s = df[col].astype("string")
        parsed = pd.to_datetime(s, errors="coerce", utc=False)
        ratio = parsed.notna().mean() if len(parsed) else 0.0
        if ratio >= 0.60:
            date_cols.append(col)
    return date_cols


def coerce_date_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)
    return df


def summarize_dates(df: pd.DataFrame, date_cols: List[str]) -> dict:
    """
    Return {col: {min, max, span_days, by_month, by_weekday}}
    """
    result: dict = {}
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    for col in date_cols:
        dt = pd.to_datetime(df[col], errors="coerce", utc=False)
        dt = dt.dropna()
        if dt.empty:
            continue

        dmin = dt.min()
        dmax = dt.max()
        span = int((dmax - dmin).days)

        by_month = (
            dt.dt.to_period("M").astype(str).value_counts().sort_index().astype(int).to_dict()
        )
        by_weekday = dt.dt.day_name().value_counts()
        by_weekday = by_weekday.reindex(weekday_order, fill_value=0).astype(int).to_dict()

        result[col] = {
            "min": dmin.date().isoformat(),
            "max": dmax.date().isoformat(),
            "span_days": span,
            "by_month": by_month,
            "by_weekday": by_weekday,
        }
    return result


def numeric_summary(df: pd.DataFrame) -> dict:
    """
    Describe numeric columns and round floats to 2 decimals.
    """
    out: dict = {}
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    percentiles = [0.25, 0.50, 0.75]
    for col in num_cols:
        desc = df[col].describe(percentiles=percentiles)
        stats = {
            "count": int(desc.get("count", 0)) if not pd.isna(desc.get("count", np.nan)) else 0,
            "mean": float(desc.get("mean")) if not pd.isna(desc.get("mean", np.nan)) else None,
            "std": float(desc.get("std")) if not pd.isna(desc.get("std", np.nan)) else None,
            "min": float(desc.get("min")) if not pd.isna(desc.get("min", np.nan)) else None,
            "25%": float(desc.get("25%")) if not pd.isna(desc.get("25%", np.nan)) else None,
            "50%": float(desc.get("50%")) if not pd.isna(desc.get("50%", np.nan)) else None,
            "75%": float(desc.get("75%")) if not pd.isna(desc.get("75%", np.nan)) else None,
            "max": float(desc.get("max")) if not pd.isna(desc.get("max", np.nan)) else None,
        }
        # round floats
        for k, v in stats.items():
            if isinstance(v, float):
                stats[k] = round(v, 2)
        out[col] = stats
    return out


def top5_categories(df: pd.DataFrame, exclude_cols: List[str] | None = None) -> dict:
    """
    Top 5 categories for each non-numeric, non-datetime column, excluding any in exclude_cols.
    """
    exclude = set(exclude_cols or [])
    out: dict = {}
    non_num_cols = df.select_dtypes(exclude=["number", "datetime"]).columns.tolist()
    for col in non_num_cols:
        if col in exclude:
            continue
        counts = df[col].astype("string").value_counts(dropna=True).head(5)
        if len(counts) > 0:
            out[col] = counts.to_dict()
    return out


def profile_dataframe(df: pd.DataFrame) -> dict:
    # detect/parse dates first, then coerce numerics
    date_cols = detect_date_columns(df)
    df = coerce_date_columns(df, date_cols)
    df = coerce_numeric_columns(df)

    # Build response
    res: dict = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": [str(c) for c in df.columns],
        "nulls": {str(c): int(df[c].isna().sum()) for c in df.columns},
        "numeric_summary": numeric_summary(df),
        "date_summary": summarize_dates(df, date_cols),
        "top5_categories": top5_categories(df, exclude_cols=date_cols),
    }
    return res


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"message": "DAFE API. Try /health or /analytics/*"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analytics/upload")
async def upload(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided.")
    try:
        df = _read_upload_to_df(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {e}")

    dsid = str(uuid.uuid4())
    DATASETS[dsid] = df

    return {
        "dataset_id": dsid,
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": [str(c) for c in df.columns],
    }

@app.get("/analytics/profile")
def profile(dataset_id: str):
    if dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="dataset_id not found")
    df = DATASETS[dataset_id]
    return profile_dataframe(df)
