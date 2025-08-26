# main.py
import uuid
from typing import Dict, Any

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from utils_io import load_table

app = FastAPI(title="Reverie Analytics API", version="1.0.0")

ALLOWED_ORIGINS = [
    "https://reveriesun.com",
    "https://www.reveriesun.com",
    "https://*.netlify.app",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

# simple in-memory store (swap for S3/db later)
DATASETS: Dict[str, pd.DataFrame] = {}

@app.get("/health")
def health(): return {"ok": True}

def _safe_round(x, nd=3):
    try:
        return None if pd.isna(x) else round(float(x), nd)
    except Exception:
        return None

def compute_numeric_summary(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    out = {}
    for col in df.columns:
        if not is_numeric_dtype(df[col]):
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            out[col] = {"count": 0, "mean": None, "std": None, "min": None,
                        "25%": None, "50%": None, "75%": None, "max": None}
            continue
        q = s.quantile([0.25, 0.5, 0.75])
        out[col] = {
            "count": int(s.count()),
            "mean": _safe_round(s.mean()),
            "std": _safe_round(s.std(ddof=1)),
            "min": _safe_round(s.min()),
            "25%": _safe_round(q.get(0.25)),
            "50%": _safe_round(q.get(0.5)),
            "75%": _safe_round(q.get(0.75)),
            "max": _safe_round(s.max()),
        }
    return out

def compute_top_categories(df: pd.DataFrame, k: int = 5) -> Dict[str, Dict[str, int]]:
    out = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        vc = df[col].astype(str).fillna("null").value_counts().head(k)
        out[col] = {str(idx): int(val) for idx, val in vc.items()}
    return out

def compute_date_summary(df: pd.DataFrame) -> Dict[str, Any]:
    out = {}
    for col in df.columns:
        s = df[col]
        if is_datetime64_any_dtype(s):
            sdt = pd.to_datetime(s, errors="coerce")
        else:
            # try to parse strings
            sdt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
        sdt = sdt.dropna()
        if sdt.empty:
            continue
        by_month = sdt.dt.to_period("M").value_counts().sort_index()
        by_weekday = sdt.dt.day_name().value_counts()
        out[col] = {
            "min": sdt.min().isoformat(),
            "max": sdt.max().isoformat(),
            "span_days": int((sdt.max() - sdt.min()).days),
            "by_month": {str(p): int(c) for p, c in by_month.items()},
            "by_weekday": {str(idx): int(val) for idx, val in by_weekday.items()},
        }
    return out

def coerce_obvious_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            cleaned = df[col].astype(str).str.replace(r"[,\$]", "", regex=True)
            num = pd.to_numeric(cleaned, errors="coerce")
            if num.notna().sum() >= max(3, int(0.6 * len(df[col]))):
                df[col] = num
    return df

def profile_dataframe(df: pd.DataFrame, preview_rows: int = 5) -> Dict[str, Any]:
    df = coerce_obvious_numeric(df.copy())
    shape = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
    columns = [str(c) for c in df.columns]
    nulls = {str(c): int(df[c].isna().sum()) for c in df.columns}
    numeric_summary = compute_numeric_summary(df)
    date_summary = compute_date_summary(df)
    top5_categories = compute_top_categories(df)

    preview = df.head(preview_rows).copy()
    for c in preview.columns:
        if is_datetime64_any_dtype(preview[c]):
            preview[c] = preview[c].astype("datetime64[ns]").astype(str)
        else:
            preview[c] = preview[c].astype(object).where(preview[c].notna(), None)

    return {
        "shape": shape,
        "columns": columns,
        "nulls": nulls,
        "numeric_summary": numeric_summary,
        "date_summary": date_summary,
        "top5_categories": top5_categories,
        "preview": preview.to_dict(orient="records"),
    }

@app.post("/analytics/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    try:
        df = load_table(content, file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")
    ds_id = str(uuid.uuid4())
    DATASETS[ds_id] = df
    return {"dataset_id": ds_id, "rows": int(df.shape[0]), "columns": int(df.shape[1]), "alias": None}

@app.get("/analytics/profile")
def profile(dataset_id: str):
    if dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="dataset not found")
    return profile_dataframe(DATASETS[dataset_id])
