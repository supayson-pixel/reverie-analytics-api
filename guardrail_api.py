# guardrail_api.py
# FastAPI service for Reverie Co: upload a dataset, then profile it.
# Endpoints:
#   GET  /health
#   POST /analytics/upload        -> {"dataset_id": "..."}
#   GET  /analytics/profile?id=…  -> summary JSON

import io
import os
import uuid
import csv
from typing import Dict, Any

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------
# App & CORS
# -----------------------------
app = FastAPI(title="Reverie Analytics API", version="1.0.0")

# Use explicit origins for production. If you ever hit CORS issues while testing,
# you can set allow_origins=["*"] temporarily.
ALLOW_ORIGINS = [
    "https://www.reveriesun.com",
    "https://reveriesun.com",
    "https://inspiring-tarsier-97b2c7.netlify.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,  # or ["*"] for widest compatibility
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Simple in-memory object store
# (dataset_id -> {"bytes": ..., "filename": ...})
# -----------------------------
DATA_STORE: Dict[str, Dict[str, Any]] = {}


# -----------------------------
# Helpers
# -----------------------------
def read_csv_robust(raw_bytes: bytes) -> pd.DataFrame:
    """
    Try hard to parse a CSV regardless of delimiter/encoding quirks.
    Handles , ; \t | and UTF-8 BOM, and repairs the common
    "one big column with comma-separated header" case.
    """
    df = None

    # 1) Pandas' autodetect (requires engine='python' for sep=None)
    try:
        df = pd.read_csv(
            io.BytesIO(raw_bytes),
            sep=None,
            engine="python",
            encoding="utf-8-sig",
        )
    except Exception:
        df = None

    # 2) If still one column or failed, use csv.Sniffer
    if df is None or df.shape[1] == 1:
        try:
            sample = raw_bytes[:2048].decode("utf-8", "ignore")
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            df2 = pd.read_csv(
                io.BytesIO(raw_bytes),
                sep=dialect.delimiter,
                encoding="utf-8-sig",
            )
            if df2.shape[1] > 1:
                df = df2
        except Exception:
            pass

    # 3) Explicit fallbacks
    if df is None or df.shape[1] == 1:
        for sep in [",", ";", "\t", "|"]:
            try:
                df2 = pd.read_csv(
                    io.BytesIO(raw_bytes),
                    sep=sep,
                    encoding="utf-8-sig",
                )
                if df2.shape[1] > 1:
                    df = df2
                    break
            except Exception:
                continue

    # 4) Last-resort: header contains commas; split manually
    if df is not None and df.shape[1] == 1 and "," in str(df.columns[0]):
        cols = [c.strip() for c in str(df.columns[0]).split(",")]
        parts = df.iloc[:, 0].astype(str).str.split(",", expand=True)
        parts.columns = cols
        df = parts

    if df is None:
        raise ValueError("Could not parse CSV with any strategy.")
    return df


def load_dataframe_from_bytes(filename: str, raw_bytes: bytes) -> pd.DataFrame:
    """
    Load a DataFrame from uploaded bytes based on extension.
    Supports CSV/TXT and Excel (XLSX/XLS).
    """
    name = (filename or "").lower()
    if name.endswith((".xlsx", ".xls")):
        # Requires 'openpyxl' for .xlsx (add to requirements if not present)
        try:
            return pd.read_excel(io.BytesIO(raw_bytes), engine="openpyxl")
        except Exception:  # fallback for older XLS
            return pd.read_excel(io.BytesIO(raw_bytes))
    # Treat txt as CSV
    return read_csv_robust(raw_bytes)


def numeric_summary(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Return describe() for numeric columns, with a stable JSON shape:
      { col: {count, mean, std, min, 25%, 50%, 75%, max}, ... }
    """
    num = df.select_dtypes(include="number")
    if num.empty:
        return {}
    desc = num.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
    # 'describe().to_dict()' already gives nested {stat -> {col -> value}} or vice versa
    # We want {col: {stat: value}}
    # Pandas uses orientation 'index' vs 'columns' depending on version; normalize here:
    out: Dict[str, Dict[str, float]] = {}
    # If keys look like stats (e.g., 'count', 'mean'), re-pivot
    if {"count", "mean", "std", "min", "25%", "50%", "75%", "max"} <= set(desc.keys()):
        for stat, col_map in desc.items():
            for col, val in col_map.items():
                out.setdefault(col, {})[stat] = None if pd.isna(val) else float(val)
    else:
        # Already in {col: {stat: value}} form
        for col, stat_map in desc.items():
            out[col] = {k: None if pd.isna(v) else float(v) for k, v in stat_map.items()}
    return out


def top5_categories(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """
    For each object/string column, return top 5 value counts.
    """
    out: Dict[str, Dict[str, int]] = {}
    obj = df.select_dtypes(include=["object", "string"])
    for col in obj.columns:
        vc = obj[col].astype(str).value_counts(dropna=False).head(5)
        out[col] = {str(k): int(v) for k, v in vc.items()}
    return out


def null_counts(df: pd.DataFrame) -> Dict[str, int]:
    return {c: int(n) for c, n in df.isna().sum().to_dict().items()}


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"message": "Reverie Analytics API. Try /health or /analytics/*"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analytics/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Accept a file (CSV/TXT/XLS/XLSX). Store bytes in memory and return dataset_id.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file provided.")

    filename = file.filename or ""
    ext = (filename.split(".")[-1] or "").lower()
    if ext not in {"csv", "xlsx", "xls", "txt"}:
        raise HTTPException(status_code=415, detail="Unsupported file type.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file.")

    dataset_id = str(uuid.uuid4())
    DATA_STORE[dataset_id] = {"bytes": content, "filename": filename}

    return {"dataset_id": dataset_id}


@app.get("/analytics/profile")
def profile_dataset(dataset_id: str = Query(..., alias="dataset_id")):
    """
    Load stored bytes into a DataFrame and return a compact profile:
      - shape (rows, columns)
      - columns
      - nulls by column
      - numeric summary
      - top 5 categories per non-numeric column
    """
    item = DATA_STORE.get(dataset_id)
    if not item:
        raise HTTPException(status_code=404, detail="dataset_id not found")

    try:
        df = load_dataframe_from_bytes(item["filename"], item["bytes"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read dataset: {e}")

    # Build response
    try:
        shape = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
        cols = [str(c) for c in df.columns]
        nulls = null_counts(df)
        nums = numeric_summary(df)
        cats = top5_categories(df)

        return {
            "shape": shape,
            "columns": cols,
            "nulls": nulls,
            "numeric_summary": nums,
            "top5_categories": cats,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profiling failed: {e}")


# -----------------------------
# Local dev entrypoint (optional)
# Render/other PaaS can run with:
#   uvicorn guardrail_api:app --host 0.0.0.0 --port $PORT
# -----------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("guardrail_api:app", host="0.0.0.0", port=port, reload=True)
