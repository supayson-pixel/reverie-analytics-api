from __future__ import annotations

import os
import io
import uuid
import pathlib
from typing import Dict, Any

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# --------- Config ---------
# Where to store uploads (Render allows writing to /tmp)
UPLOAD_DIR = pathlib.Path(os.getenv("UPLOAD_DIR", "/tmp/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Allowed extensions (front-end accepts .csv, .xlsx, .xls)
ALLOWED_EXTS = {".csv", ".xlsx", ".xls"}

# Try to support Excel; if openpyxl isn't installed, we’ll reject Excel files gracefully.
try:
    import openpyxl  # noqa: F401
    HAS_XLSX = True
except Exception:
    HAS_XLSX = False

# --------- App ---------
app = FastAPI(
    title="Reverie Analytics API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS: include BOTH www and apex, plus your preview domain.
ALLOWED_ORIGINS = [
    "https://www.reveriesun.com",
    "https://reveriesun.com",
    "https://inspiring-tarsier-97b2c7.netlify.app",  # preview
    # Local dev (optional):
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=86400,
)


# --------- Helpers ---------
def _dataset_path(dataset_id: str) -> pathlib.Path:
    return UPLOAD_DIR / f"{dataset_id}.bin"

def _save_upload(file: UploadFile) -> str:
    ext = pathlib.Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=415, detail=f"Unsupported file type {ext}. Use CSV/XLSX.")

    if ext in {".xlsx", ".xls"} and not HAS_XLSX:
        raise HTTPException(
            status_code=415,
            detail="Excel files require 'openpyxl' on the server. Install it or upload CSV."
        )

    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    dataset_id = uuid.uuid4().hex
    # Store a tiny header with extension so we know how to read later.
    # Format: b"EXT:<ext>\n" + payload
    header = f"EXT:{ext}\n".encode("utf-8")
    _dataset_path(dataset_id).write_bytes(header + data)
    return dataset_id

def _load_dataframe(dataset_id: str) -> pd.DataFrame:
    path = _dataset_path(dataset_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="dataset_id not found")

    blob = path.read_bytes()
    try:
        header, payload = blob.split(b"\n", 1)
    except ValueError:
        raise HTTPException(status_code=400, detail="Corrupt dataset file")

    header_text = header.decode("utf-8", errors="ignore")
    if not header_text.startswith("EXT:"):
        raise HTTPException(status_code=400, detail="Missing file header")
    ext = header_text[4:].strip().lower()

    bio = io.BytesIO(payload)
    if ext == ".csv":
        df = pd.read_csv(bio)
    elif ext in {".xlsx", ".xls"}:
        if not HAS_XLSX:
            raise HTTPException(
                status_code=415,
                detail="Excel files require 'openpyxl' on the server. Install it or upload CSV."
            )
        df = pd.read_excel(bio)  # uses openpyxl when available
    else:
        raise HTTPException(status_code=415, detail=f"Unsupported extension {ext}")

    return df


def _profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    # Shape & columns
    shape = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
    columns = list(map(str, df.columns))

    # Dtypes (as strings for readability)
    dtypes = {str(c): str(dt) for c, dt in df.dtypes.items()}

    # Null counts
    nulls = {str(c): int(df[c].isna().sum()) for c in df.columns}

    # Numeric summary (mimic pandas describe)
    numeric = df.select_dtypes(include="number")
    numeric_summary: Dict[str, Dict[str, Any]] = {}
    if not numeric.empty:
        desc = numeric.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
        # Reorganize to {col: {count:..., mean:..., ...}}
        wanted = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        for col in desc:
            numeric_summary[col] = {k: _maybe_round(desc[col].get(k)) for k in wanted}

    # Top 5 categories for non-numeric columns
    top5_categories: Dict[str, Dict[str, int]] = {}
    cat_df = df.select_dtypes(exclude="number")
    for col in cat_df.columns:
        vc = cat_df[col].astype(str).value_counts().head(5)
        top5_categories[str(col)] = {str(k): int(v) for k, v in vc.items()}

    return {
        "shape": shape,
        "columns": columns,
        "dtypes": dtypes,
        "nulls": nulls,
        "numeric_summary": numeric_summary,
        "top5_categories": top5_categories,
    }

def _maybe_round(x):
    if x is None:
        return None
    try:
        f = float(x)
        # round to 6 to preserve fidelity; front-end re-formats
        return round(f, 6)
    except Exception:
        return x


# --------- Routes ---------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analytics/upload")
def analytics_upload(file: UploadFile = File(...)):
    """
    Save the uploaded file and return a dataset_id.
    """
    dataset_id = _save_upload(file)
    return {"dataset_id": dataset_id, "filename": file.filename}

@app.get("/analytics/profile")
def analytics_profile(dataset_id: str = Query(..., description="dataset_id returned by /analytics/upload")):
    """
    Load the dataset by id and return a profiling summary compatible with lab.html.
    """
    df = _load_dataframe(dataset_id)
    profile = _profile_dataframe(df)
    return profile


# For running locally:  uvicorn guardrail_api:app --reload --port 8001
# (Render will run with its own command)
