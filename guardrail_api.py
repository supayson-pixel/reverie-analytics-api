# guardrail_api.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import uuid
import os
from typing import Dict, Any

app = FastAPI(title="DAFE API")

# ---- CORS: allow your live domains + preview subdomain ----
ALLOWED_ORIGINS = [
    "https://www.reveriesun.com",
    "https://reveriesun.com",
    # Netlify preview / production subdomain(s). Add others as needed.
    "https://inspiring-tarsier-97b2c7.netlify.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,   # or ["*"] during dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- simple in-memory registry for uploaded files (OK for demo) ----
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
REGISTRY: Dict[str, str] = {}  # dataset_id -> file_path

# ---------- health ----------
@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- upload ----------
@app.post("/analytics/upload")
async def upload_dataset(file: UploadFile = File(...)) -> Dict[str, Any]:
    # Accept csv/xlsx/xls only (remove txt unless you want to parse it)
    name = file.filename or ""
    ext = name.split(".")[-1].lower()
    if ext not in ("csv", "xlsx", "xls"):
        raise HTTPException(status_code=400, detail="Please upload a CSV or Excel file.")

    dataset_id = str(uuid.uuid4())
    dest_path = os.path.join(UPLOAD_DIR, f"{dataset_id}.{ext}")

    with open(dest_path, "wb") as out:
        out.write(await file.read())

    REGISTRY[dataset_id] = dest_path
    return {"dataset_id": dataset_id, "filename": name}

# ---------- profile ----------
@app.get("/analytics/profile")
def analytics_profile(dataset_id: str) -> Dict[str, Any]:
    path = REGISTRY.get(dataset_id)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="dataset_id not found (or expired).")

    ext = path.split(".")[-1].lower()
    try:
        if ext == "csv":
            df = pd.read_csv(path)
        else:
            # xlsx / xls
            df = pd.read_excel(path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    # summary
    shape = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
    columns = list(map(str, df.columns))

    # nulls
    nulls = {str(c): int(v) for c, v in df.isna().sum().items()}

    # numeric summary
    numeric_cols = df.select_dtypes(include="number")
    numeric_summary: Dict[str, Dict[str, Any]] = {}
    if not numeric_cols.empty:
        desc = numeric_cols.describe().to_dict()  # keys: count, mean, std, min, 25%, 50%, 75%, max
        # flip orientation to {col: {metric: value}}
        for metric, per_col in desc.items():
            for col, val in per_col.items():
                numeric_summary.setdefault(str(col), {})[metric] = None if pd.isna(val) else float(val)

    # top-5 categories for object-like columns
    top5_categories: Dict[str, Dict[str, int]] = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        vc = df[col].astype("string").value_counts().head(5)
        top5_categories[str(col)] = {str(k): int(v) for k, v in vc.items()}

    return {
        "shape": shape,
        "columns": columns,
        "nulls": nulls,
        "numeric_summary": numeric_summary,
        "top5_categories": top5_categories,
    }

# ---------- optional root to avoid 'Not Found' at '/' ----------
@app.get("/")
def root():
    return {"message": "DAFE API. Try /health or /analytics/*"}
