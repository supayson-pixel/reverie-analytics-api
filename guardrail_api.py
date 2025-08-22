# FastAPI analytics microservice
# Save as: guardrail_api.py

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import os, io, uuid, pathlib

# ----- config -----
UPLOAD_DIR = pathlib.Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Reverie Analytics API", version="0.1.1")

# Allow your site to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.reveriesun.com",
        "https://reveriesun.netlify.app",
        "https://inspiring-tarsier-97b2c7.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- Helpers ----------
def _load_df(path: pathlib.Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in [".csv", ".txt"]:
        return pd.read_csv(path)
    if suf in [".xlsx", ".xls"]:
        return pd.read_excel(path)  # requires openpyxl
    raise ValueError(f"Unsupported file type: {suf}")

def _profile(df: pd.DataFrame) -> dict:
    out = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": list(df.columns.map(str)),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "nulls": {c: int(n) for c, n in df.isna().sum().items()},
        "numeric_summary": {},
        "top5_categories": {},
    }

    # numeric columns
    num_cols = df.select_dtypes(include="number").columns
    for c in num_cols:
        s = df[c]
        desc = s.describe(percentiles=[.25, .5, .75]).to_dict()
        out["numeric_summary"][str(c)] = {
            k: (float(v) if pd.notna(v) else None) for k, v in desc.items()
        }

    # categorical columns
    cat_cols = df.select_dtypes(exclude="number").columns
    for c in cat_cols:
        vc = df[c].astype(str).value_counts(dropna=True).head(5).to_dict()
        out["top5_categories"][str(c)] = {str(k): int(v) for k, v in vc.items()}

    return out

# ---------- Routes ----------
@app.post("/analytics/upload")
async def analytics_upload(file: UploadFile = File(...)):
    # generate id, keep original extension
    ext = pathlib.Path(file.filename).suffix or ".csv"
    dataset_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{dataset_id}{ext}"

    # save to disk
    content = await file.read()
    dest.write_bytes(content)

    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "bytes": len(content),
        "path": str(dest),
    }

@app.get("/analytics/profile")
def analytics_profile(dataset_id: str = Query(..., min_length=6, max_length=64)):
    # find the saved file (id + any allowed extension)
    matches = list(UPLOAD_DIR.glob(f"{dataset_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="dataset_id not found")

    try:
        df = _load_df(matches[0])
        return JSONResponse(_profile(df))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Profile error: {e}")
