# guardrail_api.py
# FastAPI service for Reverie Co: upload a dataset, then profile it.

import io
import os
import uuid
import csv
from typing import Dict, Any, Optional, List

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------
# App & CORS
# -----------------------------
app = FastAPI(title="Reverie Analytics API", version="1.0.1")

ALLOW_ORIGINS = [
    "https://www.reveriesun.com",
    "https://reveriesun.com",
    "https://inspiring-tarsier-97b2c7.netlify.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,  # set to ["*"] temporarily if you ever need to debug CORS
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Simple in-memory store
# -----------------------------
DATA_STORE: Dict[str, Dict[str, Any]] = {}

# -----------------------------
# CSV helpers
# -----------------------------
CANDIDATE_DELIMS: List[str] = [",", ";", "\t", "|"]

def _repair_packed_csv(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Repair the common 'packed single column' (or single+empty column) CSV case:
    - header looks like 'a,b,c,d' (contains a delimiter)
    - df has 1 or 2 columns (second may be all empty 'Unnamed: 1')
    Returns a fixed DataFrame or None if no repair applied.
    """
    if df is None or df.shape[1] > 2:
        return None

    hdr = str(df.columns[0]).strip().strip('"').strip("'")
    # If the header doesn't contain a likely delimiter, nothing to do.
    delims_in_header = [d for d in CANDIDATE_DELIMS if d in hdr]
    if not delims_in_header:
        return None

    # If there are 2 columns and the second is NOT all empty, don't guess.
    if df.shape[1] == 2 and not df.iloc[:, 1].isna().all():
        return None

    # Choose a delimiter: whichever appears most often in the header
    delim = max(delims_in_header, key=lambda d: hdr.count(d))
    cols = [c.strip().strip('"').strip("'") for c in hdr.split(delim)]

    # Split the single "packed" column into parts
    s = df.iloc[:, 0].astype(str).str.rstrip(delim)
    parts = s.str.split(delim, expand=True)

    # If we got an extra empty column at the end, drop it
    if parts.shape[1] > 1 and parts.iloc[:, -1].replace("", pd.NA).isna().all():
        parts = parts.iloc[:, :-1]

    # Match header length to parts
    if len(cols) < parts.shape[1]:
        # pad headers
        cols = cols + [f"col_{i}" for i in range(len(cols), parts.shape[1])]
    elif len(cols) > parts.shape[1]:
        cols = cols[:parts.shape[1]]

    parts.columns = cols
    return parts


def read_csv_robust(raw_bytes: bytes) -> pd.DataFrame:
    """
    Try hard to parse a CSV regardless of delimiter/encoding quirks, including
    the 'packed one column' case with a trailing delimiter.
    """
    df: Optional[pd.DataFrame] = None

    # 1) Pandas autodetect
    try:
        df = pd.read_csv(
            io.BytesIO(raw_bytes),
            sep=None,
            engine="python",
            encoding="utf-8-sig",
        )
    except Exception:
        df = None

    # 2) Sniffer
    if df is None or df.shape[1] == 1:
        try:
            sample = raw_bytes[:4096].decode("utf-8", "ignore")
            dialect = csv.Sniffer().sniff(sample, delimiters="".join(CANDIDATE_DELIMS))
            df2 = pd.read_csv(
                io.BytesIO(raw_bytes),
                sep=dialect.delimiter,
                encoding="utf-8-sig",
            )
            if df2.shape[1] > 1:
                df = df2
        except Exception:
            pass

    # 3) Fallback: explicit candidates
    if df is None or df.shape[1] == 1:
        for sep in CANDIDATE_DELIMS:
            try:
                df2 = pd.read_csv(io.BytesIO(raw_bytes), sep=sep, encoding="utf-8-sig")
                if df2.shape[1] > 1:
                    df = df2
                    break
            except Exception:
                continue

    # 4) Repair the 'packed header + lines' case (1 column) or (1+empty column)
    if df is not None:
        fixed = _repair_packed_csv(df)
        if fixed is not None and not fixed.empty:
            df = fixed

    if df is None:
        raise ValueError("Could not parse CSV with any strategy.")
    return df


def load_dataframe_from_bytes(filename: str, raw_bytes: bytes) -> pd.DataFrame:
    name = (filename or "").lower()
    if name.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(io.BytesIO(raw_bytes), engine="openpyxl")
        except Exception:
            # fallback for legacy xls
            return pd.read_excel(io.BytesIO(raw_bytes))
    # Treat .txt like CSV
    return read_csv_robust(raw_bytes)


def numeric_summary(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    num = df.select_dtypes(include="number")
    if num.empty:
        return {}
    desc = num.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
    wanted = {"count", "mean", "std", "min", "25%", "50%", "75%", "max"}
    out: Dict[str, Dict[str, float]] = {}

    if wanted <= set(desc.keys()):
        # {stat: {col: val}}
        for stat, col_map in desc.items():
            for col, val in col_map.items():
                out.setdefault(str(col), {})[stat] = None if pd.isna(val) else float(val)
    else:
        # {col: {stat: val}}
        for col, stat_map in desc.items():
            out[str(col)] = {k: None if pd.isna(v) else float(v) for k, v in stat_map.items()}
    return out


def top5_categories(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    obj = df.select_dtypes(include=["object", "string"])
    for col in obj.columns:
        vc = obj[col].astype(str).value_counts(dropna=False).head(5)
        out[str(col)] = {str(k): int(v) for k, v in vc.items()}
    return out


def null_counts(df: pd.DataFrame) -> Dict[str, int]:
    return {str(c): int(n) for c, n in df.isna().sum().to_dict().items()}

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
    item = DATA_STORE.get(dataset_id)
    if not item:
        raise HTTPException(status_code=404, detail="dataset_id not found")
    try:
        df = load_dataframe_from_bytes(item["filename"], item["bytes"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read dataset: {e}")

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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("guardrail_api:app", host="0.0.0.0", port=port, reload=True)
