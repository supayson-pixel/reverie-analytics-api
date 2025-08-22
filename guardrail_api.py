# guardrail_api.py
# Reverie Analytics API – clean + profile uploaded CSV/XLSX
# v0.3.1

from __future__ import annotations

import io
import uuid
import csv
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
    version="0.3.1",
)

_CORS_ORIGINS = [
    "https://www.reveriesun.com",
    "https://reveriesun.com",
    "https://reveriesun.netlify.app",
    "https://inspiring-tarsier-97b2c7.netlify.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_origin_regex=r"https://.*\.netlify\.app",
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
    """Heuristic: one column has most data, others nearly empty (or only one column)."""
    if df is None or df.shape[0] == 0:
        return True
    if df.shape[1] == 1:
        return True
    nn = df.notna().mean()
    return (nn.max() > 0.8) and ((nn < 0.2).sum() >= max(1, df.shape[1] - 1))


def _strip_quotes_everywhere(df: pd.DataFrame) -> pd.DataFrame:
    """Strip surrounding single/double quotes in headers & string cells."""
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
    """
    Try several parsing strategies:
      1) pandas sniffing (sep=None, engine='python')
      2) force common separators: ',', '\t', ';', '|'
      3) row-quoted CSVs: quoting disabled so commas split, then strip quotes
    """
    # primary attempt: sniff
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
        if not _looks_broken(df):
            return _strip_quotes_everywhere(df)
    except Exception:
        pass

    # common separators
    for sep in [",", "\t", ";", "|"]:
        try:
            df2 = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
            if not _looks_broken(df2):
                return _strip_quotes_everywhere(df2)
        except Exception:
            continue

    # row-quoted CSV (whole line like "a,b,c")
    # disable quoting to treat quotes as literal, then strip them
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

    # last resort: return whatever we got via sniff; it might be 1 col with the whole line
    df_fallback = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    return _strip_quotes_everywhere(df_fallback)


def _read_dataframe_from_upload(file: UploadFile) -> pd.DataFrame:
    name = (file.filename or "").lower()

    if name.endswith((".xlsx", ".xls")):
        data = file.file.read()
        df = pd.read_excel(io.BytesIO(data))
        return _strip_quotes_everywhere(df)

    # CSV/TXT: decode robustly and remove BOM if present
    raw = file.file.read()
    try:
        text = raw.decode("utf-8-sig", errors="replace")  # handles BOM
    except Exception:
        text = raw.decode("utf-8", errors="replace")

    return _read_csv_text(text)

# -----------------------------
# Cleaning / coercions
# -----------------------------
def to_numeric_clean(s: pd.Series) -> pd.Series:
    """
    Convert strings like '$1,234.56', '45%', '1 234', '(123)' to floats.
    Keeps NaN where conversion fails. Vectorized.
    """
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    t = s.astype("string")

    has_percent = t.str.contains("%", regex=False, na=False)

    # remove everything except digits, signs, decimal sep, percent and parentheses
    t = t.str.replace(r"[^0-9\-\.,%()]", "", regex=True)

    # parentheses negatives
    neg = t.str.contains(r"\(", regex=True, na=False) & t.str.contains(r"\)", regex=True, na=False)
    t = t.str.replace(r"[()]", "", regex=True)

    both = t.str.contains(r"\.", regex=True, na=False) & t.str.contains(r",", regex=True, na=False)
    t = t.mask(both, t.str.replace(",", "", regex=False))  # 1,234.56 -> 1234.56

    comma_only = ~t.str.contains(r"\.", regex=True, na=False) & t.str.contains(r",", regex=True, na=False)
    t = t.mask(comma_only, t.str.replace(",", ".", regex=False))  # 123,45 -> 123.45

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


def top5_categories(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    non_num_cols = df.select_dtypes(exclude=["number", "datetime"]).columns.tolist()
    for col in non_num_cols:
        counts = df[col].astype("string").value_counts(dropna=True).head(5)
        if len(counts) > 0:
            out[col] = counts.to_dict()
    return out


def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    df = coerce_numeric_columns(df)
    date_cols = detect_date_columns(df)

    result: Dict[str, Any] = {}
    result["shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
    result["columns"] = df.columns.astype(str).tolist()
    result["nulls"] = {c: int(df[c].isna().sum()) for c in df.columns}
    result["numeric_summary"] = numeric_summary(df)
    result["date_summary"] = summarize_dates(df, date_cols)
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
# Local dev
# -----------------------------
if __name__ == "__main__":
    import uvicorn, os
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("guardrail_api:app", host="0.0.0.0", port=port, reload=True)
