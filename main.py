# main.py
# DAFE Sprint-1 backend — FastAPI on Render
# Endpoints:
#   GET  /health
#   POST /analytics/upload
#   GET  /analytics/profile
#   POST /analytics/aggregate
#   POST /analytics/report/pdf
#   POST /analytics/analyze

import io
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from pydantic import BaseModel

# --- Optional utils shim ------------------------------------------------------
# If your repo has helpers (e.g., utils_io.py), we’ll import them but keep safe fallbacks.
try:
    from utils_io import load_table as _load_table  # type: ignore
except Exception:
    _load_table = None

# --- App ----------------------------------------------------------------------
app = FastAPI(title="Reverie Analytics API", version="1.0")

# Allow reveriesun.com, Netlify previews, localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://reveriesun.com",
        "https://www.reveriesun.com",
        "http://localhost",
        "http://localhost:3000",
    ],
    allow_origin_regex=r"https://.*\.netlify\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory dataset store (demo scope)
DATASETS: Dict[str, Dict[str, Any]] = {}

# --- Helpers ------------------------------------------------------------------
def _infer_delimiter(sample: bytes) -> str:
    text = sample.decode("utf-8", errors="ignore")
    return "," if text.count(",") >= text.count(";") else ";"

def _load_dataframe_from_upload(file: UploadFile) -> pd.DataFrame:
    """Robust CSV/XLSX loader with common quirks handled."""
    if _load_table is not None:
        # Use project’s robust loader if available
        return _load_table(file)  # type: ignore

    name = (file.filename or "").lower()
    raw = file.file.read()
    file.file.seek(0)

    if name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
        # Excel “single column with comma-joined cells” quirk
        if df.shape[1] == 1 and isinstance(df.columns[0], str) and "," in df.columns[0]:
            df = pd.read_csv(io.StringIO(raw.decode("utf-8", errors="ignore")))
        return df

    # default: CSV
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        # Try with detected delimiter
        delim = _infer_delimiter(raw[:4096])
        df = pd.read_csv(io.StringIO(raw.decode("utf-8", errors="ignore")), sep=delim)

    return df

def _coerce_common_types(df: pd.DataFrame) -> pd.DataFrame:
    """Light cleaning: strip currency, coerce numerics and dates where obvious."""
    out = df.copy()
    # Strip currency symbols and thousand separators in object columns that look numeric
    for c in out.select_dtypes(include=["object"]).columns:
        s = out[c].astype(str).str.replace(r"[\$,]", "", regex=True)
        # Try numeric, fallback to original
        coerced = pd.to_numeric(s, errors="ignore")
        if pd.api.types.is_numeric_dtype(coerced):
            out[c] = coerced
    # Date coercion: try parse if many ISO-ish strings
    for c in out.columns:
        if out[c].dtype == "object":
            sample = out[c].dropna().astype(str).head(20)
            if len(sample) and sample.str.contains(r"\d{4}-\d{1,2}-\d{1,2}", regex=True).mean() > 0.6:
                out[c] = pd.to_datetime(out[c], errors="coerce")
    return out

def _get_df(dataset_id: str) -> pd.DataFrame:
    if dataset_id not in DATASETS:
        raise HTTPException(404, "dataset_id not found")
    return DATASETS[dataset_id]["df"]

def _top_categories(df: pd.DataFrame, k: int = 10) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    for c in df.select_dtypes(exclude=[np.number, "datetime64[ns]", "datetime64[ns, UTC]"]).columns:
        vc = df[c].value_counts(dropna=False).head(k)
        result[c] = [{"value": None if pd.isna(idx) else idx, "count": int(v)} for idx, v in vc.items()]
    return result

def _numeric_summary(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return {}
    desc = num.describe().round(3).to_dict()
    return {k: {sk: (float(sv) if pd.notna(sv) else None) for sk, sv in v.items()} for k, v in desc.items()}

def _date_summary(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            s = df[c].dropna()
            if len(s):
                out[c] = {
                    "min": s.min().isoformat(),
                    "max": s.max().isoformat(),
                    "span_days": int((s.max() - s.min()).days),
                    "by_weekday": s.dt.day_name().value_counts().to_dict(),
                    "by_month": s.dt.month_name().value_counts().to_dict(),
                }
    return out

def _chart_payload(title: str, labels: List[Any], series: Dict[str, List[float]]) -> Dict[str, Any]:
    def _fmt(v):
        try:
            return v.isoformat()
        except Exception:
            return v
    return {
        "title": title,
        "labels": [_fmt(v) for v in labels],
        "datasets": [{"label": k, "data": v} for k, v in series.items()],
    }

# --- Schemas ------------------------------------------------------------------
class AggregateSpec(BaseModel):
    dataset_id: str
    x: Optional[str] = None
    y: Optional[str] = None
    agg: Literal["sum", "count", "mean", "median", "min", "max"] = "sum"
    split_by: Optional[str] = None
    time_grain: Optional[Literal["D", "W", "M", "Q", "Y"]] = None
    type: Literal["bar", "stacked", "line", "area", "donut"] = "bar"
    limit: Optional[int] = 1000

class ReportSlide(BaseModel):
    title: Optional[str] = None
    image_base64: str  # data:image/png;base64,...

class ReportSpec(BaseModel):
    title: str = "DAFE Report"
    slides: List[ReportSlide]

class AnalyzeSpec(BaseModel):
    dataset_id: str
    type: Literal["summary","distribution","correlation","regression","time_series","groupby"]
    x: Optional[str] = None
    y: Optional[str] = None
    agg: Optional[Literal["sum","count","mean","median","min","max"]] = "sum"
    split_by: Optional[str] = None
    time_grain: Optional[Literal["D","W","M","Q","Y"]] = "D"
    forecast_horizon: Optional[int] = 12
    limit: Optional[int] = 1000

# --- Endpoints ----------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True, "time": datetime.utcnow().isoformat() + "Z"}

@app.post("/analytics/upload")
def upload(file: UploadFile = File(...)):
    df = _load_dataframe_from_upload(file)
    df = _coerce_common_types(df)
    dsid = str(uuid.uuid4())[:8]
    DATASETS[dsid] = {"df": df, "created": datetime.utcnow()}
    return {"dataset_id": dsid, "rows": int(df.shape[0]), "columns": int(df.shape[1])}

@app.get("/analytics/profile")
def profile(dataset_id: str):
    df = _get_df(dataset_id)
    nulls = {c: int(df[c].isna().sum()) for c in df.columns}
    preview = df.head(25).fillna("").to_dict(orient="records")
    return {
        "dataset_id": dataset_id,
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": list(df.columns),
        "nulls": nulls,
        "numeric_summary": _numeric_summary(df),
        "top_categories": _top_categories(df, k=10),
        "date_summary": _date_summary(df),
        "preview": preview,
    }

@app.post("/analytics/aggregate")
def aggregate(spec: AggregateSpec):
    df = _get_df(spec.dataset_id).copy()
    if spec.limit:
        df = df.head(spec.limit)

    if spec.x and pd.api.types.is_datetime64_any_dtype(df.get(spec.x, pd.Series(dtype="float64"))):
        # Time series aggregate
        if not spec.y:
            raise HTTPException(400, "y required for time-based aggregate")
        s = df[[spec.x, spec.y]].dropna()
        s[spec.x] = pd.to_datetime(s[spec.x], errors="coerce")
        s = s.dropna(subset=[spec.x]).set_index(spec.x).sort_index()
        series = pd.to_numeric(s[spec.y], errors="coerce").resample(spec.time_grain or "D").sum(min_count=1)
        labels = series.index.to_pydatetime().tolist()
        return {"kind": "time_aggregate", **_chart_payload(f"{spec.y} by {spec.time_grain or 'D'}", labels, {"value": [float(v) if pd.notna(v) else None for v in series.tolist()]})}

    # Categorical groupby (+ optional split)
    if not (spec.x and spec.y):
        raise HTTPException(400, "x and y required")
    if spec.split_by:
        pivot = df.pivot_table(
            index=spec.x,
            columns=spec.split_by,
            values=spec.y,
            aggfunc=spec.agg,
        ).fillna(0)
        labels = pivot.index.tolist()
        series = {str(col): [float(v) for v in pivot[col].tolist()] for col in pivot.columns}
        title = f"{spec.agg}({spec.y}) by {spec.x} split by {spec.split_by}"
    else:
        g = df.groupby(spec.x)[spec.y]
        agg = getattr(g, spec.agg)()
        labels = agg.index.tolist()
        series = {f"{spec.agg}({spec.y})": [float(v) for v in agg.tolist()]}
        title = f"{spec.agg}({spec.y}) by {spec.x}"

    return {"kind": "aggregate", **_chart_payload(title, labels, series)}

@app.post("/analytics/report/pdf")
def report_pdf(spec: ReportSpec):
    # Build a multi-page PDF by rendering base64 images onto figures
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # cover
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.5, spec.title, ha="center", va="center", fontsize=22)
        pdf.savefig(fig); plt.close(fig)

        # slides
        for slide in spec.slides:
            img_b64 = slide.image_base64
            # Expect "data:image/png;base64,...."
            if "," in img_b64:
                img_b64 = img_b64.split(",", 1)[1]
            import base64
            img = base64.b64decode(img_b64)
            img_buf = io.BytesIO(img)
            arr = plt.imread(img_buf, format="png")

            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            ax.imshow(arr)
            ax.axis("off")
            if slide.title:
                fig.suptitle(slide.title, y=0.98, fontsize=14)
            pdf.savefig(fig); plt.close(fig)

    buf.seek(0)
    from fastapi.responses import StreamingResponse
    return StreamingResponse(buf, media_type="application/pdf", headers={"Content-Disposition": 'attachment; filename="report.pdf"'})

# --- Analyze (Sprint-1) -------------------------------------------------------
@app.post("/analytics/analyze")
def analyze(spec: AnalyzeSpec):
    df = _get_df(spec.dataset_id).copy()

    if spec.limit:
        df = df.head(spec.limit)

    if spec.y and spec.y in df.columns:
        df[spec.y] = pd.to_numeric(df[spec.y], errors="coerce")

    if spec.type == "summary":
        desc = df.describe(include="all", datetime_is_numeric=True).fillna("").to_dict()
        return {"kind": "summary", "summary": desc}

    if spec.type == "distribution":
        if not spec.x: raise HTTPException(400, "x required")
        vc = df[spec.x].value_counts(dropna=False).sort_index()
        return {"kind": "distribution",
                **_chart_payload(f"Distribution of {spec.x}", vc.index.tolist(), {"count": vc.values.tolist()})}

    if spec.type == "correlation":
        num = df.select_dtypes(include=[np.number])
        if num.empty:
            return {"kind": "correlation", "matrix": {}}
        corr = num.corr(numeric_only=True).round(3).fillna(0).to_dict()
        return {"kind": "correlation", "matrix": corr}

    if spec.type == "groupby":
        if not (spec.x and spec.y): raise HTTPException(400, "x and y required")
        g = df.groupby(spec.x)[spec.y]
        agg = getattr(g, spec.agg)() if spec.agg else g.sum()
        return {
            "kind": "groupby",
            **_chart_payload(f"{spec.agg or 'sum'}({spec.y}) by {spec.x}",
                             agg.index.tolist(),
                             {f"{spec.agg}({spec.y})": [float(v) for v in agg.values.tolist()]})
        }

    if spec.type == "regression":
        if not (spec.x and spec.y): raise HTTPException(400, "x and y required")
        x = df[spec.x]; y = pd.to_numeric(df[spec.y], errors="coerce")
        mask = x.notna() & y.notna()
        x, y = x[mask], y[mask]
        if np.issubdtype(x.dtype, np.datetime64):
            x_ = pd.to_datetime(x).view("int64") / 1e9
        else:
            x_ = pd.to_numeric(x, errors="coerce")
        mask2 = pd.Series(x_).notna().to_numpy()
        xv = pd.Series(x_)[mask2].to_numpy().reshape(-1, 1); yv = y[mask2].to_numpy()
        if len(xv) < 2:
            return {"kind": "regression", "error": "not_enough_points"}

        xm, ym = xv.mean(), yv.mean()
        slope = float(((xv - xm) * (yv - ym)).sum() / ((xv - xm) ** 2).sum())
        intercept = float(ym - slope * xm)
        yhat = (slope * xv + intercept).ravel()

        labels = x[mask][mask2].astype(str).tolist()
        return {
            "kind": "regression",
            **_chart_payload(f"Regression {spec.y} ~ {spec.x}",
                             labels,
                             {"actual": yv.tolist(), "fit": yhat.tolist()}),
            "coefficients": {"slope": slope, "intercept": intercept},
        }

    if spec.type == "time_series":
        if not (spec.x and spec.y): raise HTTPException(400, "x and y required")
        ts = df[[spec.x, spec.y]].dropna()
        ts[spec.x] = pd.to_datetime(ts[spec.x], errors="coerce")
        ts = ts.dropna(subset=[spec.x]).set_index(spec.x).sort_index()
        series = pd.to_numeric(ts[spec.y], errors="coerce").resample(spec.time_grain or "D").sum(min_count=1)
        labels = series.index.to_pydatetime().tolist()
        values = [float(v) if pd.notna(v) else None for v in series.tolist()]

        if len(series) and spec.forecast_horizon:
            last = float(series.iloc[-1]) if pd.notna(series.iloc[-1]) else 0.0
            fidx = pd.date_range(series.index[-1], periods=spec.forecast_horizon + 1, freq=spec.time_grain or "D")[1:]
            labels += list(fidx.to_pydatetime())
            values += [last] * spec.forecast_horizon

        return {"kind": "time_series",
                **_chart_payload(f"{spec.y} over {spec.x} ({spec.time_grain or 'D'})", labels, {"value": values})}

    raise HTTPException(400, "unsupported type")
