# main.py
import io
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from utils_io import load_table  # <-- the robust loader below

# --- Add near top ---
from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Any
import pandas as pd
import numpy as np

class AnalyzeSpec(BaseModel):
    dataset_id: str
    type: Literal["summary","distribution","correlation","regression","time_series","groupby"]
    x: Optional[str] = None
    y: Optional[str] = None
    agg: Optional[Literal["sum","count","mean","median","min","max"]] = "sum"
    split_by: Optional[str] = None
    time_grain: Optional[Literal["D","W","M","Q","Y"]] = "D"
    forecast_horizon: Optional[int] = 12  # for time_series preview/forecast MVP
    limit: Optional[int] = 1000

def get_df(dataset_id: str) -> pd.DataFrame:
    # your existing in-memory / store retrieval
    return DATASETS[dataset_id]["df"]  # adjust to your structure

def chart_payload(title: str, labels: List[Any], series: Dict[str, List[float]]) -> Dict[str, Any]:
    return {"title": title, "labels": list(map(lambda v: v.isoformat() if hasattr(v, "isoformat") else v, labels)),
            "datasets": [{"label": k, "data": v} for k, v in series.items()]}

@app.post("/analytics/analyze")
def analyze(spec: AnalyzeSpec):
    df = get_df(spec.dataset_id).copy()

    # Safety: limit rows for speed
    if spec.limit:
        df = df.head(spec.limit)

    # Coerce potential numerics
    for c in [spec.y]:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if spec.type == "summary":
        desc = df.describe(include="all", datetime_is_numeric=True).fillna("").to_dict()
        return {"kind":"summary","summary": desc}

    if spec.type == "distribution":
        assert spec.x, "x required"
        vc = df[spec.x].value_counts(dropna=False).sort_index()
        return {"kind":"distribution", **chart_payload(f"Distribution of {spec.x}", vc.index.tolist(), {"count": vc.values.tolist()})}

    if spec.type == "correlation":
        num = df.select_dtypes(include=[np.number])
        corr = num.corr(numeric_only=True).round(3).fillna(0).to_dict()
        return {"kind":"correlation","matrix": corr}

    if spec.type == "groupby":
        assert spec.x and spec.y, "x and y required"
        g = df.groupby(spec.x)[spec.y]
        agg = getattr(g, spec.agg)() if spec.agg else g.sum()
        return {"kind":"groupby", **chart_payload(f"{spec.agg or 'sum'}({spec.y}) by {spec.x}", agg.index.tolist(), {f"{spec.agg}({spec.y})": agg.values.tolist()})}

    if spec.type == "regression":
        # Simple linear regression y ~ x (both numeric or x as datetime index → ordinal)
        assert spec.x and spec.y, "x and y required"
        x = df[spec.x]
        y = pd.to_numeric(df[spec.y], errors="coerce")
        mask = y.notna() & x.notna()
        x = x[mask]; y = y[mask]
        # transform x
        if np.issubdtype(x.dtype, np.datetime64):
            x_ = pd.to_datetime(x).view("int64") / 1e9  # seconds
        else:
            x_ = pd.to_numeric(x, errors="coerce")
        mask2 = x_.notna()
        x_ = x_[mask2].to_numpy().reshape(-1,1)
        y_ = y[mask2].to_numpy()

        if len(x_) < 2:
            return {"kind":"regression","error":"not_enough_points"}

        # no heavy deps: closed-form slope/intercept
        xmean, ymean = x_.mean(), y_.mean()
        slope = float(((x_ - xmean)*(y_ - ymean)).sum() / ((x_ - xmean)**2).sum())
        intercept = float(ymean - slope*xmean)
        yhat = (slope*x_ + intercept).ravel()

        labels = x[mask2].astype(str).tolist()
        return {"kind":"regression",
                **chart_payload(f"Regression {spec.y} ~ {spec.x}",
                                labels,
                                {"actual": y_.tolist(), "fit": yhat.tolist()}),
                "coefficients":{"slope": slope, "intercept": intercept}}

    if spec.type == "time_series":
        # Resample y by time grain on index = x
        assert spec.x and spec.y, "x and y required"
        ts = df[[spec.x, spec.y]].dropna()
        ts[spec.x] = pd.to_datetime(ts[spec.x], errors="coerce")
        ts = ts.dropna(subset=[spec.x]).set_index(spec.x).sort_index()
        series = ts[spec.y].astype(float).resample(spec.time_grain).sum(min_count=1)
        labels = series.index.to_pydatetime().tolist()
        base = series.tolist()

        # Forecast MVP: naive last-value hold for horizon
        if len(series) > 0 and spec.forecast_horizon:
            last = series.iloc[-1]
            f_labels = pd.date_range(series.index[-1], periods=spec.forecast_horizon+1, freq=spec.time_grain)[1:].to_pydatetime().tolist()
            f_values = [float(last)]*spec.forecast_horizon
            labels = labels + f_labels
            base = base + f_values

        return {"kind":"time_series", **chart_payload(f"{spec.y} over {spec.x} ({spec.time_grain})", labels, {"value": [float(v) if v is not None else None for v in base]})}

    return {"error":"unsupported_type"}


# ------------ App & CORS ------------
app = FastAPI(title="Reverie Analytics API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    # allow reveriesun.com and netlify.app (including deploy previews) + localhost
    allow_origin_regex=r"https://([a-z0-9-]+\.)?(reveriesun\.com|netlify\.app)$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------ In-memory store (simple for now) ------------
DATASETS: Dict[str, pd.DataFrame] = {}

# ------------ Schemas ------------
class AggregateSpec(BaseModel):
    dataset_id: str
    x: str
    y: Optional[str] = None          # optional for counts
    agg: str = "sum"                 # sum|count|mean
    split_by: Optional[str] = None   # optional categorical split
    type: str = "bar"                # bar|stackedBar|line|area|donut|histogram|boxplot|table
    time_grain: Optional[str] = None # auto|day|week|month
    filters: Optional[List[Dict[str, Any]]] = None

class Branding(BaseModel):
    logo_url: Optional[str] = None
    accent: Optional[str] = None

class Slide(BaseModel):
    layout: str                       # title|one-chart|two-charts|bullets
    title: Optional[str] = None
    subtitle: Optional[str] = None
    items: Optional[List[str]] = None
    chart: Optional[AggregateSpec] = None
    left: Optional[AggregateSpec] = None
    right: Optional[AggregateSpec] = None

class ReportSpec(BaseModel):
    title: str = "Reverie Report"
    subtitle: Optional[str] = None
    branding: Optional[Branding] = None
    slides: List[Slide]

# ------------ Health ------------
@app.get("/health")
def health():
    return {"ok": True}

# ------------ Upload ------------
@app.post("/analytics/upload")
async def upload(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = load_table(content, file.filename or "")
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="Could not parse file (empty).")
        dsid = str(uuid.uuid4())
        DATASETS[dsid] = df
        return {
            "dataset_id": dsid,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "alias": None,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Upload failed: {e}")

# ------------ Profile ------------
@app.get("/analytics/profile")
def profile(dataset_id: str):
    if dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="dataset not found")
    df = DATASETS[dataset_id].copy()

    # columns & nulls
    columns = list(df.columns)
    nulls = {c: int(df[c].isna().sum()) for c in columns}

    # numeric summary
    numeric_summary: Dict[str, Dict[str, Any]] = {}
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            desc = df[c].describe(percentiles=[0.25, 0.5, 0.75])
            numeric_summary[c] = {
                "count": float(desc.get("count", 0) or 0),
                "mean": float(desc.get("mean", 0) or 0),
                "std": float(desc.get("std", 0) or 0),
                "min": float(desc.get("min", 0) or 0),
                "25%": float(desc.get("25%", 0) or 0),
                "50%": float(desc.get("50%", 0) or 0),
                "75%": float(desc.get("75%", 0) or 0),
                "max": float(desc.get("max", 0) or 0),
            }

    # date summary
    date_summary: Dict[str, Dict[str, Any]] = {}
    for c in df.columns:
        s = pd.to_datetime(df[c], errors="coerce", utc=False, infer_datetime_format=True)
        if s.notna().sum() > 0:
            span_days = (s.max() - s.min()).days if (s.max() and s.min()) else None
            by_month = s.dt.to_period("M").astype(str).value_counts().sort_index().to_dict()
            by_weekday = s.dt.day_name().value_counts().to_dict()
            by_month = {k: int(v) for k, v in by_month.items()}
            by_weekday = {k: int(v) for k, v in by_weekday.items()}
            date_summary[c] = {
                "span_days": int(span_days) if span_days is not None else None,
                "by_month": by_month,
                "by_weekday": by_weekday,
            }

    # top 5 categories for object-like columns
    top5_categories: Dict[str, Dict[str, int]] = {}
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]):
            counts = df[c].value_counts().head(5).to_dict()
            top5_categories[c] = {str(k): int(v) for k, v in counts.items()}

    preview = df.head(5).to_dict(orient="records")

    return {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": columns,
        "nulls": nulls,
        "numeric_summary": numeric_summary,
        "date_summary": date_summary,
        "top5_categories": top5_categories,
        "preview": preview,
    }

# ------------ Aggregate helper ------------
def _apply_filters(df: pd.DataFrame, filters: Optional[List[Dict[str, Any]]]) -> pd.DataFrame:
    if not filters:
        return df
    res = df.copy()
    for f in filters:
        col = f.get("col")
        op = f.get("op")
        val = f.get("val")
        if col not in res.columns:
            continue
        if op == "eq":
            res = res[res[col] == val]
        elif op == "neq":
            res = res[res[col] != val]
        elif op == "gt":
            res = res[pd.to_numeric(res[col], errors="coerce") > float(val)]
        elif op == "lt":
            res = res[pd.to_numeric(res[col], errors="coerce") < float(val)]
        elif op == "contains":
            res = res[res[col].astype(str).str.contains(str(val), na=False, case=False)]
    return res

def _timekey(series: pd.Series, grain: Optional[str]) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce", utc=False, infer_datetime_format=True)
    if grain in (None, "", "auto"):
        grain = "month"
    if grain == "day":
        return s.dt.strftime("%Y-%m-%d")
    if grain == "week":
        return s.dt.to_period("W").astype(str)
    # default month
    return s.dt.to_period("M").astype(str)

def _aggregate(df: pd.DataFrame, spec: AggregateSpec):
    df = _apply_filters(df, spec.filters)

    if spec.x not in df.columns:
        raise HTTPException(status_code=400, detail=f"x column '{spec.x}' not found")
    ycol = spec.y
    if spec.agg not in ("sum", "count", "mean"):
        raise HTTPException(status_code=400, detail="agg must be sum|count|mean")

    # Build group key
    x_is_date = False
    try:
        x_is_date = pd.to_datetime(df[spec.x], errors="coerce").notna().any()
    except Exception:
        x_is_date = False

    if x_is_date:
        gkey = _timekey(df[spec.x], spec.time_grain)
    else:
        gkey = df[spec.x].astype(str)

    # choose series to aggregate
    if ycol and ycol in df.columns and pd.api.types.is_numeric_dtype(df[ycol]):
        values = pd.to_numeric(df[ycol], errors="coerce")
    else:
        # count mode when no numeric y
        spec.agg = "count"
        values = pd.Series([1] * len(df), index=df.index)

    # split or not
    if spec.split_by and spec.split_by in df.columns:
        split = df[spec.split_by].astype(str)
        tmp = pd.DataFrame({"x": gkey, "y": values, "split": split})
        if spec.agg == "sum":
            ret = tmp.groupby(["x", "split"], dropna=False)["y"].sum().unstack(fill_value=0).sort_index()
        elif spec.agg == "mean":
            ret = tmp.groupby(["x", "split"], dropna=False)["y"].mean().unstack(fill_value=0).sort_index()
        else:  # count
            ret = tmp.groupby(["x", "split"], dropna=False)["y"].count().unstack(fill_value=0).sort_index()
        labels = list(ret.index)
        datasets = [{"label": str(col), "data": [float(v) for v in ret[col].tolist()]} for col in ret.columns]
        return {"labels": labels, "datasets": datasets}
    else:
        tmp = pd.DataFrame({"x": gkey, "y": values})
        if spec.agg == "sum":
            ret = tmp.groupby("x", dropna=False)["y"].sum().sort_index()
        elif spec.agg == "mean":
            ret = tmp.groupby("x", dropna=False)["y"].mean().sort_index()
        else:
            ret = tmp.groupby("x", dropna=False)["y"].count().sort_index()
        labels = list(ret.index)
        data = [float(v) for v in ret.tolist()]
        return {"labels": labels, "datasets": [{"label": spec.y or "count", "data": data}]}

# ------------ Aggregate endpoint ------------
@app.post("/analytics/aggregate")
def aggregate(spec: AggregateSpec):
    if spec.dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="dataset not found")
    df = DATASETS[spec.dataset_id]
    result = _aggregate(df, spec)
    result.update({
        "x": spec.x, "y": spec.y, "agg": spec.agg,
        "split_by": spec.split_by, "time_grain": spec.time_grain or "auto",
        "type": spec.type
    })
    return result

# ------------ Report / PDF endpoint ------------
@app.post("/analytics/report/pdf")
def report_pdf(spec: ReportSpec):
    # Build the PDF in memory
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Cover slide
        plt.figure(figsize=(11, 8.5))
        plt.axis("off")
        title = spec.title or "Reverie Report"
        subtitle = spec.subtitle or datetime.now().strftime("%Y-%m-%d")
        plt.text(0.5, 0.72, title, ha="center", va="center", fontsize=28)
        plt.text(0.5, 0.64, subtitle, ha="center", va="center", fontsize=14)
        plt.text(0.5, 0.10, "Generated by Reverie Co", ha="center", va="center", fontsize=10)
        pdf.savefig(); plt.close()

        # Slide helper
        def draw_chart(ax, df, chartspec: AggregateSpec):
            agged = _aggregate(df, chartspec)
            labels = agged["labels"]
            datasets = agged["datasets"]
            ctype = chartspec.type

            if chartspec.split_by and len(datasets) > 1:
                # multi-series
                bottom = None
                for ds in datasets:
                    vals = ds["data"]
                    if ctype in ("stackedBar",):
                        ax.bar(labels, vals, bottom=bottom, label=ds["label"])
                        bottom = [a + b for a, b in zip(bottom or [0]*len(vals), vals)]
                    else:
                        ax.plot(labels, vals, label=ds["label"]) if ctype in ("line","area") else ax.bar(labels, vals, label=ds["label"])
                ax.legend()
            else:
                vals = datasets[0]["data"]
                if ctype in ("line","area"):
                    ax.plot(labels, vals)
                elif ctype == "donut":
                    wedges = ax.pie(vals, labels=labels, autopct="%1.1f%%")[0]
                    centre_circle = plt.Circle((0,0),0.60,fc="white")
                    ax.add_artist(centre_circle)
                else:
                    ax.bar(labels, vals)
            ax.set_title(f"{chartspec.agg.upper()}({chartspec.y or 'count'}) by {chartspec.x}" + (f" split by {chartspec.split_by}" if chartspec.split_by else ""))

        # Content slides
        for slide in spec.slides:
            layout = slide.layout
            plt.figure(figsize=(11, 8.5))
            if layout == "title":
                plt.axis("off")
                plt.text(0.5, 0.7, slide.title or "", ha="center", va="center", fontsize=24)
                plt.text(0.5, 0.6, slide.subtitle or "", ha="center", va="center", fontsize=14)
            elif layout == "bullets":
                plt.axis("off")
                y = 0.75
                if slide.title:
                    plt.text(0.02, 0.88, slide.title, fontsize=18, ha="left", va="center")
                for item in (slide.items or []):
                    plt.text(0.06, y, f"• {item}", fontsize=14, ha="left", va="center")
                    y -= 0.08
            elif layout == "one-chart" and slide.chart:
                ax = plt.gca()
                if slide.title:
                    plt.title(slide.title)
                # Use first dataset in memory; these chartspecs include dataset_id
                dsid = slide.chart.dataset_id
                if dsid not in DATASETS:
                    plt.text(0.5,0.5,"dataset not found",ha="center"); 
                else:
                    draw_chart(ax, DATASETS[dsid], slide.chart)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
            elif layout == "two-charts":
                left = plt.subplot(1,2,1)
                right = plt.subplot(1,2,2)
                if slide.left:
                    dsid = slide.left.dataset_id
                    if dsid in DATASETS:
                        draw_chart(left, DATASETS[dsid], slide.left)
                    left.tick_params(axis='x', rotation=45)
                if slide.right:
                    dsid = slide.right.dataset_id
                    if dsid in DATASETS:
                        draw_chart(right, DATASETS[dsid], slide.right)
                    right.tick_params(axis='x', rotation=45)
                plt.tight_layout()
            else:
                plt.axis("off")
                plt.text(0.5,0.5,"(empty slide)", ha="center", va="center")
            pdf.savefig(); plt.close()

    pdf_bytes = buf.getvalue()
    headers = {
        "Content-Disposition": 'attachment; filename="reverie_report.pdf"'
    }
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)
