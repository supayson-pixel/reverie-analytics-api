# main.py - Complete DAFE Backend
# Consolidates all endpoints: upload, profile, analyze, exports, scheduling
import io
import os
import re
import csv
import ssl
import uuid
import json
import base64
import smtplib
import typing as T
from datetime import datetime

import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Optional imports
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    from matplotlib.backends.backend_pdf import PdfPages
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

try:
    import paramiko
    _HAS_SFTP = True
except Exception:
    _HAS_SFTP = False

# --------------------------------------------------------------------------------------
# App & CORS Setup
# --------------------------------------------------------------------------------------
app = FastAPI(title="Reverie Analytics API", version="1.0")

# CORS - Allow your Netlify domains
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*")
if CORS_ALLOW_ORIGINS == "*":
    allow_origins = ["*"]
else:
    allow_origins = [o.strip() for o in CORS_ALLOW_ORIGINS.split(",")]

# Add common origins
allow_origins.extend([
    "https://reveriesun.com",
    "https://www.reveriesun.com",
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000"
])

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_origin_regex=r"https://.*\.netlify\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------------------
# In-memory stores (for demo - replace with database later)
# --------------------------------------------------------------------------------------
DATASETS: dict[str, dict] = {}     # dataset_id -> {"df": DataFrame, "meta": {...}}
ALIASES: dict[str, str] = {}       # alias -> latest dataset_id
TEMPLATES: dict[str, dict] = {}    # template_id -> template dict
FEEDBACK: list[dict] = []          # simple feedback log

# --------------------------------------------------------------------------------------
# Data Processing Utilities
# --------------------------------------------------------------------------------------
_CURRENCY_REGEX = r"[$€£¥,\(\)%\s]"

def _detect_delimiter(text: str) -> str:
    """Detect CSV delimiter from sample text"""
    try:
        dialect = csv.Sniffer().sniff(text[:2000])
        return dialect.delimiter
    except Exception:
        return ","

def _to_numeric_clean_series(s: pd.Series) -> pd.Series:
    """Coerce a text series with currency/commas/percent to float."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    
    txt = s.astype(str)
    # Handle negative numbers in parentheses
    neg = txt.str.contains(r"^\s*\(.*\)\s*$", regex=True, na=False)
    cleaned = txt.str.replace(_CURRENCY_REGEX, "", regex=True)
    out = pd.to_numeric(cleaned, errors="coerce")
    out[neg] = -out[neg]
    return out

def _read_any_to_df(
    file_name: str,
    blob: bytes,
    has_header: T.Optional[bool] = None,
    header_row: T.Optional[int] = None,
    assign_columns: T.Optional[list[str]] = None,
) -> pd.DataFrame:
    """Robust reader for CSV/XLSX with header control"""
    name = (file_name or "").lower()

    if name.endswith(".xlsx") or name.endswith(".xls"):
        if header_row is not None:
            df = pd.read_excel(io.BytesIO(blob), header=None, engine='openpyxl')
            if header_row < len(df):
                hdr = df.iloc[header_row].astype(str).tolist()
                df = df.drop(index=df.index[header_row]).reset_index(drop=True)
                df.columns = hdr
        else:
            if has_header is False:
                df = pd.read_excel(io.BytesIO(blob), header=None, engine='openpyxl')
            else:
                df = pd.read_excel(io.BytesIO(blob), engine='openpyxl')
    else:
        # CSV handling
        try:
            text = blob.decode("utf-8", errors="ignore")
        except:
            text = blob.decode("latin1", errors="ignore")
        
        sep = _detect_delimiter(text)
        
        if header_row is not None:
            df = pd.read_csv(io.StringIO(text), sep=sep, header=None)
            if header_row < len(df):
                hdr = df.iloc[header_row].astype(str).tolist()
                df = df.drop(index=df.index[header_row]).reset_index(drop=True)
                df.columns = hdr
        else:
            if has_header is False:
                df = pd.read_csv(io.StringIO(text), sep=sep, header=None)
            else:
                df = pd.read_csv(io.StringIO(text), sep=sep)

    # Clean up column names
    df.columns = [str(c).strip() for c in df.columns]
    
    # Assign custom columns if provided
    if assign_columns:
        if len(assign_columns) != df.shape[1]:
            raise HTTPException(
                status_code=400, 
                detail=f"assign_columns length {len(assign_columns)} != width {df.shape[1]}"
            )
        df.columns = [str(c) for c in assign_columns]

    return df

def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert text columns that look numeric to actual numbers"""
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_numeric_dtype(s):
            continue
        
        # Check if this looks like a numeric column
        sample = s.astype(str).str.replace(_CURRENCY_REGEX, "", regex=True).str.replace(".", "", regex=False)
        mask = sample.str.match(r"^-?\d+$", na=False)
        if mask.mean() >= 0.6:  # If 60%+ look numeric
            out[col] = _to_numeric_clean_series(s)
    
    return out

def detect_date_columns(df: pd.DataFrame) -> list[str]:
    """Find columns that look like dates"""
    date_cols: list[str] = []
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            date_cols.append(c)
            continue
        
        # Try parsing a sample
        sample = df[c].dropna().head(100)
        if len(sample) == 0:
            continue
            
        parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() >= 0.7:  # 70%+ valid dates
            date_cols.append(c)
    
    return date_cols

# --------------------------------------------------------------------------------------
# Summary/Profile Functions
# --------------------------------------------------------------------------------------
def summarize_numeric(df: pd.DataFrame, round_to: int = 2) -> dict[str, dict[str, float]]:
    """Generate numeric column summaries"""
    num = df.select_dtypes(include=[np.number])
    res: dict[str, dict[str, float]] = {}
    if num.empty:
        return res
    
    desc = num.describe(percentiles=[0.25, 0.5, 0.75]).T
    for col, row in desc.iterrows():
        res[col] = {}
        for stat, val in row.items():
            if isinstance(val, (int, float)) and not pd.isna(val):
                if stat == "count":
                    res[col][stat] = int(val)
                else:
                    res[col][stat] = round(val, round_to)
            else:
                res[col][stat] = None
    
    return res

def summarize_dates(df: pd.DataFrame, date_cols: list[str]) -> dict[str, dict]:
    """Generate date column summaries"""
    out: dict[str, dict] = {}
    for c in date_cols:
        dt = pd.to_datetime(df[c], errors="coerce")
        dt = dt.dropna()
        if dt.empty:
            out[c] = {}
            continue
        
        min_d = dt.min().date().isoformat()
        max_d = dt.max().date().isoformat()
        span = (dt.max() - dt.min()).days
        
        by_month = dt.dt.to_period("M").astype(str).value_counts().sort_index().head(12).to_dict()
        by_weekday = dt.dt.day_name().value_counts().reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], 
            fill_value=0
        ).to_dict()
        
        out[c] = {
            "min": min_d, 
            "max": max_d, 
            "span_days": int(span), 
            "by_month": by_month, 
            "by_weekday": by_weekday
        }
    
    return out

def top_k_categories(df: pd.DataFrame, k: int = 5) -> dict[str, dict]:
    """Get top categories for text columns"""
    out: dict[str, dict] = {}
    nonnum = df.select_dtypes(exclude=[np.number])
    date_like = set(detect_date_columns(df))
    
    for c in nonnum.columns:
        if c in date_like:
            continue
        
        vc = df[c].astype(str).value_counts().head(k)
        if not vc.empty:
            out[c] = vc.to_dict()
    
    return out

def dataframe_preview(df: pd.DataFrame, rows: int = 10) -> list[dict]:
    """Get sample rows as dict records"""
    return df.head(rows).fillna("").to_dict(orient="records")

# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
class UploadResponse(BaseModel):
    dataset_id: str
    rows: int
    columns: int
    alias: T.Optional[str] = None

class ProfileResponse(BaseModel):
    shape: dict
    columns: list[str]
    nulls: dict[str, int]
    numeric_summary: dict = Field(default_factory=dict)
    date_summary: dict = Field(default_factory=dict)
    top5_categories: dict = Field(default_factory=dict)
    preview: list[dict] = Field(default_factory=list)

class AnalyzeSpec(BaseModel):
    dataset_id: str
    type: str  # summary, distribution, correlation, regression, time_series, groupby
    x: T.Optional[str] = None
    y: T.Optional[str] = None
    agg: T.Optional[str] = "sum"  # sum, count, mean, median, min, max
    time_grain: T.Optional[str] = "D"  # D, W, M
    forecast_horizon: T.Optional[int] = 12
    limit: T.Optional[int] = 1000

class ReportSlide(BaseModel):
    title: T.Optional[str] = None
    image_base64: str

class ReportSpec(BaseModel):
    title: str = "DAFE Report"
    slides: list[ReportSlide]

# --------------------------------------------------------------------------------------
# Core Endpoints
# --------------------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "Reverie Analytics API", 
        "endpoints": ["/health", "/analytics/upload", "/analytics/profile", "/analytics/analyze"],
        "version": "1.0"
    }

@app.get("/health")
def health():
    return {
        "ok": True, 
        "time": datetime.utcnow().isoformat() + "Z",
        "datasets": len(DATASETS),
        "version": "1.0"
    }

def _parse_bool(b: T.Optional[str]) -> T.Optional[bool]:
    """Convert string to boolean"""
    if b is None:
        return None
    if isinstance(b, bool):
        return b
    s = str(b).strip().lower()
    if s in ("true", "1", "yes", "y", "on"):
        return True
    if s in ("false", "0", "no", "n", "off"):
        return False
    return None

@app.post("/analytics/upload", response_model=UploadResponse)
def analytics_upload(
    file: UploadFile = File(...),
    dataset_alias: T.Optional[str] = Form(None),
    has_header: T.Optional[str] = Form(None),
    header_row: T.Optional[int] = Form(None),
    assign_columns_json: T.Optional[str] = Form(None)
):
    """Upload CSV/XLSX file and return dataset_id"""
    try:
        assign_columns = json.loads(assign_columns_json) if assign_columns_json else None
        blob = file.file.read()
        
        df = _read_any_to_df(
            file.filename,
            blob,
            has_header=_parse_bool(has_header),
            header_row=header_row,
            assign_columns=assign_columns,
        )
        
        df = coerce_numeric_columns(df)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")

    dsid = str(uuid.uuid4())
    DATASETS[dsid] = {
        "df": df, 
        "meta": {
            "filename": file.filename, 
            "alias": dataset_alias,
            "uploaded": datetime.utcnow().isoformat()
        }
    }
    
    if dataset_alias:
        ALIASES[dataset_alias] = dsid
    
    return UploadResponse(
        dataset_id=dsid, 
        rows=len(df), 
        columns=df.shape[1], 
        alias=dataset_alias
    )

@app.get("/analytics/profile", response_model=ProfileResponse)
def analytics_profile(dataset_id: str, preview_rows: int = 10):
    """Get dataset profile with shape, nulls, summaries, preview"""
    if dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="dataset not found")
    
    df = DATASETS[dataset_id]["df"]
    
    shape = {"rows": int(len(df)), "columns": int(df.shape[1])}
    cols = list(map(str, df.columns))
    nulls = {c: int(df[c].isna().sum()) for c in df.columns}
    
    num_summary = summarize_numeric(df, round_to=2)
    date_cols = detect_date_columns(df)
    date_summary = summarize_dates(df, date_cols)
    cats = top_k_categories(df, k=5)
    prev = dataframe_preview(df, rows=preview_rows)
    
    return ProfileResponse(
        shape=shape,
        columns=cols,
        nulls=nulls,
        numeric_summary=num_summary,
        date_summary=date_summary,
        top5_categories=cats,
        preview=prev,
    )

# --------------------------------------------------------------------------------------
# Analysis Engine
# --------------------------------------------------------------------------------------
def _chart_payload(title: str, labels: list, datasets: list[dict]) -> dict:
    """Format data for Chart.js frontend"""
    def _fmt(v):
        try:
            return v.isoformat() if hasattr(v, 'isoformat') else v
        except:
            return str(v)
    
    return {
        "title": title,
        "labels": [_fmt(v) for v in labels],
        "datasets": datasets
    }

@app.post("/analytics/analyze")
def analytics_analyze(spec: AnalyzeSpec):
    """Run analysis on dataset"""
    if spec.dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="dataset not found")
    
    df = DATASETS[spec.dataset_id]["df"]
    
    # Apply row limit for performance
    if spec.limit and spec.limit > 0:
        df = df.head(spec.limit)
    
    # Convert y column to numeric if specified
    if spec.y and spec.y in df.columns:
        df[spec.y] = pd.to_numeric(df[spec.y], errors="coerce")
    
    if spec.type == "summary":
        return run_summary_analysis(df)
    elif spec.type == "distribution":
        return run_distribution_analysis(df, spec)
    elif spec.type == "correlation":
        return run_correlation_analysis(df)
    elif spec.type == "groupby":
        return run_groupby_analysis(df, spec)
    elif spec.type == "regression":
        return run_regression_analysis(df, spec)
    elif spec.type == "time_series":
        return run_timeseries_analysis(df, spec)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported analysis type: {spec.type}")

def run_summary_analysis(df: pd.DataFrame) -> dict:
    """Basic dataset summary"""
    desc = df.describe(include="all", datetime_is_numeric=True).fillna("").to_dict()
    
    return {
        "kind": "summary",
        "shape": {"rows": len(df), "columns": df.shape[1]},
        "summary": desc,
        "nulls": {c: int(df[c].isna().sum()) for c in df.columns}
    }

def run_distribution_analysis(df: pd.DataFrame, spec: AnalyzeSpec) -> dict:
    """Value distribution for a column"""
    if not spec.x:
        raise HTTPException(status_code=400, detail="x column required for distribution")
    
    if spec.x not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{spec.x}' not found")
    
    vc = df[spec.x].value_counts(dropna=False).sort_index().head(20)  # Limit to top 20
    
    return _chart_payload(
        f"Distribution of {spec.x}",
        vc.index.tolist(),
        [{"label": "count", "data": vc.values.tolist()}]
    )

def run_correlation_analysis(df: pd.DataFrame) -> dict:
    """Correlation matrix for numeric columns"""
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return {"kind": "correlation", "matrix": {}, "message": "No numeric columns found"}
    
    corr = num.corr(numeric_only=True).round(3).fillna(0).to_dict()
    return {"kind": "correlation", "matrix": corr}

def run_groupby_analysis(df: pd.DataFrame, spec: AnalyzeSpec) -> dict:
    """Group by analysis"""
    if not (spec.x and spec.y):
        raise HTTPException(status_code=400, detail="Both x and y columns required for groupby")
    
    if spec.x not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{spec.x}' not found")
    if spec.y not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{spec.y}' not found")
    
    # Ensure y is numeric
    y_series = pd.to_numeric(df[spec.y], errors="coerce")
    if y_series.isna().all():
        raise HTTPException(status_code=400, detail=f"Column '{spec.y}' contains no numeric data")
    
    # Create working dataframe
    work_df = df[[spec.x, spec.y]].copy()
    work_df[spec.y] = y_series
    work_df = work_df.dropna()
    
    # Group and aggregate
    agg_func = getattr(work_df.groupby(spec.x)[spec.y], spec.agg, work_df.groupby(spec.x)[spec.y].sum)
    result = agg_func().sort_values(ascending=False).head(20)  # Top 20 groups
    
    return _chart_payload(
        f"{spec.agg}({spec.y}) by {spec.x}",
        result.index.tolist(),
        [{"label": f"{spec.agg}({spec.y})", "data": [float(v) for v in result.values]}]
    )

def run_regression_analysis(df: pd.DataFrame, spec: AnalyzeSpec) -> dict:
    """Simple linear regression"""
    if not (spec.x and spec.y):
        raise HTTPException(status_code=400, detail="Both x and y columns required for regression")
    
    if spec.x not in df.columns or spec.y not in df.columns:
        raise HTTPException(status_code=400, detail="Specified columns not found")
    
    # Prepare data
    x_data = df[spec.x]
    y_data = pd.to_numeric(df[spec.y], errors="coerce")
    
    # Remove missing values
    mask = x_data.notna() & y_data.notna()
    x_data, y_data = x_data[mask], y_data[mask]
    
    if len(x_data) < 2:
        return {"kind": "regression", "error": "Not enough valid data points"}
    
    # Convert x to numeric (handle dates)
    if np.issubdtype(x_data.dtype, np.datetime64):
        x_numeric = pd.to_datetime(x_data).view("int64") / 1e9  # Convert to seconds
    else:
        x_numeric = pd.to_numeric(x_data, errors="coerce")
        if x_numeric.isna().all():
            return {"kind": "regression", "error": "X column contains no numeric data"}
    
    # Remove any remaining NaN values
    final_mask = pd.Series(x_numeric).notna()
    x_vals = x_numeric[final_mask].values.reshape(-1, 1)
    y_vals = y_data[final_mask].values
    
    if len(x_vals) < 2:
        return {"kind": "regression", "error": "Not enough valid data points after cleaning"}
    
    # Simple linear regression (avoiding sklearn dependency)
    x_mean, y_mean = x_vals.mean(), y_vals.mean()
    slope = float(((x_vals.flatten() - x_mean) * (y_vals - y_mean)).sum() / 
                  ((x_vals.flatten() - x_mean) ** 2).sum())
    intercept = float(y_mean - slope * x_mean)
    
    # Generate predictions
    y_pred = slope * x_vals.flatten() + intercept
    
    # Use original x values for labels
    labels = x_data[mask][final_mask].astype(str).tolist()
    
    return {
        "kind": "regression",
        **_chart_payload(
            f"Regression: {spec.y} ~ {spec.x}",
            labels,
            [
                {"label": "actual", "data": y_vals.tolist()},
                {"label": "predicted", "data": y_pred.tolist()}
            ]
        ),
        "coefficients": {"slope": slope, "intercept": intercept}
    }

def run_timeseries_analysis(df: pd.DataFrame, spec: AnalyzeSpec) -> dict:
    """Time series analysis with optional forecasting"""
    if not (spec.x and spec.y):
        raise HTTPException(status_code=400, detail="Both x and y columns required for time series")
    
    if spec.x not in df.columns or spec.y not in df.columns:
        raise HTTPException(status_code=400, detail="Specified columns not found")
    
    # Prepare time series data
    ts_df = df[[spec.x, spec.y]].copy()
    ts_df[spec.x] = pd.to_datetime(ts_df[spec.x], errors="coerce")
    ts_df[spec.y] = pd.to_numeric(ts_df[spec.y], errors="coerce")
    ts_df = ts_df.dropna()
    
    if ts_df.empty:
        return {"kind": "time_series", "error": "No valid time series data after cleaning"}
    
    # Set datetime index and resample
    ts_df = ts_df.set_index(spec.x).sort_index()
    
    try:
        resampled = ts_df[spec.y].resample(spec.time_grain).sum()
    except Exception:
        resampled = ts_df[spec.y].groupby(ts_df.index.date).sum()
    
    # Generate labels and data
    labels = [dt.isoformat() for dt in resampled.index.to_pydatetime()]
    values = [float(v) if pd.notna(v) else None for v in resampled.values]
    
    # Simple forecasting (repeat last value)
    if spec.forecast_horizon and len(resampled) > 0:
        last_value = float(resampled.iloc[-1]) if pd.notna(resampled.iloc[-1]) else 0.0
        
        try:
            future_dates = pd.date_range(
                start=resampled.index[-1], 
                periods=spec.forecast_horizon + 1, 
                freq=spec.time_grain
            )[1:]
            
            labels.extend([dt.isoformat() for dt in future_dates.to_pydatetime()])
            values.extend([last_value] * spec.forecast_horizon)
        except Exception:
            pass  # Skip forecasting if date generation fails
    
    return _chart_payload(
        f"{spec.y} over time ({spec.time_grain})",
        labels,
        [{"label": spec.y, "data": values}]
    )

# --------------------------------------------------------------------------------------
# PDF Report Generation
# --------------------------------------------------------------------------------------
@app.post("/analytics/report/pdf")
def generate_pdf_report(spec: ReportSpec):
    """Generate PDF report from chart images"""
    if not _HAS_MPL:
        raise HTTPException(status_code=501, detail="Matplotlib not available for PDF generation")
    
    buf = io.BytesIO()
    
    try:
        with PdfPages(buf) as pdf:
            # Cover page
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.5, spec.title, ha="center", va="center", fontsize=20, weight="bold")
            fig.text(0.5, 0.4, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", 
                    ha="center", va="center", fontsize=12)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            
            # Content slides
            for slide in spec.slides:
                fig = plt.figure(figsize=(11, 8.5))
                
                try:
                    # Decode base64 image
                    img_b64 = slide.image_base64
                    if "," in img_b64:
                        img_b64 = img_b64.split(",", 1)[1]
                    
                    img_data = base64.b64decode(img_b64)
                    img_buf = io.BytesIO(img_data)
                    
                    # Display image
                    img = plt.imread(img_buf, format="png")
                    ax = fig.add_subplot(111)
                    ax.imshow(img)
                    ax.axis("off")
                    
                    if slide.title:
                        fig.suptitle(slide.title, fontsize=16, weight="bold")
                    
                except Exception as e:
                    # Fallback: text slide
                    fig.text(0.5, 0.5, f"Error loading chart: {str(e)}", 
                            ha="center", va="center", fontsize=14)
                
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        
        buf.seek(0)
        
        return StreamingResponse(
            io.BytesIO(buf.read()),
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="dafe_report.pdf"'}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

# --------------------------------------------------------------------------------------
# Additional Endpoints
# --------------------------------------------------------------------------------------
@app.get("/datasets")
def list_datasets():
    """List all uploaded datasets"""
    datasets = []
    for dsid, data in DATASETS.items():
        meta = data.get("meta", {})
        df = data["df"]
        datasets.append({
            "dataset_id": dsid,
            "filename": meta.get("filename"),
            "alias": meta.get("alias"),
            "uploaded": meta.get("uploaded"),
            "rows": len(df),
            "columns": df.shape[1],
            "column_names": list(df.columns)
        })
    
    return {"datasets": datasets, "count": len(datasets)}

@app.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: str):
    """Delete a dataset"""
    if
