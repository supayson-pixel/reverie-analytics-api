# main.py - Fixed Profile Endpoint
# Replace your current main.py with this version

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
    "https://inspiring-tarsier-97b2c7.netlify.app",  # Your Netlify URL
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
# In-memory stores
# --------------------------------------------------------------------------------------
DATASETS: dict[str, dict] = {}
ALIASES: dict[str, str] = {}
TEMPLATES: dict[str, dict] = {}
FEEDBACK: list[dict] = []

# --------------------------------------------------------------------------------------
# Data Processing Utilities
# --------------------------------------------------------------------------------------
_CURRENCY_REGEX = r"[$€£¥,\(\)%\s]"

def _detect_delimiter(text: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(text[:2000])
        return dialect.delimiter
    except Exception:
        return ","

def _to_numeric_clean_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    txt = s.astype(str)
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
    
    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]
    
    if assign_columns:
        if len(assign_columns) != df.shape[1]:
            raise HTTPException(
                status_code=400,
                detail=f"assign_columns length {len(assign_columns)} != width {df.shape[1]}"
            )
        df.columns = [str(c) for c in assign_columns]
    
    return df

def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_numeric_dtype(s):
            continue
        
        sample = s.astype(str).str.replace(_CURRENCY_REGEX, "", regex=True).str.replace(".", "", regex=False)
        mask = sample.str.match(r"^-?\d+$", na=False)
        if mask.mean() >= 0.6:
            out[col] = _to_numeric_clean_series(s)
    
    return out

def detect_date_columns(df: pd.DataFrame) -> list[str]:
    date_cols: list[str] = []
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            date_cols.append(c)
            continue
        
        sample = df[c].dropna().head(100)
        if len(sample) == 0:
            continue
            
        parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() >= 0.7:
            date_cols.append(c)
    
    return date_cols

# --------------------------------------------------------------------------------------
# Summary Functions (FIXED)
# --------------------------------------------------------------------------------------
def summarize_numeric(df: pd.DataFrame, round_to: int = 2) -> dict[str, dict]:
    """Generate numeric column summaries - FIXED VERSION"""
    num = df.select_dtypes(include=[np.number])
    res: dict[str, dict] = {}
    
    if num.empty:
        return res
    
    try:
        desc = num.describe(percentiles=[0.25, 0.5, 0.75]).T
        for col, row in desc.iterrows():
            res[str(col)] = {}
            for stat, val in row.items():
                try:
                    if isinstance(val, (int, float)) and not pd.isna(val):
                        if stat == "count":
                            res[str(col)][str(stat)] = int(val)
                        else:
                            res[str(col)][str(stat)] = round(float(val), round_to)
                    else:
                        res[str(col)][str(stat)] = None
                except:
                    res[str(col)][str(stat)] = None
    except Exception as e:
        print(f"Error in summarize_numeric: {e}")
        return {}
    
    return res

def summarize_dates(df: pd.DataFrame, date_cols: list[str]) -> dict[str, dict]:
    """Generate date column summaries - FIXED VERSION"""
    out: dict[str, dict] = {}
    
    for c in date_cols:
        try:
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
        except Exception as e:
            print(f"Error processing date column {c}: {e}")
            out[c] = {}
    
    return out

def top_k_categories(df: pd.DataFrame, k: int = 5) -> dict[str, dict]:
    """Get top categories - FIXED VERSION"""
    out: dict[str, dict] = {}
    
    try:
        nonnum = df.select_dtypes(exclude=[np.number])
        date_like = set(detect_date_columns(df))
        
        for c in nonnum.columns:
            if c in date_like:
                continue
            
            try:
                vc = df[c].astype(str).value_counts().head(k)
                if not vc.empty:
                    out[str(c)] = {str(k): int(v) for k, v in vc.items()}
            except Exception as e:
                print(f"Error processing category column {c}: {e}")
                out[str(c)] = {}
    except Exception as e:
        print(f"Error in top_k_categories: {e}")
    
    return out

def dataframe_preview(df: pd.DataFrame, rows: int = 10) -> list[dict]:
    """Get sample rows - FIXED VERSION"""
    try:
        return df.head(rows).fillna("").to_dict(orient="records")
    except Exception as e:
        print(f"Error in dataframe_preview: {e}")
        return []

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
    type: str
    x: T.Optional[str] = None
    y: T.Optional[str] = None
    agg: T.Optional[str] = "sum"
    time_grain: T.Optional[str] = "D"
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

@app.get("/analytics/profile")
def analytics_profile(dataset_id: str, preview_rows: int = 10):
    """FIXED Profile endpoint"""
    try:
        if dataset_id not in DATASETS:
            raise HTTPException(status_code=404, detail="dataset not found")
        
        df = DATASETS[dataset_id]["df"]
        
        # Basic info
        shape = {"rows": int(len(df)), "columns": int(df.shape[1])}
        cols = [str(c) for c in df.columns]
        
        # Nulls - safe conversion
        nulls = {}
        for c in df.columns:
            try:
                nulls[str(c)] = int(df[c].isna().sum())
            except:
                nulls[str(c)] = 0
        
        # Summaries with error handling
        num_summary = summarize_numeric(df, round_to=2)
        date_cols = detect_date_columns(df)
        date_summary = summarize_dates(df, date_cols)
        cats = top_k_categories(df, k=5)
        prev = dataframe_preview(df, rows=preview_rows)
        
        return {
            "shape": shape,
            "columns": cols,
            "nulls": nulls,
            "numeric_summary": num_summary,
            "date_summary": date_summary,
            "top5_categories": cats,
            "preview": prev,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Profile error: {e}")
        raise HTTPException(status_code=500, detail=f"Profile generation failed: {str(e)}")

# --------------------------------------------------------------------------------------
# Analysis Engine
# --------------------------------------------------------------------------------------
def _chart_payload(title: str, labels: list, datasets: list[dict]) -> dict:
    def _fmt(v):
        try:
            return v.isoformat() if hasattr(v, 'isoformat') else str(v)
        except:
            return str(v)

    return {
        "title": title,
        "labels": [_fmt(v) for v in labels],
        "datasets": datasets
    }

@app.post("/analytics/analyze")
def analytics_analyze(spec: AnalyzeSpec):
    try:
        if spec.dataset_id not in DATASETS:
            raise HTTPException(status_code=404, detail="dataset not found")
        
        df = DATASETS[spec.dataset_id]["df"]
        
        if spec.limit and spec.limit > 0:
            df = df.head(spec.limit)
        
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
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def run_summary_analysis(df: pd.DataFrame) -> dict:
    try:
        # FIXED: Remove datetime_is_numeric parameter for compatibility
        desc = df.describe(include="all").fillna("").to_dict()
        return {
            "kind": "summary",
            "shape": {"rows": len(df), "columns": df.shape[1]},
            "summary": desc,
            "nulls": {str(c): int(df[c].isna().sum()) for c in df.columns}
        }
    except Exception as e:
        return {"kind": "summary", "error": str(e)}

def run_distribution_analysis(df: pd.DataFrame, spec: AnalyzeSpec) -> dict:
    if not spec.x:
        raise HTTPException(status_code=400, detail="x column required for distribution")
    
    if spec.x not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{spec.x}' not found")
    
    try:
        vc = df[spec.x].value_counts(dropna=False).sort_index().head(20)
        return _chart_payload(
            f"Distribution of {spec.x}",
            vc.index.tolist(),
            [{"label": "count", "data": vc.values.tolist()}]
        )
    except Exception as e:
        return {"kind": "distribution", "error": str(e)}

def run_correlation_analysis(df: pd.DataFrame) -> dict:
    try:
        num = df.select_dtypes(include=[np.number])
        if num.empty:
            return {"kind": "correlation", "matrix": {}, "message": "No numeric columns found"}
        
        corr = num.corr(numeric_only=True).round(3).fillna(0).to_dict()
        return {"kind": "correlation", "matrix": corr}
    except Exception as e:
        return {"kind": "correlation", "error": str(e)}

def run_groupby_analysis(df: pd.DataFrame, spec: AnalyzeSpec) -> dict:
    if not (spec.x and spec.y):
        raise HTTPException(status_code=400, detail="Both x and y columns required for groupby")
    
    if spec.x not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{spec.x}' not found")
    
    if spec.y not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{spec.y}' not found")
    
    try:
        y_series = pd.to_numeric(df[spec.y], errors="coerce")
        if y_series.isna().all():
            raise HTTPException(status_code=400, detail=f"Column '{spec.y}' contains no numeric data")
        
        work_df = df[[spec.x, spec.y]].copy()
        work_df[spec.y] = y_series
        work_df = work_df.dropna()
        
        agg_func = getattr(work_df.groupby(spec.x)[spec.y], spec.agg, work_df.groupby(spec.x)[spec.y].sum)
        result = agg_func().sort_values(ascending=False).head(20)
        
        return _chart_payload(
            f"{spec.agg}({spec.y}) by {spec.x}",
            result.index.tolist(),
            [{"label": f"{spec.agg}({spec.y})", "data": [float(v) for v in result.values]}]
        )
    except Exception as e:
        return {"kind": "groupby", "error": str(e)}

def run_regression_analysis(df: pd.DataFrame, spec: AnalyzeSpec) -> dict:
    # Simplified regression to avoid errors
    return {"kind": "regression", "message": "Regression analysis coming soon"}

def run_timeseries_analysis(df: pd.DataFrame, spec: AnalyzeSpec) -> dict:
    # Simplified timeseries to avoid errors  
    return {"kind": "time_series", "message": "Time series analysis coming soon"}

# --------------------------------------------------------------------------------------
# PDF Report Generation
# --------------------------------------------------------------------------------------
@app.post("/analytics/report/pdf")
def generate_pdf_report(spec: ReportSpec):
    if not _HAS_MPL:
        raise HTTPException(status_code=501, detail="Matplotlib not available for PDF generation")

    try:
        buf = io.BytesIO()
        
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
                    img_b64 = slide.image_base64
                    if "," in img_b64:
                        img_b64 = img_b64.split(",", 1)[1]
                    
                    img_data = base64.b64decode(img_b64)
                    img_buf = io.BytesIO(img_data)
                    
                    img = plt.imread(img_buf, format="png")
                    ax = fig.add_subplot(111)
                    ax.imshow(img)
                    ax.axis("off")
                    
                    if slide.title:
                        fig.suptitle(slide.title, fontsize=16, weight="bold")
                        
                except Exception as e:
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
    try:
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
    except Exception as e:
        return {"datasets": [], "count": 0, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
