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
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Optional deps: matplotlib for PNG charts, paramiko for SFTP
try:
    import matplotlib.pyplot as plt  # noqa
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

try:
    import paramiko  # noqa
    _HAS_SFTP = True
except Exception:
    _HAS_SFTP = False


# --------------------------------------------------------------------------------------
# In-memory stores (MVP). Move to a real DB later.
# --------------------------------------------------------------------------------------
DATASETS: dict[str, dict] = {}     # dataset_id -> {"df": DataFrame, "meta": {...}}
ALIASES: dict[str, str] = {}       # alias -> latest dataset_id
TEMPLATES: dict[str, dict] = {}    # template_id -> template dict
FEEDBACK: list[dict] = []          # simple feedback log


# --------------------------------------------------------------------------------------
# App & CORS
# --------------------------------------------------------------------------------------
app = FastAPI(title="Reverie Analytics API")

CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*")
allow_origins = ["*"] if CORS_ALLOW_ORIGINS == "*" else [o.strip() for o in CORS_ALLOW_ORIGINS.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------------------
# Utilities: parsing, headers, summaries
# --------------------------------------------------------------------------------------
_CURRENCY_REGEX = r"[$€£¥,\(\)%\s]"

def _detect_delimiter(text: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(text[:2000])
        return dialect.delimiter
    except Exception:
        return ","


def _read_any_to_df(
    file_name: str,
    blob: bytes,
    has_header: T.Optional[bool] = None,
    header_row: T.Optional[int] = None,
    assign_columns: T.Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Robust reader for CSV/XLSX with control over headers.
    - has_header=False -> read with header=None
    - header_row=i     -> read with header=None, then use row i as header
    - assign_columns   -> force-set columns after read (must match width)
    """
    name = (file_name or "").lower()

    if name.endswith(".xlsx") or name.endswith(".xls"):
        if header_row is not None:
            df = pd.read_excel(io.BytesIO(blob), header=None)
            hdr = df.iloc[header_row].astype(str).tolist()
            df = df.drop(index=df.index[header_row]).reset_index(drop=True)
            df.columns = hdr
        else:
            if has_header is False:
                df = pd.read_excel(io.BytesIO(blob), header=None)
            else:
                df = pd.read_excel(io.BytesIO(blob))  # default header row
    else:
        text = blob.decode("utf-8", errors="ignore")
        sep = _detect_delimiter(text)
        if header_row is not None:
            df = pd.read_csv(io.StringIO(text), sep=sep, header=None)
            hdr = df.iloc[header_row].astype(str).tolist()
            df = df.drop(index=df.index[header_row]).reset_index(drop=True)
            df.columns = hdr
        else:
            if has_header is False:
                df = pd.read_csv(io.StringIO(text), sep=sep, header=None)
            else:
                df = pd.read_csv(io.StringIO(text), sep=sep)  # header='infer'

    if assign_columns:
        if len(assign_columns) != df.shape[1]:
            raise HTTPException(status_code=400, detail=f"assign_columns length {len(assign_columns)} != width {df.shape[1]}")
        df.columns = [str(c) for c in assign_columns]
    else:
        # If pandas inferred headers as 0..N-1 and first row looks like strings (classic "no header" symptom), offer a light fix:
        if all(isinstance(c, (int, float)) for c in df.columns) and df.shape[0] > 0:
            # Heuristic: if at least half the first-row cells are non-numeric strings, treat them as headers.
            first = df.iloc[0].astype(str)
            looks_like_header = (first.str.len() > 0).mean() >= 0.6 and (~first.str.match(r"^-?\d+(\.\d+)?$", na=False)).mean() >= 0.6
            if looks_like_header:
                df = df.iloc[1:].reset_index(drop=True)
                df.columns = first.tolist()

    return df


def _to_numeric_clean_series(s: pd.Series) -> pd.Series:
    """Coerce a text series with currency/commas/percent to float."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    txt = s.astype(str)
    neg = txt.str.contains(r"^\s*\(.*\)\s*$", regex=True, na=False)
    cleaned = txt.str.replace(_CURRENCY_REGEX, "", regex=True)
    out = pd.to_numeric(cleaned, errors="coerce")
    out[neg] = -out[neg]
    return out


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_numeric_dtype(s):
            continue
        sample = s.astype(str).str.replace(_CURRENCY_REGEX, "", regex=True).str.replace(".", "", n=1, regex=False)
        mask = sample.str.match(r"^-?\d+(\.\d+)?$", na=False)
        if mask.mean() >= 0.6:
            out[col] = _to_numeric_clean_series(s)
    return out


def detect_date_columns(df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            out.append(c)
            continue
        parsed = pd.to_datetime(df[c], errors="coerce")
        if parsed.notna().mean() >= 0.6:
            out.append(c)
    return out


def summarize_numeric(df: pd.DataFrame, round_to: int = 2) -> dict[str, dict[str, float]]:
    num = df.select_dtypes(include="number")
    res: dict[str, dict[str, float]] = {}
    if num.empty:
        return res
    desc = num.describe(percentiles=[0.25, 0.5, 0.75]).T
    desc = desc.rename(columns={"25%": "25%", "50%": "50%", "75%": "75%"})
    for col, row in desc.iterrows():
        res[col] = {k: (round(v, round_to) if isinstance(v, (int, float)) and not pd.isna(v) else None) for k, v in row.items()}
        if res[col].get("count") is not None:
            res[col]["count"] = int(res[col]["count"])
    return res


def summarize_dates(df: pd.DataFrame, dates: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for c in dates:
        dt = pd.to_datetime(df[c], errors="coerce")
        dt = dt.dropna()
        if dt.empty:
            out[c] = {}
            continue
        min_d = dt.min().date().isoformat()
        max_d = dt.max().date().isoformat()
        span = (dt.max() - dt.min()).days
        by_month = dt.dt.to_period("M").astype(str).value_counts().sort_index().to_dict()
        by_weekday = dt.dt.day_name().value_counts().reindex(
            ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], fill_value=0
        ).to_dict()
        out[c] = {"min": min_d, "max": max_d, "span_days": int(span), "by_month": by_month, "by_weekday": by_weekday}
    return out


def top_k_categories(df: pd.DataFrame, k: int = 5) -> dict[str, dict]:
    out: dict[str, dict] = {}
    nonnum = df.select_dtypes(exclude="number")
    date_like = set(detect_date_columns(df))
    for c in nonnum.columns:
        if c in date_like:
            continue
        vc = df[c].astype(str).value_counts().head(k)
        if not vc.empty:
            out[c] = vc.to_dict()
    return out


def dataframe_preview(df: pd.DataFrame, rows: int = 10) -> list[dict]:
    return df.head(rows).to_dict(orient="records")


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
class AnalyzeSpec(BaseModel):
    type: str = Field(..., description="summary | pivot | timeseries | compose")
    params: dict = Field(default_factory=dict)


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


class IngestJSONRequest(BaseModel):
    records: list[dict]


class TemplateUpsert(BaseModel):
    template_id: str
    name: str
    source: dict = Field(..., description='{"mode":"latest_dataset_id","dataset_id":"..."} or {"mode":"alias","alias":"foo"}')
    report_spec: AnalyzeSpec
    delivery: dict = Field(..., description='{"email":["ops@acme.com"], "formats":["csv","png"], "subject":"Daily"}')


class RunTemplateRequest(BaseModel):
    template_id: str


class AssignHeadersRequest(BaseModel):
    dataset_id: str
    # supply exactly one of:
    columns: T.Optional[list[str]] = None          # full list of new column names
    header_row: T.Optional[int] = None             # use row i as header (and drop it)
    rename: T.Optional[dict[str, str]] = None      # rename some columns: {"old":"new"}


class FeedbackItem(BaseModel):
    dataset_id: T.Optional[str] = None
    spec: T.Optional[dict] = None
    rating: T.Optional[str] = Field(None, description="up|down|neutral or 1-5")
    reasons: T.Optional[list[str]] = None
    comment: T.Optional[str] = None
    duration_ms: T.Optional[int] = None
    user_id: T.Optional[str] = None
    org_id: T.Optional[str] = None
    ts_utc: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


# --------------------------------------------------------------------------------------
# Root & Health
# --------------------------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Reverie Analytics API. Try /health, /analytics/upload, /analytics/profile, /analytics/analyze, /ingest/*, /datasets/assign_headers, /templates/upsert, /jobs/run-template, /feedback"}


@app.get("/health")
def health():
    return {"ok": True, "time": datetime.utcnow().isoformat() + "Z"}


# --------------------------------------------------------------------------------------
# Core: upload (with header helpers), profile, analyze
# --------------------------------------------------------------------------------------
def _parse_bool(b: T.Optional[str]) -> T.Optional[bool]:
    if b is None:
        return None
    if isinstance(b, bool):
        return b
    s = str(b).strip().lower()
    if s in ("true","1","yes","y","on"):
        return True
    if s in ("false","0","no","n","off"):
        return False
    return None


@app.post("/analytics/upload", response_model=UploadResponse)
def analytics_upload(
    file: UploadFile = File(...),
    dataset_alias: T.Optional[str] = Form(None),
    has_header: T.Optional[str] = Form(None),         # "true"/"false" if provided
    header_row: T.Optional[int] = Form(None),
    assign_columns_json: T.Optional[str] = Form(None) # '["col1","col2",...]'
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
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")

    dsid = str(uuid.uuid4())
    DATASETS[dsid] = {"df": df, "meta": {"filename": file.filename, "alias": dataset_alias}}
    if dataset_alias:
        ALIASES[dataset_alias] = dsid
    return UploadResponse(dataset_id=dsid, rows=len(df), columns=df.shape[1], alias=dataset_alias)


@app.get("/analytics/profile", response_model=ProfileResponse)
def analytics_profile(dataset_id: str, preview_rows: int = 10):
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
        date_summary={c: date_summary.get(c, {}) for c in date_cols},
        top5_categories=cats,
        preview=prev,
    )


@app.post("/analytics/analyze")
def analytics_analyze(dataset_id: str = Body(...), spec: AnalyzeSpec = Body(...)):
    if dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="dataset not found")
    df = DATASETS[dataset_id]["df"]
    return run_analysis_from_spec(df, spec.model_dump())


# --------------------------------------------------------------------------------------
# Analysis runners (summary, pivot, timeseries, compose)
# --------------------------------------------------------------------------------------
def run_analysis_from_spec(df: pd.DataFrame, spec: dict) -> dict:
    t = (spec.get("type") or "").lower()
    p = spec.get("params") or {}

    if t == "summary":
        return {
            "shape": {"rows": len(df), "columns": df.shape[1]},
            "columns": list(df.columns),
            "nulls": {c: int(df[c].isna().sum()) for c in df.columns},
            "numeric_summary": summarize_numeric(df, round_to=2),
            "date_summary": summarize_dates(df, detect_date_columns(df)),
            "top5_categories": top_k_categories(df, k=5),
            "preview": dataframe_preview(df, rows=10),
        }

    if t == "pivot":
        index = p.get("index") or []
        values = p.get("values") or []
        agg = p.get("agg") or "sum"
        if not index or not values:
            raise HTTPException(status_code=400, detail="pivot needs index & values")
        out = df.pivot_table(index=index, values=values, aggfunc=agg).reset_index()
        return {"table": out.to_dict(orient="records")}

    if t == "timeseries":
        date_col = p.get("date_col")
        value_col = p.get("value_col")
        freq = p.get("freq", "D")
        if not date_col or not value_col:
            raise HTTPException(status_code=400, detail="timeseries needs date_col & value_col")
        tmp = df[[date_col, value_col]].dropna().copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp = tmp.dropna()
        g = tmp.groupby(pd.Grouper(key=date_col, freq=freq))[value_col].sum().reset_index()
        return {"table": g.to_dict(orient="records")}

    if t == "compose":
        return run_compose(df, p)

    return {"message": f"unknown analysis type '{t}'"}


# ---------------------- compose engine (MVP) ----------------------
_ALLOWED_FUNCS = {"to_month", "to_week", "to_day", "year", "month", "day"}

def _apply_derive(df: pd.DataFrame, derives: list[dict]) -> pd.DataFrame:
    if not derives:
        return df
    out = df.copy()
    for d in derives:
        alias = d.get("alias")
        expr = (d.get("expr") or "").strip()
        if not alias or not expr:
            continue
        # support simple functions: to_month(col), to_week(col), to_day(col), year(col), month(col), day(col)
        m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\(([^)]+)\)$", expr)
        if m and m.group(1) in _ALLOWED_FUNCS:
            fn, arg = m.group(1), m.group(2).strip()
            if arg not in out.columns:
                raise HTTPException(status_code=400, detail=f"derive: column '{arg}' not found")
            ser = out[arg]
            dt = pd.to_datetime(ser, errors="coerce")
            if fn == "to_month":
                out[alias] = dt.dt.to_period("M").astype(str)
            elif fn == "to_week":
                try:
                    out[alias] = dt.dt.isocalendar().week.astype(int)
                except Exception:
                    out[alias] = None
            elif fn == "to_day":
                out[alias] = dt.dt.date.astype(str)
            elif fn == "year":
                out[alias] = dt.dt.year
            elif fn == "month":
                out[alias] = dt.dt.month
            elif fn == "day":
                out[alias] = dt.dt.day
            continue
        # fallback: simple arithmetic "colA + colB" | "colA - colB" | "colA * colB" | "colA / colB"
        toks = re.split(r"\s*([+\-*/])\s*", expr)
        if len(toks) == 1 and expr in out.columns:
            out[alias] = pd.to_numeric(out[expr], errors="coerce")
            continue
        if len(toks) == 3 and toks[0] in out.columns and toks[2] in out.columns:
            a, op, b = toks
            A = pd.to_numeric(out[a], errors="coerce")
            B = pd.to_numeric(out[b], errors="coerce")
            if op == "+":
                out[alias] = A + B
            elif op == "-":
                out[alias] = A - B
            elif op == "*":
                out[alias] = A * B
            elif op == "/":
                out[alias] = A / B.replace({0: pd.NA})
            continue
        raise HTTPException(status_code=400, detail=f"derive: unsupported expr '{expr}'")
    return out


def _apply_filters(df: pd.DataFrame, filters: list[dict]) -> pd.DataFrame:
    if not filters:
        return df
    out = df.copy()
    for f in filters:
        col = f.get("col")
        op = (f.get("op") or "").lower()
        val = f.get("value")
        if col not in out.columns:
            raise HTTPException(status_code=400, detail=f"filter: column '{col}' not found")
        s = out[col]
        if op in ("==", "eq"):
            out = out[s == val]
        elif op in ("!=", "<>", "ne"):
            out = out[s != val]
        elif op in (">", "gt"):
            out = out[pd.to_numeric(s, errors="coerce") > _to_num(val)]
        elif op in (">=", "ge"):
            out = out[pd.to_numeric(s, errors="coerce") >= _to_num(val)]
        elif op in ("<", "lt"):
            out = out[pd.to_numeric(s, errors="coerce") < _to_num(val)]
        elif op in ("<=", "le"):
            out = out[pd.to_numeric(s, errors="coerce") <= _to_num(val)]
        elif op == "in":
            vals = val if isinstance(val, list) else [val]
            out = out[s.astype(str).isin([str(v) for v in vals])]
        elif op == "not in":
            vals = val if isinstance(val, list) else [val]
            out = out[~s.astype(str).isin([str(v) for v in vals])]
        elif op == "contains":
            out = out[s.astype(str).str.contains(str(val), na=False, case=False)]
        elif op == "startswith":
            out = out[s.astype(str).str.startswith(str(val), na=False)]
        elif op == "endswith":
            out = out[s.astype(str).str.endswith(str(val), na=False)]
        elif op == "between":
            lo, hi = val if isinstance(val, (list, tuple)) and len(val) == 2 else (None, None)
            x = pd.to_numeric(s, errors="coerce")
            if lo is not None:
                out = out[x >= _to_num(lo)]
            if hi is not None:
                out = out[x <= _to_num(hi)]
        else:
            raise HTTPException(status_code=400, detail=f"filter: unsupported op '{op}'")
    return out


def _to_num(v):
    try:
        return float(v)
    except Exception:
        return pd.NA


def _apply_select(df: pd.DataFrame, selects: list[T.Union[str, dict]]) -> pd.DataFrame:
    if not selects:
        return df
    cols = []
    out = pd.DataFrame(index=df.index)
    for s in selects:
        if isinstance(s, str):
            if s not in df.columns:
                raise HTTPException(status_code=400, detail=f"select: column '{s}' not found")
            out[s] = df[s]
            cols.append(s)
        elif isinstance(s, dict):
            alias = s.get("alias")
            expr = s.get("expr")
            if not alias or not expr:
                continue
            if expr in df.columns:
                out[alias] = df[expr]
            else:
                # very small expression support: same as derive arithmetic
                toks = re.split(r"\s*([+\-*/])\s*", expr)
                if len(toks) == 3 and toks[0] in df.columns and toks[2] in df.columns:
                    a, op, b = toks
                    A = pd.to_numeric(df[a], errors="coerce")
                    B = pd.to_numeric(df[b], errors="coerce")
                    out[alias] = {"+" : A+B, "-" : A-B, "*" : A*B, "/" : A/B.replace({0: pd.NA})}[op]
                else:
                    raise HTTPException(status_code=400, detail=f"select expr unsupported: {expr}")
            cols.append(alias)
        else:
            raise HTTPException(status_code=400, detail="select: invalid entry")
    return out[cols]


def _apply_groupby_agg(df: pd.DataFrame, groupby: list[str], aggregations: list[dict]) -> pd.DataFrame:
    if not aggregations:
        return df
    for g in groupby or []:
        if g not in df.columns:
            raise HTTPException(status_code=400, detail=f"groupby: missing column '{g}'")
    agg_map: dict[str, list[str]] = {}
    for a in aggregations:
        col = a.get("col")
        fn = (a.get("fn") or "sum").lower()
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"aggregate: missing column '{col}'")
        agg_map.setdefault(col, []).append(fn)
    out = df.groupby(groupby or []).agg(agg_map)
    # flatten columns
    out.columns = ["{}_{}".format(c[0], c[1]) if isinstance(c, tuple) else str(c) for c in out.columns.to_list()]
    out = out.reset_index()
    return out


def _apply_sort_limit(df: pd.DataFrame, sort: list[dict], limit: T.Optional[int]) -> pd.DataFrame:
    out = df.copy()
    if sort:
        by = [s["col"] for s in sort if "col" in s]
        ascending = [ (s.get("dir","asc").lower() != "desc") for s in sort ]
        for c in by:
            if c not in out.columns:
                raise HTTPException(status_code=400, detail=f"sort: missing column '{c}'")
        out = out.sort_values(by=by, ascending=ascending)
    if limit and limit > 0:
        out = out.head(int(limit))
    return out


def _apply_window(df: pd.DataFrame, windows: list[dict]) -> pd.DataFrame:
    if not windows:
        return df
    out = df.copy()
    for w in windows:
        alias = w.get("alias")
        expr = (w.get("expr") or "").strip()  # e.g., "pct_change(revenue_sum)"
        part = w.get("partition") or []
        order = w.get("order") or []
        m = re.match(r"^pct_change\(([^)]+)\)$", expr)
        if not (alias and m):
            raise HTTPException(status_code=400, detail=f"window: unsupported expr '{expr}'")
        target = m.group(1)
        if target not in out.columns:
            raise HTTPException(status_code=400, detail=f"window: target column '{target}' missing")
        if order:
            if any(c not in out.columns for c in order):
                raise HTTPException(status_code=400, detail="window: order columns missing")
        def _calc(g: pd.DataFrame) -> pd.Series:
            if order:
                g = g.sort_values(by=order)
            return g[target].pct_change()
        if part:
            out[alias] = out.groupby(part, dropna=False, group_keys=False).apply(_calc)
        else:
            out = out.sort_values(by=order) if order else out
            out[alias] = out[target].pct_change()
    return out


def run_compose(df: pd.DataFrame, params: dict) -> dict:
    # Order: derive -> filter -> select -> groupby/aggregate -> window -> sort/limit
    derives = params.get("derive") or []
    filters = params.get("filter") or []
    selects = params.get("select") or []
    groupby = params.get("groupby") or []
    aggregations = params.get("aggregate") or []
    sort = params.get("sort") or []
    limit = params.get("limit")
    chart = params.get("chart") or {}

    work = _apply_derive(df, derives)
    work = _apply_filters(work, filters)
    work = _apply_select(work, selects) if selects else work
    work = _apply_groupby_agg(work, groupby, aggregations) if aggregations else work
    work = _apply_window(work, params.get("window") or [])
    work = _apply_sort_limit(work, sort, limit)

    result = {"table": work.to_dict(orient="records")}
    if chart:
        # pass through chart intent so the frontend can render appropriately
        result["chart"] = chart
    return result


# --------------------------------------------------------------------------------------
# Ingest: CRM push (CSV / JSON)
# --------------------------------------------------------------------------------------
@app.post("/ingest/push_csv", response_model=UploadResponse)
def ingest_push_csv(
    file: UploadFile = File(...),
    dataset_alias: T.Optional[str] = Form(None),
    has_header: T.Optional[str] = Form(None),
    header_row: T.Optional[int] = Form(None),
    assign_columns_json: T.Optional[str] = Form(None),
):
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
    dsid = str(uuid.uuid4())
    DATASETS[dsid] = {"df": df, "meta": {"filename": file.filename, "alias": dataset_alias}}
    if dataset_alias:
        ALIASES[dataset_alias] = dsid
    return UploadResponse(dataset_id=dsid, rows=len(df), columns=df.shape[1], alias=dataset_alias)


@app.post("/ingest/push_json", response_model=UploadResponse)
def ingest_push_json(req: IngestJSONRequest, dataset_alias: T.Optional[str] = None):
    df = pd.DataFrame(req.records or [])
    df = coerce_numeric_columns(df)
    dsid = str(uuid.uuid4())
    DATASETS[dsid] = {"df": df, "meta": {"filename": "api-json", "alias": dataset_alias}}
    if dataset_alias:
        ALIASES[dataset_alias] = dsid
    return UploadResponse(dataset_id=dsid, rows=len(df), columns=df.shape[1], alias=dataset_alias)


# --------------------------------------------------------------------------------------
# Ingest: SFTP pull (for nightly drops)
# --------------------------------------------------------------------------------------
class SFTPPullRequest(BaseModel):
    remote_path: str
    dataset_alias: T.Optional[str] = None
    host: T.Optional[str] = None
    port: int = 22
    username: T.Optional[str] = None
    password: T.Optional[str] = None
    pkey_b64: T.Optional[str] = None  # base64-encoded private key (optional)


@app.post("/ingest/pull_sftp", response_model=UploadResponse)
def ingest_pull_sftp(req: SFTPPullRequest):
    if not _HAS_SFTP:
        raise HTTPException(status_code=501, detail="paramiko not installed on server")

    host = req.host or os.getenv("SFTP_HOST")
    port = req.port or int(os.getenv("SFTP_PORT", "22"))
    username = req.username or os.getenv("SFTP_USERNAME")
    password = req.password or os.getenv("SFTP_PASSWORD")
    pkey_b64 = req.pkey_b64 or os.getenv("SFTP_PKEY_B64")

    if not host or not username or not (password or pkey_b64):
        raise HTTPException(status_code=400, detail="SFTP credentials missing")

    key = None
    if pkey_b64:
        key_data = base64.b64decode(pkey_b64)
        key = paramiko.RSAKey.from_private_key(io.StringIO(key_data.decode("utf-8")))

    transport = paramiko.Transport((host, int(port)))
    try:
        if key:
            transport.connect(username=username, pkey=key)
        else:
            transport.connect(username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)
        with sftp.open(req.remote_path, "rb") as fh:
            blob = fh.read()
    finally:
        transport.close()

    df = _read_any_to_df(req.remote_path, blob)
    df = coerce_numeric_columns(df)
    dsid = str(uuid.uuid4())
    DATASETS[dsid] = {"df": df, "meta": {"filename": os.path.basename(req.remote_path), "alias": req.dataset_alias}}
    if req.dataset_alias:
        ALIASES[req.dataset_alias] = dsid
    return UploadResponse(dataset_id=dsid, rows=len(df), columns=df.shape[1], alias=req.dataset_alias)


# --------------------------------------------------------------------------------------
# Fix headers AFTER upload
# --------------------------------------------------------------------------------------
@app.post("/datasets/assign_headers")
def datasets_assign_headers(req: AssignHeadersRequest):
    if req.dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="dataset not found")
    df = DATASETS[req.dataset_id]["df"].copy()

    if req.header_row is not None:
        i = int(req.header_row)
        if i < 0 or i >= len(df):
            raise HTTPException(status_code=400, detail="header_row out of range")
        hdr = df.iloc[i].astype(str).tolist()
        df = df.drop(index=df.index[i]).reset_index(drop=True)
        df.columns = hdr
    elif req.columns is not None:
        if len(req.columns) != df.shape[1]:
            raise HTTPException(status_code=400, detail=f"columns length {len(req.columns)} != width {df.shape[1]}")
        df.columns = [str(c) for c in req.columns]
    elif req.rename is not None:
        missing = [k for k in req.rename.keys() if k not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"rename keys not found: {missing}")
        df = df.rename(columns=req.rename)
    else:
        raise HTTPException(status_code=400, detail="provide one of: header_row | columns | rename")

    df = coerce_numeric_columns(df)
    DATASETS[req.dataset_id]["df"] = df
    return {"ok": True, "columns": list(df.columns)}


# --------------------------------------------------------------------------------------
# Templates & Jobs (Render Cron safe)
# --------------------------------------------------------------------------------------
@app.post("/templates/upsert")
def templates_upsert(tpl: TemplateUpsert):
    TEMPLATES[tpl.template_id] = tpl.model_dump()
    return {"ok": True, "template_id": tpl.template_id}


@app.post("/jobs/run-template")
def jobs_run_template(req: RunTemplateRequest):
    tpl = TEMPLATES.get(req.template_id)
    if not tpl:
        raise HTTPException(status_code=404, detail="template not found")

    src = tpl["source"]
    dsid = None
    if src.get("mode") == "latest_dataset_id":
        dsid = src.get("dataset_id")
    elif src.get("mode") == "alias":
        dsid = ALIASES.get(src.get("alias"))

    if not dsid or dsid not in DATASETS:
        raise HTTPException(status_code=404, detail="dataset not found for template")

    df = DATASETS[dsid]["df"]

    spec = tpl["report_spec"]
    result = run_analysis_from_spec(df, spec)

    attachments: list[tuple[str, str, bytes]] = []

    if "csv" in (tpl["delivery"].get("formats") or []):
        csv_bytes = _result_to_csv_bytes(result)
        attachments.append(("report.csv", "text/csv", csv_bytes))

    if "png" in (tpl["delivery"].get("formats") or []):
        png_bytes = _result_to_chart_png(result)
        if png_bytes:
            attachments.append(("chart.png", "image/png", png_bytes))

    to_list = tpl["delivery"].get("email") or []
    subject = tpl["delivery"].get("subject") or f"Reverie Report: {tpl['name']}"
    body = f"Report: {tpl['name']}\nTemplate: {req.template_id}\nDataset: {dsid}\nGenerated: {datetime.utcnow().isoformat()}Z"
    _send_email(to_list, subject, body, attachments)
    return {"ok": True, "sent_to": to_list, "attachments": [a[0] for a in attachments]}


def _result_to_csv_bytes(result: dict) -> bytes:
    if "table" in result and isinstance(result["table"], list):
        df = pd.DataFrame(result["table"])
    elif "preview" in result and isinstance(result["preview"], list):
        df = pd.DataFrame(result["preview"])
    else:
        df = pd.json_normalize(result)
    return df.to_csv(index=False).encode("utf-8")


def _result_to_chart_png(result: dict) -> T.Optional[bytes]:
    if not _HAS_MPL:
        return None
    try:
        fig, ax = plt.subplots(figsize=(6, 3))
        table = None
        if "table" in result and isinstance(result["table"], list) and result["table"]:
            table = pd.DataFrame(result["table"])
            if table.shape[1] >= 2:
                x = table.columns[0]
                y = next((c for c in table.columns[1:] if pd.api.types.is_numeric_dtype(table[c])), None)
                if y:
                    table.plot(kind="line", x=x, y=y, ax=ax)
        if table is None:
            ax.text(0.5, 0.5, "Chart placeholder", ha="center", va="center")
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        return buf.getvalue()
    except Exception:
        return None


def _send_email(to_list: list[str], subject: str, body: str, files: list[tuple[str, str, bytes]]):
    if not to_list:
        return
    host = os.getenv("EMAIL_SMTP_HOST")
    port = int(os.getenv("EMAIL_SMTP_PORT", "587"))
    user = os.getenv("EMAIL_USERNAME")
    pwd = os.getenv("EMAIL_PASSWORD")
    sender = os.getenv("EMAIL_FROM", user)

    if not all([host, port, user, pwd, sender]):
        return

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = ", ".join(to_list)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    for fname, mime, blob in files:
        main, sub = (mime.split("/", 1) + ["octet-stream"])[:2]
        part = MIMEBase(main, sub)
        part.set_payload(blob)
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{fname}"')
        msg.attach(part)

    context = ssl.create_default_context()
    with smtplib.SMTP(host, port) as server:
        server.starttls(context=context)
        server.login(user, pwd)
        server.sendmail(sender, to_list, msg.as_string())


# --------------------------------------------------------------------------------------
# Feedback
# --------------------------------------------------------------------------------------
@app.post("/feedback")
def post_feedback(item: FeedbackItem):
    FEEDBACK.append(item.model_dump())
    return {"ok": True, "count": len(FEEDBACK)}


@app.get("/feedback/debug")
def get_feedback_debug(limit: int = 50):
    return FEEDBACK[-limit:]
