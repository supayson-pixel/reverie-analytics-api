import io
import os
import csv
import ssl
import uuid
import json
import math
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


# --------------------------------------------------------------------------------------
# App & CORS
# --------------------------------------------------------------------------------------
app = FastAPI(title="Reverie Analytics API")

# For MVP keep CORS permissive. Lock down later via env.
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
# Utilities: parsing, typing, summaries
# --------------------------------------------------------------------------------------
_NUMERIC_LIKE = ("int", "float", "Int", "Float", "complex")
_CURRENCY_CHARS = pd.Series(list("$€£¥,()%")).apply(lambda c: "\\" + c).tolist()  # for regex join
_CURRENCY_REGEX = r"[$€£¥,\(\)%\s]"

def _detect_delimiter(text: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(text[:2000])
        return dialect.delimiter
    except Exception:
        return ","


def _read_any_to_df(file_name: str, blob: bytes) -> pd.DataFrame:
    name = (file_name or "").lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(blob))
    text = blob.decode("utf-8", errors="ignore")
    sep = _detect_delimiter(text)
    return pd.read_csv(io.StringIO(text), sep=sep)


def _to_numeric_clean_series(s: pd.Series) -> pd.Series:
    """Coerce a text series with currency/commas/percent to float."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    # Strip currency/symbols; handle parentheses as negatives.
    txt = s.astype(str)
    # Handle (123.45) -> -123.45
    neg = txt.str.contains(r"^\s*\(.*\)\s*$", regex=True, na=False)
    cleaned = txt.str.replace(_CURRENCY_REGEX, "", regex=True)
    out = pd.to_numeric(cleaned, errors="coerce")
    out[neg] = -out[neg]
    return out


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # try to coerce any object-like column that looks numeric-ish
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_numeric_dtype(s):
            continue
        sample = s.astype(str).str.replace(_CURRENCY_REGEX, "", regex=True).str.replace(".", "", n=1, regex=False)
        # simple heuristic: at least 60% rows look like digits
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
    desc = desc.rename(
        columns={"count": "count", "mean": "mean", "std": "std", "min": "min", "25%": "25%", "50%": "50%", "75%": "75%", "max": "max"}
    )
    for col, row in desc.iterrows():
        res[col] = {k: (round(v, round_to) if isinstance(v, (int, float)) and not pd.isna(v) else None) for k, v in row.items()}
    # ensure integer count
    for col in res:
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
    # exclude columns that are really dates
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
    type: str = Field(..., description="summary | pivot | timeseries")
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


# --------------------------------------------------------------------------------------
# Root & Health
# --------------------------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Reverie Analytics API. Try /health, /analytics/upload, /analytics/profile, /ingest/*, /templates/upsert, /jobs/run-template"}


@app.get("/health")
def health():
    return {"ok": True, "time": datetime.utcnow().isoformat() + "Z"}


# --------------------------------------------------------------------------------------
# Core: upload, profile, analyze
# --------------------------------------------------------------------------------------
@app.post("/analytics/upload", response_model=UploadResponse)
def analytics_upload(file: UploadFile = File(...), dataset_alias: T.Optional[str] = Form(None)):
    try:
        blob = file.file.read()
        df = _read_any_to_df(file.filename, blob)
        df = coerce_numeric_columns(df)
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

    # summaries
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

    return {"message": f"unknown analysis type '{t}'"}


# --------------------------------------------------------------------------------------
# Ingest: CRM push (CSV / JSON)
# --------------------------------------------------------------------------------------
@app.post("/ingest/push_csv", response_model=UploadResponse)
def ingest_push_csv(file: UploadFile = File(...), dataset_alias: T.Optional[str] = Form(None)):
    blob = file.file.read()
    df = _read_any_to_df(file.filename, blob)
    df = coerce_numeric_columns(df)
    dsid = str(uuid.uuid4())
    DATASETS[dsid] = {"df": df, "meta": {"filename": file.filename, "alias": dataset_alias}}
    if dataset_alias:
        ALIASES[dataset_alias] = dsid
    return UploadResponse(dataset_id=dsid, rows=len(df), columns=df.shape[1], alias=dataset_alias)


@app.post("/ingest/push_json", response_model=UploadResponse)
def ingest_push_json(req: IngestJSONRequest):
    df = pd.DataFrame(req.records or [])
    df = coerce_numeric_columns(df)
    dsid = str(uuid.uuid4())
    DATASETS[dsid] = {"df": df, "meta": {"filename": "api-json"}}
    return UploadResponse(dataset_id=dsid, rows=len(df), columns=df.shape[1])


# --------------------------------------------------------------------------------------
# Ingest: SFTP pull (for nightly drops)
# --------------------------------------------------------------------------------------
class SFTPPullRequest(BaseModel):
    remote_path: str
    dataset_alias: T.Optional[str] = None
    # Optional override; if omitted we take env vars.
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

    # Connect
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

    # Parse CSV/Excel
    df = _read_any_to_df(req.remote_path, blob)
    df = coerce_numeric_columns(df)
    dsid = str(uuid.uuid4())
    DATASETS[dsid] = {"df": df, "meta": {"filename": os.path.basename(req.remote_path), "alias": req.dataset_alias}}
    if req.dataset_alias:
        ALIASES[req.dataset_alias] = dsid

    return UploadResponse(dataset_id=dsid, rows=len(df), columns=df.shape[1], alias=req.dataset_alias)


# --------------------------------------------------------------------------------------
# Templates & Jobs (for Render Cron)
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

    # Resolve dataset
    src = tpl["source"]
    dsid = None
    if src.get("mode") == "latest_dataset_id":
        dsid = src.get("dataset_id")
    elif src.get("mode") == "alias":
        dsid = ALIASES.get(src.get("alias"))

    if not dsid or dsid not in DATASETS:
        raise HTTPException(status_code=404, detail="dataset not found for template")

    df = DATASETS[dsid]["df"]

    # Run analysis
    spec = tpl["report_spec"]
    result = run_analysis_from_spec(df, spec)

    # Build artifacts
    attachments: list[tuple[str, str, bytes]] = []

    # CSV attachment
    if "csv" in (tpl["delivery"].get("formats") or []):
        csv_bytes = _result_to_csv_bytes(result)
        attachments.append(("report.csv", "text/csv", csv_bytes))

    # Optional chart
    if "png" in (tpl["delivery"].get("formats") or []):
        png_bytes = _result_to_chart_png(result)
        if png_bytes:
            attachments.append(("chart.png", "image/png", png_bytes))

    # Email
    to_list = tpl["delivery"].get("email") or []
    subject = tpl["delivery"].get("subject") or f"Reverie Report: {tpl['name']}"
    body = f"Report: {tpl['name']}\nTemplate: {req.template_id}\nDataset: {dsid}\nGenerated: {datetime.utcnow().isoformat()}Z"
    _send_email(to_list, subject, body, attachments)
    return {"ok": True, "sent_to": to_list, "attachments": [a[0] for a in attachments]}


def _result_to_csv_bytes(result: dict) -> bytes:
    # Try to find a table-like structure
    if "table" in result and isinstance(result["table"], list):
        df = pd.DataFrame(result["table"])
    elif "preview" in result and isinstance(result["preview"], list):
        df = pd.DataFrame(result["preview"])
    else:
        # last resort: flatten json
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
            # naive pick: last numeric column over first column if chartable
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
        # Fail quietly in MVP (or raise)
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
