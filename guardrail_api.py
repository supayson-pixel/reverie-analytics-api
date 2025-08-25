# guardrail_api.py
# Universal analytics API for arbitrary tabular uploads.
# FastAPI + Pandas + NumPy only.

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Literal, Any
import pandas as pd
import numpy as np
import io, uuid, re

# -----------------------------
# App & CORS
# -----------------------------
app = FastAPI(title="Reverie Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.reveriesun.com",
        "https://reveriesun.com",
        "https://reveriesun.netlify.app",
        "https://inspiring-tarsier-97b2c7.netlify.app",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# In-memory datastore
# -----------------------------
_DATA: Dict[str, pd.DataFrame] = {}

# -----------------------------
# Utilities: parsing & coercion
# -----------------------------
_CURRENCY_CHARS = re.compile(r"[\$\€\£\¥]")
_GROUPING_CHARS = re.compile(r"[,_ ]")
_PERCENT = re.compile(r"%")

def to_numeric_clean(s: pd.Series) -> pd.Series:
    # Work on string view; keep NaN if not convertible
    st = s.astype("string")
    # parentheses as negatives
    st = st.str.replace(r"^\(\s*(.*)\s*\)$", r"-\1", regex=True)
    # strip currency + grouping
    st = st.str.replace(_CURRENCY_CHARS, "", regex=True)
    st = st.str.replace(_GROUPING_CHARS, "", regex=True)
    # percent to fraction
    is_pct = st.str.contains(_PERCENT, na=False)
    st = st.str.replace(_PERCENT, "", regex=True)
    num = pd.to_numeric(st, errors="coerce")
    num.loc[is_pct] = num.loc[is_pct] / 100.0
    return num

def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        # adopt if at least half the rows convert cleanly
        cand = to_numeric_clean(out[col])
        if cand.notna().mean() >= 0.5:
            out[col] = cand
    return out

def detect_delimiter(text: str) -> str:
    # try most common delimiters
    cands = [",", "\t", ";", "|"]
    counts = {d: text.count(d) for d in cands}
    return max(counts, key=counts.get) if max(counts.values()) > 0 else ","

def read_any_table(file: UploadFile) -> pd.DataFrame:
    name = (file.filename or "").lower()
    raw = file.file.read()
    if not raw:
        raise HTTPException(400, "Empty upload")
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw))
    # CSV-like
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1", errors="ignore")
    delim = detect_delimiter(text)
    return pd.read_csv(io.StringIO(text), sep=delim, engine="python")

def describe_numeric(df: pd.DataFrame, round_to: int = 2) -> Dict[str, Dict[str, float]]:
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return {}
    desc = num_df.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
    out: Dict[str, Dict[str, float]] = {}
    for stat, cols in desc.items():
        for col, val in cols.items():
            out.setdefault(col, {})[stat] = None if pd.isna(val) else (
                round(float(val), round_to) if isinstance(val, (int, float, np.floating)) else val
            )
    # keep keys consistent
    rename = {"25%": "25%", "50%": "50%", "75%": "75%", "min": "min", "max": "max", "mean": "mean", "std": "std", "count": "count"}
    for col in list(out.keys()):
        out[col] = {rename.get(k, k): v for k, v in out[col].items()}
    return out

def date_summary(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for col in df.columns:
        dt = pd.to_datetime(df[col], errors="coerce")
        if dt.notna().sum() >= max(3, int(0.5 * len(dt))):  # mostly parseable
            s = dt.dropna()
            by_month = s.dt.to_period("M").value_counts().sort_index()
            by_wday = s.dt.day_name().value_counts().reindex(
                ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], fill_value=0
            )
            out[col] = {
                "min": None if s.empty else str(s.min().date()),
                "max": None if s.empty else str(s.max().date()),
                "span_days": None if s.empty else int((s.max() - s.min()).days),
                "by_month": {str(k): int(v) for k, v in by_month.items()},
                "by_weekday": {k: int(v) for k, v in by_wday.items()},
            }
    return out

def resample_series(df: pd.DataFrame, date_col: str, value_col: str, how: str, freq: str) -> pd.Series:
    dt = pd.to_datetime(df[date_col], errors="coerce")
    val = pd.to_numeric(df[value_col], errors="coerce")
    tmp = pd.DataFrame({"dt": dt, "val": val}).dropna()
    if tmp.empty:
        return pd.Series([], dtype=float)
    tmp = tmp.set_index("dt").sort_index()
    if how == "mean":
        s = tmp["val"].resample(freq).mean()
    elif how == "median":
        s = tmp["val"].resample(freq).median()
    else:
        s = tmp["val"].resample(freq).sum()
    # fill missing with 0 for sum, else carry NaN (chart can show gaps)
    if how == "sum":
        s = s.fillna(0.0)
    return s

# -----------------------------
# Models
# -----------------------------
class AnalyzeRequest(BaseModel):
    dataset_id: str
    type: Literal[
        "summary", "distribution", "correlation", "regression", "pivot", "preview",
        "forecast", "control_chart", "bump"
    ]
    # generic params
    columns: Optional[List[str]] = None
    target: Optional[str] = None
    features: Optional[List[str]] = None
    bins: Optional[int] = 10
    groupby: Optional[List[str]] = None
    values: Optional[List[str]] = None
    agg: Optional[Literal["sum","mean","count","min","max","median"]] = "sum"
    limit: Optional[int] = 20
    offset: Optional[int] = 0
    round_to: Optional[int] = 2

    # time-series / chart params
    date_col: Optional[str] = None
    value_col: Optional[str] = None
    category_col: Optional[str] = None
    freq: Optional[Literal["D","W","M"]] = "D"
    horizon: Optional[int] = 7
    method: Optional[Literal["linreg","ma"]] = "linreg"
    ma_window: Optional[int] = 7
    top_k: Optional[int] = 10

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"message": "Reverie Analytics API. Try /health, /analytics/upload, /analytics/profile, /analytics/analyze."}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/analytics/upload")
async def upload(file: UploadFile = File(...)):
    df = read_any_table(file)
    df.columns = [str(c).strip() for c in df.columns]
    dataset_id = str(uuid.uuid4())
    _DATA[dataset_id] = df
    return {
        "dataset_id": dataset_id,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "columns_list": list(map(str, df.columns)),
    }

@app.get("/analytics/profile")
def profile(dataset_id: str, round_to: int = 2):
    if dataset_id not in _DATA:
        raise HTTPException(404, "dataset_id not found")
    raw = _DATA[dataset_id]
    df = coerce_numeric_columns(raw)
    info = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": list(map(str, df.columns)),
        "nulls": {c: int(df[c].isna().sum()) for c in df.columns},
        "numeric_summary": describe_numeric(df, round_to=round_to),
        "date_summary": date_summary(raw),
        "preview": df.head(10).to_dict(orient="records"),
    }
    return info

@app.get("/analytics/describe")
def describe(dataset_id: str):
    if dataset_id not in _DATA:
        raise HTTPException(404, "dataset_id not found")
    df = _DATA[dataset_id]
    return {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": list(map(str, df.columns)),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "nulls": {c: int(df[c].isna().sum()) for c in df.columns},
    }

@app.post("/analytics/analyze")
def analyze(req: AnalyzeRequest):
    if req.dataset_id not in _DATA:
        raise HTTPException(404, "dataset_id not found")
    raw = _DATA[req.dataset_id]
    df = coerce_numeric_columns(raw)
    r = req.round_to or 2

    # -------- baseline analyses --------
    if req.type == "summary":
        cols = req.columns or list(df.select_dtypes(include=[np.number]).columns)
        if not cols:
            return {"type": "summary", "result": {}}
        return {"type": "summary", "result": describe_numeric(df[cols], round_to=r)}

    if req.type == "distribution":
        if not req.columns or len(req.columns) != 1:
            raise HTTPException(400, "distribution requires columns=[one numeric column]")
        col = req.columns[0]
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            return {"type": "distribution", "result": {"bins": [], "edges": []}}
        bins = req.bins or 10
        counts, edges = np.histogram(series, bins=bins)
        return {"type": "distribution", "result": {"bins": counts.tolist(), "edges": [round(float(x), r) for x in edges]}}

    if req.type == "correlation":
        cols = req.columns or list(df.select_dtypes(include=[np.number]).columns)
        corr = df[cols].corr(method="pearson").round(r).replace({np.nan: None})
        return {"type": "correlation", "result": corr.to_dict()}

    if req.type == "regression":
        if not req.target or not req.features:
            raise HTTPException(400, "regression requires target and features")
        y = pd.to_numeric(df[req.target], errors="coerce")
        X = pd.DataFrame({f: pd.to_numeric(df[f], errors="coerce") for f in req.features})
        data = pd.concat([y, X], axis=1).dropna()
        if data.empty:
            return {"type": "regression", "result": {"rows_used": 0, "coefficients": {}}}
        Y = data.iloc[:, 0].values.astype(float)
        A = data.iloc[:, 1:].values.astype(float)
        A1 = np.c_[np.ones(A.shape[0]), A]
        beta, *_ = np.linalg.lstsq(A1, Y, rcond=None)
        y_hat = A1 @ beta
        ss_res = float(np.sum((Y - y_hat) ** 2))
        ss_tot = float(np.sum((Y - Y.mean()) ** 2))
        r2 = 0.0 if ss_tot == 0 else (1.0 - ss_res / ss_tot)
        coefs = {"intercept": round(float(beta[0]), r)}
        for i, f in enumerate(req.features, start=1):
            coefs[f] = round(float(beta[i]), r)
        return {"type": "regression", "result": {"rows_used": int(A.shape[0]), "r2": round(r2, r), "coefficients": coefs}}

    if req.type == "pivot":
        if not req.groupby or not req.values:
            raise HTTPException(400, "pivot requires groupby and values")
        agg = req.agg or "sum"
        gb = raw.groupby(req.groupby, dropna=False)[req.values].agg(agg)
        out = gb.reset_index()
        return {"type": "pivot", "result": {"columns": list(out.columns), "rows": out.to_dict(orient="records")}}

    if req.type == "preview":
        start = max(0, req.offset or 0)
        end = start + max(1, req.limit or 20)
        view = raw.iloc[start:end]
        return {"type": "preview", "result": {"columns": list(view.columns), "rows": view.to_dict(orient="records"), "offset": start, "limit": end-start, "total": int(raw.shape[0])}}

    # -------- new: time-series forecast --------
    if req.type == "forecast":
        if not req.date_col or not req.value_col:
            raise HTTPException(400, "forecast requires date_col and value_col")
        how = req.agg or "sum"
        freq = req.freq or "D"
        h = int(req.horizon or 7)
        s = resample_series(raw, req.date_col, req.value_col, how, freq).dropna()
        if len(s) < 3:
            return {"type": "forecast", "result": {"history": [], "forecast": []}}
        # index -> numeric x
        x = np.arange(len(s), dtype=float)
        y = s.values.astype(float)

        if req.method == "ma":
            # simple moving average baseline
            w = int(req.ma_window or 7)
            if w < 2:
                w = 2
            y_hat_hist = pd.Series(y).rolling(window=min(w, len(y)), min_periods=1).mean().values
            slope = 0.0
            intercept = float(y_hat_hist[-1])
        else:
            # linear regression on time
            A = np.c_[np.ones_like(x), x]
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            intercept = float(beta[0])
            slope = float(beta[1])
            y_hat_hist = (A @ beta)

        # residual std for CI
        resid = y - y_hat_hist
        sd = float(np.std(resid, ddof=1)) if len(resid) > 2 else 0.0

        # build future timeline
        last_ts = s.index[-1]
        future_idx = pd.date_range(last_ts, periods=h+1, freq=freq, inclusive="right")
        x_future = np.arange(len(x), len(x) + len(future_idx), dtype=float)
        y_future = intercept + slope * x_future
        # if MA, keep flat at last MA value
        if req.method == "ma":
            y_future = np.full_like(x_future, fill_value=y_hat_hist[-1], dtype=float)

        history = [{"ts": ts.isoformat(), "y": round(float(val), r)} for ts, val in zip(s.index, y)]
        forecast = [{"ts": ts.isoformat(), "yhat": round(float(val), r), "lcl": round(float(val - 1.96 * sd), r), "ucl": round(float(val + 1.96 * sd), r)} for ts, val in zip(future_idx, y_future)]
        return {"type": "forecast", "result": {"history": history, "forecast": forecast, "method": req.method, "sd": round(sd, r)}}

    # -------- new: control chart (X-bar) --------
    if req.type == "control_chart":
        if not req.date_col or not req.value_col:
            raise HTTPException(400, "control_chart requires date_col and value_col")
        how = req.agg or "mean"
        freq = req.freq or "D"
        s = resample_series(raw, req.date_col, req.value_col, how, freq).dropna()
        if len(s) < 3:
            return {"type": "control_chart", "result": {"points": [], "ucl": None, "lcl": None, "mean": None}}
        y = s.values.astype(float)
        mu = float(np.mean(y))
        sigma = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
        ucl = mu + 3.0 * sigma
        lcl = mu - 3.0 * sigma
        pts = [{"ts": ts.isoformat(), "y": round(float(val), r), "ooc": bool(val > ucl or val < lcl)} for ts, val in zip(s.index, y)]
        return {
            "type": "control_chart",
            "result": {
                "points": pts,
                "mean": round(mu, r),
                "ucl": round(ucl, r),
                "lcl": round(lcl, r),
            },
        }

    # -------- new: bump chart (rank over time) --------
    if req.type == "bump":
        if not req.date_col or not req.category_col or not req.value_col:
            raise HTTPException(400, "bump requires date_col, category_col, value_col")
        how = req.agg or "sum"
        freq = req.freq or "M"
        top_k = int(req.top_k or 10)

        dt = pd.to_datetime(raw[req.date_col], errors="coerce")
        cat = raw[req.category_col].astype("string")
        val = pd.to_numeric(raw[req.value_col], errors="coerce")
        tmp = pd.DataFrame({"dt": dt, "cat": cat, "val": val}).dropna()
        if tmp.empty:
            return {"type": "bump", "result": {"periods": [], "series": {}, "max_rank": 0}}

        # aggregate per period & category
        tmp["period"] = tmp["dt"].dt.to_period(freq).dt.to_timestamp()
        agg_df = tmp.groupby(["period", "cat"])["val"].agg({"sum": "sum", "mean": "mean", "median": "median"}.get(how, "sum")).reset_index()
        pivot = agg_df.pivot(index="period", columns="cat", values="val").fillna(0.0).sort_index()

        # choose top categories overall to keep chart readable
        overall = pivot.sum(axis=0).sort_values(ascending=False)
        keep = set(overall.head(top_k).index)
        pivot = pivot.loc[:, [c for c in pivot.columns if c in keep]]

        # compute ranks per period (1 = highest)
        ranks = pivot.rank(axis=1, method="min", ascending=False)
        max_rank = int(ranks.max().max()) if not ranks.empty else 0

        periods = [ts.isoformat() for ts in ranks.index.to_list()]
        series = {str(cat): [None if pd.isna(v) else int(v) for v in ranks[cat].tolist()] for cat in ranks.columns}
        return {"type": "bump", "result": {"periods": periods, "series": series, "max_rank": max_rank}}

    raise HTTPException(400, f"Unknown analysis type {req.type}")

@app.post("/analytics/export")
def export_csv(req: AnalyzeRequest):
    res = analyze(req)
    t = res.get("type")
    data = res.get("result", {})
    # Normalize to table for CSV
    if t in ("pivot", "preview"):
        cols = data.get("columns", [])
        rows = data.get("rows", [])
        df = pd.DataFrame(rows, columns=cols)
    elif t == "summary":
        rows = [{"column": col, **stats} for col, stats in data.items()]
        df = pd.DataFrame(rows)
    elif t == "correlation":
        df = pd.DataFrame(data)
    else:
        return JSONResponse({"ok": False, "message": "CSV export available for preview/pivot/summary/correlation only."})
    out = io.StringIO()
    df.to_csv(out, index=False)
    out.seek(0)
    return StreamingResponse(iter([out.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=analysis.csv"})
