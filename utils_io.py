# utils_io.py
import io
import csv
import pandas as pd

ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin1", "utf-16"]
SEPS = [",", ";", "\t", "|"]

def _strip_header_noise(cols):
    clean = []
    for c in list(cols):
        s = str(c)
        s = s.lstrip("\ufeff").strip().strip('"').strip("'").strip()
        clean.append(s)
    return clean

def _maybe_split_first_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df.columns = _strip_header_noise(df.columns)

    looks_singlecol = df.shape[1] == 1
    if not looks_singlecol and df.shape[1] > 1:
        others_null_frac = df.iloc[:, 1:].isna().mean().mean()
        looks_singlecol = others_null_frac > 0.95

    first = df.iloc[:, 0].astype(str)
    delim = None
    if first.str.contains(",").mean() > 0.6: delim = ","
    elif first.str.contains(";").mean() > 0.6: delim = ";"

    if looks_singlecol and delim:
        first_clean = first.str.replace(r'^\s*"\s*|\s*"\s*$', "", regex=True)
        parts = first_clean.str.split(delim, expand=True)
        header_tokens_from_name = [t.strip() for t in str(df.columns[0]).split(",")]
        header_tokens_from_name = _strip_header_noise(header_tokens_from_name)
        if len(header_tokens_from_name) == parts.shape[1] and all(header_tokens_from_name):
            parts.columns = header_tokens_from_name
        else:
            if parts.shape[0] > 1:
                guessed = _strip_header_noise(parts.iloc[0].astype(str).tolist())
                if len(guessed) == parts.shape[1]:
                    parts.columns = guessed
                    parts = parts.iloc[1:].reset_index(drop=True)
        return parts
    return df

def load_table(file_bytes: bytes, filename: str) -> pd.DataFrame:
    name = (filename or "").lower()

    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl", dtype=object)
        return _maybe_split_first_column(df)

    sample = file_bytes[:8192].decode("utf-8", errors="ignore")
    try:
        guessed_sep = csv.Sniffer().sniff(sample, delimiters="".join(SEPS)).delimiter
    except csv.Error:
        guessed_sep = None

    for enc in ENCODINGS:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), engine="python", sep=guessed_sep, encoding=enc, dtype=object)
            df = _maybe_split_first_column(df)
            if df.shape[1] >= 1:
                return df
        except Exception:
            pass

    for enc in ENCODINGS:
        for sep in SEPS:
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), engine="python", sep=sep, encoding=enc, dtype=object)
                df = _maybe_split_first_column(df)
                if df.shape[1] >= 1:
                    return df
            except Exception:
                pass

    df = pd.read_csv(io.BytesIO(file_bytes), engine="python", sep=None, encoding="latin1", dtype=object)
    return _maybe_split_first_column(df)
