# utils_io.py
import io
import csv
import pandas as pd

ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin1", "utf-16"]
SEPS = [",", ";", "\t", "|"]

def load_table(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Robust loader:
      • Excel first (.xlsx/.xls)
      • CSV: sniff delimiter, try multiple encodings
      • Fallback: brute-force encodings × separators
      • Last resort: split single column by comma and use first row as header
    """
    name = (filename or "").lower()

    # Excel first (needs openpyxl)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")

    # CSV path
    sample = file_bytes[:8192].decode("utf-8", errors="ignore")
    try:
        guessed = csv.Sniffer().sniff(sample, delimiters="".join(SEPS)).delimiter
    except csv.Error:
        guessed = None

    # 1) guessed delimiter across encodings
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), engine="python", sep=guessed, encoding=enc)
            if df.shape[1] > 2:
                return df
        except Exception:
            pass

    # 2) brute-force
    for enc in ENCODINGS:
        for sep in SEPS:
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), engine="python", sep=sep, encoding=enc)
                if df.shape[1] > 2:
                    return df
            except Exception:
                pass

    # 3) last resort: split the single column by comma
    df = pd.read_csv(io.BytesIO(file_bytes), engine="python", sep=None, encoding="latin1")
    if df.shape[1] == 1:
        parts = df.iloc[:, 0].astype(str).str.split(",", expand=True)
        if parts.shape[0] > 1:
            parts.columns = parts.iloc[0]
            parts = parts.iloc[1:].reset_index(drop=True)
        return parts

    return df
