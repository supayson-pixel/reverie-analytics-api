# utils_io.py
import io
import csv
import pandas as pd

ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin1", "utf-16"]
SEPS = [",", ";", "\t", "|"]

def load_table(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Robust loader:
      • Excel first (.xlsx/.xls via openpyxl)
      • CSV: sniff delimiter, try multiple encodings
      • Fallback: brute-force encodings × separators
      • Last resort: split a single text column by comma
    """
    name = (filename or "").lower()

    # ---------- Excel path ----------
    if name.endswith((".xlsx", ".xls")):
        # Read as objects so we can examine text
        df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl", dtype=object)

        # If everything is in one column (common when cells contain comma-joined text),
        # auto-split by comma and set headers intelligently.
        if df.shape[1] == 1:
            only_col = df.columns[0]
            s = df[only_col].astype(str)

            # Many rows contain commas? Then it's likely CSV-in-a-cell.
            if s.str.contains(",").mean() > 0.6:
                parts = s.str.split(",", expand=True)

                # Prefer header from the original column name (e.g. "date,customer,...")
                header_tokens = [t.strip() for t in str(only_col).split(",")]
                if len(header_tokens) > 1 and len(header_tokens) == parts.shape[1]:
                    parts.columns = header_tokens
                else:
                    # Fallback: use first row as header if it looks header-ish
                    if parts.shape[0] > 1:
                        parts.columns = parts.iloc[0]
                        parts = parts.iloc[1:].reset_index(drop=True)

                return parts

        return df

    # ---------- CSV path ----------
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

    # 2) brute-force encodings × separators
    for enc in ENCODINGS:
        for sep in SEPS:
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), engine="python", sep=sep, encoding=enc)
                if df.shape[1] > 2:
                    return df
            except Exception:
                pass

    # 3) last resort: read loosely, then split single column by comma
    df = pd.read_csv(io.BytesIO(file_bytes), engine="python", sep=None, encoding="latin1")
    if df.shape[1] == 1:
        s = df.iloc[:, 0].astype(str)
        parts = s.str.split(",", expand=True)
        if parts.shape[0] > 1:
            parts.columns = parts.iloc[0]
            parts = parts.iloc[1:].reset_index(drop=True)
        return parts

    return df
