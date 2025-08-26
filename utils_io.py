# utils_io.py
import io
import csv
import pandas as pd

# Encodings & delimiters we try for CSVs
ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin1", "utf-16"]
SEPS = [",", ";", "\t", "|"]


def _strip_header_noise(cols):
    """
    Remove BOMs and stray quotes/spaces from header names.
    """
    clean = []
    for c in list(cols):
        s = str(c)
        s = s.lstrip("\ufeff").strip().strip('"').strip("'").strip()
        clean.append(s)
    return clean


def _maybe_split_first_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    If the table is effectively one text column (or other columns are almost all empty)
    and that first column contains comma- (or semicolon-) separated text, split it into
    multiple columns and set a sensible header.

    Also cleans header noise even when no split is needed.
    """
    if df is None or df.empty:
        return df

    # default: always clean header noise
    df.columns = _strip_header_noise(df.columns)

    # Decide if this "looks" like single-column data
    looks_singlecol = df.shape[1] == 1
    if not looks_singlecol and df.shape[1] > 1:
        # if most of the non-first columns are empty, it's effectively single-col
        others_null_frac = df.iloc[:, 1:].isna().mean().mean()
        looks_singlecol = others_null_frac > 0.95

    first = df.iloc[:, 0].astype(str)

    # detect likely delimiter in first column values
    delim = None
    if first.str.contains(",").mean() > 0.6:
        delim = ","
    elif first.str.contains(";").mean() > 0.6:
        delim = ";"

    if looks_singlecol and delim:
        # remove wrapping quotes around whole cell (common in some CSV exports)
        first_clean = first.str.replace(r'^\s*"\s*|\s*"\s*$', "", regex=True)
        parts = first_clean.str.split(delim, expand=True)

        # Try to derive header:
        # 1) If the *original column name* looks like a "date,customer,..." list, use that
        header_tokens_from_name = [t.strip() for t in str(df.columns[0]).split(",")]
        header_tokens_from_name = _strip_header_noise(header_tokens_from_name)
        if len(header_tokens_from_name) == parts.shape[1] and all(header_tokens_from_name):
            parts.columns = header_tokens_from_name
        else:
            # 2) Otherwise, treat the first row as header if it looks plausible
            if parts.shape[0] > 1:
                guessed = _strip_header_noise(parts.iloc[0].astype(str).tolist())
                if len(guessed) == parts.shape[1]:
                    parts.columns = guessed
                    parts = parts.iloc[1:].reset_index(drop=True)

        return parts

    return df


def load_table(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Robustly load a CSV/XLS/XLSX into a pandas DataFrame.

    Strategy:
      • Excel first (.xlsx/.xls via openpyxl), then normalize (split-first-column if needed)
      • CSV: sniff delimiter from a sample, try common encodings
      • Fallback: brute-force encodings × separators
      • Last resort: loose read + normalize
    """
    name = (filename or "").lower()

    # ---------- Excel path ----------
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl", dtype=object)
        return _maybe_split_first_column(df)

    # ---------- CSV path ----------
    # Use a UTF-8 view of first bytes to let csv.Sniffer try delimiter inference
    sample = file_bytes[:8192].decode("utf-8", errors="ignore")
    try:
        guessed_sep = csv.Sniffer().sniff(sample, delimiters="".join(SEPS)).delimiter
    except csv.Error:
        guessed_sep = None

    # 1) Try the guessed delimiter across common encodings
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(
                io.BytesIO(file_bytes),
                engine="python",
                sep=guessed_sep,
                encoding=enc,
                dtype=object,
            )
            # normalize & return even if it's single column (normalizer may split)
            df = _maybe_split_first_column(df)
            if df.shape[1] >= 1:
                return df
        except Exception:
            pass

    # 2) Brute-force: encodings × separators
    for enc in ENCODINGS:
        for sep in SEPS:
            try:
                df = pd.read_csv(
                    io.BytesIO(file_bytes),
                    engine="python",
                    sep=sep,
                    encoding=enc,
                    dtype=object,
                )
                df = _maybe_split_first_column(df)
                if df.shape[1] >= 1:
                    return df
            except Exception:
                pass

    # 3) Last resort: loose read, then normalize
    df = pd.read_csv(
        io.BytesIO(file_bytes),
        engine="python",
        sep=None,
        encoding="latin1",
        dtype=object,
    )
    return _maybe_split_first_column(df)
