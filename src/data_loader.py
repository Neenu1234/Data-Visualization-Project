import io
import os
import zipfile
from typing import Dict, List, Optional, Tuple

import duckdb
import pandas as pd


def _infer_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column_name in df.columns:
        if df[column_name].dtype == "object":
            sample = df[column_name].dropna().astype(str).head(50)
            # Fast path: ISO-like dates or numbers that look like yyyymmdd
            if sample.str.contains(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}", regex=True).mean() > 0.6:
                df[column_name] = pd.to_datetime(df[column_name], errors="ignore", infer_datetime_format=True)
            elif sample.str.contains(r"^\d{8}$", regex=True).mean() > 0.6:
                df[column_name] = pd.to_datetime(df[column_name], errors="ignore", format="%Y%m%d")
    return df


def _read_csv_smart(path_or_buffer, filename_hint: Optional[str] = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path_or_buffer, low_memory=False)
    except Exception:
        # Try semicolon
        try:
            return pd.read_csv(path_or_buffer, sep=";", low_memory=False)
        except Exception:
            # Try tab
            return pd.read_csv(path_or_buffer, sep="\t", low_memory=False)


def load_from_zip(zip_path: str) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")
    if not zip_path.lower().endswith(".zip"):
        raise ValueError("Expected a .zip file")

    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_members: List[str] = [m for m in zf.namelist() if m.lower().endswith((".csv", ".tsv"))]
        excel_members: List[str] = [m for m in zf.namelist() if m.lower().endswith((".xlsx", ".xls"))]

        if not csv_members and not excel_members:
            raise ValueError("ZIP archive does not contain CSV/TSV/XLSX files.")

        dfs: Dict[str, pd.DataFrame] = {}

        # Load CSV/TSV
        for member in csv_members:
            with zf.open(member) as f:
                bytes_buf = io.BytesIO(f.read())
                df = _read_csv_smart(bytes_buf, filename_hint=member)
                df = _infer_datetime_columns(df)
                dfs[member] = df

        # Load Excel files (first sheet only)
        for member in excel_members:
            with zf.open(member) as f:
                bytes_buf = io.BytesIO(f.read())
                df = pd.read_excel(bytes_buf)
                df = _infer_datetime_columns(df)
                dfs[member] = df

        # Choose primary df: prefer names with 'sale'/'order'/'transaction'
        priority = ["sale", "order", "transaction", "online", "retail"]
        selected_name = None
        lower_name_to_key = {k.lower(): k for k in dfs.keys()}
        for p in priority:
            for k in dfs.keys():
                if p in k.lower():
                    selected_name = k
                    break
            if selected_name:
                break
        if selected_name is None:
            # Fallback to largest row count
            selected_name = max(dfs.keys(), key=lambda k: len(dfs[k]))

        return dfs[selected_name], dfs


def load_from_path(path: str) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")
    if os.path.isdir(path):
        raise ValueError("Expected a file path, got a directory.")

    if path.lower().endswith(".zip"):
        return load_from_zip(path)
    if path.lower().endswith((".csv", ".tsv")):
        df = _read_csv_smart(path, filename_hint=path)
        df = _infer_datetime_columns(df)
        return df, {os.path.basename(path): df}
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
        df = _infer_datetime_columns(df)
        return df, {os.path.basename(path): df}

    # Try duckdb to read various formats by inference
    try:
        con = duckdb.connect()
        df = con.sql(f"SELECT * FROM read_auto('{path}')").to_df()
        df = _infer_datetime_columns(df)
        return df, {os.path.basename(path): df}
    except Exception as exc:
        raise ValueError(f"Unsupported file type for: {path}. Error: {exc}") from exc


def memory_usage_mb(df: pd.DataFrame) -> float:
    return round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
