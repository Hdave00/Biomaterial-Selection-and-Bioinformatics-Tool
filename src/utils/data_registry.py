# src/utils/data_registry.py
import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional

# Registry of dataset keys -> relative paths (adjust if your project uses different base)
DATASET_PATHS = {
    "structural": "master_data/unified_material_data.csv",
    "high_entropy": "master_data/HEA/high_entropy_alloys_properties.csv",
    "corrosion": "master_data/corrosion/corr_lookup_database.csv",
    "polymers": "master_data/polymer_tg/polymer_lookup_data.csv",
}

def get_dataset_path(key: str) -> Optional[str]:
    """Return filesystem path for given dataset key, or None if unknown."""
    return DATASET_PATHS.get(key)


def read_csv_if_exists(path: str) -> Optional[pd.DataFrame]:
    """Read CSV if it exists, otherwise return None."""
    if not path or not os.path.exists(path):
        return None
    # try common encodings and separators
    try:
        return pd.read_csv(path)
    except Exception:
        # fallback with low_memory
        try:
            return pd.read_csv(path, low_memory=False)
        except Exception:
            return None


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: strip, replace newlines, reduce spaces, replace spaces with underscores."""
    df = df.copy()
    new_cols = []
    for c in df.columns:
        nc = str(c).strip()
        nc = nc.replace("\n", " ").replace("\r", " ")
        nc = " ".join(nc.split())    # collapse whitespace
        nc = nc.replace(" ", "_")
        new_cols.append(nc)
    df.columns = new_cols
    return df


def merge_duplicate_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If numeric columns share the same base name like 'Tensile_Strength_MPa' and 'Tensile_Strength_MPa.1',
    merge them into a single column by averaging row-wise. Non-numeric duplicates are left alone.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    base_map = {}
    for col in numeric_cols:
        base = col.split(".")[0]
        base_map.setdefault(base, []).append(col)

    for base, cols in base_map.items():
        if len(cols) > 1:
            df[base] = df[cols].mean(axis=1)
            df.drop(columns=cols, inplace=True)
        else:
            if cols[0] != base:
                # rename single numeric column to base (clean ".0" or similar)
                df.rename(columns={cols[0]: base}, inplace=True)
    return df


def prepare_high_entropy_df(df):
    # strip "PROPERTY: "
    df = df.rename(columns=lambda c: c.replace("PROPERTY: ", "").strip())

    # clean spaces and parentheses
    df = df.rename(columns=lambda c: c.replace(" ", "_")
                                      .replace("(GPa)", "")
                                      .replace("(MPa)", "")
                                      .replace("(wppm)", "")
                                      .replace("(μm)", "um")
                                      .replace("μ", "u")
                                      .replace("°C", "C")
                                      .replace("/", "_")
                                      .replace("%", "pct"))

    meta = {
        "nrows": len(df),
        "columns": df.columns.tolist()
    }
    return df, meta


def prepare_structural_df(raw_df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Dict]:
    """
    Clean the unified material dataset for structural lookups.
    Returns (cleaned_df, metadata).
    """
    if raw_df is None:
        return None, {}

    df = raw_df.copy()
    df = normalize_column_names(df)
    df = merge_duplicate_numeric_columns(df)

    # Ensure expected numeric columns exist (if missing, they remain absent)
    # Convert common numeric columns to numeric dtype where possible
    for col in df.columns:
        if df[col].dtype == object:
            # try to coerce numeric-like columns
            coerced = pd.to_numeric(df[col], errors='coerce')
            if coerced.notna().sum() > 0:
                df[col] = coerced

    metadata = {
        "columns": df.columns.tolist(),
        "nrows": len(df)
    }
    return df, metadata


def prepare_corrosion_df(raw_df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Dict]:
    """Clean corrosion CSV schema and coerce common numeric columns."""
    if raw_df is None:
        return None, {}

    df = raw_df.copy()
    df = normalize_column_names(df)

    # Try to coerce rate columns to numeric if possible
    # Candidate column names from your schema:
    rate_candidates = [c for c in df.columns if "Rate" in c or "rate" in c or "rate" in c.lower()]
    for c in rate_candidates:
        # remove non-numeric chars then coerce
        df[c + "_numeric"] = pd.to_numeric(df[c].astype(str).str.replace(r"[^0-9\.\-]+", "", regex=True), errors='coerce')

    # Try to coerce Temperature and Concentration to numeric
    for c in ["Temperature_deg_C", "Concentration_(Vol_%)", "Temperature_deg_F"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    metadata = {"columns": df.columns.tolist(), "nrows": len(df)}
    return df, metadata


def prepare_polymers_df(raw_df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Dict]:
    """Clean polymer lookup file."""
    if raw_df is None:
        return None, {}

    df = raw_df.copy()
    df = normalize_column_names(df)

    # Tg and Density columns often need numeric coercion
    if "Tg" in df.columns:
        df["Tg"] = pd.to_numeric(df["Tg"], errors="coerce")
    if "Density" in df.columns:
        df["Density"] = pd.to_numeric(df["Density"], errors="coerce")

    metadata = {"columns": df.columns.tolist(), "nrows": len(df)}
    return df, metadata


def prepare_high_entropy_df(raw_df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Dict]:
    """Light normalization for HEA dataset."""
    if raw_df is None:
        return None, {}

    df = raw_df.copy()
    df = normalize_column_names(df)
    # trim whitespace in important string columns
    for c in df.select_dtypes(include=[object]).columns:
        df[c] = df[c].astype(str).str.strip()

    # coerce numerics
    for c in df.columns:
        if df[c].dtype == object:
            coerced = pd.to_numeric(df[c], errors='coerce')
            if coerced.notna().sum() > 0:
                df[c] = coerced

    metadata = {"columns": df.columns.tolist(), "nrows": len(df)}
    return df, metadata


def load_and_prepare(key: str) -> Tuple[Optional[pd.DataFrame], Dict]:
    """
    High-level helper to load and prepare dataset by key.
    Returns (df, metadata).
    """
    path = get_dataset_path(key)
    if not path:
        return None, {}

    df = read_csv_if_exists(path)
    if df is None:
        return None, {}

    if key == "structural":
        return prepare_structural_df(df)
    if key == "corrosion":
        return prepare_corrosion_df(df)
    if key == "polymers":
        return prepare_polymers_df(df)
    if key == "high_entropy":
        return prepare_high_entropy_df(df)

    # default fallback: normalized df
    df = normalize_column_names(df)
    return df, {"columns": df.columns.tolist(), "nrows": len(df)}