# src/utils/filter_engine.py
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional

def free_text_mask(df: pd.DataFrame, text: str) -> pd.Series:
    """Return boolean mask where any column contains the free-text (case-insensitive)."""
    if not text:
        return pd.Series([True] * len(df), index=df.index)
    text = str(text).strip()
    masks = []
    for c in df.columns:
        # only attempt on string-like columns
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            masks.append(df[c].astype(str).str.contains(text, case=False, na=False))
    if not masks:
        return pd.Series([False] * len(df), index=df.index)
    return pd.concat(masks, axis=1).any(axis=1)


def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply a set of filters to df.

    filters may contain:
      - "free_text": string
      - for numeric columns: {"col_name": {"min": x, "max": y}}
      - for categorical/exact match: {"col_name": {"values": [v1, v2]}}

    Returns (filtered_df, missing_columns)
    """
    if df is None:
        return pd.DataFrame(), []

    mask = pd.Series([True] * len(df), index=df.index)
    missing_cols = []

    # free_text first
    free_text = filters.get("free_text")
    if free_text:
        ft_mask = free_text_mask(df, free_text)
        mask &= ft_mask

    # iterate other filters
    for col, spec in filters.items():
        if col == "free_text":
            continue
        if col not in df.columns:
            missing_cols.append(col)
            continue

        col_ser = df[col]

        # numeric range filter
        if isinstance(spec, dict) and ("min" in spec or "max" in spec):
            try:
                # coerce to numeric for comparison
                ser_num = pd.to_numeric(col_ser, errors='coerce')
                if spec.get("min") is not None:
                    mask &= (ser_num >= spec["min"]).fillna(False)
                if spec.get("max") is not None:
                    mask &= (ser_num <= spec["max"]).fillna(False)
            except Exception:
                missing_cols.append(col)
                continue

        # categorical / exact values
        elif isinstance(spec, dict) and "values" in spec:
            vals = spec.get("values", [])
            if not vals:
                continue
            mask &= col_ser.astype(str).isin([str(v) for v in vals])

        # exact single value
        else:
            mask &= col_ser.astype(str).str.contains(str(spec), case=False, na=False)

    filtered = df[mask]
    return filtered, missing_cols