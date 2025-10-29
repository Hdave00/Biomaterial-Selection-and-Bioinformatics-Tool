import pandas as pd
from pathlib import Path

# Directory setup
DATA_DIR = Path("data/polymer_tg_density")
OUTPUT_FILE = Path("master_data/unified_polymer_data.csv")

# Helper functions
def normalize_name(series):
    return series.astype(str).str.strip().str.upper()


def load_data():
    """Load raw datasets with proper column name cleanup."""
    df_tg = pd.read_csv(DATA_DIR / "tg_density.csv")
    df_main = pd.read_csv(DATA_DIR / "TgSS_enriched_cleaned.csv")

    # Normalize column names
    df_tg.columns = df_tg.columns.str.strip().str.replace(" ", "_")
    df_main.columns = df_main.columns.str.strip().str.replace(" ", "_")

    return df_tg, df_main


def merge_polymer_data(df_tg, df_main):
    """Merge tg_density first, then append TgSS-only polymers to avoid row multiplication."""

    # Rename Tg columns
    df_tg = df_tg.rename(columns={"Tg": "Tg_density"})
    df_main = df_main.rename(columns={"Tg": "Tg_TgSS"})

    # Normalized names
    df_tg["NAME_NORM"] = normalize_name(df_tg["name"])
    df_tg["ABR_NORM"] = normalize_name(df_tg["abr"])
    df_main["NAME_NORM"] = normalize_name(df_main["Polymer_Class"])

    # Use tg_density as master; only take TgSS rows that don't exist in tg_density
    df_main_only = df_main[~df_main["NAME_NORM"].isin(df_tg["NAME_NORM"])].copy()
    df_main_only["Tg"] = df_main_only["Tg_TgSS"]
    df_main_only = df_main_only.drop(columns=["Tg_TgSS"], errors="ignore")

    # Tg_density already has Tg column renamed to Tg_density; unify
    df_tg["Tg"] = df_tg["Tg_density"]
    df_tg = df_tg.drop(columns=["Tg_density"], errors="ignore")

    # Combine datasets: tg_density first, then TgSS-only
    df_final = pd.concat([df_tg, df_main_only], ignore_index=True)

    return df_final


def clean_for_ml(df):
    """Final cleaning & preprocessing for ML compatibility"""

    # Fill missing numerics
    for col in ["Tg", "Density"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1)

    # Fill missing text categories
    for col in ["grade", "manufacturer", "abr", "name", "Polymer_Class"]:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN")

    return df


def save(df):
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nUnified dataset saved: {OUTPUT_FILE}")
    print(f"Total polymers: {len(df)}\n")
    print(df.head())


if __name__ == "__main__":
    df_tg, df_main = load_data()
    merged = merge_polymer_data(df_tg, df_main)
    merged = clean_for_ml(merged)
    save(merged)