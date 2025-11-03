#!/usr/bin/env python3
"""
corrosion_ml_pipeline.py

Machine learning pipeline for predicting corrosion rate (mm/yr) based on material and environmental features.
Trains a regression model and exports both processed data and serialized model artifacts.

Inputs:
    - data/metal_corrosion/CORR-DATA_Database.csv

Outputs:
    - master_data/corrosion/corrosion_ml_ready.csv
    - models/corrosion_rate_predictor.pkl
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re
import logging
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)


# Logging setup
log = logging.getLogger("corrosion_ml_pipeline")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# Paths
BASE = Path(__file__).resolve().parents[2]
DATA_FILE = BASE / "data" / "metal_corrosion" / "CORR-DATA_Database.csv"
MASTER_DIR = BASE / "master_data" / "corrosion"
MODEL_DIR = BASE / "models"

MASTER_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)



# Utility Functions
def safe_str(x):
    return "" if pd.isna(x) else str(x)


def parse_rate_mm(value: str) -> float:
    """Extract numeric corrosion rate (mm/yr), converting from mils if needed."""
    if pd.isna(value):
        return np.nan
    s = safe_str(value).strip()
    if not s:
        return np.nan

    num_match = re.findall(r"[-+]?\d*\.?\d+", s)
    if not num_match:
        return np.nan
    try:
        num = float(num_match[0])
    except Exception:
        return np.nan

    # Convert mils/yr to mm/yr if necessary
    if re.search(r"mil", s, flags=re.I) or re.search(r"in", s, flags=re.I):
        num *= 0.0254  # 1 mil = 0.0254 mm
    return num



# Data loading and cleaning
def load_and_clean():
    log.info("Loading corrosion dataset...")
    try:
        df = pd.read_csv(DATA_FILE, dtype=str)
    except Exception:
        df = pd.read_csv(DATA_FILE, dtype=str, encoding="latin-1")

    df.columns = [c.strip() for c in df.columns]

    # Compute numeric corrosion rate
    df["rate_mm"] = df["Rate (mm/yr) or Rating"].apply(parse_rate_mm)
    mil_rates = df["Rate (mils/yr) or Rating"].apply(parse_rate_mm)
    df["rate_mm"] = df["rate_mm"].fillna(mil_rates)

    # Drop rows without numeric target
    df = df.dropna(subset=["rate_mm"])

    # Clean up categorical text
    for col in ["Environment", "Material Group", "Material Family", "Material", "Condition/Comment"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    # Keep relevant columns
    keep_cols = [
        "Environment", "Material Group", "Material Family", "Material",
        "Temperature (deg C)", "Concentration (Vol %)", "rate_mm"
    ]
    df = df[keep_cols].copy()

    # Normalize names
    df["Clean_Name"] = df["Material"].fillna("").apply(
        lambda s: re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9\-\+\(\)\/ ]+", " ", str(s).upper())).strip()
    )

    # Convert numeric columns
    df["Temperature (deg C)"] = pd.to_numeric(df["Temperature (deg C)"], errors="coerce")
    df["Concentration (Vol %)"] = pd.to_numeric(df["Concentration (Vol %)"], errors="coerce")

    df = df.dropna(subset=["Clean_Name"])
    return df



# Model building and training
def build_and_train(df: pd.DataFrame):
    log.info("Building and training ML model for corrosion rate prediction...")

    X = df[["Environment", "Material Group", "Material Family", "Clean_Name", "Temperature (deg C)", "Concentration (Vol %)"]]
    y = df["rate_mm"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Categorical vs numerical features
    cat_cols = ["Environment", "Material Group", "Material Family", "Clean_Name"]
    num_cols = ["Temperature (deg C)", "Concentration (Vol %)"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    # Evaluate
    preds = pipe.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    log.info("Model evaluation metrics:")
    log.info(f"  - Mean Absolute Error (MAE): {mae:.4f}")
    log.info(f"  - Mean Squared Error (MSE): {mse:.4f}")
    log.info(f"  - Root Mean Squared Error (RMSE): {rmse:.4f}")
    log.info(f"  - R² Score: {r2:.3f}")

    # Save model
    model_path = MODEL_DIR / "corrosion_rate_predictor.pkl"
    joblib.dump(pipe, model_path)
    log.info(f"Saved trained model -> {model_path}")

    return pipe, mse, rmse, mae, r2



# Entry point
def run_corrosion_ml_pipeline():
    df = load_and_clean()

    out_csv = MASTER_DIR / "corrosion_ml_ready.csv"
    df.to_csv(out_csv, index=False)
    log.info(f"Wrote ML-ready corrosion dataset -> {out_csv} ({len(df)} rows)")

    model, mse, rmse, mae, r2 = build_and_train(df)
    return model, mse, rmse, mae, r2


if __name__ == "__main__":
    run_corrosion_ml_pipeline()