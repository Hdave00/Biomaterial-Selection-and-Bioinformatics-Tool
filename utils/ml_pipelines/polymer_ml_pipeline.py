#!/usr/bin/env python3
"""
polymer_tg_density_predictor.py

Predict polymer glass transition temperature (Tg) and density
from SMILES structures using molecular fingerprints and
a multi-output TensorFlow regression model.

Inputs:
    - master_data/unified_polymer_data.csv

Outputs:
    - models/polymer_tg_density_predictor.keras
    - models/polymer_tg_density_scaler_mean.npy
    - models/polymer_tg_density_scaler_scale.npy
    - master_data/polymer_tg/polymer_tg_dataset.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#  Use shared SMILES -> fingerprint function
from shared_utils import load_smiles_to_fingerprints


# Paths
BASE = Path(__file__).resolve().parents[2]
DATA_DIR = BASE / "data" / "polymer_tg_density"
MASTER_DIR = BASE / "master_data" / "polymer_tg"
MODEL_DIR = BASE / "models"

MASTER_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Load and merge unified dataset
def load_tg_density_data():
    df = pd.read_csv(BASE / "master_data" / "unified_polymer_data.csv", dtype=str)

    # Convert numeric columns
    df["Tg"] = pd.to_numeric(df["Tg"], errors="coerce")
    df["Density"] = pd.to_numeric(df["Density"], errors="coerce")

    # Remove polymers missing SMILES or Tg
    df = df.dropna(subset=["SMILES", "Tg"])

    # Clean SMILES and fill missing density
    df["SMILES"] = df["SMILES"].str.strip()
    df["Density"] = df["Density"].fillna(df["Density"].median())

    return df[["SMILES", "Tg", "Density"]]


# Preprocessing and scaling
def preprocess(df, n_bits=2048):
    print(f"[INFO] Generating fingerprints for {len(df)} polymers...")

    # Use shared utility for robust fingerprint generation
    X = load_smiles_to_fingerprints(df["SMILES"], n_bits=n_bits)

    y = df[["Tg", "Density"]].copy()
    y["Density"] = y["Density"].fillna(y["Density"].median())

    # Scale each target (Tg, Density)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)

    print(f"[INFO] Feature matrix shape: {X.shape}")
    print(f"[INFO] Target matrix shape: {y_scaled.shape}")

    return X, y_scaled, scaler_y


# Model definition (multi-output)
def build_tg_density_model(input_dim):
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(512, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)

    # Two outputs: [Tg, Density]
    outputs = tf.keras.layers.Dense(2)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="mse",
        metrics=["mae"]
    )
    return model


# Training pipeline
def train_model():
    df = load_tg_density_data()
    X, y_scaled, scaler_y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_scaled, test_size=0.2, random_state=42
    )

    model = build_tg_density_model(X.shape[1])

    print("[INFO] Starting model training...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=120,
        batch_size=32,
        verbose=1
    )

    loss, mae = model.evaluate(X_test, y_test)
    print(f"\n Test loss: {loss:.4f}, MAE (scaled): {mae:.4f}")

    # Save model + scaler
    model.save(MODEL_DIR / "polymer_tg_density_predictor.keras")
    np.save(MODEL_DIR / "polymer_tg_density_scaler_mean.npy", scaler_y.mean_)
    np.save(MODEL_DIR / "polymer_tg_density_scaler_scale.npy", scaler_y.scale_)

    df.to_csv(MASTER_DIR / "polymer_tg_dataset.csv", index=False)
    print(f"Saved processed dataset -> {MASTER_DIR}/polymer_tg_dataset.csv")

    return model, scaler_y


# Inference
def predict_tg_density(smiles):
    model = tf.keras.models.load_model(MODEL_DIR / "polymer_tg_density_predictor.keras")
    mean = np.load(MODEL_DIR / "polymer_tg_density_scaler_mean.npy")
    scale = np.load(MODEL_DIR / "polymer_tg_density_scaler_scale.npy")

    fp = load_smiles_to_fingerprints([smiles]).reshape(1, -1)
    y_pred_scaled = model.predict(fp, verbose=0)[0]
    y_pred = (y_pred_scaled * scale) + mean

    return {"Tg_pred": y_pred[0], "Density_pred": y_pred[1]}


# Run as script
if __name__ == "__main__":
    model, scaler = train_model()

    # Example prediction
    example_smiles = "*C(C)(C(=O)OC)C*"  # PMMA-like
    preds = predict_tg_density(example_smiles)
    print(f"\nExample prediction for {example_smiles}:")
    print(f"Tg ≈ {preds['Tg_pred']:.2f} °C, Density ≈ {preds['Density_pred']:.3f} g/cm³")