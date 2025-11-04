"""
crystalline_structure_predictor.py

Predicts the crystal structure (lowest distortion type) of materials
based on ionic radii, valence, electronegativity, and geometric features.

Target: 'Lowest distortion'
Type: Multi-class classification
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import logging
import os


# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# Paths
DATA_DIR = "data/crystalline_struct_dataset/"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
MODEL_PATH = "models/crystal_structure_model.pkl"
SCALER_PATH = "models/crystal_scaler.pkl"
ENCODER_PATH = "models/crystal_label_encoder.pkl"
PREDICTION_OUTPUT = "predicted_structures.csv"


# Load and preprocess
def load_data(train_path=TRAIN_PATH, test_path=TEST_PATH):
    logging.info("Loading training and test data...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    if "Lowest distortion" not in df_train.columns:
        raise ValueError("Target column 'Lowest distortion' missing from training data.")

    X_train = df_train.drop(columns=["Lowest distortion"])
    y_train = df_train["Lowest distortion"]
    X_test = df_test.copy()

    logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, y_train, X_test


def preprocess_data(X_train, y_train, X_test):
    logging.info("Encoding and scaling data...")

    # Label encode the target
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_train)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_scaled, y_encoded, X_test_scaled, scaler, encoder


# Model training
def train_model(X_train, y_train):

    logging.info("Training RandomForestClassifier model...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    return model


# Evaluation
def evaluate_model(model, X_train, y_train, encoder):

    logging.info("Evaluating model...")
    preds = model.predict(X_train)
    acc = accuracy_score(y_train, preds)
    logging.info(f"Training accuracy: {acc:.4f}")

    logging.info("Classification report:")
    logging.info("\n" + classification_report(y_train, preds, target_names=encoder.classes_))
    logging.info("Confusion matrix:")
    logging.info("\n" + str(confusion_matrix(y_train, preds)))


# Prediction
def predict(model, X_test, encoder):
    preds = model.predict(X_test)
    return encoder.inverse_transform(preds)


# Main pipeline
def run_pipeline():
    X_train, y_train, X_test = load_data()
    X_scaled, y_encoded, X_test_scaled, scaler, encoder = preprocess_data(X_train, y_train, X_test)
    model = train_model(X_scaled, y_encoded)

    evaluate_model(model, X_scaled, y_encoded, encoder)
    preds = predict(model, X_test_scaled, encoder)

    output_df = X_test.copy()
    output_df["Predicted Structure"] = preds
    output_df.to_csv(PREDICTION_OUTPUT, index=False)
    logging.info(f"Predictions saved to {PREDICTION_OUTPUT}")

    # Save model components
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    logging.info("Model, scaler, and encoder saved successfully.")


if __name__ == "__main__":
    run_pipeline()
