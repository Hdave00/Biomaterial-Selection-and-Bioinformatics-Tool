# src/inference/model_interface.py
import os
import json
import numpy as np
import pickle
import joblib
import tensorflow as tf

MODEL_REGISTRY = os.path.join(os.path.dirname(__file__), '..', '..', 'models')

def _load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def _load_npy(path):
    return np.load(path, allow_pickle=True)

def _load_keras(path):
    return tf.keras.models.load_model(path)

# Example loader for crystalline model (classification)
def load_crystalline_model(version='v1'):
    base = os.path.join(MODEL_REGISTRY, 'crystalline_structure', version)
    model = _load_pickle(os.path.join(base, 'model.pkl'))
    scaler = _load_npy(os.path.join(base, 'scaler.npy')) if os.path.exists(os.path.join(base,'scaler.npy')) else None
    label_encoder = _load_pickle(os.path.join(base, 'label_encoder.pkl')) if os.path.exists(os.path.join(base,'label_encoder.pkl')) else None
    metadata = json.load(open(os.path.join(base, 'metadata.json')))
    return {'model': model, 'scaler': scaler, 'label_encoder': label_encoder, 'meta': metadata}

# Example loader for polymer Tg predictor (keras)
def load_polymer_tg_model(version='v1'):
    base = os.path.join(MODEL_REGISTRY, 'polymer_tg_density', version)
    model = _load_keras(os.path.join(base, 'polymer_tg_density_predictor.keras'))
    scaler_mean = _load_npy(os.path.join(base, 'polymer_tg_density_scaler_mean.npy'))
    scaler_scale = _load_npy(os.path.join(base, 'polymer_tg_density_scaler_scale.npy'))
    metadata = json.load(open(os.path.join(base, 'metadata.json')))
    return {'model': model, 'scaler_mean': scaler_mean, 'scaler_scale': scaler_scale, 'meta': metadata}

# Prediction wrappers
def predict_crystalline(model_bundle, X_raw):
    """X_raw: 1D array or dict depending on metadata (ensure same feature order)"""
    model = model_bundle['model']
    scaler = model_bundle['scaler']
    le = model_bundle['label_encoder']
    # prepare numpy vector X
    X = np.asarray(X_raw).reshape(1, -1)
    if scaler is not None:
        X = (X - scaler[0]) / scaler[1]  # if you saved mean/scale as numpy arrays
    probs = model.predict_proba(X) if hasattr(model, 'predict_proba') else model.predict(X)
    pred_idx = np.argmax(probs, axis=1)[0]
    pred_label = le.inverse_transform([pred_idx])[0] if le is not None else int(pred_idx)
    return {'label': pred_label, 'proba': probs[0].tolist()}

def predict_polymer_tg(model_bundle, X_raw):
    model = model_bundle['model']
    mean = model_bundle['scaler_mean']
    scale = model_bundle['scaler_scale']
    X = np.asarray(X_raw).reshape(1, -1)
    Xs = (X - mean) / scale
    y_pred = model.predict(Xs)
    return float(y_pred.squeeze())