"""
And also utilities for:

    -Normalizing units

    -Label encoding categorical features

    -Handling missing values

    -Scaling numeric features
"""

from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
import os
import numpy as np
import logging
import pandas as pd
import joblib
import numpy as np
import json

def load_smiles_to_fingerprints(smiles_series, n_bits=2048):
    """
    Converts a pandas Series of SMILES strings into RDKit Morgan fingerprints.
    Handles malformed polymer SMILES gracefully.
    """
    log = logging.getLogger(__name__)
    bad_smiles = []

    def safe_mol_from_smiles(smi: str):
        """Sanitize polymer-like SMILES before parsing."""
        if not isinstance(smi, str) or not smi.strip():
            return None
        s = smi.strip().replace("*", "").replace("()", "")
        try:
            mol = Chem.MolFromSmiles(s)
            return mol if mol is not None else None
        except Exception:
            return None

    fps = []
    for smi in smiles_series:
        mol = safe_mol_from_smiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
            fps.append(np.array(fp))
        else:
            fps.append(np.zeros(n_bits))
            bad_smiles.append(smi)

    if bad_smiles:
        log.warning(f"Skipped {len(bad_smiles)} invalid SMILES strings during fingerprinting")

    return np.array(fps)


# For 



MODEL_PATH = "models/crystal_structure_model.pkl"
SCALER_PATH = "models/crystal_scaler.pkl"
ENCODER_PATH = "models/crystal_label_encoder.pkl"

_model = None
_scaler = None
_encoder = None

def _load_predict_objects():
    global _model, _scaler, _encoder
    if _model is None or _scaler is None or _encoder is None:
        if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(ENCODER_PATH)):
            raise FileNotFoundError("Model, scaler, or encoder files are missing in 'models/'.")
        _model = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        _encoder = joblib.load(ENCODER_PATH)
    return _model, _scaler, _encoder


def predict_single(input_dict):
    """
    Predicts crystalline structure from numeric descriptors.

    input_dict keys expected:
        v(A), v(B), r(AXII)(Å), r(AVI)(Å), r(BVI)(Å),
        EN(A), EN(B), l(A-O)(Å), l(B-O)(Å),
        ΔENR, tG, τ, μ
    """
    model, scaler, encoder = _load_predict_objects()

    cols = ["v(A)","v(B)","r(AXII)(Å)","r(AVI)(Å)","r(BVI)(Å)",
            "EN(A)","EN(B)","l(A-O)(Å)","l(B-O)(Å)",
            "ΔENR","tG","τ","μ"]

    row = [input_dict.get(c, np.nan) for c in cols]
    df_row = pd.DataFrame([row], columns=cols)
    df_row = df_row.apply(pd.to_numeric, errors="coerce")

    # later for streamlit web interfacce: fill missing with medians if stored
    # with open("models/crystal_feature_medians.json") as f:
    #     TRAIN_MEDIANS = json.load(f)
    # for c in cols:
    #     df_row[c].fillna(TRAIN_MEDIANS.get(c, 0), inplace=True)

    X_scaled = scaler.transform(df_row)
    pred_encoded = model.predict(X_scaled)
    try:
        pred_label = encoder.inverse_transform(pred_encoded.astype(int))
    except Exception:
        pred_label = pred_encoded
    return pred_label[0]


# but we also want it to be able to make the predictor usable for CSV uploads or API batch calls.
def predict_batch(input_dicts):
    model, scaler, encoder = _load_predict_objects()
    cols = ["v(A)","v(B)","r(AXII)(Å)","r(AVI)(Å)","r(BVI)(Å)",
            "EN(A)","EN(B)","l(A-O)(Å)","l(B-O)(Å)",
            "ΔENR","tG","τ","μ"]
    df = pd.DataFrame(input_dicts)[cols]
    df = df.apply(pd.to_numeric, errors="coerce")
    X_scaled = scaler.transform(df)
    preds = model.predict(X_scaled)
    try:
        return encoder.inverse_transform(preds.astype(int))
    except Exception:
        return preds