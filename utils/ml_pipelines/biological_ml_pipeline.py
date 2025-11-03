# here will be the code for bio ml pipeline

import os
import glob
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import logging


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# Helper: Convert SMILES to fingerprints
def smiles_to_fingerprint(smiles, n_bits=2048):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
        return np.array(fp)
    except Exception:
        return np.zeros(n_bits)


# Helper: Clean numeric extraction
def extract_float(value):
    try:
        if isinstance(value, str):
            value = value.replace(",", ".").split()[0]
        return float(value)
    except Exception:
        return np.nan


# Main Biological ML Pipeline
def run_biological_ml_pipeline(
        input_dir="data/cytotoxity_ionic_liquids/csv_datasets",
        cell_lines_path="data/cytotoxity_ionic_liquids/cell_lines.csv",
        methods_path="data/cytotoxity_ionic_liquids/methods.csv",
        output_path="master_data/biological/biological_ml_ready.csv"
    ):
    logging.info("Starting biological ML pipeline...")

    input_dir = Path(input_dir)
    all_files = list(input_dir.glob("*.csv"))
    if not all_files:
        logging.error(f"No cytotoxicity CSV files found in {input_dir}")
        return None

    dfs = []
    for file in all_files:
        if file.name in ["cell_lines.csv", "methods.csv"]:
            continue
        df = pd.read_csv(file)
        df["Family"] = file.stem  # tag family name (e.g. 'phosphonium')
        dfs.append(df)
    
    data = pd.concat(dfs, ignore_index=True)
    logging.info(f"Loaded {len(data)} cytotoxicity rows across {len(dfs)} families")


    # Basic cleaning
    data["Mw"] = data["Mw, g*mol-1"].apply(extract_float)
    data["CC50_mM"] = data["CC50/IC50/EC50, mM"].apply(extract_float)
    data["Incubation_h"] = data["Incubation time, h"].apply(extract_float)

    # Drop rows missing essential values
    data = data.dropna(subset=["Canonical SMILES", "CC50_mM"])
    data = data.reset_index(drop=True)


    # Merge cell line and method metadata
    try:
        cell_lines = pd.read_csv(cell_lines_path)
        methods = pd.read_csv(methods_path)
        data = data.merge(cell_lines, left_on="Cell line", right_on="Cell name", how="left")
        data = data.merge(methods, left_on="Method", right_on="Method abbreviation", how="left")
    except Exception as e:
        logging.warning(f"Could not merge metadata: {e}")


    # Generate molecular features
    logging.info("Generating molecular fingerprints...")
    fps = np.vstack(data["Canonical SMILES"].apply(smiles_to_fingerprint))
    fps_df = pd.DataFrame(fps, columns=[f"fp_{i}" for i in range(fps.shape[1])])
    data = pd.concat([data, fps_df], axis=1)

    # Add basic descriptors if possible
    molwts, logps, tpsas = [], [], []
    for smi in data["Canonical SMILES"]:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            molwts.append(Descriptors.MolWt(mol))
            logps.append(Descriptors.MolLogP(mol))
            tpsas.append(Descriptors.TPSA(mol))
        else:
            molwts.append(np.nan)
            logps.append(np.nan)
            tpsas.append(np.nan)

    data["MolWt_calc"] = molwts
    data["LogP_calc"] = logps
    data["TPSA_calc"] = tpsas


    # Encode categorical variables
    for col in ["Family", "Cell type", "Organism", "Full name of method"]:
        if col in data.columns:
            encoder = LabelEncoder()
            data[col] = encoder.fit_transform(data[col].astype(str))


    # Create label
    data["biocompatibility_label"] = data["CC50_mM"].apply(lambda x: 1 if x > 1 else 0)


    # Save ML-ready dataset
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    logging.info(f"Biological ML dataset saved -> {output_path} ({len(data)} rows)")

    return data


if __name__ == "__main__":
    run_biological_ml_pipeline()