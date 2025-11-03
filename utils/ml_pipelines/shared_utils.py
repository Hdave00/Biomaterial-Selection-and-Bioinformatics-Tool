"""
And also utilities for:

    -Normalizing units

    -Label encoding categorical features

    -Handling missing values

    -Scaling numeric features
"""

def load_smiles_to_fingerprints(smiles_series, n_bits=2048):
    
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import numpy as np

    fps = []
    for smi in smiles_series:
        try:
            mol = Chem.MolFromSmiles(smi)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
            fps.append(np.array(fp))
        except Exception:
            fps.append(np.zeros(n_bits))
    return np.array(fps)