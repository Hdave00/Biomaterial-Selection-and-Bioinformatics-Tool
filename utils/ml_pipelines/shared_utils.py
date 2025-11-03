"""
And also utilities for:

    -Normalizing units

    -Label encoding categorical features

    -Handling missing values

    -Scaling numeric features
"""

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import logging

def load_smiles_to_fingerprints(smiles_series, n_bits=2048):

    """
    Converts a pandas Series of SMILES strings into RDKit Morgan fingerprints.
    Handles malformed polymer SMILES gracefully (e.g. '*', '[Si]()', etc.)
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
            if mol is None:
                return None
            return mol
        except Exception:
            return None

    fps = []
    for smi in smiles_series:
        mol = safe_mol_from_smiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
            fps.append(np.array(fp))
        else:
            # fallback: zero-vector for failed SMILES
            fps.append(np.zeros(n_bits))
            bad_smiles.append(smi)

    if bad_smiles:
        log.warning(f"Skipped {len(bad_smiles)} invalid SMILES strings during fingerprinting")

    return np.array(fps)