# utils/feature_extractor.py
import pandas as pd

class FeatureExtractor:
    def __init__(self, mechanical_df, chemical_df, biological_df, master_index):
        self.mechanical = mechanical_df
        self.chemical = chemical_df
        self.biological = biological_df
        self.master_index = master_index

    def get_material_features(self, material_name):
        name_norm = material_name.strip().upper()
        if name_norm not in self.master_index['material_name'].values:
            return None

        # Mechanical
        mech = self.mechanical[self.mechanical['Material'].str.upper() == name_norm]
        mech_features = mech.drop(columns=['Material'], errors='ignore').mean().fillna(-1).to_dict()

        # Chemical
        chem = self.chemical[self.chemical['Polymer'].str.upper() == name_norm]
        chem_features = chem.drop(columns=['Polymer'], errors='ignore').mean().fillna(-1).to_dict()

        # Biological
        bio = self.biological[self.biological['Material'].str.upper() == name_norm]
        bio_features = bio.drop(columns=['Material'], errors='ignore').mean().fillna(-1).to_dict()

        return {
            'mechanical': mech_features,
            'chemical': chem_features,
            'biological': bio_features
        }

    def get_training_matrix(self):
        """Return X, y ready for neural network training"""
        X_mech, X_chem, X_bio, y = [], [], [], []

        for _, row in self.master_index.iterrows():
            feats = self.get_material_features(row['material_name'])
            if feats is None:
                continue

            # Skip if insufficient data
            if not feats['mechanical'] or not feats['chemical'] or not feats['biological']:
                continue

            X_mech.append(list(feats['mechanical'].values()))
            X_chem.append(list(feats['chemical'].values()))
            X_bio.append(list(feats['biological'].values()))

            # Example: biocompatibility score from cytotoxicity
            bio_score = feats['biological'].get('cytotoxicity_score', 0.5)
            y.append(bio_score)

        return X_mech, X_chem, X_bio, y