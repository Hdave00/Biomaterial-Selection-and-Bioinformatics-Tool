import pandas as pd
import numpy as np
import tensorflow as tf
from keras import Input, Dense, Concatenate, Dropout, BatchNormalization
from keras.models import Model
import pandas as pd

from sklearn.model_selection import train_test_split, TunedThresholdClassifierCV, FixedThresholdClassifier


# TODO
def predict_materials(user_input):
    
    """ Takes user inpit parameters and predicts materials to be shown. """

    # Load actual model and preprocess here in future, for now return dummy results
    # Dummy materials database
    data = {
        "Material": ["Ti-6Al-4V", "CoCrMo", "UHMWPE"],
        "Young’s Modulus (GPa)": [110, 230, 0.9],
        "Tensile Strength (MPa)": [950, 1000, 45],
        "Density (g/cm³)": [4.43, 8.29, 0.93],
        "Biocompatibility Score": [0.93, 0.88, 0.95],
        "Estimated Lifespan (yrs)": [20, 15, 10],
        "Manufacturing Method": ["EBM", "Casting", "Compression Molding"]
    }

    # initialise and load the data in a pandas dataframe to work with CSVs
    df = pd.DataFrame(data)

    # filtering example (expand this later with ML/scoring)
    if user_input["target"] == "hip implant":
        df = df[df["Young’s Modulus (GPa)"] < 150]

    return df



class MultiSourceBiocompatibilityModel:
    def __init__(self):
        self.mechanical_processor = None
        self.chemical_processor = None
        self.biological_processor = None
        
    def create_model(self):
        # Separate inputs for each data type
        mechanical_input = Input(shape=(15,), name='mechanical_props')  # Young's modulus, tensile, etc.
        chemical_input = Input(shape=(10,), name='chemical_props')      # Corrosion, pH stability, etc.
        biological_input = Input(shape=(8,), name='biological_props')   # Cytotoxicity, protein interactions
        
        # Process each domain separately
        mechanical_stream = self._create_mechanical_branch(mechanical_input)
        chemical_stream = self._create_chemical_branch(chemical_input) 
        biological_stream = self._create_biological_branch(biological_input)
        
        # Merge all streams
        merged = Concatenate()([mechanical_stream, chemical_stream, biological_stream])
        
        # Final layers for biocompatibility score
        x = Dense(128, activation='relu')(merged)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        
        # Output: Biocompatibility score (0-1)
        biocompatibility_score = Dense(1, activation='sigmoid', name='biocompatibility_score')(x)
        
        model = Model(
            inputs=[mechanical_input, chemical_input, biological_input],
            outputs=biocompatibility_score
        )
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def _create_mechanical_branch(self, input_layer):
        # Process mechanical properties (from datasets 1,2,4,7)
        x = Dense(64, activation='relu')(input_layer)
        x = BatchNormalization()(x)
        x = Dense(32, activation='relu')(x)
        return x
    
    def _create_chemical_branch(self, input_layer):
        # Process chemical properties (from datasets 5,6,8)
        x = Dense(32, activation='relu')(input_layer)
        x = BatchNormalization()(x)
        x = Dense(16, activation='relu')(x)
        return x
    
    def _create_biological_branch(self, input_layer):
        # Process biological properties (from dataset 3)
        x = Dense(32, activation='relu')(input_layer)
        x = BatchNormalization()(x)
        x = Dense(16, activation='relu')(x)
        return x
    

# Data processing strategy 

class DataManager:
    def __init__(self):
        self.datasets = {}
        
    def load_datasets(self):
        # Load each dataset separately
        self.datasets['mechanical'] = {
            'high_entropy_alloys': pd.read_csv('high_entropy_alloys.csv'),
            'alloys': pd.read_csv('alloys_dataset.csv'),
            'material_properties': pd.read_csv('materials_properties.csv'),
            'youngs_modulus': pd.read_csv('youngs_modulus.csv')
        }
        
        self.datasets['chemical'] = {
            'corrosion': pd.read_csv('metal_corrosion.csv'),
            'polymer_props': pd.read_csv('polymer_tg_density.csv'),
            'smiles_data': pd.read_csv('polymer_smiles.csv')
        }
        
        self.datasets['biological'] = {
            'cytotoxicity': pd.read_csv('cytotoxicity_ionic_liquids.csv')
        }
    
    def prepare_training_data(self):
        # Create training examples by material matching
        training_examples = []
        
        for material_name in self.get_common_materials():
            mechanical_features = self.extract_mechanical_features(material_name)
            chemical_features = self.extract_chemical_features(material_name)
            biological_features = self.extract_biological_features(material_name)
            
            # Only use examples where we have sufficient data
            if self.has_sufficient_data(mechanical_features, chemical_features, biological_features):
                training_examples.append({
                    'mechanical': mechanical_features,
                    'chemical': chemical_features, 
                    'biological': biological_features,
                    'target': self.get_biocompatibility_label(material_name)  # From cytotoxicity data
                })
        
        return training_examples
    


# In your ML model training
def get_biological_features(material_name):
    """Extract biological/chemical features for a material"""
    
    features = {
        'cytotoxicity_score': 0.5,  # Default neutral
        'corrosion_score': 0.5,     # Default neutral  
        'flexibility_score': 0.5    # Default neutral
    }
    
    # Try to find material in each dataset
    cytotoxicity_match = cytotoxicity_data[
        cytotoxicity_data['Name'].str.contains(material_name, case=False, na=False)
    ]
    if not cytotoxicity_match.empty:
        features['cytotoxicity_score'] = cytotoxicity_match['cytotoxicity_score'].iloc[0]
    
    corrosion_match = corrosion_data[
        corrosion_data['Material'].str.contains(material_name, case=False, na=False)  
    ]
    if not corrosion_match.empty:
        features['corrosion_score'] = corrosion_match['corrosion_score'].iloc[0]
        
    polymer_match = polymer_data[
        polymer_data['Polymer'].str.contains(material_name, case=False, na=False)
    ]
    if not polymer_match.empty:
        features['flexibility_score'] = polymer_match['flexibility_score'].iloc[0]
    
    return list(features.values())