import pandas as pd
import os

def create_cytotoxicity_score():
    """Process cytotoxicity datasets into a unified biocompatibility score"""
    cytotoxicity_files = [
        'ammonium.csv', 'benzimidazolium.csv', 'cholinium.csv', 'guanidinium.csv',
        'imidazolium.csv', 'morpholinium.csv', 'phosphonium.csv', 'pyridinium.csv',
        # ... all other ionic liquid files
    ]
    
    all_cytotoxicity = []
    
    for file in cytotoxicity_files:
        file_path = f'../data/cytotoxity_ionic_liquids/csv_datasets/{file}'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['source_file'] = file.replace('.csv', '')
            all_cytotoxicity.append(df)
    
    # Combine all cytotoxicity data
    cytotoxicity_combined = pd.concat(all_cytotoxicity, ignore_index=True)
    
    # Create standardized cytotoxicity score
    cytotoxicity_processed = process_cytotoxicity_data(cytotoxicity_combined)
    
    return cytotoxicity_processed

def process_cytotoxicity_data(df):
    """Convert raw cytotoxicity data into standardized scores"""
    
    # 1. Normalize concentration values (CC50/IC50/EC50)
    # Lower values = more toxic, higher values = less toxic
    df['normalized_toxicity'] = df['CC50/IC50/EC50, mM'].apply(
        lambda x: convert_to_toxicity_score(x) if pd.notna(x) else None
    )
    
    # 2. Account for different cell lines (more human-relevant lines get higher weight)
    df['cell_line_weight'] = df['Cell line'].apply(weight_cell_line_relevance)
    
    # 3. Account for different methods (standardize across measurement techniques)
    df['method_reliability'] = df['Method'].apply(weight_method_reliability)
    
    # 4. Calculate final cytotoxicity score (0-1, where 1 = most biocompatible)
    df['cytotoxicity_score'] = (
        df['normalized_toxicity'] * 0.6 +
        df['cell_line_weight'] * 0.3 +
        df['method_reliability'] * 0.1
    )
    
    return df[['Name', 'Empirical formula', 'CAS', 'SMILES', 'Cell line', 
               'Method', 'cytotoxicity_score', 'normalized_toxicity']]

def convert_to_toxicity_score(concentration):
    """Convert CC50/IC50/EC50 values to normalized toxicity score (0-1)"""
    if concentration <= 0.1:    # Highly toxic
        return 0.1
    elif concentration <= 1.0:  # Moderately toxic  
        return 0.3
    elif concentration <= 10.0: # Mildly toxic
        return 0.6
    elif concentration <= 100.0: # Low toxicity
        return 0.8
    else:                       # Very low toxicity
        return 0.95

def weight_cell_line_relevance(cell_line):
    """Weight cell lines by human relevance for medical applications"""
    human_cell_lines = ['HEK293', 'HeLa', 'HepG2', 'MCF-7']  # Example human lines
    if any(line in str(cell_line) for line in human_cell_lines):
        return 1.0
    elif 'mouse' in str(cell_line).lower() or 'rat' in str(cell_line).lower():
        return 0.7
    else:
        return 0.5  # Other cell lines
    

# TODO
def create_corrosion_compatibility():
    """Process corrosion data into chemical compatibility scores"""

    corrosion_df = pd.read_csv('../data/metal_corrosion/CORR-DATA_Database.csv')
    
    # Focus on biological-relevant environments
    biological_environments = [
        'saline', 'phosphate', 'blood', 'serum', 'water', 'NaCl', 
        'ringer', 'lactate', 'physiological'
    ]
    
    bio_corrosion = corrosion_df[
        corrosion_df['Environment'].str.contains('|'.join(biological_environments), 
        case=False, na=False)
    ]
    
    # Convert corrosion ratings to scores
    bio_corrosion['corrosion_score'] = bio_corrosion['Rate (mm/yr) or Rating'].apply(
        convert_corrosion_to_score
    )
    
    return bio_corrosion[['Material', 'Environment', 'corrosion_score']]

def convert_corrosion_to_score(corrosion_value):
    """Convert corrosion rates/ratings to compatibility score (0-1)"""
    if pd.isna(corrosion_value):
        return 0.5  # Unknown = neutral
    
    # Handle letter ratings
    if str(corrosion_value).upper() in ['A', 'RESISTANT']:
        return 0.9
    elif str(corrosion_value).upper() in ['B', 'GOOD']:
        return 0.7
    elif str(corrosion_value).upper() in ['C', 'QUESTIONABLE']:
        return 0.4
    elif str(corrosion_value).upper() in ['D', 'POOR']:
        return 0.1
    
    # Handle numeric rates (mm/yr) - lower = better
    try:
        rate = float(corrosion_value)
        if rate < 0.01:    # Excellent resistance
            return 0.9
        elif rate < 0.1:   # Good resistance  
            return 0.7
        elif rate < 0.5:   # Moderate resistance
            return 0.5
        elif rate < 1.0:   # Poor resistance
            return 0.3
        else:              # Very poor resistance
            return 0.1
    except:
        return 0.5  # Default for unparseable values
    


def create_biological_chemical_unified():
    """Create the final biological/chemical dataset for ML training"""
    
    cytotoxicity_data = create_cytotoxicity_score()
    corrosion_data = create_corrosion_compatibility() 
    polymer_data = create_polymer_biocompatibility()
    
    # For ML training, we'll need to map materials across datasets
    # This creates a sparse but valuable training set
    
    unified_bio_chemical = {
        'cytotoxicity': cytotoxicity_data,
        'corrosion': corrosion_data,
        'polymers': polymer_data
    }
    
    # Save separate files for each domain
    cytotoxicity_data.to_csv('../data/master_data/cytotoxicity_scores.csv', index=False)
    corrosion_data.to_csv('../data/master_data/corrosion_scores.csv', index=False)
    polymer_data.to_csv('../data/master_data/polymer_properties.csv', index=False)
    
    return unified_bio_chemical