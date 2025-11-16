import pandas as pd
from pathlib import Path

# Set up base paths, script is in utils/, so we only need to go up one level to reach repo root
BASE = Path(__file__).resolve().parent.parent  # This should point to biomaterial-selection/
DATA_DIR = BASE / "data"
MASTER_DIR = BASE / "master_data"
MASTER_DIR.mkdir(parents=True, exist_ok=True)

print(f"BASE directory: {BASE}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"MASTER_DIR: {MASTER_DIR}")

def create_cytotoxicity_score():
    """Process cytotoxicity datasets into a unified biocompatibility score"""
    cytotoxicity_files = [
        'ammonium.csv', 'cholinium.csv', 'phosphonium.csv', 'polycharged.csv', 
        'pyrrolidinium.csv', 'triazolium.csv', 'benzimidazolium.csv', 'guanidinium.csv', 
        'morpholinium.csv', 'piperazinium.csv', 'pyridinium.csv', 'quinolinium.csv',
        'imidazolium.csv', 'piperidinium.csv', 'pyrimidinium.csv', 'thiazolium.csv', 'other.csv',
    ]
    
    all_cytotoxicity = []
    
    # Cytotoxicity data path, imp, use the correct directory name
    cytotoxicity_dir = DATA_DIR / "cytotoxicity_ionic_liquids" / "csv_datasets"  
    print(f"Looking for cytotoxicity files in: {cytotoxicity_dir}")
    print(f"Absolute path: {cytotoxicity_dir.absolute()}")
    
    # Check if directory exists
    if not cytotoxicity_dir.exists():
        print(f"ERROR: Directory does not exist: {cytotoxicity_dir}")
        # Try alternative path without csv_datasets
        cytotoxicity_dir = DATA_DIR / "cytotoxity_ionic_liquids"  # Note the spelling bs
        print(f"Trying alternative path: {cytotoxicity_dir}")
        print(f"Absolute path: {cytotoxicity_dir.absolute()}")
    
    if cytotoxicity_dir.exists():
        actual_files = [f.name for f in cytotoxicity_dir.iterdir() if f.is_file()]
        print(f"Files found in directory: {actual_files}")
    else:
        print(f"ERROR: Cytotoxicity directory does not exist: {cytotoxicity_dir}")
        return pd.DataFrame()
    
    found_files = []
    for file in cytotoxicity_files:
        file_path = cytotoxicity_dir / file
        print(f"Checking: {file_path}")
        
        if file_path.exists():
            print(f"✓ Found: {file}")
            try:
                df = pd.read_csv(file_path)
                df['source_file'] = file.replace('.csv', '')
                all_cytotoxicity.append(df)
                found_files.append(file)
            except Exception as e:
                print(f"✗ Error reading {file}: {e}")
        else:
            print(f"✗ Not found: {file}")
    
    print(f"Successfully loaded {len(found_files)} files: {found_files}")
    
    if not all_cytotoxicity:
        print("WARNING: No cytotoxicity files were loaded!")
        return pd.DataFrame()
    
    # Combine all cytotoxicity data
    cytotoxicity_combined = pd.concat(all_cytotoxicity, ignore_index=True)
    print(f"Combined dataset shape: {cytotoxicity_combined.shape}")
    
    # Create standardized cytotoxicity score
    cytotoxicity_processed = process_cytotoxicity_data(cytotoxicity_combined)
    
    return cytotoxicity_processed

def process_cytotoxicity_data(df):
    """Convert raw cytotoxicity data into standardized scores"""
    if df.empty:
        print("WARNING: Empty DataFrame passed to process_cytotoxicity_data")
        return df
    
    print("Processing cytotoxicity data...")
    
    # first Normalize concentration values (CC50/IC50/EC50)
    # Lower values = more toxic, higher values = less toxic
    df['normalized_toxicity'] = df['CC50/IC50/EC50, mM'].apply(
        lambda x: convert_to_toxicity_score(x) if pd.notna(x) else 0.5
    )
    
    # then Account for different cell lines (more human-relevant lines get higher weight)
    df['cell_line_weight'] = df['Cell line'].apply(weight_cell_line_relevance)
    
    # then Account for different methods (standardize across measurement techniques)
    df['method_reliability'] = df['Method'].apply(weight_method_reliability)
    
    # then Calculate final cytotoxicity score (0-1, where 1 = most biocompatible)
    df['cytotoxicity_score'] = (
        df['normalized_toxicity'] * 0.6 +
        df['cell_line_weight'] * 0.3 +
        df['method_reliability'] * 0.1
    )
    
    print(f"Processed cytotoxicity data shape: {df.shape}")
    return df[['Name', 'Empirical formula', 'CAS', 'SMILES', 'Cell line', 
               'Method', 'cytotoxicity_score', 'normalized_toxicity']]

def convert_to_toxicity_score(concentration):
    """Convert CC50/IC50/EC50 values to normalized toxicity score (0-1)"""
    try:
        concentration = float(concentration)
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
    except (ValueError, TypeError):
        return 0.5  # Default for unparseable values

def weight_cell_line_relevance(cell_line):
    """Weight cell lines by human relevance for medical applications"""
    if pd.isna(cell_line):
        return 0.5
    
    cell_line_str = str(cell_line).upper()
    human_cell_lines = ['HEK293', 'HELA', 'HEPG2', 'MCF-7']  # Example human lines
    if any(line in cell_line_str for line in human_cell_lines):
        return 1.0
    elif 'MOUSE' in cell_line_str or 'RAT' in cell_line_str:
        return 0.7
    else:
        return 0.5  # Other cell lines

def weight_method_reliability(method):
    """Weight different cytotoxicity measurement methods"""
    if pd.isna(method):
        return 0.6
    
    method_str = str(method).upper()
    # High reliability methods
    high_rel = ['MTT', 'MTS', 'ALAMARBLUE', 'PRESTOBLUE', 'CELLTITER-GLO']
    # Medium reliability methods  
    med_rel = ['LDH', 'NRU', 'SRB', 'WST-1', 'WST-8']
    
    if any(m in method_str for m in high_rel):
        return 1.0
    elif any(m in method_str for m in med_rel):
        return 0.8
    else:
        return 0.6  # Other methods

# CORROSION PROCESSING
def create_corrosion_compatibility():
    """Process corrosion data into chemical compatibility scores"""
    corrosion_path = DATA_DIR / "metal_corrosion" / "CORR-DATA_Database.csv"
    
    print(f"Looking for corrosion data at: {corrosion_path}")
    print(f"Absolute path: {corrosion_path.absolute()}")
    
    if not corrosion_path.exists():
        print(f"ERROR: Corrosion file not found: {corrosion_path}")
        # Show what files exist in the directory
        corrosion_dir = DATA_DIR / "metal_corrosion"
        if corrosion_dir.exists():
            actual_files = [f.name for f in corrosion_dir.iterdir() if f.is_file()]
            print(f"Files in metal_corrosion: {actual_files}")
        return pd.DataFrame()
    
    print("Loading corrosion data...")
    corrosion_df = pd.read_csv(corrosion_path)
    print(f"Raw corrosion data shape: {corrosion_df.shape}")
    
    # Focus on biological-relevant environments
    biological_environments = [
        'saline', 'phosphate', 'blood', 'serum', 'water', 'NaCl', 
        'ringer', 'lactate', 'physiological'
    ]
    
    bio_corrosion = corrosion_df[
        corrosion_df['Environment'].str.contains('|'.join(biological_environments), 
        case=False, na=False)
    ].copy()  # Add .copy() to avoid SettingWithCopyWarning
    
    print(f"Biological-relevant corrosion data shape: {bio_corrosion.shape}")
    
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
    corrosion_str = str(corrosion_value).upper()
    if any(rating in corrosion_str for rating in ['A', 'RESISTANT']):
        return 0.9
    elif any(rating in corrosion_str for rating in ['B', 'GOOD']):
        return 0.7
    elif any(rating in corrosion_str for rating in ['C', 'QUESTIONABLE']):
        return 0.4
    elif any(rating in corrosion_str for rating in ['D', 'POOR']):
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
    except (ValueError, TypeError):
        return 0.5  # Default for unparseable values

# POLYMER PROCESSING
def create_polymer_biocompatibility():
    """Process polymer Tg data for flexibility scoring"""
    polymer_path = DATA_DIR / "polymer_tg_density" / "tg_density.csv"
    
    print(f"Looking for polymer data at: {polymer_path}")
    print(f"Absolute path: {polymer_path.absolute()}")
    
    if not polymer_path.exists():
        print(f"ERROR: Polymer file not found: {polymer_path}")
        # Show what files exist in the directory
        polymer_dir = DATA_DIR / "polymer_tg_density"
        if polymer_dir.exists():
            actual_files = [f.name for f in polymer_dir.iterdir() if f.is_file()]
            print(f"Files in polymer_tg_density: {actual_files}")
        return pd.DataFrame()
    
    print("Loading polymer data...")
    polymer_df = pd.read_csv(polymer_path)
    print(f"Raw polymer data shape: {polymer_df.shape}")
    print(f"Polymer data columns: {polymer_df.columns.tolist()}")
    
    # Check what columns actually exist and use the correct ones
    # Common polymer column names might be: 'Polymer', 'Name', 'Material', etc.
    polymer_col = None
    for col in ['Polymer', 'Name', 'Material', 'polymer']:
        if col in polymer_df.columns:
            polymer_col = col
            break
    
    if polymer_col is None:
        print(f"WARNING: No polymer name column found. Available columns: {polymer_df.columns.tolist()}")
        # Use first column as fallback
        polymer_col = polymer_df.columns[0]
        print(f"Using fallback column: {polymer_col}")
    
    # Glass transition temperature indicates flexibility
    # Lower Tg = more flexible at body temperature (37°C)
    polymer_df['flexibility_score'] = polymer_df['Tg'].apply(
        lambda x: 0.9 if x < 37 else 0.7 if x < 60 else 0.4 if x < 100 else 0.2
    )
    
    return polymer_df[[polymer_col, 'Tg', 'Density', 'flexibility_score']]

def create_biological_chemical_unified():
    """Create the final biological/chemical dataset for ML training"""
    print("Starting biological/chemical data processing...")
    
    cytotoxicity_data = create_cytotoxicity_score()
    corrosion_data = create_corrosion_compatibility() 
    polymer_data = create_polymer_biocompatibility()
    
    # Save separate files for each domain
    if not cytotoxicity_data.empty:
        output_path = MASTER_DIR / 'cytotoxicity_scores.csv'
        cytotoxicity_data.to_csv(output_path, index=False)
        print(f"Saved cytotoxicity data: {cytotoxicity_data.shape} to {output_path}")
    
    if not corrosion_data.empty:
        output_path = MASTER_DIR / 'corrosion_scores.csv'
        corrosion_data.to_csv(output_path, index=False)
        print(f"Saved corrosion data: {corrosion_data.shape} to {output_path}")
    
    if not polymer_data.empty:
        output_path = MASTER_DIR / 'polymer_properties.csv'
        polymer_data.to_csv(output_path, index=False)
        print(f"Saved polymer data: {polymer_data.shape} to {output_path}")
    
    print("Biological/chemical data processing complete!")
    
    return {
        'cytotoxicity': cytotoxicity_data,
        'corrosion': corrosion_data,
        'polymers': polymer_data
    }

# Run the processing
if __name__ == "__main__":
    create_biological_chemical_unified()