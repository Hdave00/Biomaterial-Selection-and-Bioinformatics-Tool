"""
utils/merge_data_v2.py

Improved merging / cleaning for materials_properties.csv (Kaggle).
- Prefers 'Std' column for standard when available
- Keeps original Material text and creates a Clean_Name for merging
- Extracts Material_Type with more patterns
- Normalizes E, G, Ro -> GPa, GPa, g/cm3
- Produces missing_report.csv and a summary CSV
"""

import os
import re
import pandas as pd


# adjust as repo data folder location changes
BASE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "materials_data")
INPUT_CSV = os.path.join(BASE_PATH, "materials_properties_data.csv")
OUTPUT_CSV = os.path.join(BASE_PATH, "unified_material_data.csv")
MISSING_REPORT = os.path.join(BASE_PATH, "missing_report.csv")
UNMATCHED_TYPES = os.path.join(BASE_PATH, "unmatched_material_types.csv")

# Helper funcs
def safe_str(x):
    return "" if pd.isna(x) else str(x)

def normalize_whitespace(s):
    return re.sub(r"\s+", " ", safe_str(s)).strip()

def make_clean_name(material_raw):
    """
    Create a normalized, merge-friendly material "clean name"
    - Remove punctuation except hyphen and plus
    - Collapse multiple spaces
    - Uppercase
    - Keep composition tokens (like, Ti-6Al-4V) as it is when possible
    """
    s = normalize_whitespace(material_raw)

    # remove leading standard tokens like "ANSI," if accidentally present in name
    s = re.sub(r'^\s*(ANSI|ISO|DIN|ASTM|BS|SAE|GB|JIS|GOST)[\s,:-]*', '', s, flags=re.I)

    # Replace commas with space
    s = s.replace(",", " ")

    # Keep letters, numbers, dash, plus, parentheses; remove other punct
    s = re.sub(r"[^A-Z0-9\-\+\(\)\/ ]+", " ", s.upper())
    s = re.sub(r"\s+", " ", s).strip()

    return s

def prefer_std_column(row):

    """
    Return the standard: prefer the 'Std' column if present and non-empty,
    otherwise attempt to extract from the Material text (fallback).
    """
    if "Std" in row and pd.notna(row["Std"]) and str(row["Std"]).strip() != "":
        return str(row["Std"]).strip().upper()
    
    # fallback, try to parse first token or known abbreviations in Material string
    mat = safe_str(row.get("Material", ""))

    # look for tokens at start like "ANSI", "ISO", "DIN", "ASTM", "BS", "SAE", "GB" and "NF"
    m = re.match(r'^(ANSI|ISO|DIN|ASTM|BS|SAE|GB|JIS|GOST|NF)\b', mat, flags=re.I)
    if m:
        return m.group(1).upper()
    
    # last fallback: search for any of the known codes anywhere
    for token in ["ANSI","ASTM","ISO","DIN","BS","SAE","GB","JIS","GOST","NF"]:
        if token in mat.upper():
            return token
    return "UNKNOWN"

# expanded material type matcher (ordered, longer-first)
MATERIAL_TYPE_PATTERNS = [
    # Copper alloys first (more specific)
    (r'\bMUNTZ METAL\b', 'COPPER_ALLOY'),
    (r'\bLEADED MUNTZ METAL\b', 'COPPER_ALLOY'),
    (r'\bC\d{3,5}\b', 'COPPER_ALLOY'),  # C28000, C36500 style
    
    # Cast irons
    (r'\bNODULAR CAST IRON\b', 'NODULAR_CAST_IRON'),
    (r'\bMALLEABLE CAST IRON\b', 'MALLEABLE_CAST_IRON'),
    (r'\bGRAY CAST IRON\b', 'GRAY_CAST_IRON'),
    (r'\bCAST IRON\b', 'CAST_IRON'),
    (r'\bGGG\b', 'NODULAR_CAST_IRON'),  # DIN GGG-50, GGG-60, etc.
    (r'\bGG\b', 'GRAY_CAST_IRON'),      # DIN GG-15, GG-20, etc.
    (r'\bGTS\b', 'CAST_IRON'),          # DIN GTS-40, GTS-50
    
    # Standardized steel patterns
    (r'\b(ST|USt|RSt|St)\d', 'STEEL'),           # DIN St37, USt34, RSt37
    (r'\b(Ck|Cf)\d', 'CARBON_STEEL'),            # DIN Ck10, Ck15, Cf35
    (r'\b(\d+Mn\d+|\d+Cr\d+|\d+Mo\d+)', 'ALLOY_STEEL'),  # DIN 17Mn4, 34Cr4, 15Mo3
    (r'\b(\d+CrMo\d+|\d+CrV\d+)', 'ALLOY_STEEL'),        # DIN 25CrMo4, 50CrV4
    (r'\b(\d+CrNiMo\d+|\d+Ni\d+)', 'ALLOY_STEEL'),       # DIN 17CrNiMo6, 10Ni14
    (r'\bWTSt\d', 'STEEL'),                     # DIN WTSt52-3
    
    # JIS steel patterns
    (r'\bS(NC|CM|UP|UHP|C|MnC|Cr)\d', 'ALLOY_STEEL'),  # JIS SCM430, SUP9, etc.
    (r'\bSUS\d', 'STAINLESS_STEEL'),            # JIS SUS304, SUS316, etc.
    (r'\bSUH\d', 'HEAT_RESISTANT_STEEL'),       # JIS SUH1, SUH330
    (r'\bS\d+C\b', 'CARBON_STEEL'),             # JIS S10C, S45C, etc.
    (r'\bSS\d+\b', 'STEEL'),                    # JIS SS330, SS400, SS490
    (r'\bSTKM\d', 'STEEL'),                     # JIS STKM13B, STKM16A
    (r'\bSM\d+\b', 'STEEL'),                    # JIS SM570, SM52OC
    
    # BS steel patterns
    (r'\bBS\s+(\d+[A-Z]|\d+[A-Z]\d+|\d+M\d+)', 'STEEL'),  # BS 230M07, 212M36, 070M26
    (r'\bBS\s+Grade\s+\d+', 'STEEL'),           # BS Grade 360, Grade 430
    (r'\bBS\s+\d+[A-Z]\d+', 'STEEL'),           # BS 530A36, 525A60
    (r'\bBS\s+\d+S\d+', 'STAINLESS_STEEL'),     # BS 304S15, 316S11, 430S17
    
    # CSN patterns (Czech standards)
    (r'\bCSN\s+1[01]\d{4}', 'STEEL'),           # CSN 10370, 11110, 11301, etc.
    (r'\bCSN\s+4[012]\d{4}', 'STEEL'),          # CSN 422303, 423042, etc.
    
    # NF (French) steel patterns
    (r'\bNF\s+[XZ]\w+\d', 'STEEL'),             # NF XC10, Z6C13, Z12C13
    (r'\bNF\s+\d+[A-Z]+\d', 'STEEL'),           # NF 13MF4, 35MF6
    (r'\bNF\s+[A-Z]\d+', 'STEEL'),              # NF A34-2, E24-2, A48CP
    (r'\bNF\s+\d+[CDMNS]+\d', 'ALLOY_STEEL'),   # NF 35CD4, 16MC5, 30CND8
    
    # Generic standards detection
    (r'\b(DIN|BS|CSN|NF|JIS)\s+', 'STEEL'),    # Any material with these standards
    
    # Basic material types
    (r'\bBRASS\b', 'BRASS'),
    (r'\bBRONZE\b', 'BRONZE'),
    (r'\bCOPPER-NICKEL\b', 'COPPER_NICKEL'),
    (r'\bCOPPER\b', 'COPPER'),
    (r'\bMAGNESIUM\b', 'MAGNESIUM'),
    (r'\bALUMINUM|ALUMINIUM\b', 'ALUMINUM'),
    (r'\bTITANIUM\b', 'TITANIUM'),
    (r'\bNICKEL\b', 'NICKEL'),
    (r'\bSTAINLESS\b', 'STAINLESS_STEEL'),
    (r'\bSTEEL\b', 'STEEL'),
    (r'\bALLOY\b', 'ALLOY'),
    (r'\bPEROVSKITE\b', 'PEROVSKITE'),
    (r'\bCERAMIC\b', 'CERAMIC'),
    (r'\bPOLYMER\b', 'POLYMER'),
]

# function to extract material type from a name, because this dataset has some material names as "JIS SM570" which are basically standardised materials
# jsut named after the standard
def extract_material_type_from_name(material_name):

    name_u = safe_str(material_name).upper()

    # First check specific patterns
    for patt, label in MATERIAL_TYPE_PATTERNS:
        if re.search(patt, name_u):
            return label
        
    # Handle code-based heuristics
    if re.search(r'\bC\d{3,5}\b', name_u):  # C28000, C36500 style
        return "COPPER_ALLOY"
    
    if re.search(r'\bEN\s+[A-Z0-9]+\b', name_u):
        return "STEEL"
    
    if re.search(r'\bDIN\b', name_u) and re.search(r'\bX\d', name_u):
        return "STEEL"
    
    # If it starts with a known standard but no other pattern matched
    if re.search(r'^(DIN|BS|CSN|NF|JIS|ISO|ASTM)\b', name_u):
        return "STEEL"
    
    return "UNKNOWN"

# then we want to convert the units given a dataframe
def convert_units(df):

    # rename MPa columns if present
    if "E" in df.columns and df["E"].notna().any():
        df["Youngs_Modulus_MPa"] = df["E"]
    if "G" in df.columns and df["G"].notna().any():
        df["Shear_Modulus_MPa"] = df["G"]

    # density 'Ro' appears to be kg/m3 -> convert to g/cm3
    if "Ro" in df.columns and df["Ro"].notna().any():
        df["Density_kgm3"] = df["Ro"]

    # convert to GPa and g/cm3
    if "Youngs_Modulus_MPa" in df.columns:
        df["Youngs_Modulus_GPa"] = df["Youngs_Modulus_MPa"] / 1000.0
    if "Shear_Modulus_MPa" in df.columns:
        df["Shear_Modulus_GPa"] = df["Shear_Modulus_MPa"] / 1000.0
    if "Density_kgm3" in df.columns:
        df["Density_gcm3"] = df["Density_kgm3"] / 1000.0
    return df

# Load the CSV
print("Loading:", INPUT_CSV)

# read as strings initially to avoid surprises
df = pd.read_csv(INPUT_CSV, dtype=str)  

print("Original columns:", df.columns.tolist())

# Try to force numeric columns (some CSVs have missing numeric cells)
num_cols_candidates = ["Su","Sy","E","G","A5","Bhn","HV","mu","Ro"]
for c in num_cols_candidates:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Create Material_ID if present as "ID" else, fallback to index
if "ID" in df.columns:
    df["Material_ID"] = df["ID"]
else:
    df["Material_ID"] = df.index.astype(str)


if "Material_Raw" not in df.columns:
    df["Material_Raw"] = df.get("Material", "").astype(str)

if "Clean_Name" not in df.columns:
    df["Clean_Name"] = df["Material_Raw"].apply(make_clean_name)

# Ensure Material_Name exists (same as Clean_Name for now)
if "Material_Name" not in df.columns:
    df["Material_Name"] = df["Clean_Name"]

# Standard Selection (ANSI / ISO / DIN / UNKNOWN)
df["Standard"] = df.apply(prefer_std_column, axis=1)

# Rename Desc → Notes if present
if "Desc" in df.columns:
    df = df.rename(columns={"Desc": "Notes"})

# Material Type extraction
df["Material_Type"] = df["Material_Name"].apply(extract_material_type_from_name)

# Units conversion
df = convert_units(df)

# Rename common numeric columns for clarity (if present)
rename_map = {
    "Su": "Tensile_Strength_MPa",
    "Sy": "Yield_Strength_MPa",
    "A5": "Elongation_percent",
    "Bhn": "Hardness_BHN",
    "HV": "Hardness_HV",
    "mu": "Poisson_Ratio"
}


for k, v in rename_map.items():
    if k in df.columns:
        df.rename(columns={k: v}, inplace=True)

# Select final columns that matter for this dataset
final_cols = [
    "Material_ID", "Material_Raw", "Material_Name", "Clean_Name",
    "Standard", "Material_Type",
    "Heat treatment",
    "Tensile_Strength_MPa", "Yield_Strength_MPa",
    "Youngs_Modulus_GPa", "Shear_Modulus_GPa", "Poisson_Ratio",
    "Density_gcm3", "Elongation_percent", "Hardness_BHN", "Hardness_HV",
    "Notes",
]

# use list iteration to check if there are any duplicated in the final file
final_existing = [c for c in final_cols if c in df.columns]
unified = df[final_existing].copy()

# Save missing report
missing = unified.isna().sum().sort_values(ascending=False)
missing.to_csv(MISSING_REPORT)
print("Saved missing report ->", MISSING_REPORT)

# Save unmatched material types for inspection
cols_to_show = [c for c in ["Material_ID","Material_Raw","Clean_Name"] if c in unified.columns]

unmatched = unified[unified["Material_Type"] == "UNKNOWN"][cols_to_show]

if len(unmatched) > 0:
    unmatched[cols_to_show].to_csv(UNMATCHED_TYPES, index=False)
    print(f"Unmatched material types found: {len(unmatched)}")
    print("Saved report ->", UNMATCHED_TYPES)
else:
    print("No unmatched material types!")

# Save unified table
unified.to_csv(OUTPUT_CSV, index=False)
print("Saved unified dataset ->", OUTPUT_CSV)

# Print short summary
print("Summary:")
print(" Rows:", len(unified))
print(" Distinct standards:", unified["Standard"].nunique(), "examples ->", unified["Standard"].unique()[:10])
print(" Distinct material types:", unified["Material_Type"].nunique(), "examples ->", unified["Material_Type"].unique()[:20])