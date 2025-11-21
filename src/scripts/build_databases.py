# scripts/build_databases.py

import os
from src.utils.csv_database_loader import load_csv_to_sqlite, get_db_path

# Detect project root (repo root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Folders
MASTER_DATA = os.path.join(PROJECT_ROOT, "master_data")
MASTER_DB = os.path.join(PROJECT_ROOT, "master_database")

# Output directory for all .db files
OUT_DIR = MASTER_DB
os.makedirs(OUT_DIR, exist_ok=True)

# CSV -> to DB mapping
TASKS = [
    # corrosion
    (os.path.join(MASTER_DATA, "corrosion", "corr_lookup_database.csv"),
     "corrosion.db", "corr_lookup"),

    # cytotoxicity
    (os.path.join(MASTER_DATA, "biological", "chemical_toxicity_measurements.csv"),
     "cytotoxicity.db", "cytotoxicity_lookup"),

    # polymers
    (os.path.join(MASTER_DATA, "polymer_tg", "polymer_lookup_data.csv"),
     "polymer.db", "polymer_lookup"),

    # unified structural materials -> materials.db (table: structural_materials)
    (os.path.join(MASTER_DATA, "unified_material_data.csv"),
     "materials.db", "structural_materials"),

    # high entropy alloys -> materials.db (table: high_entropy_alloys)
    (os.path.join(MASTER_DATA, "HEA", "high_entropy_alloys_properties.csv"),
     "materials.db", "high_entropy_alloys"),
]

# Run all tasks
for csv_path, dbname, table_name in TASKS:
    out_db = os.path.join(OUT_DIR, dbname)
    print(f"Processing: {csv_path} -> {out_db} [{table_name}]")

    load_csv_to_sqlite(
        csv_path=csv_path,
        db_path=out_db,
        table_name=table_name,
        if_exists="replace",
        chunksize=20000
    )

print("All DBs generated in:", OUT_DIR)
print("Commit these *.db files to the repo so Streamlit can read them.")