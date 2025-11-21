# scripts/build_databases.py

import os
import sqlite3
import pandas as pd
from src.utils.csv_database_loader import load_csv_to_sqlite, get_db_path, normalize_columns

# Detect project root (repo root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Folders
MASTER_DATA = os.path.join(PROJECT_ROOT, "master_data")
MASTER_DB = os.path.join(PROJECT_ROOT, "master_database")

# Output directory for all .db files
OUT_DIR = os.path.join(PROJECT_ROOT, "master_database")
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

# Keep track of DBs we have already written to
initialized_dbs = set()

for csv_path, dbname, table_name in TASKS:
    out_db = os.path.join(OUT_DIR, dbname)
    print(f"Processing: {csv_path} -> {out_db} [{table_name}]")

    # first table in this DB -> replace, subsequent -> append
    if dbname not in initialized_dbs:
        mode = "replace"
        initialized_dbs.add(dbname)
    else:
        mode = "append"

    if mode == "replace":

        # temp DB -> atomic replace is fine
        load_csv_to_sqlite(
            csv_path=csv_path,
            db_path=out_db,
            table_name=table_name,
            if_exists=mode,
            chunksize=20000
        )
    else:
        # append mode → write directly into the DB
        df = pd.read_csv(csv_path, chunksize=20000)
        conn = sqlite3.connect(out_db)
        for chunk in df if hasattr(df, "__iter__") else [df]:
            chunk.columns = normalize_columns(chunk.columns.tolist())
            chunk.to_sql(table_name, conn, if_exists="append", index=False)
        conn.close()

print("All DBs generated in:", OUT_DIR)
print("Commit these NEW *.db files to the repo so Streamlit can read them.")