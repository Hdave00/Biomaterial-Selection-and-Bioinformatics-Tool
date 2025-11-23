import sqlite3
import pandas as pd
import os
import shutil

# --------------------------------------------------------
# DIRECTORY SETUP
# --------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DB_DIR = os.path.join(PROJECT_ROOT, "runtime_dbs")
BASE_DB_DIR = os.environ.get("BIOMAT_DB_DIR", DEFAULT_DB_DIR)
os.makedirs(BASE_DB_DIR, exist_ok=True)

def get_db_path(name):
    return os.path.join(BASE_DB_DIR, name)

# Single master DB file
MASTER_DB_NAME = "biomat.db"

def get_master_db_path():
    return get_db_path(MASTER_DB_NAME)


# --------------------------------------------------------
# COPY DEV DBs INTO RUNTIME (if exist in repo)
# --------------------------------------------------------

REPO_DB_DIR = os.path.join(PROJECT_ROOT, "master_database")

def ensure_runtime_dbs_copied():
    if not os.path.isdir(REPO_DB_DIR):
        return
    for fname in os.listdir(REPO_DB_DIR):
        if not fname.lower().endswith(".db"):
            continue
        src = os.path.join(REPO_DB_DIR, fname)
        dst = get_db_path(fname)
        try:
            if (not os.path.exists(dst)) or (os.path.getsize(src) != os.path.getsize(dst)):
                shutil.copy2(src, dst)
        except Exception:
            pass

ensure_runtime_dbs_copied()


# --------------------------------------------------------
# SQL HELPERS
# --------------------------------------------------------

def run_sql(db_path, sql):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    df = pd.read_sql_query(sql, conn)
    conn.close()
    return df


def load_csv_to_sqlite(csv_path, db_path, table_name, if_exists='replace'):

    """ ---- How to run function in python console: ----

        from src.utils.csv_database_loader import load_csv_to_sqlite, get_master_db_path

        db = get_master_db_path()

        load_csv_to_sqlite("master_data/corrosion/corr_lookup_database.csv", db, "corrosion_lookup")
        load_csv_to_sqlite("master_data/biological/chemical_toxicity_measurements.csv", db, "cytotoxicity_lookup")
        load_csv_to_sqlite("master_data/unified_material_data.csv", db, "unified_material_data")
        load_csv_to_sqlite("master_data/polymer_tg/polymer_lookup_data.csv", db, "polymer_lookup")
        load_csv_to_sqlite("master_data/HEA/high_entropy_alloys_properties.csv", db, "hea_lookup")
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df.columns = [
        c.strip().replace(" ", "_").replace("%","pct").replace("°","C").replace("/","_")
        for c in df.columns
    ]

    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    conn.commit()
    conn.close()
    print(f"Loaded {len(df)} rows into table '{table_name}' in DB '{db_path}'")


def get_distinct_values(table_name: str, col: str, limit: int = 500):
    db_path = get_master_db_path()
    sql = f"SELECT DISTINCT {col} FROM {table_name} WHERE {col} IS NOT NULL LIMIT {limit}"
    return pd.read_sql_query(sql, sqlite3.connect(db_path, check_same_thread=False))[col].astype(str).tolist()


def create_index(db_path: str, table_name: str, column_name: str, unique: bool = False):

    conn = sqlite3.connect(db_path, check_same_thread=False)
    idx_name = f"idx_{table_name}_{column_name}"
    unique_sql = "UNIQUE" if unique else ""
    try:
        conn.execute(f"CREATE {unique_sql} INDEX IF NOT EXISTS {idx_name} ON {table_name}({column_name})")
        conn.commit()
    finally:
        conn.close()


def query_table_columns(table_name: str, columns: list[str] | None = None, where: dict | None = None, limit: int | None = None):
    
    db_path = get_master_db_path()
    cols = ", ".join(columns) if columns else "*"
    sql = f"SELECT {cols} FROM {table_name}"
    params = []
    if where:
        conds = []
        for k, v in where.items():
            conds.append(f"{k} = ?")
            params.append(v)
        sql += " WHERE " + " AND ".join(conds)
    if limit:
        sql += f" LIMIT {int(limit)}"
    return pd.read_sql_query(sql, sqlite3.connect(db_path, check_same_thread=False), params=params if params else None)


def query_table(db_path, table_name, where: dict | None = None):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")

    sql = f"SELECT * FROM {table_name}"
    params = None

    if where:
        conditions = []
        params = []
        for col, value in where.items():
            conditions.append(f"{col} = ?")
            params.append(value)
        sql += " WHERE " + " AND ".join(conditions)

    df = pd.read_sql_query(sql, conn, params=params) if params else pd.read_sql_query(sql, conn)
    conn.close()
    return df