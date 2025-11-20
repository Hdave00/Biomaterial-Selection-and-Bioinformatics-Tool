# NOTE generalised function to load any csv into an SQLite database, currently just functions without a main executable function because all datasets vary and 
# i just wanted something quick for now.


import sqlite3
import pandas as pd
import os
import shutil

# Use a writable runtime directory for DB files THAT WORKS ON STREAMLIT
BASE_DB_DIR = os.environ.get("BIOMAT_DB_DIR", "/tmp/master_database")
os.makedirs(BASE_DB_DIR, exist_ok=True)


# Helper: path to DB stored inside runtime writable dir
def get_db_path(name):
    return os.path.join(BASE_DB_DIR, name)


# Path in the repository where dev-time DBs live (read-only on cloud)
REPO_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "master_database")

def ensure_runtime_dbs_copied():

    """
    Copy any .db files from the repo master_database into the writable runtime dir.
    Safe: copies only if target missing or sizes differ.
    """

    if not os.path.isdir(REPO_DB_DIR):
        return
    
    for fname in os.listdir(REPO_DB_DIR):
        if not fname.lower().endswith(".db"):
            continue
        src = os.path.join(REPO_DB_DIR, fname)
        dst = get_db_path(fname)
        try:
            # copy only if not present or different size quick check
            if (not os.path.exists(dst)) or (os.path.getsize(src) != os.path.getsize(dst)):
                shutil.copy2(src, dst)
        except Exception:
            # forgo errors because on some deployments REPO_DB_DIR may be missing
            pass

# run at import time (idempotent)
ensure_runtime_dbs_copied()



def run_sql(db_path, sql):

    """
    Execute any SQL query and return a DataFrame. Mainly for debugging and testing table joins.
    """

    conn = sqlite3.connect(db_path, check_same_thread=False)
    df = pd.read_sql_query(sql, conn)
    conn.close()
    return df


def load_csv_to_sqlite(csv_path, db_path, table_name, if_exists='replace'):

    """
    Load a CSV file into a SQLite database.
    
    :param csv_path: Path to CSV
    :param db_path: Path to SQLite DB
    :param table_name: Target table in DB
    :param if_exists: 'replace' or 'append'

    Example of how to call this function, use in python console for safety:

    >>> from src.utils.csv_database_loader import load_csv_to_sqlite, get_db_path
    >>> csv_path = "path/to/.csv"
    >>> db_path = get_db_path("database_name.db")
    >>> load_csv_to_sqlite(csv_path=csv_path, db_path=db_path, table_name="name_of_table", if_exists="replace")

    """

    # use path checking, if file not foundm raise error
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # read the csv as a dataframe
    df = pd.read_csv(csv_path)

    # not strict, but keep for now to normalize column names
    df.columns = [c.strip().replace(" ", "_").replace("%","pct").replace("°","C").replace("/","_") for c in df.columns]
    
    # create a connection, covert the given dataframe of CSV into an SQL table.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    df.to_sql(table_name, conn, if_exists=if_exists, index=False)

    # commit the change and close connection
    conn.commit()
    conn.close()
    print(f"Loaded {len(df)} rows into {table_name}")


def query_table(db_path, table_name, where: dict | None = None):

    """
    Query a SQLite table and return a DataFrame.
    If `where` is provided, it should be a dict of {column: value}.

    NOTE --- This function is the one called in materials_app.py in the local search render to query the database of different domains.

    :param db_path: Path to SQLite DB
    :param table_name: Name of the table
    :param where: optional WHERE clause (string)

    Example:
        query_table("materials.db", "structural_materials",
                    columns=["Material_Name", "Density_gcm3"],
                    where="Density_gcm3 > 5",
                    limit=10)
    """

    # clean column format and parameterized queries to prevent injection 
    conn = sqlite3.connect(db_path, check_same_thread=False)

    sql = f"SELECT * FROM {table_name}"

    params = None

    if where:
        conds = []
        params = []

        for k, v in where.items():
            conds.append(f"{k} = ?")
            params.append(v)
        sql += " WHERE " + " AND ".join(conds)
    
    df = pd.read_sql_query(sql, conn, params=params) if params else pd.read_sql_query(sql, conn)
    conn.close()
    return df
