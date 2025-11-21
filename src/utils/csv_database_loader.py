# NOTE generalised function to load any csv into an SQLite database, currently just functions without a main executable function because all datasets vary and 
# i just wanted something quick for now.


# src/utils/csv_database_loader.py
import os
import sqlite3
import pandas as pd
import shutil
import tempfile
import re
from typing import Tuple, Optional


# Path in repository where dev-time DBs live (read-only on cloud)
REPO_DB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "master_database"
)

def get_db_path(name: str) -> str:
    """Return full runtime-writable path for a DB filename."""

    return os.path.join(REPO_DB_DIR, name)

def copy_repo_dbs_to_runtime():
    """
    Copy any .db files from the repo master_database into the writable runtime dir.
    Call this explicitly from a maintenance script if you want to seed runtime DBs.
    """
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
            # don't raise; repository may be missing in some deployments
            pass

# ---------- Utilities ----------
def _sanitize_table_name(name: str) -> str:
    # Allow only letters, numbers, underscores
    if not re.match(r'^[A-Za-z0-9_]+$', name):
        raise ValueError("Invalid table name. Only A-Za-z0-9_ and underscore allowed.")
    return name

def _normalize_columns(cols):
    """
    - strip whitespace
    - replace spaces with _
    - replace special chars
    - deduplicate by appending suffixes for exact duplicates
    """
    cleaned = []
    seen = {}
    for c in cols:
        c0 = str(c).strip()
        c0 = c0.replace(" ", "_").replace("%", "pct").replace("°", "deg").replace("/", "_").replace("$", "")
        # collapse multiple underscores
        c0 = re.sub(r'__+', '_', c0)
        if not c0:
            c0 = "col"
        # dedupe
        if c0 in seen:
            seen[c0] += 1
            c_unique = f"{c0}_{seen[c0]}"
        else:
            seen[c0] = 0
            c_unique = c0
        cleaned.append(c_unique)
    return cleaned

# ---------- CSV -> SQLite (safe) ----------
def load_csv_to_sqlite(csv_path: str, db_path: str, table_name: str, if_exists: str = "replace", chunksize: Optional[int]=20000):
    """
    Load CSV safely into SQLite.
    - writes to a temporary DB file then atomically replaces target
    - chunksize: if set, pandas will stream CSV to reduce memory usage
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    table_name = _sanitize_table_name(table_name)

    # Read first row to normalize headers (without loading whole file)
    # Use pandas to parse headers reliably
    first_df = pd.read_csv(csv_path, nrows=5)  # small read for header inference
    normalized_cols = _normalize_columns(first_df.columns.tolist())

    # Build temp DB path in same directory as final (atomic move)
    db_dir = os.path.dirname(db_path)
    os.makedirs(db_dir, exist_ok=True)
    fd, tmp_db = tempfile.mkstemp(prefix="tmp_", suffix=".db", dir=db_dir)
    os.close(fd)
    try:
        conn = sqlite3.connect(tmp_db, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")  # safer for concurrent readers
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.commit()

        # Read CSV in chunks (if chunksize provided)
        total = 0
        if chunksize:
            iterator = pd.read_csv(csv_path, chunksize=2000)
            first_chunk = True
            for chunk in iterator:
                # normalize column names to match the header normalization
                chunk.columns = _normalize_columns(chunk.columns.tolist())
                # ensure order/columns align with the initial normalization
                # reindex missing cols as NaN, drop extras
                chunk = chunk.reindex(columns=normalized_cols, fill_value=None)
                chunk.to_sql(table_name, conn, if_exists=("replace" if first_chunk else "append"), index=False)
                first_chunk = False
                total += len(chunk)
        else:
            df = pd.read_csv(csv_path)
            df.columns = _normalize_columns(df.columns.tolist())
            df.to_sql(table_name, conn, if_exists=if_exists, index=False)
            total = len(df)

        conn.commit()
        conn.close()

        # atomic replace
        os.replace(tmp_db, db_path)
        print(f"Loaded {total} rows into {table_name} at {db_path}")
    except Exception:
        # ensure temp file removal on failure
        try:
            os.remove(tmp_db)
        except Exception:
            pass
        raise

# ---------- Query helper ----------
def query_table(db_path: str, table_name: str, where: dict | None = None, limit: Optional[int] = None) -> Tuple[pd.DataFrame, dict]:
    """
    Read-only query. Returns (DataFrame, metadata).
    `where` must be a dict of exact matches (ANDed). For range/complex queries, use `run_sql`.
    """
    table_name = _sanitize_table_name(table_name)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB not found at {db_path}")

    conn = sqlite3.connect(db_path, check_same_thread=False)
    sql = f"SELECT * FROM {table_name}"
    params = []
    if where:
        conds = []
        for k, v in where.items():
            # basic column name sanitation
            if not re.match(r'^[A-Za-z0-9_]+$', k):
                raise ValueError("Invalid column name in where.")
            conds.append(f"{k} = ?")
            params.append(v)
        sql += " WHERE " + " AND ".join(conds)
    if limit and isinstance(limit, int) and limit > 0:
        sql += f" LIMIT {limit}"

    df = pd.read_sql_query(sql, conn, params=params if params else None)
    # lightweight metadata
    meta = {"nrows": len(df), "columns": df.columns.tolist()}
    conn.close()
    return df, meta

def run_sql(db_path: str, sql: str) -> pd.DataFrame:
    """Run arbitrary read SQL (for debugging)."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    df = pd.read_sql_query(sql, conn)
    conn.close()
    return df


if __name__ == "__main__":
    from src.utils.csv_database_loader import copy_repo_dbs_to_runtime
    copy_repo_dbs_to_runtime()