# NOTE generalised function to load any csv into an SQLite database, currently just functions without a main executable function because all datasets vary and 
# i just wanted something quick for now.


import sqlite3
import pandas as pd
import os

# Set a fixed directory for databases to be crated if a folder dooes or does not exist
BASE_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "master_database")

os.makedirs(BASE_DB_DIR, exist_ok=True)


def get_db_path(name):
    """
    Returns a full path to a DB inside /master_database.
    Example: get_db_path("materials.db")
    """
    return os.path.join(BASE_DB_DIR, name)


def run_sql(db_path, sql):

    """
    Execute any SQL query and return a DataFrame. Mainly for debugging and testing table joins.
    """

    conn = sqlite3.connect(db_path)
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
    conn = sqlite3.connect(db_path)
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
    conn = sqlite3.connect(db_path)

    sql = f"SELECT * FROM {table_name}"

    params = None

    if where:
        conditions = []
        params = []

        for col, value in where.items():
            conditions.append(f"{col} = ?")
            params.append(value)

        sql += " WHERE " + " AND ".join(conditions)

    # Only pass params if they exist
    if params is not None:
        df = pd.read_sql_query(sql, conn, params=params)
    else:
        df = pd.read_sql_query(sql, conn)

    conn.close()
    return df
