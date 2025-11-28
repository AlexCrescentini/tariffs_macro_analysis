import os, requests, gzip, io, pandas as pd, re, keyword 

def process_data(output_folder, conn, data_list, dataset_name=None):

    print(f" Processing {dataset_name} data...")
    os.makedirs(output_folder, exist_ok=True)
    figaro_tables = [] if dataset_name == "a_figaro_data_y" else None # Initialize list for unified FIGARO table below

    for url in data_list:
        full_url = url[0] + url[1] + url[2]
        try:
            print(f"  Downloading {url[1]}...")
            df = download_data(full_url, url[1], timeout=60)
            print("  Cleaning table name for SQL...")
            df, table_name = make_valid_table(df, table_name=url[1])
            print("  Saving data as SQL table...")
            save_sql_table(df, conn, table_name)
            if dataset_name == "a_figaro_data_y": # For unified FIGARO table below
                figaro_tables.append(table_name)

        except Exception as e:
            print(f"  Failed to process {url[1]}: {e}")

    # Create also a unified FIGARO IO table
    if dataset_name == "a_figaro_data_y" and figaro_tables and len(figaro_tables) > 1:
        print("  Creating unified FIGARO IO table (naio_10_fcp_ii) inside DuckDB...")
        try:
            union_query = " UNION ALL ".join([f"SELECT * FROM {tbl}" for tbl in figaro_tables])
            conn.execute(f"CREATE OR REPLACE TABLE naio_10_fcp_ii AS {union_query}")
            print("  Unified FIGARO IO table created successfully (naio_10_fcp_ii)\n")
        except Exception as e:
            print(f"  Failed to create unified FIGARO IO table: {e}")
    elif dataset_name == "a_figaro_data_y":
        print(f"  Expected multiple FIGARO tables, found {len(figaro_tables) if figaro_tables else 0}. Skipping union.\n")

    print(f"  Completed!\n")

def download_data(url, filename, timeout=60):
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    content = io.BytesIO(response.content)
    try:
        if content.getbuffer().nbytes > 2 and content.getvalue()[:2] == b'\x1f\x8b': # if gzip
            with gzip.GzipFile(fileobj=content) as f:
                df = pd.read_csv(f, low_memory=False)
        else: # if csv
            df = pd.read_csv(content, low_memory=False)

    except Exception as e:
        print(f"   Error reading {filename}: {e}")
        df = pd.DataFrame()
    return df

def make_valid_table(df, table_name=None):
    ## correct table columns name
    reserved_keys = ["TRANSACTION", "TABLE", "USER", "INDEX", "SELECT", "WHERE", "GROUP", "ORDER"] # List of SQL reserved keywords that cannot be used directly as column names. These will be automatically prefixed with '_' when saving tables to avoid SQL conflicts.
    if reserved_keys:
        df.columns = [f"_{c}" if c.upper() in reserved_keys else c for c in df.columns]
    ## correct table name
    table_name = re.sub(r'\W', '_', table_name) # Replace any non-alphanumeric characters with underscores
    if table_name and table_name[0].isdigit(): # If it starts with a digit, prepend an underscore
        table_name = '_' + table_name
    if table_name in keyword.kwlist: # If it's a Python keyword (e.g. 'class', 'for', 'try'), append an underscore
        table_name += '_'
    return df, table_name

def save_sql_table(df, conn, table_name=None):
    try:
        if df.empty:
            print("    Skipped empty dataset!")
            return
        conn.register("tmp_df", df)
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM tmp_df")
        conn.unregister("tmp_df")
    except Exception as e:
        print(f"    Error saving table {table_name}: {e}")