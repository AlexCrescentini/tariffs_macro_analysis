# %% PACKAGES
import os
import duckdb
from datetime import datetime
from pathlib import Path
from api_url_list import (a_figaro_data_yearly, b_other_data_yearly, c_sectoral_data_quarterly, d_gdp_data_quarterly, e_exchange_rate_data_yearly_quarterly)
from utils import process_data

# %% SETUP PATHS
SCRIPT_DIR = Path(__file__).parent if '__file__' in globals() else Path.cwd()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# %% Create time-stamped folder and DuckDB database
output_folder = DATA_DIR / "output" / "downloaded" / f"db_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
output_folder.mkdir(parents=True, exist_ok=False)
db_path = output_folder / "dataset_GTAB.duckdb"
conn = duckdb.connect(str(db_path))
print("=" * 30)
print(f"DuckDB database created at: {db_path}\n")

# %% Process data (downloading, cleaning, SQL saving)
process_data(output_folder, conn, a_figaro_data_yearly, "a_figaro_data_y")
process_data(output_folder, conn, b_other_data_yearly, "b_other_data_yearly")
process_data(output_folder, conn, c_sectoral_data_quarterly, "c_sectoral_data_quarterly")
process_data(output_folder, conn, d_gdp_data_quarterly, "d_gdp_data_quarterly")
process_data(output_folder, conn, e_exchange_rate_data_yearly_quarterly, "e_exchange_rate_data_yearly_quarterly")

# %% List all SQL tables
tables = conn.execute("SHOW TABLES").fetchall()
print(f"=== TABLES IN THE DATASET ({os.path.basename(db_path)}) ===")
for t in tables:
    print("-", t[0])

# %% Close connection to the DuckDB database
conn.close()
print(f"Connection to DuckDB database ({db_path}) closed!\n")
print("=" * 30)
