# %% PACKAGES
import duckdb
import pandas as pd
import numpy as np
from scipy.io import savemat
from scipy.interpolate import interp1d
from datetime import datetime
from pathlib import Path
import tarfile

# %% SETUP PATHS
SCRIPT_DIR = Path(__file__).parent if '__file__' in globals() else Path.cwd()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# %% OPEN CONNECTION TO DUCKDB DATABASE
db_folder = DATA_DIR / "output" / "downloaded"
db_path = DATA_DIR / "output" / "downloaded" / "db_2025-11-12_16-54-10" / "dataset_GTAB.duckdb"
if not db_path.exists():
    raise FileNotFoundError(f"Database non trovato: {db_path}")
conn = duckdb.connect(database=str(db_path), read_only=True)
print("=" * 30)
print(f"Connected to DuckDB database: {db_path}\n")

# %% DATASETS
exchange_rate_table_y = ['ert_bil_eur_a_1_0']
figaro_tables_y = ['naio_10_fcp_ii']
other_tables_y = ['OECD_SDD_NAD_DSD_NAMAIN10_DF_TABLE6_2_0',
    'OECD_SDD_NAD_DSD_NAMAIN10_DF_TABLE9A_2_0',
    'OECD_SDD_NAD_DSD_NAMAIN10_DF_TABLE7',
    'OECD_SDD_NAD_DSD_NAMAIN10_DF_TABLE3_1_0',
    'OECD_SDD_TPS_DSD_LFS_DF_IALFS_UNE_M_1_0',
    'OECD_SDD_NAD_DSD_NAMAIN10_DF_TABLE1_INCOME_2_0',
    'OECD_SDD_NAD_DSD_NASEC10_DF_TABLE14_1_1',
    'OECD_SDD_NAD_DSD_NASEC10_DF_TABLE10_1_1',
    'OECD_SDD_NAD_DSD_NASEC10_DF_TABLE13_BAL_1_1',
    'OECD_SDD_NAD_DSD_NASEC10_DF_TABLE11_1_1',
    'OECD_SDD_TPS_DSD_ALFS_DF_ALFS_EMP_EES',
    'nama_10_nfa_st_1_0',
    'OECD_ELS_SPD_DSD_SOCX_AGG_DF_PUB_PRV',
    'OECD_SDD_NAD_DSD_NASEC10_DF_TABLE12_REV_1_1',
    'OECD_SDD_NAD_DSD_NASEC10_DF_TABLE9B_1_1']
exchange_rate_table_q = ['ert_bil_eur_q_1_0']
sectoral_tables_q = ['OECD_SDD_NAD_DSD_NASEC20_DF_T720R_Q_1_1',
    'OECD_SDD_NAD_DSD_NASEC20_DF_T7PSD_Q_1_1',
    'OECD_SDD_NAD_DSD_NASEC20_DF_T710R_Q_1_1']
gdp_tables_q = ['OECD_SDD_NAD_DSD_NAMAIN1_DF_QNA_EXPENDITURE_NATIO_CURR_1_1']

# %% PRELIMINARIES
geos = [#OECD OECD-30 codes
    'AUT','AUS','BEL','CAN','CZE','DEU','DNK','EST','GRC','ESP','FIN',
    'FRA','HUN','IRL','ITA','JPN','KOR','LTU','LUX','LVA','MEX','NLD',
    'NOR','POL','PRT','SWE','SVN','SVK','GBR','USA']
geos_2 = [#Eurostat OECD-30 codes
    'AT','AU','BE','CA','CZ','DE','DK','EE','EL','ES','FI',
    'FR','HU','IE','IT','JP','KR','LT','LU','LV','MX','NL',
    'NO','PL','PT','SE','SI','SK','UK','US']
geos_sql = str(tuple(geos_2)) # Convert countries into SQL string
currencies = [#Currencies OECD-30 codes
    'EUR','AUD','EUR','CAD','CZK','EUR','DKK','EUR','EUR','EUR','EUR',
    'EUR','HUF','EUR','EUR','JPY','KRW','EUR','EUR','EUR','MXN','EUR',
    'NOK','PLN','EUR','SEK','EUR','EUR','GBP','USD']
calibration_years = list(range(2010, 2023)) # Calibration years (annual data)
years_num = np.array([datetime(year, 12, 31).toordinal() + 366 for year in calibration_years]).reshape(-1, 1).astype(float)
years_sql = str(tuple(calibration_years)) # SQL
calibration_years_q = list(range(2010, 2025)); # Calibration years of initial conditions (quarterly data)
calibration_quarters = [f"{year}-Q{q}" for year in calibration_years_q for q in range(1, 5)]
quarters_num = np.array([datetime(year, q*3, [31,30,30,31][q-1]).toordinal() + 366 for year in calibration_years_q for q in range(1, 5)]).reshape(-1, 1).astype(float)
quarters_sql = str(tuple(calibration_quarters)) # SQL
estim_years_q = list(range(2005, 2025)); # Estimation years of exogenous variables and AR1 coefficents (quarterly data)
calibration_quarters_est = [f"{year}-Q{q}" for year in estim_years_q for q in range(1, 5)]
quarters_est_num = np.array([datetime(year, q*3, [31,30,30,31][q-1]).toordinal() + 366 for year in estim_years_q for q in range(1, 5)]).reshape(-1, 1).astype(float)
quarters_est_sql = str(tuple(calibration_quarters_est)) # SQL
industries = [# 64-industries classification
    'A01','A02','A03','B','C10-12','C13-15','C16','C17','C18','C19','C20','C21','C22','C23',
    'C24','C25','C26','C27','C28','C29','C30','C31_32','C33','D35','E36','E37-39','F','G45',
    'G46','G47','H49','H50','H51','H52','H53','I','J58','J59_60','J61','J62_63','K64','K65',
    'K66','L','M69_70','M71','M72','M73','M74_75','N77','N78','N79','N80-82','O84','P85','Q86',
    'Q87_88','R90-92','R93','S94','S95','S96','T','U']
industries_sql = str(tuple(industries)) # SQL
industries_19 = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S'] # 19-industries classification
industries_19_sql = str(tuple(industries_19)) # SQL

# %% A) ANNUAL DATA EXTRACTION
# %% A.1) Exchange Rate Data
df = conn.execute(f"""
    SELECT TIME_PERIOD, currency, OBS_VALUE
    FROM {exchange_rate_table_y[0]}
    WHERE TIME_PERIOD IN {years_sql}
    AND statinfo = 'AVG'
    AND currency = 'USD'
    ORDER BY TIME_PERIOD
""").fetchdf()
eur_to_usd = df.set_index("TIME_PERIOD")["OBS_VALUE"].values   # how many EUR for 1 USD?
nom_exchange_rate = {}
for i, geo in enumerate(geos):
    curr = currencies[i]
    if curr != 'EUR':
        curr_sql = str((curr,))
        df = conn.execute(f"""
            SELECT TIME_PERIOD, currency, OBS_VALUE
            FROM {exchange_rate_table_y[0]}
            WHERE TIME_PERIOD IN {years_sql}
            AND statinfo = 'AVG'
            AND currency IN {curr_sql}
            ORDER BY TIME_PERIOD
        """).fetchdf()
        eur_to_lcu = df.set_index("TIME_PERIOD")["OBS_VALUE"].values # how many EUR for 1 LCU?
    else:
        eur_to_lcu = np.ones(len(eur_to_usd))  # how many EUR for 1 EUR?
    nom_exchange_rate[geo] = (eur_to_usd / eur_to_lcu).reshape(-1, 1)

# %% A.2) Eurostat Figaro IO Data
figaro = {}
figaro["interm_consumption"] = np.zeros((len(geos_2)+1, len(geos_2), len(industries), len(industries), len(calibration_years)))
figaro["gov_consumption"] = np.zeros((len(geos_2)+1, len(geos_2), len(industries), len(calibration_years)))
figaro["hh_consumption"] = np.zeros((len(geos_2)+1, len(geos_2), len(industries), len(calibration_years)))
figaro["fixed_cf"] = np.zeros((len(geos_2)+1, len(geos_2), len(industries), len(calibration_years)))
figaro["inv_chang"] = np.zeros((len(geos_2)+1, len(geos_2), len(industries), len(calibration_years)))
figaro["exports"] = np.zeros((len(geos_2), len(industries), len(calibration_years)))
figaro["imports"] = np.zeros((len(geos_2), len(industries), len(calibration_years)))
figaro["operating_surplus"] = np.zeros((len(geos_2), len(industries), len(calibration_years)))
figaro["compensation_employees"] = np.zeros((len(geos_2), len(industries), len(calibration_years)))
figaro["taxes_production"] = np.zeros((len(geos_2), len(industries), len(calibration_years)))
figaro["taxes_products"] = np.zeros((len(geos_2), len(industries), len(calibration_years)))
figaro["taxes_products_gov"] = np.zeros((len(geos_2), len(calibration_years)))
figaro["taxes_products_hh"] = np.zeros((len(geos_2), len(calibration_years)))
figaro["taxes_products_fixed_cf"] = np.zeros((len(geos_2), len(calibration_years)))
figaro["taxes_products_inv_chang"] = np.zeros((len(geos_2), len(calibration_years)))
figaro["taxes_products_export"] = np.zeros((len(geos_2), len(calibration_years)))
# Create mappings from codes to indices
iG = {c:i for i,c in enumerate(geos_2)}                # geos mapping
iG_ROW = {o:i for i,o in enumerate(geos_2 + ["ROW"])}  # geos + ROW mapping
iI = {a:i for i,a in enumerate(industries)}            # industries mapping
iY = {t:i for i,t in enumerate(calibration_years)}     # years mapping
df = conn.execute(f"""
SELECT TIME_PERIOD, CASE WHEN c_orig IN {geos_sql} THEN c_orig WHEN c_orig = 'DOM' THEN 'DOM' ELSE 'ROW' END AS c_orig_with_ROW, ind_ava, c_dest, ind_use, SUM(OBS_VALUE) AS sum_value
FROM {figaro_tables_y[0]}
WHERE TIME_PERIOD IN {years_sql}
    AND c_dest IN {geos_sql}
    AND ind_use IN {industries_sql}
    AND c_orig <> 'DOM'
    AND ind_ava IN {industries_sql}
GROUP BY
    TIME_PERIOD, c_orig_with_ROW, c_dest, ind_ava, ind_use
ORDER BY
    TIME_PERIOD, c_orig_with_ROW, c_dest, ind_ava, ind_use
""").fetchdf() #pandas Dataframe object
figaro["interm_consumption"][df["c_orig_with_ROW"].map(iG_ROW).to_numpy(), df["c_dest"].map(iG).to_numpy(), df["ind_ava"].map(iI).to_numpy(), df["ind_use"].map(iI).to_numpy(), df["TIME_PERIOD"].map(iY).to_numpy()] = df["sum_value"].to_numpy()
df = conn.execute(f"""
SELECT TIME_PERIOD, CASE WHEN c_orig IN {geos_sql} THEN c_orig WHEN c_orig = 'DOM' THEN 'DOM' ELSE 'ROW' END AS c_orig_with_ROW, ind_ava, c_dest, SUM(OBS_VALUE) AS sum_value
FROM {figaro_tables_y[0]}
WHERE TIME_PERIOD IN {years_sql}
    AND c_dest IN {geos_sql}
    AND ind_use IN ('P3_S13')
    AND c_orig <> 'DOM'
    AND ind_ava IN {industries_sql}
GROUP BY
    TIME_PERIOD, c_orig_with_ROW, c_dest, ind_ava, ind_use
ORDER BY
    TIME_PERIOD, c_orig_with_ROW, c_dest, ind_ava, ind_use
""").fetchdf()
figaro["gov_consumption"][df["c_orig_with_ROW"].map(iG_ROW).to_numpy(), df["c_dest"].map(iG).to_numpy(), df["ind_ava"].map(iI).to_numpy(), df["TIME_PERIOD"].map(iY).to_numpy()] = df["sum_value"].to_numpy()
df = conn.execute(f"""
SELECT TIME_PERIOD, CASE WHEN c_orig IN {geos_sql} THEN c_orig WHEN c_orig = 'DOM' THEN 'DOM' ELSE 'ROW' END AS c_orig_with_ROW, ind_ava, c_dest, SUM(OBS_VALUE) AS sum_value
FROM {figaro_tables_y[0]}
WHERE TIME_PERIOD IN {years_sql}
    AND c_dest IN {geos_sql}
    AND ind_use IN ('P3_S14','P3_S15')
    AND c_orig <> 'DOM'
    AND ind_ava IN {industries_sql}
GROUP BY
    TIME_PERIOD, c_orig_with_ROW, c_dest, ind_ava
ORDER BY
    TIME_PERIOD, c_orig_with_ROW, c_dest, ind_ava
""").fetchdf()
figaro["hh_consumption"][df["c_orig_with_ROW"].map(iG_ROW).to_numpy(), df["c_dest"].map(iG).to_numpy(), df["ind_ava"].map(iI).to_numpy(), df["TIME_PERIOD"].map(iY).to_numpy()] = df["sum_value"].to_numpy()
df = conn.execute(f"""
SELECT TIME_PERIOD, CASE WHEN c_orig IN {geos_sql} THEN c_orig WHEN c_orig = 'DOM' THEN 'DOM' ELSE 'ROW' END AS c_orig_with_ROW, ind_ava, c_dest, SUM(OBS_VALUE) AS sum_value
FROM {figaro_tables_y[0]}
WHERE TIME_PERIOD IN {years_sql}
    AND c_dest IN {geos_sql}
    AND ind_use IN ('P51G')
    AND c_orig <> 'DOM'
    AND ind_ava IN {industries_sql}
GROUP BY
    TIME_PERIOD, c_orig_with_ROW, c_dest, ind_ava
ORDER BY
    TIME_PERIOD, c_orig_with_ROW, c_dest, ind_ava
""").fetchdf()
figaro["fixed_cf"][df["c_orig_with_ROW"].map(iG_ROW).to_numpy(), df["c_dest"].map(iG).to_numpy(), df["ind_ava"].map(iI).to_numpy(), df["TIME_PERIOD"].map(iY).to_numpy()] = df["sum_value"].to_numpy() 
df = conn.execute(f"""
SELECT TIME_PERIOD, CASE WHEN c_orig IN {geos_sql} THEN c_orig WHEN c_orig = 'DOM' THEN 'DOM' ELSE 'ROW' END AS c_orig_with_ROW, ind_ava, c_dest, SUM(OBS_VALUE) AS sum_value
FROM {figaro_tables_y[0]}
WHERE TIME_PERIOD IN {years_sql}
    AND c_dest IN {geos_sql}
    AND ind_use IN ('P5M')
    AND c_orig <> 'DOM'
    AND ind_ava IN {industries_sql}
GROUP BY
    TIME_PERIOD, c_orig_with_ROW, c_dest, ind_ava
ORDER BY
    TIME_PERIOD, c_orig_with_ROW, c_dest, ind_ava
""").fetchdf()
figaro["inv_chang"][df["c_orig_with_ROW"].map(iG_ROW).to_numpy(), df["c_dest"].map(iG).to_numpy(), df["ind_ava"].map(iI).to_numpy(), df["TIME_PERIOD"].map(iY).to_numpy()] = df["sum_value"].to_numpy()
df = conn.execute(f"""
SELECT TIME_PERIOD, c_orig, ind_ava, SUM(OBS_VALUE) AS sum_value
FROM {figaro_tables_y[0]}
WHERE TIME_PERIOD IN {years_sql}
AND c_orig IN {geos_sql}
AND ind_ava IN {industries_sql}
AND c_dest NOT IN {geos_sql}
GROUP BY TIME_PERIOD, c_orig, ind_ava
ORDER BY TIME_PERIOD, c_orig, ind_ava
""").fetchdf()
figaro["exports"][df["c_orig"].map(iG).to_numpy(), df["ind_ava"].map(iI).to_numpy(), df["TIME_PERIOD"].map(iY).to_numpy()] = df["sum_value"].to_numpy()
df = conn.execute(f"""
SELECT TIME_PERIOD, c_dest, ind_ava, SUM(OBS_VALUE) AS sum_value
FROM {figaro_tables_y[0]}
WHERE TIME_PERIOD IN {years_sql}
AND c_dest IN {geos_sql}
AND c_orig NOT IN {geos_sql} AND c_orig <> 'DOM'
AND ind_ava IN {industries_sql}
GROUP BY TIME_PERIOD, c_dest, ind_ava
ORDER BY TIME_PERIOD, c_dest, ind_ava
""").fetchdf()
figaro["imports"][df["c_dest"].map(iG).to_numpy(), df["ind_ava"].map(iI).to_numpy(), df["TIME_PERIOD"].map(iY).to_numpy()] = df["sum_value"].to_numpy()
df = conn.execute(f"""
SELECT TIME_PERIOD, c_dest, ind_use, SUM(OBS_VALUE) AS sum_value
FROM {figaro_tables_y[0]}
WHERE TIME_PERIOD IN {years_sql}
    AND c_dest IN {geos_sql}
    AND ind_use IN {industries_sql}
    AND c_orig = 'DOM'
    AND ind_ava IN ('B2A3G')
GROUP BY
    TIME_PERIOD, c_dest, ind_use
ORDER BY
    TIME_PERIOD, c_dest, ind_use
""").fetchdf()
figaro["operating_surplus"][df["c_dest"].map(iG).to_numpy(), df["ind_use"].map(iI).to_numpy(), df["TIME_PERIOD"].map(iY).to_numpy()] = df["sum_value"].to_numpy()
df = conn.execute(f"""
SELECT TIME_PERIOD, c_dest, ind_use, SUM(OBS_VALUE) AS sum_value
FROM {figaro_tables_y[0]}
WHERE TIME_PERIOD IN {years_sql}
    AND c_dest IN {geos_sql}
    AND ind_use IN {industries_sql}
    AND c_orig = 'DOM'
    AND ind_ava IN ('D1')
GROUP BY
    TIME_PERIOD, c_dest, ind_use
ORDER BY
    TIME_PERIOD, c_dest, ind_use
""").fetchdf()
figaro["compensation_employees"][df["c_dest"].map(iG).to_numpy(), df["ind_use"].map(iI).to_numpy(), df["TIME_PERIOD"].map(iY).to_numpy()] = df["sum_value"].to_numpy()
df = conn.execute(f"""
SELECT TIME_PERIOD, c_dest, ind_use, SUM(OBS_VALUE) AS sum_value
FROM {figaro_tables_y[0]}
WHERE TIME_PERIOD IN {years_sql}
    AND c_dest IN {geos_sql}
    AND ind_use IN {industries_sql}
    AND c_orig = 'DOM'
    AND ind_ava IN ('D29X39')
GROUP BY
    TIME_PERIOD, c_dest, ind_use
ORDER BY
    TIME_PERIOD, c_dest, ind_use
""").fetchdf()
figaro["taxes_production"][df["c_dest"].map(iG).to_numpy(), df["ind_use"].map(iI).to_numpy(), df["TIME_PERIOD"].map(iY).to_numpy()] = df["sum_value"].to_numpy()
df = conn.execute(f"""
SELECT TIME_PERIOD, c_dest, ind_use, SUM(OBS_VALUE) AS sum_value
FROM {figaro_tables_y[0]}
WHERE TIME_PERIOD IN {years_sql}
    AND c_dest IN {geos_sql}
    AND ind_use IN {industries_sql}
    AND c_orig = 'DOM'
    AND ind_ava IN ('D21X31')
GROUP BY
    TIME_PERIOD, c_dest, ind_use
ORDER BY
    TIME_PERIOD, c_dest, ind_use
""").fetchdf()
figaro["taxes_products"][df["c_dest"].map(iG).to_numpy(), df["ind_use"].map(iI).to_numpy(), df["TIME_PERIOD"].map(iY).to_numpy()] = df["sum_value"].to_numpy()
df = conn.execute(f"""
SELECT TIME_PERIOD, c_dest, SUM(OBS_VALUE) AS sum_value
FROM {figaro_tables_y[0]}
WHERE TIME_PERIOD IN {years_sql}
    AND c_dest IN {geos_sql}
    AND ind_use IN ('P3_S13')
    AND c_orig = 'DOM'
    AND ind_ava IN ('D21X31')
GROUP BY
    TIME_PERIOD, c_dest
ORDER BY
    TIME_PERIOD, c_dest
""").fetchdf()
figaro["taxes_products_gov"][df["c_dest"].map(iG).to_numpy(), df["TIME_PERIOD"].map(iY).to_numpy()] = df["sum_value"].to_numpy()
df = conn.execute(f"""
SELECT TIME_PERIOD, c_dest, SUM(OBS_VALUE) AS sum_value
FROM {figaro_tables_y[0]}
WHERE TIME_PERIOD IN {years_sql}
    AND c_dest IN {geos_sql}
    AND ind_use IN ('P3_S14','P3_S15')
    AND c_orig = 'DOM'
    AND ind_ava IN ('D21X31')
GROUP BY
    TIME_PERIOD, c_dest
ORDER BY
    TIME_PERIOD, c_dest
""").fetchdf()
figaro["taxes_products_hh"][df["c_dest"].map(iG).to_numpy(), df["TIME_PERIOD"].map(iY).to_numpy()] = df["sum_value"].to_numpy()
df = conn.execute(f"""
SELECT TIME_PERIOD, c_dest, SUM(OBS_VALUE) AS sum_value
FROM {figaro_tables_y[0]}
WHERE TIME_PERIOD IN {years_sql}
    AND c_dest IN {geos_sql}
    AND ind_use IN ('P51G')
    AND c_orig = 'DOM'
    AND ind_ava IN ('D21X31')
GROUP BY
    TIME_PERIOD, c_dest
ORDER BY
    TIME_PERIOD, c_dest
""").fetchdf()
figaro["taxes_products_fixed_cf"][df["c_dest"].map(iG).to_numpy(), df["TIME_PERIOD"].map(iY).to_numpy()] = df["sum_value"].to_numpy()
df = conn.execute(f"""
SELECT TIME_PERIOD, c_dest, SUM(OBS_VALUE) AS sum_value
FROM {figaro_tables_y[0]}
WHERE TIME_PERIOD IN {years_sql}
    AND c_dest IN {geos_sql}
    AND ind_use IN ('P5M')
    AND c_orig = 'DOM'
    AND ind_ava IN ('D21X31')
GROUP BY
    TIME_PERIOD, c_dest
ORDER BY
    TIME_PERIOD, c_dest
""").fetchdf()
figaro["taxes_products_inv_chang"][df["c_dest"].map(iG).to_numpy(), df["TIME_PERIOD"].map(iY).to_numpy()] = df["sum_value"].to_numpy()
figaro["taxes_products_export"] = np.zeros((len(iG), len(iY)), dtype=float)

# %% A.3) Other Data
other = {}
iI19 = {a: i for i, a in enumerate(industries_19)} # Mappings for by-industry variables
iY = {t: i for i, t in enumerate(calibration_years)} # Mappings for by-industry variables
for i, geo in enumerate(geos):
    if geo not in other:
        other[geo] = {}

    ## BY-INDUSTRY VARIABLES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # CONSUMPTION OF FIXED CAPITAL
    df = conn.execute(f"""
        SELECT TIME_PERIOD, ACTIVITY, OBS_VALUE
        FROM {other_tables_y[0]}
        WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S1'
            AND COUNTERPART_SECTOR = 'S1'
            AND _TRANSACTION = 'P51C'
            AND INSTR_ASSET = '_Z'
            AND ACTIVITY IN {industries_19_sql}
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
        ORDER BY TIME_PERIOD, ACTIVITY
    """).fetchdf()
    capital_consumption = np.zeros((len(industries_19), len(calibration_years)))
    capital_consumption[df["ACTIVITY"].map(iI19).to_numpy(), df["TIME_PERIOD"].map(iY).to_numpy()] = df["OBS_VALUE"].to_numpy()
    other[geo]['capital_consumption'] = capital_consumption
    
    # FIXED ASSETS (TOTAL FIXED ASSETS NET)
    df = conn.execute(f"""
        SELECT TIME_PERIOD, ACTIVITY, OBS_VALUE
        FROM {other_tables_y[1]}
        WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S1'
            AND COUNTERPART_SECTOR = 'S1'
            AND _TRANSACTION = 'LE'
            AND INSTR_ASSET = 'N11N'
            AND ACTIVITY IN {industries_19_sql}
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
        ORDER BY TIME_PERIOD, ACTIVITY
    """).fetchdf()
    fixed_assets = np.zeros((len(industries_19), len(calibration_years)))
    fixed_assets[df["ACTIVITY"].map(iI19).to_numpy(), df["TIME_PERIOD"].map(iY).to_numpy()] = df["OBS_VALUE"].to_numpy()
    other[geo]['fixed_assets'] = fixed_assets

    # DWELLINGS (NET)
    df = conn.execute(f"""
        SELECT TIME_PERIOD, ACTIVITY, OBS_VALUE
        FROM {other_tables_y[1]}
        WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S1'
            AND COUNTERPART_SECTOR = 'S1'
            AND _TRANSACTION = 'LE'
            AND INSTR_ASSET = 'N111N'
            AND ACTIVITY IN {industries_19_sql}
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
        ORDER BY TIME_PERIOD, ACTIVITY
    """).fetchdf()
    dwellings = np.zeros((len(industries_19), len(calibration_years)))
    dwellings[df["ACTIVITY"].map(iI19).to_numpy(), df["TIME_PERIOD"].map(iY).to_numpy()] = df["OBS_VALUE"].to_numpy()
    other[geo]['dwellings'] = dwellings
    
    # EMPLOYEES
    df = conn.execute(f"""
        SELECT TIME_PERIOD, ACTIVITY, OBS_VALUE
        FROM {other_tables_y[2]}
        WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S1'
            AND COUNTERPART_SECTOR = 'S1'
            AND _TRANSACTION = 'SAL'
            AND INSTR_ASSET = '_Z'
            AND ACTIVITY IN {industries_19_sql}
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'PS'
            AND PRICE_BASE = '_Z'
        ORDER BY TIME_PERIOD, ACTIVITY
    """).fetchdf()
    employees = np.full((len(industries_19), len(calibration_years)), np.nan) 
    employees[df["ACTIVITY"].map(iI19).to_numpy(), df["TIME_PERIOD"].map(iY).to_numpy()] = 1000 * df["OBS_VALUE"].to_numpy()
    other[geo]['employees'] = employees
    
    # FIRMS: we assume average firm size of 10 employees
    other[geo]['firms'] = np.round(employees / 10)

    ## NATIONAL VARIABLES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # POPULATION
    df = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {other_tables_y[3]}
        WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S1'
            AND COUNTERPART_SECTOR = 'S1'
            AND _TRANSACTION = 'POP'
            AND INSTR_ASSET = '_Z'
            AND ACTIVITY = '_Z'
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'PS'
            AND PRICE_BASE = '_Z'
        ORDER BY TIME_PERIOD
    """).fetchdf()
    other[geo]['population'] = 1000 * df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1, 1)
    
    # UNEMPLOYMENT RATE
    df = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {other_tables_y[4]}
        WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A'
            AND ADJUSTMENT = 'Y'
            AND REF_AREA = '{geo}'
            AND MEASURE = 'UNE_LF_M'
            AND UNIT_MEASURE = 'PT_LF_SUB'
            AND TRANSFORMATION = '_Z'
            AND SEX = '_T'
            AND AGE = 'Y_GE15'
        ORDER BY TIME_PERIOD
    """).fetchdf()
    other[geo]['unemployment_rate'] = 0.01 * df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1, 1)
    
    # WAGES
    df = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {other_tables_y[5]}
        WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S1'
            AND COUNTERPART_SECTOR = 'S1'
            AND _TRANSACTION = 'D11'
            AND INSTR_ASSET = '_Z'
            AND ACTIVITY = '_T'
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
        ORDER BY TIME_PERIOD
    """).fetchdf()
    other[geo]['wages'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1, 1)
    
    ## SECTORAL VARIABLES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # MIXED INCOME (operating surplus and mixed income net received by Households and NPISH)
    df1 = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {other_tables_y[6]}
        WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S1M'
            AND COUNTERPART_SECTOR = 'S1'
            AND ACCOUNTING_ENTRY = 'C'
            AND _TRANSACTION = 'B2A3G'
            AND INSTR_ASSET = '_Z'
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
        ORDER BY TIME_PERIOD
    """).fetchdf()
    df2 = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {other_tables_y[6]}
        WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S1M'
            AND COUNTERPART_SECTOR = 'S1'
            AND ACCOUNTING_ENTRY = 'C'
            AND _TRANSACTION = 'P51CB'
            AND INSTR_ASSET = '_Z'
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
        ORDER BY TIME_PERIOD
    """).fetchdf()
    operating_surplus_and_mixed_income_gross = df1.set_index("TIME_PERIOD")["OBS_VALUE"].values
    consumption_fixed_capital = df2.set_index("TIME_PERIOD")["OBS_VALUE"].values
    if len(operating_surplus_and_mixed_income_gross) == len(consumption_fixed_capital):
        mixed_income = operating_surplus_and_mixed_income_gross - consumption_fixed_capital
    else:
        mixed_income = np.array([])
    other[geo]['mixed_income'] = mixed_income.reshape(-1, 1)

    # PROPERTY INCOME (income received from property by Households and NPISH)
    df = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {other_tables_y[6]}
        WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S1M'
            AND COUNTERPART_SECTOR = 'S1'
            AND ACCOUNTING_ENTRY = 'C'
            AND _TRANSACTION = 'D4'
            AND INSTR_ASSET = '_Z'
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
        ORDER BY TIME_PERIOD
    """).fetchdf()
    other[geo]['property_income'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1, 1)
    
    # FIRM INTEREST (interest paid by Financial and Non-Financial corporations)
    df1 = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {other_tables_y[6]}
        WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S11'
            AND COUNTERPART_SECTOR = 'S1'
            AND ACCOUNTING_ENTRY = 'D'
            AND _TRANSACTION = 'D41'
            AND INSTR_ASSET = '_Z'
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
        ORDER BY TIME_PERIOD
    """).fetchdf()
    df2 = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {other_tables_y[6]}
        WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S12'
            AND COUNTERPART_SECTOR = 'S1'
            AND ACCOUNTING_ENTRY = 'D'
            AND _TRANSACTION = 'D41'
            AND INSTR_ASSET = '_Z'
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
        ORDER BY TIME_PERIOD
    """).fetchdf()
    firm_interest_non_financial = df1.set_index("TIME_PERIOD")["OBS_VALUE"].values
    firm_interest_financial = df2.set_index("TIME_PERIOD")["OBS_VALUE"].values
    if len(firm_interest_non_financial) == len(firm_interest_financial):
        firm_interest = firm_interest_non_financial + firm_interest_financial
    else:
        firm_interest = np.array([])
    other[geo]['firm_interest'] = firm_interest.reshape(-1, 1)
    
    # CORPORATE TAX (taxes on income paid by Non-Financial and Financial Corporations)
    df1 = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {other_tables_y[6]}
        WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S11'
            AND COUNTERPART_SECTOR = 'S1'
            AND ACCOUNTING_ENTRY = 'D'
            AND _TRANSACTION = 'D51'
            AND INSTR_ASSET = '_Z'
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
        ORDER BY TIME_PERIOD
    """).fetchdf()
    df2 = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {other_tables_y[6]}
        WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S12'
            AND COUNTERPART_SECTOR = 'S1'
            AND ACCOUNTING_ENTRY = 'D'
            AND _TRANSACTION = 'D51'
            AND INSTR_ASSET = '_Z'
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
        ORDER BY TIME_PERIOD
    """).fetchdf()
    corporate_tax_non_financial = df1.set_index("TIME_PERIOD")["OBS_VALUE"].values
    corporate_tax_financial = df2.set_index("TIME_PERIOD")["OBS_VALUE"].values
    if len(corporate_tax_non_financial) == len(corporate_tax_financial):
        corporate_tax = corporate_tax_non_financial + corporate_tax_financial
    else:
        corporate_tax = np.array([])
    other[geo]['corporate_tax'] = corporate_tax.reshape(-1, 1)
    
    # INTEREST GOVERNMENT DEBT (interest paid by the General Government on debt)
    df = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {other_tables_y[6]}
        WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S13'
            AND COUNTERPART_SECTOR = 'S1'
            AND ACCOUNTING_ENTRY = 'D'
            AND _TRANSACTION = 'D41'
            AND INSTR_ASSET = '_Z'
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
        ORDER BY TIME_PERIOD
    """).fetchdf()
    other[geo]['interest_government_debt'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1, 1)

    # SOCIAL BENEFITS (social benefits other than social transfers in kind paid by the General Government)
    df = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {other_tables_y[6]}
        WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S13'
            AND COUNTERPART_SECTOR = 'S1'
            AND ACCOUNTING_ENTRY = 'D'
            AND _TRANSACTION = 'D62'
            AND INSTR_ASSET = '_Z'
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
        ORDER BY TIME_PERIOD
    """).fetchdf()
    other[geo]['social_benefits'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1, 1)

    # SOCIAL CONTRIBUTIONS (net social contributions received by the General Government)
    df = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {other_tables_y[6]}
        WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S13'
            AND COUNTERPART_SECTOR = 'S1'
            AND ACCOUNTING_ENTRY = 'C'
            AND _TRANSACTION = 'D61'
            AND INSTR_ASSET = '_Z'
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
        ORDER BY TIME_PERIOD
    """).fetchdf()
    other[geo]['social_contributions'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1, 1)
    
    # INCOME TAX (current taxes on income, wealth, etc. received by the General Government)
    df = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {other_tables_y[7]}
        WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S13'
            AND COUNTERPART_SECTOR = '_Z'
            AND ACCOUNTING_ENTRY = 'C'
            AND _TRANSACTION = 'D5'
            AND INSTR_ASSET = '_Z'
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
        ORDER BY TIME_PERIOD
    """).fetchdf()
    other[geo]['income_tax'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1, 1)
    
    # CAPITAL TAXES (capital taxes received by the General Government)
    df = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {other_tables_y[7]}
        WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S13'
            AND COUNTERPART_SECTOR = '_Z'
            AND ACCOUNTING_ENTRY = 'C'
            AND _TRANSACTION = 'D91'
            AND INSTR_ASSET = '_Z'
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
        ORDER BY TIME_PERIOD
    """).fetchdf()
    other[geo]['capital_taxes'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1, 1)

    # GOVERNMENT DEFICIT (net lending(+) / net borrowing (-) of the General Government)
    df = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {other_tables_y[8]}
        WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S13'
            AND COUNTERPART_SECTOR = 'S1'
            AND ACCOUNTING_ENTRY = 'B'
            AND _TRANSACTION = 'B9'
            AND INSTR_ASSET = '_Z'
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
        ORDER BY TIME_PERIOD
    """).fetchdf()
    other[geo]['government_deficit'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1, 1)

    # UNEMPLOYMENT BENEFITS (General Government expenditure for unemployment benefits)
    df = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {other_tables_y[9]}
        WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S13'
            AND COUNTERPART_SECTOR = '_Z'
            AND ACCOUNTING_ENTRY = 'D'
            AND _TRANSACTION = 'OTE'
            AND INSTR_ASSET = '_Z'
            AND EXPENDITURE = 'GF1005'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
        ORDER BY TIME_PERIOD
    """).fetchdf()
    other[geo]['unemployment_benefits'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1, 1)
    
    # PENSION BENEFITS (General Government expenditure for pension benefits)
    df = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {other_tables_y[9]}
        WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S13'
            AND COUNTERPART_SECTOR = '_Z'
            AND ACCOUNTING_ENTRY = 'D'
            AND _TRANSACTION = 'OTE'
            AND INSTR_ASSET = '_Z'
            AND EXPENDITURE = 'GF1002'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
        ORDER BY TIME_PERIOD
    """).fetchdf()
    other[geo]['pension_benefits'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1, 1)

    ## ADJUSTMENTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Employees
    df_employees = pd.DataFrame(other[geo]['employees'])
    df_filled = df_employees.interpolate(axis=1, method='nearest').ffill(axis=1).bfill(axis=1)
    other[geo]['employees'] = df_filled.values
    if np.isnan(other[geo]['employees']).any():
        df = conn.execute(f"""
            SELECT TIME_PERIOD, ACTIVITY, OBS_VALUE 
            FROM {other_tables_y[10]}
            WHERE TIME_PERIOD IN {years_sql}
                AND FREQ = 'A' 
                AND REF_AREA = '{geo}' 
                AND ACTIVITY IN {industries_19_sql}
                AND UNIT_MEASURE = 'PS' 
                AND SEX = '_T' 
            ORDER BY TIME_PERIOD, ACTIVITY
        """).fetchdf()
        employees_temp = np.zeros((len(industries_19), len(calibration_years)))
        employees_temp[df["ACTIVITY"].map(iI19).to_numpy(), df["TIME_PERIOD"].map(iY).to_numpy()] = 1000 * df["OBS_VALUE"].to_numpy()
        df_alt = pd.DataFrame(employees_temp).replace(0, np.nan)
        df_alt_filled = df_alt.interpolate(axis=1, method='nearest').ffill(axis=1).bfill(axis=1)
        other[geo]['employees'] = df_alt_filled.values
    if np.isnan(other[geo]['employees']).any():
        other[geo]['employees'] = np.nan_to_num(other[geo]['employees'], nan=0)
    other[geo]['employees'] = np.round(other[geo]['employees'])

    # Firms: we assume average firm size of 10 employees
    other[geo]['firms'] = np.round(other[geo]['employees'] / 10)

    # Dwellings (net)
    if 'dwellings' in other[geo] and np.isnan(other[geo]['dwellings']).any():
        other[geo]['dwellings'][np.isnan(other[geo]['dwellings'])] = 0

    # Country-specific adjustments
    if geo == 'AUS':  # Australia - industry E is included in D, split based on employees
        numerator = other[geo]['employees'][4, :]
        denominator = other[geo]['employees'][3, :] + other[geo]['employees'][4, :]
        ratio_E_ED_employees = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
        other[geo]['capital_consumption'][4, :] = other[geo]['capital_consumption'][3, :] * ratio_E_ED_employees
        other[geo]['capital_consumption'][3, :] *= (1 - ratio_E_ED_employees)
        other[geo]['fixed_assets'][4, :] = other[geo]['fixed_assets'][3, :] * ratio_E_ED_employees
        other[geo]['fixed_assets'][3, :] *= (1 - ratio_E_ED_employees)
    elif geo == 'CAN': 
        # FIXED ASSETS: Canada - industry E is split into industries D and N, take 50% each
        other[geo]['fixed_assets'][4, :] = 0.5 * other[geo]['fixed_assets'][3, :] + 0.5 * other[geo]['fixed_assets'][13, :]
        other[geo]['fixed_assets'][3, :] *= 0.5
        other[geo]['fixed_assets'][13, :] *= 0.5
        # CONSUMPTION OF FIXED CAPITAL: Canada - from Canadian data
        tar_path = db_folder / "CAN" / "36100096.csv.tar.gz"
        with tarfile.open(tar_path, 'r:gz') as tar:
            csv_file = tar.extractfile(tar.getmembers()[0])
            capital_consumption_df = pd.read_csv(csv_file, low_memory=False)

        filtered_df = capital_consumption_df[
            (capital_consumption_df['REF_DATE'].astype(int) >= calibration_years[0]) &
            (capital_consumption_df['REF_DATE'].astype(int) <= calibration_years[-1]) &
            (capital_consumption_df['GEO'] == 'Canada') &
            (capital_consumption_df['Prices'] == 'Current prices') &
            (capital_consumption_df['Flows and stocks'] == 'Linear depreciation') &
            (capital_consumption_df['Assets'] == 'Total non-residential')
        ].copy()
        columns_to_remove = ['GEO','DGUID','UOM_ID','SCALAR_ID','VECTOR','COORDINATE',
            'Prices','Flows and stocks','Assets','STATUS','SYMBOL','TERMINATED',
            'DECIMALS','UOM','SCALAR_FACTOR'
        ]
        filtered_df.drop(columns=columns_to_remove, inplace=True)
        industry_mapping = {
            'Agriculture, forestry, fishing and hunting': 'A',
            'Mining, quarrying and oil and gas extraction': 'B',
            'Manufacturing': 'C',
            'Utilities': 'D',
            'Construction': 'F',
            'Wholesale trade': 'G',
            'Retail trade': 'G',
            'Transportation and warehousing': 'H',
            'Information and cultural industries': 'J',
            'Finance and insurance': 'K',
            'Real estate and rental and leasing': 'L',
            'Holding companies': 'K',
            'Professional, scientific and technical services': 'M',
            'Administrative and support, waste management and remediation services': 'N and E',
            'Educational services': 'P',
            'Health care and social assistance': 'Q',
            'Hospitals': 'Q',
            'Nursing and residential care facilities': 'Q',
            'Arts, entertainment and recreation': 'R',
            'Accommodation and food services': 'I',
            'Other services (except public administration)': 'S',
            'Non-profit institutions serving households': 'S',
            'Defence services': 'O',
            'Other federal government services': 'O',
            'Other provincial and territorial government services': 'O',
            'Other municipal government services': 'O',
            'Other Indigenous government services': 'O'
        }
        filtered_df['Code'] = filtered_df['Industry'].map(industry_mapping)
        merged_df = filtered_df[filtered_df['Code'].notna()]
        pivot_df = merged_df.pivot_table(index='Code',
            columns='REF_DATE',
            values='VALUE',
            aggfunc='sum'
        )
        pivot_df = pivot_df.sort_index()
        year_cols = [int(y) for y in calibration_years]
        output_cols = [col for col in year_cols if col in pivot_df.columns]
        grouped_array = pivot_df[output_cols].to_numpy()
        ratio_E_EN_employees = other[geo]['employees'][4, :] / (other[geo]['employees'][4, :] + other[geo]['employees'][13, :])
        other[geo]['capital_consumption'][0:4, :] = grouped_array[0:4, :]
        other[geo]['capital_consumption'][4, :] = grouped_array[12, :] * ratio_E_EN_employees
        other[geo]['capital_consumption'][5:13, :] = grouped_array[4:12, :]
        other[geo]['capital_consumption'][13, :] = grouped_array[12, :] * (1 - ratio_E_EN_employees)
        other[geo]['capital_consumption'][14:, :] = grouped_array[13:, :]

    elif geo == 'POL':  # Poland - use Eurostat data source
        df = conn.execute(f"""
            SELECT TIME_PERIOD, nace_r2, OBS_VALUE 
            FROM {other_tables_y[11]} 
            WHERE TIME_PERIOD IN {years_sql}
                AND freq = 'A'
                AND geo = 'PL'
                AND asset10 = 'N11N'
                AND nace_r2 IN {industries_19_sql}
                AND unit = 'CRC_MNAC'
            ORDER BY TIME_PERIOD, nace_r2
        """).fetchdf()
        fixed_assets = np.zeros((len(industries_19), len(calibration_years)))
        fixed_assets[df["nace_r2"].map(iI19).to_numpy(), df["TIME_PERIOD"].map(iY).to_numpy()] = df["OBS_VALUE"].to_numpy()
        other[geo]['fixed_assets'] = fixed_assets

    # elif geo == 'JPN':  # Japan - set specific industries to zero
    #     industries_to_zero_indices = [4, 13, 18]
    #     other[geo]['capital_consumption'][industries_to_zero_indices, :] = 0
    #     other[geo]['fixed_assets'][industries_to_zero_indices, :] = 0

    elif geo == 'MEX':
        # FIXED ASSETS: Use USA shares
        df = conn.execute(f"""
            SELECT TIME_PERIOD, ACTIVITY, OBS_VALUE 
            FROM {other_tables_y[1]}
            WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A' 
            AND REF_AREA = 'USA' 
            AND SECTOR = 'S1' 
            AND COUNTERPART_SECTOR = 'S1' 
            AND _TRANSACTION = 'LE' 
            AND INSTR_ASSET = 'N11N' 
            AND ACTIVITY IN {industries_19_sql}
            AND EXPENDITURE = '_Z' 
            AND UNIT_MEASURE = 'XDC' 
            AND PRICE_BASE = 'V' 
            ORDER BY TIME_PERIOD, ACTIVITY
        """).fetchdf()
        fixed_assets_USA = np.zeros((len(industries_19), len(calibration_years)))
        fixed_assets_USA[df["ACTIVITY"].map(iI19).to_numpy(), df["TIME_PERIOD"].map(iY).to_numpy()] = df["OBS_VALUE"].to_numpy()
        shares = fixed_assets_USA / fixed_assets_USA.sum(axis=0)
        df = conn.execute(f"""
            SELECT TIME_PERIOD, OBS_VALUE 
            FROM {other_tables_y[14]}
            WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A' 
            AND REF_AREA = '{geo}' 
            AND SECTOR = 'S1' 
            AND COUNTERPART_SECTOR = 'S1' 
            AND _TRANSACTION = '_Z' 
            AND INSTR_ASSET = 'N11N' 
            AND EXPENDITURE = '_Z' 
            AND UNIT_MEASURE = 'XDC' 
            AND PRICE_BASE = 'V' 
            ORDER BY TIME_PERIOD
        """).fetchdf()
        fixed_assets_total_MEX = df.set_index("TIME_PERIOD")["OBS_VALUE"].values
        other[geo]['fixed_assets'] = shares * fixed_assets_total_MEX
        # CONSUMPTION OF FIXED CAPITAL: use ECB data and USA shares
        capital_consumption_ECB = pd.read_excel(db_folder / "MEX" / "ECB Data Portal wide_20250311173304.xlsx", sheet_name='Sheet1', header=0)
        capital_consumption_ECB.columns = ['TIME', 'TIME_PERIOD', 'VALUE']
        # Ensure TIME_PERIOD is just the year as integer
        capital_consumption_ECB['TIME_PERIOD'] = capital_consumption_ECB['TIME_PERIOD'].astype(str).str[:4].astype(int)
        capital_consumption_array = capital_consumption_ECB.loc[(capital_consumption_ECB['TIME_PERIOD'] >= calibration_years[0]) & (capital_consumption_ECB['TIME_PERIOD'] <= calibration_years[-1]), 'VALUE'].to_numpy().reshape(1, -1)
        other[geo]['capital_consumption'] = shares * capital_consumption_array
    elif geo == 'GBR':
        if calibration_years[-1] == 2022:
            data_matrix = other[geo]['fixed_assets']
            last_value = data_matrix[-1, -1] 
            if last_value == 0:
                V_2020 = data_matrix[:, -3]
                V_2021 = data_matrix[:, -2]
                V_2022_extrapolated = V_2021 + (V_2021 - V_2020)
                data_matrix[:, -1] = V_2022_extrapolated
                other[geo]['fixed_assets'] = data_matrix
    elif geo == 'USA':
        if calibration_years[-1] == 2022:
                data_matrix = other[geo]['capital_consumption']
                V_2020 = data_matrix[:, -3]
                V_2021 = data_matrix[:, -2]
                V_2022_extrapolated = V_2021 + (V_2021 - V_2020)
                data_matrix[:, -1] = V_2022_extrapolated
                other[geo]['capital_consumption'] = data_matrix

    if geo == 'AUS':
            aus_df = pd.read_excel(db_folder / "AUS" / "5204036_Household_Income_Account.xlsx", sheet_name='Data1', skiprows=5)
            start_idx = calibration_years[0] - 1960
            end_idx = calibration_years[-1] - 1960 + 1
            col_28 = pd.to_numeric(aus_df.iloc[start_idx:end_idx, 28], errors='coerce')
            col_11 = pd.to_numeric(aus_df.iloc[start_idx:end_idx, 11], errors='coerce')
            social_contributions_series = (col_28 - col_11).astype(float)             
            other[geo]['social_contributions'] = social_contributions_series.to_numpy().reshape(-1, 1)

    if geo == 'CAN':
        df = conn.execute(f"""
            SELECT TIME_PERIOD, OBS_VALUE
            FROM {other_tables_y[6]}
            WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S11'
            AND COUNTERPART_SECTOR = 'S1'
            AND ACCOUNTING_ENTRY = 'D'
            AND _TRANSACTION = 'D5'
            AND INSTR_ASSET = '_Z'
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
            ORDER BY TIME_PERIOD
        """).fetchdf()
        corporate_tax_non_financial_corporations = df.set_index("TIME_PERIOD")["OBS_VALUE"].values            
        df = conn.execute(f"""
            SELECT TIME_PERIOD, OBS_VALUE
            FROM {other_tables_y[6]}
            WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S12'
            AND COUNTERPART_SECTOR = 'S1'
            AND ACCOUNTING_ENTRY = 'D'
            AND _TRANSACTION = 'D5'
            AND INSTR_ASSET = '_Z'
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
            ORDER BY TIME_PERIOD
        """).fetchdf()
        corporate_tax_financial_corporations = df.set_index("TIME_PERIOD")["OBS_VALUE"].values
        other[geo]['corporate_tax'] = (corporate_tax_non_financial_corporations + corporate_tax_financial_corporations).reshape(-1, 1)

    if geo == 'KOR':
        df = conn.execute(f"""
            SELECT TIME_PERIOD, OBS_VALUE
            FROM {other_tables_y[13]}
            WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S13'
            AND COUNTERPART_SECTOR = '_Z'
            AND ACCOUNTING_ENTRY = 'C'
            AND _TRANSACTION = 'D5'
            AND INSTR_ASSET = '_Z'
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
            ORDER BY TIME_PERIOD
        """).fetchdf()
        other[geo]['income_tax'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1, 1)
        df = conn.execute(f"""
            SELECT TIME_PERIOD, OBS_VALUE
            FROM {other_tables_y[13]}
            WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S13'
            AND COUNTERPART_SECTOR = '_Z'
            AND ACCOUNTING_ENTRY = 'C'
            AND _TRANSACTION = 'D91'
            AND INSTR_ASSET = '_Z'
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
            ORDER BY TIME_PERIOD
        """).fetchdf()
        other[geo]['capital_taxes'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1, 1)

    if geo == 'MEX':
        df = conn.execute(f"""
            SELECT TIME_PERIOD, OBS_VALUE
            FROM {other_tables_y[6]}
            WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S1'
            AND COUNTERPART_SECTOR = 'S1'
            AND ACCOUNTING_ENTRY = 'D'
            AND _TRANSACTION = 'D11'
            AND INSTR_ASSET = '_Z'
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
            ORDER BY TIME_PERIOD
        """).fetchdf()
        other[geo]['wages'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1, 1)

    if geo == 'PRT':
        df = conn.execute(f"""
            SELECT TIME_PERIOD, OBS_VALUE
            FROM {other_tables_y[13]}
            WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S13'
            AND COUNTERPART_SECTOR = '_Z'
            AND ACCOUNTING_ENTRY = 'C'
            AND _TRANSACTION = 'D91'
            AND INSTR_ASSET = '_Z'
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
            ORDER BY TIME_PERIOD
        """).fetchdf()
        other[geo]['capital_taxes'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1, 1)
    elif geo in ['CAN', 'EST', 'SWE']:
        other[geo]['capital_taxes'] = np.zeros(len(calibration_years)).reshape(-1, 1)
        
    if geo in ['CAN', 'KOR', 'USA']:
        df = conn.execute(f"""
            SELECT TIME_PERIOD, OBS_VALUE
            FROM {other_tables_y[9]}
            WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S13'
            AND COUNTERPART_SECTOR = '_Z'
            AND ACCOUNTING_ENTRY = 'D'
            AND _TRANSACTION = 'OTE'
            AND INSTR_ASSET = '_Z'
            AND EXPENDITURE = '_T'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
            ORDER BY TIME_PERIOD
        """).fetchdf()
        total_gov_exp = df.set_index("TIME_PERIOD")["OBS_VALUE"].values
        df = conn.execute(f"""
            SELECT TIME_PERIOD, OBS_VALUE
            FROM {other_tables_y[12]}
            WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A'
            AND REF_AREA = '{geo}'
            AND UNIT_MEASURE = 'PT_OTE_S13'
            AND EXPEND_SOURCE = 'ES10'
            AND SPENDING_TYPE = '_T'
            AND PROGRAMME_TYPE = 'TP71'
            AND PRICE_BASE = '_Z'
            ORDER BY TIME_PERIOD
        """).fetchdf()
        pct_unemp_benefits = df.set_index("TIME_PERIOD")["OBS_VALUE"].values
        if total_gov_exp.shape == pct_unemp_benefits.shape and np.isfinite(pct_unemp_benefits).all():
            other[geo]['unemployment_benefits'] = (total_gov_exp * pct_unemp_benefits / 100).reshape(-1, 1)
        if other[geo]['unemployment_benefits'].size == 0:
            df = conn.execute(f"""
                SELECT TIME_PERIOD, OBS_VALUE
                FROM {other_tables_y[12]}
                WHERE TIME_PERIOD IN {years_sql}
                AND FREQ = 'A'
                AND REF_AREA = '{geo}'
                AND UNIT_MEASURE = 'XDC'
                AND EXPEND_SOURCE = 'ES10'
                AND SPENDING_TYPE = '_T'
                AND PROGRAMME_TYPE = 'TP71'
                AND PRICE_BASE = 'V'
                ORDER BY TIME_PERIOD
            """).fetchdf()
            other[geo]['unemployment_benefits'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1, 1)
    elif geo == 'MEX':
        other[geo]['unemployment_benefits'] = np.zeros(len(calibration_years)).reshape(-1, 1)

    if geo == 'USA':
        if calibration_years[-1] == 2022:
            wages = np.array(other[geo]['wages'], dtype=float).flatten()
            x_2 = np.arange(1, len(wages) + 1)
            target_point = x_2[-1] + 1
            f = interp1d(x_2, wages, kind='linear', fill_value='extrapolate')
            extrapolated_value = f(target_point)
            other[geo]['wages'] = np.append(wages, extrapolated_value).reshape(-1, 1)

    if geo in ['CAN', 'KOR', 'MEX', 'USA']:
        df = conn.execute(f"""
            SELECT TIME_PERIOD, OBS_VALUE
            FROM {other_tables_y[9]}
            WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S13'
            AND COUNTERPART_SECTOR = '_Z'
            AND ACCOUNTING_ENTRY = 'D'
            AND _TRANSACTION = 'OTE'
            AND INSTR_ASSET = '_Z'
            AND EXPENDITURE = '_T'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
            ORDER BY TIME_PERIOD
        """).fetchdf()
        total_gov_exp = df.set_index("TIME_PERIOD")["OBS_VALUE"].values
        df = conn.execute(f"""
            SELECT TIME_PERIOD, OBS_VALUE
            FROM {other_tables_y[12]}
            WHERE TIME_PERIOD IN {years_sql}
            AND FREQ = 'A'
            AND REF_AREA = '{geo}'
            AND UNIT_MEASURE = 'PT_OTE_S13'
            AND EXPEND_SOURCE = 'ES10'
            AND SPENDING_TYPE = '_T'
            AND PROGRAMME_TYPE = 'TP11'
            AND PRICE_BASE = '_Z'
            ORDER BY TIME_PERIOD
        """).fetchdf()
        pct_pension_benefits = df.set_index("TIME_PERIOD")["OBS_VALUE"].values
        if total_gov_exp.shape == pct_pension_benefits.shape and np.isfinite(pct_pension_benefits).all():
            other[geo]['pension_benefits'] = (total_gov_exp * pct_pension_benefits / 100).reshape(-1, 1)
        if other[geo]['pension_benefits'].size == 0:
            df = conn.execute(f"""
                SELECT TIME_PERIOD, OBS_VALUE
                FROM {other_tables_y[12]}
                WHERE TIME_PERIOD IN {years_sql}
                AND FREQ = 'A'
                AND REF_AREA = '{geo}'
                AND UNIT_MEASURE = 'XDC'
                AND EXPEND_SOURCE = 'ES10'
                AND SPENDING_TYPE = '_T'
                AND PROGRAMME_TYPE = 'TP11'
                AND PRICE_BASE = 'V'
                ORDER BY TIME_PERIOD
            """).fetchdf()
            other[geo]['pension_benefits'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1, 1)

    # By-industry data adjustments: merging industries R and S
    for field in other[geo]:
        field_data = other[geo][field]
        if field_data.shape[0] == 19:
            summed_row = field_data[17, :] + field_data[18, :]
            other[geo][field] = np.vstack([field_data[:17, :], summed_row])

# %% A.4) Save annual data to .mat file
savemat(
    str(DATA_DIR / "output" / "processed" / "data_y.mat"), 
    {
        "geos": geos, 
        "geos_2": geos_2, 
        "currencies": currencies, 
        "years_num": years_num, 
        "nom_exchange_rate": nom_exchange_rate, 
        "figaro": figaro, 
        "other": other
    }, 
    do_compression=True
)

# %% B) QUARTERLY DATA EXTRACTION
# %% B.1) Exchange Rate Data
df = conn.execute(f"""
    SELECT TIME_PERIOD, currency, OBS_VALUE
    FROM {exchange_rate_table_q[0]}
    WHERE TIME_PERIOD IN {quarters_sql}
    AND statinfo = 'AVG'
    AND currency = 'USD'
    ORDER BY TIME_PERIOD
""").fetchdf()
eur_to_usd = df.set_index("TIME_PERIOD")["OBS_VALUE"].values   # how many EUR for 1 USD?
nom_exchange_rate = {}
for i, geo in enumerate(geos):
    curr = currencies[i]
    if curr != 'EUR':
        curr_sql = str((curr,))
        df = conn.execute(f"""
            SELECT TIME_PERIOD, currency, OBS_VALUE
            FROM {exchange_rate_table_q[0]}
            WHERE TIME_PERIOD IN {quarters_sql}
            AND statinfo = 'AVG'
            AND currency IN {curr_sql}
            ORDER BY TIME_PERIOD
        """).fetchdf()
        eur_to_lcu = df.set_index("TIME_PERIOD")["OBS_VALUE"].values # how many EUR for 1 LCU?
    else:
        eur_to_lcu = np.ones(len(eur_to_usd))  # how many EUR for 1 EUR?
    nom_exchange_rate[geo] = (eur_to_usd / eur_to_lcu).reshape(-1, 1)

# %% B.2) Sectoral Data
sectoral = {}
for i, geo in enumerate(geos):
    if geo not in sectoral:
        sectoral[geo] = {}
    # AUS data are in a different dataset
    table = sectoral_tables_q[2] if geo == "AUS" else sectoral_tables_q[0]
    table_gov = sectoral_tables_q[1]
    df = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {table}
        WHERE TIME_PERIOD IN {quarters_sql}
            AND FREQ = 'Q'
            AND REF_AREA = '{geo}' 
            AND SECTOR = 'S1M' 
            AND COUNTERPART_SECTOR = 'S1' 
            AND ACCOUNTING_ENTRY = 'A' 
            AND _TRANSACTION = 'LE' 
            AND INSTR_ASSET = 'F2' 
            AND UNIT_MEASURE = 'XDC' 
            AND PRICE_BASE = 'V' 
            AND CURRENCY_DENOM = '_T' 
        ORDER BY TIME_PERIOD
        """).fetchdf()
    sectoral[geo]['households_cash_q'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1, 1)
    df = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {table}
        WHERE TIME_PERIOD IN {quarters_sql}
            AND FREQ = 'Q'
            AND REF_AREA = '{geo}' 
            AND SECTOR = 'S11' 
            AND COUNTERPART_SECTOR = 'S1' 
            AND ACCOUNTING_ENTRY = 'A' 
            AND _TRANSACTION = 'LE' 
            AND INSTR_ASSET = 'F2' 
            AND UNIT_MEASURE = 'XDC' 
            AND PRICE_BASE = 'V' 
            AND CURRENCY_DENOM = '_T' 
        ORDER BY TIME_PERIOD
        """).fetchdf()
    sectoral[geo]['firm_cash_q'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1, 1)
    df = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {table}
        WHERE TIME_PERIOD IN {quarters_sql}
            AND FREQ = 'Q'
            AND REF_AREA = '{geo}' 
            AND SECTOR = 'S11'
            AND COUNTERPART_SECTOR = 'S1'
            AND ACCOUNTING_ENTRY = 'L'
            AND _TRANSACTION = 'LE'
            AND INSTR_ASSET = 'F4'
            AND MATURITY = 'T'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
            AND CURRENCY_DENOM = '_T'
        ORDER BY TIME_PERIOD
        """).fetchdf()
    sectoral[geo]['firm_debt_q'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1, 1)
    df = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {table}
        WHERE TIME_PERIOD IN {quarters_sql}
            AND FREQ = 'Q'
            AND REF_AREA = '{geo}' 
            AND SECTOR = 'S12K'
            AND COUNTERPART_SECTOR = 'S1'
            AND ACCOUNTING_ENTRY = 'L'
            AND _TRANSACTION = 'LE'
            AND INSTR_ASSET = 'F5'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
            AND CURRENCY_DENOM = '_T'
        ORDER BY TIME_PERIOD
        """).fetchdf()
    sectoral[geo]['bank_equity_q'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1, 1)
    df = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {table_gov}
        WHERE TIME_PERIOD IN {quarters_sql}
            AND FREQ = 'Q'
            AND REF_AREA = '{geo}' 
            AND SECTOR = 'S13'
            AND COUNTERPART_SECTOR = 'S1'
            AND ACCOUNTING_ENTRY = 'L'
            AND _TRANSACTION = 'LE'
            AND INSTR_ASSET = 'FD4'
            AND MATURITY = 'T'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
            AND DEBT_BREAKDOWN = 'INST'
            AND CURRENCY_DENOM = '_T'
        ORDER BY TIME_PERIOD
        """).fetchdf()
    sectoral[geo]['gov_debt_q'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1, 1)
    if geo == 'KOR': # KOR gov_debt_q data: % Data are available only at Q4 values, and from 2011 until 2023. So 2010 and 2024 also fully missing.
        gov_debt = sectoral[geo]['gov_debt_q'].flatten()
        if calibration_years_q[0] == 2010 and calibration_years_q[-1] == 2024:
            n = len(gov_debt)
            slope_start = gov_debt[1] - gov_debt[0]
            slope_end = gov_debt[-1] - gov_debt[-2]
            q4_2010 = gov_debt[0] - slope_start
            q4_2024 = gov_debt[-1] + slope_end
            q4 = np.concatenate([[q4_2010], gov_debt, [q4_2024]])
            q_all = np.full(4 * len(q4), np.nan)
            q_all[3::4] = q4
            valid_mask = ~np.isnan(q_all)
            valid_idx = np.where(valid_mask)[0]
            f = interp1d(valid_idx, q_all[valid_idx], kind='linear', fill_value='extrapolate')
            q_all = f(np.arange(len(q_all)))
            sectoral[geo]['gov_debt_q'] = q_all.reshape(-1, 1)
        elif calibration_years_q[0] == 2010 and calibration_years_q[-1] == 2023:
            n = len(gov_debt)
            slope_start = gov_debt[1] - gov_debt[0]
            q4_2010 = gov_debt[0] - slope_start
            q4 = np.concatenate([[q4_2010], gov_debt])
            q_all = np.full(4 * len(q4), np.nan)
            q_all[3::4] = q4
            valid_mask = ~np.isnan(q_all)
            valid_idx = np.where(valid_mask)[0]
            f = interp1d(valid_idx, q_all[valid_idx], kind='linear', fill_value='extrapolate')
            q_all = f(np.arange(len(q_all)))
            sectoral[geo]['gov_debt_q'] = q_all.reshape(-1, 1)
        else:
            slope_end = gov_debt[-1] - gov_debt[-2]
            q4_2024 = gov_debt[-1] + slope_end
            q4 = np.concatenate([gov_debt, [q4_2024]])
            q_all = np.full(4 * len(q4), np.nan)
            q_all[3::4] = q4
            valid_mask = ~np.isnan(q_all)
            valid_idx = np.where(valid_mask)[0]
            f = interp1d(valid_idx, q_all[valid_idx], kind='linear', fill_value='extrapolate')
            q_all = f(np.arange(len(q_all)))
            sectoral[geo]['gov_debt_q'] = q_all.reshape(-1, 1)

# %% B.3) Save quarterly data to .mat file
savemat(
    str(DATA_DIR / "output" / "processed" / "data_q.mat"), 
    {
        "geos": geos, 
        "geos_2": geos_2, 
        "currencies": currencies, 
        "quarters_num": quarters_num, 
        "nom_exchange_rate": nom_exchange_rate, 
        "sectoral": sectoral
    }, 
    do_compression=True
)

# %% C) QUARTERLY DATA EXTRACTION (for estimation of exogenous vars AR1 coefficients)
# %% C.1) Exchange Rate Data
df = conn.execute(f"""
    SELECT TIME_PERIOD, currency, OBS_VALUE
    FROM {exchange_rate_table_q[0]}  
    WHERE TIME_PERIOD IN {quarters_est_sql}
    AND statinfo = 'AVG'
    AND currency = 'USD'
    ORDER BY TIME_PERIOD
""").fetchdf()
eur_to_usd = df.set_index("TIME_PERIOD")["OBS_VALUE"].values   # how many EUR for 1 USD?
nom_exchange_rate_est = {}
for i, geo in enumerate(geos):
    curr = currencies[i]
    if curr != 'EUR':
        curr_sql = str((curr,))
        df = conn.execute(f"""
            SELECT TIME_PERIOD, currency, OBS_VALUE
            FROM {exchange_rate_table_q[0]}            
            WHERE TIME_PERIOD IN {quarters_est_sql}
            AND statinfo = 'AVG'
            AND currency IN {curr_sql}
            ORDER BY TIME_PERIOD
        """).fetchdf()
        eur_to_lcu = df.set_index("TIME_PERIOD")["OBS_VALUE"].values # how many EUR for 1 LCU?
    else:
        eur_to_lcu = np.ones(len(eur_to_usd))  # how many EUR for 1 EUR?
    nom_exchange_rate_est[geo] = (eur_to_usd / eur_to_lcu).reshape(-1, 1)

# %% C.2) GDP Data
gdp_data = {}
for i, geo in enumerate(geos):
    if geo not in gdp_data:
        gdp_data[geo] = {}
    price_base_real = 'Q' if geo == 'MEX' else 'L'
    df = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {gdp_tables_q[0]}
        WHERE TIME_PERIOD IN {quarters_est_sql}
            AND FREQ = 'Q'
            AND ADJUSTMENT = 'Y'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S1'
            AND COUNTERPART_SECTOR = 'S1'
            AND _TRANSACTION = 'B1GQ'
            AND INSTR_ASSET = '_Z'
            AND ACTIVITY = '_Z'
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
        ORDER BY TIME_PERIOD
    """).fetchdf()
    gdp_data[geo]['nom_gdp_q'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1,1)
    df = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {gdp_tables_q[0]}
        WHERE TIME_PERIOD IN {quarters_est_sql}
            AND FREQ = 'Q'
            AND ADJUSTMENT = 'Y'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S1'
            AND COUNTERPART_SECTOR = 'S1'
            AND _TRANSACTION = 'B1GQ'
            AND INSTR_ASSET = '_Z'
            AND ACTIVITY = '_Z'
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = '{price_base_real}'
        ORDER BY TIME_PERIOD
    """).fetchdf()
    gdp_data[geo]['real_gdp_q'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1,1)
    gdp_data[geo]['gdp_deflator_q'] = gdp_data[geo]['nom_gdp_q']/gdp_data[geo]['real_gdp_q']
    df = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {gdp_tables_q[0]}
        WHERE TIME_PERIOD IN {quarters_est_sql}
            AND FREQ = 'Q'
            AND ADJUSTMENT = 'Y'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S1M'
            AND COUNTERPART_SECTOR = 'S1'
            AND _TRANSACTION = 'P3'
            AND INSTR_ASSET = '_Z'
            AND ACTIVITY = '_Z'
            AND EXPENDITURE = '_T'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
        ORDER BY TIME_PERIOD
    """).fetchdf()
    gdp_data[geo]['nom_hh_consumption_q'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1,1)
    df = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {gdp_tables_q[0]}
        WHERE TIME_PERIOD IN {quarters_est_sql}
            AND FREQ = 'Q'
            AND ADJUSTMENT = 'Y'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S1M'
            AND COUNTERPART_SECTOR = 'S1'
            AND _TRANSACTION = 'P3'
            AND INSTR_ASSET = '_Z'
            AND ACTIVITY = '_Z'
            AND EXPENDITURE = '_T'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = '{price_base_real}'
        ORDER BY TIME_PERIOD
    """).fetchdf()
    gdp_data[geo]['real_hh_consumption_q'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1,1)
    gdp_data[geo]['hh_consumption_deflator_q'] = gdp_data[geo]['nom_hh_consumption_q']/gdp_data[geo]['real_hh_consumption_q']
    df = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {gdp_tables_q[0]}
        WHERE TIME_PERIOD IN {quarters_est_sql}
            AND FREQ = 'Q'
            AND ADJUSTMENT = 'Y'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S13'
            AND COUNTERPART_SECTOR = 'S1'
            AND _TRANSACTION = 'P3'
            AND INSTR_ASSET = '_Z'
            AND ACTIVITY = '_Z'
            AND EXPENDITURE = '_T'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
        ORDER BY TIME_PERIOD
    """).fetchdf()
    gdp_data[geo]['nom_gov_consumption_q'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1,1)
    df = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {gdp_tables_q[0]}
        WHERE TIME_PERIOD IN {quarters_est_sql}
            AND FREQ = 'Q'
            AND ADJUSTMENT = 'Y'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S13'
            AND COUNTERPART_SECTOR = 'S1'
            AND _TRANSACTION = 'P3'
            AND INSTR_ASSET = '_Z'
            AND ACTIVITY = '_Z'
            AND EXPENDITURE = '_T'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = '{price_base_real}'
        ORDER BY TIME_PERIOD
    """).fetchdf()
    gdp_data[geo]['real_gov_consumption_q'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1,1)
    gdp_data[geo]['gov_consumption_deflator_q'] = gdp_data[geo]['nom_gov_consumption_q']/gdp_data[geo]['real_gov_consumption_q']
    df = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {gdp_tables_q[0]}
        WHERE TIME_PERIOD IN {quarters_est_sql}
            AND FREQ = 'Q'
            AND ADJUSTMENT = 'Y'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S1'
            AND COUNTERPART_SECTOR = 'S1'
            AND _TRANSACTION = 'P3'
            AND INSTR_ASSET = '_Z'
            AND ACTIVITY = '_Z'
            AND EXPENDITURE = '_T'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
        ORDER BY TIME_PERIOD
    """).fetchdf()
    gdp_data[geo]['nom_final_consumption_q'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1,1)
    df = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {gdp_tables_q[0]}
        WHERE TIME_PERIOD IN {quarters_est_sql}
            AND FREQ = 'Q'
            AND ADJUSTMENT = 'Y'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S1'
            AND COUNTERPART_SECTOR = 'S1'
            AND _TRANSACTION = 'P3'
            AND INSTR_ASSET = '_Z'
            AND ACTIVITY = '_Z'
            AND EXPENDITURE = '_T'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = '{price_base_real}'
        ORDER BY TIME_PERIOD
    """).fetchdf()
    gdp_data[geo]['real_final_consumption_q'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1,1)
    gdp_data[geo]['final_consumption_deflator_q'] = gdp_data[geo]['nom_final_consumption_q']/gdp_data[geo]['real_final_consumption_q']
    df = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {gdp_tables_q[0]}
        WHERE TIME_PERIOD IN {quarters_est_sql}
            AND FREQ = 'Q'
            AND ADJUSTMENT = 'Y'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S1'
            AND COUNTERPART_SECTOR = 'S1'
            AND _TRANSACTION = 'P51G'
            AND INSTR_ASSET = '_Z'
            AND ACTIVITY = '_T'
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
        ORDER BY TIME_PERIOD
    """).fetchdf()
    gdp_data[geo]['nom_fixed_cf_q'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1,1)
    df = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {gdp_tables_q[0]}
        WHERE TIME_PERIOD IN {quarters_est_sql}
            AND FREQ = 'Q'
            AND ADJUSTMENT = 'Y'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S1'
            AND COUNTERPART_SECTOR = 'S1'
            AND _TRANSACTION = 'P51G'
            AND INSTR_ASSET = '_Z'
            AND ACTIVITY = '_T'
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = '{price_base_real}'
        ORDER BY TIME_PERIOD
    """).fetchdf()
    gdp_data[geo]['real_fixed_cf_q'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1,1)
    gdp_data[geo]['fixed_cf_deflator_q'] = gdp_data[geo]['nom_fixed_cf_q']/gdp_data[geo]['real_fixed_cf_q']
    df = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {gdp_tables_q[0]}
        WHERE TIME_PERIOD IN {quarters_est_sql}
            AND FREQ = 'Q'
            AND ADJUSTMENT = 'Y'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S1'
            AND COUNTERPART_SECTOR = 'S1'
            AND _TRANSACTION = 'P6'
            AND INSTR_ASSET = '_Z'
            AND ACTIVITY = '_Z'
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
        ORDER BY TIME_PERIOD
    """).fetchdf()
    gdp_data[geo]['nom_exports_q'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1,1)
    df = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {gdp_tables_q[0]}
        WHERE TIME_PERIOD IN {quarters_est_sql}
            AND FREQ = 'Q'
            AND ADJUSTMENT = 'Y'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S1'
            AND COUNTERPART_SECTOR = 'S1'
            AND _TRANSACTION = 'P6'
            AND INSTR_ASSET = '_Z'
            AND ACTIVITY = '_Z'
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = '{price_base_real}'
        ORDER BY TIME_PERIOD
    """).fetchdf()
    gdp_data[geo]['real_exports_q'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1,1)
    gdp_data[geo]['exports_deflator_q'] = gdp_data[geo]['nom_exports_q']/gdp_data[geo]['real_exports_q']
    df = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {gdp_tables_q[0]}
        WHERE TIME_PERIOD IN {quarters_est_sql}
            AND FREQ = 'Q'
            AND ADJUSTMENT = 'Y'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S1'
            AND COUNTERPART_SECTOR = 'S1'
            AND _TRANSACTION = 'P7'
            AND INSTR_ASSET = '_Z'
            AND ACTIVITY = '_Z'
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = 'V'
        ORDER BY TIME_PERIOD
    """).fetchdf()
    gdp_data[geo]['nom_imports_q'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1,1)
    df = conn.execute(f"""
        SELECT TIME_PERIOD, OBS_VALUE
        FROM {gdp_tables_q[0]}
        WHERE TIME_PERIOD IN {quarters_est_sql}
            AND FREQ = 'Q'
            AND ADJUSTMENT = 'Y'
            AND REF_AREA = '{geo}'
            AND SECTOR = 'S1'
            AND COUNTERPART_SECTOR = 'S1'
            AND _TRANSACTION = 'P7'
            AND INSTR_ASSET = '_Z'
            AND ACTIVITY = '_Z'
            AND EXPENDITURE = '_Z'
            AND UNIT_MEASURE = 'XDC'
            AND PRICE_BASE = '{price_base_real}'
        ORDER BY TIME_PERIOD
    """).fetchdf()
    gdp_data[geo]['real_imports_q'] = df.set_index("TIME_PERIOD")["OBS_VALUE"].values.reshape(-1,1)
    gdp_data[geo]['imports_deflator_q'] = gdp_data[geo]['nom_imports_q']/gdp_data[geo]['real_imports_q']

# %% C.3) Federal Reserve (FED) Data
fed_table = pd.read_excel(db_folder / 'USA' / 'FEDFUNDS.xlsx', sheet_name='Quarterly')
fed_table['DATE'] = pd.to_datetime(fed_table.iloc[:, 0], format='%Y-%m-%d')
fed_table['dates_fed'] = (fed_table['DATE'] - pd.Timedelta(days=1))
fed_table['dates_fed_ordinal'] = fed_table['dates_fed'].apply(lambda x: x.toordinal() + 366)
values_fed = fed_table.iloc[:, 1].values
mask = (fed_table['dates_fed_ordinal'] >= quarters_est_num.min()) & (fed_table['dates_fed_ordinal'] <= quarters_est_num.max())
fed_policy_rate = 1/100*values_fed[mask].reshape(-1,1)

# %% C.4) Save quarterly data for estimation to .mat file
savemat(
    str(DATA_DIR / "output" / "processed" / "data_q_est.mat"), 
    {
        "geos": geos, 
        "geos_2": geos_2, 
        "currencies": currencies, 
        "quarters_est_num": quarters_est_num, 
        "nom_exchange_rate_est": nom_exchange_rate_est, 
        "gdp_data": gdp_data, 
        "fed_policy_rate": fed_policy_rate
    }, 
    do_compression=True
)

# %% CLOSE CONNECTION TO DUCKDB DATABASE
conn.close()
print(f"Connection to DuckDB database ({db_path}) closed!")
print("=" * 30)
