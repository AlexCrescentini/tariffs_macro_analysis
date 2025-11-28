"""
List of all datasets used for GTAB calibration
"""

# (a) EUROSTAT FIGARO IO data (yearly)
a_figaro_data_yearly = [
    ("https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/", "naio_10_fcp_ii1", "?format=SDMX-CSV&compressed=true&attributes=none"),
    ("https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/", "naio_10_fcp_ii2", "?format=SDMX-CSV&compressed=true&attributes=none"),
    ("https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/", "naio_10_fcp_ii3", "?format=SDMX-CSV&compressed=true&attributes=none")
]

# (b) Other OECD and EUROSTAT data (yearly)
b_other_data_yearly = [
    ("https://sdmx.oecd.org/public/rest/data/", "OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE3,1.0", "?detail=full&format=csvfile"),
    ("https://sdmx.oecd.org/public/rest/data/", "OECD.SDD.NAD,DSD_NASEC10@DF_TABLE14,1.1", "?detail=full&format=csvfile"),
    ("https://sdmx.oecd.org/public/rest/data/", "OECD.SDD.NAD,DSD_NASEC10@DF_TABLE10,1.1", "?detail=full&format=csvfile"),
    ("https://sdmx.oecd.org/public/rest/data/", "OECD.SDD.NAD,DSD_NASEC10@DF_TABLE13_BAL,1.1", "?detail=full&format=csvfile"),
    ("https://sdmx.oecd.org/public/rest/data/", "OECD.SDD.NAD,DSD_NASEC10@DF_TABLE11,1.1", "?detail=full&format=csvfile"),
    ("https://sdmx.oecd.org/public/rest/data/", "OECD.ELS.SPD,DSD_SOCX_AGG@DF_PUB_PRV", "?detail=full&format=csvfile"),
    ("https://sdmx.oecd.org/public/rest/data/", "OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE7", "?detail=full&format=csvfile"),
    ("https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/", "nama_10_nfa_st/1.0", "?compress=false&format=csvdata&formatVersion=2.0&lang=en&labels=id"),
    ("https://sdmx.oecd.org/public/rest/data/", "OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE1_INCOME,2.0", "?detail=full&format=csvfile"),
    ("https://sdmx.oecd.org/public/rest/data/", "OECD.SDD.NAD,DSD_NASEC10@DF_TABLE12_REV,1.1", "?detail=full&format=csvfile"),
    ("https://sdmx.oecd.org/public/rest/data/", "OECD.SDD.NAD,DSD_NASEC10@DF_TABLE9B,1.1", "?detail=full&format=csvfile"),
    ("https://sdmx.oecd.org/public/rest/data/", "OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE9A,2.0", "?detail=full&format=csvfile"),
    ("https://sdmx.oecd.org/public/rest/data/", "OECD.SDD.TPS,DSD_ALFS@DF_ALFS_EMP_EES", "?detail=full&format=csvfile"),
    ("https://sdmx.oecd.org/public/rest/data/", "OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE6,2.0", "?detail=full&format=csvfile"),
    ("https://sdmx.oecd.org/public/rest/data/", "OECD.SDD.TPS,DSD_LFS@DF_IALFS_UNE_M,1.0", "?detail=full&format=csvfile")
]

# (c) Sectoral OECD data (quarterly)
c_sectoral_data_quarterly = [
    ("https://sdmx.oecd.org/public/rest/data/", "OECD.SDD.NAD,DSD_NASEC20@DF_T720R_Q,1.1", "?detail=full&format=csvfile"),
    ("https://sdmx.oecd.org/public/rest/data/", "OECD.SDD.NAD,DSD_NASEC20@DF_T7PSD_Q,1.1", "?detail=full&format=csvfile"),
    ("https://sdmx.oecd.org/public/rest/data/", "OECD.SDD.NAD,DSD_NASEC20@DF_T710R_Q,1.1", "?detail=full&format=csvfile")
]

# (d) GDP data OECD (quarterly)
d_gdp_data_quarterly = [
    ("https://sdmx.oecd.org/public/rest/data/", "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_NATIO_CURR,1.1", "?detail=full&format=csvfile")
]

# (e) Exchange rate data (annual + quarterly)
e_exchange_rate_data_yearly_quarterly = [
    ("https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/", "ert_bil_eur_a/1.0", "?compress=false&format=csvdata&formatVersion=2.0&lang=en&labels=id"),
    ("https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/", "ert_bil_eur_q/1.0", "?compress=false&format=csvdata&formatVersion=2.0&lang=en&labels=id")
]