## Macroeconomic Analysis of Tariff Shocks

A macroeconomic model for policy and research applications, designed to analyse **how tariff changes ripple through global trade and the wider economy**. 

The project includes a fully automated data pipeline, a calibrated multi-country macroeconomic model with over a billion agents across 30 OECD countries, and a suite of policy simulations to assess the impact of tariffs on global trade, thus requiring expertise in data engineering, economic modelling, and policy-oriented analysis.

This work has been presented at leading institutions, including the **[European Central Bank](https://www.ecb.europa.eu/press/conferences/html/20250626_abm4policy.en.html)**, and is currently being tested at the **Bank of Canada**, underscoring its relevance for real-world policy analysis.

#### Project Overview

This repository mirrors the project workflow, organized into three main steps:

&nbsp;&nbsp;&nbsp;&nbsp;**1. Data Preparation** ([`data/`](data)) — *Python code provided in this repository*  
&nbsp;&nbsp;&nbsp;&nbsp;**2. Model Simulation** ([`model/`](model)) — *MATLAB code not shared (the model is not yet published)*  
&nbsp;&nbsp;&nbsp;&nbsp;**3. Policy Analysis** ([`policy/`](policy)) — *Python code provided in this repository*  

For steps 1 and 3, Jupyter notebooks are available at:

- [`notebooks/01_data_download.ipynb`](notebooks/01_data_download.ipynb) and [`notebooks/01_data_process.ipynb`](notebooks/01_data_process.ipynb)
- [`notebooks/03_policy_analysis.ipynb`](notebooks/03_policy_analysis.ipynb)

**Python packages:** `pandas`, `numpy`, `requests`, `duckdb`, `matplotlib`, `scipy`, `geopandas`, `plotly`

#### Repository Structure

```
trade_macro_analysis/
├── data/
│   ├── scripts/          # Python scripts for data pipeline
│   │   ├── main_download.py
│   │   ├── main_process.py
│   │   └── utils.py
│   └── output/
│       ├── downloaded/   # Raw data from OECD/Eurostat APIs
│       ├── processed/    # Cleaned data (.mat files for MATLAB)
│       └── figs/         # Data pipeline visualizations
│
├── model/
│   ├── scripts/          # MATLAB code for model simulation
│   │   └── main.m
│   └── output/           # Simulation results (.mat files)
│
├── policy/
│   ├── scripts/          # Python scripts for analysis
│   │   ├── main_plot.py
│   │   └── utils.py
│   └── output/           # Policy analysis figures and charts
│
├── notebooks/            # Step-by-step workflow notebooks
│   ├── 01_data_download.ipynb
│   ├── 01_data_process.ipynb
│   └── 03_policy_analysis.ipynb
│
└── docs/                 # Presentations and documentation
    └── slides_ECB.pdf
```
---

### 1. Data Preparation (Python)

End-to-end automatic workflow to prepare data for the model calibration ([Figure 1](#fig-data-pipeline) summarizes). This includes:

- Downloading **OECD and Eurostat datasets via API** through [`main_download.py`](data/main_download.py)/[`01_data_download.ipynb`](notebooks/01_data_download.ipynb). Data include time series, national accounts, sectoral accounts, input–output tables, and other indicators for 30 OECD countries ([Figure 2](#fig-oecd-map)). The tables downloaded are cleaned and stored in **DuckDB**.  
- Processing of data needed for the model calibration through  [`main_extract.py`](data/main_extract.py)/[`01_data_process.ipynb`](notebooks/01_data_process.ipynb). This requires handling missing values, zeros, and NaNs, and harmonizing data across industries and countries calibrated. Final series are extracted using **SQL** queries and exported as MATLAB-ready `.mat` files for the model.

<a id="fig-data-pipeline"></a>

<p align="center">
  <strong>Figure 1. Data pipeline</strong>
</p>

<p align="center">
  <img src="data/output/figs/data_pipeline_short.png" width="700">
</p>

<a id="fig-oecd-map"></a>

<p align="center">
  <strong>Figure 2. Calibrated Economies: 30 OECD Countries</strong>
</p>

<p align="center">
  <img src="data/output/figs/map_calibration.png" width="700">
</p>

---

### 2. Model Simulation (MATLAB)

The workflow of this section consists of:

- **Calibration** of parameters and initial conditions using data gathered from [Section 1](#1-data-preparation-python)
- **Scenario setup** — define tariff-shock policies as described in [Section 3](#3-policy-results-for-usa-can-trade-war-python)
- **Model simulation** — run the global trade model based on the scenario set above

> **Note:** As the model is not yet published, it is not shared here. An overview is available in the latest [Slides](docs/slides_ECB.pdf).

<!-- ([Figure 2](#fig-model-struct) and [Figure 3](#fig-seq-events) illustrate the model structure and the sequence of events within one quarter) -->

<!-- <a id="fig-model-struct"></a>
<p align="center"><em>Figure 2. Economy structure: F countries with sectors (I_f, H_f, J_f, B_f), a central bank, and the rest of the world (L, M). </em></p>
<p align="center">
  <img src="figs/structure_model.png" width="95%">
</p>

<a id="fig-seq-events"></a>
<p align="center"><em>Figure 3. Sequence of events within one model period (quarter).</em></p>
<p align="center">
  <img src="figs/sequence_of_events.png" width="95%">
</p>

**Steps description:**

&nbsp;&nbsp;&nbsp;&nbsp;**Step 1 — Expectations** — firms, households, and banks form expectations; RoW plans demand and supply.  
&nbsp;&nbsp;&nbsp;&nbsp;**Step 2 — Central Bank** — sets US policy rate; firms and banks plan credit and supply.  
&nbsp;&nbsp;&nbsp;&nbsp;**Step 3 — Credit Market** — banks set lending rates and provide loans to firms.  
&nbsp;&nbsp;&nbsp;&nbsp;**Step 4 — Labor Market** — firms post vacancies; households supply labor.  
&nbsp;&nbsp;&nbsp;&nbsp;**Step 5 — Production** — firms produce goods using labor and capital.  
&nbsp;&nbsp;&nbsp;&nbsp;**Step 6 — Wages & Inputs** — firms pay wages; demand intermediate and investment goods.  
&nbsp;&nbsp;&nbsp;&nbsp;**Step 7 — Households** — allocate income between consumption and savings.  
&nbsp;&nbsp;&nbsp;&nbsp;**Step 8 — Goods Market** — all demands and supplies meet; tariffs and taxes applied.  
&nbsp;&nbsp;&nbsp;&nbsp;**Step 9 — Aggregation** — prices, output, and inflation aggregated across agents and sectors.  
&nbsp;&nbsp;&nbsp;&nbsp;**Step 10 — Accounting** — update stocks, flows, and balance sheets for consistency.  
&nbsp;&nbsp;&nbsp;&nbsp;**Step 11 — Insolvency** — bankrupt agents exit and are replaced; next quarter begins. -->

---

### 3. Policy Analysis (Python)

This section applies the model to analyze the **USA–Canada trade war**, examining how tariff shocks propagate through bilateral trade flows and affect macroeconomic outcomes. The analysis focuses on three key dimensions:

- **Trade flows** — Changes in bilateral trade shares and trade diversion effects
- **Macroeconomic aggregates** — Real GDP growth and CPI inflation dynamics
- **Inflation decomposition** — Producer price inflation by components (cost-push, demand-pull, expectations) and by sector (across industries)

#### Simulation Scenarios

The policy analysis compares three scenarios to isolate the effects of unilateral tariffs and retaliatory measures:

| Scenario | Description |
|----------|-------------|
| **⚫ Baseline** | No additional tariffs (current trade policy) |
| **🔵 Unilateral Tariffs** | USA imposes 25% tariffs on all imports from Canada |
| **🔴 Retaliation** | Canada retaliates with 25% tariffs on all imports from USA (on top of Unilateral scenario) |

The figures below visualize the results from these simulations:

<p align="center">
  <strong>Figure 3. Trade: Bilateral Shares</strong>
</p>

<p align="center">
  <img src="policy/output/1_trade_shares.png" width="700">
</p>

<p align="center">
  <strong>Figure 4. Real GDP Growth and CPI Inflation (2025–2026 quarterly cumulated)</strong>
</p>

<p align="center">
  <img src="policy/output/2_gdp_inflation.png" width="700">
</p>

<p align="center">
  <strong>Figure 5. PPI Inflation by Component</strong>
</p>

<p align="center">
  <img src="policy/output/3_inflation_comp.png" width="700">
</p>

<p align="center">
  <strong>Figure 6. PPI Inflation by Sector</strong> <a href="https://gist.githack.com/AlexCrescentini/3d21c1344e2ec31baf1872bc9bad1812/raw/4_inflation_sect.html" target="_blank">(interactive version)</a>
</p>

<p align="center">
  <img src="policy/output/4_inflation_sect.gif" width="700">
</p>

---

<!-- ### How to store files for GitHub

* **Store compressed files only** (`.csv.gz` or `.tar.gz`) inside `data/raw/`.
* Prefer a single file per dataset; keep the raw filename as given by the source (e.g. `36100096.csv.gz`).
* Add `data/raw/*.csv` to `.gitignore` to avoid accidentally committing decompressed files.
* If some files are >100 MB consider Git LFS or hosting raw data externally and including download scripts.

Example `.gitignore` snippet:

```
# raw uncompressed data
/data/raw/*.csv
/data/raw/*.tmp
```

---

## Data pipeline: key scripts (suggested)

### `src/data_pipeline/download.py`

* takes the dataset list (you already keep it as a Python list)
* downloads files into `data/raw/`
* verifies checksums or `Content-Length` when available
* retries transient failures

### `src/data_pipeline/compress.py`

* converts `.csv` → `.csv.gz` (or `.tar.gz` if the tool creates tar.gz)
* keeps optional `-k` flag to preserve originals

Example function to load compressed CSV in Python (pandas):

```python
import pandas as pd

def load_csv_gz(path):
    # pandas handles gzip automatically when extension is .gz
    return pd.read_csv(path, compression="gzip", low_memory=False)
```

### `src/data_pipeline/ingest_duckdb.py`

* read compressed CSVs in chunks
* create typed tables in DuckDB
* store a snapshot file: `data/duckdb/gtab.duckdb`

Snippet (duckdb + pandas):

```python
import duckdb
import pandas as pd

con = duckdb.connect('data/duckdb/gtab.duckdb')

df = pd.read_csv('data/raw/36100096.csv.gz', compression='gzip')
con.register('tmp_df', df)
con.execute('CREATE TABLE IF NOT EXISTS capital_consumption AS SELECT * FROM tmp_df')
``` -->
<!-- --- -->


