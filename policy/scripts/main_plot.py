# %% === SETTINGS AND DATA LOADING === %%

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.animation import FuncAnimation, PillowWriter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import HTML, display
import base64
from pathlib import Path
from utils import matstruct_to_dict, remove_bad_seeds, shaded_error_bar, ci

# %% SETUP PATHS
SCRIPT_DIR = Path(__file__).parent if '__file__' in globals() else Path.cwd()
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # scripts -> policy -> root
POLICY_DIR = PROJECT_ROOT / "policy"
MODEL_DIR = PROJECT_ROOT / "model"
output_folder = POLICY_DIR / "output"
output_folder.mkdir(parents=True, exist_ok=True)
data_path = MODEL_DIR / "output"

# %% OTHER STUFF

# OECD-30 calibrated countries
geos_OECD30_oecd_codes = ['AUT','AUS','BEL','CAN','CZE','DEU','DNK','EST','GRC','ESP','FIN','FRA','HUN','IRL','ITA','JPN','KOR','LTU','LUX','LVA','MEX','NLD','NOR','POL','PRT','SWE','SVN','SVK','GBR','USA']
selected_countries = ['USA','CAN']
scenario_indices = [0, 1, 2]
scenario_names_dev = ['S1S0', 'S2S0']
scenarios = {}
scenarios_cleaned = {country: {f"scenario{s}": {} for s in scenario_indices} for country in selected_countries}
for country in selected_countries:
    raw_scenarios = [
        matstruct_to_dict(
            loadmat(str(data_path / f"{s}_{country}.mat"), squeeze_me=True, struct_as_record=False)[f"scenario{s}"]
        ) 
        for s in scenario_indices
    ]
    cleaned_scenarios = remove_bad_seeds(*raw_scenarios)
    for s in scenario_indices:
        scenarios_cleaned[country][f"scenario{s}"] = cleaned_scenarios[s]

# Plotting stuff
scenario_labels = ['Baseline', 'Unilateral Tariffs', 'Retaliation']
full_ecb_map = np.array([ # ECB color map
    [0, 50, 153],
    [255, 75, 0],
    [101, 184, 0],
    [255, 180, 0],
    [0, 177, 234],
    [0, 120, 22],
    [129, 57, 198],
    [92, 92, 92],
    [152, 161, 208],
    [253, 221, 167],
    [246, 177, 131],
    [206, 225, 175]
])/255
black = 'k'
blue = full_ecb_map[0]
yellow = full_ecb_map[3]
white = np.array([1,1,1])
light_gray = (white + full_ecb_map[7])/2
red = np.array([220, 60, 60]) / 255
blue = np.array([50, 100, 200]) / 255 
#scenario_colors = ['k', 'b', 'r']
scenario_colors = [black, blue, red]
industry_names = [
    'A01','A02','A03','B','C10T12','C13T15','C16','C17','C18','C19','C20','C21','C22','C23','C24',
    'C25','C26','C27','C28','C29','C30','C31_32','C33','D35','E36','E37T39','F','G45','G46','G47',
    'H49','H50','H51','H52','H53','I','J58','J59_60','J61','J62_63','K64','K65','K66','L','M69_70',
    'M71','M72','M73','M74_75','N77','N78','N79','N80T82','O84','P85','Q86','Q87_88','R90T92','R93',
    'S94','S95','S96'
]

# Pdf for all figures
multi_pdf_path = os.path.join(output_folder, '0_all_figures.pdf')
pdf = PdfPages(multi_pdf_path)

# %% === TRADE: BILATERAL SHARES === %%

def get_data(country, from_country):
    raw = np.asarray(scenarios_cleaned[country]['scenario0']["quarters_num"], dtype=float)
    quarters_dates = pd.to_datetime(raw - 719529, unit="D")
    quarters = pd.PeriodIndex(quarters_dates, freq='Q')
    x = np.arange(len(quarters))
    data0 = 100 * scenarios_cleaned[country]['scenario0'][f"nominal_imports_from_{from_country}_tot_quarterly"]
    data1 = 100 * scenarios_cleaned[country]['scenario1'][f"nominal_imports_from_{from_country}_tot_quarterly"]
    data2 = 100 * scenarios_cleaned[country]['scenario2'][f"nominal_imports_from_{from_country}_tot_quarterly"]
    return quarters, x, data0, data1, data2

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Top-left tile
quarters, x, data0, data1, data2 = get_data('USA', 'usa')
shaded_error_bar(axs[0, 0], x, np.mean(data0, axis=1), ci(data0.T), color=black, linewidth=2)
shaded_error_bar(axs[0, 0], x, np.mean(data1, axis=1), ci(data1.T), color=blue, linewidth=2)
shaded_error_bar(axs[0, 0], x, np.mean(data2, axis=1), ci(data2.T), color=red, linewidth=2)
axs[0, 0].set_title('USA <- USA', fontweight='bold')

# Top-right tile
quarters, x, data0, data1, data2 = get_data('USA', 'can')
shaded_error_bar(axs[0, 1], x, np.mean(data0, axis=1), ci(data0.T), color=black, linewidth=2)
shaded_error_bar(axs[0, 1], x, np.mean(data1, axis=1), ci(data1.T), color=blue, linewidth=2)
shaded_error_bar(axs[0, 1], x, np.mean(data2, axis=1), ci(data2.T), color=red, linewidth=2)
axs[0, 1].set_title('USA <- CAN', fontweight='bold')

# Bottom-left tile   
quarters, x, data0, data1, data2 = get_data('CAN', 'usa')
shaded_error_bar(axs[1, 0], x, np.mean(data0, axis=1), ci(data0.T), color=black, linewidth=2)
shaded_error_bar(axs[1, 0], x, np.mean(data1, axis=1), ci(data1.T), color=blue, linewidth=2)
shaded_error_bar(axs[1, 0], x, np.mean(data2, axis=1), ci(data2.T), color=red, linewidth=2)
axs[1, 0].set_title('CAN <- USA', fontweight='bold')

# Bottom-right tile
quarters, x, data0, data1, data2 = get_data('CAN', 'can')
shaded_error_bar(axs[1, 1], x, np.mean(data0, axis=1), ci(data0.T), color=black, linewidth=2)
shaded_error_bar(axs[1, 1], x, np.mean(data1, axis=1), ci(data1.T), color=blue, linewidth=2)
shaded_error_bar(axs[1, 1], x, np.mean(data2, axis=1), ci(data2.T), color=red, linewidth=2)
axs[1, 1].set_title('CAN <- CAN', fontweight='bold')

# Axis formatting
for ax_row in axs:
    for ax in ax_row:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(x)
        ax.set_xticklabels([f"Q{q.quarter}" for q in quarters], rotation=0, ha='center')
        #ax.set_xticklabels([f"{q.year} Q{q.quarter}" for q in quarters], rotation=45, ha='right')
        ax.set_xlim(x[0], x[-1])
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1))

# Common title
#fig.suptitle('Bilateral Trade Share (%)', fontsize=14, y=1.02)  # y>1 moves it above the figure

# Common legend         
handles = [mpatches.Patch(color=scenario_colors[i], label=scenario_labels[i]) for i in range(len(scenario_labels))]
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(scenario_labels), fontsize=12)

plt.tight_layout()
plt.show()

# Pdf, PNG, multi-page pdf
fig.savefig(os.path.join(output_folder, '1_trade_shares.pdf'), format='pdf', bbox_inches='tight')
fig.savefig(os.path.join(output_folder, '1_trade_shares.png'), format='png', dpi=300, bbox_inches='tight')
pdf.savefig(fig, bbox_inches='tight')

# %% === REAL GDP GROWTH AND CPI INFLATION (CUMULATIVE) === %%

# Prepare data
nC = len(selected_countries)
nS = len(scenario_indices)-1
gdp_dev = np.zeros((nS, nC))
cpi_dev = np.zeros((nS, nC))
x = np.arange(nC)
bar_width = 0.8 / nS
x_offsets = np.linspace(-bar_width*nS/2 + bar_width/2, bar_width*nS/2 - bar_width/2, nS)

# Calculate deviations
for i, c in enumerate(selected_countries):
    d0_gdp = 100 * scenarios_cleaned[c]['scenario0']['real_gdp_growth_quarterly']
    d0_cpi = 100 * scenarios_cleaned[c]['scenario0']['household_consumption_deflator_growth_quarterly']
    for s_idx, s in enumerate(scenario_indices[1:]):
        dX_gdp = 100 * scenarios_cleaned[c][f"scenario{s}"]['real_gdp_growth_quarterly']
        dX_cpi = 100 * scenarios_cleaned[c][f"scenario{s}"]['household_consumption_deflator_growth_quarterly']
        gdp_dev[s_idx, i] = np.mean(np.sum(dX_gdp[:8, :] - d0_gdp[:8, :], axis=0))
        cpi_dev[s_idx, i] = np.mean(np.sum(dX_cpi[:8, :] - d0_cpi[:8, :], axis=0))

fig, axs = plt.subplots(2, 1, figsize=(12, 7))

# Top tile   
ax = axs[0]
for s_idx in range(nS):
    ax.bar(x + x_offsets[s_idx], gdp_dev[s_idx], width=bar_width, color=scenario_colors[s_idx+1], edgecolor='black', linewidth=0.8)
#ax.set_title("2025–2026 Cumulated Quarterly Real GDP Growth", fontweight='bold')
axs[0].text(
    0.02, 0.15, 
    "Real GDP Growth", 
    transform=axs[0].transAxes, 
    fontsize=12, 
    fontweight='bold', 
    va='top', 
    ha='left',
    bbox=dict(
    facecolor='white',   
    edgecolor='black',  
    linewidth=0.8
    )
)

# Bottom tile   
ax = axs[1]
for s_idx in range(nS):
    ax.bar(x + x_offsets[s_idx], cpi_dev[s_idx], width=bar_width, color=scenario_colors[s_idx+1], edgecolor='black', linewidth=0.8)
#ax.set_title("2025–2026 Cumulated Quarterly CPI Inflation", fontweight='bold')
axs[1].text(
    0.02, 0.15, 
    "CPI Inflation", 
    transform=axs[1].transAxes, 
    fontsize=12, 
    fontweight='bold', 
    va='top', 
    ha='left',
    bbox=dict(
    facecolor='white',   
    edgecolor='black',  
    linewidth=0.8
    )
)

# Axis formatting
for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(0, linestyle='--', color=black)
    ax.set_xticks(x)
    ax.set_xticklabels(selected_countries)
    ax.set_ylabel(r"p.p. deviations from $\mathbf{Baseline}$")
    ax.set_xlim(x[0] - 0.9, x[-1] + 0.9)
    
# Common legend
patches = [mpatches.Patch(color=scenario_colors[i+1], label=scenario_labels[i+1])for i in range(nS)]
fig.legend(handles=patches, loc='lower center', ncol=nS, bbox_to_anchor=(0.5, -0.05), fontsize=12)

plt.tight_layout()
plt.show()

# Pdf, PNG, multi-page pdf
fig.savefig(os.path.join(output_folder, "2_gdp_inflation.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(output_folder, "2_gdp_inflation.png"), dpi=300, bbox_inches='tight')
pdf.savefig(fig, bbox_inches='tight')

# %% === PPI INFLATION BY COMPONENT === %%  

# Prepare components and colors
components = ['infl_ppi_cpu_quarterly', 'infl_ppi_dpl_quarterly', 'infl_ppi_exp_quarterly', 'infl_ppi_rsd_quarterly']
component_names = ['Cost-push', 'Demand-pull', 'Expectations', 'Residual']

# Colors inflation components
component_colors = np.array([blue, yellow, white, light_gray])

# Function to get inflation deviations and total
def get_inflation_data(country, scenario_index):
    # Baseline and scenario data
    sc0 = scenarios_cleaned[country]['scenario0']
    sc = scenarios_cleaned[country][f'scenario{scenario_index}']
    T = sc0[components[0]].shape[0] - 1
    # Quarters
    raw = np.asarray(sc0['quarters_num'][1:1+T], dtype=float)
    quarters_dates = pd.to_datetime(raw - 719529, unit='D')
    quarters = pd.PeriodIndex(quarters_dates, freq='Q')
    x = np.arange(T)
    # Component deviations (stacked bars) – vectorized
    comp_dev = np.column_stack([100 * (np.mean(sc[comp][1:1+T, :], axis=1) - np.mean(sc0[comp][1:1+T, :], axis=1))for comp in components])
    # Total inflation line
    total_infl = 100 * (np.mean(sc['infl_ppi_quarterly'][1:1+T, :], axis=1) - np.mean(sc0['infl_ppi_quarterly'][1:1+T, :], axis=1))
    
    return quarters, x, comp_dev, total_infl

fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharey='row')  # share y-axis row-wise

# Top-left tile
quarters, x, comp_dev, total_infl = get_inflation_data('USA', 1)
pos_bottom = np.zeros_like(x, dtype=float)
neg_bottom = np.zeros_like(x, dtype=float)
for j in range(comp_dev.shape[1]):
    vals = comp_dev[:, j]
    bottom = np.where(vals >= 0, pos_bottom, neg_bottom)
    axs[0, 0].bar(x, vals, bottom=bottom, color=component_colors[j], edgecolor=black)
    pos_bottom += np.where(vals >= 0, vals, 0)
    neg_bottom += np.where(vals < 0, vals, 0)
axs[0, 0].plot(x, total_infl, 'k-', linewidth=1.5)
axs[0, 0].set_title('USA', fontweight='bold')
axs[0, 0].set_ylabel("p.p. deviations from $\mathbf{Baseline}$")

# Top-right tile
quarters, x, comp_dev, total_infl = get_inflation_data('CAN', 1)
pos_bottom = np.zeros_like(x, dtype=float)
neg_bottom = np.zeros_like(x, dtype=float)
for j in range(comp_dev.shape[1]):
    vals = comp_dev[:, j]
    bottom = np.where(vals >= 0, pos_bottom, neg_bottom)
    axs[0, 1].bar(x, vals, bottom=bottom, color=component_colors[j], edgecolor=black)
    pos_bottom += np.where(vals >= 0, vals, 0)
    neg_bottom += np.where(vals < 0, vals, 0)
axs[0, 1].plot(x, total_infl, 'k-', linewidth=1.5)
axs[0, 1].set_title('CAN', fontweight='bold')
axs[0, 1].text(
    0.98, 0.98,
    scenario_labels[1],
    transform=axs[0, 1].transAxes,
    fontsize=12,
    fontweight='bold',
    color=blue,
    va='top',
    ha='right',
    bbox=dict(
    facecolor='white',   
    edgecolor='black',  
    linewidth=0.8
    )
)

# Bottom-left tile
quarters, x, comp_dev, total_infl = get_inflation_data('USA', 2)
pos_bottom = np.zeros_like(x, dtype=float)
neg_bottom = np.zeros_like(x, dtype=float)
for j in range(comp_dev.shape[1]):
    vals = comp_dev[:, j]
    bottom = np.where(vals >= 0, pos_bottom, neg_bottom)
    axs[1, 0].bar(x, vals, bottom=bottom, color=component_colors[j], edgecolor=black)
    pos_bottom += np.where(vals >= 0, vals, 0)
    neg_bottom += np.where(vals < 0, vals, 0)
axs[1, 0].plot(x, total_infl, 'k-', linewidth=1.5)
axs[1, 0].set_title('USA', fontweight='bold')
axs[1, 0].set_ylabel("p.p. deviations from $\mathbf{Baseline}$")

# Bottom-right tile
quarters, x, comp_dev, total_infl = get_inflation_data('CAN', 2)
pos_bottom = np.zeros_like(x, dtype=float)
neg_bottom = np.zeros_like(x, dtype=float)
for j in range(comp_dev.shape[1]):
    vals = comp_dev[:, j]
    bottom = np.where(vals >= 0, pos_bottom, neg_bottom)
    axs[1, 1].bar(x, vals, bottom=bottom, color=component_colors[j], edgecolor=black)
    pos_bottom += np.where(vals >= 0, vals, 0)
    neg_bottom += np.where(vals < 0, vals, 0)
axs[1, 1].plot(x, total_infl, 'k-', linewidth=1.5)
axs[1, 1].set_title('CAN', fontweight='bold')
axs[1, 1].text(
    0.98, 0.98,
    scenario_labels[2],
    transform=axs[1, 1].transAxes,
    fontsize=12,
    fontweight='bold',
    color=red,
    va='top',
    ha='right',
    bbox=dict(
    facecolor='white',   
    edgecolor='black',  
    linewidth=0.8
    )
)

# Axis formatting
for i in range(2):
    for j in range(2):
        ax = axs[i, j]
        ax.set_xticks(x)
        ax.set_xticklabels([f"Q{q.quarter}" for q in quarters], rotation=0, ha='center')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.axhline(0, linestyle='--', color=black, linewidth=0.75)
        
# Common legend
handles = [mpatches.Patch(color=component_colors[j], label=component_names[j]) for j in range(len(component_names))]
handles.append(mpatches.Patch(color=black, label='Total'))
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(handles), fontsize=12)

# Pdf, PNG, multi-page pdf
plt.tight_layout()
plt.show()
fig.savefig(os.path.join(output_folder, "3_inflation_comp.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(output_folder, "3_inflation_comp.png"), dpi=300, bbox_inches='tight')
pdf.savefig(fig, bbox_inches='tight')

# %% === PPI INFLATION BY SECTOR === %%
startYear = 2024
startQuarter = 4
manual_ylim = {
    "S1S0_USA": [-0.4, 0.4],
    "S1S0_CAN": [-1.1, 1.1],
    "S2S0_USA": [-0.4, 0.4],
    "S2S0_CAN": [-4.0, 4.0]
}

# Helper functions
def get_P(sc):
    return np.transpose(sc['deflator_ppi_fg'], (1, 2, 0))

def get_Ptot(sc):
    return np.transpose(sc['deflator_ppi_f'], (1, 0))

def pct_exact(x):
    """Quarter-to-quarter percent change"""
    if x.ndim == 3:
        return (x[:, :, 1:] / x[:, :, :-1] - 1) * 100
    else:
        return (x[:, 1:] / x[:, :-1] - 1) * 100

# Determine actual T_all based on available data (after pct_exact calculation)
sc0_sample = scenarios_cleaned[selected_countries[0]]['scenario0']
infl0_sample = pct_exact(get_P(sc0_sample))
T_all = infl0_sample.shape[2]

# Colors for sectors
cmap = matplotlib.colormaps["tab20"].resampled(len(industry_names))
sector_colors = np.array([cmap(i) for i in range(len(industry_names))])

# Fixed order of sectors (alphabetical)
sorted_labels = sorted(industry_names)
label_to_x = {label: i + 1 for i, label in enumerate(sorted_labels)}

output_pdf = os.path.join(output_folder, f'4_inflation_sect.pdf')
if os.path.exists(output_pdf):
    os.remove(output_pdf)

with PdfPages(output_pdf) as pdf_pages:
    for qi in range(T_all):
        qIdx = startQuarter + qi - 1
        year = startYear + qIdx // 4
        quarter = qIdx % 4 + 1
        quarter_label = f"{year} Q{quarter}"
        
        fig, axs = plt.subplots(2, len(selected_countries), figsize=(12, 10), sharex=True)
        
        # Quarter annotation (top right of entire figure)
        axs[0, -1].text(
            0.98, 0.98, quarter_label,
            transform=axs[0, -1].transAxes,
            fontsize=11, fontweight='bold',
            va='top', ha='right',
            bbox=dict(facecolor='white', edgecolor='black', linewidth=0.8)
        )
        
        # Loop over scenarios (rows)
        for s_idx, s in enumerate(scenario_indices[1:]):
            scenario_name = scenario_names_dev[s_idx]
            
            # Header annotation for each row
            header_text = 'Unilateral Tariffs' if scenario_name == 'S1S0' else 'Retaliation'
            header_color = 'blue' if scenario_name == 'S1S0' else 'red'
            axs[s_idx, -1].text(
                0.98, 0.88, header_text,
                transform=axs[s_idx, -1].transAxes,
                fontsize=11, fontweight='bold', color=header_color,
                va='top', ha='right',
                bbox=dict(facecolor='white', edgecolor='black', linewidth=0.8)
            )
            
            # Loop over countries (columns)
            for ci, c in enumerate(selected_countries):
                ax = axs[s_idx, ci]
                
                sc0 = scenarios_cleaned[c]['scenario0']
                scX = scenarios_cleaned[c][f'scenario{s}']
                
                infl0_s = pct_exact(get_P(sc0))
                inflX_s = pct_exact(get_P(scX))
                deltag = inflX_s - infl0_s
                
                infl0_f = pct_exact(get_Ptot(sc0))
                inflX_f = pct_exact(get_Ptot(scX))
                deltaf = inflX_f - infl0_f
                
                sector_mean = np.mean(deltag[:, :, qi], axis=0)
                
                # Plot markers in fixed alphabetical order
                for label in sorted_labels:
                    sec_idx = industry_names.index(label)
                    x_pos = label_to_x[label]
                    m = sector_mean[sec_idx]
                    ax.plot(x_pos, m, 'o', markersize=6, linewidth=1.5,
                           color=sector_colors[sec_idx], zorder=3)
                
                # Set x-axis
                xticks = np.arange(1, len(sorted_labels) + 1)
                ax.set_xticks(xticks)
                ax.set_xticklabels(sorted_labels, fontsize=6, rotation=90)
                
                # Reference lines
                ax.axhline(np.mean(deltaf[:, 0]), linestyle='--', color='k', linewidth=2)
                ax.axhline(np.mean(deltaf[:, qi]), linestyle=':', color='k', linewidth=2)
                
                # Axis formatting
                ax.set_ylabel(r'p.p. deviations from $\mathbf{Baseline}$', fontsize=6)
                ax.set_title(c, fontweight='bold', fontsize=7)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticks_position('bottom')
                
                # Set y-axis limits
                key = f"{scenario_name}_{c}"
                if key in manual_ylim:
                    lim = manual_ylim[key]
                else:
                    q10 = np.quantile(deltag[:, :, qi], 0.10)
                    q90 = np.quantile(deltag[:, :, qi], 0.90)
                    pad = 0.6 * (q90 - q10)
                    lim = [q10 - pad, q90 + pad]
                ax.set_ylim(lim)
                yt = np.linspace(lim[0], lim[1], 5)
                ax.set_yticks(yt)
                ax.set_yticklabels([f"{v:.1f}" for v in yt])
        
        fig.tight_layout()
        pdf_pages.savefig(fig, bbox_inches='tight')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

# %% === PPI INFLATION BY SECTOR (GIF) === %%

startYear = 2024
startQuarter = 4
manual_ylim = {
    "S1S0_USA": [-0.4, 0.4],
    "S1S0_CAN": [-1.1, 1.1],
    "S2S0_USA": [-0.4, 0.4],
    "S2S0_CAN": [-4.0, 4.0]
}

# Helper functions
def get_P(sc):
    return np.transpose(sc['deflator_ppi_fg'], (1, 2, 0))
def get_Ptot(sc):
    return np.transpose(sc['deflator_ppi_f'], (1, 0))
def pct_exact(x):
    """Quarter-to-quarter percent change"""
    if x.ndim == 3:
        return (x[:, :, 1:] / x[:, :, :-1] - 1) * 100
    else:
        return (x[:, 1:] / x[:, :-1] - 1) * 100

# Colors for sectors
cmap = matplotlib.colormaps["tab20"].resampled(len(industry_names))
sector_colors = np.array([cmap(i) for i in range(len(industry_names))])

# Fixed order of sectors (alphabetical)
sorted_labels = sorted(industry_names)
label_to_x = {label: i + 1 for i, label in enumerate(sorted_labels)}

# Determine actual T_all based on available data (after pct_exact calculation)
sc0_sample = scenarios_cleaned[selected_countries[0]]['scenario0']
infl0_sample = pct_exact(get_P(sc0_sample))
T_all = infl0_sample.shape[2]

fig, axs = plt.subplots(2, len(selected_countries), figsize=(12, 10), sharex=True, dpi=150)

def update(qi):
    qIdx = startQuarter + qi - 1
    year = startYear + qIdx // 4
    quarter = qIdx % 4 + 1
    quarter_label = f"{year} Q{quarter}"
    
    # Clear all axes
    for i in range(2):
        for j in range(len(selected_countries)):
            axs[i, j].clear()
    
    # Quarter annotation (top right of entire figure)
    axs[0, -1].text(
        0.98, 0.98,
        quarter_label,
        transform=axs[0, -1].transAxes,
        fontsize=11,
        fontweight='bold',
        va='top',
        ha='right',
        bbox=dict(facecolor='white', edgecolor='black', linewidth=0.8)
    )
    
    # Loop over scenarios (rows)
    for s_idx, s in enumerate(scenario_indices[1:]):
        scenario_name = scenario_names_dev[s_idx]
        
        # Header annotation for each row
        header_text = 'Unilateral Tariffs' if scenario_name == 'S1S0' else 'Retaliation'
        header_color = 'blue' if scenario_name == 'S1S0' else 'red'
        axs[s_idx, -1].text(
            0.98, 0.88,
            header_text,
            transform=axs[s_idx, -1].transAxes,
            fontsize=11,
            fontweight='bold',
            color=header_color,
            va='top',
            ha='right',
            bbox=dict(facecolor='white', edgecolor='black', linewidth=0.8)
        )
        
        # Loop over countries (columns)
        for ci, c in enumerate(selected_countries):
            ax = axs[s_idx, ci]
            sc0 = scenarios_cleaned[c]['scenario0']
            scX = scenarios_cleaned[c][f'scenario{s}']
            
            infl0_s = pct_exact(get_P(sc0))
            inflX_s = pct_exact(get_P(scX))
            deltag = inflX_s - infl0_s
            
            infl0_f = pct_exact(get_Ptot(sc0))
            inflX_f = pct_exact(get_Ptot(scX))
            deltaf = inflX_f - infl0_f
            
            sector_mean = np.mean(deltag[:, :, qi], axis=0)
            
            # Plot markers in fixed alphabetical order
            for label in sorted_labels:
                sec_idx = industry_names.index(label)
                x_pos = label_to_x[label]
                m = sector_mean[sec_idx]
                ax.plot(x_pos, m, 'o', markersize=6, linewidth=1.5, color=sector_colors[sec_idx], zorder=3)
            
            # Set x-axis - show labels on BOTH rows with rotation
            xticks = np.arange(1, len(sorted_labels) + 1)
            ax.set_xticks(xticks)
            ax.set_xticklabels(sorted_labels, fontsize=6, rotation=90)
            
            # Reference lines
            ax.axhline(np.mean(deltaf[:, 0]), linestyle='--', color='k', linewidth=2)
            ax.axhline(np.mean(deltaf[:, qi]), linestyle=':', color='k', linewidth=2)
            
            # Axis formatting
            ax.set_ylabel(r'p.p. deviations from $\mathbf{Baseline}$', fontsize=6)
            ax.set_title(c, fontweight='bold', fontsize=7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            
            # Set y-axis limits
            key = f"{scenario_name}_{c}"
            if key in manual_ylim:
                lim = manual_ylim[key]
            else:
                q10 = np.quantile(deltag[:, :, qi], 0.10)
                q90 = np.quantile(deltag[:, :, qi], 0.90)
                pad = 0.6 * (q90 - q10)
                lim = [q10 - pad, q90 + pad]
            ax.set_ylim(lim)
            yt = np.linspace(lim[0], lim[1], 5)
            ax.set_yticks(yt)
            ax.set_yticklabels([f"{v:.1f}" for v in yt])
    
    fig.tight_layout()

# Create animation
ani = FuncAnimation(fig, update, frames=T_all, repeat=True)

# Save as GIF
output_gif = os.path.join(output_folder, f'4_inflation_sect.gif')
ani.save(output_gif, writer=PillowWriter(fps=1, bitrate=1800))

# Display GIF embedded in notebook (works in both browser and VSCode)
print("\n=== PPI Inflation by Sector (Animated GIF) ===\n")
with open(output_gif, 'rb') as f:
    gif_data = f.read()
    gif_base64 = base64.b64encode(gif_data).decode('ascii')
    display(HTML(f'''
        <div style="text-align: center;">
            <h3>PPI Inflation by Sector (Animated GIF)</h3>
            <img src="data:image/gif;base64,{gif_base64}" style="max-width: 100%; height: auto;">
        </div>
    '''))

plt.close(fig)

# %% === PPI INFLATION BY SECTOR (INTERACTIVE PLOTLY) === %%

startYear = 2024
startQuarter = 4
manual_ylim = {
    "S1S0_USA": [-0.4, 0.4],
    "S1S0_CAN": [-1.1, 1.1],
    "S2S0_USA": [-0.4, 0.4],
    "S2S0_CAN": [-4.0, 4.0]
}

# Helper functions
def get_P(sc):
    return np.transpose(sc['deflator_ppi_fg'], (1, 2, 0))
def get_Ptot(sc):
    return np.transpose(sc['deflator_ppi_f'], (1, 0))
def pct_exact(x):
    """Quarter-to-quarter percent change"""
    if x.ndim == 3:
        return (x[:, :, 1:] / x[:, :, :-1] - 1) * 100
    else:
        return (x[:, 1:] / x[:, :-1] - 1) * 100

# Colors for sectors
cmap = matplotlib.colormaps["tab20"].resampled(len(industry_names))
sector_colors_mpl = np.array([cmap(i) for i in range(len(industry_names))])
# Convert matplotlib colors to hex for Plotly
sector_colors_hex = {industry_names[i]: matplotlib.colors.rgb2hex(sector_colors_mpl[i]) 
                      for i in range(len(industry_names))}

# Fixed order of sectors (alphabetical)
sorted_labels = sorted(industry_names)
label_to_x = {label: i + 1 for i, label in enumerate(sorted_labels)}

# Determine actual T_all based on available data (after pct_exact calculation)
sc0_sample = scenarios_cleaned[selected_countries[0]]['scenario0']
infl0_sample = pct_exact(get_P(sc0_sample))
T_all = infl0_sample.shape[2]  # Get actual number of time periods after pct calculation

row_titles = ['<b style="color:blue">Unilateral Tariffs</b>', '<b style="color:red">Retaliation</b>']
fig_plotly = make_subplots(
    rows=2, cols=len(selected_countries),
    subplot_titles=[f'{selected_countries[0]}', f'{selected_countries[1]}', '', ''],
    shared_xaxes='columns',  # Changed to share x-axes by columns
    vertical_spacing=0.15,
    horizontal_spacing=0.1,
    row_titles=row_titles
)

# Store all traces for slider
all_traces = []

# Loop over quarters
for qi in range(T_all):
    qIdx = startQuarter + qi - 1
    year = startYear + qIdx // 4
    quarter = qIdx % 4 + 1
    quarter_label = f"{year} Q{quarter}"
    
    quarter_traces = []
    
    # Loop over scenarios (rows)
    for s_idx, s in enumerate(scenario_indices[1:]):
        scenario_name = scenario_names_dev[s_idx]
        
        # Loop over countries (columns)
        for ci, c in enumerate(selected_countries):
            sc0 = scenarios_cleaned[c]['scenario0']
            scX = scenarios_cleaned[c][f'scenario{s}']
            
            infl0_s = pct_exact(get_P(sc0))
            inflX_s = pct_exact(get_P(scX))
            deltag = inflX_s - infl0_s
            
            infl0_f = pct_exact(get_Ptot(sc0))
            inflX_f = pct_exact(get_Ptot(scX))
            deltaf = inflX_f - infl0_f
            
            sector_mean = np.mean(deltag[:, :, qi], axis=0)
            
            # Prepare data in fixed alphabetical order
            x_positions = []
            y_values = []
            colors = []
            hover_texts = []
            
            for label in sorted_labels:
                sec_idx = industry_names.index(label)
                x_pos = label_to_x[label]
                m = sector_mean[sec_idx]
                
                x_positions.append(x_pos)
                y_values.append(m)
                colors.append(sector_colors_hex[label])
                hover_texts.append(f"{label}<br>Deviation: {m:.2f} p.p.")
            
            # Add scatter trace for this scenario/country/quarter
            trace = go.Scatter(
                x=x_positions,
                y=y_values,
                mode='markers',
                marker=dict(
                    color=colors,
                    size=10,
                    line=dict(width=1, color='white')
                ),
                text=hover_texts,
                hovertemplate='%{text}<extra></extra>',
                name=f"{scenario_name} {c} {quarter_label}",
                visible=(qi == 0),  # Only first quarter visible initially
                showlegend=False
            )
            
            fig_plotly.add_trace(trace, row=s_idx+1, col=ci+1)
            quarter_traces.append(trace)
            
            # Add reference lines (horizontal)
            ref_line_0 = go.Scatter(
                x=[0.5, len(sorted_labels) + 0.5],
                y=[np.mean(deltaf[:, 0]), np.mean(deltaf[:, 0])],
                mode='lines',
                line=dict(color='black', dash='dash', width=2),
                showlegend=False,
                visible=(qi == 0),
                hoverinfo='skip'
            )
            fig_plotly.add_trace(ref_line_0, row=s_idx+1, col=ci+1)
            quarter_traces.append(ref_line_0)
            
            ref_line_qi = go.Scatter(
                x=[0.5, len(sorted_labels) + 0.5],
                y=[np.mean(deltaf[:, qi]), np.mean(deltaf[:, qi])],
                mode='lines',
                line=dict(color='black', dash='dot', width=2),
                showlegend=False,
                visible=(qi == 0),
                hoverinfo='skip'
            )
            fig_plotly.add_trace(ref_line_qi, row=s_idx+1, col=ci+1)
            quarter_traces.append(ref_line_qi)
    
    all_traces.append(quarter_traces)

# Create slider steps
steps = []
for qi in range(T_all):
    qIdx = startQuarter + qi - 1
    year = startYear + qIdx // 4
    quarter = qIdx % 4 + 1
    quarter_label = f"{year} Q{quarter}"
    
    # Calculate visibility for this quarter
    visible = []
    for q_idx in range(T_all):
        for trace in all_traces[q_idx]:
            visible.append(q_idx == qi)
    
    step = dict(
        method="update",
        args=[{"visible": visible}],
        label=quarter_label
    )
    steps.append(step)

sliders = [dict(
    active=0,
    yanchor="top",
    y=-0.08,
    xanchor="left",
    currentvalue=dict(
        prefix="<b>Quarter:</b> ",
        visible=True,
        xanchor="center"
    ),
    pad=dict(b=10, t=10),
    len=0.9,
    x=0.05,
    steps=steps
)]

# Update layout
fig_plotly.update_layout(
    title=dict(
        text="<b>Inflation (PPI) by Sector</b>",
        x=0.5,
        xanchor='center'
    ),
    sliders=sliders,
    height=900,
    template='plotly_white',
    hovermode='closest'
)

# Update x-axes and y-axes for all subplots
for s_idx in range(2):  # scenarios
    for ci in range(len(selected_countries)):  # selected_countries
        # Update x-axes (industries) - show labels on BOTH rows with rotation
        fig_plotly.update_xaxes(
            tickmode='array',
            tickvals=list(range(1, len(sorted_labels) + 1)),
            ticktext=sorted_labels,
            tickangle=-90,
            showticklabels=True,  # Explicitly show labels on all subplots
            row=s_idx+1, col=ci+1
        )
        
        # Set y-axis limits
        scenario_name = scenario_names_dev[s_idx]
        key = f"{scenario_name}_{selected_countries[ci]}"
        if key in manual_ylim:
            y_range = manual_ylim[key]
        else:
            # Calculate from data
            all_deltas = []
            for qi in range(T_all):
                sc0 = scenarios_cleaned[selected_countries[ci]]['scenario0']
                scX = scenarios_cleaned[selected_countries[ci]][f'scenario{scenario_indices[1:][s_idx]}']
                infl0_s = pct_exact(get_P(sc0))
                inflX_s = pct_exact(get_P(scX))
                deltag = inflX_s - infl0_s
                all_deltas.append(deltag[:, :, qi])
            all_deltas = np.concatenate(all_deltas)
            q10 = np.quantile(all_deltas, 0.10)
            q90 = np.quantile(all_deltas, 0.90)
            pad = 0.6 * (q90 - q10)
            y_range = [q10 - pad, q90 + pad]
        
        fig_plotly.update_yaxes(
            title_text='p.p. deviations from <b>Baseline</b>' if ci == 0 else '',
            range=y_range,
            row=s_idx+1, col=ci+1
        )

# Save standalone HTML
output_html = os.path.join(output_folder, f'4_inflation_sect.html')
fig_plotly.write_html(output_html, include_plotlyjs='cdn')

# Display markdown with link to interactive version
display(HTML('''
<p align="center">
  <strong>PPI Inflation by Sector</strong> 
  <a href="https://gist.githack.com/AlexCrescentini/3d21c1344e2ec31baf1872bc9bad1812/raw/4_inflation_sect.html" target="_blank">(CLICK for interactive version)</a>
</p>
'''))

# %% Close multi-page pdf

pdf.close()
