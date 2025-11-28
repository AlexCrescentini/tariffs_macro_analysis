import os
import geopandas as gpd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set working directory to the location of this script
os.chdir(os.path.dirname(__file__))

# Updated list: OECD-30 calibrated countries
geos_calibrated = ['AUT','AUS','BEL','CAN','CZE','DEU','DNK','EST','GRC','ESP','FIN',
    'FRA','HUN','IRL','ITA','JPN','KOR','LTU','LUX','LVA','MEX','NLD',
    'NOR','POL','PRT','SWE','SVN','SVK','GBR','USA']

# Load world map shapefile, applies some corrections, save index of calibrated countries
world = gpd.read_file(r'./ne_110m_admin_0_countries.shp')
world = world[world['NAME'] != 'Antarctica']
world.loc[world['NAME'] == 'France', 'ISO_A3'] = 'FRA'
world.loc[world['NAME'] == 'Norway', 'ISO_A3'] = 'NOR'
highlight_calibrated_idx = world['ISO_A3'].isin(geos_calibrated)

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
world[~highlight_calibrated_idx].plot(ax=ax, color=[0.9, 0.9, 0.9], edgecolor='black')
world[highlight_calibrated_idx].plot(ax=ax, color=[0.2, 0.4, 0.8], edgecolor='black')
legend_handles = [mpatches.Patch(color=[0.2, 0.4, 0.8], label='OECD-30'), mpatches.Patch(color=[0.9, 0.9, 0.9], label='RoW')]
ax.legend(handles=legend_handles, loc='lower left', fontsize=12)
ax.set_axis_off()
plt.show()
fig.savefig('map_calibration.png', format='png', bbox_inches='tight')