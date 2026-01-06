import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import os
import geopandas as gpd
from tqdm.auto import tqdm
from matplotlib.gridspec import GridSpec
from shapely.geometry import Point

# --- Global Plot Settings ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['axes.labelweight'] = 'bold'

# --- Paths ---
base_path = '/content/drive/MyDrive/mukund_iitr/PHD/PHD/MY PHD WORK/objective 3/CropLizer/Analysis/Results/Benefits Analysis'
file_path = os.path.join(base_path, 'Benefit Analysis.xlsx')
shp_path = os.path.join(base_path, '1971_2016_2024 districts aligned/shapefile/state2016_GCS.shp')
# Converted path for GAUL file in Colab
gaul_path = '/content/drive/MyDrive/mukund_iitr/PHD/PHD/MY PHD WORK/objective 3/CropLizer/Analysis/Results/Model Accuracy/GAUL_2024_L1.zip'

# --- Load Data ---
print("Loading data...")
df = pd.read_excel(file_path)
gdf_india = gpd.read_file(shp_path)

# --- 1. Shapefile Integration & Preparation ---
print("Loading FAO GAUL data...")
try:
    # Load GAUL L1 data
    gdf_gaul = gpd.read_file(gaul_path)

    # Filter for Bangladesh and Nepal
    target_countries = ['Bangladesh', 'Nepal']

    # Check for correct column name based on user feedback (gaul0_name)
    if 'gaul0_name' in gdf_gaul.columns:
        neighbors = gdf_gaul[gdf_gaul['gaul0_name'].isin(target_countries)].copy()
        neighbors['Plot_Category'] = neighbors['gaul0_name']
    elif 'ADM0_NAME' in gdf_gaul.columns:
        neighbors = gdf_gaul[gdf_gaul['ADM0_NAME'].isin(target_countries)].copy()
        neighbors['Plot_Category'] = neighbors['ADM0_NAME']
    else:
        # Fallback if column names differ, print available columns to help debug
        print(f"Warning: Country name column not found. Available columns: {gdf_gaul.columns}")
        neighbors = gpd.GeoDataFrame(columns=['geometry', 'Plot_Category'])

except Exception as e:
    print(f"Error loading GAUL data: {e}")
    neighbors = gpd.GeoDataFrame(columns=['geometry', 'Plot_Category'])

# Reproject neighbors to match India's CRS if necessary
if not neighbors.empty and gdf_india.crs != neighbors.crs:
    neighbors = neighbors.to_crs(gdf_india.crs)

# Standardize columns for merging
gdf_india['Plot_Category'] = 'India'

# Keep only geometry and category for the merge to avoid schema conflicts
gdf_india_clean = gdf_india[['geometry', 'Plot_Category']]
neighbors_clean = neighbors[['geometry', 'Plot_Category']]

# Create the new combined shapefile
combined_gdf = pd.concat([gdf_india_clean, neighbors_clean], ignore_index=True)

# Define Color Map
country_colors = {
    'India': '#f0f1fa',
    'Bangladesh': '#f2cece',
    'Nepal': '#d0f2ce'
}

# --- 2. Data Processing ---
target_col = 'Current_Reliability'
required_cols = [target_col, 'LAT', 'LONG', 'CRLPARHA']
df_clean = df[required_cols].replace([np.inf, -np.inf], np.nan).dropna()
total_points = len(df_clean)

# Create Geometry for points to check intersection
points_geom = [Point(xy) for xy in zip(df_clean.LONG, df_clean.LAT)]
gdf_points = gpd.GeoDataFrame(df_clean, geometry=points_geom, crs=combined_gdf.crs)

# Spatial Join to identify which polygons contain points
# 'inner' join keeps only polygons that have points, we use this to get indices
if not combined_gdf.empty:
    polys_with_points = gpd.sjoin(combined_gdf, gdf_points, how='inner', predicate='intersects')
    indices_with_points = polys_with_points.index.unique()
else:
    indices_with_points = []

# --- Pre-calculate Size Bins (Terciles) for Consistency ---
s_var = df_clean['CRLPARHA']
s_min, s_max = s_var.min(), s_var.max()

q33 = np.percentile(s_var, 33.333)
q66 = np.percentile(s_var, 66.666)

bin_labels = [
    f'T1 (< {q33:.1f})',
    f'T2 ({q33:.1f}-{q66:.1f})',
    f'T3 (> {q66:.1f})'
]

bins = [s_min - 0.1, q33, q66, s_max + 0.1]
df_clean['Size_Class'] = pd.cut(df_clean['CRLPARHA'], bins=bins, labels=bin_labels)

# --- Plotting Configuration ---
mm_to_inch = 1 / 25.4
fig_width = 180 * mm_to_inch
fig_height = 180 * mm_to_inch

fig = plt.figure(figsize=(fig_width, fig_height))

gs = GridSpec(2, 2, figure=fig,
              width_ratios=[4, 1.2],
              height_ratios=[1.2, 4],
              wspace=0.0, hspace=0.0)

ax_lon_prof = fig.add_subplot(gs[0, 0])
ax_map = fig.add_subplot(gs[1, 0])
ax_lat_prof = fig.add_subplot(gs[1, 1])

# Determine Aesthetics
vmin = np.percentile(df_clean[target_col], 2.5)
vmax = np.percentile(df_clean[target_col], 97.5)
map_cmap = 'turbo'
sizes = 10 + (s_var - s_min) / (s_max - s_min) * 150

# Extents
min_lon, max_lon = df_clean['LONG'].min(), df_clean['LONG'].max()
min_lat, max_lat = df_clean['LAT'].min(), df_clean['LAT'].max()

row_min_lon = df_clean.loc[df_clean['LONG'].idxmin()]
row_max_lon = df_clean.loc[df_clean['LONG'].idxmax()]
row_min_lat = df_clean.loc[df_clean['LAT'].idxmin()]
row_max_lat = df_clean.loc[df_clean['LAT'].idxmax()]

def calculate_rolling_stats(x, y, window_frac=0.15):
    sort_idx = np.argsort(x)
    x_sorted = x.iloc[sort_idx].reset_index(drop=True)
    y_sorted = y.iloc[sort_idx].reset_index(drop=True)
    w = int(len(x) * window_frac)
    if w < 5: w = 5
    roller = y_sorted.rolling(window=w, center=True, min_periods=int(w/2))
    y_median = roller.median()
    return x_sorted, y_median

# --- 1. Main Spatial Map (Bottom Left) ---
ax_map.set_aspect('equal', adjustable='datalim')

# A. Create the Buffer/Dissolve Layer
# Buffer by 0.1, dissolve all, plot with thick black line
if not combined_gdf.empty:
    buffered_boundary = combined_gdf.copy()
    # Note: Geometry is likely geographic (lat/lon). 0.01 is in degrees.
    buffered_boundary['geometry'] = buffered_boundary.geometry.buffer(0.01)
    dissolved_boundary = buffered_boundary.dissolve()
    dissolved_boundary.plot(ax=ax_map, facecolor='none', edgecolor='black', linewidth=1, zorder=3)

# B. Plot the Polygons with Color Coding
# Iterate to handle colors and line widths individually based on point containment
for idx, row in tqdm(combined_gdf.iterrows(), total=len(combined_gdf), desc="Plotting Map Polygons"):
    # Determine face color
    f_color = country_colors.get(row['Plot_Category'], '#e6e8fa')

    # Determine edge width and color
    if idx in indices_with_points:
        e_color = 'black'
        l_width = 0.5
        z_ord = 3 # Bring features with data to front
    else:
        e_color = 'grey'
        l_width = 0.5
        z_ord = 2

    # Plot single feature
    gpd.GeoSeries(row.geometry).plot(
        ax=ax_map,
        facecolor=f_color,
        edgecolor=e_color,
        linewidth=l_width,
        alpha=1,
        zorder=z_ord
    )

# C. Plot Scatter Points
sc = ax_map.scatter(df_clean['LONG'], df_clean['LAT'], c=df_clean[target_col], s=sizes,
                    cmap=map_cmap, vmin=vmin, vmax=vmax, alpha=0.9, edgecolors='none', zorder=4)

# D. Projection rays
current_xlim = ax_map.get_xlim()
current_ylim = ax_map.get_ylim()

ax_map.plot([row_min_lon['LONG'], row_min_lon['LONG']], [row_min_lon['LAT'], current_ylim[1]],
            color='black', linestyle=':', linewidth=1.2, alpha=0.7, zorder = 5)
ax_map.plot([row_max_lon['LONG'], row_max_lon['LONG']], [row_max_lon['LAT'], current_ylim[1]],
            color='black', linestyle=':', linewidth=1.2, alpha=0.7, zorder = 5)
ax_map.plot([row_min_lat['LONG'], current_xlim[1]+10], [row_min_lat['LAT'], row_min_lat['LAT']],
            color='black', linestyle=':', linewidth=1.2, alpha=0.7, zorder = 5)
ax_map.plot([row_max_lat['LONG'], current_xlim[1]+10], [row_max_lat['LAT'], row_max_lat['LAT']],
            color='black', linestyle=':', linewidth=1.2, alpha=0.7, zorder = 5)
ax_map.text(0.01, 0.95, f"a",
                 transform=ax_map.transAxes, fontsize=12, color='black', fontweight='bold', va='bottom')

ax_map.set_xlim(current_xlim)
ax_map.set_ylim(current_ylim)
ax_map.set_xlabel('Longitude', fontweight='bold', fontsize=12)
ax_map.set_ylabel('Latitude', fontweight='bold', fontsize=12)
ax_map.grid(True, linestyle='--', alpha=0.3)

# E. Legends
# 1. Size Legend
representative_sizes = [
    (bins[0] + bins[1])/2 if bins[0] > 0 else bins[1]/2,
    (bins[1] + bins[2])/2,
    (bins[2] + bins[3])/2
]
labels = bin_labels
size_handles = []
custom_pt_sizes = [20, 60, 100]
for i, (s, label) in enumerate(zip(representative_sizes, labels)):
    pt_size = custom_pt_sizes[i]
    size_handles.append(mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                                      markersize=np.sqrt(pt_size), label=label, alpha=0.6))

# Create size legend first and add it as an artist so it isn't overwritten
leg_size = ax_map.legend(handles=size_handles, loc='lower left', ncol=1, frameon=False,
                    fontsize=12, title='Area (ha)', title_fontsize=12,
                    bbox_to_anchor=(-0.05, 0.02)) # Explicit bbox to ensure visibility
leg_size.get_title().set_weight('bold')
leg_size.get_title().set_ha('right')
leg_size.get_title().set_position((-20, 0))
ax_map.add_artist(leg_size)

# 2. Country Legend
legend_patches = [
    mpatches.Patch(facecolor=country_colors['India'], edgecolor='grey', label='India'),
    mpatches.Patch(facecolor=country_colors['Bangladesh'], edgecolor='grey', label='Bangladesh'),
    mpatches.Patch(facecolor=country_colors['Nepal'], edgecolor='grey', label='Nepal')
]
# This call becomes the 'main' legend of the axes
ax_map.legend(handles=legend_patches, loc='upper left', frameon=False, fontsize=12,
                    bbox_to_anchor=(0.57, 0.45))


# --- Define Binned Colormap for Profiles ---
try:
    cmap_binned = plt.colormaps['Blues'].resampled(5)
except:
    cmap_binned = plt.cm.get_cmap('Blues', 5)

# --- 2. Longitude Profile (Top Left) ---
hb1 = ax_lon_prof.hexbin(df_clean['LONG'], df_clean[target_col], gridsize=(20,30),
                         cmap=cmap_binned, mincnt=1, linewidths=0, edgecolors='white', alpha=0.9)

lx, ly_med = calculate_rolling_stats(df_clean['LONG'], df_clean[target_col])
ax_lon_prof.plot(lx, ly_med, color='red', linewidth=1.5, linestyle='-', label='Median')

mean_of_median_lon = ly_med.mean()
pct_above_lon = (df_clean[target_col] > mean_of_median_lon).sum() / total_points * 100
pct_below_lon = (df_clean[target_col] < mean_of_median_lon).sum() / total_points * 100

ax_lon_prof.axhline(mean_of_median_lon, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
ax_lon_prof.text(0.01, 0.85, f"c",
                 transform=ax_lon_prof.transAxes, fontsize=12, color='black', fontweight='bold', va='bottom')
ax_lon_prof.text(0.02, (mean_of_median_lon - 84.5)/5.5 + 0.05, f"{pct_above_lon:.1f}%",
                 transform=ax_lon_prof.transAxes, fontsize=12, color='black', fontweight='bold', va='bottom')
ax_lon_prof.text(0.02, (mean_of_median_lon - 84.5)/5.5 - 0.05, f"{pct_below_lon:.1f}%",
                 transform=ax_lon_prof.transAxes, fontsize=12, color='black', fontweight='bold', va='top')

ax_lon_prof.axvline(min_lon, color='black', linestyle=':', alpha=0.7, linewidth=1.2)
ax_lon_prof.axvline(max_lon, color='black', linestyle=':', alpha=0.7, linewidth=1.2)

ax_lon_prof.set_ylabel('Median\nReliability', fontweight='bold', fontsize=12)
ax_lon_prof.tick_params(labelbottom=False)
ax_lon_prof.grid(True, linestyle='--', alpha=0.3)
ax_lon_prof.yaxis.set_major_locator(ticker.FixedLocator([85, 87, 89]))
ax_lon_prof.set_ylim(84.5, 90)

# --- 3. Latitude Profile (Bottom Right) ---
hb2 = ax_lat_prof.hexbin(df_clean[target_col], df_clean['LAT'], gridsize=(45,10),
                         cmap=cmap_binned, mincnt=1, linewidths=0, edgecolors='white', alpha=0.9)

lax, lay_med = calculate_rolling_stats(df_clean['LAT'], df_clean[target_col])
ax_lat_prof.plot(lay_med, lax, color='red', linewidth=1.5, linestyle='-')

mean_of_median_lat = lay_med.mean()
pct_above_lat = (df_clean[target_col] > mean_of_median_lat).sum() / total_points * 100
pct_below_lat = (df_clean[target_col] < mean_of_median_lat).sum() / total_points * 100

ax_lat_prof.axvline(mean_of_median_lat, color='black', linestyle='--', linewidth=1.5, alpha=0.8)

ax_lat_prof.text((mean_of_median_lat - 84)/6.0 - 0.03, 0.95, f"{pct_above_lat:.1f}%",
                 transform=ax_lat_prof.transAxes, fontsize=12, color='black', fontweight='bold', ha='left', va='top')
ax_lat_prof.text((mean_of_median_lat - 84)/6.0 - 0.13, 0.95, f"{pct_below_lat:.1f}%",
                 transform=ax_lat_prof.transAxes, fontsize=12, color='black', fontweight='bold', ha='right', va='top')
ax_lat_prof.text(0.015, 0.95, f"d",
                 transform=ax_lat_prof.transAxes, fontsize=12, color='black', fontweight='bold', va='bottom')

ax_lat_prof.text(0.5, 1.3, f"Total\nPoints\n{total_points:,.0f}", transform=ax_lat_prof.transAxes,
                 fontsize=12, color='black', fontweight='bold', ha='center', va='top')

ax_lat_prof.axhline(min_lat, color='black', linestyle=':', alpha=0.7, linewidth=1.2)
ax_lat_prof.axhline(max_lat, color='black', linestyle=':', alpha=0.7, linewidth=1.2)

ax_lat_prof.set_xlabel('Median\nReliability', fontweight='bold', fontsize=12)
ax_lat_prof.xaxis.tick_top()
ax_lat_prof.xaxis.set_label_position('top')
ax_lat_prof.tick_params(labelleft=False, left=False)
ax_lat_prof.grid(True, linestyle='--', alpha=0.3)
ax_lat_prof.xaxis.set_major_locator(ticker.FixedLocator([85, 87, 89]))
ax_lat_prof.set_xlim(84, 90.5)

# --- 4. Boxplot of Size Class vs Reliability (Inset Inside Map) ---
ax_ins = ax_map.inset_axes([0.735, 0.795, 0.265, 0.205])

box_data = []
for i, label in enumerate(bin_labels):
    subset = df_clean[df_clean['Size_Class'] == label][target_col].values
    if len(subset) > 0:
        box_data.append(subset)
    else:
        box_data.append([])

bp = ax_ins.boxplot(box_data, positions=range(1, len(bin_labels) + 1), patch_artist=True, widths=0.5,
                    flierprops=dict(marker='o', markerfacecolor='gray', markersize=0.5, alpha=0.5, markeredgecolor='none'))

for patch in bp['boxes']:
    patch.set_facecolor("grey")
    patch.set_alpha(0.6)
    patch.set_edgecolor('black')
    patch.set_linewidth(1)

for element in ['whiskers', 'caps', 'medians']:
    plt.setp(bp[element], color='black', linewidth=1)
plt.setp(bp['medians'], color='black', linewidth=1.0)

ax_ins.set_ylim(75, 98)
ax_ins.grid(True, linestyle='--', alpha=0.3)
ax_ins.set_facecolor('white')
ax_ins.patch.set_alpha(0.85)

ax_ins.set_xticks(range(1, len(bin_labels) + 1))
ax_ins.set_xticklabels(["T1","T2","T3"], rotation=0, ha='center', fontsize=12)
ax_ins.set_ylabel('Reliability', fontweight='bold', fontsize=12)
ax_ins.tick_params(axis='y', labelsize=12)
ax_ins.tick_params(axis='x', labelsize=12)
ax_ins.yaxis.set_major_locator(ticker.FixedLocator([80, 86, 92]))
ax_ins.text(0.74, 0.95, f"b",
                 transform=ax_map.transAxes, fontsize=12, color='black', fontweight='bold', va='bottom')

# --- 5. Colorbar and Legends ---
cbar_ax_lat = ax_lat_prof.inset_axes([0.05, 0.08, 0.9, 0.02])
cbar_lat = fig.colorbar(hb2, cax=cbar_ax_lat, orientation='horizontal')
cbar_lat.set_label('Count', fontsize=12, labelpad=-40, ha='right', x=0.4)
cbar_lat.ax.tick_params(labelsize=12)
cbar_lat.ax.patch.set_facecolor('white')
tick_labels_c = [500,1250]
cbar_lat.ax.xaxis.set_major_locator(ticker.FixedLocator(tick_labels_c))
cbar_lat.ax.set_xticklabels(tick_labels_c, fontsize=12)

cax = ax_map.inset_axes([0.43, 0.08, 0.3, 0.02])
cbar = fig.colorbar(sc, cax=cax, orientation='horizontal', ticks=[vmin, (vmin+vmax)/2, vmax])
cbar.ax.xaxis.set_ticks_position('bottom')
cbar.set_label('Reliability Score', fontsize=12, labelpad=-40, x=0.5, color='black')
tick_labels = [f'{x:.2f}' for x in [vmin, (vmin+vmax)/2, vmax]]
cbar.ax.set_xticklabels(tick_labels, fontsize=12)

# --- Final Synchronization ---
ax_lon_prof.set_xlim(65.96, 99.56)
ax_lat_prof.set_ylim(ax_map.get_ylim())

ax_lon_prof.xaxis.set_major_locator(ticker.FixedLocator([65, 70, 75, 80, 85, 90, 95, 100]))
ax_map.xaxis.set_major_locator(ticker.FixedLocator([65, 70, 75, 80, 85, 90, 95, 100]))
ax_lat_prof.yaxis.set_major_locator(ax_map.yaxis.get_major_locator())

output_path = os.path.join(base_path, 'Reliability_Spatial_Profile_Density_Box_final_v1_1.png')
# plt.savefig(output_path, dpi=300, bbox_inches='tight')
# print(f"Plot saved to: {output_path}")
plt.show()
