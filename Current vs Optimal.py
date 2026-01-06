import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
from scipy.stats import gaussian_kde
import os
import geopandas as gpd
from tqdm.auto import tqdm
from google.colab import drive
from shapely.geometry import Point

# Global Plot Settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['axes.labelweight'] = 'bold'

# Mount Google Drive
drive.mount('/content/drive')

# Define paths
base_path = '/content/drive/MyDrive/mukund_iitr/PHD/PHD/MY PHD WORK/objective 3/CropLizer/Analysis/Results/Benefits Analysis'
file_path = os.path.join(base_path, 'Benefit Analysis.xlsx')
shp_path = os.path.join(base_path, '1971_2016_2024 districts aligned/shapefile/state2016_GCS.shp')
gaul_path = '/content/drive/MyDrive/mukund_iitr/PHD/PHD/MY PHD WORK/objective 3/CropLizer/Analysis/Results/Model Accuracy/GAUL_2024_L1.zip'

# Load data
print("Loading Excel Data...")
df = pd.read_excel(file_path)

# --- Shapefile Integration Logic ---
print("Processing Shapefiles...")
gdf_india = gpd.read_file(shp_path)
combined_gdf = gpd.GeoDataFrame()
country_colors = {
    'India': '#f0f1fa',
    'Bangladesh': '#f2cece',
    'Nepal': '#d0f2ce'
}

try:
    # Load GAUL L1 data
    gdf_gaul = gpd.read_file(gaul_path)

    # Filter for Bangladesh and Nepal
    target_countries = ['Bangladesh', 'Nepal']

    # Check for correct column name
    if 'gaul0_name' in gdf_gaul.columns:
        neighbors = gdf_gaul[gdf_gaul['gaul0_name'].isin(target_countries)].copy()
        neighbors['Plot_Category'] = neighbors['gaul0_name']
    elif 'ADM0_NAME' in gdf_gaul.columns:
        neighbors = gdf_gaul[gdf_gaul['ADM0_NAME'].isin(target_countries)].copy()
        neighbors['Plot_Category'] = neighbors['ADM0_NAME']
    else:
        print(f"Warning: Country name column not found. Available columns: {gdf_gaul.columns}")
        neighbors = gpd.GeoDataFrame(columns=['geometry', 'Plot_Category'])

except Exception as e:
    print(f"Error loading GAUL data: {e}")
    neighbors = gpd.GeoDataFrame(columns=['geometry', 'Plot_Category'])

# Reproject neighbors to match India's CRS
if not neighbors.empty and gdf_india.crs != neighbors.crs:
    neighbors = neighbors.to_crs(gdf_india.crs)

# Standardize columns for merging
gdf_india['Plot_Category'] = 'India'
gdf_india_clean = gdf_india[['geometry', 'Plot_Category']]
neighbors_clean = neighbors[['geometry', 'Plot_Category']]

# Create the new combined shapefile (Ensure it is a GeoDataFrame)
combined_gdf = pd.concat([gdf_india_clean, neighbors_clean], ignore_index=True)
combined_gdf = gpd.GeoDataFrame(combined_gdf, crs=gdf_india.crs)

# --- Identify Polygons with Data (Aesthetic Logic) ---
print("Identifying polygons with data points...")
if not combined_gdf.empty:
    # Create geometry for points to check intersection
    # Ensure points have same CRS as shapefile
    points_geom = [Point(xy) for xy in zip(df['LONG'], df['LAT'])]
    gdf_points = gpd.GeoDataFrame(df, geometry=points_geom, crs=combined_gdf.crs)

    # Spatial join to find polygons containing points
    polys_with_points = gpd.sjoin(combined_gdf, gdf_points, how='inner', predicate='intersects')
    indices_with_points = polys_with_points.index.unique()

    # Mark polygons
    combined_gdf['has_points'] = combined_gdf.index.isin(indices_with_points)
else:
    combined_gdf['has_points'] = False

# --- Pre-calculate Map Layers (Optimization) ---
# This block ensures we calculate aesthetics only once, not inside the loop
print("Pre-calculating map geometries and layers...")
dissolved_boundary = None
map_layers = {} # Keys: (country, has_points)

if not combined_gdf.empty:
    # 1. Dissolve the boundary once (with buffer for smoothness)
    buffered_gdf = combined_gdf.copy()
    buffered_gdf['geometry'] = buffered_gdf.geometry.buffer(0.01)
    dissolved_boundary = buffered_gdf.dissolve()

    # 2. Group by category AND data presence for fast vectorized plotting with correct styles
    for (country, has_pts), group in combined_gdf.groupby(['Plot_Category', 'has_points']):
        map_layers[(country, has_pts)] = group

# Calculate Fertilizer Use Efficiency (FUE)
df['Current_FUE'] = df['Current_Yield_Pred'] / df['Current_NPK_Sum']
df['Opt_FUE'] = df['Opt_Balanced_Yield'] / df['Opt_Balanced_TotalNPK']

# Define pair groups
pairs_yld_frt = [
    ('Current_Yield_Pred', 'Opt_Balanced_Yield', 'YLD', 'kg/ha'),
    ('Current_NPK_Sum', 'Opt_Balanced_TotalNPK', 'FRT', 'kg NPK/ha')
]

pairs_fue_bcr = [
    ('Current_FUE', 'Opt_FUE', 'FUE', 'YLD/FRT'),
    ('Current_BCR', 'Opt_Balanced_BCR', 'BCR', 'BCR')
]

groups = [
    (pairs_yld_frt, 'yield_and_fert_maps'),
    (pairs_fue_bcr, 'fue_and_bcr_maps')
]

# Colors
map_cmap = 'turbo'
violin_colors = ['#9c3636', '#0d6391']
w_map = 1.0
w_spacer = 0.1
w_violin = 0.7
mm_to_inch = 1 / 25.4
fig_width = 190 * mm_to_inch

# --- Helper Functions ---

def draw_half_violin(ax, data, side, color):
    min_val, max_val = np.min(data), np.max(data)
    buffer = (max_val - min_val) * 0.2
    y_grid = np.linspace(min_val - buffer, max_val + buffer, 200)

    kde = gaussian_kde(data)
    density = kde(y_grid)
    density = density / density.max() * 0.35

    q1, median, q3 = np.percentile(data, [25, 50, 75])
    low_whisker, high_whisker = np.percentile(data, [5, 95])
    box_width = 0.08

    if side == 'left':
        ax.fill_betweenx(y_grid, -density, 0, facecolor=color, alpha=0.4, edgecolor='none')
        ax.fill_betweenx(y_grid, -density, 0, facecolor='none', alpha=1, edgecolor='black', linewidth=0.5)

        x_center = -0.1
        ax.plot([x_center, x_center], [low_whisker, q1], color='black', linewidth=0.8, alpha=0.8)
        ax.plot([x_center, x_center], [q3, high_whisker], color='black', linewidth=0.8, alpha=0.8)
        rect = mpatches.Rectangle((x_center - box_width/2, q1), box_width, q3-q1,
                                  facecolor=color, edgecolor='black', linewidth=0.8, alpha=0.6, zorder=2)
        ax.add_patch(rect)
        ax.scatter(x_center, median, color='white', s=10, zorder=3)

    else: # right
        ax.fill_betweenx(y_grid, 0, density, facecolor=color, alpha=0.4, edgecolor='none')
        ax.fill_betweenx(y_grid, density, 0, facecolor='none', alpha=1, edgecolor='black', linewidth=0.5)

        x_center = 0.1
        ax.plot([x_center, x_center], [low_whisker, q1], color='black', linewidth=0.8, alpha=0.8)
        ax.plot([x_center, x_center], [q3, high_whisker], color='black', linewidth=0.8, alpha=0.8)
        rect = mpatches.Rectangle((x_center - box_width/2, q1), box_width, q3-q1,
                                  facecolor=color, edgecolor='black', linewidth=0.8, alpha=0.6, zorder=2)
        ax.add_patch(rect)
        ax.scatter(x_center, median, color='white', s=10, zorder=3)

def plot_spatial_map(ax, lat, lon, values, vmin, vmax, cmap):
    ax.set_aspect('equal')

    # 1. Plot Background Countries (Using pre-calculated layers)
    if map_layers:
        for (country, has_pts), layer in map_layers.items():
            f_color = country_colors.get(country, '#e6e8fa')

            # Aesthetic logic:
            # If polygon has data points -> Black edge, Z-order 2 (higher)
            # If polygon has NO data points -> Grey edge, Z-order 1 (lower)
            if has_pts:
                e_color = 'black'
                z_ord = 2
            else:
                e_color = 'grey'
                z_ord = 1

            layer.plot(ax=ax, facecolor=f_color, edgecolor=e_color, linewidth=0.5, alpha=1, zorder=z_ord)

    # Overlay dissolved boundary (Using pre-calculated geometry)
    if dissolved_boundary is not None:
        dissolved_boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5, zorder=3)

    # 2. Plot Scatter Data
    # Fixed size since legend is removed
    sizes = 1

    sc = ax.scatter(lon, lat, c=values, s=sizes, cmap=cmap, vmin=vmin, vmax=vmax, norm='log', alpha=0.9, edgecolors='none', zorder=4)
    ax.set_axis_off()
    return sc

# --- Main Plotting Loop wrapped in Function ---

def create_plot(pairs_subset, filename_suffix):
    # Adjust height for 2 rows (approx 2.3 times the base unit)
    fig_height = (69.25 * 2.3) * mm_to_inch
    fig = plt.figure(figsize=(fig_width, fig_height))

    # GridSpec: 2 rows, 4 columns
    gs = fig.add_gridspec(nrows=2, ncols=4, width_ratios=[w_map, w_map, w_spacer, w_violin],
                          height_ratios=[1, 1],
                          wspace=0.15, hspace=0.0,
                          left=0.05, right=0.95, top=0.9, bottom=0.1)

    for i, (col1, col2, title, unit) in tqdm(enumerate(pairs_subset), total=len(pairs_subset), desc=f'Plotting {filename_suffix}'):
        # Prepare data
        mask = df[[col1, col2, 'LAT', 'LONG', 'CRLPARHA']].notna().all(axis=1)
        df_clean = df[mask]

        vals_curr = df_clean[col1]
        vals_opt = df_clean[col2]

        # Determine color limits
        combined_vals = pd.concat([vals_curr, vals_opt])
        vmin = np.percentile(combined_vals, 2.5)
        vmax = np.percentile(combined_vals, 97.5)
        if vmin <= 0: vmin = 0.001

        # Create axes
        ax_curr = fig.add_subplot(gs[i, 0])
        ax_opt = fig.add_subplot(gs[i, 1])
        ax_violin = fig.add_subplot(gs[i, 3])

        # 1. Plot Maps
        plot_spatial_map(ax_curr, df_clean['LAT'], df_clean['LONG'],
                         vals_curr, vmin, vmax, map_cmap)

        sc2 = plot_spatial_map(ax_opt, df_clean['LAT'], df_clean['LONG'],
                               vals_opt, vmin, vmax, map_cmap)

        # Row Title on the far left
        ax_curr.text(-0.05, 0.5, title, fontsize=12, fontweight='bold', rotation=90,
                     va='center', ha='right', transform=ax_curr.transAxes)

        # Colorbar
        cax = ax_opt.inset_axes([-0.6, -0.02, 0.7, 0.03])
        ax_opt.text(1.3, -0.02, unit, fontsize=12, fontweight='normal', rotation=0,
                  va='center', ha='left', transform=ax_curr.transAxes)
        tick_locs = np.geomspace(vmin, vmax, 4)
        cbar = fig.colorbar(sc2, cax=cax, orientation='horizontal', alpha=0.3, ticks=tick_locs)

        tick_labels = [f'{x:,.1f}' if x < 10 else f'{int(x):,}' for x in tick_locs]
        cbar.ax.set_xticklabels(tick_labels, fontsize=12)
        cbar.ax.xaxis.set_ticks_position('bottom')
        cbar.ax.minorticks_off()
        cbar.ax.tick_params(which='minor', size=0)
        cbar.outline.set_visible(True)

        # 2. Match Heights
        ylims = ax_curr.get_ylim()
        xlims = ax_curr.get_xlim()
        map_ratio = (ylims[1] - ylims[0]) / (xlims[1] - xlims[0])

        violin_aspect = map_ratio * (w_map / (w_violin+0.08))
        ax_violin.set_box_aspect(violin_aspect)
        ax_violin.set_anchor('C')

        # 3. Plot Violins
        data1 = vals_curr.replace([np.inf, -np.inf], np.nan).dropna()
        data2 = vals_opt.replace([np.inf, -np.inf], np.nan).dropna()

        draw_half_violin(ax_violin, data1, 'left', violin_colors[0])
        draw_half_violin(ax_violin, data2, 'right', violin_colors[1])

        # Percentage Change Label
        median_curr = np.median(data1)
        median_opt = np.median(data2)
        pct_change = ((median_opt - median_curr) / median_curr) * 100
        ax_violin.text(0.95, 0.05, f'{pct_change:+.1f}%', ha='right', va='bottom',
                       transform=ax_violin.transAxes, fontsize=12, fontweight='bold', color='black')

        # Formatting Violin Axes
        all_data = np.concatenate([data1, data2])
        y_min_ci = np.percentile(all_data, 0)
        y_max_ci = np.percentile(all_data, 99.5)
        if title == "FUE": y_max_ci = np.percentile(all_data, 97.5)

        view_buffer = (y_max_ci - y_min_ci) * 0.1
        ax_violin.set_ylim(y_min_ci - view_buffer, y_max_ci + view_buffer)
        ax_violin.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax_violin.tick_params(axis='y', labelsize=12)

        ax_violin.set_xlim(-0.5, 0.5)
        ax_violin.set_xticks([])
        if unit:
            ax_violin.set_xlabel(unit, fontsize=12, fontweight='normal')
            ax_violin.xaxis.set_label_coords(0.01, -0.05)

        ax_violin.spines['top'].set_visible(False)
        ax_violin.spines['right'].set_visible(False)
        ax_violin.spines['bottom'].set_visible(False)
        ax_violin.spines['left'].set_color('black')
        ax_violin.grid(False)

    # --- Headers (Applied to top row of the current figure) ---
    ax_top_left = fig.axes[0]
    ax_top_mid = fig.axes[1]
    ax_top_right = fig.axes[2]
    ax_top_left.text(-0.1, 0.95, 'a', fontsize=12, fontweight='bold', ha='center', va='bottom', transform=ax_top_left.transAxes)
    ax_top_mid.text(-0.1, 0.95, 'b', fontsize=12, fontweight='bold', ha='center', va='bottom', transform=ax_top_mid.transAxes)
    ax_top_right.text(-0.12, 1.05, 'c', fontsize=12, fontweight='bold', ha='center', va='bottom', transform=ax_top_right.transAxes)
    ax_top_left.text(0.5, 0.95, 'Current', fontsize=12, fontweight='bold', ha='center', va='bottom', transform=ax_top_left.transAxes)
    ax_top_mid.text(0.5, 0.95, 'Optimized', fontsize=12, fontweight='bold', ha='center', va='bottom', transform=ax_top_mid.transAxes)
    ax_top_right.text(0.5, 1.05, 'Distribution', fontsize=12, fontweight='bold', ha='center', va='bottom', transform=ax_top_right.transAxes)

    # --- Legends ---
    # 1. Country Legend
    country_patches = [
        mpatches.Patch(facecolor=country_colors['India'], edgecolor='grey', label='India'),
        mpatches.Patch(facecolor=country_colors['Bangladesh'], edgecolor='grey', label='Bangladesh'),
        mpatches.Patch(facecolor=country_colors['Nepal'], edgecolor='grey', label='Nepal')
    ]

    # 2. Violin Color Legend
    violin_patches = [
        mpatches.Patch(color=violin_colors[0], label='Current'),
        mpatches.Patch(color=violin_colors[1], label='Optimized')
    ]

    legend1 = fig.legend(handles=country_patches, loc='lower center', ncol=3,
                         fontsize=12, frameon=False, bbox_to_anchor=(0.32, -0.02), columnspacing=0.5, handletextpad=0.3)
    legend2 = fig.legend(handles=violin_patches, loc='lower center', ncol=2,
                         fontsize=12, frameon=False, bbox_to_anchor=(0.85, -0.02), columnspacing=0.5, handletextpad=0.3)

    # Save
    output_path = os.path.join(base_path, f'{filename_suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.show()

# Run for both groups
for pairs_subset, suffix in groups:
    create_plot(pairs_subset, suffix)
