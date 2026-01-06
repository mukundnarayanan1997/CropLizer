import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch, Patch
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances_argmin_min
from tqdm.auto import tqdm
import geopandas as gpd
from shapely.geometry import Point
import os
import joblib

# Try importing drive for Colab
try:
    from google.colab import drive
except ImportError:
    pass

# --- 1. Configuration & Constants ---

FEATURE_NAME_MAP = {
    'VARTYPE': 'Variety', 'SRATEHA': 'Seed Rate', 'Organic_ha': 'Manure',
    'TDATE_yday': 'Sowing Date', 'EST_line': 'Line Sowing', 'EST_binary': 'Transplanting',
    'IRRINU': 'Irrigation Count', 'PCRR': 'Residue', 'Field_duration': 'Duration',
    'GYP_ha': 'Gypsum', 'BOR_ha': 'Boron', 'Zn_ha': 'Zinc'
}

UNIT_MAP = {
    'SRATEHA': 'kg/ha', 'Organic_ha': 'kg/ha', 'GYP_ha': 'kg/ha',
    'BOR_ha': 'kg/ha', 'Zn_ha': 'kg/ha', 'TDATE_yday': 'DOY',
    'Field_duration': 'days', 'IRRINU': 'no.', 'PCRR': '%',
    'VARTYPE': '', 'EST_line': '', 'EST_binary': ''
}

FEATURES = [
    'LAT', 'LONG', 'SEASON', 'CRLPARHA', 'GEN', 'EDU', 'SOCCAT', 'CRARHA', 'HHMEM',
    'DISMR', 'SOPER', 'DCLASS',
    'SoilGrids_bdod', 'SoilGrids_clay', 'SoilGrids_nitrogen', 'SoilGrids_ocd',
    'SoilGrids_phh2o', 'SoilGrids_sand', 'SoilGrids_silt', 'SoilGrids_soc',
    'total_precip', 'num_dry_days', 'avg_dspell_length', 'monsoon_onset', 'monsoon_length',
    'PCRR', 'TDATE_yday', 'Field_duration', 'VARTYPE', 'EST_line', 'EST_binary', 'SRATEHA',
    'IRRIAVA', 'IRRINU', 'WSEV', 'INSEV', 'DISEV', 'DRSEV', 'FLSEV',
    'GYP_ha', 'BOR_ha', 'Zn_ha', 'Organic_ha'
]

PRACTICE_FEATURES = [
    'VARTYPE', 'SRATEHA', 'Organic_ha', 'TDATE_yday', 'EST_line', 'EST_binary',
    'IRRINU', 'PCRR', 'Field_duration', 'GYP_ha', 'BOR_ha', 'Zn_ha'
]

ECONOMIC_DEFAULTS = {
    'FGPRICE_quintal': 3368.0, 'Urea_price': 250.0, 'DAP_price': 1400.0,
    'MOP_price': 250.0, 'Seed_price': 105.08, 'Labour_cost': 48533.0,
    'Fixed_cost': 2610.97
}

NUMERIC_FEATURES = [
    'LAT', 'LONG', 'CRLPARHA', 'CRARHA', 'HHMEM', 'DISMR', 'SoilGrids_bdod', 'SoilGrids_clay',
    'SoilGrids_nitrogen', 'SoilGrids_ocd', 'SoilGrids_phh2o', 'SoilGrids_sand', 'SoilGrids_silt',
    'SoilGrids_soc', 'total_precip', 'num_dry_days', 'avg_dspell_length', 'monsoon_onset',
    'monsoon_length', 'PCRR', 'TDATE_yday', 'Field_duration', 'SRATEHA', 'IRRINU',
    'GYP_ha', 'BOR_ha', 'Zn_ha', 'Organic_ha'
]
CATEGORICAL_FEATURES = [c for c in FEATURES if c not in NUMERIC_FEATURES]
TARGETS = ['yield_kg_ha', 'N_kg_ha', 'P2O5_ha', 'K2O_ha']

# Set global font size
plt.rcParams.update({'font.size': 11})

# --- 2. Model & Optimization Logic ---

def load_and_train():
    print("Loading Data...")
    if not os.path.exists('NUE_survey_dataset_v2.csv'):
        print("Dataset not found. Please ensure 'NUE_survey_dataset_v2.csv' is in the directory.")
        return None, None

    df = pd.read_csv('NUE_survey_dataset_v2.csv', low_memory=False, on_bad_lines='skip')
    df = df[[c for c in FEATURES + TARGETS if c in df.columns]].copy()

    for c in NUMERIC_FEATURES + TARGETS:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for c in CATEGORICAL_FEATURES:
        if c in df.columns: df[c] = df[c].fillna('Unknown').astype(str)
    df.dropna(subset=TARGETS, inplace=True)

    colab_path = '/content/drive/MyDrive/mukund_iitr/PHD/PHD/MY PHD WORK/objective 3/CropLizer/Analysis/Methods/Optimization/Models/best_model_RandomForest_pipeline.joblib'
    local_path = r'G:\My Drive\mukund_iitr\PHD\PHD\MY PHD WORK\objective 3\CropLizer\Analysis/Methods/Optimization/Models/best_model_RandomForest_pipeline.joblib'

    model_path = colab_path if os.path.exists(colab_path) else local_path

    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        try:
            model = joblib.load(model_path)

            # --- PATCH START: Fix for sklearn version incompatibility ---
            try:
                if hasattr(model, 'named_steps') and 'model' in model.named_steps:
                    rf_model = model.named_steps['model']
                    if hasattr(rf_model, 'estimators_'):
                        for estimator in rf_model.estimators_:
                            if not hasattr(estimator, 'monotonic_cst'):
                                estimator.monotonic_cst = None
            except Exception as e:
                print(f"Warning: Could not patch model for version compatibility: {e}")
            # --- PATCH END ---

            print("Validating model compatibility...")
            model.predict(df.head(1)[FEATURES])

        except Exception as e:
            print(f"Failed to load or validate model (Switching to Fallback): {e}")
            model = None
    else:
        print(f"Model file not found at {model_path}. Training fallback model...")
        model = None

    if model is None:
        print("Training Random Forest (Fallback)...")
        preprocessor = ColumnTransformer([
            ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scl', StandardScaler())]), NUMERIC_FEATURES),
            ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('enc', OneHotEncoder(handle_unknown='ignore'))]), CATEGORICAL_FEATURES)
        ])

        model = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=50, max_depth=15, n_jobs=-1, random_state=42))
        ])

        model.fit(df[FEATURES], df[TARGETS])

    return model, df

def calculate_fue_bcr(row, y_pred, n_pred, p_pred, k_pred):
    total_npk = max(1, n_pred + p_pred + k_pred)
    fue = y_pred / total_npk

    area = row.get('CRLPARHA', 1.0)
    if pd.isna(area) or area <= 0: area = 1.0

    dap = p_pred / 0.46
    n_rem = n_pred - (dap * 0.18)
    urea = n_rem / 0.46 if n_rem > 0 else 0
    mop = k_pred / 0.60

    fert_cost = (urea * ECONOMIC_DEFAULTS['Urea_price'] + dap * ECONOMIC_DEFAULTS['DAP_price'] + mop * ECONOMIC_DEFAULTS['MOP_price']) / 50.0
    seed_cost = row.get('SRATEHA', 30) * area * ECONOMIC_DEFAULTS['Seed_price']
    total_cost = fert_cost + seed_cost + (ECONOMIC_DEFAULTS['Labour_cost'] * area) + (ECONOMIC_DEFAULTS['Fixed_cost'] * area)

    revenue = (y_pred * area / 100) * ECONOMIC_DEFAULTS['FGPRICE_quintal']
    bcr = revenue / total_cost if total_cost > 0 else 0

    return fue, bcr

def get_optimization_steps(row, model, df_ref):
    np.random.seed(1997)

    NUM_SIMS = 20000
    grid = {}

    for col in PRACTICE_FEATURES:
        if col in CATEGORICAL_FEATURES:
            options = [x for x in df_ref[col].unique() if x != 'Unknown']
            grid[col] = np.random.choice(options, NUM_SIMS)
        else:
            curr = row.get(col, 0)
            if pd.isna(curr): curr = 0
            grid[col] = np.random.uniform(curr * 0.5, curr * 1.5, NUM_SIMS)

    sim_df = pd.DataFrame(grid)
    for col in FEATURES:
        if col not in PRACTICE_FEATURES:
            sim_df[col] = row[col]

    preds = model.predict(sim_df[FEATURES])
    best_idx = np.argmax(preds[:, 0])
    best_config = sim_df.iloc[best_idx].to_dict()

    steps = []

    curr_df = pd.DataFrame([row])[FEATURES]
    base_preds = model.predict(curr_df)[0]
    fue_base, bcr_base = calculate_fue_bcr(row, *base_preds)

    lat_val = row.get('LAT', 0.0)
    lon_val = row.get('LONG', 0.0)

    steps.append({
        'Step': 'Current',
        'FUE': fue_base,
        'BCR': bcr_base,
        'Label': 'Base',
        'Desc': f"Current Status ({lat_val:.2f}° N, {lon_val:.2f}° E)"
    })

    current_state = row.copy()

    def fmt_val(c, v):
        if isinstance(v, str):
            return v.replace('_', ' ')

        if c == 'TDATE_yday':
            try:
                doy = int(round(v))
                doy = max(1, min(365, doy))
                return pd.to_datetime(f"2023-{doy}", format="%Y-%j").strftime('%d-%b')
            except:
                return str(int(round(v)))

        if c in ['IRRINU', 'Field_duration']:
            try:
                return f"{int(round(v))}"
            except:
                return str(v)

        if isinstance(v, (float, int)):
            return f"{v:.2f}"

        return str(v).replace('_', ' ')

    for k in PRACTICE_FEATURES:
        curr_val = row.get(k)
        opt_val = best_config.get(k)

        val_from = fmt_val(k, curr_val)
        val_to = fmt_val(k, opt_val)

        if val_from != val_to:
            current_state[k] = opt_val

            step_df = pd.DataFrame([current_state])[FEATURES]
            p = model.predict(step_df)[0]
            fue, bcr = calculate_fue_bcr(current_state, *p)

            label_friendly = FEATURE_NAME_MAP.get(k, k)
            unit = UNIT_MAP.get(k, "")

            sep = " " if unit else ""
            unit_display = unit

            if k == 'TDATE_yday':
                sep = ""
                unit_display = ""

            desc = f"{label_friendly}: {val_from}{sep}{unit_display} → {val_to}{sep}{unit_display}"

            steps.append({'Step': k, 'FUE': fue, 'BCR': bcr, 'Label': label_friendly, 'Desc': desc})

    if len(steps) > 6:
        base = steps[0]
        final = steps[-1]
        intermediates = steps[1:]

        indices = np.linspace(0, len(intermediates)-1, 5, dtype=int)

        selected_inter = [intermediates[i] for i in indices]

        if selected_inter[-1] != final:
            selected_inter[-1] = final

        steps = [base] + selected_inter

    return pd.DataFrame(steps)

# --- 3. Visualization ---

def plot_waterfall(ax, df, metric, color, show_xaxis_labels=True, show_step_labels=True, x_limits=None):
    values = df[metric].values
    descs = df['Desc'].values
    deltas = [values[0]] + [values[i] - values[i-1] for i in range(1, len(values))]

    y = np.arange(len(values))

    if x_limits:
        ax.set_xlim(x_limits)
    else:
        x_min_val = min(values)
        x_max_val = max(values)
        x_range = x_max_val - x_min_val
        x_lower = max(0, x_min_val - (x_range * 0.1))
        x_upper = x_max_val + (x_range * 1.5)
        if x_range < 10:
             x_lower = x_min_val * 0.9
             x_upper = x_max_val * 1.1
        ax.set_xlim(x_lower, x_upper)

    ax.set_ylim(y[0] - 0.6, y[-1] + 0.6)

    bar_height = 0.5
    ax.barh(y[0], values[0], color='grey', alpha=0.5, height=bar_height)

    for i in range(1, len(values)):
        left = values[i-1]
        change = deltas[i]
        c = color if change >= 0 else '#d62728'
        ax.barh(y[i], change, left=left, color=c, height=bar_height)

        ax.plot([values[i-1], values[i-1]], [y[i-1]+(bar_height/2), y[i]-(bar_height/2)], '-', color='black', alpha=0.3, linewidth=1)

    ax.plot([values[-1], values[-1]], [y[0]-(bar_height/2), y[-1]+(bar_height/2)], '--', color='black', alpha=0.3, linewidth=1)

    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.tick_params(axis='y', which='both', left=False, right=False)

    cur_xlim = ax.get_xlim()
    cur_x_range = cur_xlim[1] - cur_xlim[0]

    for i, v in enumerate(values):
        val_str = f"{v:.2f}"  # Changed to 2 decimal places
        pos_x = v + (cur_x_range * 0.02)
        ax.text(pos_x, y[i], val_str, ha='left', va='center', fontsize=10, fontweight='bold', color='black')

    if show_step_labels:
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(y)
        ax2.set_yticklabels(descs, fontsize=10)

        for spine in ax2.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(1)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

def process_and_plot():
    print("Mounting Google Drive...")
    try:
        drive.mount('/content/drive')
    except:
        pass

    model, df = load_and_train()

    if model is None:
        return

    N_CLUSTERS = 3
    print("Clustering Regions...")
    coords = df[['LAT', 'LONG']].dropna()
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    df['Cluster'] = np.nan
    df.loc[coords.index, 'Cluster'] = kmeans.fit_predict(coords)

    centroids = kmeans.cluster_centers_
    TARGET_REFS = {
        'LGP': np.array([23.51, 89.30]),
        'UGP': np.array([28.27, 83.92]),
        'SIA': np.array([17.74, 82.63])
    }

    print("Optimizing Representatives...")
    plot_data = []

    for i in tqdm(range(N_CLUSTERS), desc="Processing Clusters"):
        cluster_points = df[df['Cluster'] == i][['LAT', 'LONG']]
        cluster_centroid = centroids[i]
        best_target_name = min(TARGET_REFS, key=lambda k: np.linalg.norm(TARGET_REFS[k] - cluster_centroid))
        target_coords = TARGET_REFS[best_target_name]

        dists = np.linalg.norm(cluster_points.values - target_coords, axis=1)
        sorted_indices = np.argsort(dists)

        found_rep = False
        fallback_data = None
        check_count = 0
        MAX_CHECKS = 100

        max_change = -1.0
        best_rep_data = None

        for idx in sorted_indices:
            candidate_idx = cluster_points.index[idx]
            candidate_row = df.loc[candidate_idx]

            if not candidate_row[PRACTICE_FEATURES].isna().any():
                candidate_dict = candidate_row.to_dict()
                steps_df = get_optimization_steps(candidate_dict, model, df)

                curr_fue = steps_df.iloc[0]['FUE']
                curr_bcr = steps_df.iloc[0]['BCR']
                opt_fue = steps_df.iloc[-1]['FUE']
                opt_bcr = steps_df.iloc[-1]['BCR']

                if fallback_data is None:
                    fallback_data = (steps_df, candidate_dict['LAT'], candidate_dict['LONG'])

                change = opt_fue - curr_fue + opt_bcr - curr_bcr
                if change > max_change:
                    max_change = change
                    best_rep_data = (steps_df, candidate_dict['LAT'], candidate_dict['LONG'])
                    found_rep = True

                check_count += 1
                if check_count >= MAX_CHECKS:
                    break

        if found_rep:
            plot_data.append(best_rep_data)
        else:
            if fallback_data:
                print(f"Cluster {i}: No representative found. Using closest valid point.")
                plot_data.append(fallback_data)
            else:
                closest_idx, _ = pairwise_distances_argmin_min([centroids[i]], cluster_points)
                rep_idx = cluster_points.index[closest_idx[0]]
                rep_row = df.loc[rep_idx].to_dict()
                steps_df = get_optimization_steps(rep_row, model, df)
                plot_data.append((steps_df, rep_row['LAT'], rep_row['LONG']))

    all_fue_values = []
    all_bcr_values = []
    for data, _, _ in plot_data:
        all_fue_values.extend(data['FUE'].values)
        all_bcr_values.extend(data['BCR'].values)

    def get_common_limits(values):
        v_min = min(values)
        v_max = max(values)
        v_range = v_max - v_min
        pad = v_range * 0.5
        lower = v_min - pad
        upper = v_max + pad
        return (lower, upper)

    fue_xlim = get_common_limits(all_fue_values)
    bcr_xlim = get_common_limits(all_bcr_values)

    print("Generating Plot...")

    cluster_order = np.argsort([c[0] for c in centroids])[::-1]
    region_names = ["UGP", "LGP", "SIA"]

    fig = plt.figure(figsize=(7.48, 5.04))
    gs = gridspec.GridSpec(N_CLUSTERS, 3, width_ratios=[1, 1, 1], wspace=0.0, hspace=0.0)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    base_path = '/content/drive/MyDrive/mukund_iitr/PHD/PHD/MY PHD WORK/objective 3/CropLizer/Analysis/Results/Benefits Analysis'
    shp_path = os.path.join(base_path, '1971_2016_2024 districts aligned/shapefile/state2016_GCS.shp')
    gaul_path = '/content/drive/MyDrive/mukund_iitr/PHD/PHD/MY PHD WORK/objective 3/CropLizer/Analysis/Results/Model Accuracy/GAUL_2024_L1.zip'

    combined_gdf = gpd.GeoDataFrame()

    if os.path.exists(shp_path):
        india = gpd.read_file(shp_path)
        india['Plot_Category'] = 'India'
        combined_gdf = pd.concat([combined_gdf, india[['geometry', 'Plot_Category']]], ignore_index=True)

    if os.path.exists(gaul_path):
        neighbors = gpd.read_file(gaul_path)
        target_countries = ['Bangladesh', 'Nepal']
        if 'gaul0_name' in neighbors.columns:
            neighbors = neighbors[neighbors['gaul0_name'].isin(target_countries)].copy()
            neighbors['Plot_Category'] = neighbors['gaul0_name']
        elif 'ADM0_NAME' in neighbors.columns:
            neighbors = neighbors[neighbors['ADM0_NAME'].isin(target_countries)].copy()
            neighbors['Plot_Category'] = neighbors['ADM0_NAME']

        if not neighbors.empty:
            if not combined_gdf.empty and combined_gdf.crs != neighbors.crs:
                neighbors = neighbors.to_crs(combined_gdf.crs)
            combined_gdf = pd.concat([combined_gdf, neighbors[['geometry', 'Plot_Category']]], ignore_index=True)

    if not combined_gdf.empty:
        combined_gdf = gpd.GeoDataFrame(combined_gdf, crs=combined_gdf.crs)
        points_geom = [Point(xy) for xy in zip(df['LONG'], df['LAT'])]
        gdf_points = gpd.GeoDataFrame(df, geometry=points_geom, crs=combined_gdf.crs)
        polys_with_points = gpd.sjoin(combined_gdf, gdf_points, how='inner', predicate='intersects')
        indices_with_points = polys_with_points.index.unique()
        combined_gdf['has_points'] = combined_gdf.index.isin(indices_with_points)

    for row_idx, cluster_idx in enumerate(cluster_order):
        color = colors[cluster_idx % len(colors)]
        data, rep_lat, rep_lon = plot_data[cluster_idx]

        ax_map = fig.add_subplot(gs[row_idx, 0])

        cluster_points = df[df['Cluster'] == cluster_idx]
        min_lon, max_lon = cluster_points['LONG'].min(), cluster_points['LONG'].max()
        min_lat, max_lat = cluster_points['LAT'].min(), cluster_points['LAT'].max()

        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2

        lon_span = max(max_lon - min_lon, 2.0)
        lat_span = max(max_lat - min_lat, 2.0)

        target_lon = lon_span * 1.8
        target_lat = lat_span * 1.8

        ax_map.set_xlim(center_lon - target_lon/2, center_lon + target_lon/2)
        ax_map.set_ylim(center_lat - target_lat/2, center_lat + target_lat/2)
        ax_map.set_aspect('equal', adjustable='datalim')

        if not combined_gdf.empty:
            country_colors = {'India': '#f0f1fa', 'Bangladesh': '#f2cece', 'Nepal': '#d0f2ce'}
            for (country, has_pts), group in combined_gdf.groupby(['Plot_Category', 'has_points']):
                f_color = country_colors.get(country, '#e6e8fa')
                if has_pts:
                    e_color = 'black'; z_ord = 2; lw = 0.8
                else:
                    e_color = 'grey'; z_ord = 1; lw = 0.5
                group.plot(ax=ax_map, facecolor=f_color, edgecolor=e_color, linewidth=lw, alpha=1, zorder=z_ord)

        ax_map.scatter(cluster_points['LONG'], cluster_points['LAT'], c=color, s=1, alpha=0.7, zorder=3)
        ax_map.scatter(rep_lon, rep_lat, c='black', marker='x', s=20, linewidth=2.5, zorder=4)

        ax_map.set_xticks([])
        ax_map.set_yticks([])
        for spine in ax_map.spines.values():
            spine.set_visible(True); spine.set_edgecolor('black'); spine.set_linewidth(1)

        ax_map.set_ylabel(region_names[row_idx], fontsize=12, fontweight='bold', labelpad=2)

        ax_fue = fig.add_subplot(gs[row_idx, 1])
        is_bottom = (row_idx == (N_CLUSTERS - 1))
        plot_waterfall(ax_fue, data, 'FUE', color, show_xaxis_labels=is_bottom, show_step_labels=False, x_limits=fue_xlim)

        con = ConnectionPatch(
            xyA=(rep_lon, rep_lat), coordsA=ax_map.transData,
            xyB=(0, 0.1), coordsB=ax_fue.transAxes,
            axesA=ax_map, axesB=ax_fue,
            color="black", arrowstyle="-|>", linewidth=1, zorder=5
        )
        ax_fue.add_artist(con)

        ax_bcr = fig.add_subplot(gs[row_idx, 2])
        plot_waterfall(ax_bcr, data, 'BCR', color, show_xaxis_labels=is_bottom, show_step_labels=True, x_limits=bcr_xlim)

        if row_idx == 0:
            ax_map.set_title("Regions", fontsize=12, fontweight='bold', pad=10)
            ax_fue.set_title("FUE", fontsize=12, fontweight='bold', pad=10)
            ax_bcr.set_title("BCR", fontsize=12, fontweight='bold', pad=10)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='LGP', markerfacecolor='#1f77b4', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='UGP', markerfacecolor='#ff7f0e', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='SIA', markerfacecolor='#2ca02c', markersize=10),
        Line2D([0], [0], marker='x', color='w', label='Example Farm', markeredgecolor='black', markersize=10, markeredgewidth=2),
        Line2D([0], [0], linestyle='--', color='black', alpha=0.6, label='Optimal'),
        Patch(facecolor='grey', alpha=0.5, label='Base Value'),
        Patch(facecolor='#f0f1fa', edgecolor='grey', label='India'),
        Patch(facecolor='#f2cece', edgecolor='grey', label='Bangladesh'),
        Patch(facecolor='#d0f2ce', edgecolor='grey', label='Nepal')
    ]

    fig.legend(handles=legend_elements, loc='lower center', ncol=5, bbox_to_anchor=(0.65, -0.01), frameon=False)

    output_file = 'UI Step Benefits.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as '{output_file}'")

if __name__ == "__main__":
    process_and_plot()
