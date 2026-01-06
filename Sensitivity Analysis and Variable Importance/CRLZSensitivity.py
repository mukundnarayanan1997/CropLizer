# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm.auto import tqdm
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# --- PATHS ---
MODEL_SAVE_DIR = "/scratch/mukund_n.iitr/CropLizer/v3/Models"
DATA_SAVE_DIR = "/scratch/mukund_n.iitr/CropLizer/v3/Data"
PLOT_SAVE_DIR = "/scratch/mukund_n.iitr/CropLizer/v3/Sensitivity Plots"

# --- FEATURES ---
TARGETS = ['yield_kg_ha', 'N_kg_ha', 'P2O5_ha', 'K2O_ha']
CATEGORICAL_FEATURES = [
    'SEASON', 'GEN', 'EDU', 'SOCCAT', 'SOPER', 'DCLASS',
    'VARTYPE', 'EST_line', 'EST_binary', 'IRRIAVA',
    'WSEV', 'INSEV', 'DISEV', 'DRSEV', 'FLSEV'
]
NUMERIC_FEATURES = [
    'LAT', 'LONG', 'CRLPARHA', 'CRARHA', 'HHMEM', 'DISMR',
    'SoilGrids_bdod', 'SoilGrids_clay', 'SoilGrids_nitrogen', 'SoilGrids_ocd',
    'SoilGrids_phh2o', 'SoilGrids_sand', 'SoilGrids_silt', 'SoilGrids_soc',
    'total_precip', 'num_dry_days', 'avg_dspell_length', 'monsoon_onset', 'monsoon_length',
    'PCRR', 'TDATE_yday', 'Field_duration', 'SRATEHA', 'IRRINU',
    'GYP_ha', 'BOR_ha', 'Zn_ha', 'Organic_ha'
]

# --- UI FEATURE NAME MAPPING (From app.py) ---
# NOTE: Unicode characters (subscripts/superscripts) removed for compatibility
FEATURE_NAME_MAP = {
    'LAT': 'Latitude',
    'LONG': 'Longitude',
    'SEASON': 'Season',
    'CRLPARHA': 'Plot Area (ha)',
    'GEN': 'Farmer Gender',
    'EDU': 'Farmer Education Level',
    'SOCCAT': 'Social Category',
    'CRARHA': 'Total Farm Area (ha)',
    'HHMEM': 'Household Members',
    'DISMR': 'Distance to Market (km)',
    'SOPER': 'Perceived Soil Quality',
    'DCLASS': 'Perceived Drainage',
    'SoilGrids_bdod': 'Soil Bulk Density (kg/dm3)',
    'SoilGrids_clay': 'Soil Clay Content (%)',
    'SoilGrids_nitrogen': 'Soil Nitrogen (g/kg)',
    'SoilGrids_ocd': 'Soil Organic Carbon Density (kg/m3)',
    'SoilGrids_phh2o': 'Soil pH (H2O)',
    'SoilGrids_sand': 'Soil Sand Content (%)',
    'SoilGrids_silt': 'Soil Silt Content (%)',
    'SoilGrids_soc': 'Soil Organic Carbon (g/kg)',
    'total_precip': 'Total Annual Precipitation (mm)',
    'num_dry_days': 'Number of Dry Days (annual)',
    'avg_dspell_length': 'Avg. Dry Spell Length (days)',
    'monsoon_onset': 'Monsoon Onset (day of year)',
    'monsoon_length': 'Monsoon Length (days)',
    'PCRR': 'Previous Crop Residue Retained (%)',
    'TDATE_yday': 'Planting Date',
    'Field_duration': 'Crop Duration (days)',
    'VARTYPE': 'Rice Variety Type',
    'EST_line': 'Establishment (Line or Random)',
    'EST_binary': 'Establishment (Seed or Transplant)',
    'SRATEHA': 'Seed Rate', 
    'IRRIAVA': 'Irrigation Available (yes/no)',
    'IRRINU': 'Number of Irrigations',
    'WSEV': 'Weed Severity',
    'INSEV': 'Insect Pest Severity',
    'DISEV': 'Disease Severity',
    'DRSEV': 'Drought Severity',
    'FLSEV': 'Flood Severity',
    'N_kg_ha': 'Nitrogen (N) Rate',
    'P2O5_ha': 'Phosphorous (P2O5) Rate',
    'K2O_ha': 'Potassium (K2O) Rate',
    'GYP_ha': 'Applied Gypsum (kg/ha)',
    'BOR_ha': 'Applied Boron (kg/ha)',
    'Zn_ha': 'Applied Zinc (kg/ha)',
    'Organic_ha': 'Applied Organic Matter (kg/ha)',
    
    # Target fallback if needed
    'yield_kg_ha': 'Yield (kg/ha)',
}

# --- REVERSE MAPPINGS FOR CATEGORIES (From app.py) ---
REVERSE_INTERPRETABLE_MAPS = {
    'EDU': {
        '1.0': 'No Formal Schooling', '2.0': 'Primary', '3.0': 'Matriculation',
        '4.0': 'Senior Secondary', '5.0': 'Bachelors', '6.0': 'Masters', '7.0': 'PhD',
        'Unknown': 'Unknown'
    },
    'SOPER': {'1.0': 'Low', '2.0': 'Medium', '3.0': 'High', 'Unknown': 'Unknown'},
    'DCLASS': {'1.0': 'Very Lowland', '2.0': 'Lowland', '3.0': 'Mediumland', '4.0': 'Upland', 'Unknown': 'Unknown'},
    'WSEV': {'0.0': 'None', '1.0': 'Low', '2.0': 'Medium', '3.0': 'High', 'Unknown': 'Unknown'},
    'INSEV': {'0.0': 'None', '1.0': 'Low', '2.0': 'Medium', '3.0': 'High', 'Unknown': 'Unknown'},
    'DISEV': {'0.0': 'None', '1.0': 'Low', '2.0': 'Medium', '3.0': 'High', 'Unknown': 'Unknown'},
    'DRSEV': {'0.0': 'None', '1.0': 'Low', '2.0': 'Medium', '3.0': 'High', 'Unknown': 'Unknown'},
    'FLSEV': {'0.0': 'None', '1.0': 'Low', '2.0': 'Medium', '3.0': 'High', 'Unknown': 'Unknown'},
    'SEASON': {'Kharif': 'Kharif', 'Rabi': 'Rabi', 'Unknown': 'Unknown'},
    'GEN': {'Male': 'Male', 'Female': 'Female', 'Unknown': 'Unknown'},
    'SOCCAT': {'SC': 'SC', 'ST': 'ST', 'OBC': 'OBC', 'General': 'General', 'Unknown': 'Unknown'},
    'VARTYPE': {
        'basmati': 'Basmati', 
        'local': 'Local', 
        'improved': 'Improved', 
        'hybrid': 'Hybrid', 
        'Traditional_Local': 'Traditional Local', 
        'Traditional_local': 'Traditional Local',
        'Unknown': 'Unknown'
    },
    'EST_line': {'Line': 'Line', 'Random': 'Random', 'Unknown': 'Unknown'},
    'EST_binary': {'Direct_seed': 'Direct Seed', 'Transplanted': 'Transplanted', 'Unknown': 'Unknown'},
    'IRRIAVA': {'yes': 'Yes', 'no': 'No', 'Unknown': 'Unknown'},
}

def get_colloquial(name):
    """Returns the colloquial name from FEATURE_NAME_MAP or falls back to title case."""
    if name in FEATURE_NAME_MAP:
        return FEATURE_NAME_MAP[name]
    return name.replace('_', ' ').title()

def get_unit(name):
    """Infers unit from colloquial name."""
    colloq = get_colloquial(name)
    if '(kg/ha)' in colloq: return 'kg/ha'
    if '(ha)' in colloq: return 'ha'
    if 'Precipitation' in colloq or 'Rain' in colloq: return 'mm'
    if 'Days' in colloq or 'Onset' in colloq or 'Length' in colloq or 'DOY' in colloq: return 'days'
    if '%' in colloq: return '%'
    if 'pH' in colloq: return 'pH scale'
    if 'Density' in colloq: return 'g/cm3' # Matches kg/dm3
    if 'Latitude' in colloq or 'Longitude' in colloq: return 'degrees'
    if 'Count' in colloq or 'Number' in colloq: return 'count'
    if 'Distance' in colloq: return 'km'
    if 'Seed Rate' in colloq: return 'kg/ha'
    return '-'

def get_feature_names(column_transformer):
    """Extracts feature names from ColumnTransformer."""
    output_features = []
    
    # Handle numeric features (passthrough or scaled)
    if 'num' in column_transformer.named_transformers_:
        output_features.extend(NUMERIC_FEATURES)
    
    # Handle categorical features (OneHotEncoder)
    if 'cat' in column_transformer.named_transformers_:
        cat_trans = column_transformer.named_transformers_['cat']
        if hasattr(cat_trans, 'named_steps'): # It's a pipeline
            encoder = cat_trans.named_steps['encoder']
        else:
            encoder = cat_trans # It's just the encoder
            
        if hasattr(encoder, 'get_feature_names_out'):
            cat_features = encoder.get_feature_names_out(CATEGORICAL_FEATURES)
            output_features.extend(cat_features)
            
    return output_features

def fit_best_distribution(data):
    """Fits norm, weibull, lognorm, expon, chi2 and returns best fit based on R2."""
    distributions = [stats.norm, stats.weibull_min, stats.lognorm, stats.expon, stats.chi2]
    
    y, x = np.histogram(data, bins='auto', density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0
    
    best_dist = None
    best_params = {}
    best_r2 = -np.inf
    
    for dist in distributions:
        try:
            params = dist.fit(data)
            pdf = dist.pdf(x, *params)
            current_r2 = r2_score(y, pdf)
            if current_r2 > best_r2:
                best_r2 = current_r2
                best_dist = dist
                best_params = params
        except:
            continue
            
    return best_dist, best_params, best_r2

def main():
    # Ensure plot directory exists
    os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

    # 1. Load Model and Data
    pipeline_path = os.path.join(MODEL_SAVE_DIR, "best_model_RandomForest_pipeline.joblib")
    data_path = "/scratch/mukund_n.iitr/CropLizer/Data/NUE_survey_dataset_v2.csv"
    
    print(f"Loading pipeline from {pipeline_path}...")
    pipeline = joblib.load(pipeline_path)
    model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['preprocessor']
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)
    
    # --- CRITICAL FIX: Ensure categorical columns are mapped keys (e.g. '1.0' string) ---
    print("Preprocessing categorical columns for consistency...")
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            # Check if this column maps to float-like keys (e.g., '1.0' instead of '1')
            is_numeric_key_map = False
            if col in REVERSE_INTERPRETABLE_MAPS:
                # Inspect keys to see if they are formatted like "1.0"
                keys = [k for k in REVERSE_INTERPRETABLE_MAPS[col].keys() if k != 'Unknown']
                if keys and keys[0].replace('.', '', 1).isdigit():
                    is_numeric_key_map = True
            
            if is_numeric_key_map:
                # Force conversion to float then to formatted string "1.0"
                # This handles cases where CSV loads int 1, but map expects '1.0'
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else 'Unknown')
            else:
                # Standard conversion
                df[col] = df[col].fillna('Unknown').astype(str)
    
    # Filter valid data
    # Only dropna on features we actually have
    available_features = [f for f in NUMERIC_FEATURES + CATEGORICAL_FEATURES if f in df.columns]
    df = df[available_features].dropna()
    
    # --- VARIABLE IMPORTANCE ---
    print("Calculating Variable Importance...")
    feature_names = get_feature_names(preprocessor)
    importances = model.feature_importances_
    
    # Aggregate one-hot encoded importances back to original categorical features
    aggregated_importances = {}
    for feat in NUMERIC_FEATURES:
        if feat in feature_names:
            idx = feature_names.index(feat)
            aggregated_importances[feat] = importances[idx]
            
    for cat in CATEGORICAL_FEATURES:
        total_imp = 0
        for i, name in enumerate(feature_names):
            if name.startswith(f"{cat}_"):
                total_imp += importances[i]
        aggregated_importances[cat] = total_imp

    # Sort and Plot Importance
    sorted_imp = dict(sorted(aggregated_importances.items(), key=lambda item: item[1], reverse=True))
    
    # Prepare plot labels
    top_keys = list(sorted_imp.keys())[:20][::-1]
    top_values = list(sorted_imp.values())[:20][::-1]
    top_labels = [get_colloquial(k) for k in top_keys]

    plt.figure(figsize=(12, 8))
    plt.barh(top_labels, top_values)
    plt.xlabel('Importance Score')
    plt.title('Top 20 Variable Importances (Random Forest)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_SAVE_DIR, "RF_Variable_Importance.png"))
    plt.close()
    
    # --- MONTE CARLO SIMULATION ---
    print("Starting Monte Carlo Simulation...")
    N_SAMPLES = 10000
    synthetic_data = {}
    
    # Numerical: Fit distributions and sample
    print("Fitting distributions for numerical variables...")
    for col in tqdm(NUMERIC_FEATURES):
        if col in df.columns:
            data = df[col].values
            dist, params, score = fit_best_distribution(data)
            if dist:
                samples = dist.rvs(*params, size=N_SAMPLES)
                samples = np.clip(samples, data.min(), data.max())
                synthetic_data[col] = samples
            else:
                synthetic_data[col] = np.random.choice(data, size=N_SAMPLES, replace=True)
        else:
             synthetic_data[col] = np.zeros(N_SAMPLES)

    # Categorical: Stratified Sampling
    print("Sampling categorical variables...")
    for col in tqdm(CATEGORICAL_FEATURES):
        if col in df.columns:
            probs = df[col].value_counts(normalize=True)
            synthetic_data[col] = np.random.choice(probs.index, size=N_SAMPLES, p=probs.values)
        else:
            synthetic_data[col] = np.array(['Unknown'] * N_SAMPLES)
            
    synthetic_df = pd.DataFrame(synthetic_data)
    
    # Predict Targets
    print("Predicting targets for synthetic data...")
    predictions = pipeline.predict(synthetic_df)
    
    # --- SENSITIVITY ANALYSIS (NUMERICAL) ---
    print("Calculating Numerical Sensitivity (Slopes)...")
    sensitivity_results = []
    feature_sensitivity_map = {f: {} for f in NUMERIC_FEATURES}

    for i, target in enumerate(TARGETS):
        target_pred = predictions[:, i]
        
        for feature in NUMERIC_FEATURES:
            X = synthetic_df[feature].values.reshape(-1, 1)
            y = target_pred
            
            reg = LinearRegression().fit(X, y)
            slope = reg.coef_[0]
            sensitivity_results.append({
                'Target': target,
                'Feature': feature,
                'Sensitivity': slope
            })
            feature_sensitivity_map[feature][target] = slope
            
    sens_df = pd.DataFrame(sensitivity_results)
    
    # Plotting Faceted Bar Plot
    targets_unique = sens_df['Target'].unique()
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, target in enumerate(targets_unique):
        subset = sens_df[sens_df['Target'] == target].sort_values(by='Sensitivity', key=abs, ascending=False).head(10)
        
        feature_labels = [get_colloquial(f) for f in subset['Feature']]
        target_label = get_colloquial(target)
        
        axes[i].bar(feature_labels, subset['Sensitivity'], color='skyblue', edgecolor='black')
        axes[i].set_title(f"Sensitivity for {target_label}")
        axes[i].set_ylabel(f"Change in {target_label} / Unit Input")
        axes[i].tick_params(axis='x', rotation=45, labelsize=10)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_SAVE_DIR, "MonteCarlo_Numerical_Sensitivity.png"))
    plt.close()
    
    # --- SENSITIVITY ANALYSIS (CATEGORICAL) ---
    print("Generating Categorical Box Plots...")
    
    top_categorical = [k for k in sorted_imp.keys() if k in CATEGORICAL_FEATURES][:5]
    
    for cat_feature in tqdm(top_categorical):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        cat_label = get_colloquial(cat_feature)
        
        for i, target in enumerate(TARGETS):
            target_vals = predictions[:, i]
            target_label = get_colloquial(target)
            
            categories = synthetic_df[cat_feature].unique()
            readable_labels = []
            for c in categories:
                c_str = str(c)
                if cat_feature in REVERSE_INTERPRETABLE_MAPS:
                    lbl = REVERSE_INTERPRETABLE_MAPS[cat_feature].get(c_str, c_str)
                else:
                    lbl = c_str
                readable_labels.append(lbl)
                
            data_to_plot = [target_vals[synthetic_df[cat_feature] == c] for c in categories]
            
            valid_cats = []
            valid_data = []
            for c_lbl, d in zip(readable_labels, data_to_plot):
                if len(d) > 10:
                    valid_cats.append(c_lbl)
                    valid_data.append(d)
            
            if valid_data:
                axes[i].boxplot(valid_data, labels=valid_cats)
                axes[i].set_title(f"{target_label} vs {cat_label}")
                axes[i].set_ylabel(target_label)
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_SAVE_DIR, f"Sensitivity_Categorical_{cat_feature}.png"))
        plt.close()

    # --- FULL TABLE GENERATION ---
    print("Generating comprehensive summary table...")
    
    target_stats = {}
    for i, target in enumerate(TARGETS):
        target_vals = predictions[:, i]
        target_stats[f"Mean_{target}"] = np.mean(target_vals)
        target_stats[f"SD_{target}"] = np.std(target_vals)

    table_data = []
    all_features = list(sorted_imp.keys())
    
    for feature in tqdm(all_features):
        imp = sorted_imp.get(feature, 0)
        colloq = get_colloquial(feature)
        
        if feature in NUMERIC_FEATURES:
            row = {}
            row['Variable Name'] = feature
            row['Variable Colloquial Name'] = colloq
            row['Variable Type'] = 'Numerical'
            row['Variable Importance'] = imp
            row['Reference Unit'] = get_unit(feature)
            
            for target in TARGETS:
                slope = feature_sensitivity_map.get(feature, {}).get(target, np.nan)
                row[f"Change in {target.split('_')[0]}"] = slope
            
            # Global stats for numerical
            for target in TARGETS:
                t_short = target.split('_')[0] 
                if 'yield' in t_short: t_short = 'Yield'
                elif 'P2O5' in t_short: t_short = 'P'
                elif 'K2O' in t_short: t_short = 'K'
                
                row[f"Mean {t_short}"] = f"{target_stats[f'Mean_{target}']:.2f}"
                row[f"SD {t_short}"] = f"{target_stats[f'SD_{target}']:.2f}"
            
            table_data.append(row)

        else:
            # Categorical - Generate ONE row per Category
            
            # Get unique categories present in the synthetic data
            if feature in synthetic_df.columns:
                series = synthetic_df[feature].astype(str)
                unique_cats = sorted(series.unique())
            else:
                unique_cats = []

            if not unique_cats:
                 # Fallback if no data
                 row = {}
                 row['Variable Name'] = feature
                 row['Variable Colloquial Name'] = colloq
                 row['Variable Type'] = 'Categorical'
                 row['Variable Importance'] = imp
                 row['Reference Unit'] = 'No Data'
                 table_data.append(row)
                 continue

            for cat_str in unique_cats:
                row = {}
                row['Variable Name'] = feature
                row['Variable Colloquial Name'] = colloq
                row['Variable Type'] = 'Categorical'
                row['Variable Importance'] = imp
                
                # Map Name
                if feature in REVERSE_INTERPRETABLE_MAPS:
                     label = REVERSE_INTERPRETABLE_MAPS[feature].get(cat_str, cat_str)
                else:
                     label = cat_str
                
                row['Reference Unit'] = label # The Category Name
                
                # Sensitivities are NaN
                for target in TARGETS:
                    row[f"Change in {target.split('_')[0]}"] = np.nan

                # Calculate specific stats for this category
                mask = (series == cat_str)
                
                for i, target in enumerate(TARGETS):
                    t_short = target.split('_')[0]
                    if 'yield' in t_short: t_short = 'Yield'
                    elif 'P2O5' in t_short: t_short = 'P'
                    elif 'K2O' in t_short: t_short = 'K'
                    
                    # Subset predictions
                    vals = predictions[mask, i]
                    
                    if len(vals) > 0:
                        row[f"Mean {t_short}"] = f"{np.mean(vals):.2f}"
                        row[f"SD {t_short}"] = f"{np.std(vals):.2f}"
                    else:
                        row[f"Mean {t_short}"] = "0.00"
                        row[f"SD {t_short}"] = "0.00"

                table_data.append(row)
        
    summary_df = pd.DataFrame(table_data)
    
    cols_order = [
        'Variable Name', 'Variable Colloquial Name', 'Variable Type', 'Variable Importance', 
        'Reference Unit',
        'Change in yield', 'Change in N', 'Change in P', 'Change in K',
        'Mean Yield', 'SD Yield', 'Mean N', 'SD N', 'Mean P', 'SD P', 'Mean K', 'SD K'
    ]
    actual_cols = summary_df.columns.tolist()
    final_cols = [c for c in cols_order if c in actual_cols] + [c for c in actual_cols if c not in cols_order]
    summary_df = summary_df[final_cols]
    
    save_path = os.path.join(PLOT_SAVE_DIR, "Full_Variable_Sensitivity_Table.csv")
    summary_df.to_csv(save_path, index=False)
    print(f"Table saved to: {save_path}")

    print("Analysis Complete. Plots and Table saved.")

if __name__ == "__main__":
    main()
