import pandas as pd
import numpy as np
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
import joblib
import requests
from datetime import datetime, timedelta
import os
from tqdm.auto import tqdm
from soilgrids import SoilGrids
from pyproj import Transformer
import itertools
from deep_translator import GoogleTranslator
import croplizer_chat
import importlib 

# --- 0. GLOBAL VARS ---
model = None
data_df = None
reliability_knn = None
reliability_dist_scale = 1.0
scaler_y = None
MODEL_FILENAME = 'farm_advisor_model.joblib'
TRANS_CACHE = {}

# --- 1. LANGUAGE SETTINGS ---
LANGUAGES = {
    'English': 'en',
    'हिंदी': 'hi',
    'తెలుగు': 'te',
    'தமிழ்': 'ta',
    'संस्कृत':'sa',
    'ਪੰਜਾਬੀ': 'pa',
    'বাংলা': 'bn',
    'ಕನ್ನಡ': 'kn',
    'മലയാളം': 'ml',
    'मराठी': 'mr',
    'ગુજરાતી': 'gu',
    'ઓડિયા': 'or',
    'اردو': 'ur',
    'Español': 'es',
    'Français': 'fr',
    '中国人':'zh-Hans',
    '日本語':'ja'
}

def translate_text(text, target_lang_code):
    if target_lang_code == 'en':
        return text
    
    key = (text, target_lang_code)
    if key in TRANS_CACHE:
        return TRANS_CACHE[key]

    try:
        translator = GoogleTranslator(source='auto', target=target_lang_code)
        translated = translator.translate(text)
        TRANS_CACHE[key] = translated
        return translated
    except Exception as e:
        print(f"Translation Error: {e}")
        return text

# --- 2. FEATURE DEFINITIONS ---

TARGETS = ['yield_kg_ha', 'N_kg_ha', 'P2O5_ha', 'K2O_ha']

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

# Variables to vary in grid search
PRACTICE_FEATURES_TO_OPTIMIZE = [
    'VARTYPE', 'SRATEHA', 'Organic_ha', 'TDATE_yday',
    'EST_line', 'EST_binary', 'IRRINU', 'PCRR', 'Field_duration',
    'GYP_ha', 'BOR_ha', 'Zn_ha'
]

# Variables that must remain constant during optimization
CONSTANTS_FOR_OPTIMIZATION = [
    'LAT', 'LONG', 'SEASON', 'CRLPARHA', 'GEN', 'EDU', 'SOCCAT', 'CRARHA', 'HHMEM',
    'DISMR', 'SOPER', 'DCLASS',
    'SoilGrids_bdod', 'SoilGrids_clay', 'SoilGrids_nitrogen', 'SoilGrids_ocd',
    'SoilGrids_phh2o', 'SoilGrids_sand', 'SoilGrids_silt', 'SoilGrids_soc',
    'total_precip', 'num_dry_days', 'avg_dspell_length', 'monsoon_onset', 'monsoon_length',
    'IRRIAVA', 'WSEV', 'INSEV', 'DISEV', 'DRSEV', 'FLSEV','W_YIELD', 'W_PROFIT', 'W_ENV'
]

CATEGORICAL_PRACTICES = [col for col in PRACTICE_FEATURES_TO_OPTIMIZE if col in CATEGORICAL_FEATURES]
NUMERIC_PRACTICES = [col for col in PRACTICE_FEATURES_TO_OPTIMIZE if col in NUMERIC_FEATURES]

# Detailed Economic Defaults
ECONOMIC_DEFAULTS = {
    'FGPRICE_quintal': 3368.0,
    'Urea_price': 250.0,    # Per 50kg
    'DAP_price': 1400.0,    # Per 50kg
    'MOP_price': 250.0,     # Per 50kg
    'Seed_price': 105.08,   # Per kg
    'Labour_cost': 48533.0,
    'Insecticides_cost': 2286.91,
    'Irrigation_cost': 2771.01,
    'Insurance_cost': 64.61,
    'Misc_cost': 314.44,
    'Interest_working_cost': 1362.0,
    'Rent_owned_cost': 19761.0,
    'Rent_leased_cost': 918.9,
    'Land_revenue_cost': 60.29,
    'Depreciation_cost': 550.37,
    'Interest_fixed_cost': 2610.97
}

ECONOMIC_COLS = list(ECONOMIC_DEFAULTS.keys())

# --- 3. UI FEATURE NAME MAPPING ---

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
    'SoilGrids_bdod': 'Soil Bulk Density (kg/dm³)',
    'SoilGrids_clay': 'Soil Clay Content (%)',
    'SoilGrids_nitrogen': 'Soil Nitrogen (g/kg)',
    'SoilGrids_ocd': 'Soil Organic Carbon Density (kg/m³)',
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
    'P2O5_ha': 'Phosphorous (P₂O₅) Rate',
    'K2O_ha': 'Potassium (K₂O) Rate',
    'GYP_ha': 'Applied Gypsum (kg/ha)',
    'BOR_ha': 'Applied Boron (kg/ha)',
    'Zn_ha': 'Applied Zinc (kg/ha)',
    'Organic_ha': 'Applied Organic Matter (kg/ha)',
    'W_YIELD': 'Yield Priority',
    'W_PROFIT': 'Profit/BCR Priority',
    'W_ENV': 'Environmental Priority (Low NPK)',
    # Economic Labels
    'FGPRICE_quintal': 'Rice Price (INR/quintal)',
    'Urea_price': 'Urea Price (INR/50kg)',
    'DAP_price': 'DAP Price (INR/50kg)',
    'MOP_price': 'MOP Price (INR/50kg)',
    'Seed_price': 'Seed Cost (INR/kg)',
    'Labour_cost': 'Labour Cost (INR/ha)',
    'Insecticides_cost': 'Insecticides Cost (INR/ha)',
    'Irrigation_cost': 'Irrigation Charges (INR/ha)',
    'Insurance_cost': 'Crop Insurance Cost (INR/ha)',
    'Misc_cost': 'Miscellaneous Cost (INR/ha)',
    'Interest_working_cost': 'Interest on Working Capital (INR/ha)',
    'Rent_owned_cost': 'Rental Value of Owned Land (INR/ha)',
    'Rent_leased_cost': 'Rent Paid for Leased-in Land (INR/ha)',
    'Land_revenue_cost': 'Land Revenue, Taxes, Cesses (INR/ha)',
    'Depreciation_cost': 'Depreciation on Implements & Building (INR/ha)',
    'Interest_fixed_cost': 'Interest on Fixed Capital (INR/ha)'
}

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

INTERPRETABLE_MAPS = {
    col: {name: val for val, name in mappings.items()}
    for col, mappings in REVERSE_INTERPRETABLE_MAPS.items()
}


# --- 4. DATA LOADING & PREPROCESSING ---

def load_and_preprocess_data(filepath):
    global NUMERIC_FEATURES, CATEGORICAL_FEATURES, FEATURES

    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath, low_memory=False, on_bad_lines='skip')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None
    
    print(f"Successfully loaded {len(df)} rows initially.")

    all_cols_to_load = FEATURES + TARGETS
    all_cols_to_load = list(dict.fromkeys(all_cols_to_load))
    # Filter out columns not in csv
    all_cols_to_load = [c for c in all_cols_to_load if c in df.columns]

    df = df[all_cols_to_load]

    NUMERIC_FEATURES = [col for col in NUMERIC_FEATURES if col in df.columns]
    CATEGORICAL_FEATURES = [col for col in CATEGORICAL_FEATURES if col in df.columns]
    FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    numeric_cols_to_convert = [col for col in NUMERIC_FEATURES + TARGETS if col in df.columns]
    for col in numeric_cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    initial_rows = len(df)
    df = df.dropna(subset=TARGETS)
    print(f"Dropped {initial_rows - len(df)} rows due to missing yield or N/P/K data.")

    if 'yield_kg_ha' in df.columns and len(df) > 0:
        print("Filtering yield outliers based on 90th percentile and IQR...")
        before_filter_count = len(df)
        
        Q1 = df['yield_kg_ha'].quantile(0.25)
        Q3 = df['yield_kg_ha'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - (1.5 * IQR)
        upper_bound_iqr = Q3 + (1.5 * IQR)
        upper_bound_p90 = df['yield_kg_ha'].quantile(0.90)
        
        final_upper_bound = min(upper_bound_iqr, upper_bound_p90)
        
        df = df[
            (df['yield_kg_ha'] >= lower_bound) &
            (df['yield_kg_ha'] <= final_upper_bound)
        ]
        
        after_filter_count = len(df)
        print(f"Dropped {before_filter_count - after_filter_count} rows from yield outlier filtering.")
        print(f"Kept yield data between {lower_bound:.2f} and {final_upper_bound:.2f} kg/ha.")

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna('Unknown').astype(str)

    print("Data loading and cleaning complete.")
    return df

# --- 5. MODEL TRAINING & TUNING ---

def train_prediction_model(data_df):
    print("Starting model training with fixed hyperparameters...")

    X = data_df[FEATURES]
    y = data_df[TARGETS]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='passthrough'
    )

    best_params = {
        'n_estimators': 112,
        'max_depth': 30,
        'min_samples_split': 2,
        'min_samples_leaf': 2,
        'max_features': 0.28898808111582136
    }
    
    print(f"Using fixed hyperparameters: {best_params}")

    final_model = RandomForestRegressor(
        n_jobs=-1,
        **best_params 
    )
    
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', final_model)
    ])

    print("Fitting final model on training data...")
    final_pipeline.fit(X_train, y_train)

    score = final_pipeline.score(X_test, y_test)
    print(f"Final model R^2 score on test set: {score:.4f}")

    return final_pipeline


# --- 6. API DATA FETCHING ---

def fetch_external_data(lat, lon, progress=gr.Progress()):
    if lat is None or lon is None:
        gr.Warning("Latitude or Longitude is missing. Cannot fetch data.")
        return (gr.update(),) * 13 

    lat = round(float(lat), 4)
    lon = round(float(lon), 4)
    
    outputs = {
        'total_precip': gr.update(),
        'num_dry_days': gr.update(),
        'avg_dspell_length': gr.update(),
        'monsoon_onset': gr.update(),
        'monsoon_length': gr.update(),
        'SoilGrids_bdod': gr.update(),
        'SoilGrids_clay': gr.update(),
        'SoilGrids_nitrogen': gr.update(),
        'SoilGrids_ocd': gr.update(),
        'SoilGrids_phh2o': gr.update(),
        'SoilGrids_sand': gr.update(),
        'SoilGrids_silt': gr.update(),
        'SoilGrids_soc': gr.update()
    }

    try:
        progress(0.2, desc="Fetching Climate Data...")
        
        prev_year = datetime.now().year - 1
        start_date = f"{prev_year}-01-01"
        end_date = f"{prev_year}-12-31"
        
        climate_url = "https://archive-api.open-meteo.com/v1/archive"
        climate_params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "precipitation_sum",
            "timezone": "auto"
        }
        r = requests.get(climate_url, params=climate_params, timeout=10)
        r.raise_for_status() 
        
        climate_data = r.json().get('daily', {})
        precip_data = climate_data.get('precipitation_sum', [])
        
        if precip_data and all(v is not None for v in precip_data):
            # 1. Total Precip and Dry Days
            total_precip = sum(precip_data)
            num_dry_days = sum(1 for p in precip_data if p == 0)
            
            outputs['total_precip'] = round(total_precip, 1)
            outputs['num_dry_days'] = num_dry_days

            # 2. Avg Dry Spell Length
            dry_streaks = []
            current_streak = 0
            for p in precip_data:
                if p == 0:
                    current_streak += 1
                else:
                    if current_streak > 0:
                        dry_streaks.append(current_streak)
                    current_streak = 0
            if current_streak > 0:
                dry_streaks.append(current_streak)
            
            avg_dspell = sum(dry_streaks) / len(dry_streaks) if dry_streaks else 0
            outputs['avg_dspell_length'] = round(avg_dspell, 1)

            # 3. Monsoon Onset and Length
            onset_day = None
            last_rain_day = None
            
            check_start_idx = 120 
            
            for i in range(check_start_idx, len(precip_data) - 2):
                three_day_sum = precip_data[i] + precip_data[i+1] + precip_data[i+2]
                if three_day_sum > 25.0:
                    onset_day = i + 1 # 1-based index (DOY)
                    break
            
            if onset_day:
                outputs['monsoon_onset'] = onset_day
                # Find last significant rain day (e.g. > 2mm) after onset
                for j in range(len(precip_data) - 1, onset_day, -1):
                    if precip_data[j] > 2.0:
                        last_rain_day = j + 1
                        break
                
                if last_rain_day:
                    outputs['monsoon_length'] = last_rain_day - onset_day
                else:
                    outputs['monsoon_length'] = 0
            else:
                 outputs['monsoon_onset'] = 0 # No clear onset
                 outputs['monsoon_length'] = 0

            gr.Info(f"Climate data calculated for {prev_year}.")
        else:
            gr.Warning("Could not fetch valid climate data.")

    except requests.exceptions.RequestException as e:
        error_msg = f"Open-Meteo API Error: Could not connect. {e}"
        print(error_msg) 
        gr.Warning(error_msg) 
    except Exception as e:
        error_msg = f"Open-Meteo Data Error: {e}"
        print(error_msg) 
        gr.Warning(error_msg) 

    try:
        progress(0.6, desc="Fetching SoilGrids Data...")
        
        soil_grids = SoilGrids()
        properties_to_fetch = ["bdod", "clay", "nitrogen", "ocd", "phh2o", "sand", "silt", "soc"]
        soil_results = {}
        
        target_crs = "EPSG:3857" 
        transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        
        x, y = transformer.transform(lon, lat)
        
        buffer_m = 125 
        west_proj = x - buffer_m
        east_proj = x + buffer_m
        south_proj = y - buffer_m
        north_proj = y + buffer_m
        
        temp_tif_file = "temp_soilgrids_data.tif"
        
        progress(0.6, desc="Fetching SoilGrids Data (Property 1/8)...")

        for i, prop_name in enumerate(properties_to_fetch):
            try:
                service_id = prop_name
                coverage_id = f"{prop_name}_0-5cm_mean"
                
                data = soil_grids.get_coverage_data(
                    service_id=service_id,
                    coverage_id=coverage_id,
                    west=west_proj,  
                    south=south_proj, 
                    east=east_proj,  
                    north=north_proj, 
                    crs="urn:ogc:def:crs:EPSG::152160", 
                    width=1,  
                    height=1, 
                    output=temp_tif_file,
                )
                
                if data is None:
                    raise Exception("SoilGrids data (xarray.DataArray) was not returned.")

                mid_x_index = len(data.coords['x']) // 2
                mid_y_index = len(data.coords['y']) // 2
                
                mean_val = data.isel(band=0, x=mid_x_index, y=mid_y_index).item()
                
                if not np.isnan(mean_val):
                    soil_results[prop_name] = mean_val
                else:
                    print(f"Warning: No valid data found for SoilGrids property '{prop_name}'.")

                progress(0.6 + (0.4 * (i+1) / len(properties_to_fetch)), desc=f"Fetching SoilGrids ({i+1}/{len(properties_to_fetch)})...")

            except Exception as e:
                print(f"Warning: Could not fetch SoilGrids property '{prop_name}'. Error: {e}")
                gr.Warning(f"Could not fetch SoilGrids property '{prop_name}'.")
            finally:
                if os.path.exists(temp_tif_file):
                    try:
                        os.remove(temp_tif_file)
                    except Exception as e:
                        print(f"Warning: Could not remove temp file {temp_tif_file}. Error: {e}")

        
        if not soil_results:
            gr.Warning("No SoilGrids data could be fetched for this location.")
        else:
            if 'bdod' in soil_results:
                outputs['SoilGrids_bdod'] = round(soil_results['bdod'] / 100, 2)
            if 'clay' in soil_results:
                outputs['SoilGrids_clay'] = round(soil_results['clay'] / 10, 1)
            if 'nitrogen' in soil_results:
                outputs['SoilGrids_nitrogen'] = round(soil_results['nitrogen'] / 100, 3)
            if 'ocd' in soil_results:
                outputs['SoilGrids_ocd'] = round(soil_results['ocd'] / 10, 1)
            if 'phh2o' in soil_results:
                outputs['SoilGrids_phh2o'] = round(soil_results['phh2o'] / 10, 2)
            if 'sand' in soil_results:
                outputs['SoilGrids_sand'] = round(soil_results['sand'] / 10, 1)
            if 'silt' in soil_results:
                outputs['SoilGrids_silt'] = round(soil_results['silt'] / 10, 1)
            if 'soc' in soil_results:
                outputs['SoilGrids_soc'] = round(soil_results['soc'] / 10, 2)
                
            gr.Info("SoilGrids data fetched successfully.")

    except Exception as e:
        error_msg = f"SoilGrids Error: Could not fetch data using the 'soilgrids' library. This might still be a network issue (like 'NameResolutionError') or a problem with the library itself. Error: {e}"
        print(error_msg) 
        gr.Warning(error_msg) 

    progress(1.0, desc="Data Fetching Complete")
    
    return (
        outputs['total_precip'], outputs['num_dry_days'], outputs['avg_dspell_length'],
        outputs['monsoon_onset'], outputs['monsoon_length'],
        outputs['SoilGrids_bdod'], outputs['SoilGrids_clay'], outputs['SoilGrids_nitrogen'],
        outputs['SoilGrids_ocd'], outputs['SoilGrids_phh2o'], outputs['SoilGrids_sand'],
        outputs['SoilGrids_silt'], outputs['SoilGrids_soc']
    )


# --- 7. FERTILIZER MIX CALCULATOR ---

def get_fertilizer_quantities(n_pred, p_pred, k_pred, plot_area_ha):
    n_target_rate = max(0, n_pred)
    p_target_rate = max(0, p_pred)
    k_target_rate = max(0, k_pred)

    mop_rate_kg_ha = k_target_rate / 0.60 if k_target_rate > 0 else 0
    dap_rate_kg_ha = p_target_rate / 0.46 if p_target_rate > 0 else 0
    n_from_dap_rate = dap_rate_kg_ha * 0.18
    n_remaining_rate = n_target_rate - n_from_dap_rate
    urea_rate_kg_ha = (n_remaining_rate / 0.46) if n_remaining_rate > 0 else 0
    if n_remaining_rate < 0:
        urea_rate_kg_ha = 0

    total_mop = mop_rate_kg_ha * plot_area_ha
    total_dap = dap_rate_kg_ha * plot_area_ha
    total_urea = urea_rate_kg_ha * plot_area_ha

    return total_urea, total_dap, total_mop

def format_fertilizer_plan(opt_urea, opt_dap, opt_mop, curr_urea, curr_dap, curr_mop, plot_area_ha):
    
    def format_line(name, opt_val, curr_val):
        opt_per_ha = opt_val / plot_area_ha if plot_area_ha > 0 else 0
        curr_per_ha = curr_val / plot_area_ha if plot_area_ha > 0 else 0
        
        if curr_per_ha > 0:
            pct_change = ((opt_per_ha - curr_per_ha) / curr_per_ha) * 100
            diff_str = f"[{pct_change:+.1f}%]"
        elif opt_per_ha > 0:
            diff_str = "[+New]"
        else:
            diff_str = "[0%]"
            
        return f"- **{name}:** Current: {curr_val:.1f} kg ({curr_per_ha:.1f} kg/ha) → **Optimal: {opt_val:.1f} kg** ({opt_per_ha:.1f} kg/ha) {diff_str}"

    plan_summary = (
        f"**Suggested Optimized Fertilizer Plan (Total for {plot_area_ha:.2f} ha Plot):**\n"
        f"{format_line('Urea (46-0-0)', opt_urea, curr_urea)}\n"
        f"{format_line('DAP (18-46-0)', opt_dap, curr_dap)}\n"
        f"{format_line('MOP (0-0-60)', opt_mop, curr_mop)}\n\n"
        f"*Values in brackets are per hectare. Percentages show change required from current predicted practice to match the optimized yield plan.*"
    )
    return plan_summary

# --- 8. OPTIMIZATION HELPERS ---

def doy_to_date_str(doy, year=None):
    """Convert Day of Year to DD-Mon format."""
    if year is None: year = datetime.now().year
    try:
        date_obj = datetime(year, 1, 1) + timedelta(days=int(doy) - 1)
        return date_obj.strftime("%d-%b")
    except:
        return str(doy)

def date_str_to_doy(date_val):
    """Convert Date string or datetime object to Day of Year integer."""
    try:
        if isinstance(date_val, str):
            date_obj = datetime.strptime(date_val, "%Y-%m-%d")
        elif isinstance(date_val, datetime):
            date_obj = date_val
        else:
            return float(date_val) if date_val else np.nan
            
        return float(date_obj.timetuple().tm_yday)
    except:
        return np.nan

def generate_grid_and_optimize(user_inputs, progress):
    """
    Generates a dataframe of practice combinations using Randomized Search logic.
    Ensures the search space includes the user's current practice to prevent artificial jumps.
    """
    
    # Extract economic factors from inputs
    crlparha = pd.to_numeric(user_inputs.get('CRLPARHA', 1.0), errors='coerce')
    if crlparha <= 0: crlparha = 1.0

    prices = {k: pd.to_numeric(user_inputs.get(k, 0), errors='coerce') for k in ECONOMIC_COLS}
    for k in prices:
        if pd.isna(prices[k]) or prices[k] < 0: prices[k] = 0.0

    # Calculate Fixed Costs per Hectare (Excluding Seeds and Fertilizers)
    fixed_operational_cost_ha = (
        prices['Labour_cost'] +
        prices['Insecticides_cost'] +
        prices['Irrigation_cost'] +
        prices['Insurance_cost'] +
        prices['Misc_cost'] +
        prices['Interest_working_cost'] +
        prices['Rent_owned_cost'] +
        prices['Rent_leased_cost'] +
        prices['Land_revenue_cost'] +
        prices['Depreciation_cost'] +
        prices['Interest_fixed_cost']
    )

    NUM_SIMULATIONS = 20000
    sim_data = {}

    # 1. Handle Categorical Practices
    for col in CATEGORICAL_PRACTICES:
        options = data_df[col].unique().tolist()
        if 'Unknown' in options: options.remove('Unknown')
        sim_data[col] = np.random.choice(options, size=NUM_SIMULATIONS)

    # 2. Handle Numerical Practices with Bounds Checking
    season = user_inputs.get('SEASON', 'Kharif')
    
    for col in NUMERIC_PRACTICES:
        series = data_df[col].dropna()
        curr_val = pd.to_numeric(user_inputs.get(col, 0.0), errors='coerce')
        if pd.isna(curr_val): curr_val = 0.0

        if series.empty:
            sim_data[col] = np.full(NUM_SIMULATIONS, curr_val)
            continue

        p05 = series.quantile(0.05)
        p95 = series.quantile(0.95)

        # Dynamic Bounds: Ensure current value is included in the range
        min_bound = min(p05, curr_val)
        max_bound = max(p95, curr_val)

        # Special Variable Logic
        if col == 'TDATE_yday':
            if season == 'Kharif':
                min_bound, max_bound = max(152, min_bound), min(304, max_bound)
            elif season == 'Rabi':
                # Simplified Rabi logic
                pass 

        elif col == 'Field_duration':
            min_bound, max_bound = max(80.0, min_bound), min(160.0, max_bound)
        elif col == 'IRRINU':
            min_bound, max_bound = max(0.0, min_bound), min(30.0, max_bound)
        elif col in ['BOR_ha', 'Zn_ha']:
            min_bound, max_bound = max(0.0, min_bound), min(10.0, max_bound)
        elif col == 'GYP_ha':
            min_bound, max_bound = max(0.0, min_bound), min(1000.0, max_bound)
        elif col == 'Organic_ha':
            min_bound, max_bound = max(0.0, min_bound), min(20000.0, max_bound)
        elif col == 'SRATEHA':
            min_bound, max_bound = max(10.0, min_bound), min(150.0, max_bound)
        elif col == 'PCRR':
             min_bound, max_bound = max(0.0, min_bound), min(100.0, max_bound)

        if min_bound > max_bound: min_bound, max_bound = max_bound, min_bound
        
        # Use Uniform distribution for smoother sampling
        sim_data[col] = np.random.uniform(min_bound, max_bound, NUM_SIMULATIONS)

    # 3. Create DataFrame
    base_row = {k: user_inputs.get(k) for k in CONSTANTS_FOR_OPTIMIZATION}
    grid_df = pd.DataFrame([base_row] * NUM_SIMULATIONS)
    practices_df = pd.DataFrame(sim_data)
    full_grid_df = pd.concat([grid_df, practices_df], axis=1)

    # Explicitly add the current user input as the last row
    current_input_row = pd.DataFrame([user_inputs])
    current_input_row = current_input_row.reindex(columns=FEATURES, fill_value=np.nan)
    
    # Handle numeric conversion for the current row
    for col in NUMERIC_FEATURES:
        if col in current_input_row.columns:
            current_input_row[col] = pd.to_numeric(current_input_row[col], errors='coerce')
    
    # Combine
    full_grid_df = pd.concat([full_grid_df, current_input_row], axis=0, ignore_index=True)

    # 4. Preprocessing
    for col in NUMERIC_FEATURES:
        if col in full_grid_df.columns:
            full_grid_df[col] = pd.to_numeric(full_grid_df[col], errors='coerce')
    for col in CATEGORICAL_FEATURES:
        if col in full_grid_df.columns:
            full_grid_df[col] = full_grid_df[col].fillna('Unknown').astype(str)

    full_grid_df = full_grid_df.reindex(columns=FEATURES, fill_value=np.nan)
    
    # 5. Predict
    progress(0.5, desc=f"Simulating {NUM_SIMULATIONS} randomized scenarios...")
    predictions = model.predict(full_grid_df)
    
    # 6. Calculate Indices
    N_rates = np.maximum(0, predictions[:, 1])
    P_rates = np.maximum(0, predictions[:, 2])
    K_rates = np.maximum(0, predictions[:, 3])

    mop_rates_kg_ha = K_rates / 0.60
    dap_rates_kg_ha = P_rates / 0.46
    n_from_dap = dap_rates_kg_ha * 0.18
    n_remaining = N_rates - n_from_dap
    urea_rates_kg_ha = np.where(n_remaining > 0, n_remaining / 0.46, 0)

    total_mop = mop_rates_kg_ha * crlparha
    total_dap = dap_rates_kg_ha * crlparha
    total_urea = urea_rates_kg_ha * crlparha

    # Include Fixed Costs in the optimization calculation
    # Divide 50kg bag price by 50 to get per kg price
    fertilizer_cost = (
        total_urea * (prices['Urea_price'] / 50.0) + 
        total_dap * (prices['DAP_price'] / 50.0) + 
        total_mop * (prices['MOP_price'] / 50.0)
    )

    # Seed Cost Calculation: Rate (kg/ha) * Area (ha) * Price (per kg)
    # Using 'SRATEHA' from the simulated practices DataFrame
    seed_rates = full_grid_df['SRATEHA'].fillna(0).values
    seed_cost = seed_rates * crlparha * prices['Seed_price']

    # Total Cost = Fertilizer + Seed + Operational/Fixed
    costs = fertilizer_cost + seed_cost + (fixed_operational_cost_ha * crlparha)
    
    yields = predictions[:, 0]
    revenues = (yields * crlparha / 100.0) * prices['FGPRICE_quintal']
    
    bcrs = np.divide(revenues, costs, out=np.zeros_like(revenues), where=costs!=0)
    total_npk = N_rates + P_rates + K_rates

    def normalize(v):
        min_v, max_v = np.min(v), np.max(v)
        if max_v - min_v < 1e-6: return np.zeros_like(v)
        return (v - min_v) / (max_v - min_v)

    norm_yield = normalize(yields)
    norm_bcr = normalize(bcrs)
    norm_npk = normalize(total_npk)
    
    # Weights: Dynamic based on user input (Defaults to 1.0)
    raw_w_yield = float(user_inputs.get('W_YIELD', 1.0))
    raw_w_profit = float(user_inputs.get('W_PROFIT', 1.0))
    raw_w_env = float(user_inputs.get('W_ENV', 1.0))

    w_yield = raw_w_yield ** 3
    w_profit = raw_w_profit ** 3
    w_env = raw_w_env ** 3

    composite_index = w_yield * norm_yield + w_profit * norm_bcr + w_env * (1 - norm_npk)
    
    best_idx = np.argmax(composite_index)
    best_score = composite_index[best_idx]
    user_score = composite_index[-1]
    best_pred = predictions[best_idx]
    
    # Extract best practices
    if best_idx == len(full_grid_df) - 1:
        best_practices = user_inputs
    else:
        best_practices = practices_df.iloc[best_idx].to_dict()
    
    return best_pred, best_practices, best_score, user_score


# --- 9. PREDICTION FUNCTION ---

def predict_yield_and_fertilizer(inputs, state, lang_selection, progress=gr.Progress(track_tqdm=True)):
    
    progress(0, desc="Initializing...")
    
    # Determine target language code
    target_lang_code = LANGUAGES.get(lang_selection, 'en')

    # --- 1. Prepare User Input DataFrame ---
    user_inputs_processed = inputs.copy()
    
    # Construct DF
    input_df = pd.DataFrame([user_inputs_processed])
    for col in NUMERIC_FEATURES:
        if col in input_df.columns: 
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
    for col in CATEGORICAL_FEATURES:
        if col in input_df.columns: 
            input_df[col] = input_df[col].fillna('Unknown').astype(str)
            
    input_df = input_df.reindex(columns=FEATURES, fill_value=np.nan)
    input_df[CATEGORICAL_FEATURES] = input_df[CATEGORICAL_FEATURES].fillna('Unknown')
    input_df[NUMERIC_FEATURES] = input_df[NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan)

    # --- 2. Current Practice Prediction & Reliability Calculation ---
    try:
        # A. Standard Ensemble Prediction (The weighted average)
        prediction = model.predict(input_df)[0]
        
        # B. Calculate Reliability via Distance-Based Method (KNN on OUTPUTS)
        # We check if the predicted Yield/N/P/K combination is close to combinations seen in training
        
        global reliability_knn, reliability_dist_scale, scaler_y
        
        if reliability_knn is not None and scaler_y is not None:
            # 1. Reshape prediction to 2D array (1 sample, 4 features)
            pred_vector = prediction.reshape(1, -1)
            
            # 2. Scale the prediction using the scaler fitted on y_train
            pred_scaled = scaler_y.transform(pred_vector)
            
            # 3. Find distances to 5 nearest neighbors in TARGET space (y_train)
            distances, _ = reliability_knn.kneighbors(pred_scaled)
            avg_dist = np.mean(distances[0])
            
            # Calculate Score: 100 * exp(-distance / scale)
            if reliability_dist_scale > 0:
                 reliability_score = 100.0 * np.exp(-avg_dist / reliability_dist_scale)
            else:
                 reliability_score = 0.0 
        else:
            reliability_score = 0.0 # KNN not available

    except Exception as e:
        print(f"Prediction Error: {e}")
        return f"Error: {e}", "", "", "", "", "", "", "", state

    yield_pred = prediction[0]
    n_pred = prediction[1]
    p_pred = prediction[2]
    k_pred = prediction[3]

    # --- 3. Optimization Logic with Caching ---
    
    # Identify current constants
    current_constants = {k: inputs.get(k) for k in CONSTANTS_FOR_OPTIMIZATION}
    
    cached_constants = state.get('constants', {})
    cached_result = state.get('best_pred', None)
    
    # Check if constants changed
    constants_changed = True
    if cached_result is not None:
        # Simple dict comparison
        if str(current_constants) == str(cached_constants):
            constants_changed = False
            
    if constants_changed:
        progress(0.2, desc="Constants changed. Running Grid Search Optimization...")
        best_pred, best_practices, best_score, user_score = generate_grid_and_optimize(inputs, progress)
        state['constants'] = current_constants
        state['best_pred'] = best_pred
        state['best_practices'] = best_practices
        state['best_score'] = best_score
        state['user_score'] = user_score
    else:
        progress(0.2, desc="Constants unchanged. Using Cached Optimization...")
        best_pred = state['best_pred']
        best_practices = state['best_practices']
        best_score = state.get('best_score', 0)
        user_score = state.get('user_score', 0)

    opt_yield = best_pred[0]
    opt_n = best_pred[1]
    opt_p = best_pred[2]
    opt_k = best_pred[3]
    
    # Generate Advice Dynamically
    advice_lines = []
    crlparha = pd.to_numeric(inputs['CRLPARHA'], errors='coerce')
    if pd.isna(crlparha) or crlparha <= 0: crlparha = 1.0 
    
    for col in PRACTICE_FEATURES_TO_OPTIMIZE:
        user_val = inputs.get(col)
        best_val = best_practices.get(col)
        
        # Formatting for UI and Comparison
        user_val_ui = str(user_val)
        best_val_ui = str(best_val)

        if col == 'TDATE_yday':
            user_val_ui = doy_to_date_str(user_val)
            best_val_ui = doy_to_date_str(best_val)
        elif col == 'SRATEHA':
            user_val_kg = pd.to_numeric(user_val, errors='coerce') * crlparha
            best_val_kg = pd.to_numeric(best_val, errors='coerce') * crlparha
            user_val_ui = f"{user_val_kg:.1f} kg"
            best_val_ui = f"{best_val_kg:.1f} kg"
        elif col in NUMERIC_PRACTICES:
            user_val_ui = f"{pd.to_numeric(user_val, errors='coerce'):.1f}"
            best_val_ui = f"{pd.to_numeric(best_val, errors='coerce'):.1f}"
        elif col in REVERSE_INTERPRETABLE_MAPS:
            user_val_ui = REVERSE_INTERPRETABLE_MAPS[col].get(str(user_val), str(user_val))
            best_val_ui = REVERSE_INTERPRETABLE_MAPS[col].get(str(best_val), str(best_val))

        if user_val_ui != best_val_ui:
             feat_name = FEATURE_NAME_MAP.get(col, col)
             advice_lines.append(f"- **{feat_name}:** Change from '{user_val_ui}' to '{best_val_ui}'")

    # --- 4. Economics & Formatting ---
    
    if user_score >= (best_score - 1e-4):
        opt_yield = yield_pred
        opt_n = n_pred
        opt_p = p_pred
        opt_k = k_pred
        advice_summary = "Your current practices are predicted to be better than the grid search alternatives! **No changes recommended.**"
    elif not advice_lines:
        advice_summary = "Your current practices are optimal based on our simulation. **No changes recommended.**"
        opt_yield = yield_pred 
        opt_n = n_pred
        opt_p = p_pred
        opt_k = k_pred
    else:
          advice_summary = "**To achieve the highest predicted yield (balanced with cost), consider these changes:**\n\n" + "\n".join(advice_lines)

    # Translate Advice
    advice_summary = translate_text(advice_summary, target_lang_code)

    crlparha = pd.to_numeric(inputs['CRLPARHA'], errors='coerce')
    if pd.isna(crlparha) or crlparha <= 0: crlparha = 1.0 

    prices = {k: pd.to_numeric(inputs.get(k), errors='coerce') for k in ECONOMIC_COLS}
    for k, v in prices.items():
        if pd.isna(v) or v < 0: prices[k] = 0.0

    def calculate_economics(yield_ha, n_ha, p_ha, k_ha, area_ha, prices_dict, srate_ha):
        total_urea, total_dap, total_mop = get_fertilizer_quantities(n_ha, p_ha, k_ha, area_ha)
        
        # Fertilizer Costs
        cost_urea = total_urea * (prices_dict['Urea_price'] / 50.0)
        cost_dap = total_dap * (prices_dict['DAP_price'] / 50.0)
        cost_mop = total_mop * (prices_dict['MOP_price'] / 50.0)
        
        # Seed Cost (Variable based on rate)
        cost_seed = srate_ha * area_ha * prices_dict['Seed_price']

        # Fixed Per-Hectare Costs Aggregation
        fixed_cost_sum = (
            prices_dict['Labour_cost'] +
            prices_dict['Insecticides_cost'] +
            prices_dict['Irrigation_cost'] +
            prices_dict['Insurance_cost'] +
            prices_dict['Misc_cost'] +
            prices_dict['Interest_working_cost'] +
            prices_dict['Rent_owned_cost'] +
            prices_dict['Rent_leased_cost'] +
            prices_dict['Land_revenue_cost'] +
            prices_dict['Depreciation_cost'] +
            prices_dict['Interest_fixed_cost']
        )
        total_fixed_cost = fixed_cost_sum * area_ha
        
        cost = cost_urea + cost_dap + cost_mop + cost_seed + total_fixed_cost
        
        total_quintals = (yield_ha * area_ha) / 100
        revenue = total_quintals * prices_dict['FGPRICE_quintal']
        
        profit = revenue - cost
        bcr = revenue / cost if cost > 0 else 0
        
        return revenue, cost, profit, bcr, total_urea, total_dap, total_mop

    # For user, srate_ha is in inputs['SRATEHA']
    user_srate_ha = pd.to_numeric(inputs.get('SRATEHA', 30.0), errors='coerce')
    
    # For optimal, srate_ha is in best_practices['SRATEHA']
    opt_srate_ha = pd.to_numeric(best_practices.get('SRATEHA', 30.0), errors='coerce')

    curr_rev, curr_cost, curr_profit, curr_bcr, \
    curr_urea, curr_dap, curr_mop = calculate_economics(yield_pred, n_pred, p_pred, k_pred, crlparha, prices, user_srate_ha)

    opt_rev, opt_cost, opt_profit, opt_bcr, \
    opt_urea, opt_dap, opt_mop = calculate_economics(opt_yield, opt_n, opt_p, opt_k, crlparha, prices, opt_srate_ha)

    economic_table_md = f"""
    | Metric | Your Practice (Predicted) | Optimized Practice (Simulated) |
    | :--- | :--- | :--- |
    | **Yield (kg/ha)** | **{yield_pred:.1f}** | **{opt_yield:.1f}** |
    | Total Revenue (INR) | {curr_rev:,.0f} | {opt_rev:,.0f} |
    | Total Cost (INR) | {curr_cost:,.0f} | {opt_cost:,.0f} |
    | **Net Profit (INR)** | **{curr_profit:,.0f}** | **{opt_profit:,.0f}** |
    | **BCR (Benefit-Cost)** | **{curr_bcr:.2f}** | **{opt_bcr:.2f}** |
    """
    # Translate Table
    economic_table_md = translate_text(economic_table_md, target_lang_code)

    fertilizer_plan = format_fertilizer_plan(opt_urea, opt_dap, opt_mop, curr_urea, curr_dap, curr_mop, crlparha)
    # Translate Plan
    fertilizer_plan = translate_text(fertilizer_plan, target_lang_code)

    yield_output = f"{yield_pred:.2f} kg/ha"
    n_output = f"{n_pred:.2f} kg/ha"
    p_output = f"{p_pred:.2f} kg/ha"
    k_output = f"{k_pred:.2f} kg/ha"
    reliability_output = f"{reliability_score:.1f} / 100"

    return yield_output, n_output, p_output, k_output, reliability_output, advice_summary, fertilizer_plan, economic_table_md, state


# --- 10. LAUNCH THE APP ---

if __name__ == "__main__":
    
    importlib.reload(croplizer_chat)
    print("Reloaded croplizer_chat module.")
    data_df = load_and_preprocess_data("NUE_survey_dataset_v2.csv")
    if data_df is None:
        print("Failed to load data. Exiting.")
        exit()

    if os.path.exists(MODEL_FILENAME):
        print(f"Loading existing model from {MODEL_FILENAME}...")
        try:
            model = joblib.load(MODEL_FILENAME)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}. Retraining...")
            model = train_prediction_model(data_df)
    else:
        print(f"No saved model found. Training new model...")
        model = train_prediction_model(data_df)

    # --- 11. FIT RELIABILITY MODEL (KNN) ---
    print("Fitting reliability (KNN) model on TRAINING TARGETS...")
    try:
        # NEW: Fit on Targets (y) instead of Features (X)
        y_train = data_df[TARGETS]
        
        # We must scale targets because Yield (~5000) dominates N/P/K (~100)
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train)
        
        # Use NearestNeighbors to find distance to training targets
        reliability_knn = NearestNeighbors(n_neighbors=5, n_jobs=-1)
        reliability_knn.fit(y_train_scaled)
        
        # Calculate Scale Factor for reliability score normalization
        sample_size = min(2000, y_train_scaled.shape[0])
        sample_indices = np.random.choice(y_train_scaled.shape[0], sample_size, replace=False)
        
        # Find distances within the training target space
        sample_dists, _ = reliability_knn.kneighbors(y_train_scaled[sample_indices])
        
        avg_train_dist = np.mean(sample_dists)
        
        # Calibrate scale
        if avg_train_dist > 0:
            reliability_dist_scale = -avg_train_dist / np.log(0.90)
        else:
            reliability_dist_scale = 1.0 
            
        print(f"Reliability model fitted on Outputs. Scale factor: {reliability_dist_scale:.4f}")
        
    except Exception as e:
        print(f"Error fitting reliability model: {e}")
        reliability_knn = None
        scaler_y = None
    # 3. Define Bridging Functions for Chatbot
    def chat_predict_bridge(inputs, state, lang_code):
        """Wraps the main prediction function for the chatbot module."""
        # Ensure imports/globals are available
        return predict_yield_and_fertilizer(inputs, state, lang_code, progress=gr.Progress())

    def chat_translate_bridge(text, lang_code):
        return translate_text(text, LANGUAGES.get(lang_code, 'en'))

    def chat_data_bridge(lat, lon):
        return fetch_external_data(lat, lon, progress=gr.Progress())
    print("Launching Gradio app...")

    def create_input_component(name):
        label = FEATURE_NAME_MAP.get(name, name)

        if name not in FEATURES and name not in ECONOMIC_COLS:
            return None 

        if name in ECONOMIC_DEFAULTS:
             default_value = ECONOMIC_DEFAULTS[name]
        elif name in data_df.columns:
            if name in NUMERIC_FEATURES: default_value = data_df[name].median()
            else: default_value = data_df[name].mode()[0]
        else: default_value = 0.0 

        if name in CATEGORICAL_FEATURES:
            ui_choices = list(INTERPRETABLE_MAPS[name].keys())
            default_model_val = str(default_value)
            default_ui_val = REVERSE_INTERPRETABLE_MAPS[name].get(default_model_val, 'Unknown')
            if default_ui_val not in ui_choices: default_ui_val = ui_choices[0] if ui_choices else 'Unknown'

            if name in ['EDU', 'VARTYPE', 'SOCCAT']:
                return gr.Dropdown(label=label, choices=ui_choices, value=default_ui_val, elem_id=name)
            else:
                return gr.Radio(label=label, choices=ui_choices, value=default_ui_val, elem_id=name)
        else:
            if pd.isna(default_value): default_value = 0.0
                
            if name == 'CRLPARHA':
                default_crarha_max = 10.0
                if 'CRARHA' in data_df.columns and not data_df['CRARHA'].isnull().all():
                    default_crarha_max = max(10.0, float(data_df['CRARHA'].max()))
                return gr.Slider(label=label, minimum=0.0, maximum=default_crarha_max, value=round(default_value, 1), step=0.1, elem_id=name)
            elif name == 'SRATEHA':
                return gr.Slider(label=label, minimum=10, maximum=150, value=round(default_value, 0), step=1, elem_id=name)
            elif name == 'TDATE_yday':
                # Special handling for Date Input
                return gr.DateTime(label=label, type="datetime", elem_id=name, include_time=False)
            elif name in ['LAT', 'LONG']:
                return gr.Number(label=label, value=round(default_value, 4), elem_id=name)
            else:
                return gr.Number(label=label, value=round(default_value, 1), elem_id=name)

    js_get_location = """
    () => {
        return new Promise((resolve) => {
            if (!navigator.geolocation) {
                alert("Geolocation is not supported by your browser.");
                resolve([null, null]);
                return;
            }
            const inIframe = (window.self !== window.top);
            navigator.permissions.query({name:'geolocation'}).then((permissionStatus) => {
                const handleSuccess = (position) => {
                    resolve([position.coords.latitude, position.coords.longitude]);
                };
                const handleError = (error) => { 
                    if (permissionStatus.state === 'denied') {
                        alert("Location permission is blocked.");
                    } else {
                        alert("Unable to retrieve location.");
                    }
                    resolve([null, null]);
                };
                navigator.geolocation.getCurrentPosition(handleSuccess, handleError, { timeout: 10000 });
            });
        });
    }
    """

    # Custom CSS to force tabs to wrap and not collapse, and force radio into one column
    css = """
    /* Force Language Radio to be vertical/one-column */
    #lang_radio .wrap {
        display: flex !important;
        flex-direction: column !important;
        gap: 0px !important; /* Eliminate gap between items */
    }
    #lang_radio label {
        margin-bottom: 10px !important;
        padding: 2px 8px !important; /* Compact padding */
        width: 100% !important; /* Ensure it fills the narrower column */
    }
    
    /* NEW: Constrain the Language Column Width */
    #lang_col {
        min_width: 120px !important;
        max_width: 180px !important;
        width: auto !important;
    }

    /* Fix Tab truncation */
    .tab-nav {
        flex-wrap: wrap !important;
    }
    .tab-nav button {
        white-space: normal !important;
        height: auto !important;
        padding: 5px 10px !important;
    }
    """
    # JS for Chat Auto-Send
    js_chat_loc = """
    async (x) => {
        if (x === 'Auto') {
            return new Promise((resolve) => {
                if (!navigator.geolocation) { resolve("Auto"); return; }
                navigator.geolocation.getCurrentPosition(
                    (pos) => { resolve(`${pos.coords.latitude.toFixed(4)}, ${pos.coords.longitude.toFixed(4)}`); },
                    (err) => { resolve("Auto"); }
                );
            });
        }
        return x;
    }
    """
    # Define favicon as both static HTML (for Chrome App install) and JS (for Browser Tab)
    # We use the same SVG Data URI for both.
    favicon_html = """
    <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🌾</text></svg>">
    <link rel="apple-touch-icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🌾</text></svg>">
    <script>
    function setFavicon() {
        const link = document.querySelector("link[rel*='icon']") || document.createElement('link');
        link.type = 'image/svg+xml';
        link.rel = 'shortcut icon';
        link.href = 'data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🌾</text></svg>';
        document.getElementsByTagName('head')[0].appendChild(link);
    }
    // Auto-refresh chat every 12 hours (43,200,000 milliseconds)
    setInterval(() => {
        const buttons = Array.from(document.querySelectorAll('button'));
        const restartBtn = buttons.find(btn => btn.innerText.includes('Restart Chat'));
        if (restartBtn) {
            console.log("Auto-refreshing chat...");
            restartBtn.click();
        }
    }, 43200000);
    // Run on load and after delay to fight Gradio's defaults
    window.addEventListener('DOMContentLoaded', setFavicon);
    window.addEventListener('load', setFavicon);
    setTimeout(setFavicon, 1500);
    </script>
    """

    with gr.Blocks(theme=gr.themes.Soft(), css=css, title="CropLizer", head=favicon_html) as iface:
        state = gr.State({}) # Store optimization cache here

        with gr.Row():
            # --- Sidebar: Language Selector ---
            with gr.Column(scale=0, min_width=120, elem_id="lang_col"):
                # gr.Markdown("### 🗣️ Language")
                lang_selector = gr.Radio(
                    choices=list(LANGUAGES.keys()), 
                    value='English', 
                    label="Select Language",
                    show_label=False,
                    container=False,
                    interactive=True,
                    elem_id="lang_radio"  # <--- Added ID
                )
            
            # --- Main Content ---
            with gr.Column(scale=5):
                main_title = gr.Markdown(
                    """
                    # 🌾 CropLizer: An Agro-Socio-Edapho-Climatological Tool for Farmers
                    """
                )
                with gr.Tabs():
                    # --- TAB 1: CHATBOT (New Integration) ---
                    with gr.TabItem("🤖 Quickbot Advisor"):
                        chatbot = gr.Chatbot(height=500, type="tuples", label="Farm Assistant")
                        chat_state = gr.State({'step': 0, 'data': {}})
                        
                        with gr.Row():
                            chat_opts = gr.Radio(choices=['Auto'], label="Quick Options", visible=True)
                            chat_msg = gr.Textbox(show_label=False, placeholder="Type answer here...", autofocus=True)
                            chat_clear = gr.Button("⟲")

                        # Chat Event Logic
                        def respond(msg, hist, st, lang, opts):
                            # Pass bridges to the module
                            return croplizer_chat.process_turn(
                                msg, hist, st, lang, 
                                chat_translate_bridge, chat_data_bridge, chat_predict_bridge
                            )
                        # NEW: Dedicated function to force-start the chat
                        def init_chat(lang):
                            # Force empty history, empty state, and empty message to trigger greeting
                            return croplizer_chat.process_turn(
                                "", [], {'step': 0, 'data': {}}, lang,
                                chat_translate_bridge, chat_data_bridge, chat_predict_bridge
                            )

                        # Submit Text
                        chat_msg.submit(
                            fn=lambda x: x, inputs=[chat_msg], outputs=[chat_msg], js=js_chat_loc
                        ).then(
                            fn=respond,
                            inputs=[chat_msg, chatbot, chat_state, lang_selector, chat_opts],
                            outputs=[chatbot, chat_state, chat_opts]
                        ).then(lambda: "", None, chat_msg)

                        # Submit Option (Auto/Buttons)
                        chat_opts.select(
                            fn=None, inputs=[chat_opts], outputs=[chat_msg], js=js_chat_loc
                        ).then(
                            fn=respond,
                            inputs=[chat_msg, chatbot, chat_state, lang_selector, chat_opts],
                            outputs=[chatbot, chat_state, chat_opts]
                        ).then(lambda: "", None, chat_msg)

                        # Reset Chat: Call init_chat directly
                        chat_clear.click(
                            fn=init_chat,
                            inputs=[lang_selector],
                            outputs=[chatbot, chat_state, chat_opts]
                        )
                        
                        # Trigger initial greeting on app load
                        iface.load(
                            fn=init_chat,
                            inputs=[lang_selector],
                            outputs=[chatbot, chat_state, chat_opts]
                        )
                    with gr.TabItem("📝 Manual Advanced Input"):        
                        inputs_dict = {} 
                        
                        with gr.Row():
                            with gr.Column(scale=2):
                                header_1 = gr.Markdown("### 1. 🌍 Farm Location")
                                with gr.Row():
                                    inputs_dict['LAT'] = create_input_component('LAT')
                                    inputs_dict['LONG'] = create_input_component('LONG')
                                
                                get_loc_btn = gr.Button("Get My Location (GPS)")
                                lat_js = gr.Number(visible=False)
                                lon_js = gr.Number(visible=False)

                                header_2 = gr.Markdown("### 2. 📝 Farm Details")
                                with gr.Tabs() as tabs:
                                    with gr.TabItem("👨‍🌾Farm/Farmer") as tab_farm_farmer:
                                        with gr.Row():
                                            with gr.Column():
                                                header_plot = gr.Markdown("#### Plot Details")
                                                inputs_dict['SEASON'] = create_input_component('SEASON')
                                                inputs_dict['CRLPARHA'] = create_input_component('CRLPARHA')
                                                inputs_dict['CRARHA'] = create_input_component('CRARHA')
                                            with gr.Column():
                                                header_farmer = gr.Markdown("#### Farmer Profile")
                                                inputs_dict['GEN'] = create_input_component('GEN')
                                                inputs_dict['EDU'] = create_input_component('EDU')
                                                inputs_dict['SOCCAT'] = create_input_component('SOCCAT')
                                                inputs_dict['HHMEM'] = create_input_component('HHMEM')
                                                inputs_dict['DISMR'] = create_input_component('DISMR')

                                    with gr.TabItem("₹ Economics") as tab_economics:
                                        header_prices = gr.Markdown("#### Market Prices")
                                        with gr.Row():
                                            inputs_dict['FGPRICE_quintal'] = create_input_component('FGPRICE_quintal')
                                            inputs_dict['Seed_price'] = create_input_component('Seed_price')
                                        with gr.Row():
                                            inputs_dict['Urea_price'] = create_input_component('Urea_price')
                                            inputs_dict['DAP_price'] = create_input_component('DAP_price')
                                            inputs_dict['MOP_price'] = create_input_component('MOP_price')
                                        
                                        header_op_costs = gr.Markdown("#### Operational Costs (per ha)")
                                        with gr.Row():
                                            inputs_dict['Labour_cost'] = create_input_component('Labour_cost')
                                            inputs_dict['Irrigation_cost'] = create_input_component('Irrigation_cost')
                                        with gr.Row():
                                            inputs_dict['Insecticides_cost'] = create_input_component('Insecticides_cost')
                                            inputs_dict['Misc_cost'] = create_input_component('Misc_cost')
                                        
                                        header_fixed_costs = gr.Markdown("#### Fixed/Overhead Costs (per ha)")
                                        with gr.Row():
                                            inputs_dict['Rent_owned_cost'] = create_input_component('Rent_owned_cost')
                                            inputs_dict['Rent_leased_cost'] = create_input_component('Rent_leased_cost')
                                        with gr.Row():
                                            inputs_dict['Depreciation_cost'] = create_input_component('Depreciation_cost')
                                            inputs_dict['Interest_fixed_cost'] = create_input_component('Interest_fixed_cost')
                                        with gr.Row():
                                            inputs_dict['Interest_working_cost'] = create_input_component('Interest_working_cost')
                                            inputs_dict['Insurance_cost'] = create_input_component('Insurance_cost')
                                            inputs_dict['Land_revenue_cost'] = create_input_component('Land_revenue_cost')


                                    with gr.TabItem("Soil/Climate") as tab_soil_climate:
                                        fetch_data_btn = gr.Button("Fetch Climate & Soil")
                                        with gr.Row():
                                            with gr.Column():
                                                header_climate = gr.Markdown("#### Climate Data")
                                                inputs_dict['total_precip'] = create_input_component('total_precip')
                                                inputs_dict['num_dry_days'] = create_input_component('num_dry_days')
                                                inputs_dict['avg_dspell_length'] = create_input_component('avg_dspell_length')
                                                inputs_dict['monsoon_onset'] = create_input_component('monsoon_onset')
                                                inputs_dict['monsoon_length'] = create_input_component('monsoon_length')
                                            with gr.Column():
                                                header_soil_perc = gr.Markdown("#### Soil (Perceived)")
                                                inputs_dict['SOPER'] = create_input_component('SOPER')
                                                inputs_dict['DCLASS'] = create_input_component('DCLASS')
                                        header_soil_grid = gr.Markdown("#### Soil (Grids Data)")
                                        with gr.Row():
                                            inputs_dict['SoilGrids_nitrogen'] = create_input_component('SoilGrids_nitrogen')
                                            inputs_dict['SoilGrids_phh2o'] = create_input_component('SoilGrids_phh2o')
                                            inputs_dict['SoilGrids_soc'] = create_input_component('SoilGrids_soc')
                                        with gr.Row():
                                            inputs_dict['SoilGrids_clay'] = create_input_component('SoilGrids_clay')
                                            inputs_dict['SoilGrids_sand'] = create_input_component('SoilGrids_sand')
                                            inputs_dict['SoilGrids_silt'] = create_input_component('SoilGrids_silt')
                                        with gr.Row():
                                            inputs_dict['SoilGrids_bdod'] = create_input_component('SoilGrids_bdod')
                                            inputs_dict['SoilGrids_ocd'] = create_input_component('SoilGrids_ocd')

                                    with gr.TabItem("𓃽𓃽𓀚 Practices") as tab_practices:
                                        with gr.Row():
                                            with gr.Column():
                                                header_cult = gr.Markdown("#### Cultivation")
                                                inputs_dict['VARTYPE'] = create_input_component('VARTYPE')
                                                inputs_dict['EST_line'] = create_input_component('EST_line')
                                                inputs_dict['EST_binary'] = create_input_component('EST_binary')
                                                inputs_dict['SRATE_kg'] = gr.Number(label="Seed Rate (Total kg)", value=30.0, elem_id='SRATE_kg')
                                                inputs_dict['TDATE_yday'] = create_input_component('TDATE_yday')
                                                inputs_dict['Field_duration'] = create_input_component('Field_duration')
                                                inputs_dict['PCRR'] = create_input_component('PCRR')
                                            with gr.Column():
                                                header_irri = gr.Markdown("#### Irrigation")
                                                inputs_dict['IRRIAVA'] = create_input_component('IRRIAVA')
                                                inputs_dict['IRRINU'] = create_input_component('IRRINU')
                                                header_stress = gr.Markdown("#### Stresses")
                                                inputs_dict['WSEV'] = create_input_component('WSEV')
                                                inputs_dict['INSEV'] = create_input_component('INSEV')
                                                inputs_dict['DISEV'] = create_input_component('DISEV')
                                                inputs_dict['DRSEV'] = create_input_component('DRSEV')
                                                inputs_dict['FLSEV'] = create_input_component('FLSEV')

                                    with gr.TabItem("💩 Nutrients") as tab_nutrients:
                                        header_other_nut = gr.Markdown("#### Other Nutrients")
                                        with gr.Row():
                                            inputs_dict['Organic_ha'] = create_input_component('Organic_ha')
                                            inputs_dict['GYP_ha'] = create_input_component('GYP_ha')
                                        with gr.Row():
                                            inputs_dict['BOR_ha'] = create_input_component('BOR_ha')
                                            inputs_dict['Zn_ha'] = create_input_component('Zn_ha')


                            with gr.Column(scale=3):
                                header_3 = gr.Markdown("### 3. 📈 Prediction Results")
                                yield_out = gr.Textbox(label="Predicted Yield")
                                reliability_out = gr.Textbox(label="Prediction Reliability Score")
                                nutrients_header = gr.Markdown("#### Predicted Nutrient Rates (Current Practice)")
                                n_out = gr.Textbox(label="Predicted Nitrogen (N) Rate")
                                p_out = gr.Textbox(label="Predicted Phosphorous (P₂O₅) Rate")
                                k_out = gr.Textbox(label="Predicted Potassium (K₂O) Rate")

                                gr.Markdown("---")
                                plan_header = gr.Markdown("### 💡 Suggested Fertilizer Plan (Optimized for Best Yield & Profit)")
                                plan_out = gr.Markdown(label="Fertilizer Mix")

                                gr.Markdown("---")
                                econ_header = gr.Markdown("### 💰 Economic Analysis")
                                economic_out = gr.Markdown(label="Profit & BCR Comparison")

                                gr.Markdown("---")
                                advice_header = gr.Markdown("### 🌾 Yield Optimization Advice")
                                advice_out = gr.Markdown(label="Optimization Advice")

                        submit_btn = gr.Button("Predict Yield, NPK & Profitability", variant="primary")
                        
                    with gr.TabItem("⚙️ Settings") as tab_settings:
                        with gr.Row():
                            with gr.Column(scale=1):
                                header_weights = gr.Markdown("### ⚙️ Optimization Priorities")
                                weights_desc = gr.Markdown("Adjust how much importance the AI gives to each factor when suggesting changes (0 = Ignore, 1 = High Priority).")
                                inputs_dict['W_YIELD'] = gr.Slider(0.0, 1.0, value=1.0, step=0.1, label="Yield Priority", interactive=True)
                                inputs_dict['W_PROFIT'] = gr.Slider(0.0, 1.0, value=1.0, step=0.1, label="Profit/BCR Priority", interactive=True)
                                inputs_dict['W_ENV'] = gr.Slider(0.0, 1.0, value=1.0, step=0.1, label="Environmental Priority (Low NPK)", interactive=True)
                # --- Dynamic Label Updates ---
                def update_ui_language(lang_selection):
                    target_lang_code = LANGUAGES.get(lang_selection, 'en')
                    
                    # --- BATCH TRANSLATION LOGIC ---
                    # 1. Collect all texts to translate in order
                    
                    # A. Markdown Components
                    markdown_texts = [
                        "# 🌾 CropLizer: An Agro-Socio-Edapho-Climatological Tool for Farmers",
                        "### 1. 🌍 Farm Location",
                        "### 2. 📝 Farm Details",
                        "#### Plot Details",
                        "#### Farmer Profile",
                        "#### Market Prices",
                        "#### Operational Costs (per ha)",
                        "#### Fixed/Overhead Costs (per ha)",
                        "#### Climate Data",
                        "#### Soil (Perceived)",
                        "#### Soil (Grids Data)",
                        "#### Cultivation",
                        "#### Irrigation",
                        "#### Stresses",
                        "#### Other Nutrients",
                        "### 3. 📈 Prediction Results",
                        "#### Predicted Nutrient Rates (Current Practice)",
                        "### 💡 Suggested Fertilizer Plan (Optimized for Best Yield & Profit)",
                        "### 💰 Economic Analysis",
                        "### 🌾 Yield Optimization Advice",
                        "### ⚙️ Optimization Priorities",
                        "Adjust how much importance the AI gives to each factor when suggesting changes (0 = Ignore, 1 = High Priority)."
                    ]
                    
                    # B. Input Labels & Choices
                    # We need to track which inputs have choices to map back correctly
                    input_text_map = [] # List of {'type': 'label'/'choice', 'key': feature_key, 'original': text}
                    
                    for key, component in inputs_dict.items():
                        if component is None: continue
                        
                        # Label
                        if key == 'SRATE_kg':
                            english_label = "Seed Rate (Total kg)"
                        else:
                            english_label = FEATURE_NAME_MAP.get(key, key)
                        
                        input_text_map.append({'type': 'label', 'key': key, 'original': english_label})
                        
                        # Choices (if categorical)
                        if key in CATEGORICAL_FEATURES:
                            english_choices = list(INTERPRETABLE_MAPS[key].keys())
                            for choice in english_choices:
                                input_text_map.append({'type': 'choice', 'key': key, 'original': choice})

                    # C. Output Labels
                    output_label_texts = [
                        "Predicted Yield",
                        "Prediction Reliability Score",
                        "Predicted Nitrogen (N) Rate",
                        "Predicted Phosphorous (P₂O₅) Rate",
                        "Predicted Potassium (K₂O) Rate",
                        "Fertilizer Mix",
                        "Profit & BCR Comparison",
                        "Optimization Advice"
                    ]

                    # D. Button Texts
                    button_texts = [
                        "Predict Yield, NPK & Profitability",
                        "Get My Location (GPS)",
                        "Fetch Climate & Soil"
                    ]

                    # E. Tab Labels
                    # E. Tab Labels
                    tab_texts = [
                        "👨‍🌾Farm/Farmer",
                        "₹ Economics",
                        "Soil/Climate",
                        "𓃽𓃽𓀚 Practices",
                        "💩 Nutrients",
                        "⚙️ Settings"
                    ]

                    # --- EXECUTE BATCH TRANSLATION ---
                    # Combine all lists into one massive list
                    all_texts = markdown_texts + [item['original'] for item in input_text_map] + output_label_texts + button_texts + tab_texts
                    
                    # Remove duplicates to save API calls (optional but good practice) - keep order for mapping
                    # Actually, keeping duplicates is safer for simple mapping back by index. 
                    # Given the scale (~100 items), batch translation is fine with duplicates.
                    
                    try:
                        if target_lang_code == 'en':
                            translated_texts = all_texts
                        else:
                            # Use batch translation
                            translator = GoogleTranslator(source='auto', target=target_lang_code)
                            translated_texts = translator.translate_batch(all_texts)
                    except Exception as e:
                        print(f"Batch Translation Error: {e}")
                        translated_texts = all_texts # Fallback to English

                    # --- MAP BACK TO UPDATES ---
                    updates = []
                    idx = 0
                    
                    # A. Markdown
                    markdown_components = [
                        main_title, header_1, header_2, header_plot, header_farmer,
                        header_prices, header_op_costs, header_fixed_costs, header_climate,
                        header_soil_perc, header_soil_grid, header_cult, header_irri,
                        header_stress, header_other_nut, header_3, nutrients_header,
                        plan_header, econ_header, advice_header,
                        header_weights,weights_desc 
                    ]
                    for comp in markdown_components:
                        updates.append(gr.update(value=translated_texts[idx]))
                        idx += 1
                    
                    # B. Inputs
                    # We need to reconstruct the choices lists
                    # Since input_text_map is linear, we iterate it and group by key
                    
                    # Helper to retrieve translated text from the linear list
                    def get_trans():
                        nonlocal idx
                        t = translated_texts[idx]
                        idx += 1
                        return t

                    current_input_key = None
                    current_choices = []
                    current_label = ""
                    
                    # We process input_text_map linearly. 
                    # When key changes or list ends, we push the update for the previous key.
                    
                    # However, logic is tricky because we iterate inputs_dict in the same order as we built input_text_map
                    # Let's iterate inputs_dict again and pop from input_text_map/translated_texts
                    
                    # Re-create the iterator logic used to build the list
                    # We know the order: Label, then Choices (if any)
                    
                    for key, component in inputs_dict.items():
                        if component is None: continue
                        
                        # 1. Get Label Translation
                        trans_label = get_trans()
                        
                        # 2. Get Choices Translation (if any)
                        if key in CATEGORICAL_FEATURES:
                            english_choices = list(INTERPRETABLE_MAPS[key].keys())
                            new_choices = []
                            for original_choice in english_choices:
                                trans_choice = get_trans()
                                # Use tuple (Display Text, Internal Value)
                                # Internal value is the original_choice (which maps to numeric/code via INTERPRETABLE_MAPS logic if needed, 
                                # but here 'choices' in gradio uses the second element as value)
                                # Wait, INTERPRETABLE_MAPS keys ARE the internal values for our logic in predict_wrapper?
                                # No, keys are English Labels. Values are internal codes.
                                # Predict wrapper maps UI Value -> Internal Code using INTERPRETABLE_MAPS.get(ui_value).
                                # So the Value of the component MUST be the English Label.
                                new_choices.append((trans_choice, original_choice))
                            
                            updates.append(gr.update(label=trans_label, choices=new_choices))
                        else:
                            updates.append(gr.update(label=trans_label))

                    # C. Outputs
                    output_components = [
                        yield_out, reliability_out, n_out, p_out, k_out,
                        plan_out, economic_out, advice_out
                    ]
                    for comp in output_components:
                        updates.append(gr.update(label=translated_texts[idx]))
                        idx += 1

                    # D. Buttons
                    button_components = [submit_btn, get_loc_btn, fetch_data_btn]
                    for comp in button_components:
                        updates.append(gr.update(value=translated_texts[idx]))
                        idx += 1

                    # E. Tabs
                    tab_components = [
                        tab_farm_farmer, tab_economics, tab_soil_climate,
                        tab_practices, tab_nutrients,
                        tab_settings
                    ]
                    for comp in tab_components:
                        updates.append(gr.update(label=translated_texts[idx]))
                        idx += 1
                    
                    return updates

                # Collect all components that need updating for the outputs list
                # Order must match the order of appends in update_ui_language
                all_ui_outputs = (
                    [comp for comp, _ in [
                        (main_title, ""), (header_1, ""), (header_2, ""), (header_plot, ""), (header_farmer, ""),
                        (header_prices, ""), (header_op_costs, ""), (header_fixed_costs, ""), (header_climate, ""),
                        (header_soil_perc, ""), (header_soil_grid, ""), (header_cult, ""), (header_irri, ""),
                        (header_stress, ""), (header_other_nut, ""), (header_3, ""), (nutrients_header, ""),
                        (plan_header, ""), (econ_header, ""), (advice_header, ""),(header_weights, ""), (weights_desc, "")
                    ]] +
                    list(inputs_dict.values()) +
                    [yield_out, reliability_out, n_out, p_out, k_out, plan_out, economic_out, advice_out] +
                    [submit_btn, get_loc_btn, fetch_data_btn] +
                    [tab_farm_farmer, tab_economics, tab_soil_climate, tab_practices, tab_nutrients]
                )

                lang_selector.change(
                    fn=update_ui_language,
                    inputs=lang_selector,
                    outputs=all_ui_outputs
                )

                if 'CRARHA' in inputs_dict and 'CRLPARHA' in inputs_dict:
                    def update_plot_area_max(farm_area, current_plot_area):
                        try: farm_area_float = float(farm_area)
                        except: farm_area_float = 0.0
                        try: current_plot_area_float = float(current_plot_area)
                        except: current_plot_area_float = 0.0
                        new_max = max(0.1, farm_area_float) 
                        new_value = min(current_plot_area_float, new_max)
                        return gr.update(maximum=new_max, value=new_value)
                    
                    inputs_dict['CRARHA'].blur(
                        fn=update_plot_area_max,
                        inputs=[inputs_dict['CRARHA'], inputs_dict['CRLPARHA']],
                        outputs=inputs_dict['CRLPARHA'],
                        show_progress=False
                    )

                all_input_components = []
                all_input_names = []
                
                for name in FEATURES:
                    if name in inputs_dict:
                        all_input_components.append(inputs_dict[name])
                        all_input_names.append(name)
                
                if 'SRATE_kg' in inputs_dict:
                    all_input_components.append(inputs_dict['SRATE_kg'])
                    all_input_names.append('SRATE_kg')
                        
                for name in ECONOMIC_COLS:
                    if name in inputs_dict:
                        all_input_components.append(inputs_dict[name])
                        all_input_names.append(name)
                        
                for name in ['W_YIELD', 'W_PROFIT', 'W_ENV']:
                    if name in inputs_dict:
                        all_input_components.append(inputs_dict[name])
                        all_input_names.append(name)
                filtered_components = []
                filtered_names = [] 
                for comp, name in zip(all_input_components, all_input_names):
                    if comp is not None:
                        filtered_components.append(comp)
                        filtered_names.append(name)
                
                # Wrapper to handle inputs including State
                def predict_wrapper(*args, progress=gr.Progress(track_tqdm=True)):
                    # The last argument is the state, second to last is lang
                    current_state = args[-1]
                    lang_val = args[-2]
                    input_values = args[:-2]
                    
                    inputs_dict_wrapper = dict(zip(filtered_names, input_values))

                    # Convert TDATE_yday from DatePicker (datetime) to Day-of-Year (float)
                    if 'TDATE_yday' in inputs_dict_wrapper:
                        date_val = inputs_dict_wrapper['TDATE_yday']
                        doy_val = date_str_to_doy(date_val)
                        inputs_dict_wrapper['TDATE_yday'] = doy_val

                    crlparha_val = pd.to_numeric(inputs_dict_wrapper.get('CRLPARHA'), errors='coerce')
                    srate_kg_val = pd.to_numeric(inputs_dict_wrapper.get('SRATE_kg'), errors='coerce')
                    
                    if crlparha_val and crlparha_val > 0 and srate_kg_val is not None:
                        inputs_dict_wrapper['SRATEHA'] = srate_kg_val / crlparha_val
                    else:
                        inputs_dict_wrapper['SRATEHA'] = np.nan 

                    for key, ui_value in inputs_dict_wrapper.items():
                        if key in INTERPRETABLE_MAPS:
                            inputs_dict_wrapper[key] = INTERPRETABLE_MAPS[key].get(ui_value, 'Unknown')

                    return predict_yield_and_fertilizer(inputs_dict_wrapper, current_state, lang_val, progress)
                
                get_loc_btn.click(
                    fn=None, 
                    js=js_get_location, 
                    inputs=None, 
                    outputs=[lat_js, lon_js] 
                )
                
                def update_vis(val):
                    return val if val is not None else gr.update() 
                    
                lat_js.change(fn=update_vis, inputs=lat_js, outputs=inputs_dict['LAT'])
                lon_js.change(fn=update_vis, inputs=lon_js, outputs=inputs_dict['LONG'])

                api_output_components = [
                    inputs_dict['total_precip'], inputs_dict['num_dry_days'], inputs_dict['avg_dspell_length'],
                    inputs_dict['monsoon_onset'], inputs_dict['monsoon_length'],
                    inputs_dict['SoilGrids_bdod'], inputs_dict['SoilGrids_clay'], inputs_dict['SoilGrids_nitrogen'],
                    inputs_dict['SoilGrids_ocd'], inputs_dict['SoilGrids_phh2o'], inputs_dict['SoilGrids_sand'],
                    inputs_dict['SoilGrids_silt'], inputs_dict['SoilGrids_soc']
                ]
                
                fetch_data_btn.click(
                    fn=fetch_external_data,
                    inputs=[inputs_dict['LAT'], inputs_dict['LONG']],
                    outputs=api_output_components
                )

                submit_btn.click(
                    fn=predict_wrapper,
                    inputs=filtered_components + [lang_selector, state], 
                    outputs=[
                        yield_out, n_out, p_out, k_out, reliability_out,
                        advice_out, plan_out, economic_out, state
                    ]
                )

        iface.launch(share=True, debug=True)
