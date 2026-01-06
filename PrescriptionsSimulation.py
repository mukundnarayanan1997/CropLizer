import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm
import joblib
import os
import warnings

# Suppress feature name mismatch warnings from sklearn
warnings.filterwarnings("ignore", message="X does not have valid feature names, but StandardScaler was fitted with feature names")

# --- 1. CONFIGURATION & FEATURES ---
MODEL_FILENAME = 'farm_advisor_model.joblib'
DATA_FILENAME = 'NUE_survey_dataset_v2.csv'
OUTPUT_FILENAME = 'optimized_farming_results.csv'
NUM_SIMULATIONS = 20000

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

PRACTICE_FEATURES_TO_OPTIMIZE = [
    'VARTYPE', 'SRATEHA', 'Organic_ha', 'TDATE_yday',
    'EST_line', 'EST_binary', 'IRRINU', 'PCRR', 'Field_duration',
    'GYP_ha', 'BOR_ha', 'Zn_ha'
]

CONSTANTS_FOR_OPTIMIZATION = [f for f in FEATURES if f not in PRACTICE_FEATURES_TO_OPTIMIZE]
CATEGORICAL_PRACTICES = [col for col in PRACTICE_FEATURES_TO_OPTIMIZE if col in CATEGORICAL_FEATURES]
NUMERIC_PRACTICES = [col for col in PRACTICE_FEATURES_TO_OPTIMIZE if col in NUMERIC_FEATURES]

WEIGHT_PROFILES = {
    'Balanced': (1.0, 1.0, 1.0),
    'Yield_Max': (3.0, 0.5, 0.5),
    'Eco_Max': (0.5, 3.0, 0.5),
    'Profit_Max': (0.5, 0.5, 3.0)
}

# --- 2. DATA LOADING ---
def load_and_preprocess_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, low_memory=False, on_bad_lines='skip')

    all_cols = list(dict.fromkeys(FEATURES + TARGETS + ['FGPRICE_quintal', 'Urea_price', 'DAP_price', 'MOP_price']))
    existing_cols = [c for c in all_cols if c in df.columns]
    df = df[existing_cols]

    num_cols = [c for c in NUMERIC_FEATURES + TARGETS if c in df.columns]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=TARGETS)

    if 'yield_kg_ha' in df.columns:
        Q1 = df['yield_kg_ha'].quantile(0.25)
        Q3 = df['yield_kg_ha'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = min(Q3 + 1.5 * IQR, df['yield_kg_ha'].quantile(0.90))
        lower_bound = Q1 - 1.5 * IQR
        df = df[(df['yield_kg_ha'] >= lower_bound) & (df['yield_kg_ha'] <= upper_bound)]

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)

    print(f"Data cleaned. Valid observations: {len(df)}")
    return df

# --- 3. MODEL TRAINING & RELIABILITY ---
def get_model_and_tools(df):
    if os.path.exists(MODEL_FILENAME):
        print(f"Loading model from {MODEL_FILENAME}...")
        pipeline = joblib.load(MODEL_FILENAME)
    else:
        print("Training new model...")
        X = df[FEATURES]
        y = df[TARGETS]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

        model = RandomForestRegressor(
            n_jobs=-1,
            n_estimators=186,
            max_depth=28,
            min_samples_split=17,
            min_samples_leaf=10,
            max_features=0.2075561632492809
        )

        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, MODEL_FILENAME)

    print("Fitting reliability model on targets...")
    # Explicitly use .values to ensure StandardScaler fits on a numpy array (no feature names)
    y = df[TARGETS].values
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)

    knn = NearestNeighbors(n_neighbors=5, n_jobs=-1)
    knn.fit(y_scaled)

    sample_indices = np.random.choice(y_scaled.shape[0], min(2000, y_scaled.shape[0]), replace=False)
    dists, _ = knn.kneighbors(y_scaled[sample_indices])
    avg_dist = np.mean(dists)
    scale_factor = -avg_dist / np.log(0.90) if avg_dist > 0 else 1.0

    return pipeline, knn, scaler_y, scale_factor

# --- 4. BATCH OPTIMIZATION LOGIC ---
def process_partition(partition_df, model, knn, scaler_y, scale_factor, df_ref):
    """
    Processes a partition of the dataframe.
    Iterates through rows within the partition and applies the optimization logic.
    """
    results = []

    # Pre-calculate quantiles/options to avoid repetition in loop
    numeric_constraints = {}
    for col in NUMERIC_PRACTICES:
        series = df_ref[col].dropna()
        numeric_constraints[col] = (series.quantile(0.05), series.quantile(0.95))

    categorical_options = {}
    for col in CATEGORICAL_PRACTICES:
        categorical_options[col] = df_ref[col].unique()

    # Disable threading in sklearn to prevent oversubscription in Dask workers
    if hasattr(model, 'steps'):
        try:
            model.named_steps['model'].n_jobs = 1
        except:
            pass

    # Use tqdm for progress tracking within partition
    for _, row_data in tqdm(partition_df.iterrows(), total=len(partition_df), desc="Simulating Partition", leave=False):
        crlparha = row_data.get('CRLPARHA', 1.0)
        if pd.isna(crlparha) or crlparha <= 0: crlparha = 1.0

        price_fg = row_data.get('FGPRICE_quintal', 3368.0)
        price_urea = row_data.get('Urea_price', 10.0)
        price_dap = row_data.get('DAP_price', 25.0)
        price_mop = row_data.get('MOP_price', 18.0)
        if pd.isna(price_fg): price_fg = 3368.0

        sim_data = {}
        for col in CATEGORICAL_PRACTICES:
            sim_data[col] = np.random.choice(categorical_options[col], size=NUM_SIMULATIONS)

        season = row_data.get('SEASON', 'Kharif')

        for col in NUMERIC_PRACTICES:
            curr_val = row_data.get(col, 0.0)
            if pd.isna(curr_val): curr_val = 0.0

            p05, p95 = numeric_constraints[col]
            min_b, max_b = min(p05, curr_val), max(p95, curr_val)

            if col == 'TDATE_yday' and season == 'Kharif':
                min_b, max_b = max(152, min_b), min(304, max_b)
            elif col == 'Field_duration':
                min_b, max_b = max(80.0, min_b), min(160.0, max_b)
            elif col == 'IRRINU':
                min_b, max_b = max(0.0, min_b), min(30.0, max_b)
            elif col == 'SRATEHA':
                min_b, max_b = max(10.0, min_b), min(150.0, max_b)

            if min_b > max_b: min_b, max_b = max_b, min_b
            sim_data[col] = np.random.uniform(min_b, max_b, NUM_SIMULATIONS)

        base_data = {k: row_data.get(k) for k in CONSTANTS_FOR_OPTIMIZATION}
        grid_df = pd.DataFrame([base_data] * NUM_SIMULATIONS)
        practices_df = pd.DataFrame(sim_data)
        full_grid = pd.concat([grid_df, practices_df], axis=1)

        for col in NUMERIC_FEATURES:
            full_grid[col] = pd.to_numeric(full_grid[col], errors='coerce')
        for col in CATEGORICAL_FEATURES:
            full_grid[col] = full_grid[col].astype(str)

        full_grid = full_grid.reindex(columns=FEATURES, fill_value=0)

        preds = model.predict(full_grid)

        yields = preds[:, 0]
        n_rates = np.maximum(0, preds[:, 1])
        p_rates = np.maximum(0, preds[:, 2])
        k_rates = np.maximum(0, preds[:, 3])

        mop_kg = k_rates / 0.60
        dap_kg = p_rates / 0.46
        n_from_dap = dap_kg * 0.18
        n_rem = np.maximum(0, n_rates - n_from_dap)
        urea_kg = n_rem / 0.46

        cost_ha = (urea_kg * price_urea) + (dap_kg * price_dap) + (mop_kg * price_mop)
        rev_ha = (yields / 100.0) * price_fg
        profit_ha = rev_ha - cost_ha
        bcr = np.divide(rev_ha, cost_ha, out=np.zeros_like(rev_ha), where=cost_ha!=0)
        total_npk = n_rates + p_rates + k_rates

        def norm(v):
            mn, mx = np.min(v), np.max(v)
            if mx - mn < 1e-6: return np.zeros_like(v)
            return (v - mn) / (mx - mn)

        n_yield = norm(yields)
        n_bcr = norm(bcr)
        n_fert_inv = 1.0 - norm(total_npk)

        res_row = row_data.to_dict()

        orig_input = pd.DataFrame([row_data]).reindex(columns=FEATURES)
        for col in NUMERIC_FEATURES: orig_input[col] = pd.to_numeric(orig_input[col], errors='coerce')
        for col in CATEGORICAL_FEATURES: orig_input[col] = orig_input[col].fillna('Unknown').astype(str)

        orig_pred = model.predict(orig_input)[0]
        res_row['Current_Yield_Pred'] = orig_pred[0]
        res_row['Current_NPK_Sum'] = orig_pred[1] + orig_pred[2] + orig_pred[3]

        orig_pred_scaled = scaler_y.transform(orig_pred.reshape(1, -1))
        dist, _ = knn.kneighbors(orig_pred_scaled)
        res_row['Current_Reliability'] = 100.0 * np.exp(-np.mean(dist) / scale_factor)

        for profile_name, weights in WEIGHT_PROFILES.items():
            w_y, w_f, w_b = weights
            score = (w_y * n_yield) + (w_f * n_fert_inv) + (w_b * n_bcr)

            best_idx = np.argmax(score)
            prefix = f"Opt_{profile_name}_"

            res_row[prefix + 'Yield'] = yields[best_idx]
            res_row[prefix + 'Profit'] = profit_ha[best_idx]
            res_row[prefix + 'BCR'] = bcr[best_idx]
            res_row[prefix + 'TotalNPK'] = total_npk[best_idx]

            best_target = np.array([[yields[best_idx], n_rates[best_idx], p_rates[best_idx], k_rates[best_idx]]])
            best_target_scaled = scaler_y.transform(best_target)
            dist_opt, _ = knn.kneighbors(best_target_scaled)
            res_row[prefix + 'Reliability'] = 100.0 * np.exp(-np.mean(dist_opt) / scale_factor)

            res_row[prefix + 'Urea'] = urea_kg[best_idx]
            res_row[prefix + 'DAP'] = dap_kg[best_idx]
            res_row[prefix + 'VARTYPE'] = full_grid.iloc[best_idx]['VARTYPE']
            res_row[prefix + 'Srate'] = full_grid.iloc[best_idx]['SRATEHA']

        results.append(res_row)

    return pd.DataFrame(results)

# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    df = load_and_preprocess_data(DATA_FILENAME)
    model, knn, scaler_y, scale_factor = get_model_and_tools(df)

    # Configure Dask Parallelization
    n_partitions = max(2, os.cpu_count())
    ddf = dd.from_pandas(df, npartitions=n_partitions)

    print(f"Starting Dask optimization for {len(df)} rows ({NUM_SIMULATIONS} sims/row) across {n_partitions} partitions...")

    # Infer metadata by processing the first row (prevents schema errors in map_partitions)
    first_row_df = df.iloc[:1]
    meta_df = process_partition(first_row_df, model, knn, scaler_y, scale_factor, df)

    with ProgressBar():
        result_ddf = ddf.map_partitions(
            process_partition,
            model=model,
            knn=knn,
            scaler_y=scaler_y,
            scale_factor=scale_factor,
            df_ref=df,
            meta=meta_df
        )
        result_df = result_ddf.compute()

    result_df.to_csv(OUTPUT_FILENAME, index=False)
    print(f"Optimization complete. Results saved to {OUTPUT_FILENAME}")
