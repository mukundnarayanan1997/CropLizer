import pandas as pd
import numpy as np
import os
import joblib
import optuna
import optuna.visualization as vis
import openpyxl
from tqdm.auto import tqdm
import scipy.sparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

# Model Imports
from sklearn.linear_model import LinearRegression
# SVR and MultiOutputRegressor removed
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# TensorFlow / Keras Imports
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

# --- FOLDER PATHS ---
MODEL_SAVE_DIR = "/scratch/mukund_n.iitr/CropLizer/v3/Models"
DATA_SAVE_DIR = "/scratch/mukund_n.iitr/CropLizer/v3/Data"
PLOT_SAVE_DIR = "/scratch/mukund_n.iitr/CropLizer/v3/Optimization Plots"
STUDY_SAVE_DIR = "/scratch/mukund_n.iitr/CropLizer/v3/Optuna Studies"

# --- 1. FEATURE DEFINITIONS ---

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

ECONOMIC_COLS = ['FGPRICE_quintal', 'Urea_price']

# --- 2. GLOBAL VARS FOR TRAINING ---
X_train_transformed = None
y_train_df = None
X_test_transformed = None
y_test_df = None
X_train_3d = None
X_test_3d = None
NUM_FEATURES = 0


# --- 3. DATA LOADING & PREPROCESSING ---

def load_and_preprocess_data(filepath):
    global NUMERIC_FEATURES, CATEGORICAL_FEATURES, FEATURES

    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath, low_memory=False, on_bad_lines='skip')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

    print(f"Successfully loaded {len(df)} rows initially.")

    all_cols_to_load = FEATURES + TARGETS + ECONOMIC_COLS
    all_cols_to_load = list(dict.fromkeys(all_cols_to_load))

    missing_cols = [col for col in all_cols_to_load if col not in df.columns]
    if missing_cols:
        print(f"Warning: The following columns are missing from the CSV: {missing_cols}")
        all_cols_to_load = [col for col in all_cols_to_load if col in df.columns]

    df = df[all_cols_to_load]

    NUMERIC_FEATURES = [col for col in NUMERIC_FEATURES if col in df.columns]
    CATEGORICAL_FEATURES = [col for col in CATEGORICAL_FEATURES if col in df.columns]
    FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    numeric_cols_to_convert = [col for col in NUMERIC_FEATURES + TARGETS + ECONOMIC_COLS if col in df.columns]
    for col in numeric_cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    initial_rows = len(df)
    df = df.dropna(subset=TARGETS)
    print(f"Dropped {initial_rows - len(df)} rows due to missing target data.")

    if 'yield_kg_ha' in df.columns and len(df) > 0:
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

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna('Unknown').astype(str)

    print("Data loading and cleaning complete.")
    return df


# --- 4. KERAS MODEL BUILDERS ---

def build_transformer_encoder_block(inputs, num_heads, key_dim, ff_dim, dropout=0.1):
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout
    )(inputs, inputs)
    attn_output = layers.Add()([inputs, attn_output])
    attn_output = layers.LayerNormalization(epsilon=1e-6)(attn_output)

    ffn_output = layers.Dense(ff_dim, activation="relu")(attn_output)
    ffn_output = layers.Dense(inputs.shape[-1])(ffn_output)
    ffn_output = layers.Dropout(dropout)(ffn_output)
    
    ffn_output = layers.Add()([attn_output, ffn_output])
    ffn_output = layers.LayerNormalization(epsilon=1e-6)(ffn_output)
    
    return ffn_output

def build_nn_model(trial, input_shape):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_shape,)))

    n_layers = trial.suggest_int('nn_n_layers', 1, 3)
    for i in range(n_layers):
        units = trial.suggest_int(f'nn_units_l{i}', 32, 256, log=True)
        model.add(layers.Dense(units, activation='relu'))
        dropout_rate = trial.suggest_float(f'nn_dropout_l{i}', 0.1, 0.5)
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(len(TARGETS)))

    learning_rate = trial.suggest_float('nn_learning_rate', 1e-4, 1e-2, log=True)
    model.compile(
        loss='mean_squared_error',
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
    )
    return model

def build_lstm_model(trial, input_shape_3d):
    inputs = layers.Input(shape=input_shape_3d)
    
    lstm_units = trial.suggest_int('lstm_units', 32, 256, log=True)
    x = layers.LSTM(lstm_units)(inputs)
    
    dropout_rate = trial.suggest_float('lstm_dropout', 0.1, 0.5)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(len(TARGETS))(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)

    learning_rate = trial.suggest_float('lstm_learning_rate', 1e-4, 1e-2, log=True)
    model.compile(
        loss='mean_squared_error',
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
    )
    return model

def build_transformer_model(trial, input_shape_3d):
    inputs = layers.Input(shape=input_shape_3d)

    num_heads = trial.suggest_int('transformer_num_heads', 2, 8, step=2)
    key_dim = trial.suggest_int('transformer_key_dim', 32, 128, log=True)
    ff_dim = trial.suggest_int('transformer_ff_dim', key_dim * 2, key_dim * 4, step=32)
    dropout_rate = trial.suggest_float('transformer_dropout', 0.1, 0.5)
    
    x = build_transformer_encoder_block(
        inputs, 
        num_heads=num_heads, 
        key_dim=key_dim, 
        ff_dim=ff_dim, 
        dropout=dropout_rate
    )
    
    x = layers.GlobalAveragePooling1D()(x)
    
    outputs = layers.Dense(len(TARGETS))(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)

    learning_rate = trial.suggest_float('transformer_learning_rate', 1e-4, 1e-2, log=True)
    model.compile(
        loss='mean_squared_error',
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
    )
    return model

# --- 5. OPTUNA OBJECTIVE FUNCTION ---

def objective(trial, model_name):
    global X_train_transformed, y_train_df, X_test_transformed, y_test_df
    global X_train_3d, X_test_3d, NUM_FEATURES

    model = None

    try:
        if model_name == 'LinearRegression':
            model = LinearRegression()

        elif model_name == 'DecisionTree':
            max_depth = trial.suggest_int('dt_max_depth', 5, 50, log=True)
            min_samples_split = trial.suggest_int('dt_min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('dt_min_samples_leaf', 1, 20)
            
            model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf
            )

        elif model_name == 'RandomForest':
            n_estimators = trial.suggest_int('rf_n_estimators', 50, 300)
            max_depth = trial.suggest_int('rf_max_depth', 10, 50, log=True)
            min_samples_split = trial.suggest_int('rf_min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('rf_min_samples_leaf', 1, 20)
            max_features = trial.suggest_float('rf_max_features', 0.1, 1.0)
            
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                n_jobs=-1
            )

        elif model_name == 'NN':
            model = build_nn_model(trial, input_shape=NUM_FEATURES)

        elif model_name == 'LSTM':
            model = build_lstm_model(trial, input_shape_3d=(1, NUM_FEATURES))

        elif model_name == 'Transformer':
            model = build_transformer_model(trial, input_shape_3d=(1, NUM_FEATURES))

        # --- Fit Model ---
        if model_name in ['LinearRegression', 'DecisionTree', 'RandomForest']:
            model.fit(X_train_transformed, y_train_df)
            y_train_pred = model.predict(X_train_transformed)
            y_test_pred = model.predict(X_test_transformed)

        elif model_name in ['NN']:
            epochs = trial.suggest_int('nn_epochs', 20, 100)
            batch_size = trial.suggest_int('nn_batch_size', 32, 128, log=True)
            
            early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            model.fit(
                X_train_transformed, y_train_df,
                validation_split=0.2,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=0
            )
            
            y_train_pred = model.predict(X_train_transformed)
            y_test_pred = model.predict(X_test_transformed)

        elif model_name in ['LSTM', 'Transformer']:
            epochs = trial.suggest_int(f'{model_name.lower()}_epochs', 20, 100)
            batch_size = trial.suggest_int(f'{model_name.lower()}_batch_size', 32, 128, log=True)

            early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            model.fit(
                X_train_3d, y_train_df,
                validation_split=0.2,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=0
            )
            
            y_train_pred = model.predict(X_train_3d)
            y_test_pred = model.predict(X_test_3d)

        train_r2 = r2_score(y_train_df, y_train_pred)
        test_r2 = r2_score(y_test_df, y_test_pred)

        if not np.isfinite(train_r2): train_r2 = -1.0
        if not np.isfinite(test_r2): test_r2 = -1.0

        return train_r2, test_r2

    except Exception as e:
        print(f"Error during trial {trial.number} for {model_name}: {e}")
        return -1.0, -1.0


# --- 6. RETRAIN AND SAVE BEST MODEL ---

def retrain_and_save_model(trial, model_name, preprocessor, X, y):
    params = trial.params
    print(f"\nRetraining best {model_name} model on full dataset...")
    
    model = None
    is_keras_model = False

    if model_name == 'LinearRegression':
        model = LinearRegression()

    elif model_name == 'DecisionTree':
        model = DecisionTreeRegressor(
            max_depth=params['dt_max_depth'],
            min_samples_split=params['dt_min_samples_split'],
            min_samples_leaf=params['dt_min_samples_leaf']
        )

    elif model_name == 'RandomForest':
        model = RandomForestRegressor(
            n_estimators=params['rf_n_estimators'],
            max_depth=params['rf_max_depth'],
            min_samples_split=params['rf_min_samples_split'],
            min_samples_leaf=params['rf_min_samples_leaf'],
            max_features=params['rf_max_features'],
            n_jobs=-1
        )

    elif model_name in ['NN', 'LSTM', 'Transformer']:
        is_keras_model = True
        
        X_transformed = preprocessor.fit_transform(X)
        if scipy.sparse.issparse(X_transformed):
            X_transformed = X_transformed.toarray()
            
        num_features_full = X_transformed.shape[1]

        if model_name == 'NN':
            model = build_nn_model(trial, input_shape=num_features_full)
            epochs = params['nn_epochs']
            batch_size = params['nn_batch_size']
            print(f"Fitting final {model_name} model...")
            model.fit(X_transformed, y, epochs=epochs, batch_size=batch_size, verbose=1)
            
        elif model_name in ['LSTM', 'Transformer']:
            X_transformed_3d = np.expand_dims(X_transformed, axis=1)
            input_shape_3d_full = (1, num_features_full)
            
            if model_name == 'LSTM':
                model = build_lstm_model(trial, input_shape_3d_full)
                epochs = params['lstm_epochs']
                batch_size = params['lstm_batch_size']
            else:
                model = build_transformer_model(trial, input_shape_3d_full)
                epochs = params['transformer_epochs']
                batch_size = params['transformer_batch_size']

            print(f"Fitting final {model_name} model...")
            model.fit(X_transformed_3d, y, epochs=epochs, batch_size=batch_size, verbose=1)

    if is_keras_model:
        preprocessor_filename = os.path.join(MODEL_SAVE_DIR, f'best_model_{model_name}_preprocessor.joblib')
        model_filename = os.path.join(MODEL_SAVE_DIR, f'best_model_{model_name}.keras')
        
        joblib.dump(preprocessor, preprocessor_filename)
        model.save(model_filename)
        print(f"Saved Keras model to '{model_filename}' and preprocessor to '{preprocessor_filename}'")
    else:
        final_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        print(f"Fitting final {model_name} pipeline...")
        final_pipeline.fit(X, y)
        
        model_filename = os.path.join(MODEL_SAVE_DIR, f'best_model_{model_name}_pipeline.joblib')
        joblib.dump(final_pipeline, model_filename)
        print(f"Saved scikit-learn pipeline to '{model_filename}'")


# --- 7. MAIN EXECUTION ---

def main():
    global X_train_transformed, y_train_df, X_test_transformed, y_test_df
    global X_train_3d, X_test_3d, NUM_FEATURES

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(DATA_SAVE_DIR, exist_ok=True)
    os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
    os.makedirs(STUDY_SAVE_DIR, exist_ok=True)
    print(f"Models will be saved to: {MODEL_SAVE_DIR}")
    print(f"Data will be saved to: {DATA_SAVE_DIR}")
    print(f"Plots will be saved to: {PLOT_SAVE_DIR}")
    print(f"Optuna studies will be saved to: {STUDY_SAVE_DIR}")

    data_df = load_and_preprocess_data("/scratch/mukund_n.iitr/CropLizer/Data/NUE_survey_dataset_v2.csv")
    if data_df is None:
        print("Failed to load data. Exiting.")
        return

    X = data_df[FEATURES]
    y = data_df[TARGETS]

    X_train, X_test, y_train_df, y_test_df = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print("Saving common train/test splits...")
    try:
        X_train.to_csv(os.path.join(DATA_SAVE_DIR, "common_X_train.csv"), index=False)
        X_test.to_csv(os.path.join(DATA_SAVE_DIR, "common_X_test.csv"), index=False)
        y_train_df.to_csv(os.path.join(DATA_SAVE_DIR, "common_y_train.csv"), index=False)
        y_test_df.to_csv(os.path.join(DATA_SAVE_DIR, "common_y_test.csv"), index=False)
        print(f"Successfully saved train/test CSVs to {DATA_SAVE_DIR}")
    except Exception as e:
        print(f"Error saving train/test splits: {e}")

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

    print("Fitting preprocessor...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    if scipy.sparse.issparse(X_train_transformed):
        print("Converting sparse matrix to dense array for Keras...")
        X_train_transformed = X_train_transformed.toarray()
        X_test_transformed = X_test_transformed.toarray()

    NUM_FEATURES = X_train_transformed.shape[1]
    
    X_train_3d = np.expand_dims(X_train_transformed, axis=1)
    X_test_3d = np.expand_dims(X_test_transformed, axis=1)

    print(f"Data preprocessed. Number of features: {NUM_FEATURES}")

    results_data = []
    sl_no = 1
    best_trials_per_model = {}
    MODELS_TO_RUN = ['LinearRegression', 'DecisionTree', 'RandomForest', 'NN', 'LSTM', 'Transformer']
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    for model_name in MODELS_TO_RUN:
        print(f"\n--- Processing Model: {model_name} ---")
        
        storage_path = f"sqlite:///{os.path.join(STUDY_SAVE_DIR, f'{model_name}_study.db')}"
        
        study = optuna.create_study(
            storage=storage_path,
            study_name=model_name,
            directions=['maximize', 'maximize'],
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            load_if_exists=True
        )

        print(f"Starting/Resuming Optuna optimization for {model_name} (100 trials)...")
        n_trials = 100
        
        n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_remaining = max(0, n_trials - n_completed)
        
        if n_remaining == 0:
            print(f"Study for {model_name} already has {n_completed} trials. Skipping optimization.")
        else:
            print(f"Study has {n_completed} completed trials. Running {n_remaining} more.")
            with tqdm(total=n_remaining, desc=f"Trials for {model_name}") as pbar:
                def tqdm_callback(study, trial):
                    pbar.update(1)

                study.optimize(
                    lambda trial: objective(trial, model_name),
                    n_trials=n_remaining,
                    callbacks=[tqdm_callback]
                )
        print(f"Optuna study for {model_name} complete.")
        print(f"Study results saved to: {storage_path.replace('sqlite:///', '')}")

        # --- 5c. Save Optimization Plots for this model ---
        print(f"Saving optimization plots for {model_name}...")
        
        if len(study.trials) > 0:
            # History
            try:
                fig_history = vis.plot_optimization_history(study, target_names=["Train R2", "Test R2"])
                fig_history.write_image(os.path.join(PLOT_SAVE_DIR, f"{model_name}_optimization_history.png"))
                print(f"Saved optimization history for {model_name}")
            except Exception as e:
                print(f"Could not save optimization history for {model_name}: {e}")

            # Importances (needs > 1 completed trial usually and variance)
            try:
                fig_importance = vis.plot_param_importances(study, target=lambda t: t.values[1], target_name="Test R2")
                fig_importance.write_image(os.path.join(PLOT_SAVE_DIR, f"{model_name}_param_importances.png"))
                print(f"Saved param importances for {model_name}")
            except Exception as e:
                print(f"Could not save param importances for {model_name}: {e}")

            # Pareto (Multi-objective)
            try:
                fig_pareto = vis.plot_pareto_front(study, target_names=["Train R2", "Test R2"])
                fig_pareto.write_image(os.path.join(PLOT_SAVE_DIR, f"{model_name}_pareto_front.png"))
                print(f"Saved pareto front for {model_name}")
            except Exception as e:
                print(f"Could not save pareto front for {model_name}: {e}")
        else:
            print("No trials completed, skipping plots.")

        print(f"Found {len(study.best_trials)} trials on the Pareto front for {model_name}.")
        
        best_model_test_r2 = -np.inf
        best_model_trial = None

        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            
            train_r2, test_r2 = trial.values
            
            if test_r2 > best_model_test_r2:
                best_model_test_r2 = test_r2
                best_model_trial = trial

            # --- MODIFIED SECTION: SAVE ALL HYPERPARAMETERS IN ONE ROW ---
            params_dict = trial.params
            results_data.append([
                sl_no, model_name, str(params_dict), train_r2, test_r2
            ])
            
            sl_no += 1
        
        if best_model_trial:
            best_trials_per_model[model_name] = (best_model_trial, best_model_test_r2)
            print(f"Best trial for {model_name}: Trial {best_model_trial.number} (Test R2: {best_model_test_r2:.4f})")
        else:
            print(f"No completed trials found for {model_name}.")
    
    print("\nProcessing combined results for Excel...")
    if not results_data:
        print("No results data collected. Skipping Excel save.")
    else:
        # --- MODIFIED DATAFRAME COLUMNS ---
        results_df = pd.DataFrame(results_data, columns=[
            'sl no', 'model name', 'hyperparameters', 'train r2', 'test r2'
        ])
        
        results_df.sort_values(by='test r2', ascending=False, inplace=True)
        
        excel_path = "model_hyperparameter_results.xlsx"
        try:
            results_df.to_excel(excel_path, index=False)
            print(f"\nSuccessfully saved combined hyperparameter results to {excel_path}")
        except Exception as e:
            print(f"\nError saving Excel file: {e}")

    if best_trials_per_model:
        print(f"\nFound best trials for {len(best_trials_per_model)} model types.")
        
        for model_name, (trial, test_r2) in best_trials_per_model.items():
            print(f"\nBest model for {model_name}:")
            print(f"  Test R2: {test_r2:.4f}")
            print(f"  Train R2: {trial.values[0]:.4f}")
            print(f"  Params: {trial.params}")
            
            retrain_and_save_model(trial, model_name, preprocessor, X, y)
            
    else:
        print("\nNo successful trials completed. No models will be saved.")


if __name__ == "__main__":
    main()
