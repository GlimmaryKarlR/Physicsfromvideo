import pandas as pd
import numpy as np
import json
import joblib
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os
from tqdm.autonotebook import tqdm # Used for progress bars
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress warnings
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# warnings.filterfilterwarnings("ignore", category=InconsistentVersionWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)

# --- GLOBAL CONFIGURATION (8 COLUMNS, 6 FEATURES) ---
PREFIX = '120625'
DOMAIN_NAME = 'A'
TIMESTEPS = 1
FEATURES = 6 
BATCH_SIZE = 32
LEARNING_RATE = 0.000005
EPOCHS = 25
TARGET_CLASSES = range(1, 11) # Classes 1 through 10

# --- CRITICAL LOCAL FILE PATH CONFIGURATION ---
# Base directory where your combined CSVs are located (e.g., C:/data/)
LOCAL_BASE_PATH = Path(r'C:\Users\KarlRoesch\physVLA\bulkcsvout') 

# Output directory for models and scalers
OUTPUT_DIR = LOCAL_BASE_PATH / 'model_deployment_segmented'

# CRITICAL: Pattern to collect multiple files if they exist.
# Assuming your final CSV name might be like 'vector_features_part_1.csv'
# If you only have one file named 'vector_features.csv', adjust this pattern.
FILE_NAME_PATTERN = 'vector_features*.csv' 
# ----------------------------------------------------

# --- CRITICAL COLUMN NAME UPDATE ---
# This matches the final column schema from the CSV generation script.
COLUMN_NAMES = [
    'timestep',         # ts_id (Used for sorting/indexing, but dropped for features)
    'vector_class_id',  # vector_class (Target column for segmentation)
    'decomp_x',         # Feature 1 (Normalized)
    'decomp_y',         # Feature 2 (Normalized)
    'start_x',          # Feature 3
    'start_y',          # Feature 4
    'end_x',            # Feature 5
    'end_y'             # Feature 6
]
FEATURE_COLUMNS = COLUMN_NAMES[2:] # ['decomp_x', 'decomp_y', 'start_x', 'start_y', 'end_x', 'end_y']
EXPECTED_COLUMNS = 8 # Total columns in the CSV

# Ensure the calculated FEATURES matches the column list length
if len(FEATURE_COLUMNS) != FEATURES:
    raise ValueError(f"Feature count mismatch: FEATURES is {FEATURES}, but {len(FEATURE_COLUMNS)} were defined in FEATURE_COLUMNS.")


# ----------------------------------------------------
# 1. Setup and Path Initialization
# ----------------------------------------------------
print("Initializing local paths...")

# Remove Google Drive dependencies
# drive.mount('/content/drive', force_remount=True) # REMOVED

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Input Directory Base: {LOCAL_BASE_PATH}")
print(f"Output directory ensured at: {OUTPUT_DIR}")

# ----------------------------------------------------
# 2. Data Loading (Load and Aggregate Multiple Files)
# ----------------------------------------------------

all_domain_data = []
print(f"\n--- Aggregating data from {LOCAL_BASE_PATH} ---")

# CRITICAL: Use glob to find all files matching the pattern
input_files = list(LOCAL_BASE_PATH.glob(FILE_NAME_PATTERN))

if not input_files:
    print(f"\n❌ FATAL: No files found matching pattern '{FILE_NAME_PATTERN}' in {LOCAL_BASE_PATH}. Exiting.")
    exit()

for FULL_FILE_PATH in tqdm(input_files, desc="Loading and Combining Files"):
    try:
        # Use header=None if the CSVs don't have a header row
        df_temp = pd.read_csv(FULL_FILE_PATH, header=0, names=COLUMN_NAMES, skipinitialspace=True)

        # Validation Check
        if df_temp.shape[1] != EXPECTED_COLUMNS:
            print(f"\n❌ ERROR: File {FULL_FILE_PATH.name} has {df_temp.shape[1]} columns, but {EXPECTED_COLUMNS} were expected. Skipping.")
            continue
        
        all_domain_data.append(df_temp)
    except Exception as e:
        print(f"\n❌ Error processing file {FULL_FILE_PATH.name}: {e}. Skipping this file.")

if not all_domain_data:
    print("\n❌ FATAL: No valid data loaded from any input file. Exiting.")
    exit()

df_combined = pd.concat(all_domain_data, ignore_index=True)
print(f"Data aggregation complete: {len(df_combined)} total rows from {len(all_domain_data)} file(s).")

unique_classes = sorted(df_combined['vector_class_id'].unique())
print(f"Verification: Classes found in the aggregated dataset: {unique_classes}")


# ----------------------------------------------------
# 3. AUTOMATED SEGMENTED TRAINING LOOP
# ----------------------------------------------------

print(f"\n--- Starting Automated Segmented Training for {len(TARGET_CLASSES)} Classes ---")

for TARGET_CLASS in tqdm(TARGET_CLASSES, desc="Training Classes"):

    # 3.1 SEGMENTATION
    # Use the new column name 'vector_class_id' for filtering
    df_segment = df_combined[df_combined['vector_class_id'] == TARGET_CLASS].copy()

    if df_segment.empty:
        print(f"\nSkipping Class {TARGET_CLASS}: No data found in the aggregated set.")
        continue

    # 3.2 SCALING AND SEQUENCE CREATION
    data_raw = df_segment[FEATURE_COLUMNS].values.astype(np.float32)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_raw)

    def create_sequences(data, timesteps):
        # Time-series regression: X_t -> Y_t+1 (Predicting the next feature vector)
        X, Y = [], []
        for i in range(len(data) - timesteps):
            X.append(data[i:(i + timesteps)])
            Y.append(data[i + timesteps])
        return np.array(X), np.array(Y)

    X_seq, Y_target = create_sequences(data_scaled, TIMESTEPS)

    # Prepare data for Dense layer: Flatten input sequences
    X_final = X_seq.reshape(X_seq.shape[0], TIMESTEPS * FEATURES)
    Y_final = Y_target.reshape(Y_target.shape[0], FEATURES) # Regression target (6 features)

    # Handle insufficient sequence data
    if X_final.shape[0] < 2:
        print(f"\nSkipping Class {TARGET_CLASS}: Not enough sequence data after segmentation.")
        continue

    # Splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, Y_final, test_size=0.3, random_state=42
    )

    input_dim = X_train.shape[1]

    # 3.3 MODEL DEFINITION (Regression)
    model = Sequential([
        Dense(units=32, activation='relu', input_shape=(input_dim,)),
        Dense(units=16, activation='relu'),
        Dense(units=FEATURES, activation='linear') # Output units = 6 features
    ])

    stable_optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=stable_optimizer, loss='mse') # Mean Squared Error (MSE)

    # 3.4 TRAINING
    # Verbose=0 to hide training output during the loop
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=0 
    )

    # 3.5 SERIALIZATION (SAVING ASSETS)
    model_name = f'{PREFIX}_model_{TARGET_CLASS}.keras'
    scaler_name = f'{PREFIX}_scaler_{TARGET_CLASS}.joblib'

    save_model(model, OUTPUT_DIR / model_name)
    joblib.dump(scaler, OUTPUT_DIR / scaler_name)

    # print(f"\n✅ Successfully trained and saved assets for Class {TARGET_CLASS}.")

print("\n--- ALL SEGMENTED REGRESSION TRAINING RUNS ARE COMPLETE ---")
print(f"Total {len(TARGET_CLASSES)} models and {len(TARGET_CLASSES)} scalers saved in: {OUTPUT_DIR}")