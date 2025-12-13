import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib
import os
import numpy as np
from tqdm.keras import TqdmCallback 
from tqdm import tqdm 

# --- CONFIGURATION (UPDATED TO INCLUDE SPECTRAL FEATURES) ---

# Features derived by kinematicid.py (2D) + spectral_geometry_extractor.py (3D)
INPUT_FEATURE_COLUMNS = [
    # Original 2D Kinematic Features
    'vx', 'vy', 'ax', 'ay', 'omega', 'size_A', 
    # NEW 3D Spectral Z-Depth Features
    'parallax_delta_x',       
    'parallax_delta_y',       
    'parallax_delta_z',       
    'parallax_hypotenuse_length' 
]

# --- NN TARGET DEFINITIONS (UNCHANGED) ---
# 1. Prediction NN 2: Dynamics/Force Prediction Targets (Y_DYNAMICS)
DYNAMICS_TARGET_COLUMNS = [
    'vx_prime',       
    'vy_prime',       
    'ax_prime',       
    'ay_prime',       
    'omega_prime', 
    'size_A_prime' 
]

# 2. Prediction NN 1: Geometric Prediction Targets (Y_GEOMETRY)
GEOMETRY_TARGET_COLUMNS = [
    'start_x_prime', 
    'start_y_prime', 
    'end_x_prime',   
    'end_y_prime'    
]

# --- UTILITY: Keras Model Creation (UNCHANGED) ---
def create_model(input_dim, output_dim):
    """Creates a basic sequential Keras model for regression."""
    model = Sequential()
    # The input dimension now includes the 4 new spectral features
    model.add(Dense(32, activation='relu', input_shape=(input_dim,))) 
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(output_dim, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

# --- 1. KNN CLASSIFICATION TRAINING (NOW USES EXPANDED X) ---

def train_knn_classifier(X: pd.DataFrame, y: pd.Series, model_dir: str):
    """Trains and saves the KNN model and its dedicated scaler for event identification."""
    print("  -> Starting KNN Classification Training (Expanded Feature Set)...")
    
    # Scale data for KNN
    knn_scaler = StandardScaler()
    X_scaled = knn_scaler.fit_transform(X)
    
    # Train KNN Model
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_scaled, y)
    
    # Save the global KNN model and its scaler
    joblib.dump(knn_model, os.path.join(model_dir, 'knn_event_classifier.joblib'))
    joblib.dump(knn_scaler, os.path.join(model_dir, 'knn_feature_scaler.joblib'))
    
    print("  ✅ KNN Classifier and Scaler saved successfully.")

# --- 2. DUAL NN PREDICTION TRAINING (NOW USES EXPANDED X) ---

def train_nn_prediction_models(X: pd.DataFrame, Y_dynamics: pd.DataFrame, Y_geometry: pd.DataFrame, model_dir: str, class_id: int):
    """Trains and saves both Dynamics and Geometry NNs for a specific class ID."""
    
    # Define training parameters
    EPOCHS = 50
    BATCH_SIZE = 32

    # Input Scaling (Shared by both NNs for this class)
    input_scaler = MinMaxScaler() 
    X_scaled = input_scaler.fit_transform(X)
    joblib.dump(input_scaler, os.path.join(model_dir, f'scaler_input_class_{class_id}.joblib'))
    
    # ------------------ DYNAMICS NN TRAINING ------------------
    print(f"  -> Training Dynamics NN (Forces) for Class {class_id}...")
    
    # Dynamics Target Scaling
    dyn_scaler = MinMaxScaler()
    Y_dyn_scaled = dyn_scaler.fit_transform(Y_dynamics)
    
    # Split data 
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_dyn_scaled, test_size=0.2, random_state=42)
    
    # Create and train Dynamics model with TqdmCallback
    dyn_model = create_model(X_train.shape[1], y_train.shape[1])
    dyn_model.fit(
        X_train, y_train, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        verbose=0, 
        callbacks=[TqdmCallback(verbose=1)] 
    )
    
    # Save the Dynamics NN and the corresponding DYNAMICS target scaler
    dyn_model.save(os.path.join(model_dir, f'model_class_{class_id}_dynamics.h5'))
    joblib.dump(dyn_scaler, os.path.join(model_dir, f'scaler_target_class_{class_id}_dynamics.joblib'))
    
    print(f"\n  ✅ Dynamics NN (Forces) saved.")

    # ------------------ GEOMETRY NN TRAINING ------------------
    print(f"  -> Training Geometry NN (Start/End) for Class {class_id}...")
    
    # Geometry Target Scaling
    geom_scaler = MinMaxScaler()
    Y_geom_scaled = geom_scaler.fit_transform(Y_geometry)
    
    # Split data 
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_geom_scaled, test_size=0.2, random_state=42)
    
    # Create and train Geometry model with TqdmCallback
    geom_model = create_model(X_train.shape[1], y_train.shape[1])
    geom_model.fit(
        X_train, y_train, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        verbose=0, 
        callbacks=[TqdmCallback(verbose=1)]
    )
    
    # Save the Geometry NN and the corresponding GEOMETRY target scaler
    geom_model.save(os.path.join(model_dir, f'model_class_{class_id}_geometry.h5'))
    joblib.dump(geom_scaler, os.path.join(model_dir, f'scaler_target_class_{class_id}_geometry.joblib'))
    
    print(f"\n  ✅ Geometry NN (Start/End) saved.")

# --- PUBLIC ENTRY POINT: RENAMED FROM 'main' TO 'run_module_logic' ---
def run_module_logic(master_training_csv_path: str, model_dir: str):
    """
    Main entry point for local_nn_training.py. Loads the master CSV (which should
    now contain both 2D kinematic and 3D spectral features) and trains all models.
    """
    print("🤖 GlimmarianEmbeddedSystems V1 Training Mode")
    
    if not os.path.exists(master_training_csv_path):
        print(f"Error: Master CSV not found at {master_training_csv_path}")
        return

    try:
        # **CRITICAL FIX: Use Python engine and skip bad lines.**
        # This prevents the "Expected 18 fields, saw 19" error caused by stray commas.
        df = pd.read_csv(
            master_training_csv_path,
            engine='python', 
            on_bad_lines='skip' 
        )
    except Exception as e:
        print(f"FATAL ERROR reading master CSV: {e}")
        return
    
    # --- Data Integrity Check ---
    required_feature_cols = INPUT_FEATURE_COLUMNS
    missing_cols = [col for col in required_feature_cols if col not in df.columns]
    
    if missing_cols:
        print(f"\nFATAL ERROR: The master training CSV is missing critical feature columns:")
        # Highlight the new missing spectral features
        print(f"  Missing: {missing_cols}") 
        print("  Please ensure the data was merged correctly with 'spectral_depth_data.csv' before running this module.")
        return

    # --- Step 1: Train KNN Classifier (Global Model) ---
    X_knn = df[required_feature_cols] # Uses the full, expanded feature set
    y_knn = df['vector_class_id']
    train_knn_classifier(X_knn, y_knn, model_dir)

    # --- Step 2: Train Dual NNs (Per Class Model) ---
    unique_classes = df['vector_class_id'].unique()
    
    # Use tqdm to show overall class progress
    for class_id in tqdm(unique_classes, desc="Overall Class Training"):
        print(f"\n--- Starting Training for Class ID: {class_id} ---")
        df_class = df[df['vector_class_id'] == class_id].copy()
        
        # Prepare inputs and targets for this specific class
        X = df_class[required_feature_cols]
        
        # Check if all required target columns exist
        required_targets = DYNAMICS_TARGET_COLUMNS + GEOMETRY_TARGET_COLUMNS
        if not all(col in df_class.columns for col in required_targets):
            print(f"FATAL: Missing one or more target columns ({required_targets}) in data for Class {class_id}. Check kinematicid.py output.")
            continue

        Y_dynamics = df_class[DYNAMICS_TARGET_COLUMNS]
        Y_geometry = df_class[GEOMETRY_TARGET_COLUMNS]
        
        # Ensure sufficient data for training
        if len(X) < 100:
              print(f"⚠️ Skipping Class {class_id}: Insufficient data ({len(X)} lines). Need at least 100.")
              continue

        # Execute the dual training process
        train_nn_prediction_models(X, Y_dynamics, Y_geometry, model_dir, class_id)
    
    print("\nTraining workflow complete. Models now utilize Z-Depth features for enhanced prediction.")

if __name__ == '__main__':
    # Placeholder path configuration - adjust these paths for actual use
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Set the model output directory
    MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
    
    # Set the input data file path (This should be the *merged* CSV)
    MASTER_CSV_PATH = os.path.join(BASE_DIR, "..", "data", "training_sequences", "combined_kinematic_sequences.csv")
    
    # Ensure the model output directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    run_module_logic(MASTER_CSV_PATH, MODEL_DIR)