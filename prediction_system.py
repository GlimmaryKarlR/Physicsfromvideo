# C:\Users\KarlRoesch\physVLA\robobuild\prediction_system.py

import os
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras
from pathlib import Path

# --- CONFIGURATION (Must match kinematicid.py) ---
INPUT_FEATURE_COLUMNS = ['vx', 'vy', 'ax', 'ay', 'omega', 'size_A']
DYNAMICS_TARGET_COLUMNS = ['vx_prime', 'vy_prime', 'ax_prime', 'ay_prime', 'omega_prime', 'size_A_prime']
GEOMETRY_TARGET_COLUMNS = ['start_x_prime', 'start_y_prime', 'end_x_prime', 'end_y_prime']

# --- MODEL/SCALER FILENAMES ---
DYNAMICS_MODEL_NAME = "dynamics_model.h5"
GEOMETRY_MODEL_NAME = "geometry_model.h5"
SCALER_X_NAME = "scaler_X.joblib"
SCALER_YD_NAME = "scaler_YD.joblib" # Y_D: Dynamics Target Scaler
SCALER_YG_NAME = "scaler_YG.joblib" # Y_G: Geometry Target Scaler

# --- 1. UTILITIES ---

def load_models_and_scalers(model_dir: str):
    """Loads the dual Keras models and all three Joblib scalers."""
    print("    -> Loading Keras models and Joblib scalers...")
    model_dir_path = Path(model_dir)
    
    try:
        # Load Models
        dynamics_model = keras.models.load_model(model_dir_path / DYNAMICS_MODEL_NAME)
        geometry_model = keras.models.load_model(model_dir_path / GEOMETRY_MODEL_NAME)
        
        # Load Scalers
        scaler_X = joblib.load(model_dir_path / SCALER_X_NAME)
        scaler_YD = joblib.load(model_dir_path / SCALER_YD_NAME)
        scaler_YG = joblib.load(model_dir_path / SCALER_YG_NAME)
        
        return dynamics_model, geometry_model, scaler_X, scaler_YD, scaler_YG
        
    except FileNotFoundError as e:
        print(f"    ❌ ERROR: Required model or scaler file not found: {e}")
        # Suggest retraining if models are missing
        print("    HINT: You may need to run the 'learn' mode first to train the models.")
        raise
    except Exception as e:
        print(f"    ❌ ERROR loading models/scalers: {e}")
        raise

def translate_prediction_to_robot_action(df_predicted: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the predicted dynamic and geometric features into a final 
    robot control command format (e.g., interpolated move, end-effector state).
    
    This is highly domain-specific and requires interpolation logic.
    """
    print("    -> Translating predictions into robot action commands...")
    
    # Placeholder for translation logic:
    # 1. Determine the path and velocity profile from predicted geometry/dynamics.
    # 2. Interpolate the discrete predicted points into smooth robot joint/Cartesian commands.
    
    # We will create a simple Cartesian command for the first predicted point
    if df_predicted.empty:
        return pd.DataFrame()
        
    # Take the first predicted point as the target move for the robot
    first_prediction = df_predicted.iloc[0]
    
    # The action command needs to be the predicted start/end point of the line at t+1 (Y_GEO)
    # The action should specify the desired movement.
    
    # Example Robot Action Command Format:
    ACTION_COLUMNS = ['Command', 'X', 'Y', 'Z', 'Velocity_Scale', 'Tool_State']
    
    # Using the predicted midpoint for the new line segment as the robot's target X, Y
    # Note: Z is often assumed or set to a known plane/offset
    predicted_midpoint_x = (first_prediction['start_x_prime'] + first_prediction['end_x_prime']) / 2
    predicted_midpoint_y = (first_prediction['start_y_prime'] + first_prediction['end_y_prime']) / 2
    
    # Use predicted velocity scale from dynamics (e.g., magnitude of vx_prime, vy_prime)
    pred_vx = first_prediction['vx_prime']
    pred_vy = first_prediction['vy_prime']
    predicted_vel_scale = np.sqrt(pred_vx**2 + pred_vy**2)
    
    # Create the action row
    action_data = {
        'Command': 'MOVE_TO_MIDPOINT', 
        'X': predicted_midpoint_x, 
        'Y': predicted_midpoint_y, 
        'Z': 0.0, # Placeholder Z value
        'Velocity_Scale': predicted_vel_scale,
        'Tool_State': 'ACTIVATE' # Example tool action
    }
    
    # Return a DataFrame with the action commands
    return pd.DataFrame([action_data], columns=ACTION_COLUMNS)

# --- 2. CORE PREDICTION LOGIC ---

def run_prediction(
    df_features: pd.DataFrame, 
    dynamics_model, 
    geometry_model, 
    scaler_X, 
    scaler_YD, 
    scaler_YG
) -> pd.DataFrame:
    """Performs the dual prediction using the loaded models and scalers."""
    
    if df_features.empty:
        print("    ⚠️ Input features DataFrame is empty. Skipping prediction.")
        return pd.DataFrame()

    # 1. Prepare Input (X)
    X = df_features[INPUT_FEATURE_COLUMNS].values
    
    # 2. Scale Input (X)
    X_scaled = scaler_X.transform(X)
    
    # Models expect a 3D tensor: (samples, time steps, features). Since kinematicid.py 
    # already generates features based on sequences, time_steps = 1.
    X_input = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    
    # 3. Dual Prediction
    print("    -> Predicting Dynamics (Y_D) and Geometry (Y_G) simultaneously...")
    
    # Predict scaled outputs
    YD_pred_scaled = dynamics_model.predict(X_input)
    YG_pred_scaled = geometry_model.predict(X_input)

    # 4. Inverse Transform Outputs
    YD_pred = scaler_YD.inverse_transform(YD_pred_scaled)
    YG_pred = scaler_YG.inverse_transform(YG_pred_scaled)
    
    # 5. Combine results into a DataFrame
    df_predicted_dynamics = pd.DataFrame(YD_pred, columns=DYNAMICS_TARGET_COLUMNS)
    df_predicted_geometry = pd.DataFrame(YG_pred, columns=GEOMETRY_TARGET_COLUMNS)
    
    # Combine the predictions with the original features for context
    df_predicted = pd.concat(
        [df_features.reset_index(drop=True), df_predicted_dynamics, df_predicted_geometry], 
        axis=1
    )
    
    return df_predicted

# --- 3. PUBLIC ENTRY POINT (run_module_logic) ---

def run_module_logic(
    temp_prediction_csv: str, 
    model_dir: str, 
    master_csv_path: str, # Required for normalization/context, though not explicitly used for loading data here
    action_csv_path: str
):
    """
    The unified function called by main.py to execute the full prediction workflow.
    
    Args:
        temp_prediction_csv (str): Path to the single temporary CSV with features to predict (X).
        model_dir (str): Directory containing the Keras models and Joblib scalers.
        master_csv_path (str): Path to the master training CSV (context/reference).
        action_csv_path (str): Final output path for the robot commands.
    """
    print("\n[Prediction System] Starting Robot Action Prediction...")
    
    try:
        # 1. Load Models and Scalers
        dynamics_model, geometry_model, scaler_X, scaler_YD, scaler_YG = load_models_and_scalers(model_dir)
        
        # 2. Load Input Features (X)
        df_features = pd.read_csv(temp_prediction_csv)
        if df_features.empty:
            print("    ⚠️ Temporary prediction CSV is empty. No features to predict.")
            return

        # 3. Run Prediction
        df_predicted = run_prediction(
            df_features, 
            dynamics_model, 
            geometry_model, 
            scaler_X, 
            scaler_YD, 
            scaler_YG
        )
        
        # 4. Translate Predictions to Action Commands
        df_action_commands = translate_prediction_to_robot_action(df_predicted)
        
        # 5. Save Action Commands
        if not df_action_commands.empty:
            df_action_commands.to_csv(action_csv_path, index=False, header=True)
            print(f"    ✅ Action commands saved to: {Path(action_csv_path).name}")
        else:
            print("    ⚠️ No robot actions generated.")

    except Exception as e:
        print(f"    🛑 Prediction System failed: {e}")
        raise # Re-raise to halt the workflow in main.py
        
    print("    -> Prediction System finished execution.")

# --- Standalone Execution Block (Optional) ---

if __name__ == "__main__":
    # This block is for direct testing of the script
    print("Running prediction_system.py in STANDALONE mode.")
    
    # NOTE: To run this standalone, you MUST manually set up a directory with:
    # 1. Trained models (dynamics_model.h5, geometry_model.h5)
    # 2. Scalers (scaler_X.joblib, scaler_YD.joblib, scaler_YG.joblib)
    # 3. A test input CSV (temp_prediction_features.csv)
    pass