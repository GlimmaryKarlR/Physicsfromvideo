# C:\Users\KarlRoesch\physVLA\robobuild\kinematicid.py - FINAL, CLEANED, AND HANDLE-BASED VERSION

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple

# --- CONFIGURATION ---
DELTA_TIME = 1.0 / 30.0 # Time between frames (1 / FPS)
SEQUENCE_LENGTH = 4 # Frames needed: t-1, t, t+1, t+2

# --- COLUMN DEFINITIONS ---

# 1. Columns expected in the raw input CSV (from main.py merged data)
SPECTRAL_FEATURE_COLUMNS = [
    'parallax_delta_x',
    'parallax_delta_y',
    'parallax_delta_z',
    'parallax_hypotenuse_length'
]
INPUT_COLUMNS = [
    'timestep', 'handle', 'vector_class_id', 
    'start_x', 'start_y', 'end_x', 'end_y'
] + SPECTRAL_FEATURE_COLUMNS # <-- ADDED SPECTRAL INPUTS

# 2. Features derived by this script
INPUT_FEATURE_COLUMNS = ['vx', 'vy', 'ax', 'ay', 'omega', 'size_A']

# 3. Target definitions
DYNAMICS_TARGET_COLUMNS = ['vx_prime', 'vy_prime', 'ax_prime', 'ay_prime', 'omega_prime', 'size_A_prime']
GEOMETRY_TARGET_COLUMNS = ['start_x_prime', 'start_y_prime', 'end_x_prime', 'end_y_prime']

# 4. Combined output columns for the final combined_kinematic_sequences.csv
# CRITICAL FIX: Changed 'timestep' to 'frame_id' to match the rename in step 4 of run_module_logic.
OUTPUT_COLUMNS = (
    ['vector_class_id', 'handle', 'frame_id'] +  # <-- FIX APPLIED HERE
    INPUT_FEATURE_COLUMNS +                     # Calculated 2D Inputs (X_KINEMATIC)
    SPECTRAL_FEATURE_COLUMNS +                  # Raw 3D Inputs (X_SPECTRAL)
    DYNAMICS_TARGET_COLUMNS +                   # Calculated Dynamics Targets (Y_DYNAMICS)
    GEOMETRY_TARGET_COLUMNS                     # Raw Geometry Targets (Y_GEOMETRY)
)

# --- 1. GEOMETRIC & KINEMATIC UTILITIES (UNCHANGED) ---

def calculate_midpoint_and_angle(p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Calculates midpoint coordinates, line length, and angle (rotation)."""
    midpoint = (p1 + p2) / 2
    length = np.linalg.norm(p2 - p1)
    # Use 2D distance for angle, as Z-axis is often depth, not rotation axis
    angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) 
    return midpoint, length, angle

def calculate_kinematics(
    p1: np.ndarray, a1: float, m1: float, 
    p2: np.ndarray, a2: float, m2: float, 
    p3: np.ndarray, a3: float, m3: float, 
    dt: float
) -> Dict[str, float]:
    """Calculates the 6 kinematic features based on three sequential midpoints (p1, p2, p3)."""
    
    # Velocity (p1 -> p2)
    v_x = (p2[0] - p1[0]) / dt
    v_y = (p2[1] - p1[1]) / dt
    
    # Velocity Prime (p2 -> p3) 
    v_prime_x = (p3[0] - p2[0]) / dt
    v_prime_y = (p3[1] - p2[1]) / dt

    # Acceleration
    a_x = (v_prime_x - v_x) / dt
    a_y = (v_prime_y - v_y) / dt
    
    # Angular Velocity
    angle_diff = a2 - a1
    while angle_diff > np.pi:
        angle_diff -= 2 * np.pi
    while angle_diff < -np.pi:
        angle_diff += 2 * np.pi
        
    omega = angle_diff / dt

    # Size Proxy
    size_A = m2 

    return {
        'vx': v_x, 'vy': v_y, 
        'ax': a_x, 'ay': a_y, 
        'omega': omega, 
        'size_A': size_A
    }

# --- 2. TRACKING AND SEQUENCE GENERATION (CRITICALLY REFACTORED) ---

def track_and_extract_features(df_raw: pd.DataFrame, dt: float) -> pd.DataFrame:
    """
    Uses the 'handle' column to track a single entity through time and extracts
    the necessary 4-frame kinematic sequences (t-1, t, t+1, t+2).
    """
    global SEQUENCE_LENGTH
    
    if len(df_raw) < SEQUENCE_LENGTH: 
        # Ensure returned DataFrame has the full structure, even if empty
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # 1. Calculate midpoints and angles for the entire sequence of this handle
    df_raw['p1'] = df_raw.apply(lambda row: np.array([row['start_x'], row['start_y']]), axis=1)
    df_raw['p2'] = df_raw.apply(lambda row: np.array([row['end_x'], row['end_y']]), axis=1)
    
    pre_calc = df_raw.apply(lambda row: calculate_midpoint_and_angle(row['p1'], row['p2']), axis=1, result_type='expand')
    df_raw['midpoint'] = pre_calc[0]
    df_raw['length'] = pre_calc[1]
    df_raw['angle'] = pre_calc[2]
    
    kinematic_sequences = []
    
    # Loop over the dataframe for this handle (t is the center of the 4-frame window)
    for i in range(1, len(df_raw) - (SEQUENCE_LENGTH - 2)): 
        
        # t-1, t, t+1, t+2
        row_t_1 = df_raw.iloc[i-1]
        row_t = df_raw.iloc[i]
        row_t_plus_1 = df_raw.iloc[i+1]
        row_t_plus_2 = df_raw.iloc[i+2]
        
        # --- Extract kinematic components for t-1, t, t+1 ---
        p_t_1, a_t_1, m_t_1 = row_t_1[['midpoint', 'angle', 'length']]
        p_t, a_t, m_t = row_t[['midpoint', 'angle', 'length']]
        p_t_plus_1, a_t_plus_1, m_t_plus_1 = row_t_plus_1[['midpoint', 'angle', 'length']]

        # 1. Calculate Input Features (X) at time 't' using sequence (t-1, t, t+1)
        features_t = calculate_kinematics(p_t_1, a_t_1, m_t_1, p_t, a_t, m_t, p_t_plus_1, a_t_plus_1, m_t_plus_1, dt)
        
        # 2. Calculate Target Features (Y_DYN) at time 't+1' using sequence (t, t+1, t+2)
        p_t_plus_2, a_t_plus_2, m_t_plus_2 = row_t_plus_2[['midpoint', 'angle', 'length']]
        features_t_prime = calculate_kinematics(p_t, a_t, m_t, p_t_plus_1, a_t_plus_1, m_t_plus_1, p_t_plus_2, a_t_plus_2, m_t_plus_2, dt)
        
        # --- Combine and Store ---
        sequence = {k: v for k, v in features_t.items()} # 2D Kinematic Inputs
        
        # Dynamics Targets (Y_DYN)
        for k, v in features_t_prime.items():
            sequence[k + '_prime'] = v 

        # Geometry Targets (Y_GEO) - Position of the line at t+1
        sequence['start_x_prime'] = row_t_plus_1['start_x']
        sequence['start_y_prime'] = row_t_plus_1['start_y']
        sequence['end_x_prime'] = row_t_plus_1['end_x']
        sequence['end_y_prime'] = row_t_plus_1['end_y']
        
        # --- CRITICAL FIX: Pass through 3D Spectral Features at time 't' ---
        for col in SPECTRAL_FEATURE_COLUMNS:
             # The raw spectral data is the input X_SPECTRAL, taken at time t
             sequence[col] = row_t[col]
        
        # Context/Meta-data
        sequence['vector_class_id'] = row_t['vector_class_id']
        sequence['handle'] = row_t['handle'] 
        sequence['timestep'] = row_t['timestep'] # <-- This is what gets renamed to 'frame_id' later

        kinematic_sequences.append(sequence)

    # Note: OUTPUT_COLUMNS includes 'frame_id', but the DataFrame is created with 'timestep'. 
    # This list comprehension ensures the DataFrame creation succeeds by substituting 'frame_id' back to 'timestep' for the initial DataFrame columns.
    return pd.DataFrame(kinematic_sequences, columns=[col if col != 'frame_id' else 'timestep' for col in OUTPUT_COLUMNS])


# --- 3. MAIN FUNCTION WRAPPER (CRITICALLY REFACTORED) ---

def run_module_logic(input_file_path: str, output_file_path: str, mode: str = 'learn'):
    """
    Entry point for feature generation, called by main.py. 
    Groups by 'handle' for reliable tracking and generates 2D kinematic features.
    """
    print(f"    -> KinematicID running in '{mode.upper()}' mode for 2D feature generation.")
    
    # 1. Load the single raw CSV file
    try:
        df_raw = pd.read_csv(input_file_path, low_memory=False)
        
        # --- Renaming Check (Only for internal consistency) ---
        rename_map = {'frame_id': 'timestep'}
        df_raw.rename(columns=rename_map, inplace=True)

        # The column check: ensure all geometric inputs + SPECTRAL inputs are present
        required_cols = INPUT_COLUMNS
        
        if not all(col in df_raw.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df_raw.columns]
            print(f"    ❌ FATAL: Input CSV is missing required columns. Missing: {missing_cols}")
            print(f"    ❌ Found: {df_raw.columns.tolist()}")
            raise ValueError("Input CSV missing critical geometric/spectral columns.")

    except Exception as e:
        print(f"    ❌ FATAL: Failed to read input CSV {Path(input_file_path).name}. Error: {e}")
        raise 
    
    # 2. Filter, Sort, and Group
    # Ensure types are correct for grouping and calculation
    df_raw['vector_class_id'] = df_raw['vector_class_id'].astype(float)
    df_raw['timestep'] = df_raw['timestep'].astype(int)
    
    # Filter for class 1.0 (Assuming class 1 is the primary object of interest)
    df_class = df_raw[df_raw['vector_class_id'] == 1.0].copy() 
    df_class.sort_values(by=['handle', 'timestep'], inplace=True)
    
    if df_class.empty:
        print(f"    ⚠️ No entities found for vector_class_id=1.0 in {Path(input_file_path).name}.")
        return

    # 3. Core feature generation: Group by handle and apply tracking function
    print(f"    -> Grouping {len(df_class)} raw entries by unique entity handle...")
    
    # Apply the tracking function to each sequence belonging to one 'handle'
    df_features_list = []
    for handle, df_group in df_class.groupby('handle'):
        # For a group, df_group contains all frames for that single entity
        df_features_list.append(track_and_extract_features(df_group, DELTA_TIME))
        
    df_features = pd.concat([df for df in df_features_list if not df.empty], ignore_index=True)
    
    if df_features.empty:
        print(f"    ⚠️ No kinematic features (sequences of length {SEQUENCE_LENGTH}) generated for {Path(input_file_path).name}.")
        return

    # 4. Write Output based on Mode
    # NOTE: The 'timestep' column is renamed back to 'frame_id' 
    df_features.rename(columns={'timestep': 'frame_id'}, inplace=True) 
    
    try:
        if mode == 'learn':
            # --- CRITICAL WRITE FIX APPLIED HERE ---
            if not os.path.exists(output_file_path):
                # If file doesn't exist, create it with header using 'w' mode (overwrite)
                write_mode = 'w'
                write_header = True
                print(f"    -> Master CSV not found. Writing in OVERWRITE mode (first session run) to set full header.")
            else:
                # If file exists, append without header using 'a' mode
                write_mode = 'a'
                write_header = False
                print(f"    -> Master CSV found. Writing in APPEND mode (subsequent run).")
                
            # Execute the write with the determined mode and header settings
            df_features[OUTPUT_COLUMNS].to_csv(
                output_file_path, 
                mode=write_mode, 
                index=False, 
                header=write_header
            )
            print(f"    ✅ Generated {len(df_features)} sequences (2D + 3D) and appended to master kinematic CSV.")
            
        elif mode == 'predict':
            # Use 'w' mode for prediction since the file is temporary/session-specific
            df_features[OUTPUT_COLUMNS].to_csv(output_file_path, mode='w', index=False, header=True)
            print(f"    ✅ Generated {len(df_features)} sequences (2D + 3D) and saved to temporary prediction CSV.")
            
        else:
            raise ValueError(f"Invalid mode '{mode}' provided to kinematicid.main.")
            
    except Exception as e:
        print(f"    ❌ Failed to write output CSV: {e}")
        raise

    print("    -> KinematicID module finished execution.")