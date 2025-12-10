# C:\Users\KarlRoesch\physVLA\robobuild\kinematicid.py

import pandas as pd
import numpy as np
import os
from scipy.spatial.distance import cdist
from pathlib import Path
from typing import Dict, List, Tuple

# --- CONFIGURATION ---
# These are kept for reference, but the file paths are now handled by main.py
DELTA_TIME = 1.0 / 30.0 # Time between frames (1 / FPS)
SEQUENCE_LENGTH = 4 # Frames needed: t-1, t, t+1, t+2

# --- SUBSAMPLING SAFEGUARD ---
# Maximum number of vectors allowed in a frame for tracking before random subsampling is applied.
MAX_VECTORS_PER_FRAME = 1000 

# --- COLUMN DEFINITIONS ---
# All columns set to float64 for maximum parsing robustness.
INPUT_DTYPES = {
    # NOTE: These column names should match the output of read_dxf_and_generate_csv.py
    'timestep': 'float64',        # (Assumed equivalent to 'frame_id')
    'vector_class_id': 'float64', # (Assumed equivalent to 'class_id')
    'decomp_x': 'float64',        # (Not used in provided logic, but kept for structure)
    'decomp_y': 'float64',        # (Not used in provided logic, but kept for structure)
    'start_x': 'float64',         
    'start_y': 'float64',         
    'end_x': 'float64',           
    'end_y': 'float64',           
    # NOTE: The raw CSV from the previous stage likely contains more columns like 'length', etc.
}
# Using a simplified version of the input columns based on the logic's requirements
INPUT_COLUMNS = ['timestep', 'vector_class_id', 'start_x', 'start_y', 'end_x', 'end_y'] 

# Features derived by kinematicid.py
INPUT_FEATURE_COLUMNS = ['vx', 'vy', 'ax', 'ay', 'omega', 'size_A']
DYNAMICS_TARGET_COLUMNS = ['vx_prime', 'vy_prime', 'ax_prime', 'ay_prime', 'omega_prime', 'size_A_prime']
GEOMETRY_TARGET_COLUMNS = ['start_x_prime', 'start_y_prime', 'end_x_prime', 'end_y_prime']

# Combined output columns for the final combined_kinematic_sequences.csv
OUTPUT_COLUMNS = ['vector_class_id', 'timestep'] + INPUT_FEATURE_COLUMNS + DYNAMICS_TARGET_COLUMNS + GEOMETRY_TARGET_COLUMNS

# --- 1. GEOMETRIC & KINEMATIC UTILITIES ---

def calculate_midpoint_and_angle(p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Calculates midpoint coordinates, line length, and angle (rotation)."""
    midpoint = (p1 + p2) / 2
    length = np.linalg.norm(p2 - p1)
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

# --- 2. TRACKING AND SEQUENCE GENERATION ---

def track_and_extract_features(df_raw: pd.DataFrame, dt: float) -> pd.DataFrame:
    """
    Tracks lines across sequential timestamps, calculates features, and applies subsampling
    to individual frames if their vector count exceeds MAX_VECTORS_PER_FRAME.
    """
    global MAX_VECTORS_PER_FRAME
    global SEQUENCE_LENGTH
    
    # 1. Pre-process and calculate midpoints/angles for all lines
    df_raw['p1'] = df_raw.apply(lambda row: np.array([row['start_x'], row['start_y']]), axis=1)
    df_raw['p2'] = df_raw.apply(lambda row: np.array([row['end_x'], row['end_y']]), axis=1)
    
    pre_calc = df_raw.apply(lambda row: calculate_midpoint_and_angle(row['p1'], row['p2']), axis=1, result_type='expand')
    df_raw['midpoint'] = pre_calc[0]
    df_raw['length'] = pre_calc[1]
    df_raw['angle'] = pre_calc[2]
    
    timestamps = sorted(df_raw['timestep'].unique())
    if len(timestamps) < SEQUENCE_LENGTH: 
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    kinematic_sequences = []
    
    # Function to apply subsampling
    def subsample_frame(df_frame, t_val):
        if len(df_frame) > MAX_VECTORS_PER_FRAME:
            # print(f"  [SUBSAMPLE] Timestep {int(t_val)} reduced from {len(df_frame)} to {MAX_VECTORS_PER_FRAME} vectors.")
            return df_frame.sample(n=MAX_VECTORS_PER_FRAME, random_state=42).reset_index(drop=True)
        return df_frame.reset_index(drop=True)

    # Loop over all possible starting frames (t)
    for i in range(1, len(timestamps) - (SEQUENCE_LENGTH - 2)): 
        t_current = timestamps[i]
        t_minus_1 = timestamps[i-1]
        t_plus_1 = timestamps[i+1]
        t_plus_2 = timestamps[i+2]
        
        # Filter DataFrames for the four time steps
        df_t_1 = df_raw[df_raw['timestep'] == t_minus_1]
        df_t = df_raw[df_raw['timestep'] == t_current]
        df_t_plus_1 = df_raw[df_raw['timestep'] == t_plus_1]
        df_t_plus_2 = df_raw[df_raw['timestep'] == t_plus_2]

        if df_t.empty or df_t_1.empty or df_t_plus_1.empty or df_t_plus_2.empty: continue

        # --- Subsample all required frames if they exceed the max limit ---
        df_t_1_sub = subsample_frame(df_t_1, t_minus_1)
        df_t_sub = subsample_frame(df_t, t_current)
        df_t_plus_1_sub = subsample_frame(df_t_plus_1, t_plus_1)
        df_t_plus_2_sub = subsample_frame(df_t_plus_2, t_plus_2)

        # The loop iterates over the subsampled CURRENT frame (df_t_sub)
        for line_idx_t in df_t_sub.index:
            p_t = df_t_sub.loc[line_idx_t, 'midpoint']
            
            # --- Tracking for INPUT (X) at time t ---
            
            # Tracking t-1 (Closest line in previous frame to current line's midpoint)
            midpoints_t_1 = np.vstack(df_t_1_sub['midpoint'].to_numpy())
            idx_t_1 = np.argmin(cdist(p_t.reshape(1, -1), midpoints_t_1)) 
            
            # Tracking t+1 (Closest line in next frame to current line's midpoint)
            midpoints_t_plus_1 = np.vstack(df_t_plus_1_sub['midpoint'].to_numpy())
            idx_t_plus_1 = np.argmin(cdist(p_t.reshape(1, -1), midpoints_t_plus_1))
            
            # Get properties from SUBSAMPLED dataframes
            p_t_1, a_t_1, m_t_1 = df_t_1_sub.loc[idx_t_1, ['midpoint', 'angle', 'length']]
            p_t, a_t, m_t = df_t_sub.loc[line_idx_t, ['midpoint', 'angle', 'length']]
            p_t_plus_1, a_t_plus_1, m_t_plus_1 = df_t_plus_1_sub.loc[idx_t_plus_1, ['midpoint', 'angle', 'length']]

            # Calculate Input Features (X) at time 't' using sequence (t-1, t, t+1)
            features_t = calculate_kinematics(p_t_1, a_t_1, m_t_1, p_t, a_t, m_t, p_t_plus_1, a_t_plus_1, m_t_plus_1, dt)
            
            
            # --- Tracking for TARGET (Y) at time t+1 ---
            
            # Tracking t+2 (Closest line in t+2 frame to t+1 line's midpoint)
            midpoints_t_plus_2 = np.vstack(df_t_plus_2_sub['midpoint'].to_numpy())
            idx_t_plus_2 = np.argmin(cdist(p_t_plus_1.reshape(1, -1), midpoints_t_plus_2)) 
            
            # Get properties from SUBSAMPLED dataframes
            p_t_plus_2, a_t_plus_2, m_t_plus_2 = df_t_plus_2_sub.loc[idx_t_plus_2, ['midpoint', 'angle', 'length']]
            
            # Calculate Target Features (Y) at time 't+1' using sequence (t, t+1, t+2)
            features_t_prime = calculate_kinematics(p_t, a_t, m_t, p_t_plus_1, a_t_plus_1, m_t_plus_1, p_t_plus_2, a_t_plus_2, m_t_plus_2, dt)
            
            # --- Combine and Store ---
            
            # 1. Input Features (X)
            sequence = {k: v for k, v in features_t.items()} 
            
            # 2. Dynamics Targets (Y_DYN)
            for k, v in features_t_prime.items():
                    sequence[k + '_prime'] = v 

            # 3. Geometry Targets (Y_GEO) - Position of the line at t+1 (from its index)
            sequence['start_x_prime'] = df_t_plus_1_sub.loc[idx_t_plus_1, 'start_x']
            sequence['start_y_prime'] = df_t_plus_1_sub.loc[idx_t_plus_1, 'start_y']
            sequence['end_x_prime'] = df_t_plus_1_sub.loc[idx_t_plus_1, 'end_x']
            sequence['end_y_prime'] = df_t_plus_1_sub.loc[idx_t_plus_1, 'end_y']
            
            # 4. Context/Meta-data
            sequence['vector_class_id'] = df_t_sub.loc[line_idx_t, 'vector_class_id']
            sequence['timestep'] = t_current

            kinematic_sequences.append(sequence)

    df_final = pd.DataFrame(kinematic_sequences)
    if df_final.empty:
        # Return an empty DataFrame with the correct column structure
        return pd.DataFrame(columns=OUTPUT_COLUMNS) 
        
    return df_final[OUTPUT_COLUMNS]


# --- 3. MAIN FUNCTION WRAPPER (Renamed to run_module_logic) ---

def run_module_logic(input_file_path: str, output_file_path: str, mode: str = 'learn'):
    """
    Renamed entry point for feature generation, called by main.py. 
    Loads a single raw CSV, generates features, and writes them to the specified output file based on the mode.
    
    Args:
        input_file_path (str): Path to the raw CSV generated by the vision system.
        output_file_path (str): Path to either the master CSV ('learn') or a temp CSV ('predict').
        mode (str): 'learn' (append to master) or 'predict' (overwrite temp file).
    """
    print(f"    -> KinematicID running in '{mode.upper()}' mode.")
    global INPUT_COLUMNS, INPUT_DTYPES, DELTA_TIME, OUTPUT_COLUMNS
    
    # 1. Load the single raw CSV file
    try:
        # NOTE: The header=None/names=INPUT_COLUMNS assumes the previous stage writes without a header 
        # or that the column names are exactly as expected. We use a more robust read.
        df_raw = pd.read_csv(
            input_file_path, 
            sep=',', 
            encoding='utf8', 
            # We assume the read_dxf_and_generate_csv.py output columns are:
            # ['frame_id', 'length', 'class_id', 'start_x', 'start_y', 'end_x', 'end_y', ...]
            # We rename them to match the logic internally:
            names=INPUT_COLUMNS + ['length_placeholder'], # Add placeholder to match number of columns needed
            skiprows=1, # Assuming the previous script wrote a header row
            low_memory=False
        )
        # Rename columns to match the internal logic if necessary
        df_raw.rename(columns={'timestep': 'timestep', 'vector_class_id': 'vector_class_id'}, inplace=True)
        
    except Exception as e:
        print(f"    ❌ FATAL: Failed to read input CSV {Path(input_file_path).name}. Error: {e}")
        raise # Stop the process
        # return # Use return instead of raise if you want the main menu to reappear

    df_features = pd.DataFrame()
    if not df_raw.empty:
        # Filter for class 1 and sort by timestep
        # NOTE: Assumes 'vector_class_id' is the column name corresponding to the geometric class ID.
        df_class = df_raw[df_raw['vector_class_id'] == 1.0].copy()
        df_class.sort_values(by='timestep', inplace=True)
        
        # Core feature generation (includes subsampling)
        df_features = track_and_extract_features(df_class, DELTA_TIME)
    
    if df_features.empty:
        print(f"    ⚠️ No features generated for {Path(input_file_path).name}.")
        return

    # 2. Write Output based on Mode
    try:
        if mode == 'learn':
            # Append features to the master training CSV (Checking if header is needed)
            write_header = not os.path.exists(output_file_path)
            df_features.to_csv(output_file_path, mode='a', index=False, header=write_header)
            print(f"    ✅ Generated {len(df_features)} sequences and appended to master CSV.")
            
        elif mode == 'predict':
            # Overwrite/Create temporary prediction CSV
            df_features.to_csv(output_file_path, mode='w', index=False, header=True)
            print(f"    ✅ Generated {len(df_features)} sequences and saved to temporary prediction CSV.")
            
        else:
            raise ValueError(f"Invalid mode '{mode}' provided to kinematicid.main.")
            
    except Exception as e:
        print(f"    ❌ Failed to write output CSV: {e}")
        raise

    print("    -> KinematicID module finished execution.")

# The if __name__ == "__main__": block is omitted to ensure the script runs only when called as a module.