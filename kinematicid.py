import pandas as pd
import numpy as np
import os
from scipy.spatial.distance import cdist
from pathlib import Path
from typing import Dict, List, Tuple

# --- CONFIGURATION ---
# 1. Input: Directory containing the raw CSV files.
INPUT_CSV_DIR = Path(r"C:\Users\KarlRoesch\physVLA\bulkcsvout")
# 2. Output: Directory for the final, sequenced CSVs used for training.
OUTPUT_DIR = Path("C:/Users/KarlRoesch/physVLA/training_sequences")
# 3. Time difference between frames (required for velocity/acceleration)
#    ADJUST THIS VALUE based on your video source FPS.
DELTA_TIME = 1.0 / 30.0 

# --- COLUMN DEFINITIONS ---
# Raw Input Columns
INPUT_COLUMNS = [
    'timestep', 
    'vector_class_id', 
    'decomp_x', 
    'decomp_y', 
    'start_x', 
    'start_y', 
    'end_x', 
    'end_y'
]
# Kinematic Features (New Columns)
FEATURE_COLUMNS = ['vx', 'vy', 'ax', 'ay', 'omega', 'size_A']

# --- 1. GEOMETRIC & KINEMATIC UTILITIES ---

def calculate_midpoint_and_angle(p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Calculates midpoint coordinates, line length, and angle (rotation)."""
    midpoint = (p1 + p2) / 2
    length = np.linalg.norm(p2 - p1) # Proxy for size/mass
    angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    return midpoint, length, angle

def calculate_kinematics(
    p1: np.ndarray, a1: float, m1: float, 
    p2: np.ndarray, a2: float, m2: float, 
    p3: np.ndarray, a3: float, m3: float, 
    dt: float
) -> Dict[str, float]:
    """Calculates the 6 kinematic features based on three sequential midpoints (p1, p2, p3)."""
    
    # Velocity (at frame 2, using p1 -> p2 displacement)
    v_x = (p2[0] - p1[0]) / dt
    v_y = (p2[1] - p1[1]) / dt
    
    # Velocity Prime (using p2 -> p3 displacement)
    v_prime_x = (p3[0] - p2[0]) / dt
    v_prime_y = (p3[1] - p2[1]) / dt

    # Acceleration (at frame 2)
    a_x = (v_prime_x - v_x) / dt
    a_y = (v_prime_y - v_y) / dt
    
    # Angular Velocity (at frame 2)
    angle_diff = a2 - a1
    while angle_diff > np.pi:
        angle_diff -= 2 * np.pi
    while angle_diff < -np.pi:
        angle_diff += 2 * np.pi
        
    omega = angle_diff / dt

    # Size Proxy (at frame 2)
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
    Tracks lines across sequential timestamps and calculates kinematic features.
    """
    
    SEQUENCE_LENGTH = 3 
    
    # 1. Pre-process and calculate midpoints/angles for all lines
    df_raw['p1'] = df_raw.apply(lambda row: np.array([row['start_x'], row['start_y']]), axis=1)
    df_raw['p2'] = df_raw.apply(lambda row: np.array([row['end_x'], row['end_y']]), axis=1)
    
    pre_calc = df_raw.apply(lambda row: calculate_midpoint_and_angle(row['p1'], row['p2']), axis=1, result_type='expand')
    df_raw['midpoint'] = pre_calc[0]
    df_raw['length'] = pre_calc[1]
    df_raw['angle'] = pre_calc[2]
    
    timestamps = sorted(df_raw['timestep'].unique())
    if len(timestamps) < SEQUENCE_LENGTH:
        return pd.DataFrame()

    kinematic_data = []
    
    # 2. Iterate through sequences of 3 frames (t-1, t, t+1)
    for i in range(1, len(timestamps) - 1):
        t_minus_1 = timestamps[i-1]
        t_current = timestamps[i]
        t_plus_1 = timestamps[i+1]
        
        # Get data frames for the three time steps
        df_t_1 = df_raw[df_raw['timestep'] == t_minus_1].reset_index(drop=True)
        df_t = df_raw[df_raw['timestep'] == t_current].reset_index(drop=True)
        df_t_plus_1 = df_raw[df_raw['timestep'] == t_plus_1].reset_index(drop=True)

        if df_t.empty or df_t_1.empty or df_t_plus_1.empty: continue

        # For every line/CAD element identified in the CURRENT frame (t):
        for line_idx_t in df_t.index:
            p_t = df_t.loc[line_idx_t, 'midpoint']
            a_t = df_t.loc[line_idx_t, 'angle']
            m_t = df_t.loc[line_idx_t, 'length']
            
            # Find closest line in t-1 (Midpoint Tracking backwards)
            midpoints_t_1 = np.vstack(df_t_1['midpoint'].to_numpy())
            distances_t_1 = cdist(p_t.reshape(1, -1), midpoints_t_1)
            idx_t_1 = np.argmin(distances_t_1)
            
            # Find closest line in t+1 (Midpoint Tracking forwards)
            midpoints_t_plus_1 = np.vstack(df_t_plus_1['midpoint'].to_numpy())
            distances_t_plus_1 = cdist(p_t.reshape(1, -1), midpoints_t_plus_1)
            idx_t_plus_1 = np.argmin(distances_t_plus_1)
            
            # Extract properties of matched lines
            p_t_1 = df_t_1.loc[idx_t_1, 'midpoint']
            a_t_1 = df_t_1.loc[idx_t_1, 'angle']
            m_t_1 = df_t_1.loc[idx_t_1, 'length']
            
            p_t_plus_1 = df_t_plus_1.loc[idx_t_plus_1, 'midpoint']
            a_t_plus_1 = df_t_plus_1.loc[idx_t_plus_1, 'angle']
            m_t_plus_1 = df_t_plus_1.loc[idx_t_plus_1, 'length']

            # Calculate the kinematic vector for the line at time t
            features = calculate_kinematics(
                p_t_1, a_t_1, m_t_1, p_t, a_t, m_t, 
                p_t_plus_1, a_t_plus_1, m_t_plus_1, dt
            )
            
            # Include the original metadata from the current frame (t)
            features['vector_class_id'] = df_t.loc[line_idx_t, 'vector_class_id']
            features['timestep'] = t_current
            
            kinematic_data.append(features)

    # 3. Final structuring
    df_final = pd.DataFrame(kinematic_data)
    if df_final.empty:
        return pd.DataFrame()
        
    return df_final[['vector_class_id', 'timestep'] + FEATURE_COLUMNS]


def process_all_raw_data(input_dir: Path, output_dir: Path):
    """
    Main function to loop through all CSV files, process data, and save sequences.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_raw_files = list(input_dir.glob('*.csv'))

    if not all_raw_files:
        print(f"FATAL: No CSV files found in {input_dir}")
        return

    print(f"Found {len(all_raw_files)} CSV files to process.")
    
    combined_output_file = output_dir / "combined_kinematic_sequences.csv"
    
    # ⚠️ WARNING: The following line will NOT delete the file if you want to run this multiple times.
    # If you need to overwrite, you must uncomment the removal logic:
    # if combined_output_file.exists():
    #     os.remove(combined_output_file)
    #     print(f"Removed existing file: {combined_output_file.name}")

    for file_path in all_raw_files:
        print(f"\n--- Processing: {file_path.name} ---")
        
        try:
            # Load raw data using the specified INPUT_COLUMNS
            df_raw = pd.read_csv(file_path, header=None, names=INPUT_COLUMNS)
            
            # Process each unique vector_class_id within the file separately
            for class_id in df_raw['vector_class_id'].unique():
                df_class = df_raw[df_raw['vector_class_id'] == class_id].copy()
                
                # Sort by timestep to ensure correct sequential data
                df_class.sort_values(by='timestep', inplace=True)
                
                print(f"  -> Extracting features for Class {class_id} ({len(df_class)} lines)...")
                
                # Extract kinematic features
                df_features = track_and_extract_features(df_class, DELTA_TIME)
                
                if not df_features.empty:
                    # Append to the combined output file
                    write_header = not combined_output_file.exists()
                    df_features.to_csv(
                        combined_output_file, 
                        mode='a', 
                        index=False, 
                        header=write_header
                    )
                    print(f"  ✅ Saved {len(df_features)} sequences for Class {class_id}.")
                else:
                    print(f"  ⚠️ Skipped Class {class_id}: Insufficient sequential data found.")

        except Exception as e:
            print(f"FATAL ERROR processing {file_path.name}: {e}")
            continue

    print(f"\n--- Process Complete ---")
    print(f"Final training sequence file created: {combined_output_file}")


if __name__ == "__main__":
    
    # Ensure the input directory path uses forward slashes or raw string literals
    INPUT_CSV_DIR = Path(r"C:\Users\KarlRoesch\physVLA\bulkcsvout")
    
    process_all_raw_data(INPUT_CSV_DIR, OUTPUT_DIR)