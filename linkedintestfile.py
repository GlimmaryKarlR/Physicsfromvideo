import pandas as pd
import os

# --- Configuration ---
# Set the base directory relative to the script execution location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Assumes the script is run from the 'robobuild' folder or similar project structure
MASTER_CSV_PATH = os.path.join(BASE_DIR, '..', 'data', 'training_sequences', 'combined_kinematic_sequences.csv')

def load_data():
    """Loads the kinematic sequence data into a Pandas DataFrame."""
    try:
        print(f"Attempting to load data from: {MASTER_CSV_PATH}")
        df = pd.read_csv(MASTER_CSV_PATH)
        print(f"\n✅ Data loaded successfully. Total sequences: {len(df):,}")
        print("-" * 30)
        return df
    except FileNotFoundError:
        print(f"\n🛑 ERROR: Master CSV not found at {MASTER_CSV_PATH}")
        print("Please ensure you have run the 'learn' mode successfully and the file exists.")
        return None
    except Exception as e:
        print(f"\n🛑 An error occurred during loading: {e}")
        return None

def analyze_kinematic_data(df: pd.DataFrame):
    """Performs common kinematic queries and analysis using Pandas."""

    # 1. Identify critical acceleration events (Anomaly Detection)
    # Define a threshold (e.g., predicted acceleration greater than 500 units)
    ACCELERATION_THRESHOLD = 500
    critical_accel = df[df['ay_prime'] > ACCELERATION_THRESHOLD]

    print(f"1. Critical Events (ay_prime > {ACCELERATION_THRESHOLD}): {len(critical_accel):,}")
    if not critical_accel.empty:
        # Display the first 5 events and key columns
        print("   -> Top 5 Critical Acceleration Events:")
        print(critical_accel[['frame_index', 'vx', 'ay', 'ay_prime', 'vector_class_id']].head())
    print("-" * 30)


    # 2. Filter data for a specific kinematic event class (e.g., Class ID 4)
    # This is useful for training specialized models or analyzing known failure modes
    TARGET_CLASS_ID = 4
    class_data = df[df['vector_class_id'] == TARGET_CLASS_ID]

    print(f"2. Data for Kinematic Class ID {TARGET_CLASS_ID}: {len(class_data):,}")
    if not class_data.empty:
        # Calculate the average predicted angular velocity for this class
        avg_omega_prime = class_data['omega_prime'].mean()
        print(f"   -> Average Predicted Angular Velocity (omega_prime) for Class {TARGET_CLASS_ID}: {avg_omega_prime:.4f}")
    print("-" * 30)


    # 3. Time Series Query: Identify predicted geometric drift (Start X position change)
    # Filter for sequences where the predicted start position deviates significantly from the current one
    GEOMETRY_DEVIATION_THRESHOLD = 2.0
    drift_sequences = df[abs(df['start_x_prime'] - df['start_x']) > GEOMETRY_DEVIATION_THRESHOLD]

    print(f"3. Predicted Geometric Drift (Start X deviation > {GEOMETRY_DEVIATION_THRESHOLD}): {len(drift_sequences):,}")
    if not drift_sequences.empty:
        # Display the predicted change in X position
        drift_sequences['delta_x_prime'] = drift_sequences['start_x_prime'] - drift_sequences['start_x']
        print("   -> Top 5 Predicted Drift Sequences (focusing on geometry):")
        print(drift_sequences[['frame_index', 'start_x', 'start_x_prime', 'delta_x_prime', 'vector_class_id']].head())
    print("-" * 30)


if __name__ == '__main__':
    data = load_data()
    if data is not None:
        analyze_kinematic_data(data)