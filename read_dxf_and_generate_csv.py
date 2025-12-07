
import ezdxf
import os
import glob
import numpy as np
import csv
from math import ceil
from tqdm import tqdm

# --- Configuration (UPDATE THESE) ---
DXF_DIRECTORY = r"C:\Users\KarlRoesch\physVLA\bulkdxfout"
CSV_OUTPUT_FILE = r"C:\Users\KarlRoesch\physVLA\bulkcsvout\vector_features.csv"
NUM_CLASSES = 10
CSV_HEADER = [
    "timestep",
    "vector_class_id",
    "decomp_x",
    "decomp_y",
    "start_x",
    "start_y",
    "end_x",
    "end_y"
]

# --- Helper Functions ---

def calculate_line_features(entity):
    """
    Extracts P1, P2, length, and decomposition from a simple LINE entity.
    """
    
    # Accessing coordinates explicitly using .x and .y attributes (Safest method)
    x1 = entity.dxf.start.x
    y1 = entity.dxf.start.y
    x2 = entity.dxf.end.x
    y2 = entity.dxf.end.y

    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    
    # Calculate length (Euclidean distance)
    length = np.linalg.norm(p2 - p1)
    
    return p1, p2, length

def get_all_vector_lengths(dxf_files):
    """Pass 1: Reads all DXF files to collect all segment lengths for global classification."""
    all_lengths = []
    
    for dxf_file in tqdm(dxf_files, desc="Pass 1: Collecting All Lengths"):
        try:
            doc = ezdxf.readfile(dxf_file)
            msp = doc.modelspace()
            
            # Query simple 'LINE' entities
            for entity in msp.query('LINE'): 
                _, _, length = calculate_line_features(entity)
                if length > 0.0001: # Check for a non-zero length
                    all_lengths.append(length)
                        
        except Exception:
            continue
            
    return all_lengths

def calculate_class_metrics(all_lengths, num_classes=10):
    """Calculates L_min, L_max, and Delta_L for classification."""
    if not all_lengths:
        return None, None, None

    lengths = np.array(all_lengths)
    L_min = lengths.min()
    L_max = lengths.max()

    if L_max == L_min:
        return L_min, L_max, 1 

    delta_L = (L_max - L_min) / num_classes
    
    return L_min, L_max, delta_L

def assign_class_id(length, L_min, delta_L, num_classes):
    """Assigns the 1-10 class ID based on pre-calculated metrics."""
    if delta_L is None or delta_L == 0:
        return 1

    relative_position = length - L_min
    class_id = int(ceil(relative_position / delta_L))
    
    return max(1, min(class_id, num_classes))

# --- Main Processing Function ---

def read_dxf_and_generate_csv():
    """Main function to run both passes and write the final CSV."""
    dxf_files = glob.glob(os.path.join(DXF_DIRECTORY, "*.dxf"))
    dxf_files.sort() 
    
    if not dxf_files:
        print(f"Error: No DXF files found in {DXF_DIRECTORY}")
        return

    # --- PASS 1: Global Classification Analysis ---
    all_lengths = get_all_vector_lengths(dxf_files)
    L_min, L_max, delta_L = calculate_class_metrics(all_lengths, NUM_CLASSES)

    if L_min is None:
        print("ERROR: No valid vectors found with length > 0. Cannot classify.")
        return

    print(f"\n--- Global Classification Metrics ---")
    print(f"Total Vector Range: {L_min:.2f} to {L_max:.2f}")
    print(f"Class Interval Width (Delta L): {delta_L:.2f}")
    print(f"Number of Vectors Classified: {len(all_lengths)}")
    
    # --- PASS 2: Sequential Feature Extraction and CSV Output ---
    
    with open(CSV_OUTPUT_FILE, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(CSV_HEADER)
        
        global_timestep = 0 
        
        for dxf_file in tqdm(dxf_files, desc="Pass 2: Generating Features and CSV"):
            global_timestep += 1
            
            try:
                doc = ezdxf.readfile(dxf_file)
                msp = doc.modelspace()
                
                for entity in msp.query('LINE'): 
                    
                    p1, p2, length = calculate_line_features(entity)

                    if length > 0.0001:
                        
                        start_x, start_y = p1
                        end_x, end_y = p2

                        # 1. Calculate Dimensional Length Components (Delta X and Delta Y)
                        decomp_x_dim = end_x - start_x
                        decomp_y_dim = end_y - start_y
                        
                        # 2. Normalization to Unit Vector (Relative to Start Point)
                        # This scales the length along each axis so that the total vector length is 1.
                        decomp_x_norm = decomp_x_dim / length
                        decomp_y_norm = decomp_y_dim / length
                        
                        # 3. Vector Class ID
                        class_id = assign_class_id(length, L_min, delta_L, NUM_CLASSES)
                        
                        # Create the CSV row (rounded for clean output)
                        row = [
                            global_timestep,
                            class_id,
                            round(decomp_x_norm, 4), # Normalized Length along X-axis (-1 to 1)
                            round(decomp_y_norm, 4), # Normalized Length along Y-axis (-1 to 1)
                            round(start_x, 4),
                            round(start_y, 4),
                            round(end_x, 4),
                            round(end_y, 4)
                        ]
                        
                        writer.writerow(row)
                            
            except Exception as e:
                print(f"\nWarning: Could not process {os.path.basename(dxf_file)}. Skipping. Error: {e}")
                continue

    print(f"\n✅ Successfully generated feature CSV.")
    print(f"Output saved to {CSV_OUTPUT_FILE}")
    print(f"Total time steps processed: {global_timestep}")

if __name__ == "__main__":
    read_dxf_and_generate_csv()