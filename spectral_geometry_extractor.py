import os
import cv2
import ezdxf
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

# --- 0. FILE PATH CONFIGURATION ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# --- 1. GEOMETRY & PARALLAX CONSTANTS ---

# Simulated Z-depth offset between the two spectral views (in millimeters)
SIMULATED_Z_OFFSET_MM = 500

# Epsilon values for Ramer-Douglas-Peucker line simplification (cv2.approxPolyDP)
# 380nm (Camera 1) - Detailed, standard epsilon
EPSILON_380NM = 0.005 
# 780nm (Camera 2) - Simplified, larger epsilon for longer, more robust line segments
EPSILON_780NM = 0.05 

# --- 2. CORE UTILITY FUNCTIONS ---

def get_line_midpoints(
    image_path: Path, 
    epsilon: float, 
    z_offset: float, 
    wavelength_label: str, 
    dxf_output_path: Path
) -> List[Tuple[float, float, float]]:
    """
    Performs image processing (edge detection, contour finding, simplification) 
    to extract line midpoints and exports a DXF for visual verification.
    """
    # Load image (BGR format by default in OpenCV)
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        # print(f"Error: Could not load image {image_path.name}") # Suppressing non-critical print
        return []

    # Simple edge detection (Canny is standard)
    edges = cv2.Canny(img, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    midpoints = []
    
    # Create DXF for verification
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    for contour in contours:
        # Approximate polygon with Ramer-Douglas-Peucker algorithm
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon * peri, True)
        
        # We only care about segments, so iterate over the approximated vertices
        for i in range(len(approx) - 1):
            start_x, start_y = approx[i][0]
            end_x, end_y = approx[i+1][0]
            
            # Calculate midpoint
            mid_x = (start_x + end_x) / 2.0
            mid_y = (start_y + end_y) / 2.0
            midpoints.append((mid_x, mid_y, z_offset))

            # Export line to DXF for visualization
            msp.add_line((start_x, start_y, z_offset), (end_x, end_y, z_offset), 
                         dxfattribs={'layer': wavelength_label, 'color': 1 if wavelength_label == '380nm' else 2})

    # Save the DXF file using the provided output path
    dxf_filename = dxf_output_path / f"{image_path.stem}.dxf"
    doc.saveas(dxf_filename)
    
    return midpoints

def calculate_parallax_vector(frame_id: int, midpoints_380: List[Tuple[float, float, float]], midpoints_780: List[Tuple[float, float, float]]) -> List[Dict[str, Any]]:
    """
    Calculates the 3D Hypotenuse Vector between the closest midpoints in Euclidean space.
    Note: frame_id is now expected to be an INTEGER.
    """
    if not midpoints_380 or not midpoints_780:
        return []

    spectral_vectors = []
    
    # Convert 780nm midpoints to a NumPy array for efficient distance searching
    m780_array = np.array(midpoints_780)

    for m380_x, m380_y, m380_z in midpoints_380:
        
        # 1. Find the closest match in the 780nm data (Euclidean distance in the XY image plane)
        # Calculate squared differences in X and Y
        diff_sq = (m780_array[:, 0] - m380_x)**2 + (m780_array[:, 1] - m380_y)**2
        
        # Find the index of the minimum distance
        min_idx = np.argmin(diff_sq)
        
        # Retrieve the closest 780nm midpoint
        closest_m780 = m780_array[min_idx]
        m780_x, m780_y, m780_z = closest_m780
        
        # 2. Calculate the Hypotenuse Vector Components
        delta_x = m780_x - m380_x
        delta_y = m780_y - m380_y
        delta_z = m780_z - m380_z 

        # Hypotenuse Length (The magnitude of the 3D parallax vector)
        hypotenuse_length = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
        
        # Store features using the requested column names
        spectral_vectors.append({
            'frame_id': frame_id, # frame_id is now guaranteed to be an integer
            # Kinematic Base Position (The point from which the vector starts)
            'midpoint_380_x': m380_x,
            'midpoint_380_y': m380_y,
            # The Hypotenuse Vector Components (New Z-Depth Features)
            'parallax_delta_x': delta_x,
            'parallax_delta_y': delta_y,
            'parallax_delta_z': delta_z, 
            'parallax_hypotenuse_length': hypotenuse_length
        })
        
    return spectral_vectors


# --- 3. MAIN EXECUTION LOOP (FIXED LOGIC) ---

def run_module_logic(spectral_frames_dir: str, dxf_output_dir: str, depth_csv_path: str):
    """
    Main function to run the entire spectral depth extraction pipeline, 
    using explicit input/output paths from the main orchestrator.
    """
    
    # Convert string paths to Path objects for easy manipulation
    input_folder = Path(spectral_frames_dir)
    dxf_output_folder = Path(dxf_output_dir)
    final_csv_path = Path(depth_csv_path)

    print("-> Starting Spectral Geometry Extractor (Z-Depth Creation)...")
    
    # 1. Setup
    dxf_output_folder.mkdir(parents=True, exist_ok=True)
    all_final_vectors = []
    
    # Find all unique numeric frame IDs (e.g., '000', '026', '052', ...)
    frame_files = [f.stem for f in input_folder.iterdir() if f.suffix == '.jpg']
    
    # FIXED LOGIC: This ensures we isolate only the numeric index
    unique_frame_id_strings = set()
    for f_stem in frame_files:
        # Splits '380nm_000' into ['380nm', '000'] and takes the numeric part
        if '_' in f_stem:
            unique_frame_id_strings.add(f_stem.split('_')[-1])

    # 2. Process Frames in Pairs
    print(f"-> Found {len(unique_frame_id_strings)} unique frames to process.")
    
    # Iterate over the clean numeric strings
    for frame_id_str in tqdm(sorted(list(unique_frame_id_strings)), desc="Processing Frame Pairs", unit="pair"):
        
        # CRITICAL STEP: Convert to INTEGER here for the final CSV consistency
        try:
            frame_id_int = int(frame_id_str)
        except ValueError:
            print(f"Warning: Could not convert frame ID '{frame_id_str}' to integer. Skipping.")
            continue

        # 2.1. Get file paths using the clean numeric string
        path_380 = input_folder / f"380nm_{frame_id_str}.jpg"
        path_780 = input_folder / f"780nm_{frame_id_str}.jpg"
        
        if not path_380.exists() or not path_780.exists():
            print(f"Warning: Skipping frame {frame_id_str} due to missing 380nm or 780nm file.")
            continue
            
        # 2.2. Extract Midpoints and Generate DXF (Z=0)
        midpoints_380 = get_line_midpoints(path_380, EPSILON_380NM, z_offset=0.0, wavelength_label='380nm', dxf_output_path=dxf_output_folder)
        
        # 2.3. Extract Midpoints and Generate DXF (Z=-500mm)
        midpoints_780 = get_line_midpoints(path_780, EPSILON_780NM, z_offset=-SIMULATED_Z_OFFSET_MM, wavelength_label='780nm', dxf_output_path=dxf_output_folder)
        
        # 2.4. Calculate the Hypotenuse Vector (The Z-Depth Feature)
        # PASS THE INTEGER ID
        vectors = calculate_parallax_vector(frame_id_int, midpoints_380, midpoints_780)
        all_final_vectors.extend(vectors)

    # 3. Save Final Features to CSV
    if all_final_vectors:
        df_output = pd.DataFrame(all_final_vectors)
        
        # Ensure the frame_id column is explicitly of integer type (for merging)
        # This acts as an extra safety measure against mixed data types
        df_output['frame_id'] = df_output['frame_id'].astype(int) 
        
        # Append data to the CSV if it exists, otherwise write new header
        write_header = not final_csv_path.exists()
        df_output.to_csv(final_csv_path, mode='a', header=write_header, index=False)
        
        print(f"\n✅ Spectral depth vectors saved successfully to: {final_csv_path}")
        print(f"   Total vectors generated: {len(df_output):,}")
    else:
        print("\n🛑 No spectral depth vectors were generated. Check input images and constants.")


if __name__ == '__main__':
    # For local testing, use the hardcoded paths, but main.py uses run_module_logic
    SPECTRAL_FRAMES_FOLDER = Path(BASE_PATH) / '..' / 'data' / 'temp_spectral_frames'
    SPECTRAL_DXF_OUTPUT = Path(BASE_PATH) / '..' / 'data' / 'temp_spectral_dxf'
    FINAL_DEPTH_FEATURES_CSV = Path(BASE_PATH) / '..' / 'data' / 'training_sequences' / 'spectral_depth_data.csv'

    run_module_logic(
        str(SPECTRAL_FRAMES_FOLDER), 
        str(SPECTRAL_DXF_OUTPUT), 
        str(FINAL_DEPTH_FEATURES_CSV)
    )