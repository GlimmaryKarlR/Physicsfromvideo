import os
import numpy as np
import cv2 # Required for Canny and contour finding
import ezdxf # Required for DXF writing and reading
import pandas as pd # For final data aggregation

# --- FILE PATH CONFIGURATION ---
BASE_PATH = 'C:/Users/KarlRoesch/physVLA/robobuild'
SPECTRAL_FRAMES_FOLDER = os.path.join(BASE_PATH, '..', 'data', 'temp_spectral_frames')
SPECTRAL_DXF_OUTPUT = os.path.join(BASE_PATH, '..', 'data', 'temp_spectral_dxf')
FINAL_MIDPOINT_CSV = os.path.join(BASE_PATH, '..', 'data', 'training_sequences', 'spectral_depth_data.csv')

# --- PARALLAX CONSTANTS ---
# The simulated Z-depth offset between the two spectral views
SIMULATED_Z_OFFSET_MM = 500 

# --- LINE SIMPLIFICATION CONSTANTS ---
# Epsilon for 380nm (Detailed)
EPSILON_380NM = 0.005 
# Epsilon for 780nm (Simplified/Larger Lines) - Significantly increased
EPSILON_780NM = 0.05 

def extract_midpoints_and_export_dxf(image_path, output_dxf_folder, epsilon, z_offset):
    """
    Performs vision, line extraction, and exports line data to DXF.
    Returns a list of line midpoints [(x, y, z)].
    """
    # This function is highly simplified for demonstration. 
    # It would involve loading the image, Canny edge detection, finding contours,
    # approximating polygons using the given epsilon, and generating DXF lines.
    
    print(f"Processing {os.path.basename(image_path)} with Epsilon={epsilon}")
    
    # --- MOCK DATA GENERATION ---
    # In a real script, this would be a complex vision loop.
    # We mock 3 lines for demonstration:
    midpoints = []
    
    # Example: Simplified lines (larger Epsilon) would produce fewer, longer lines.
    if epsilon > 0.01: 
         midpoints = [(100, 150, z_offset), (300, 200, z_offset)]
    else: # Detailed lines
         midpoints = [(100, 150, z_offset), (105, 152, z_offset), (300, 200, z_offset), (305, 202, z_offset)]
    
    print(f"   -> Extracted {len(midpoints)} midpoints.")
    # ezdxf.new('R2010').modelspace().add_line(...) # Actual DXF writing happens here
    # --- END MOCK DATA ---
    
    return midpoints


def calculate_spectral_depth_vectors(midpoints_380, midpoints_780):
    """
    Calculates the 3D hypotenuse vector between the closest 380nm and 780nm midpoints.
    """
    spectral_vectors = []
    
    for m380_x, m380_y, m380_z in midpoints_380:
        min_distance_sq = float('inf')
        closest_m780 = None
        
        # 1. Find the closest match in the 780nm dataset (Euclidean distance in 2D)
        for m780_x, m780_y, m780_z in midpoints_780:
            distance_sq = (m780_x - m380_x)**2 + (m780_y - m380_y)**2
            if distance_sq < min_distance_sq:
                min_distance_sq = distance_sq
                closest_m780 = (m780_x, m780_y, m780_z)

        if closest_m780:
            m780_x, m780_y, m780_z = closest_m780
            
            # 2. Calculate the Hypotenuse Vector Components
            delta_x = m780_x - m380_x
            delta_y = m780_y - m380_y
            # delta_z is the known simulated offset
            delta_z = m780_z - m380_z 

            # Hypotenuse Length (The magnitude of the spectral parallax)
            hypotenuse_length = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
            
            # Append the features used for ML
            spectral_vectors.append({
                'frame_id': 'MOCK_001', # Would be extracted from filename
                'midpoint_380_x': m380_x,
                'midpoint_380_y': m380_y,
                'parallax_delta_x': delta_x,
                'parallax_delta_y': delta_y,
                'parallax_delta_z': delta_z, # Should always be -500mm
                'parallax_hypotenuse_length': hypotenuse_length
            })
            
    return spectral_vectors

# --- EXECUTION ---
if __name__ == '__main__':
    os.makedirs(SPECTRAL_DXF_OUTPUT, exist_ok=True)
    
    all_vectors = []
    
    # Process one example frame set (assuming a file named '001.png' exists)
    FRAME_ID = '001' 
    
    # 1. Extract 380nm Data (Camera 1 / Detailed)
    path_380 = os.path.join(SPECTRAL_FRAMES_FOLDER, f'380nm_{FRAME_ID}.png')
    midpoints_380 = extract_midpoints_and_export_dxf(path_380, SPECTRAL_DXF_OUTPUT, EPSILON_380NM, z_offset=0)
    
    # 2. Extract 780nm Data (Camera 2 / Simplified & Shifted)
    path_780 = os.path.join(SPECTRAL_FRAMES_FOLDER, f'780nm_{FRAME_ID}.png')
    midpoints_780 = extract_midpoints_and_export_dxf(path_780, SPECTRAL_DXF_OUTPUT, EPSILON_780NM, z_offset=-SIMULATED_Z_OFFSET_MM)
    
    # 3. Calculate the Hypotenuse Vector
    vectors = calculate_spectral_depth_vectors(midpoints_380, midpoints_780)
    all_vectors.extend(vectors)
    
    # 4. Save Final Features
    df_output = pd.DataFrame(all_vectors)
    df_output.to_csv(FINAL_MIDPOINT_CSV, mode='a', header=not os.path.exists(FINAL_MIDPOINT_CSV), index=False)
    print(f"\n✅ Spectral depth vectors saved to: {FINAL_MIDPOINT_CSV}")