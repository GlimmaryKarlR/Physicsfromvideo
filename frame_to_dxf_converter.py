# C:\Users\KarlRoesch\physVLA\robobuild\frame_to_dxf_converter.py

import os
import sys
import shutil
from pathlib import Path
import cv2
import ezdxf
from tqdm import tqdm 
from typing import List

# --- Configuration Parameters (Using user-provided logic) ---
CANNY_THRESHOLDS = (5, 10) # Used for aggressive edge finding on low-contrast images
POLY_DP_EPSILON_FACTOR = 0.05 # High epsilon for simplifying contours significantly
# ------------------------------------------

# --- Helper Functions (UNCHANGED) ---

def initialize_dxf_doc(frame_num):
    """Creates a new DXF document for a single frame."""
    try:
        # Use a stable DXF version (R2000 specified in your logic)
        doc = ezdxf.new(dxfversion='R2000') 
        msp = doc.modelspace()
        # Create a layer specific to the frame number
        doc.layers.new(f'Frame_{frame_num}', dxfattribs={'color': 7})
        return doc, msp
    except Exception as e:
        print(f"    ⚠️ Error initializing DXF for frame {frame_num}: {e}")
        return None, None

def detect_and_add_lines(msp, image, frame_num):
    """
    Applies OpenCV Contour Detection and Polygon Approximation logic
    to the frame and adds the resulting LINE segments to the DXF model space (msp).
    
    Returns:
        bool: True if at least one LINE entity was added, False otherwise.
    """
    lines_added = 0
    
    if frame_num % 10 == 0:
        print(f"    -> Detecting contours in frame {frame_num}...")

    # --- CV2 Image Processing ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, CANNY_THRESHOLDS[0], CANNY_THRESHOLDS[1])
    
    # Find contours (RETR_EXTERNAL for outer contours, CHAIN_APPROX_SIMPLE for compression)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    
    # --- EZDXF Conversion ---
    for contour in contours:
        # 1. Approximate the contour
        epsilon = POLY_DP_EPSILON_FACTOR * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Flatten points structure: [[[x, y]], ...] -> [(x, y), ...]
        points = [tuple(p[0]) for p in approx] 
        
        # 2. Iterate through approximated points and create LINE entities
        if len(points) >= 2:
            num_vertices = len(points)
            
            for i in range(num_vertices):
                p1 = points[i]
                # p2 wraps back to p1 for closed contours/polygons
                p2 = points[(i + 1) % num_vertices] 
                
                # Add a simple LINE entity for each segment (assuming Z=0)
                p1_3d = (p1[0], p1[1], 0)
                p2_3d = (p2[0], p2[1], 0)
                
                msp.add_line(p1_3d, p2_3d, dxfattribs={'layer': f'Frame_{frame_num}'})
                lines_added += 1

    return lines_added > 0

# --- Core Processing Logic (Directory Reader - FIXED) ---

def process_image_dir_to_multiple_dxfs(image_dir: Path, output_dir: Path) -> int:
    """
    Core logic: Takes an image directory, generates one DXF file per image.
    """
    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the list of image files to process
    image_files: List[Path] = sorted(
        list(image_dir.glob("*.jpg")) + 
        list(image_dir.glob("*.jpeg")) + 
        list(image_dir.glob("*.png"))
    )
    
    if not image_files:
        raise ValueError(f"Image directory '{image_dir.name}' is empty or contains no recognized images (jpg/png).")

    # The stem derivation is no longer needed for the filename, but kept for context if you need it later.
    # video_stem = image_dir.name.replace('temp_frames_', '') 
    
    frames_with_geometry = 0
    
    # Determine padding based on the largest possible index.
    # Assumes the largest index is the highest number in the filenames.
    # We use a default of 5 digits if no files exist, although image_files check already handles this.
    digit = 5 
    if image_files:
        try:
            # Simple check to estimate max digits needed from a filename that contains only numbers
            max_index = max(int(f.stem.split('_')[-1].split('nm')[-1]) for f in image_files)
            digit = len(str(max_index))
        except:
            # Default to 5 digits if complex naming or parsing fails
            digit = 5 


    print(f"    -> Found {len(image_files)} image frames to process.")
    
    # Use tqdm for progress over the list of files
    for image_path in tqdm(image_files, desc="    Generating DXFs"):
        
        # 1. Extract frame number from the filename (FIXED LOGIC)
        frame_num = -1
        frame_id_str_raw = image_path.stem # e.g., '000' or '380nm_000'
        
        try:
            # Check if spectral prefix exists (e.g., '380nm_000' -> '000')
            if 'nm_' in frame_id_str_raw:
                # This correctly isolates the numeric index
                frame_id_str = frame_id_str_raw.split('_')[-1] 
            # Check if it's a simple numeric frame (e.g., '000')
            else:
                frame_id_str = frame_id_str_raw
            
            # Now safely convert the clean numeric string to an integer
            frame_num = int(frame_id_str)
            
        except ValueError:
            # This handles cases like 'cncturbine_frame_000' or malformed strings
            print(f"    ⚠️ Warning: Could not parse simple numeric ID from filename: {image_path.name}. Using integer -1.")
            frame_num = -1 


        # 2. Read the image frame 
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        # 3. Initialize DXF 
        doc, msp = initialize_dxf_doc(frame_num)
        if doc is None: 
            continue

        # 4. Detection and DXF update
        if detect_and_add_lines(msp, image, frame_num):
            
            # 5. Save DXF for this frame (FIXED Naming)
            # CRITICAL FIX: Only use the padded numeric index for the output DXF filename
            # The padding is now calculated based on the directory content for consistency.
            output_filename = f"{frame_num:0{digit}d}.dxf"
            
            dxf_filename = output_dir / output_filename
            doc.saveas(dxf_filename)
            frames_with_geometry += 1

    print(f"    ✅ Successfully generated {frames_with_geometry} DXF files from {len(image_files)} image frames scanned.")
    return frames_with_geometry


# --- Public Entry Point for main.py (UNCHANGED) ---

def run_module_logic(input_image_dir_str: str, output_dir_str: str):
    """
    The unified function called by main.py to start the bulk DXF conversion.
    """
    print("    -> Initializing bulk frame-to-DXF conversion (1 DXF per image)...")
    
    input_image_dir = Path(input_image_dir_str)
    output_dir = Path(output_dir_str)
    
    try:
        frames_processed = process_image_dir_to_multiple_dxfs(input_image_dir, output_dir)
        
        if frames_processed == 0:
            raise ValueError("No line geometry was detected or added across all image frames. (Check the Canny/Contour parameters in 'detect_and_add_lines'.)")
        
    except Exception as e:
        print(f"    🛑 An error occurred during DXF processing: {e}")
        # Clean up the output DXF directory on failure
        if output_dir.exists():
             shutil.rmtree(output_dir)
        raise # Re-raise to halt the workflow in main.py

    print("    -> Frame-to-DXF module finished execution.")