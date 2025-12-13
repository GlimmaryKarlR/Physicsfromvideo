# C:\Users\KarlRoesch\physVLA\robobuild\read_dxf_and_generate_csv.py

import os
import sys
import ezdxf
import csv
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any

# --- Helper Functions (UPDATED to include Handle) ---

def parse_dxf_file(dxf_path: Path) -> List[Dict[str, Any]]:
    """
    Reads a single DXF file and extracts geometric properties (start, end points, length) 
    and the CRITICAL unique entity HANDLE from LINE entities.
    """
    entities_data = []
    
    try:
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
    except Exception as e:
        print(f"        ⚠️ Error reading DXF file {dxf_path.name}: {e}. Skipping.")
        return []

    # Iterate over all entities (LINEs, POLYLINEs, etc.)
    for entity in msp:
        if entity.dxftype() == 'LINE':
            # Extract relevant data for kinematic analysis
            start = entity.dxf.start
            end = entity.dxf.end
            
            # Calculate length (Euclidean distance in 3D space)
            length = ((end[0] - start[0])**2 + (end[1] - start[1])**2 + (end[2] - start[2])**2)**0.5
            
            # --- CRITICAL FIX: Extract the DXF Entity Handle ---
            try:
                # The handle is the unique, persistent ID for the entity.
                handle = entity.dxf.handle
            except AttributeError:
                # Fallback if handle is somehow missing (e.g., in ancient DXF versions)
                handle = f"NO_HANDLE_{len(entities_data)}" 

            entities_data.append({
                # --- ADDED: The unique entity handle ---
                'handle': handle, 
                
                'entity_type': 'LINE',
                'start_x': start[0],
                'start_y': start[1],
                'start_z': start[2],
                'end_x': end[0],
                'end_y': end[1],
                'end_z': end[2],
                'length': length
            })
    return entities_data


# --- Core Processing Logic (UPDATED to include Handle fieldname) ---

def process_dxf_directory_to_csv(dxf_directory_path: Path, csv_output_path: Path) -> int:
    """
    Reads all DXF files in the input directory, aggregates geometric data, 
    and writes a single raw feature CSV file.
    """
    
    if not dxf_directory_path.exists() or not dxf_directory_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {dxf_directory_path}")

    dxf_files: List[Path] = sorted(list(dxf_directory_path.glob("*.dxf")))
    
    if not dxf_files:
        raise ValueError(f"DXF directory '{dxf_directory_path.name}' is empty or contains no recognized DXF files.")

    print(f"    -> Found {len(dxf_files)} DXF files to aggregate.")
    
    all_raw_data = []
    
    # --- CRITICAL FIX: Add 'handle' to the list of expected column headers ---
    fieldnames = [
        'frame_id', 'handle', 'entity_type', 
        'start_x', 'start_y', 'start_z', 
        'end_x', 'end_y', 'end_z', 
        'length'
    ]

    # Process each DXF file
    for dxf_path in tqdm(dxf_files, desc="    Parsing DXF files"):
        
        # 1. Extract the frame ID from the filename
        frame_id = -1 
        
        try:
            # Assume format: INDEX.dxf (e.g., '000.dxf')
            frame_id_str = dxf_path.stem 
            frame_id = int(frame_id_str)
            
        except Exception:
            # If the stem is not a valid integer, use the fallback -1.
            print(f"        ⚠️ Could not parse frame ID from {dxf_path.name} (Expected simple numeric name). Using integer -1.")
            frame_id = -1 
            
        # 2. Parse the DXF file
        entities_list = parse_dxf_file(dxf_path)
        
        # 3. Append frame data and index to the master list
        for entity in entities_list:
            # frame_id is now guaranteed to be an integer
            entity['frame_id'] = frame_id
            all_raw_data.append(entity)

    # --- Write the master CSV file ---
    try:
        with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_raw_data)
        
        print(f"    ✅ Successfully aggregated {len(all_raw_data)} geometric entities.")
        return len(all_raw_data)

    except Exception as e:
        print(f"    🛑 Error writing CSV file: {e}")
        raise


# --- Public Entry Point for main.py (UNCHANGED) ---

def run_module_logic(dxf_directory_path_str: str, csv_output_path_str: str):
    """
    The unified function called by main.py to generate the raw kinematic feature CSV.
    """
    print("    -> Starting DXF file aggregation...")
    
    dxf_directory_path = Path(dxf_directory_path_str)
    csv_output_path = Path(csv_output_path_str)
    
    try:
        entities_processed = process_dxf_directory_to_csv(dxf_directory_path, csv_output_path)
        
        if entities_processed == 0:
            raise ValueError("No geometric entities were successfully parsed from the DXF files.")
        
    except Exception as e:
        print(f"    🛑 An error occurred during CSV generation: {e}")
        raise # Re-raise to halt the workflow in main.py

    print("    -> DXF-to-CSV module finished execution.")