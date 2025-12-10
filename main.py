# C:\Users\KarlRoesch\physVLA\robobuild\main.py

import sys
import os
from pathlib import Path
import time
import shutil

# --- PACKAGE IMPORTS ---
import cv2 # Required for reading video properties
import frame_to_dxf_converter
import read_dxf_and_generate_csv
import kinematicid
import local_nn_training
import prediction_system
import video_to_frames # New module for frame extraction

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 

CONFIG = {
    # Directory to store raw CSVs and the intermediate DXF/Frame folders
    "RAW_CSV_DIR": os.path.join(BASE_DIR, "..", "data", "bulkcsvout"), 
    
    "TRAINING_DIR": os.path.join(BASE_DIR, "..", "data", "training_sequences"),
    "MODEL_DIR": os.path.join(BASE_DIR, "..", "models"), 
    "ACTION_CSV": os.path.join(BASE_DIR, "..", "action_commands", "robot_action.csv"),
    "MASTER_TRAINING_CSV": os.path.join(BASE_DIR, "..", "data", "training_sequences", "combined_kinematic_sequences.csv"),
}

# Ensure all necessary output directories exist
for key, path in CONFIG.items():
    if key not in ["MASTER_TRAINING_CSV", "ACTION_CSV"]: 
        Path(path).mkdir(parents=True, exist_ok=True)
Path(CONFIG["ACTION_CSV"]).parent.mkdir(parents=True, exist_ok=True)


# --- UX and ART ELEMENTS ---

def main_menu_art():
    """Prints the main menu ASCII art."""
    print("   ____            ___                                 ")
    print("  / /\ \           | |(_)                                ")
    print(" / /  \ \      __ _| |_ _ ___ __  _ __ __    ____ ___ __  __ ")
    print("< <    > >    / _  | | |  _   _ \|  _  _  \ / _  | __|| | | |")
    print(" \ \  / /    | (_| | | | | | | | | | | | | | (_| | |  | |_| |")
    print("  \_\/_/      \___ |_|_|_| |_| |_|_| |_| |_|\____|_|  \__   |")
    print("               __/ |                                   __/  |")
    print("              |____/                                  |____/ ")
    print("\n--- GlimmarianEmbeddedSystem.0.0.3 - Robotics Physics Engine ---")
    print("-----------------------------------------------------------")
    print("Type Learn to process a video and train physics models. LEARN a new task.")
    print("Type Do to process video and generate robot action commands. DO a task it knows how to do.")
    print("Type 'exit' to quit.")

# --- UTILITY & FILE INPUT FUNCTION ---

def get_video_path() -> str | None:
    """Prompts the user to enter the path of the video file to process."""
    print("\n🎥 **Video Upload Mode**")
    
    while True:
        video_path = input("Please enter the full path to the video file: ").strip().strip('"')
        if video_path.lower() == 'exit':
            return None
        
        path_obj = Path(video_path)
        
        if not path_obj.exists() or not path_obj.is_file():
            print(f"Error: File not found at '{video_path}'. Please check the path.")
        else:
            return video_path

# --- CORE LOGIC FUNCTIONS ---

def run_vision_and_geometry(video_path: str) -> str:
    """
    Runs video-to-frames (targeting 10 frames), bulk DXF conversion, and raw CSV generation.
    """
    print(f"\n[Processor] Running Vision Pipeline on {os.path.basename(video_path)}...")
    
    # 1. Define Output Paths and Directories
    file_name = Path(video_path).stem 
    temp_frames_dir = Path(CONFIG["RAW_CSV_DIR"]) / f"temp_frames_{file_name}" 
    dxf_bulk_dir = Path(CONFIG["RAW_CSV_DIR"]) / f"temp_dxf_bulk_{file_name}"
    raw_csv_path = Path(CONFIG["RAW_CSV_DIR"]) / f"raw_{file_name}.csv"
    
    # --- Frame Calculation ---
    TARGET_FRAMES = 10
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if total_frames <= TARGET_FRAMES:
            # If video is short, capture every frame
            n_frame_interval = 1
        else:
            # Calculate interval to get roughly TARGET_FRAMES
            n_frame_interval = max(1, total_frames // TARGET_FRAMES)
            
        print(f"  🎬 Total Video Frames: {total_frames}")
        print(f"  🎯 Target Frames: {TARGET_FRAMES}. Calculated Interval (n_frame): {n_frame_interval}")

    except Exception as e:
        print(f"  ❌ ERROR reading video properties for frame calculation: {e}")
        return ""
    
    # --- PHASE A: Video to Frames Extraction ---
    print(f"  -> Phase A: Video to Image Frames Extraction.")
    try:
        # Pass the calculated n_frame_interval
        video_to_frames.run_module_logic(video_path, str(temp_frames_dir), n_frame=n_frame_interval)
    except Exception as e:
        print(f"  ❌ ERROR in Phase A (Frame Extraction): {e}")
        return ""
    
    # --- PHASE B: Frame Directory to Bulk DXF ---
    print(f"  -> Phase B: Image Frames to BULK DXF geometry.")
    try:
        # Ensure DXF output directory is clean
        if dxf_bulk_dir.exists():
            shutil.rmtree(dxf_bulk_dir)
        dxf_bulk_dir.mkdir(parents=True, exist_ok=True)
            
        # Call DXF converter, using the temporary frames folder as input
        frame_to_dxf_converter.run_module_logic(str(temp_frames_dir), str(dxf_bulk_dir))
    except Exception as e:
        print(f"  ❌ ERROR in Phase B (Bulk DXF conversion): {e}")
        # Cleanup intermediate files on failure
        if dxf_bulk_dir.exists(): shutil.rmtree(dxf_bulk_dir)
        if temp_frames_dir.exists(): shutil.rmtree(temp_frames_dir)
        return ""
    
    # --- PHASE C: DXF Aggregation and CSV Generation ---
    print(f"  -> Phase C: Parsing DXF directory and generating raw line CSV.")
    try:
        # Read the bulk DXF folder and output a single raw CSV
        read_dxf_and_generate_csv.run_module_logic(str(dxf_bulk_dir), str(raw_csv_path))
        
        # Cleanup ALL intermediate directories
        if dxf_bulk_dir.exists():
            shutil.rmtree(dxf_bulk_dir)
            print(f"  🧹 Cleaned up intermediate DXF directory: {dxf_bulk_dir.name}")
        if temp_frames_dir.exists():
            shutil.rmtree(temp_frames_dir)
            print(f"  🧹 Cleaned up intermediate Frames directory: {temp_frames_dir.name}")
            
    except Exception as e:
        print(f"  ❌ ERROR in Phase C (CSV generation): {e}")
        if dxf_bulk_dir.exists(): shutil.rmtree(dxf_bulk_dir)
        if temp_frames_dir.exists(): shutil.rmtree(temp_frames_dir)
        return ""

    print(f"  ✅ Raw CAD lines successfully converted to CSV at: {raw_csv_path.name}")
    return str(raw_csv_path)

# --- WORKFLOW MODE FUNCTIONS ---

def workflow_learn_unlearned_task(raw_csv_path: str):
    """
    LEARN MODE: Generates kinematic features, appends to master data, and retrains all models.
    """
    
    # 1. Generate Kinematic Features and Append (using mode='learn')
    print("\n[KinematicID] Generating Kinematic Features...")
    try:
        kinematicid.run_module_logic(raw_csv_path, CONFIG["MASTER_TRAINING_CSV"], mode='learn')
        print("  ✅ Kinematic features appended to MASTER training CSV.")
    except Exception as e:
        print(f"  ❌ ERROR during KinematicID processing: {e}")
        return
        
    # 2. Train/Update Models
    print("\n[NN Training] Retraining Keras Models and Joblib Scalers...")
    try:
        local_nn_training.run_module_logic(CONFIG["MASTER_TRAINING_CSV"], CONFIG["MODEL_DIR"])
        print("  ✅ Training and saving COMPLETE. Baseline skills updated.")
    except Exception as e:
        print(f"  ❌ ERROR during NN training: {e}")
        return

def workflow_do_known_task(raw_csv_path: str):
    """
    DO MODE: Generates features, uses trained models to predict actions, and outputs command CSV.
    """
    
    # 1. Generate Kinematic Features for Prediction (using mode='predict')
    temp_prediction_csv = os.path.join(CONFIG["RAW_CSV_DIR"], "temp_prediction_features.csv")
    print("\n[KinematicID] Generating Kinematic Features for Prediction...")
    try:
        kinematicid.run_module_logic(raw_csv_path, temp_prediction_csv, mode='predict') 
        print("  ✅ Prediction features generated.")
    except Exception as e:
        print(f"  ❌ ERROR during KinematicID processing: {e}")
        return
        
    # 2. Run Prediction System
    print("\n[Prediction] Generating Robot Action Commands...")
    try:
        prediction_system.run_module_logic(
            temp_prediction_csv, 
            CONFIG["MODEL_DIR"], 
            CONFIG["MASTER_TRAINING_CSV"], 
            CONFIG["ACTION_CSV"]
        )
        print(f"  ✅ Prediction COMPLETE. Action CSV saved to: {os.path.basename(CONFIG['ACTION_CSV'])}")
    except Exception as e:
        print(f"  ❌ ERROR during Prediction: {e}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_prediction_csv):
            os.remove(temp_prediction_csv)

# --- MAIN MENU LOGIC ---

def main_menu():
    """Presents the main menu options to the user and manages the workflow."""
    print("")
    main_menu_art()
    
    while True:
        choice = input("\nSelect Mode (learn/do/exit): ").strip().lower()

        if choice in ('learn', 'do'):
            start_time = time.time()
            
            # 1. Get video path from user
            uploaded_video_path = get_video_path()
            if uploaded_video_path is None: 
                continue 

            # 2. Run vision pipeline (Outputs raw CSV)
            raw_csv_path = run_vision_and_geometry(uploaded_video_path)
            if not raw_csv_path: continue 

            # 3. Execute the chosen workflow
            if choice == 'learn':
                workflow_learn_unlearned_task(raw_csv_path)
            elif choice == 'do':
                workflow_do_known_task(raw_csv_path)
                
            # 4. Clean up the final raw CSV generated from vision processing
            if os.path.exists(raw_csv_path):
                os.remove(raw_csv_path)
                print(f"  🧹 Cleaned up final raw CSV: {Path(raw_csv_path).name}")
            
            end_time = time.time()
            print(f"\n[SUMMARY] Total time elapsed for '{choice}' workflow: {end_time - start_time:.2f} seconds.")
            
        elif choice == 'exit':
            print("Exiting Robot Control System.")
            sys.exit(0)
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()