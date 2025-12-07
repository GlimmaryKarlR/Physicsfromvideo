import cv2
import ezdxf
import os
import glob
from tqdm import tqdm 
import numpy as np

# --- Configuration (UPDATE THESE) ---
# NEW: Directory containing all video files
VIDEO_INPUT_DIR = r"C:\Users\KarlRoesch\physVLA\bulkvidin" 
OUTPUT_DIR = r"C:\Users\KarlRoesch\physVLA\bulkdxfout"
FRAME_SKIP_INTERVAL = 5
CANNY_THRESHOLDS = (5, 10) 
POLY_DP_EPSILON_FACTOR = 0.05 
# ------------------------------------------

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_single_video(video_path):
    """Core logic to process one video file and output DXF files."""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return 0 # Return 0 DXF files processed

    video_filename_base = os.path.splitext(os.path.basename(video_path))[0]
    frame_count = 0
    dxf_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nProcessing video: {video_filename_base} ({total_frames} frames total)")
    
    with tqdm(total=total_frames, desc=f"Frames to DXF for {video_filename_base}") as pbar:
        while cap.isOpened():
            ret, image = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            pbar.update(1)

            if frame_count % FRAME_SKIP_INTERVAL == 0:
                
                # --- CV2 Image Processing ---
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)

                edges = cv2.Canny(blurred, CANNY_THRESHOLDS[0], CANNY_THRESHOLDS[1])
                
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
                
                # --- EZDXF Conversion ---
                doc = ezdxf.new("R2000")
                msp = doc.modelspace()
                
                for contour in contours:
                    # 1. Approximate the contour 
                    epsilon = POLY_DP_EPSILON_FACTOR * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)

                    points = [tuple(p[0]) for p in approx] 
                    
                    if len(points) >= 2:
                        num_vertices = len(points)
                        
                        for i in range(num_vertices):
                            p1 = points[i]
                            p2 = points[(i + 1) % num_vertices]
                            
                            # Add a simple LINE entity for each segment
                            msp.add_line(p1, p2) 

                # Save the DXF file. Filename includes video base name and frame number.
                output_filename = f"{video_filename_base}_frame_{frame_count:05d}.dxf"
                output_file_path = os.path.join(OUTPUT_DIR, output_filename)
                doc.saveas(output_file_path)
                dxf_count += 1

    cap.release()
    return dxf_count

def process_multiple_videos():
    """Main function to find all videos and loop through them."""
    
    # Use glob to find all files matching common video extensions
    video_extensions = ['*.mp4', '*.avi', '*.mov']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(VIDEO_INPUT_DIR, ext)))
    
    if not video_files:
        print(f"Error: No video files found in {VIDEO_INPUT_DIR}")
        return

    print(f"Found {len(video_files)} videos to process.")
    
    total_dxf_files = 0
    
    for video_file in video_files:
        dxf_count = process_single_video(video_file)
        total_dxf_files += dxf_count

    print(f"\n🎉 Batch processing complete.")
    print(f"Total DXF files generated across all videos: {total_dxf_files}")

if __name__ == "__main__":
    process_multiple_videos()