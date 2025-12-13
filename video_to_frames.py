# C:\Users\KarlRoesch\physVLA\robobuild\video_to_frames.py

import cv2
import os
import shutil
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple
from PIL import Image
import numpy as np

# --- 0. FILE PATH CONFIGURATION & COLOR SCIENCE ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# CIE 1931 data and supporting classes/functions 
# Wavelengths from 380nm to 780nm in 5nm steps.
cie_wavelengths = np.arange(380, 780.1, 5)
cie_colour_match_data = np.array([
    [0.0014,0.0000,0.0065], [0.0022,0.0001,0.0105], [0.0042,0.0001,0.0201],
    [0.0076,0.0002,0.0362], [0.0143,0.0004,0.0679], [0.0232,0.0006,0.1102],
    [0.0435,0.0012,0.2074], [0.0776,0.0022,0.3713], [0.1344,0.0040,0.6456],
    [0.2148,0.0073,1.0391], [0.2839,0.0116,1.3856], [0.3285,0.0168,1.6230],
    [0.3483,0.0230,1.7471], [0.3481,0.0298,1.7826], [0.3362,0.0380,1.7721],
    [0.3187,0.0480,1.7441], [0.2908,0.0600,1.6692], [0.2511,0.0739,1.5281],
    [0.1954,0.0910,1.2876], [0.1421,0.1126,1.0419], [0.0956,0.1390,0.8130],
    [0.0580,0.1693,0.6162], [0.0320,0.2080,0.4652], [0.0147,0.2586,0.3533],
    [0.0049,0.3230,0.2720], [0.0024,0.4073,0.2123], [0.0093,0.5030,0.1582],
    [0.0291,0.6082,0.1117], [0.0633,0.7100,0.0782], [0.1096,0.7932,0.0573],
    [0.1655,0.8620,0.0422], [0.2257,0.9149,0.0298], [0.2904,0.9540,0.0203],
    [0.3597,0.9803,0.0134], [0.4334,0.9950,0.0087], [0.5121,1.0000,0.0057],
    [0.5945,0.9950,0.0039], [0.6784,0.9786,0.0027], [0.7621,0.9520,0.0021],
    [0.8425,0.9154,0.0018], [0.9163,0.8700,0.0017], [0.9786,0.8163,0.0014],
    [1.0263,0.7570,0.0011], [1.0567,0.6949,0.0010], [1.0622,0.6310,0.0008],
    [1.0456,0.5668,0.0006], [1.0026,0.5030,0.0003], [0.9384,0.4412,0.0002],
    [0.8544,0.3810,0.0002], [0.7514,0.3210,0.0001], [0.6424,0.2650,0.0000],
    [0.5419,0.2170,0.0000], [0.4479,0.1750,0.0000], [0.3608,0.1382,0.0000],
    [0.2835,0.1070,0.0000], [0.2187,0.0816,0.0000], [0.1649,0.0610,0.0000],
    [0.1212,0.0446,0.0000], [0.0874,0.0320,0.0000], [0.0636,0.0232,0.0000],
    [0.0468,0.0170,0.0000], [0.0329,0.0119,0.0000], [0.0227,0.0082,0.0000],
    [0.0158,0.0057,0.0000], [0.0114,0.0041,0.0000], [0.0081,0.0029,0.0000],
    [0.0058,0.0021,0.0000], [0.0041,0.0015,0.0000], [0.0029,0.0010,0.0000],
    [0.0020,0.0007,0.0000], [0.0014,0.0005,0.0000], [0.0010,0.0004,0.0000],
    [0.0007,0.0002,0.0000], [0.0005,0.0002,0.0000], [0.0003,0.0001,0.0000],
    [0.0002,0.0001,0.0000], [0.0002,0.0001,0.0000], [0.0001,0.0000,0.0000],
    [0.0001,0.0000,0.0000], [0.0001,0.0000,0.0000], [0.0000,0.0000,0.0000]
])

def linear_interpolate(x, x_values, y_values):
    if x <= x_values[0]: return y_values[0]
    if x >= x_values[-1]: return y_values[-1]

    for i in range(len(x_values) - 1):
        if x_values[i] <= x <= x_values[i+1]:
            x0, y0 = x_values[i], y_values[i]
            x1, y1 = x_values[i+1], y_values[i+1]
            return y0 + (y1 - y0) * ((x - x0) / (x1 - x0))
    return y_values[-1]

class ColourSystem:
    def __init__(self, name, xRed, yRed, xGreen, yGreen, xBlue, yBlue, xWhite, yWhite, gamma):
        self.name = name
        self.xRed, self.yRed = xRed, yRed
        self.xGreen, self.yGreen = xGreen, yGreen
        self.xBlue, self.yBlue = xBlue, yBlue
        self.xWhite, self.yWhite = xWhite, yWhite
        self.gamma = gamma

SMPTEsystem = ColourSystem(
    "SMPTE", 0.630, 0.340, 0.310, 0.595, 0.155, 0.070, 0.3127, 0.3291, gamma=0
)

def xyz_to_rgb(cs, xc, yc, zc):
    xr, yr, zr = cs.xRed, cs.yRed, 1 - (cs.xRed + cs.yRed)
    xg, yg, zg = cs.xGreen, cs.yGreen, 1 - (cs.xGreen + cs.yGreen)
    xb, yb, zb = cs.xBlue, cs.yBlue, 1 - (cs.xBlue + cs.yBlue)

    xw, yw, zw = cs.xWhite, cs.yWhite, 1 - (cs.xWhite + cs.yWhite)

    rx = (yg * zb) - (yb * zg); ry = (xb * zg) - (xg * zb); rz = (xg * yb) - (xb * yg)
    gx = (yb * zr) - (yr * zb); gy = (xr * zb) - (xb * zr); gz = (xb * yr) - (xr * yb)
    bx = (yr * zg) - (yg * zr); by = (xg * zr) - (xr * zg); bz = (xr * yg) - (xg * yr)

    if yw == 0: raise ValueError("White point Y coordinate (yw) cannot be zero for scaling.")

    rw = ((rx * xw) + (ry * yw) + (rz * zw)) / yw
    gw = ((gx * xw) + (gy * yw) + (gz * zw)) / yw
    bw = ((bx * xw) + (by * yw) + (bz * zw)) / yw

    if rw != 0: rx /= rw; ry /= rw; rz /= rw
    if gw != 0: gx /= gw; gy /= gw; gz /= gw
    if bw != 0: bx /= bw; by /= bw; bz /= bw

    r = (rx * xc) + (ry * yc) + (rz * zc)
    g = (gx * xc) + (gy * yc) + (gz * zc)
    b = (bx * xc) + (by * yc) + (bz * zc)
    return r, g, b

def constrain_rgb(r, g, b):
    w = min(0, r, g, b)
    w = -w
    if w > 0:
        return r + w, g + w, b + w, True
    return r, g, b, False

def gamma_correct(val, gamma_type=None, gamma_value=None):
    if gamma_type == "Rec709":
        cc = 0.018
        if val < cc:
            return val * ((1.099 * pow(cc, 0.45)) - 0.099) / cc
        else:
            return (1.099 * pow(val, 0.45)) - 0.099
    elif gamma_value is not None and gamma_value > 0:
        return pow(val, 1.0 / gamma_value)
    return val

def gamma_correct_rgb(cs, r, g, b):
    gamma_type_for_func = "Rec709" if cs.gamma == 0 else None
    gamma_value_for_func = cs.gamma if cs.gamma != 0 else None
    r = gamma_correct(r, gamma_type_for_func, gamma_value_for_func)
    g = gamma_correct(g, gamma_type_for_func, gamma_value_for_func)
    b = gamma_correct(b, gamma_type_for_func, gamma_value_for_func)
    return r, g, b

def norm_rgb(r, g, b):
    greatest = max(r, g, b)
    if greatest > 0:
        r /= greatest; g /= greatest; b /= greatest
    return r, g, b

def get_wavelength_rgb(wavelength, cs):
    if not (380 <= wavelength <= 750): return (0, 0, 0)
    
    x_bar = linear_interpolate(wavelength, cie_wavelengths, cie_colour_match_data[:, 0])
    y_bar = linear_interpolate(wavelength, cie_wavelengths, cie_colour_match_data[:, 1])
    z_bar = linear_interpolate(wavelength, cie_wavelengths, cie_colour_match_data[:, 2])

    xyz_sum = x_bar + y_bar + z_bar
    if xyz_sum == 0: return (0, 0, 0)
    
    xc_norm = x_bar / xyz_sum
    yc_norm = y_bar / xyz_sum
    zc_norm = z_bar / xyz_sum

    r_linear, g_linear, b_linear = xyz_to_rgb(cs, xc_norm, yc_norm, zc_norm)
    r_constrained, g_constrained, b_constrained, _ = constrain_rgb(r_linear, g_linear, b_linear)
    r_gamma, g_gamma, b_gamma = gamma_correct_rgb(cs, r_constrained, g_constrained, b_constrained)
    r_final_norm, g_final_norm, b_final_norm = norm_rgb(r_gamma, g_gamma, b_gamma)
    
    return (int(r_final_norm * 255), int(g_final_norm * 255), int(b_final_norm * 255))


def apply_color_shift_to_frame(frame_data: np.ndarray, shift_rgb_color: Tuple[int, int, int], wavelength_label: str, output_path: Path, filename_prefix: str, blend_factor: float = 0.6):
    """
    Applies a specific RGB color tint to a frame and saves it as a JPG.
    """
    try:
        # Convert OpenCV's BGR format to PIL's RGB format for color manipulation
        frame_rgb = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        pixels = img.load()
        width, height = img.size
        target_r, target_g, target_b = shift_rgb_color

        for x in range(width):
            for y in range(height):
                r, g, b = pixels[x, y]

                # Blend the original pixel color with the calculated spectral color.
                new_r = int(r * (1 - blend_factor) + target_r * blend_factor)
                new_g = int(g * (1 - blend_factor) + target_g * blend_factor)
                new_b = int(b * (1 - blend_factor) + target_b * blend_factor)

                pixels[x, y] = (min(255, max(0, new_r)),
                                 min(255, max(0, new_g)),
                                 min(255, max(0, new_b)))

        # Output format: 380nm_000.jpg
        output_filename = output_path / f"{wavelength_label}_{filename_prefix}.jpg"
        img.save(output_filename)

    except Exception as e:
        raise Exception(f"Error processing frame for {wavelength_label}: {e}")


# --- 2. CORE VIDEO PROCESSING (FIXED) ---

def save_every_nth_frame_and_spectral_shift(video_path: str, base_output_dir: str, spectral_output_dir: str, n_frame: int = 1) -> int:
    """
    Captures every nth frame, saves the original, and applies spectral shifts to a second directory,
    using ONLY the padded frame index as the filename (e.g., 000.jpg).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    Error: Could not open video file: {os.path.basename(video_path)}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # CRITICAL: Calculate padding based on total frames for consistent filenames
    digit = len(str(total_frames))
    
    # Define the two wavelengths for spectral depth generation
    WAVELENGTHS_TO_PROCESS = [380, 780]
    cs = SMPTEsystem
    rgb_shifts = {w: get_wavelength_rgb(w, cs) for w in WAVELENGTHS_TO_PROCESS}

    frame_count = 0
    saved_frame_count = 0
    
    spectral_path_obj = Path(spectral_output_dir)
    
    print(f"    -> Extracting base frames to: {os.path.basename(base_output_dir)}/")
    print(f"    -> Applying spectral shifts to: {os.path.basename(spectral_output_dir)}/")

    progress_iterator = tqdm(total=total_frames, desc="    Processing frames", unit="fr", leave=False)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break # End of video stream

            if frame_count % n_frame == 0:
                # *** FIX APPLIED HERE: Use only the padded index as the base filename ***
                frame_idx_str = str(frame_count).zfill(digit)
                output_filename_base = frame_idx_str 
                
                # A. Save Original Base Frame: Format is now '000.jpg'
                original_path = os.path.join(base_output_dir, f"{output_filename_base}.jpg")
                cv2.imwrite(original_path, frame)
                
                # B. Apply and Save Spectral Shifts: Format is now '380nm_000.jpg'
                for wavelength, color_shift in rgb_shifts.items():
                    wavelength_label = f"{wavelength}nm"
                    apply_color_shift_to_frame(frame, color_shift, wavelength_label, spectral_path_obj, output_filename_base)

                saved_frame_count += 1

            frame_count += 1
            progress_iterator.update(1)
            
    finally:
        cap.release()
        progress_iterator.close()

    return saved_frame_count


# --- 3. PUBLIC ENTRY POINT (UNCHANGED) ---

def run_module_logic(video_path: str, base_output_dir: str, n_frame: int = 1):
    """
    The unified function called by main.py to start frame and spectral extraction.
    """
    print("  -> Initializing frame and spectral depth extraction.")
    
    # We must derive the SPECTRAL output path based on the structure defined in main.py's CONFIG.
    BASE_PATH_LOCAL = os.path.dirname(os.path.abspath(__file__))
    SPECTRAL_OUTPUT_DIR = os.path.join(BASE_PATH_LOCAL, '..', 'data', 'temp_spectral_frames')


    # Clean up and recreate BOTH output directories
    base_output_path_obj = Path(base_output_dir)
    if base_output_path_obj.exists():
        shutil.rmtree(base_output_path_obj)
    base_output_path_obj.mkdir(parents=True, exist_ok=True)
    
    spectral_output_path_obj = Path(SPECTRAL_OUTPUT_DIR)
    if spectral_output_path_obj.exists():
        shutil.rmtree(spectral_output_path_obj)
    spectral_output_path_obj.mkdir(parents=True, exist_ok=True)


    saved_count = save_every_nth_frame_and_spectral_shift(video_path, base_output_dir, SPECTRAL_OUTPUT_DIR, n_frame)
    
    if saved_count == 0:
        raise ValueError("Video frame extraction failed or resulted in zero frames.")
        
    print(f"  ✅ Successfully saved {saved_count} original frames and {saved_count * 2} spectral frames.")
    print(f"     Original frames saved to: {base_output_dir}")
    print(f"     Spectral frames saved to: {SPECTRAL_OUTPUT_DIR}")
    return saved_count

# --- Standalone Execution Block (Optional) ---
if __name__ == "__main__":
    print("This module is designed to be run via main.py.")