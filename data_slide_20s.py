import os
import numpy as np
import glob
import re

SAMPLING_RATE = 100
WINDOW_LEN_SEC = 20
WINDOW_LEN_PTS = int(WINDOW_LEN_SEC * SAMPLING_RATE)    
TARGET_PRE_P_PTS = int(10 * SAMPLING_RATE)          

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCE_DIR = os.path.join(BASE_DIR, "data_samples", "1_continuous data")
OUTPUT_DIR = os.path.join(BASE_DIR, "data_samples", "5_data_20s")

def process_all_files():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    txt_files = glob.glob(os.path.join(SOURCE_DIR, "*_pidx_*.txt"))
    
    for fpath in txt_files:
        fname = os.path.basename(fpath)
        pidx = int(re.search(r'_pidx_(\d+)', fname).group(1))

        data = np.loadtxt(fpath)
        if data.ndim == 1: data = data.reshape(-1, 1)
        
        start_idx = max(0, pidx - TARGET_PRE_P_PTS)
        end_idx = start_idx + WINDOW_LEN_PTS
        
        slice_data = data[start_idx : end_idx]
        
        if len(slice_data) < WINDOW_LEN_PTS:
            pad_width = WINDOW_LEN_PTS - len(slice_data)
            padding = np.zeros((pad_width, data.shape[1]))
            slice_data = np.vstack([slice_data, padding])
        
        new_p_idx = pidx - start_idx
        
        base_name = re.sub(r'_pidx_\d+', '', fname.replace('.txt', ''))
        out_name = f"{base_name}_pidx_{new_p_idx}.npy"
        np.save(os.path.join(OUTPUT_DIR, out_name), slice_data.astype(np.float32))
        

if __name__ == "__main__":
    process_all_files()
    print("Done")