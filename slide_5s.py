
import os
import glob
import hashlib
import numpy as np
import random
from joblib import Parallel, delayed
from tqdm import tqdm
import time


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEARCH_ROOT_DIR = os.path.join(BASE_DIR, "data_samples","5_data_20s")
SAVE_ROOT_DIR = os.path.join(BASE_DIR, "data_samples", "4_train_5s","EQ")


SAMPLING_RATE = 100
WIN_LEN_PTS = 500  

AUGMENT_TIMES = 10

PROBS = {'A': 0.5, 'B': 0.4, 'C': 0.1}

N_JOBS = -1

def get_valid_start_range(total_len, p_idx, policy):
    if policy == 'A':   
        desired_p_min, desired_p_max = 200, 300
    elif policy == 'B':
        desired_p_min, desired_p_max = 400, 480
    elif policy == 'C':
        return 0, max(0, p_idx - WIN_LEN_PTS)
    else:
        return None

    s_min = p_idx - desired_p_max
    s_max = p_idx - desired_p_min
    
    valid_s_min = max(0, s_min)
    valid_s_max = min(total_len - WIN_LEN_PTS, s_max)
    
    if valid_s_min > valid_s_max:
        return None
        
    return valid_s_min, valid_s_max

def process_source_file(npy_path):
    try:

        fname = os.path.basename(npy_path)
        base_name = fname.replace('.npy', '')

        md5_hash = hashlib.md5(base_name.encode('utf-8')).hexdigest()
        seed_val = int(md5_hash, 16) % (2**32)
        
        random.seed(seed_val)
        np.random.seed(seed_val)
        
        try:
            p_idx_20s = int(base_name.split('_pidx_')[-1])
        except:
            return f"Skip: Filename parse error {fname}"
        
            
        os.makedirs(SAVE_ROOT_DIR, exist_ok=True)
        
        data_20s = np.load(npy_path)
        total_len = len(data_20s)
        
        for i in range(AUGMENT_TIMES):
            policy = random.choices(list(PROBS.keys()), weights=list(PROBS.values()))[0]
            valid_range = get_valid_start_range(total_len, p_idx_20s, policy)

            if valid_range is None and policy == 'B':
                policy = 'A'
                valid_range = get_valid_start_range(total_len, p_idx_20s, 'A')
            
            if valid_range is None:
                continue 
                
            start_min, start_max = valid_range
            slice_start = random.randint(start_min, start_max)
            slice_end = slice_start + WIN_LEN_PTS
            slice_data = data_20s[slice_start : slice_end]
            
            if policy == 'C':
                new_pidx = -1 
            else:
                new_pidx = p_idx_20s - slice_start
            
            new_fname = f"{base_name}_Policy{policy}_pidx_{new_pidx}.npy"
            
            save_path = os.path.join(SAVE_ROOT_DIR, new_fname)
            np.save(save_path, slice_data.astype(np.float32))
            
        return "Success"

    except Exception as e:
        return f"Error: {str(e)} | File: {npy_path}"

if __name__ == "__main__":
    t0 = time.time()
    
    search_pattern = os.path.join(SEARCH_ROOT_DIR, "**", "*.npy")
    
    all_files = glob.glob(search_pattern, recursive=True)
    
    if not all_files:
        exit()
        
    
    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(process_source_file)(f) 
        for f in tqdm(all_files, desc="Processing")
    )
    
    success = results.count("Success")
    errors = [r for r in results if r != "Success" and not r.startswith("Skip")]
    
    print(f"Success: {success}")
    print(f"Error: {len(errors)}")
    
    if errors:
        with open("cut_5s_errors.log", "w") as f:
            f.write("\n".join(errors))
        print("Save Error_log as cut_5s_errors.log")