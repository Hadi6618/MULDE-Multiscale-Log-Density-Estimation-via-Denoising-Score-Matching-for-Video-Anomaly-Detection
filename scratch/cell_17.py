import os
import glob
import tqdm
import datetime
import numpy as np

def compute_train_statistics():
    print("--- Computing Global Training Set Feature Statistics ---")
    
    # 1. Discover all train feature files
    all_train_files = sorted(glob.glob(os.path.join(TRAIN_FEATURE_DIR, "*.npz")))
    
    # Filter out smoke test file so it doesn't double count or contaminate stats
    all_train_files = [f for f in all_train_files if "smoke_test" not in os.path.basename(f)]
    
    if len(all_train_files) == 0:
        print("No real training feature files found! Checking for smoke test file...")
        # If there are no other training files, fallback to smoke_test for dry-run continuity
        all_train_files = sorted(glob.glob(os.path.join(TRAIN_FEATURE_DIR, "smoke_test_video.npz")))
        if len(all_train_files) == 0:
            print("No training feature files found! Cannot calculate statistics.")
            return None
        
    print(f"Streaming through {len(all_train_files)} training feature files...")
    
    n_frames = 0
    sum_x = np.zeros(1152, dtype=np.float64)
    sum_x2 = np.zeros(1152, dtype=np.float64)
    
    for path in tqdm.tqdm(all_train_files, desc="Streaming Stats"):
        try:
            data = np.load(path)
            feats = data['features'].astype(np.float64) # [num_frames, 1152]
            
            sum_x += feats.sum(axis=0)
            sum_x2 += (feats ** 2).sum(axis=0)
            n_frames += feats.shape[0]
        except Exception as e:
            print(f"Skipping file {path} due to load error: {e}")
            
    if n_frames == 0:
        print("Total loaded training frames is 0. Cannot compute stats.")
        return None
        
    # Calculate global mean and standard deviation
    global_mean = sum_x / n_frames
    global_variance = (sum_x2 / n_frames) - (global_mean ** 2)
    # Clamp small negative variances from numerical errors to 0
    global_variance = np.clip(global_variance, 0.0, None)
    global_std = np.sqrt(global_variance)
    
    # Check for features with zero variance to avoid divide-by-zero during normalization
    zero_var_mask = global_std < 1e-8
    if zero_var_mask.any():
        print(f"Warning: Found {zero_var_mask.sum()} feature dimensions with near-zero variance. Fixing std to 1.0 for these features.")
        global_std[zero_var_mask] = 1.0
        
    print(f"\nSuccessfully accumulated statistics over {n_frames} frames.")
    print(f"Mean Vector - min: {global_mean.min():.5f}, max: {global_mean.max():.5f}, avg: {global_mean.mean():.5f}")
    print(f"Std Vector  - min: {global_std.min():.5f}, max: {global_std.max():.5f}, avg: {global_std.mean():.5f}")
    
    # Save statistics atomically
    stats_dict = {
        "mean": global_mean.astype(np.float32),
        "std": global_std.astype(np.float32),
        "num_frames": int(n_frames),
        "computed_at": datetime.datetime.now().isoformat()
    }
    
    save_npz_atomic(FEATURE_STATS_PATH, stats_dict)
    print(f"✓ Saved training feature statistics to: {FEATURE_STATS_PATH}")
    return FEATURE_STATS_PATH

stats_path = compute_train_statistics()