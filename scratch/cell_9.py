import re
import os
import glob
import tempfile
import shutil
import numpy as np
import torch
import cv2
from decord import VideoReader, cpu

# 1. Natural sorting helper
def natural_keys(text):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

# 2. Decode/load training AVI video
def load_train_video(path):
    vr = VideoReader(path, ctx=cpu(0))
    num_frames = len(vr)
    fps = vr.get_avg_fps()
    return vr, num_frames, fps

# 3. Load testing JPG frame sequence
def load_test_frames(folder_path):
    frame_paths = sorted(glob.glob(os.path.join(folder_path, "*.jpg")), key=natural_keys)
    num_frames = len(frame_paths)
    fps = 24.0 # Standard fps for ShanghaiTech test sequences
    return frame_paths, num_frames, fps

# 4. Generate clip indices with temporal stride 4 centered around target frame i
def generate_clip_indices(i, num_frames):
    indices = []
    for k in range(16):
        idx = i - 30 + 4 * k
        idx = max(0, min(idx, num_frames - 1))
        indices.append(idx)
    return np.array(indices, dtype=np.int64)

# 5. Preprocess a single 16-frame spatiotemporal clip
def preprocess_clip(frames_or_paths, is_train, vr=None, target_size=(224, 224)):
    """
    frames_or_paths: For train (is_train=True), this is a list of frame indices to load via 'vr'.
                    For test (is_train=False), this is a list of absolute JPG file paths to load.
    is_train: bool
    vr: VideoReader instance (needed if is_train=True)
    """
    processed_frames = []
    
    if is_train:
        # Load from VideoReader
        assert vr is not None, "VideoReader instance must be provided for train split."
        # decord can fetch multiple frames as a batch, converting to numpy RGB
        frames_np = vr.get_batch(frames_or_paths).asnumpy() # [16, H, W, C]
        for img in frames_np:
            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            img_float = img_resized.astype(np.float32) / 255.0
            processed_frames.append(img_float)
    else:
        # Load from JPG paths
        for path in frames_or_paths:
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Failed to read image at {path}")
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)
            img_float = img_resized.astype(np.float32) / 255.0
            processed_frames.append(img_float)
            
    # Stack along time axis: [16, H, W, C]
    clip_np = np.stack(processed_frames, axis=0)
    
    # Convert to PyTorch Tensor: [16, H, W, C] -> [C, T, H, W]
    clip_tensor = torch.tensor(clip_np).permute(3, 0, 1, 2) # [3, 16, 224, 224]
    
    # Standardize with Hiera-L video stats: mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.255]
    mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.255]).view(3, 1, 1, 1)
    clip_tensor = (clip_tensor - mean) / std
    
    return clip_tensor # [3, 16, 224, 224]

# 6. Atomic save function for NPZ files (FUSE/Google Drive Safe)
def save_npz_atomic(file_path, data_dict):
    dir_name = os.path.dirname(file_path)
    os.makedirs(dir_name, exist_ok=True)
    
    # Create temp file on the LOCAL Colab SSD filesystem (guarantees fast, robust POSIX writes)
    fd, temp_path = tempfile.mkstemp(suffix=".tmp")
    try:
        os.close(fd)
        np.savez_compressed(temp_path, **data_dict)
        # Use shutil.copy2 to copy from local SSD to Google Drive (sequential FUSE-friendly write)
        shutil.copy2(temp_path, file_path)
        os.remove(temp_path)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e