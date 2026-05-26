import os
import gc
import cv2
import json
import datetime
import tqdm
import shutil
import numpy as np
import torch

# =============================================================================
# Pre-caching functions: decode & preprocess every frame exactly ONCE.
# This eliminates the 16x redundant decoding that caused the 7-minute bottleneck.
# Memory: a 764-frame video cached at [764, 3, 224, 224] float32 ≈ 460 MB.
# =============================================================================

def preprocess_all_frames_train(vr, num_frames, target_size=(224, 224), chunk_size=128):
    """Pre-decode and preprocess ALL frames from a training video.
    Uses chunked decord loading to limit peak memory.
    Returns numpy array [num_frames, 3, 224, 224] (float32, Hiera-normalized).
    """
    mean = np.array([0.45, 0.45, 0.45], dtype=np.float32).reshape(1, 3, 1, 1)
    std  = np.array([0.225, 0.225, 0.255], dtype=np.float32).reshape(1, 3, 1, 1)

    all_frames = np.empty((num_frames, 3, target_size[0], target_size[1]), dtype=np.float32)

    for start in range(0, num_frames, chunk_size):
        end = min(start + chunk_size, num_frames)
        indices = list(range(start, end))
        frames_np = vr.get_batch(indices).asnumpy()  # [chunk, H, W, C] RGB uint8

        for j, img in enumerate(frames_np):
            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            img_float = img_resized.astype(np.float32) / 255.0
            all_frames[start + j] = img_float.transpose(2, 0, 1)  # HWC -> CHW

        del frames_np

    # Vectorized Hiera normalization across all frames at once
    all_frames = (all_frames - mean) / std
    return all_frames  # [num_frames, 3, 224, 224]


def preprocess_all_frames_test(frame_paths, target_size=(224, 224)):
    """Pre-load and preprocess ALL JPG frames from a test folder.
    Returns numpy array [num_frames, 3, 224, 224] (float32, Hiera-normalized).
    """
    num_frames = len(frame_paths)
    mean = np.array([0.45, 0.45, 0.45], dtype=np.float32).reshape(1, 3, 1, 1)
    std  = np.array([0.225, 0.225, 0.255], dtype=np.float32).reshape(1, 3, 1, 1)

    all_frames = np.empty((num_frames, 3, target_size[0], target_size[1]), dtype=np.float32)

    for j, path in enumerate(frame_paths):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to read image at {path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)
        img_float = img_resized.astype(np.float32) / 255.0
        all_frames[j] = img_float.transpose(2, 0, 1)  # HWC -> CHW

    # Vectorized Hiera normalization
    all_frames = (all_frames - mean) / std
    return all_frames  # [num_frames, 3, 224, 224]


# =============================================================================
# Main extraction function (optimized with frame pre-caching, dynamic OOM recovery, and FP16 autocast)
# =============================================================================

def extract_features_for_video(video_source, is_train, out_npz_path, model, device, batch_size=8):
    """
    Optimized feature extraction with pre-cached preprocessed frames.
    Each frame is decoded, resized, and normalized exactly ONCE.
    Clips are assembled via fast numpy index slicing (zero re-decoding).

    Includes dynamic CUDA OOM recovery: if the GPU runs out of memory,
    it automatically halves the batch size and retries.

    Uses torch.cuda.amp.autocast to leverage T4 Tensor Cores for massive speedups.
    Returns (out_npz_path, num_frames, fps)
    """
    if video_source == 'mock':
        # Synthetic mock extraction (unchanged)
        num_frames = 45
        fps = 24.0
        video_id = "mock_video_01"
        split = "train" if is_train else "test"
        source_path = "mock_source"

        print(f"Running mock feature extraction for {video_id} ({split})...")
        features = np.random.randn(num_frames, 1152).astype(np.float32)
        frame_indices = np.arange(num_frames, dtype=np.int64)
        clip_indices = np.zeros((num_frames, 16), dtype=np.int64)
        for i in range(num_frames):
            clip_indices[i] = generate_clip_indices(i, num_frames)
    else:
        # Real extraction
        split = "train" if is_train else "test"
        if is_train:
            video_id = os.path.splitext(os.path.basename(video_source))[0]
            vr, num_frames, fps = load_train_video(video_source)
        else:
            video_id = os.path.basename(video_source)
            frame_paths, num_frames, fps = load_test_frames(video_source)

        source_path = video_source
        print(f"Extracting features for {video_id} ({split})... Total frames: {num_frames}")

        # ---- PHASE 1: Pre-cache all preprocessed frames (each frame decoded ONCE) ----
        print(f"  Phase 1/2: Pre-caching {num_frames} preprocessed frames...")
        if is_train:
            cached_frames = preprocess_all_frames_train(vr, num_frames)
            del vr  # Release VideoReader to free memory
        else:
            cached_frames = preprocess_all_frames_test(frame_paths)

        gc.collect()
        mem_mb = cached_frames.nbytes / (1024 * 1024)
        print(f"  Frame cache ready: {cached_frames.shape} — {mem_mb:.1f} MB")

        # ---- PHASE 2: Assemble clips via slicing + batched GPU inference (with OOM recovery) ----
        print(f"  Phase 2/2: Running batched Hiera-L inference...")
        frame_indices = np.arange(num_frames, dtype=np.int64)
        clip_indices = np.zeros((num_frames, 16), dtype=np.int64)

        # Pre-compute all clip indices
        for i in range(num_frames):
            clip_indices[i] = generate_clip_indices(i, num_frames)

        current_batch_size = batch_size
        success = False
        features_list = []

        while not success and current_batch_size >= 1:
            try:
                features_list = []
                # Batched inference loop
                for batch_start in tqdm.tqdm(range(0, num_frames, current_batch_size), desc=f"Batches (size={current_batch_size})"):
                    batch_end = min(batch_start + current_batch_size, num_frames)
                    batch_clips = []

                    for i in range(batch_start, batch_end):
                        # Fast numpy slicing from the pre-cached array
                        clip_frames = cached_frames[clip_indices[i]]   # [16, 3, 224, 224]
                        # Permute to Hiera input format: [C, T, H, W]
                        clip_tensor = torch.from_numpy(clip_frames.copy()).permute(1, 0, 2, 3)
                        batch_clips.append(clip_tensor)

                    # Stack to [B, C, T, H, W] and move to GPU
                    stacked = torch.stack(batch_clips, dim=0).to(device)
                    with torch.no_grad():
                        # Enable mixed precision (autocast) on CUDA to leverage T4 Tensor Cores
                        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                            feats = model(stacked)  # [B, 1152]
                        # Ensure features are converted back to FP32 before saving
                        features_list.append(feats.float().cpu().numpy())

                    del stacked, batch_clips

                features = np.concatenate(features_list, axis=0)  # [num_frames, 1152]
                success = True
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n[WARNING] CUDA Out of Memory with batch_size={current_batch_size}. Retrying with batch_size={current_batch_size // 2}...")
                    current_batch_size //= 2
                    # Clear variables to recover memory
                    if 'stacked' in locals(): del stacked
                    if 'batch_clips' in locals(): del batch_clips
                    if 'features_list' in locals(): del features_list
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if current_batch_size < 1:
                        raise e
                else:
                    raise e

        # Free the cache to reclaim RAM
        del cached_frames
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Package metadata as a compact JSON string
    metadata = {
        "model": MODEL_NAME,
        "checkpoint": CHECKPOINT_NAME,
        "hiera_commit": HIERA_COMMIT,
        "extraction_date": datetime.datetime.now().isoformat(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "temporal_stride": TEMPORAL_STRIDE,
        "clip_len": NUM_FRAMES_PER_CLIP,
        "preprocessing": {
            "resize": f"{IMAGE_SIZE}x{IMAGE_SIZE}",
            "mean": [0.45, 0.45, 0.45],
            "std": [0.225, 0.225, 0.255]
        }
    }

    data_to_save = {
        "features": features.astype(np.float32),
        "frame_indices": frame_indices.astype(np.int64),
        "clip_indices": clip_indices.astype(np.int64),
        "video_id": video_id,
        "split": split,
        "source_path": source_path,
        "num_frames": int(num_frames),
        "fps": float(fps),
        "metadata_json": json.dumps(metadata, indent=2)
    }

    save_npz_atomic(out_npz_path, data_to_save)
    print(f"✓ Saved: {out_npz_path}")
    return out_npz_path, int(num_frames), float(fps)

# Execute Smoke Test
print("--- Launching Smoke Test ---")
# Use local fast SSD files for smoke test validation to completely bypass Google Drive FUSE latency
local_smoke_train_out = "/content/smoke_test_train_local.npz"
local_smoke_test_out = "/content/smoke_test_test_local.npz"

smoke_train_out = os.path.join(TRAIN_FEATURE_DIR, "smoke_test_video.npz")
smoke_test_out = os.path.join(TEST_FEATURE_DIR, "smoke_test_video.npz")

if len(train_videos) > 0 and len(test_folders) > 0:
    # Use actual video/folders
    print("Real dataset files discovered. Running real smoke test...")
    real_train_video = train_videos[0]
    real_test_folder = test_folders[0]
    extract_features_for_video(real_train_video, is_train=True, out_npz_path=local_smoke_train_out, model=model, device=device)
    extract_features_for_video(real_test_folder, is_train=False, out_npz_path=local_smoke_test_out, model=model, device=device)
else:
    # Dry run mock smoke test
    print("No raw dataset files detected at paths. Executing mock smoke test...")
    extract_features_for_video("mock", is_train=True, out_npz_path=local_smoke_train_out, model=model, device=device)
    extract_features_for_video("mock", is_train=False, out_npz_path=local_smoke_test_out, model=model, device=device)

# Verify reloading and statistics on LOCAL SSD paths (100% reliable)
for path in [local_smoke_train_out, local_smoke_test_out]:
    assert os.path.exists(path), f"Smoke Test Failure: Output file {path} was not created."
    loaded = np.load(path)
    print(f"\nVerifying loaded file: {os.path.basename(path)}")
    print(f" - features shape: {loaded['features'].shape}")
    print(f" - num_frames: {loaded['num_frames']}")
    print(f" - video_id: {loaded['video_id']}")

    # Check NaN/Inf
    assert np.isfinite(loaded['features']).all(), "Smoke Test Failure: Features contain non-finite values (NaN/Inf)."
    assert loaded['features'].shape[1] == 1152, f"Smoke Test Failure: Expected feature size 1152, got {loaded['features'].shape[1]}"
    print(" - NaN/Inf check: Passed (all values finite)")
    print(" - Feature dimensions check: Passed")

    # Print sample metadata
    meta = json.loads(str(loaded['metadata_json']))
    print(f" - Metadata loaded: model={meta['model']}, date={meta['extraction_date']}")

# After successful verification, copy the smoke test files to their final Google Drive destinations
print("\nCopying verified smoke test artifacts to Google Drive...")
shutil.copy2(local_smoke_train_out, smoke_train_out)
shutil.copy2(local_smoke_test_out, smoke_test_out)
print(f"✓ Saved to Drive: {smoke_train_out}")
print(f"✓ Saved to Drive: {smoke_test_out}")

print("\n✓ Smoke test executed and validated successfully!")