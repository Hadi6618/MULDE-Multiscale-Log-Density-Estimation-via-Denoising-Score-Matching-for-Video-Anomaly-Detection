# ShanghaiTech Hiera-L Feature Extraction Plan

## Summary
Build a Colab notebook pipeline that mounts Google Drive, reads ShanghaiTech training `.avi` videos and testing JPG-frame folders, extracts one centered Hiera-L feature per frame, and saves raw 1152-D NPZ feature artifacts ready for MULDE training/evaluation.

Author details to preserve: MULDE uses one-class normal training, standardizes video features with training-set statistics, and uses frame-centric Hiera-L features from 16-frame sequences, producing 1152-D vectors. Sources: [MULDE paper](https://ar5iv.labs.arxiv.org/html/2403.14497v1), [supplement](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Micorek_MULDE_Multiscale_Log-Density_CVPR_2024_supplemental.pdf), [MULDE repo](https://github.com/jakubmicorek/MULDE-Multiscale-Log-Density-Estimation-via-Denoising-Score-Matching-for-Video-Anomaly-Detection), [Hiera repo](https://github.com/facebookresearch/hiera).

## Key Implementation Choices
- Use Hiera commit `b12b842542ee5c757fcfec8c41f6b56fcbe89b65`, model `hiera_large_16x224`, checkpoint `mae_k400_ft_k400`.
- Load the Kinetics-400 fine-tuned model, then replace `model.head = torch.nn.Identity()` so the output is the pooled 1152-D representation before the classifier.
- For every target frame `i`, sample 16 source frames with temporal stride 4:
  `clip_indices = clamp(i - 30 + 4*k, 0, num_frames - 1)` for `k = 0..15`.
  This gives one feature per original frame and keeps frame-level labels/scores aligned 1:1.
- Decode training videos from `.avi`; load testing videos from naturally sorted `.jpg` files inside each test-video folder.
- Preprocess clips as RGB float32 in `[0,1]`, resize to `16 x 224 x 224`, normalize with Hiera video stats `mean=[0.45,0.45,0.45]`, `std=[0.225,0.225,0.255]`, and run tensors as `[B,C,T,H,W]`.
- Save raw unstandardized features. Compute and save training-set mean/std separately for MULDE standardization.

## Drive Output Interface
Use this Drive layout, with path constants editable at the top of the notebook:

```text
/content/drive/MyDrive/MULDE_ShanghaiTech/
  raw/ShanghaiTech/
    training/videos/*.avi
    testing/frames/<video_id>/*.jpg
  features/hiera_large_16x224_mae_k400_ft_k400_centered_s4/
    train/<video_id>.npz
    test/<video_id>.npz
    manifest_train.csv
    manifest_test.csv
    train_feature_stats.npz
    extraction_config.json
```

Each per-video `.npz` contains:
- `features`: float32 array `[num_frames, 1152]`
- `frame_indices`: int64 array `[num_frames]`
- `clip_indices`: int64 array `[num_frames, 16]`
- `video_id`, `split`, `source_path`, `num_frames`, `fps`
- `metadata_json`: model, checkpoint, preprocessing, commit hash, extraction date, and Colab/PyTorch versions

The manifests contain one row per video: `split`, `video_id`, `source_path`, `feature_path`, `num_frames`, `fps`, `status`.

## Notebook Flow
- Cell 1: mount Drive, define paths, model/config constants, and random seeds.
- Cell 2: install pinned dependencies: PyTorch-compatible `timm`, Hiera from the pinned GitHub commit, `decord`, `opencv-python-headless`, `pandas`, `tqdm`.
- Cell 3: scan dataset; assert training files are `.avi` and testing entries are JPG folders.
- Cell 4: define reusable decode/load, natural sort, clip-index generation, preprocessing, and atomic NPZ-save helpers.
- Cell 5: load Hiera-L, remove classifier head, validate output shape is `[batch,1152]`.
- Cell 6: smoke-test one train video and one test folder, checking frame count, finite values, and saved reload.
- Cell 7: run full extraction with resume support: skip valid existing NPZs, write temp files first, then rename.
- Cell 8: concatenate training features in streaming batches to compute `train_feature_stats.npz`.
- Cell 9: create a compact MULDE-ready index file listing all feature files and training stats.

## Test Plan
- Verify every output video has `features.shape[0] == original_num_frames` and `features.shape[1] == 1152`.
- Verify no NaN/Inf values and nonzero feature variance.
- Verify first/last frames use boundary clamping correctly.
- Verify train/test manifests match expected ShanghaiTech counts: 330 train videos and 107 test videos if the dataset copy is complete.
- Reload several NPZ files from Drive in a fresh Colab runtime and confirm the arrays and metadata are intact.

## Assumptions
- You chose centered per-frame alignment and NPZ output.
- The paper does not specify the exact spatial crop/resize policy; this plan uses full-frame resize to preserve surveillance-scene content while still matching Hiera’s `224x224` input.
- Stored features remain raw Hiera features; MULDE standardization uses `train_feature_stats.npz` during density-model training and testing.
