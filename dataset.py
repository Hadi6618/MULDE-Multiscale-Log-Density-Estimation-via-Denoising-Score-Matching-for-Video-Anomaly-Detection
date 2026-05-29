import os
import glob
import numpy as np
import torch

def get_dataset(seed=None, train_dir=None, test_dir=None, test_labels_dir=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    print(f"Loading training features from {train_dir}...")
    train_files = sorted(glob.glob(os.path.join(train_dir, "*.npz")))
    train_files = [f for f in train_files if "smoke_test" not in os.path.basename(f)]
    
    data_train_list = []
    for f in train_files:
        data = np.load(f)['features']
        data_train_list.append(data)
    
    if len(data_train_list) == 0:
        raise ValueError(f"No training features found in {train_dir}")
    
    data_train = np.vstack(data_train_list)
    labels_train = np.zeros(len(data_train), dtype=np.uint8)
    print(f"Loaded training data: {data_train.shape}")

    print(f"Loading testing features from {test_dir}...")
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.npz")))
    test_files = [f for f in test_files if "smoke_test" not in os.path.basename(f)]
    
    data_test_list = []
    labels_test_list = []
    
    for f in test_files:
        basename = os.path.basename(f)
        video_name = os.path.splitext(basename)[0]  # e.g., '01_0014'
        
        # Load features
        feat = np.load(f)['features']
        
        # Load labels
        label_path = os.path.join(test_labels_dir, f"{video_name}.npy")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Missing ground truth label for {video_name} at {label_path}")
            
        label = np.load(label_path)
        
        # Ensure sizes match exactly
        if feat.shape[0] != label.shape[0]:
            print(f"Warning: Frame count mismatch for {video_name}: feat={feat.shape[0]}, label={label.shape[0]}. Truncating to minimum.")
            min_len = min(feat.shape[0], label.shape[0])
            feat = feat[:min_len]
            label = label[:min_len]
            
        data_test_list.append(feat)
        labels_test_list.append(label)

    if len(data_test_list) == 0:
        raise ValueError(f"No testing features found in {test_dir}")

    data_test = np.vstack(data_test_list)
    labels_test = np.concatenate(labels_test_list).astype(np.uint8)
    
    print(f"Loaded testing data: {data_test.shape}")
    print(f"Loaded testing labels: {labels_test.shape} (Anomalies: {labels_test.sum()})")

    id_to_type = {
        0: "normal",
        1: "anomaly"
    }

    return data_train, labels_train, data_test, labels_test, id_to_type


def create_meshgrid_from_data(data, n_points=100, meshgrid_offset=1):
    x_min, x_max = data[:, 0].min() - meshgrid_offset, data[:, 0].max() + meshgrid_offset
    y_min, y_max = data[:, 1].min() - meshgrid_offset, data[:, 1].max() + meshgrid_offset
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_points), np.linspace(y_min, y_max, n_points))
    return xx, yy