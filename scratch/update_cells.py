import json
import os

path = r"c:\Projects\Graduate Project\MULDE-Multiscale-Log-Density-Estimation-via-Denoising-Score-Matching-for-Video-Anomaly-Detection\ShanghaiTech_Hiera_L_Feature_Extraction.ipynb"

if not os.path.exists(path):
    raise FileNotFoundError(f"Notebook file not found at: {path}")

with open(path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# 1. Update Cell 9
cell_9 = notebook['cells'][9]
cell_9_source = "".join(cell_9['source'])

# Replace suffix=".tmp" with suffix=".npz"
if 'suffix=".tmp"' in cell_9_source:
    cell_9_source = cell_9_source.replace('suffix=".tmp"', 'suffix=".npz"')
    print("Found and replaced suffix=\".tmp\" with suffix=\".npz\" in Cell 9")
else:
    print("WARNING: suffix=\".tmp\" not found in Cell 9!")

# Split back to lines with trailing newlines
cell_9['source'] = [line + '\n' for line in cell_9_source.split('\n')[:-1]] + [cell_9_source.split('\n')[-1]]


# 2. Update Cell 13
cell_13 = notebook['cells'][13]
cell_13_source = "".join(cell_13['source'])

# Replace torch.cuda.amp.autocast with torch.amp.autocast
if 'torch.cuda.amp.autocast(enabled=(device.type == \'cuda\'))' in cell_13_source:
    cell_13_source = cell_13_source.replace(
        "torch.cuda.amp.autocast(enabled=(device.type == 'cuda'))",
        "torch.amp.autocast('cuda', enabled=(device.type == 'cuda'))"
    )
    print("Updated autocast deprecation in Cell 13")
else:
    print("autocast pattern not found or already updated in Cell 13")

cell_13['source'] = [line + '\n' for line in cell_13_source.split('\n')[:-1]] + [cell_13_source.split('\n')[-1]]


# Save back to disk
with open(path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("Notebook successfully updated on disk!")
