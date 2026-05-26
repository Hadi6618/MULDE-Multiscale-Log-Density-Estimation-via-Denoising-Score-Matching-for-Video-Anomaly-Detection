import json

path = r"c:\Projects\Graduate Project\MULDE-Multiscale-Log-Density-Estimation-via-Denoising-Score-Matching-for-Video-Anomaly-Detection\ShanghaiTech_Hiera_L_Feature_Extraction.ipynb"

with open(path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

cell = notebook['cells'][9]
source = "".join(cell['source'])
# Print the save_npz_atomic function definition
lines = source.split('\n')
for idx, line in enumerate(lines):
    if 'def save_npz_atomic' in line or idx >= len(lines) - 20:
        print(f"{idx+1}: {line}")
