import json

path = r"c:\Projects\Graduate Project\MULDE-Multiscale-Log-Density-Estimation-via-Denoising-Score-Matching-for-Video-Anomaly-Detection\ShanghaiTech_Hiera_L_Feature_Extraction.ipynb"

with open(path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for idx, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        if 'save_npz_atomic' in source:
            print(f"Found 'save_npz_atomic' in cell {idx}")
        if 'savez_compressed' in source:
            print(f"Found 'savez_compressed' in cell {idx}")
        if 'np.save' in source:
            print(f"Found 'np.save' in cell {idx}")
