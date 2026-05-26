import json
import sys

# Reconfigure stdout to utf-8 just in case
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

path = r"c:\Projects\Graduate Project\MULDE-Multiscale-Log-Density-Estimation-via-Denoising-Score-Matching-for-Video-Anomaly-Detection\ShanghaiTech_Hiera_L_Feature_Extraction.ipynb"

with open(path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

print(f"Total cells: {len(notebook['cells'])}")
cells_to_inspect = [9, 13, 15]

for idx in cells_to_inspect:
    if idx < len(notebook['cells']):
        cell = notebook['cells'][idx]
        print(f"\n==================================================")
        print(f"CELL {idx} ({cell['cell_type']})")
        print(f"==================================================")
        source = "".join(cell.get('source', []))
        print(source)
