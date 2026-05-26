import json
import os

path = r"c:\Projects\Graduate Project\MULDE-Multiscale-Log-Density-Estimation-via-Denoising-Score-Matching-for-Video-Anomaly-Detection\ShanghaiTech_Hiera_L_Feature_Extraction.ipynb"

with open(path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

os.makedirs('scratch', exist_ok=True)

for idx in [9, 13, 17]:
    cell = notebook['cells'][idx]
    source = "".join(cell.get('source', []))
    with open(f"scratch/cell_{idx}.py", "w", encoding="utf-8") as out:
        out.write(source)
print("Extracted cells 9, 13, and 17 to scratch/cell_<idx>.py")
