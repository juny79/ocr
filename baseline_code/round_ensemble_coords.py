#!/usr/bin/env python3
"""
Round ensemble coordinates to integers and regenerate CSV
"""

import json
import numpy as np
from pathlib import Path

# Load ensemble JSON
print("Loading ensemble JSON...")
with open('outputs/submissions/ensemble_3models.json', 'r') as f:
    data = json.load(f)

# Round all coordinates to integers
print("Rounding coordinates to integers...")
for img_name, img_data in data['images'].items():
    for word_id, word_data in img_data['words'].items():
        points = word_data['points']
        # Round to nearest integer
        rounded_points = [[int(round(x)), int(round(y))] for x, y in points]
        word_data['points'] = rounded_points

# Save rounded JSON
output_json = Path('outputs/submissions/ensemble_3models_rounded.json')
print(f"Saving rounded JSON to {output_json}...")
with open(output_json, 'w') as f:
    json.dump(data, f, indent=2)

print(f"✅ Rounded JSON saved: {output_json}")

# Convert to CSV
print("Converting to CSV...")
import pandas as pd

rows = []
for filename, content in data['images'].items():
    polygons = []
    for idx, word in content['words'].items():
        points = word['points']
        polygon = ' '.join([' '.join(map(str, point)) for point in points])
        polygons.append(polygon)
    
    polygons_str = '|'.join(polygons)
    rows.append([filename, polygons_str])

df = pd.DataFrame(rows, columns=['filename', 'polygons'])
output_csv = 'outputs/submissions/ensemble_3models_rounded.csv'
df.to_csv(output_csv, index=False)

print(f"✅ Rounded CSV saved: {output_csv}")

# Compare file sizes
import os
original_size = os.path.getsize('outputs/submissions/ensemble_3models.csv') / (1024**2)
rounded_size = os.path.getsize(output_csv) / (1024**2)

print(f"\n📊 파일 크기 비교:")
print(f"  원본 (float): {original_size:.1f} MB")
print(f"  반올림 (int): {rounded_size:.1f} MB")
print(f"  감소율: {(1 - rounded_size/original_size)*100:.1f}%")
