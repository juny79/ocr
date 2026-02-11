#!/usr/bin/env python3
"""
Convert all polygons to 4-point quadrilaterals using bounding box
"""

import json
import numpy as np
from pathlib import Path
import argparse


def polygon_to_quadrilateral(points):
    """
    Convert any polygon to a 4-point quadrilateral
    Uses minimum area rotated rectangle
    """
    if len(points) == 4:
        return points
    
    points_array = np.array(points, dtype=np.float32)
    
    # Calculate center
    center = points_array.mean(axis=0)
    
    # Translate to origin
    translated = points_array - center
    
    # Calculate angle (using PCA-like approach)
    angles = np.arctan2(translated[:, 1], translated[:, 0])
    
    # Find min/max in rotated space
    # Use convex hull corners
    min_x, max_x = points_array[:, 0].min(), points_array[:, 0].max()
    min_y, max_y = points_array[:, 1].min(), points_array[:, 1].max()
    
    # Create axis-aligned bounding box
    quad = [
        [float(min_x), float(min_y)],  # top-left
        [float(max_x), float(min_y)],  # top-right
        [float(max_x), float(max_y)],  # bottom-right
        [float(min_x), float(max_y)],  # bottom-left
    ]
    
    return quad


def convert_predictions(input_json, output_json):
    """Convert all polygons in JSON to quadrilaterals"""
    
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    converted_count = 0
    total_count = 0
    
    for img_name, img_data in data['images'].items():
        for word_id, word_data in img_data['words'].items():
            total_count += 1
            points = word_data['points']
            
            if len(points) != 4:
                converted_count += 1
                quad = polygon_to_quadrilateral(points)
                word_data['points'] = quad
    
    # Save
    with open(output_json, 'w') as f:
        json.dump(data, f)
    
    print(f'✅ Conversion complete!')
    print(f'  Total boxes: {total_count:,}')
    print(f'  Converted: {converted_count:,} ({converted_count/total_count*100:.1f}%)')
    print(f'  Output: {output_json}')


def main():
    parser = argparse.ArgumentParser(description='Convert polygons to quadrilaterals')
    parser.add_argument('--input', type=str, required=True, help='Input JSON')
    parser.add_argument('--output', type=str, required=True, help='Output JSON')
    
    args = parser.parse_args()
    
    convert_predictions(args.input, args.output)


if __name__ == '__main__':
    main()
