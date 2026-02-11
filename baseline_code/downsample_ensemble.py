#!/usr/bin/env python3
"""
Downsample polygons with too many points
Keep polygons reasonable (max 25 points)
"""

import json
import numpy as np


def downsample_polygon(points, max_points=25):
    """Downsample polygon to max_points"""
    if len(points) <= max_points:
        return points
    
    # Sample evenly
    points_array = np.array(points)
    indices = np.linspace(0, len(points) - 1, max_points).astype(int)
    downsampled = points_array[indices]
    
    return downsampled.tolist()


def process_ensemble(input_json, output_json, max_points=25):
    """Process ensemble and downsample large polygons"""
    
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    stats = {'total': 0, 'downsampled': 0, 'original_avg': 0, 'new_avg': 0}
    total_orig_points = 0
    total_new_points = 0
    
    for img_name, img_data in data['images'].items():
        for word_id, word_data in img_data['words'].items():
            stats['total'] += 1
            points = word_data['points']
            total_orig_points += len(points)
            
            if len(points) > max_points:
                stats['downsampled'] += 1
                new_points = downsample_polygon(points, max_points)
                # Round to integers
                new_points = [[int(round(x)), int(round(y))] for x, y in new_points]
                word_data['points'] = new_points
                total_new_points += len(new_points)
            else:
                # Just round coordinates
                word_data['points'] = [[int(round(x)), int(round(y))] for x, y in points]
                total_new_points += len(points)
    
    stats['original_avg'] = total_orig_points / stats['total']
    stats['new_avg'] = total_new_points / stats['total']
    
    # Save
    with open(output_json, 'w') as f:
        json.dump(data, f)
    
    print(f"✅ Downsampling complete!")
    print(f"   Total boxes: {stats['total']:,}")
    print(f"   Downsampled: {stats['downsampled']:,} ({stats['downsampled']/stats['total']*100:.1f}%)")
    print(f"   Original avg points/box: {stats['original_avg']:.1f}")
    print(f"   New avg points/box: {stats['new_avg']:.1f}")
    print(f"   Output: {output_json}")


if __name__ == '__main__':
    process_ensemble(
        'outputs/submissions/ensemble_3models.json',
        'outputs/submissions/ensemble_downsampled.json',
        max_points=25
    )
