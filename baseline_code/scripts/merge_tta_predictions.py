#!/usr/bin/env python
"""TTA Postprocessing - 두 예측 결과를 병합"""
import json
import sys
from pathlib import Path
from shapely.geometry import Polygon


def calculate_iou(box1_points, box2_points):
    """두 박스의 IoU 계산"""
    try:
        poly1 = Polygon(box1_points)
        poly2 = Polygon(box2_points)
        
        if not poly1.is_valid or not poly2.is_valid:
            return 0.0
        
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        
        return intersection / union if union > 0 else 0.0
    except:
        return 0.0


def merge_predictions(pred1_path, pred2_path, output_path, iou_threshold=0.3):
    """두 예측 결과를 IoU 기반으로 병합"""
    
    with open(pred1_path, 'r') as f:
        pred1 = json.load(f)
    
    with open(pred2_path, 'r') as f:
        pred2 = json.load(f)
    
    merged = {'images': {}}
    
    for img_name in pred1['images'].keys():
        boxes1 = list(pred1['images'][img_name]['words'].values())
        boxes2 = list(pred2['images'][img_name]['words'].values()) if img_name in pred2['images'] else []
        
        # 모든 박스 수집
        all_boxes = boxes1 + boxes2
        
        # NMS-like merging
        merged_boxes = []
        used = set()
        
        for i, box1 in enumerate(all_boxes):
            if i in used:
                continue
            
            # 겹치는 박스들 찾기
            group = [box1]
            points1 = box1['points']
            
            for j in range(i + 1, len(all_boxes)):
                if j in used:
                    continue
                
                points2 = all_boxes[j]['points']
                iou = calculate_iou(points1, points2)
                
                if iou > iou_threshold:
                    group.append(all_boxes[j])
                    used.add(j)
            
            # 평균 좌표로 병합
            if len(group) > 1:
                avg_points = []
                for pt_idx in range(4):
                    avg_x = sum([b['points'][pt_idx][0] for b in group]) / len(group)
                    avg_y = sum([b['points'][pt_idx][1] for b in group]) / len(group)
                    avg_points.append([avg_x, avg_y])
                merged_boxes.append({'points': avg_points})
            else:
                merged_boxes.append(box1)
        
        merged['images'][img_name] = {
            'words': {str(i): box for i, box in enumerate(merged_boxes)}
        }
    
    # 저장
    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)
    
    print(f"Merged predictions saved to: {output_path}")
    print(f"Total images: {len(merged['images'])}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python merge_tta_predictions.py <pred1.json> <pred2.json> <output.json> [iou_threshold]")
        sys.exit(1)
    
    pred1 = sys.argv[1]
    pred2 = sys.argv[2]
    output = sys.argv[3]
    iou_threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
    
    merge_predictions(pred1, pred2, output, iou_threshold)
