#!/usr/bin/env python
"""2-Fold Ensemble Prediction with Voting"""
import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from shapely.geometry import Polygon
from tqdm import tqdm


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


def ensemble_predictions(pred_paths, output_path, voting_threshold=2, iou_threshold=0.5):
    """
    여러 Fold 예측을 앙상블
    
    Args:
        pred_paths: 예측 JSON 파일 경로 리스트
        output_path: 출력 JSON 경로
        voting_threshold: 최소 투표 수 (2-Fold면 1 또는 2)
        iou_threshold: 같은 박스로 간주할 IoU 임계값
    """
    
    # 모든 예측 로드
    predictions = []
    for path in pred_paths:
        with open(path, 'r') as f:
            predictions.append(json.load(f))
        print(f"Loaded: {path}")
    
    n_folds = len(predictions)
    print(f"Ensemble: {n_folds} folds, voting_threshold={voting_threshold}, iou_threshold={iou_threshold}")
    
    # 이미지별 앙상블
    ensemble_result = {'images': {}}
    
    for img_name in tqdm(predictions[0]['images'].keys(), desc="Ensemble"):
        # 모든 Fold의 박스 수집
        all_boxes = []
        for pred in predictions:
            if img_name in pred['images']:
                boxes = list(pred['images'][img_name]['words'].values())
                all_boxes.extend([(box, 1) for box in boxes])
        
        # Voting: 겹치는 박스들을 그룹화
        final_boxes = []
        used = set()
        
        for i, (box1, _) in enumerate(all_boxes):
            if i in used:
                continue
            
            # 이 박스와 겹치는 박스들 찾기
            group = [box1]
            votes = 1
            points1 = box1['points']
            
            for j in range(i + 1, len(all_boxes)):
                if j in used:
                    continue
                
                box2, _ = all_boxes[j]
                points2 = box2['points']
                iou = calculate_iou(points1, points2)
                
                if iou > iou_threshold:
                    group.append(box2)
                    votes += 1
                    used.add(j)
            
            # 투표 수가 임계값 이상이면 채택
            if votes >= voting_threshold:
                # 평균 좌표로 병합
                avg_points = []
                for pt_idx in range(4):
                    avg_x = sum([b['points'][pt_idx][0] for b in group]) / len(group)
                    avg_y = sum([b['points'][pt_idx][1] for b in group]) / len(group)
                    avg_points.append([avg_x, avg_y])
                
                final_boxes.append({
                    'points': avg_points,
                    'votes': votes
                })
        
        # 결과 저장
        ensemble_result['images'][img_name] = {
            'words': {str(i): {'points': box['points']} for i, box in enumerate(final_boxes)}
        }
    
    # 저장
    with open(output_path, 'w') as f:
        json.dump(ensemble_result, f, indent=2)
    
    print(f"\nEnsemble complete!")
    print(f"Output: {output_path}")
    print(f"Total images: {len(ensemble_result['images'])}")
    
    # 통계
    total_boxes = sum([len(img['words']) for img in ensemble_result['images'].values()])
    avg_boxes = total_boxes / len(ensemble_result['images'])
    print(f"Total boxes: {total_boxes}")
    print(f"Average boxes per image: {avg_boxes:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ensemble K-Fold predictions')
    parser.add_argument('--fold0', required=True, help='Fold 0 prediction JSON')
    parser.add_argument('--fold1', required=True, help='Fold 1 prediction JSON')
    parser.add_argument('--output', default='outputs/ensemble_2fold.json', help='Output JSON')
    parser.add_argument('--voting', type=int, default=1, help='Voting threshold (1 or 2)')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold')
    
    args = parser.parse_args()
    
    ensemble_predictions(
        pred_paths=[args.fold0, args.fold1],
        output_path=args.output,
        voting_threshold=args.voting,
        iou_threshold=args.iou
    )
