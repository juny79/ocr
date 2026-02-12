"""
Best Model Error Case Analysis
최고 성능 모델 (Fold3, H-Mean 98.63%) 기반 에러 케이스 분석
"""

import json
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def parse_polygon(poly_str: str) -> np.ndarray:
    """폴리곤 문자열을 numpy 배열로 변환"""
    coords = list(map(int, poly_str.split()))
    return np.array(coords).reshape(-1, 2)


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """두 폴리곤의 IoU 계산"""
    from shapely.geometry import Polygon
    
    try:
        poly1 = Polygon(box1)
        poly2 = Polygon(box2)
        
        if not poly1.is_valid or not poly2.is_valid:
            return 0.0
            
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        
        if union == 0:
            return 0.0
            
        return intersection / union
    except:
        return 0.0


def match_predictions_to_ground_truth(
    pred_boxes: List[np.ndarray], 
    gt_boxes: List[np.ndarray], 
    iou_threshold: float = 0.5
) -> Dict:
    """
    예측 박스와 GT 박스를 매칭
    """
    matched_pairs = []
    false_positives = []
    false_negatives = list(range(len(gt_boxes)))
    
    # 각 예측에 대해 가장 높은 IoU의 GT를 찾음
    for pred_idx, pred_box in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx in false_negatives:
            gt_box = gt_boxes[gt_idx]
            iou = calculate_iou(pred_box, gt_box)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            matched_pairs.append((pred_idx, best_gt_idx, best_iou))
            false_negatives.remove(best_gt_idx)
        else:
            false_positives.append((pred_idx, best_iou))
    
    return {
        'matched': matched_pairs,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


def load_validation_ground_truth(val_json_path: str) -> Dict:
    """Validation GT 데이터 로드"""
    with open(val_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    gt_data = {}
    for img_name, img_info in data['images'].items():
        boxes = []
        for word_id, word_info in img_info['words'].items():
            points = word_info['points']
            # 4개 점을 평평한 배열로 변환
            boxes.append(np.array(points))
        gt_data[img_name] = {
            'boxes': boxes,
            'img_w': img_info['img_w'],
            'img_h': img_info['img_h']
        }
    
    return gt_data


def load_predictions(pred_file: str) -> Dict:
    """제출 파일에서 예측 결과 로드"""
    predictions = {}
    
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',',1)
            if len(parts) != 2:
                continue
                
            img_name = parts[0]
            pred_str = parts[1]
            
            if not pred_str:
                predictions[img_name] = []
                continue
            
            boxes = []
            for poly_str in pred_str.split('|'):
                if poly_str.strip():
                    box = parse_polygon(poly_str.strip())
                    boxes.append(box)
            
            predictions[img_name] = boxes
    
    return predictions


def analyze_per_image_performance(gt_data: Dict, predictions: Dict, iou_threshold: float = 0.5) -> pd.DataFrame:
    """이미지별 성능 분석"""
    results = []
    
    for img_name in gt_data.keys():
        gt_boxes = gt_data[img_name]['boxes']
        pred_boxes = predictions.get(img_name, [])
        
        # 매칭 수행
        match_result = match_predictions_to_ground_truth(pred_boxes, gt_boxes, iou_threshold)
        
        # 메트릭 계산
        tp = len(match_result['matched'])
        fp = len(match_result['false_positives'])
        fn = len(match_result['false_negatives'])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'image_name': img_name,
            'gt_count': len(gt_boxes),
            'pred_count': len(pred_boxes),
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'img_w': gt_data[img_name]['img_w'],
            'img_h': gt_data[img_name]['img_h']
        })
    
    return pd.DataFrame(results)


def visualize_error_case(
    img_path: str, 
    gt_boxes: List[np.ndarray], 
    pred_boxes: List[np.ndarray],
    match_result: Dict,
    save_path: str
):
    """에러 케이스 시각화"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Image not found: {img_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # Ground Truth
    axes[0].imshow(img)
    axes[0].set_title(f'Ground Truth ({len(gt_boxes)} boxes)', fontsize=14, fontweight='bold')
    for box in gt_boxes:
        poly = patches.Polygon(box, linewidth=2, edgecolor='green', facecolor='none')
        axes[0].add_patch(poly)
    axes[0].axis('off')
    
    # Predictions
    axes[1].imshow(img)
    axes[1].set_title(f'Predictions ({len(pred_boxes)} boxes)', fontsize=14, fontweight='bold')
    for box in pred_boxes:
        poly = patches.Polygon(box, linewidth=2, edgecolor='blue', facecolor='none')
        axes[1].add_patch(poly)
    axes[1].axis('off')
    
    # Error Overlay
    axes[2].imshow(img)
    axes[2].set_title('Error Analysis', fontsize=14, fontweight='bold')
    
    # True Positives (녹색)
    for pred_idx, gt_idx, iou in match_result['matched']:
        poly = patches.Polygon(pred_boxes[pred_idx], linewidth=2, edgecolor='green', facecolor='none', alpha=0.7)
        axes[2].add_patch(poly)
    
    # False Positives (빨강)
    for pred_idx, iou in match_result['false_positives']:
        poly = patches.Polygon(pred_boxes[pred_idx], linewidth=3, edgecolor='red', facecolor='none')
        axes[2].add_patch(poly)
    
    # False Negatives (노랑)
    for gt_idx in match_result['false_negatives']:
        poly = patches.Polygon(gt_boxes[gt_idx], linewidth=3, edgecolor='yellow', facecolor='none')
        axes[2].add_patch(poly)
    
    axes[2].axis('off')
    
    # 범례
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label=f'True Positive ({len(match_result["matched"])})'),
        Line2D([0], [0], color='red', lw=2, label=f'False Positive ({len(match_result["false_positives"])})'),
        Line2D([0], [0], color='yellow', lw=2, label=f'False Negative ({len(match_result["false_negatives"])})')
    ]
    axes[2].legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    # 경로 설정
    VAL_JSON = '/data/ephemeral/home/data/datasets/jsons/val.json'
    IMG_DIR = '/data/ephemeral/home/data/datasets/images'
    OUTPUT_DIR = Path('/data/ephemeral/home/error_analysis')
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Validation predictions 찾기 (가장 최근 파일 사용)
    # 실제로는 best model을 사용한 validation prediction이 필요
    # 여기서는 validation set에 대해 모델을 실행해야 함
    
    print("=" * 80)
    print("Error Case Analysis for Best Model (Fold 3, H-Mean 98.63%)")
    print("=" * 80)
    print()
    
    # 1. Ground Truth 로드
    print("Loading validation ground truth...")
    gt_data = load_validation_ground_truth(VAL_JSON)
    print(f"  - Loaded {len(gt_data)} validation images")
    print()
    
    # 2. 예측 결과 로드 (validation 예측이 필요)
    # 주의: 실제로는 모델을 실행하거나, validation prediction 파일이 필요
    print("Note: To complete this analysis, we need to:")
    print("  1. Run the best model (Fold 3) on validation set")
    print("  2. Generate predictions in the same format as submission")
    print("  3. Compare with ground truth")
    print()
    
    # 3. 대안: test.py를 사용한 validation 평가
    print("Alternative approach:")
    print("  Run: python baseline_code/runners/test.py \\")
    print("       --config baseline_code/configs/test.yaml \\")
    print("       --checkpoint <best_fold3_checkpoint.pth> \\")
    print("       --data_dir data/datasets \\")
    print("       --output_dir error_analysis")
    print()
    
    # 샘플 분석 (만약 예측 파일이 있다면)
    # PRED_FILE = 'baseline_code/outputs/validation_predictions_fold3.csv'
    # if Path(PRED_FILE).exists():
    #     print(f"Loading predictions from {PRED_FILE}...")
    #     predictions = load_predictions(PRED_FILE)
    #     
    #     # 이미지별 성능 분석
    #     df_results = analyze_per_image_performance(gt_data, predictions)
    #     
    #     # 결과 저장
    #     df_results.to_csv(OUTPUT_DIR / 'per_image_performance.csv', index=False)
    #     
    #     # 상위 에러 케이스 추출
    #     df_errors = df_results[
    #         (df_results['f1_score'] < 0.95) | 
    #         (df_results['false_positives'] > 0) | 
    #         (df_results['false_negatives'] > 0)
    #     ].sort_values('f1_score')
    #     
    #     # Top 20 에러 케이스 시각화
    #     for idx, row in df_errors.head(20).iterrows():
    #         img_name = row['image_name']
    #         img_path = Path(IMG_DIR) / img_name
    #         
    #         if not img_path.exists():
    #             continue
    #         
    #         gt_boxes = gt_data[img_name]['boxes']
    #         pred_boxes = predictions.get(img_name, [])
    #         match_result = match_predictions_to_ground_truth(pred_boxes, gt_boxes)
    #         
    #         save_path = OUTPUT_DIR / f'error_{idx:03d}_{row["f1_score"]:.3f}_{img_name}'
    #         visualize_error_case(img_path, gt_boxes, pred_boxes, match_result, save_path)
    
    print("Analysis script created successfully!")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
