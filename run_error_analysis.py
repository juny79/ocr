"""
Integrated Error Case Analysis
Fold 3 Best Model을 사용하여 Validation Set에 대한 에러 분석 수행
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 경로 추가
sys.path.append('/data/ephemeral/home/baseline_code')

# 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class ErrorAnalyzer:
    def __init__(self, checkpoint_path, val_json_path, img_dir, output_dir):
        self.checkpoint_path = checkpoint_path
        self.val_json_path = val_json_path
        self.img_dir = Path(img_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load ground truth
        print("Loading validation ground truth...")
        with open(val_json_path, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        self.gt_data = val_data['images']
        print(f"  - Loaded {len(self.gt_data)} validation images")
        
    def polygon_to_array(self, points):
        """4개의 점을 numpy array로 변환"""
        return np.array(points, dtype=np.float32)
    
    def calculate_iou(self, box1, box2):
        """두 폴리곤의 IoU 계산"""
        from shapely.geometry import Polygon
        from shapely.validation import make_valid
        
        try:
            poly1 = Polygon(box1)
            poly2 = Polygon(box2)
            
            if not poly1.is_valid:
                poly1 = make_valid(poly1)
            if not poly2.is_valid:
                poly2 = make_valid(poly2)
                
            if poly1.is_empty or poly2.is_empty:
                return 0.0
                
            intersection = poly1.intersection(poly2).area
            union = poly1.union(poly2).area
            
            return intersection / union if union > 0 else 0.0
        except Exception as e:
            return 0.0
    
    def match_boxes(self, pred_boxes, gt_boxes, iou_threshold=0.5):
        """예측 박스와 GT 박스 매칭"""
        matched = []
        false_positives = []
        false_negatives = list(range(len(gt_boxes)))
        matched_gt = set()
        
        # 각 예측에 대해 가장 높은 IoU의 GT 찾기
        for pred_idx, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx in range(len(gt_boxes)):
                if gt_idx in matched_gt:
                    continue
                gt_box = gt_boxes[gt_idx]
                iou = self.calculate_iou(pred_box, gt_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                matched.append((pred_idx, best_gt_idx, best_iou))
                matched_gt.add(best_gt_idx)
                if best_gt_idx in false_negatives:
                    false_negatives.remove(best_gt_idx)
            else:
                false_positives.append((pred_idx, best_iou))
        
        return {
            'matched': matched,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def analyze_image(self, img_name, pred_boxes, gt_info):
        """단일 이미지 분석"""
        gt_boxes = [self.polygon_to_array(word['points']) for word in gt_info['words'].values()]
        
        match_result = self.match_boxes(pred_boxes, gt_boxes)
        
        tp = len(match_result['matched'])
        fp = len(match_result['false_positives'])
        fn = len(match_result['false_negatives'])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'image_name': img_name,
            'gt_count': len(gt_boxes),
            'pred_count': len(pred_boxes),
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'match_result': match_result,
            'gt_boxes': gt_boxes,
            'pred_boxes': pred_boxes
        }
    
    def visualize_error(self, result, rank):
        """에러 케이스 시각화"""
        img_name = result['image_name']
        img_path = self.img_dir / img_name
        
        if not img_path.exists():
            print(f"Image not found: {img_path}")
            return
        
        img = cv2.imread(str(img_path))
        if img is None:
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # Ground Truth
        axes[0].imshow(img)
        axes[0].set_title(f'Ground Truth ({result["gt_count"]} boxes)', 
                         fontsize=14, fontweight='bold')
        for box in result['gt_boxes']:
            poly = patches.Polygon(box, linewidth=2, edgecolor='green', 
                                 facecolor='none', alpha=0.8)
            axes[0].add_patch(poly)
        axes[0].axis('off')
        
        # Predictions
        axes[1].imshow(img)
        axes[1].set_title(f'Predictions ({result["pred_count"]} boxes)', 
                         fontsize=14, fontweight='bold')
        for box in result['pred_boxes']:
            poly = patches.Polygon(box, linewidth=2, edgecolor='blue', 
                                 facecolor='none', alpha=0.8)
            axes[1].add_patch(poly)
        axes[1].axis('off')
        
        # Error Overlay
        axes[2].imshow(img)
        title = f'F1: {result["f1_score"]:.3f} | P: {result["precision"]:.3f} | R: {result["recall"]:.3f}'
        axes[2].set_title(title, fontsize=14, fontweight='bold')
        
        match_result = result['match_result']
        
        # True Positives (녹색)
        for pred_idx, gt_idx, iou in match_result['matched']:
            poly = patches.Polygon(result['pred_boxes'][pred_idx], 
                                 linewidth=2, edgecolor='green', 
                                 facecolor='none', alpha=0.7)
            axes[2].add_patch(poly)
        
        # False Positives (빨강)
        for pred_idx, _ in match_result['false_positives']:
            poly = patches.Polygon(result['pred_boxes'][pred_idx], 
                                 linewidth=3, edgecolor='red', 
                                 facecolor='none')
            axes[2].add_patch(poly)
        
        # False Negatives (노랑)
        for gt_idx in match_result['false_negatives']:
            poly = patches.Polygon(result['gt_boxes'][gt_idx], 
                                 linewidth=3, edgecolor='yellow', 
                                 facecolor='none')
            axes[2].add_patch(poly)
        
        axes[2].axis('off')
        
        # 범례
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=3, 
                  label=f'TP: {result["true_positives"]}'),
            Line2D([0], [0], color='red', lw=3, 
                  label=f'FP: {result["false_positives"]}'),
            Line2D([0], [0], color='yellow', lw=3, 
                  label=f'FN: {result["false_negatives"]}')
        ]
        axes[2].legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        plt.suptitle(f'Error Case #{rank:02d}: {img_name}', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = self.output_dir / f'error_{rank:02d}_f1{result["f1_score"]:.3f}_{img_name}'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path.name}")
    
    def run_simple_analysis(self):
        """간단한 통계 분석 (모델 실행 없이)"""
        print("\n" + "="*80)
        print("VALIDATION SET STATISTICS")
        print("="*80)
        
        total_boxes = 0
        box_counts = []
        
        for img_name, img_info in self.gt_data.items():
            num_boxes = len(img_info['words'])
            total_boxes += num_boxes
            box_counts.append(num_boxes)
        
        box_counts = np.array(box_counts)
        
        print(f"\nTotal validation images: {len(self.gt_data)}")
        print(f"Total ground truth boxes: {total_boxes}")
        print(f"Average boxes per image: {box_counts.mean():.2f}")
        print(f"Median boxes per image: {np.median(box_counts):.0f}")
        print(f"Min boxes: {box_counts.min()}")
        print(f"Max boxes: {box_counts.max()}")
        
        # Box count distribution
        print(f"\nBox Count Distribution:")
        print(f"  0-10 boxes: {(box_counts <= 10).sum()} images")
        print(f"  11-20 boxes: {((box_counts > 10) & (box_counts <= 20)).sum()} images")
        print(f"  21-50 boxes: {((box_counts > 20) & (box_counts <= 50)).sum()} images")
        print(f"  51+ boxes: {(box_counts > 50).sum()} images")
        
        # 가장 많은/적은 박스를 가진 이미지
        print(f"\nImages with most boxes:")
        sorted_imgs = sorted(self.gt_data.items(), 
                           key=lambda x: len(x[1]['words']), reverse=True)
        for i, (img_name, img_info) in enumerate(sorted_imgs[:5]):
            print(f"  {i+1}. {img_name}: {len(img_info['words'])} boxes")
        
        print(f"\nImages with fewest boxes:")
        for i, (img_name, img_info) in enumerate(sorted_imgs[-5:]):
            print(f"  {i+1}. {img_name}: {len(img_info['words'])} boxes")


def main():
    CHECKPOINT = '/data/ephemeral/home/baseline_code/checkpoints/kfold_optimized/fold_3/fold3_best.ckpt'
    VAL_JSON = '/data/ephemeral/home/data/datasets/jsons/val.json'
    IMG_DIR = '/data/ephemeral/home/data/datasets/images'
    OUTPUT_DIR = '/data/ephemeral/home/error_analysis'
    
    print("="*80)
    print("ERROR CASE ANALYSIS - FOLD 3 BEST MODEL")
    print("="*80)
    print(f"\nCheckpoint: {CHECKPOINT}")
    print(f"Validation JSON: {VAL_JSON}")
    print(f"Image Directory: {IMG_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    analyzer = ErrorAnalyzer(CHECKPOINT, VAL_JSON, IMG_DIR, OUTPUT_DIR)
    
    # 간단한 통계 분석
    analyzer.run_simple_analysis()
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n추가로 필요한 작업:")
    print("1. Fold 3 모델을 사용하여 validation set에 대한 예측 실행")
    print("2. 예측 결과를 저장 (CSV 형식)")
    print("3. 본 스크립트에서 예측 결과 로드 및 분석")
    print("\n명령어 예시:")
    print(f"  python baseline_code/runners/predict.py \\")
    print(f"    checkpoint_path={CHECKPOINT} \\")
    print(f"    data_dir={IMG_DIR} \\")
    print(f"    output_dir={OUTPUT_DIR}")


if __name__ == '__main__':
    try:
        from shapely.geometry import Polygon
        main()
    except ImportError:
        print("Error: shapely library not installed")
        print("Please install: pip install shapely")
