"""
Final Error Case Analysis
Fold 3 예측 결과를 이용한 Validation Set 에러 케이스 분석
"""

import json
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class ErrorCaseAnalyzer:
    def __init__(self, pred_path, gt_path, img_dir, output_dir):
        self.img_dir = Path(img_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load predictions and ground truth
        print("Loading predictions and ground truth...")
        with open(pred_path, 'r', encoding='utf-8') as f:
            self.predictions = json.load(f)['images']
        
        with open(gt_path, 'r', encoding='utf-8') as f:
            self.ground_truth = json.load(f)['images']
        
        print(f"  - Predictions: {len(self.predictions)} images")
        print(f"  - Ground Truth: {len(self.ground_truth)} images")
        
    def polygon_to_array(self, points):
        """Convert points to numpy array"""
        return np.array(points, dtype=np.float32)
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two polygons"""
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
        except:
            return 0.0
    
    def match_boxes(self, pred_boxes, gt_boxes, iou_threshold=0.5):
        """Match predicted boxes with ground truth boxes"""
        matched = []
        false_positives = []
        matched_gt = set()
        
        # For each prediction, find best matching GT
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
            else:
                false_positives.append((pred_idx, best_iou))
        
        # Find false negatives (GT boxes not matched)
        false_negatives = [i for i in range(len(gt_boxes)) if i not in matched_gt]
        
        return {
            'matched': matched,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def analyze_single_image(self, img_name):
        """Analyze single image"""
        # Get boxes
        pred_info = self.predictions.get(img_name, {'words': {}})
        gt_info = self.ground_truth.get(img_name, {'words': {}})
        
        pred_boxes = [self.polygon_to_array(w['points']) for w in pred_info['words'].values()]
        gt_boxes = [self.polygon_to_array(w['points']) for w in gt_info['words'].values()]
        
        # Match boxes
        match_result = self.match_boxes(pred_boxes, gt_boxes)
        
        # Calculate metrics
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
            'pred_boxes': pred_boxes,
            'avg_iou': np.mean([iou for _, _, iou in match_result['matched']]) if match_result['matched'] else 0.0
        }
    
    def analyze_all(self):
        """Analyze all images"""
        print("\nAnalyzing all images...")
        results = []
        
        common_images = set (self.predictions.keys()) & set(self.ground_truth.keys())
        print(f"  - Common images: {len(common_images)}")
        
        for img_name in tqdm(sorted(common_images)):
            result = self.analyze_single_image(img_name)
            results.append(result)
        
        return results
    
    def generate_report(self, results):
        """Generate analysis report"""
        print("\n" + "="*80)
        print("ERROR CASE ANALYSIS REPORT - FOLD 3 BEST MODEL")
        print("="*80)
        
        # Convert to DataFrame
        df = pd.DataFrame([{k: v for k, v in r.items() 
                           if k not in ['match_result', 'gt_boxes', 'pred_boxes']} 
                          for r in results])
        
        # Save detailed results
        csv_path = self.output_dir / 'per_image_performance.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved detailed results: {csv_path}")
        
        # Overall metrics
        total_tp = df['true_positives'].sum()
        total_fp = df['false_positives'].sum()
        total_fn = df['false_negatives'].sum()
        
        overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0
        
        print(f"\n{'OVERALL VALIDATION PERFORMANCE':^80}")
        print("-"*80)
        print(f"  Total Images:           {len(df):,}")
        print(f"  Total GT Boxes:         {df['gt_count'].sum():,}")
        print(f"  Total Pred Boxes:       {df['pred_count'].sum():,}")
        print(f"  True Positives:         {total_tp:,}")
        print(f"  False Positives:        {total_fp:,}")
        print(f"  False Negatives:        {total_fn:,}")
        print(f"\n  Precision:              {overall_p:.4f}  ({overall_p*100:.2f}%)")
        print(f"  Recall:                 {overall_r:.4f}  ({overall_r*100:.2f}%)")
        print(f"  F1-Score (H-Mean):      {overall_f1:.4f}  ({overall_f1*100:.2f}%)")
        print(f"  Average IoU:            {df['avg_iou'].mean():.4f}")
        
        # Error statistics
        perfect_images = df[df['f1_score'] == 1.0]
        error_images = df[df['f1_score'] < 1.0]
        
        print(f"\n{'ERROR STATISTICS':^80}")
        print("-"*80)
        print(f"  Perfect predictions:    {len(perfect_images)} ({len(perfect_images)/len(df)*100:.1f}%)")
        print(f"  Images with errors:     {len(error_images)} ({len(error_images)/len(df)*100:.1f}%)")
        
        # Error distribution
        print(f"\n  Error Distribution:")
        print(f"    F1 >= 0.99:           {len(df[df['f1_score'] >= 0.99])} images")
        print(f"    0.95 <= F1 < 0.99:    {len(df[(df['f1_score'] >= 0.95) & (df['f1_score'] < 0.99)])} images")
        print(f"    0.90 <= F1 < 0.95:    {len(df[(df['f1_score'] >= 0.90) & (df['f1_score'] < 0.95)])} images")
        print(f"    F1 < 0.90:            {len(df[df['f1_score'] < 0.90])} images")
        
        # Worst cases
        worst_cases = df.sort_values('f1_score').head(20)
        print(f"\n{'TOP 20 WORST CASES':^80}")
        print("-"*80)
        print(worst_cases[['image_name', 'f1_score', 'precision', 'recall', 
                          'false_positives', 'false_negatives']].to_string(index=False))
        
        return df, results
    
    def visualize_error(self, result, rank):
        """Visualize single error case"""
        img_name = result['image_name']
        img_path = self.img_dir / img_name
        
        if not img_path.exists():
            return False
        
        img = cv2.imread(str(img_path))
        if img is None:
            return False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # Ground Truth
        axes[0].imshow(img)
        axes[0].set_title(f'Ground Truth\n{result["gt_count"]} boxes', 
                         fontsize=14, fontweight='bold')
        for box in result['gt_boxes']:
            poly = patches.Polygon(box, linewidth=2, edgecolor='green', 
                                 facecolor='none', alpha=0.8)
            axes[0].add_patch(poly)
        axes[0].axis('off')
        
        # Predictions
        axes[1].imshow(img)
        axes[1].set_title(f'Predictions\n{result["pred_count"]} boxes', 
                         fontsize=14, fontweight='bold')
        for box in result['pred_boxes']:
            poly = patches.Polygon(box, linewidth=2, edgecolor='blue', 
                                 facecolor='none', alpha=0.8)
            axes[1].add_patch(poly)
        axes[1].axis('off')
        
        # Error Overlay
        axes[2].imshow(img)
        title = f'Error Analysis\nF1={result["f1_score"]:.3f}, P={result["precision"]:.3f}, R={result["recall"]:.3f}'
        axes[2].set_title(title, fontsize=14, fontweight='bold')
        
        match_result = result['match_result']
        
        # True Positives (green)
        for pred_idx, gt_idx, iou in match_result['matched']:
            if pred_idx < len(result['pred_boxes']):
                poly = patches.Polygon(result['pred_boxes'][pred_idx], 
                                     linewidth=2, edgecolor='green', 
                                     facecolor='none', alpha=0.7)
                axes[2].add_patch(poly)
        
        # False Positives (red)
        for pred_idx, _ in match_result['false_positives']:
            if pred_idx < len(result['pred_boxes']):
                poly = patches.Polygon(result['pred_boxes'][pred_idx], 
                                     linewidth=3, edgecolor='red', 
                                     facecolor='none')
                axes[2].add_patch(poly)
        
        # False Negatives (yellow)
        for gt_idx in match_result['false_negatives']:
            if gt_idx < len(result['gt_boxes']):
                poly = patches.Polygon(result['gt_boxes'][gt_idx], 
                                     linewidth=3, edgecolor='yellow', 
                                     facecolor='none')
                axes[2].add_patch(poly)
        
        axes[2].axis('off')
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=3, label=f'TP: {result["true_positives"]}'),
            Line2D([0], [0], color='red', lw=3, label=f'FP: {result["false_positives"]}'),
            Line2D([0], [0], color='yellow', lw=3, label=f'FN: {result["false_negatives"]}')
        ]
        axes[2].legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        plt.suptitle(f'Error Case #{rank:02d}: {img_name}', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        output_name = f'error_{rank:02d}_f1{result["f1_score"]:.3f}_{img_name.replace("/", "_")}.png'
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True


def main():
    # Configuration
    PRED_PATH = '/data/ephemeral/home/baseline_code/predictions/kfold/fold3_predictions.json'
    GT_PATH = '/data/ephemeral/home/data/datasets/jsons/val.json'
    IMG_DIR = '/data/ephemeral/home/data/datasets/images'
    OUTPUT_DIR = '/data/ephemeral/home/error_analysis'
    
    print("="*80)
    print("ERROR CASE ANALYSIS - FOLD 3 BEST MODEL (98.63% H-Mean)")
    print("="*80)
    print(f"\nPredictions: {PRED_PATH}")
    print(f"Ground Truth: {GT_PATH}")
    print(f"Images: {IMG_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Create analyzer
    analyzer = ErrorCaseAnalyzer(PRED_PATH, GT_PATH, IMG_DIR, OUTPUT_DIR)
    
    # Run analysis
    results = analyzer.analyze_all()
    
    # Generate report
    df, results_list = analyzer.generate_report(results)
    
    # Visualize error cases
    print(f"\n{'VISUALIZING ERROR CASES':^80}")
    print("-"*80)
    
    error_results = [r for r in results_list if r['f1_score'] < 1.0]
    error_results_sorted = sorted(error_results, key=lambda x: x['f1_score'])
    
    print(f"\nGenerating visualizations for top 30 error cases...")
    vis_count = 0
    for rank, result in enumerate(error_results_sorted[:30], 1):
        if analyzer.visualize_error(result, rank):
            vis_count += 1
            if vis_count % 10 == 0:
                print(f"  Processed {vis_count} images...")
    
    print(f"✓ Generated {vis_count} visualization images")
    
    print(f"\n{'ANALYSIS COMPLETE':^80}")
    print("="*80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print(f"  - per_image_performance.csv  (모든 이미지별 성능 지표)")
    print(f"  - error_XX_*.png              (에러 케이스 시각화)")
    print("\n" + "="*80)


if __name__ == '__main__':
    try:
        import shapely
        main()
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install required libraries: pip install shapely")
