#!/usr/bin/env python
"""Simple TTA with horizontal flip for ResNet50"""
import os
import sys
import json
import torch
import albumentations as A
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.getcwd())
from ocr.datasets import OCRDataset, DBTransforms
import lightning.pytorch as pl


def merge_boxes(boxes_list):
    """여러 예측 박스를 NMS로 병합"""
    import numpy as np
    from shapely.geometry import Polygon
    
    all_boxes = []
    for boxes in boxes_list:
        all_boxes.extend(boxes)
    
    if not all_boxes:
        return []
    
    # IoU 기반 중복 제거
    merged = []
    used = set()
    
    for i, box1 in enumerate(all_boxes):
        if i in used:
            continue
            
        poly1 = Polygon(box1['points'])
        group = [box1]
        
        for j, box2 in enumerate(all_boxes[i+1:], start=i+1):
            if j in used:
                continue
            poly2 = Polygon(box2['points'])
            
            try:
                iou = poly1.intersection(poly2).area / poly1.union(poly2).area
                if iou > 0.3:  # 30% 이상 겹치면 같은 박스로 간주
                    group.append(box2)
                    used.add(j)
            except:
                continue
        
        # 평균 좌표로 병합
        if len(group) > 1:
            avg_points = []
            for pt_idx in range(4):
                avg_x = sum([b['points'][pt_idx][0] for b in group]) / len(group)
                avg_y = sum([b['points'][pt_idx][1] for b in group]) / len(group)
                avg_points.append([avg_x, avg_y])
            merged.append({'points': avg_points})
        else:
            merged.append(box1)
    
    return merged


def predict_with_tta(checkpoint_path, output_dir='outputs/tta_predictions'):
    """TTA로 예측 (원본 + 수평 플립)"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 모델 로드
    print("Loading model...")
    from ocr.lightning_modules import OCRPLModule
    model = OCRPLModule.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    
    # 데이터셋
    base_transform = DBTransforms(transforms=[
        A.LongestMaxSize(max_size=960, p=1.0),
        A.PadIfNeeded(min_width=960, min_height=960, border_mode=0, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = OCRDataset(
        image_path='/data/ephemeral/home/data/datasets/images/test',
        annotation_path=None,
        transform=base_transform
    )
    
    print(f"Total test images: {len(dataset)}")
    
    # TTA 예측
    final_predictions = {'images': {}}
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="TTA Prediction"):
            image, _, meta = dataset[idx]
            img_name = meta['ori_image_name']
            
            all_boxes = []
            
            # 1. 원본 예측
            img_tensor = image.unsqueeze(0).to(device)
            output = model(img_tensor)
            boxes_orig = model.postprocess(output, [meta])
            all_boxes.append(boxes_orig[0] if boxes_orig else [])
            
            # 2. 수평 플립 예측
            img_flipped = torch.flip(img_tensor, dims=[3])  # W 방향 플립
            output_flip = model(img_flipped)
            boxes_flip = model.postprocess(output_flip, [meta])
            
            # 플립 박스 좌표 복원
            if boxes_flip and boxes_flip[0]:
                for box in boxes_flip[0]:
                    for point in box['points']:
                        point[0] = 960 - point[0]  # x 좌표 반전
                all_boxes.append(boxes_flip[0])
            
            # 박스 병합
            merged_boxes = merge_boxes(all_boxes)
            
            # 저장
            final_predictions['images'][img_name] = {
                'words': {str(i): box for i, box in enumerate(merged_boxes)}
            }
    
    # 저장
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/tta_predictions.json"
    with open(output_path, 'w') as f:
        json.dump(final_predictions, f, indent=2)
    
    print(f"\nTTA predictions saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    checkpoint = "/data/ephemeral/home/baseline_code/outputs/resnet50_fold0/checkpoints/epoch=19-step=14700.ckpt"
    predict_with_tta(checkpoint)
