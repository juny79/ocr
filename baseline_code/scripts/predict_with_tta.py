#!/usr/bin/env python
"""
Test-Time Augmentation (TTA) for OCR Detection
여러 변환(원본, 수평 플립)을 적용하여 앙상블
"""
import os
import sys
import json
import numpy as np
import torch
import albumentations as A
from pathlib import Path
from tqdm import tqdm
import cv2

sys.path.append(os.getcwd())
from ocr.datasets import OCRDataset, DBTransforms
from ocr.lightning_modules import OCRLightningModule
import lightning.pytorch as pl


def merge_predictions(pred_list):
    """여러 TTA 예측을 병합"""
    merged = {}
    
    for pred in pred_list:
        for img_name, content in pred['images'].items():
            if img_name not in merged:
                merged[img_name] = {'words': {}}
            
            # 모든 예측 박스 수집
            for word_id, word_info in content['words'].items():
                # 중복 제거를 위한 키 생성
                points_tuple = tuple(map(tuple, word_info['points']))
                merged[img_name]['words'][points_tuple] = word_info
    
    # Tuple 키를 일반 인덱스로 변환
    final_result = {'images': {}}
    for img_name, content in merged.items():
        final_result['images'][img_name] = {'words': {}}
        for idx, (_, word_info) in enumerate(content['words'].items()):
            final_result['images'][img_name]['words'][str(idx)] = word_info
    
    return final_result


def predict_with_tta(model_path, preset='augmented_resnet50_aggressive', output_dir='outputs/tta_predictions'):
    """TTA를 사용한 예측"""
    
    # 1. 기본 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. 모델 로드
    model = OCRLightningModule.load_from_checkpoint(model_path)
    model.to(device)
    model.eval()
    
    # 3. TTA 변환 정의
    tta_transforms = {
        'original': None,
        'hflip': A.HorizontalFlip(p=1.0),
    }
    
    # 4. 데이터셋 준비
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
    
    # 5. TTA 예측
    all_predictions = []
    
    for tta_name, tta_aug in tta_transforms.items():
        print(f"Running TTA: {tta_name}")
        predictions = {'images': {}}
        
        with torch.no_grad():
            for idx in tqdm(range(len(dataset))):
                image, _, meta = dataset[idx]
                
                # TTA 적용
                if tta_aug:
                    augmented = tta_aug(image=image.numpy().transpose(1, 2, 0))
                    image = torch.from_numpy(augmented['image'].transpose(2, 0, 1))
                
                image = image.unsqueeze(0).to(device)
                
                # 예측
                output = model(image)
                boxes = model.postprocess(output, meta)
                
                # TTA 역변환 (hflip인 경우 박스 좌표 반전)
                if tta_name == 'hflip':
                    for box in boxes:
                        for point in box['points']:
                            point[0] = 960 - point[0]  # x 좌표 반전
                
                # 저장
                img_name = meta['ori_image_name']
                predictions['images'][img_name] = {
                    'words': {str(i): box for i, box in enumerate(boxes)}
                }
        
        all_predictions.append(predictions)
    
    # 6. 예측 병합
    print("Merging TTA predictions...")
    merged_pred = merge_predictions(all_predictions)
    
    # 7. 저장
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/tta_predictions.json"
    with open(output_path, 'w') as f:
        json.dump(merged_pred, f, indent=2)
    
    print(f"TTA predictions saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint path')
    parser.add_argument('--preset', default='augmented_resnet50_aggressive')
    parser.add_argument('--output', default='outputs/tta_predictions')
    args = parser.parse_args()
    
    predict_with_tta(args.checkpoint, args.preset, args.output)
