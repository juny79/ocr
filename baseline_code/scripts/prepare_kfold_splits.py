#!/usr/bin/env python3
"""
K-Fold 데이터 분할 준비
외부 데이터 포함 (4,698개 이미지) → 5-Fold Split
"""
import json
from pathlib import Path
from sklearn.model_selection import KFold
import numpy as np


def create_kfold_splits(json_path: Path, output_dir: Path, n_splits: int = 5):
    """
    UFO JSON 데이터를 K-Fold로 분할
    
    Args:
        json_path: 전체 데이터 JSON 경로
        output_dir: 출력 디렉토리
        n_splits: Fold 개수 (기본 5)
    """
    print(f"🔄 K-Fold 데이터 분할 시작 (n_splits={n_splits})")
    
    # 데이터 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    image_names = list(data['images'].keys())
    print(f"📊 전체 이미지: {len(image_names)}개")
    
    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # K-Fold split
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_info = {}
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(image_names)):
        print(f"\n📁 Fold {fold_idx}")
        
        # Train/Val 이미지 리스트
        train_images = [image_names[i] for i in train_idx]
        val_images = [image_names[i] for i in val_idx]
        
        print(f"  - Train: {len(train_images)}개")
        print(f"  - Val: {len(val_images)}개")
        
        # Train JSON 생성
        train_data = {"images": {}}
        for img_name in train_images:
            train_data["images"][img_name] = data["images"][img_name]
        
        train_json_path = output_dir / f"train_fold_{fold_idx}.json"
        with open(train_json_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        # Val JSON 생성
        val_data = {"images": {}}
        for img_name in val_images:
            val_data["images"][img_name] = data["images"][img_name]
        
        val_json_path = output_dir / f"val_fold_{fold_idx}.json"
        with open(val_json_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        
        # Fold 정보 저장
        fold_info[f"fold_{fold_idx}"] = {
            "train_json": str(train_json_path),
            "val_json": str(val_json_path),
            "train_count": len(train_images),
            "val_count": len(val_images)
        }
        
        print(f"  ✅ {train_json_path.name}")
        print(f"  ✅ {val_json_path.name}")
    
    # Fold 매핑 정보 저장
    fold_mapping_path = output_dir / "fold_mapping.json"
    with open(fold_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(fold_info, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ K-Fold 분할 완료!")
    print(f"💾 Fold 매핑: {fold_mapping_path}")
    
    return fold_info


if __name__ == "__main__":
    # 외부 데이터 포함 JSON
    json_path = Path("/data/ephemeral/home/data/datasets/jsons/train_augmented_full.json")
    output_dir = Path("/data/ephemeral/home/data/datasets/kfold_splits")
    
    fold_info = create_kfold_splits(json_path, output_dir, n_splits=5)
    
    print("\n📊 Fold 통계:")
    for fold_name, info in fold_info.items():
        print(f"  {fold_name}: Train {info['train_count']} / Val {info['val_count']}")
