#!/usr/bin/env python3
"""
외부 데이터셋들을 기존 train.json과 병합
"""
import json
from pathlib import Path


def merge_all_datasets():
    """SROIE + CORD + 기존 데이터 통합"""
    
    print("🔄 전체 데이터셋 병합 시작\n")
    
    # 파일 경로
    base_json = Path("/data/ephemeral/home/data/datasets/jsons/train.json")
    sroie_json = Path("/data/ephemeral/home/data/datasets/jsons/train_augmented.json")  # SROIE 포함
    cord_json = Path("/data/ephemeral/home/data/datasets/jsons/cord_ufo.json")
    output_json = Path("/data/ephemeral/home/data/datasets/jsons/train_augmented_full.json")
    
    # 1. 기존 데이터 로드 (이미 SROIE 포함되어 있음)
    with open(sroie_json, 'r', encoding='utf-8') as f:
        merged_data = json.load(f)
    
    print(f"📊 기존 + SROIE 데이터: {len(merged_data['images'])}개 이미지")
    
    # 2. CORD 데이터 추가
    with open(cord_json, 'r', encoding='utf-8') as f:
        cord_data = json.load(f)
    
    print(f"📊 CORD-v2 데이터: {len(cord_data['images'])}개 이미지")
    
    added = 0
    for img_name, img_info in cord_data["images"].items():
        if img_name not in merged_data["images"]:
            merged_data["images"][img_name] = img_info
            added += 1
    
    print(f"✅ CORD-v2 {added}개 이미지 추가")
    print(f"📊 최종 데이터셋: {len(merged_data['images'])}개 이미지")
    
    # 데이터 출처 통계
    sroie_count = sum(1 for img in merged_data['images'].values() if 'SROIE' in img.get('tags', []))
    cord_count = sum(1 for img in merged_data['images'].values() if 'CORD' in img.get('tags', []))
    original_count = len(merged_data['images']) - sroie_count - cord_count
    
    print(f"\n📈 데이터 구성:")
    print(f"  - 원본 데이터: {original_count}개")
    print(f"  - SROIE: {sroie_count}개")
    print(f"  - CORD-v2: {cord_count}개")
    print(f"  - 총합: {len(merged_data['images'])}개")
    
    # 3. 저장
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 통합 JSON 저장: {output_json}")
    
    # 파일 크기 비교
    import os
    base_size = os.path.getsize(base_json) / (1024 * 1024)
    output_size = os.path.getsize(output_json) / (1024 * 1024)
    print(f"\n📏 파일 크기:")
    print(f"  - 원본: {base_size:.1f} MB")
    print(f"  - 통합: {output_size:.1f} MB (x{output_size/base_size:.2f})")


if __name__ == "__main__":
    merge_all_datasets()
