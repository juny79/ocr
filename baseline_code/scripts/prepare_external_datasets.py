#!/usr/bin/env python3
"""
외부 데이터셋 통합 스크립트
SROIE + CORD-v2 → UFO JSON 변환 및 병합
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List
import argparse
from tqdm import tqdm


def convert_sroie_to_ufo(sroie_base: Path, output_dir: Path) -> Dict:
    """SROIE 데이터셋을 UFO JSON 형식으로 변환"""
    
    print("🔄 SROIE 데이터셋 변환 중...")
    
    ufo_data = {"images": {}}
    
    # SROIE 이미지 및 라벨 경로
    img_dir = sroie_base / "data" / "img"
    key_dir = sroie_base / "data" / "key"
    
    if not img_dir.exists():
        print(f"⚠️  SROIE 이미지 디렉토리 없음: {img_dir}")
        return ufo_data
    
    json_files = list(key_dir.glob("*.json"))
    print(f"📊 SROIE JSON 파일: {len(json_files)}개")
    
    for json_file in tqdm(json_files, desc="SROIE 변환"):
        img_id = json_file.stem
        img_files = list(img_dir.glob(f"{img_id}.*"))
        
        if not img_files:
            continue
            
        img_file = img_files[0]
        
        # SROIE JSON 읽기
        with open(json_file, 'r', encoding='utf-8') as f:
            sroie_data = json.load(f)
        
        # UFO 형식으로 변환
        words = {}
        for idx, item in enumerate(sroie_data.get("valid_line", [])):
            # SROIE는 quad 형식: [x1,y1, x2,y2, x3,y3, x4,y4]
            points = item.get("words", [])
            if len(points) == 8:
                words[f"word_{idx:04d}"] = {
                    "transcription": item.get("text", ""),
                    "points": [
                        [points[0], points[1]],  # top-left
                        [points[2], points[3]],  # top-right
                        [points[4], points[5]],  # bottom-right
                        [points[6], points[7]]   # bottom-left
                    ]
                }
        
        ufo_data["images"][img_file.name] = {
            "words": words,
            "img_w": item.get("img_w", 1000),
            "img_h": item.get("img_h", 1000),
            "tags": ["SROIE"],
            "num_patches": None,
            "source": "external"
        }
    
    print(f"✅ SROIE 변환 완료: {len(ufo_data['images'])}개 이미지")
    return ufo_data


def convert_cord_to_ufo(cord_base: Path, output_dir: Path, split: str = "train") -> Dict:
    """CORD-v2 데이터셋을 UFO JSON 형식으로 변환"""
    
    print(f"🔄 CORD-v2 ({split}) 데이터셋 변환 중...")
    
    ufo_data = {"images": {}}
    
    # CORD-v2 구조: train/ 또는 dev/
    split_dir = cord_base / split
    
    if not split_dir.exists():
        print(f"⚠️  CORD-v2 split 디렉토리 없음: {split_dir}")
        return ufo_data
    
    img_dir = split_dir / "image"
    json_dir = split_dir / "json"
    
    if not img_dir.exists() or not json_dir.exists():
        print(f"⚠️  CORD-v2 이미지/JSON 디렉토리 없음")
        return ufo_data
    
    json_files = list(json_dir.glob("*.json"))
    print(f"📊 CORD-v2 {split} JSON 파일: {len(json_files)}개")
    
    for json_file in tqdm(json_files, desc=f"CORD {split} 변환"):
        img_id = json_file.stem
        img_files = list(img_dir.glob(f"{img_id}.*"))
        
        if not img_files:
            continue
            
        img_file = img_files[0]
        
        # CORD JSON 읽기
        with open(json_file, 'r', encoding='utf-8') as f:
            cord_data = json.load(f)
        
        # UFO 형식으로 변환
        words = {}
        word_idx = 0
        
        # CORD는 nested 구조: valid_line -> words
        for line in cord_data.get("valid_line", []):
            for word_info in line.get("words", []):
                quad = word_info.get("quad", {})
                
                # CORD quad: {"x1": ..., "y1": ..., "x2": ..., "y2": ..., ...}
                if all(k in quad for k in ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]):
                    words[f"word_{word_idx:04d}"] = {
                        "transcription": word_info.get("text", ""),
                        "points": [
                            [quad["x1"], quad["y1"]],  # top-left
                            [quad["x2"], quad["y2"]],  # top-right
                            [quad["x3"], quad["y3"]],  # bottom-right
                            [quad["x4"], quad["y4"]]   # bottom-left
                        ]
                    }
                    word_idx += 1
        
        img_info = cord_data.get("meta", {}).get("image_size", {})
        ufo_data["images"][img_file.name] = {
            "words": words,
            "img_w": img_info.get("width", 1000),
            "img_h": img_info.get("height", 1000),
            "tags": ["CORD-v2"],
            "num_patches": None,
            "source": "external"
        }
    
    print(f"✅ CORD-v2 {split} 변환 완료: {len(ufo_data['images'])}개 이미지")
    return ufo_data


def merge_datasets(base_json: Path, external_jsons: List[Dict], output_path: Path):
    """베이스 데이터셋과 외부 데이터셋 병합"""
    
    print(f"🔗 데이터셋 병합 중...")
    
    # 베이스 데이터 로드
    with open(base_json, 'r', encoding='utf-8') as f:
        merged = json.load(f)
    
    base_count = len(merged.get("images", {}))
    print(f"📊 베이스 데이터: {base_count}개 이미지")
    
    # 외부 데이터 병합
    for ext_data in external_jsons:
        ext_count = len(ext_data.get("images", {}))
        merged["images"].update(ext_data["images"])
        print(f"   + 외부 데이터: {ext_count}개 이미지")
    
    total_count = len(merged["images"])
    print(f"✅ 병합 완료: 총 {total_count}개 이미지 (+{total_count - base_count})")
    
    # 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    
    print(f"💾 저장 완료: {output_path}")
    return merged


def copy_external_images(sroie_base: Path, cord_base: Path, output_img_dir: Path):
    """외부 데이터셋 이미지를 통합 디렉토리로 복사"""
    
    print("🖼️  외부 이미지 복사 중...")
    
    output_img_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    
    # SROIE 이미지 복사
    sroie_img_dir = sroie_base / "data" / "img"
    if sroie_img_dir.exists():
        for img_file in tqdm(list(sroie_img_dir.glob("*.*")), desc="SROIE 이미지"):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                shutil.copy2(img_file, output_img_dir / img_file.name)
                copied += 1
    
    # CORD train 이미지 복사
    cord_train_img = cord_base / "train" / "image"
    if cord_train_img.exists():
        for img_file in tqdm(list(cord_train_img.glob("*.*")), desc="CORD train 이미지"):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                shutil.copy2(img_file, output_img_dir / img_file.name)
                copied += 1
    
    # CORD dev 이미지 복사
    cord_dev_img = cord_base / "dev" / "image"
    if cord_dev_img.exists():
        for img_file in tqdm(list(cord_dev_img.glob("*.*")), desc="CORD dev 이미지"):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                shutil.copy2(img_file, output_img_dir / img_file.name)
                copied += 1
    
    print(f"✅ 이미지 복사 완료: {copied}개")


def main():
    parser = argparse.ArgumentParser(description="외부 데이터셋 통합")
    parser.add_argument("--sroie-dir", type=str, 
                        default="/data/ephemeral/home/data/external_datasets/sroie_raw",
                        help="SROIE 데이터셋 디렉토리")
    parser.add_argument("--cord-dir", type=str,
                        default="/data/ephemeral/home/data/external_datasets/cord-v2",
                        help="CORD-v2 데이터셋 디렉토리")
    parser.add_argument("--base-json", type=str,
                        default="/data/ephemeral/home/data/datasets/jsons/train.json",
                        help="베이스 train.json 파일")
    parser.add_argument("--output-json", type=str,
                        default="/data/ephemeral/home/data/datasets/jsons/train_augmented.json",
                        help="출력 JSON 파일")
    parser.add_argument("--output-img-dir", type=str,
                        default="/data/ephemeral/home/data/datasets/images/all",
                        help="통합 이미지 디렉토리")
    parser.add_argument("--skip-images", action="store_true",
                        help="이미지 복사 건너뛰기 (JSON만 생성)")
    
    args = parser.parse_args()
    
    sroie_base = Path(args.sroie_dir)
    cord_base = Path(args.cord_dir)
    base_json = Path(args.base_json)
    output_json = Path(args.output_json)
    output_img_dir = Path(args.output_img_dir)
    
    print("=" * 80)
    print("🚀 외부 데이터셋 통합 시작")
    print("=" * 80)
    
    # 1. SROIE 변환
    sroie_data = convert_sroie_to_ufo(sroie_base, output_json.parent)
    
    # 2. CORD-v2 변환 (train + dev)
    cord_train_data = convert_cord_to_ufo(cord_base, output_json.parent, split="train")
    cord_dev_data = convert_cord_to_ufo(cord_base, output_json.parent, split="dev")
    
    # 3. 병합
    external_jsons = [sroie_data, cord_train_data, cord_dev_data]
    merged_data = merge_datasets(base_json, external_jsons, output_json)
    
    # 4. 이미지 복사 (옵션)
    if not args.skip_images:
        copy_external_images(sroie_base, cord_base, output_img_dir)
    else:
        print("⏭️  이미지 복사 건너뛰기")
    
    print("=" * 80)
    print("✅ 외부 데이터셋 통합 완료!")
    print("=" * 80)
    print(f"📄 통합 JSON: {output_json}")
    print(f"🖼️  이미지 디렉토리: {output_img_dir}")
    print(f"📊 총 이미지: {len(merged_data['images'])}개")


if __name__ == "__main__":
    main()
