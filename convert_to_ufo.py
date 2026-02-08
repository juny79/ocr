"""
외부 데이터셋을 UFO JSON 포맷으로 변환
SROIE, CORD-v2, WildReceipt 지원
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict
import argparse


def convert_sroie_to_ufo(sroie_root: str, output_file: str):
    """SROIE 데이터를 UFO JSON으로 변환"""
    print(f"\n📝 SROIE 변환 시작...")
    
    sroie_path = Path(sroie_root)
    
    # SROIE 데이터 폴더 찾기
    img_dir = None
    box_dir = None
    
    # 가능한 경로들
    possible_img = [
        sroie_path / "data" / "img",
        sroie_path / "data_train",
        sroie_path / "task2" / "data_train",
    ]
    
    possible_box = [
        sroie_path / "data" / "box",
        sroie_path / "data_train",
        sroie_path / "task2" / "data_train",
    ]
    
    for p in possible_img:
        if p.exists():
            img_dir = p
            break
    
    for p in possible_box:
        if p.exists():
            box_dir = p
            break
    
    if not img_dir or not box_dir:
        print(f"❌ SROIE 폴더를 찾을 수 없습니다.")
        print(f"   img_dir: {img_dir}, box_dir: {box_dir}")
        return None
    
    output_data = {"images": {}}
    img_count = 0
    word_count = 0
    
    # 이미지 파일 처리
    for img_file in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
        img_name = img_file.name
        txt_file = box_dir / f"{img_file.stem}.txt"
        
        if not txt_file.exists():
            continue
        
        words = {}
        word_idx = 1
        
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 8:
                        continue
                    
                    try:
                        coords = [int(parts[i]) for i in range(8)]
                        points = [[coords[j], coords[j+1]] for j in range(0, 8, 2)]
                        
                        words[f"{word_idx:04d}"] = {
                            "points": points,
                            "orientation": "Horizontal"
                        }
                        word_idx += 1
                        word_count += 1
                    except (ValueError, IndexError):
                        continue
            
            if words:
                output_data["images"][img_name] = {"words": words}
                img_count += 1
                if img_count % 100 == 0:
                    print(f"  ✓ {img_count} images processed...")
        
        except Exception as e:
            print(f"  ⚠️ Error processing {txt_file}: {e}")
    
    # 저장
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ SROIE 변환 완료: {img_count} images, {word_count} words")
    print(f"   Output: {output_file}")
    
    return output_data


def convert_cord_to_ufo(cord_cache: str, output_file: str):
    """CORD-v2 HF 캐시 데이터를 UFO JSON으로 변환"""
    print(f"\n📝 CORD-v2 변환 시작...")
    
    cache_path = Path(cord_cache)
    
    # Hugging Face 캐시에서 parquet 파일 찾기
    parquet_files = list(cache_path.rglob("*.parquet"))
    
    if not parquet_files:
        print(f"❌ CORD parquet 파일을 찾을 수 없습니다: {cache_path}")
        return None
    
    try:
        import pandas as pd
    except ImportError:
        print("⚠️ pandas 설치 필요: pip install pandas")
        return None
    
    output_data = {"images": {}}
    img_count = 0
    word_count = 0
    
    # 각 parquet 파일 처리
    for parquet_file in parquet_files:
        try:
            df = pd.read_parquet(parquet_file)
            
            for idx, row in df.iterrows():
                img_name = row.get('image', {}).get('path', f"image_{idx}.jpg")
                if isinstance(img_name, dict):
                    img_name = img_name.get('path', f"image_{idx}.jpg")
                
                words = {}
                word_idx = 1
                
                # CORD JSON 구조: words 필드
                if 'words' in row:
                    words_data = row['words']
                    if isinstance(words_data, list):
                        for word_info in words_data:
                            if isinstance(word_info, dict) and 'quad' in word_info:
                                points = word_info['quad']
                                if len(points) >= 4:
                                    words[f"{word_idx:04d}"] = {
                                        "points": points,
                                        "orientation": word_info.get('orientation', 'Horizontal')
                                    }
                                    word_idx += 1
                                    word_count += 1
                
                if words:
                    output_data["images"][img_name] = {"words": words}
                    img_count += 1
                    
                    if img_count % 100 == 0:
                        print(f"  ✓ {img_count} images processed...")
        
        except Exception as e:
            print(f"  ⚠️ Error reading {parquet_file}: {e}")
    
    # 저장
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ CORD 변환 완료: {img_count} images, {word_count} words")
    print(f"   Output: {output_file}")
    
    return output_data


def convert_wildreceipt_to_ufo(wr_root: str, output_file: str):
    """WildReceipt 데이터를 UFO JSON으로 변환"""
    print(f"\n📝 WildReceipt 변환 시작...")
    
    wr_path = Path(wr_root)
    
    # WildReceipt 폴더 구조 찾기
    img_dir = None
    anno_dir = None
    
    # pseudo_label 구조
    if (wr_path / "images").exists():
        img_dir = wr_path / "images"
    elif (wr_path / "image").exists():
        img_dir = wr_path / "image"
    
    if (wr_path / "annotations").exists():
        anno_dir = wr_path / "annotations"
    elif (wr_path / "anno").exists():
        anno_dir = wr_path / "anno"
    
    if not img_dir:
        print(f"❌ WildReceipt 이미지 폴더를 찾을 수 없습니다: {wr_path}")
        return None
    
    output_data = {"images": {}}
    img_count = 0
    word_count = 0
    
    # 이미지 파일 처리
    for img_file in sorted(img_dir.glob("**/*.jpg")) + sorted(img_dir.glob("**/*.png")):
        img_name = img_file.name
        
        # 라벨 파일 찾기
        words = {}
        word_idx = 1
        
        if anno_dir:
            # TXT 파일 시도
            txt_file = anno_dir / f"{img_file.stem}.txt"
            if txt_file.exists():
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split(',')
                            if len(parts) >= 8:
                                try:
                                    coords = [int(parts[i]) for i in range(8)]
                                    points = [[coords[j], coords[j+1]] for j in range(0, 8, 2)]
                                    words[f"{word_idx:04d}"] = {
                                        "points": points,
                                        "orientation": "Horizontal"
                                    }
                                    word_idx += 1
                                    word_count += 1
                                except (ValueError, IndexError):
                                    continue
                except Exception as e:
                    print(f"  ⚠️ Error reading {txt_file}: {e}")
            
            # JSON 파일 시도
            if not words:
                json_file = anno_dir / f"{img_file.stem}.json"
                if json_file.exists():
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            anno_data = json.load(f)
                            
                            # 가능한 필드들
                            for key in ['annotations', 'lines', 'words']:
                                if key in anno_data:
                                    items = anno_data[key] if isinstance(anno_data[key], list) else [anno_data[key]]
                                    for item in items:
                                        if 'quad' in item and len(item['quad']) >= 4:
                                            words[f"{word_idx:04d}"] = {
                                                "points": item['quad'],
                                                "orientation": item.get('orientation', 'Horizontal')
                                            }
                                            word_idx += 1
                                            word_count += 1
                    except Exception as e:
                        print(f"  ⚠️ Error reading {json_file}: {e}")
        
        if words:
            output_data["images"][img_name] = {"words": words}
            img_count += 1
            
            if img_count % 100 == 0:
                print(f"  ✓ {img_count} images processed...")
    
    # 저장
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ WildReceipt 변환 완료: {img_count} images, {word_count} words")
    print(f"   Output: {output_file}")
    
    return output_data


def merge_datasets(json_files: list, output_file: str, include_base=False):
    """여러 JSON을 병합"""
    print(f"\n📝 데이터셋 병합 시작...")
    
    merged = {"images": {}}
    total_imgs = 0
    total_words = 0
    
    # 대회 데이터 포함 여부
    if include_base:
        base_file = "/data/ephemeral/home/data/datasets/jsons/train.json"
        if os.path.exists(base_file):
            print(f"  Including base dataset: {base_file}")
            with open(base_file, 'r', encoding='utf-8') as f:
                base_data = json.load(f)
                if "images" in base_data:
                    merged["images"].update(base_data["images"])
                    total_imgs += len(base_data["images"])
    
    # 외부 데이터셋 추가
    for json_file in json_files:
        if not os.path.exists(json_file):
            print(f"  ⚠️ File not found: {json_file}")
            continue
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if "images" in data:
                merged["images"].update(data["images"])
                total_imgs += len(data["images"])
                print(f"  ✓ Added {len(data['images'])} from {os.path.basename(json_file)}")
    
    # 단어 수 계산
    for img_data in merged["images"].values():
        if "words" in img_data:
            total_words += len(img_data["words"])
    
    # 저장
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 병합 완료: {total_imgs} images, {total_words} words")
    print(f"   Output: {output_file}")
    
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert external datasets to UFO format")
    parser.add_argument("--sroie", type=str, help="SROIE root directory")
    parser.add_argument("--cord", type=str, help="CORD cache directory")
    parser.add_argument("--wildreceipt", type=str, help="WildReceipt root directory")
    parser.add_argument("--merge", action="store_true", help="Merge datasets")
    parser.add_argument("--include_base", action="store_true", help="Include base dataset in merge")
    parser.add_argument("--output_dir", type=str, default="./converted_data", help="Output directory")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    json_files = []
    
    if args.sroie:
        output_file = os.path.join(args.output_dir, "sroie.json")
        convert_sroie_to_ufo(args.sroie, output_file)
        json_files.append(output_file)
    
    if args.cord:
        output_file = os.path.join(args.output_dir, "cord.json")
        convert_cord_to_ufo(args.cord, output_file)
        json_files.append(output_file)
    
    if args.wildreceipt:
        output_file = os.path.join(args.output_dir, "wildreceipt.json")
        convert_wildreceipt_to_ufo(args.wildreceipt, output_file)
        json_files.append(output_file)
    
    if args.merge and json_files:
        merge_output = os.path.join(args.output_dir, "train_all_external.json")
        merge_datasets(json_files, merge_output, include_base=args.include_base)
