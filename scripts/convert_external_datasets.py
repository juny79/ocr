"""
SROIE, CORD-v2, WildReceipt를 대회 포맷(UFO/JSON)으로 변환하는 스크립트

사용법:
1. SROIE: python convert_external_datasets.py --dataset sroie --input_dir ./sroie_raw --output_dir ./converted
2. CORD-v2: python convert_external_datasets.py --dataset cord --input_dir ./cord_raw --output_dir ./converted  
3. WildReceipt: python convert_external_datasets.py --dataset wildreceipt --input_dir ./wildreceipt_raw --output_dir ./converted
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def convert_sroie(input_dir: str, output_path: str) -> Dict:
    """
    SROIE 포맷 변환
    
    Input: 
        - images/train/x1.jpg
        - x1.txt (x1,y1,x2,y2,x3,y3,x4,y4,text)
    
    Output:
        - UFO JSON: points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    input_path = Path(input_dir)
    images_dir = input_path / "images" / "train"
    annotations_dir = input_path / "annotations" if (input_path / "annotations").exists() else input_path
    
    output_data = {"images": {}}
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    for img_file in images_dir.glob("*.jpg"):
        img_name = img_file.name
        txt_file = annotations_dir / f"{img_file.stem}.txt"
        
        if not txt_file.exists():
            print(f"⚠️ Warning: {txt_file} not found, skipping {img_name}")
            continue
        
        words = {}
        word_idx = 1
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 9:
                    continue
                
                try:
                    coords = [int(parts[i]) for i in range(8)]
                    points = [[coords[j], coords[j+1]] for j in range(0, 8, 2)]
                    
                    # SROIE GT는 text 정보를 가지고 있지만 Detection 평가에서는 사용 안 함
                    words[f"{word_idx:04d}"] = {
                        "points": points,
                        "orientation": "Horizontal"
                    }
                    word_idx += 1
                except (ValueError, IndexError) as e:
                    print(f"⚠️ Error parsing line in {txt_file}: {line.strip()}")
                    continue
        
        output_data["images"][img_name] = {"words": words}
        print(f"✓ Converted {img_name}: {len(words)} words")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ SROIE conversion complete: {output_path}")
    print(f"  Total images: {len(output_data['images'])}")
    return output_data


def convert_cord(input_dir: str, output_path: str) -> Dict:
    """
    CORD-v2 포맷 변환
    
    Input:
        - json/train.json (nested structure)
    
    Output:
        - UFO JSON: points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    input_path = Path(input_dir)
    
    # CORD 데이터셋 구조 확인
    possible_json_paths = [
        input_path / "json" / "train.json",
        input_path / "train.json",
        input_path / "jsons" / "train.json",
    ]
    
    json_file = None
    for path in possible_json_paths:
        if path.exists():
            json_file = path
            break
    
    if json_file is None:
        raise FileNotFoundError(f"CORD train.json not found in {input_path}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        cord_data = json.load(f)
    
    output_data = {"images": {}}
    
    # CORD 구조: {"images": {"image_name.jpg": {"words": {"0001": ...}}}}
    if "images" in cord_data:
        for img_name, img_info in cord_data["images"].items():
            words = {}
            word_idx = 1
            
            if "words" in img_info:
                for word_id, word_info in img_info["words"].items():
                    if "points" in word_info:
                        points = word_info["points"]
                        
                        # 최소 4개 점 이상이어야 유효
                        if len(points) >= 4:
                            words[f"{word_idx:04d}"] = {
                                "points": points,
                                "orientation": word_info.get("orientation", "Horizontal")
                            }
                            word_idx += 1
            
            if words:
                output_data["images"][img_name] = {"words": words}
                print(f"✓ Converted {img_name}: {len(words)} words")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ CORD conversion complete: {output_path}")
    print(f"  Total images: {len(output_data['images'])}")
    return output_data


def convert_wildreceipt(input_dir: str, output_path: str) -> Dict:
    """
    WildReceipt 포맷 변환
    
    Input:
        - images/train/x_0_0.jpg
        - annotations/train/x_0_0.json (또는 .txt)
    
    Output:
        - UFO JSON: points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    input_path = Path(input_dir)
    images_dir = input_path / "images"
    annotations_dir = input_path / "annotations"
    
    output_data = {"images": {}}
    
    # 가능한 구조들 확인
    if not images_dir.exists():
        images_dir = input_path / "train" / "images"
        annotations_dir = input_path / "train" / "annotations"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    for img_file in sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png")):
        img_name = img_file.name
        
        # 먼저 json 파일 시도
        json_file = annotations_dir / f"{img_file.stem}.json"
        txt_file = annotations_dir / f"{img_file.stem}.txt"
        
        words = {}
        word_idx = 1
        
        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    anno_data = json.load(f)
                
                # WildReceipt JSON 구조 파싱
                if "annotations" in anno_data:
                    for anno in anno_data["annotations"]:
                        if "quad" in anno:
                            points = anno["quad"]
                            if len(points) >= 4:
                                words[f"{word_idx:04d}"] = {
                                    "points": points,
                                    "orientation": "Horizontal"
                                }
                                word_idx += 1
                elif "lines" in anno_data:
                    for line in anno_data["lines"]:
                        if "quad" in line:
                            points = line["quad"]
                            if len(points) >= 4:
                                words[f"{word_idx:04d}"] = {
                                    "points": points,
                                    "orientation": "Horizontal"
                                }
                                word_idx += 1
            except Exception as e:
                print(f"⚠️ Error parsing {json_file}: {e}")
        
        elif txt_file.exists():
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
                        except (ValueError, IndexError):
                            continue
        
        if words:
            output_data["images"][img_name] = {"words": words}
            print(f"✓ Converted {img_name}: {len(words)} words")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ WildReceipt conversion complete: {output_path}")
    print(f"  Total images: {len(output_data['images'])}")
    return output_data


def merge_datasets(json_files: List[str], output_path: str) -> Dict:
    """
    여러 개의 JSON 파일을 하나로 병합
    """
    merged_data = {"images": {}}
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "images" in data:
            merged_data["images"].update(data["images"])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Merged {len(json_files)} datasets")
    print(f"  Total images: {len(merged_data['images'])}")
    print(f"  Output: {output_path}")
    return merged_data


def main():
    parser = argparse.ArgumentParser(description="Convert external OCR datasets to UFO format")
    parser.add_argument("--dataset", type=str, choices=["sroie", "cord", "wildreceipt", "merge"],
                        required=True, help="Dataset to convert")
    parser.add_argument("--input_dir", type=str, help="Input directory path")
    parser.add_argument("--input_files", type=str, nargs="+", help="Input JSON files (for merge)")
    parser.add_argument("--output_dir", type=str, default="./converted_data",
                        help="Output directory")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.dataset == "sroie":
        if not args.input_dir:
            raise ValueError("--input_dir required for SROIE")
        output_path = os.path.join(args.output_dir, "sroie.json")
        convert_sroie(args.input_dir, output_path)
    
    elif args.dataset == "cord":
        if not args.input_dir:
            raise ValueError("--input_dir required for CORD")
        output_path = os.path.join(args.output_dir, "cord.json")
        convert_cord(args.input_dir, output_path)
    
    elif args.dataset == "wildreceipt":
        if not args.input_dir:
            raise ValueError("--input_dir required for WildReceipt")
        output_path = os.path.join(args.output_dir, "wildreceipt.json")
        convert_wildreceipt(args.input_dir, output_path)
    
    elif args.dataset == "merge":
        if not args.input_files:
            raise ValueError("--input_files required for merge")
        output_path = os.path.join(args.output_dir, "train_all_external.json")
        merge_datasets(args.input_files, output_path)


if __name__ == "__main__":
    main()
