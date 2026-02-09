#!/usr/bin/env python3
"""
외부 데이터셋(SROIE, CORD-v2)를 UFO 포맷으로 변환하고 기존 데이터와 통합
"""
import json
import shutil
from pathlib import Path
from typing import Dict, List
import argparse
from tqdm import tqdm
from PIL import Image


def convert_sroie_to_ufo(sroie_base: Path, output_dir: Path) -> Dict:
    """SROIE 데이터셋을 UFO JSON 형식으로 변환"""
    
    print("🔄 SROIE 데이터셋 변환 중...")
    
    ufo_data = {"images": {}}
    
    # SROIE 구조: data/img, data/box (CSV 파일)
    img_dir = sroie_base / "data" / "img"
    box_dir = sroie_base / "data" / "box"
    
    if not box_dir.exists():
        print(f"⚠️  SROIE box 디렉토리 없음: {box_dir}")
        return ufo_data
    
    box_files = list(box_dir.glob("*.csv"))
    print(f"📊 SROIE 박스 파일: {len(box_files)}개")
    
    for box_file in tqdm(box_files, desc="SROIE 변환"):
        img_id = box_file.stem
        img_files = list(img_dir.glob(f"{img_id}.*"))
        
        if not img_files:
            continue
            
        img_file = img_files[0]
        
        # 이미지 크기 읽기
        try:
            with Image.open(img_file) as img:
                img_w, img_h = img.size
        except Exception as e:
            print(f"⚠️  이미지 읽기 실패 ({img_id}): {e}")
            continue
        
        # CSV 파일 읽기 (x1,y1,x2,y2,x3,y3,x4,y4,text)
        words = {}
        try:
            with open(box_file, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split(',')
                    if len(parts) < 9:  # 8개 좌표 + 텍스트
                        continue
                    
                    try:
                        coords = [int(parts[i]) for i in range(8)]
                        text = ','.join(parts[8:])  # 텍스트에 쉼표가 있을 수 있음
                        
                        points = [
                            [coords[0], coords[1]],  # 좌상단
                            [coords[2], coords[3]],  # 우상단
                            [coords[4], coords[5]],  # 우하단
                            [coords[6], coords[7]]   # 좌하단
                        ]
                        
                        words[f"word_{idx:04d}"] = {
                            "transcription": text,
                            "points": points
                        }
                    except (ValueError, IndexError):
                        continue
        except Exception as e:
            print(f"⚠️  CSV 읽기 실패 ({img_id}): {e}")
            continue
        
        if not words:  # 바운딩 박스가 없으면 스킵
            continue
        
        ufo_data["images"][img_file.name] = {
            "words": words,
            "img_w": img_w,
            "img_h": img_h,
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
    
    # CORD-v2 구조 탐색
    possible_paths = [
        (cord_base / split / "image", cord_base / split / "json"),
        (cord_base / "image" / split, cord_base / "json" / split),
        (cord_base / "images" / split, cord_base / "annotations" / split),
    ]
    
    img_dir, json_dir = None, None
    for img_path, json_path in possible_paths:
        if img_path.exists() and json_path.exists():
            img_dir, json_dir = img_path, json_path
            break
    
    if img_dir is None:
        print(f"⚠️  CORD-v2 구조를 찾을 수 없음: {cord_base}")
        # 구조 탐색
        print("📂 CORD-v2 디렉토리 구조:")
        for item in cord_base.rglob("*"):
            if item.is_dir():
                depth = len(item.relative_to(cord_base).parts)
                if depth <= 2:
                    print(f"  {'  ' * depth}{item.name}/")
        return ufo_data
    
    json_files = list(json_dir.glob("*.json"))
    print(f"📊 CORD-v2 {split} JSON 파일: {len(json_files)}개")
    
    for json_file in tqdm(json_files, desc=f"CORD {split} 변환"):
        img_id = json_file.stem
        img_files = list(img_dir.glob(f"{img_id}.*"))
        
        if not img_files:
            continue
            
        img_file = img_files[0]
        
        # 이미지 크기 읽기
        try:
            with Image.open(img_file) as img:
                img_w, img_h = img.size
        except Exception as e:
            print(f"⚠️  이미지 읽기 실패 ({img_id}): {e}")
            continue
        
        # CORD JSON 읽기
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                cord_data = json.load(f)
        except Exception as e:
            print(f"⚠️  JSON 읽기 실패 ({img_id}): {e}")
            continue
        
        # UFO 형식으로 변환
        words = {}
        valid_line = cord_data.get("valid_line", [])
        
        for idx, item in enumerate(valid_line):
            # CORD는 quad 형식
            points = item.get("quad", {})
            if not points:
                continue
            
            # quad 좌표 변환
            quad_points = [
                [points.get("x1", 0), points.get("y1", 0)],
                [points.get("x2", 0), points.get("y2", 0)],
                [points.get("x3", 0), points.get("y3", 0)],
                [points.get("x4", 0), points.get("y4", 0)]
            ]
            
            words[f"word_{idx:04d}"] = {
                "transcription": item.get("text", ""),
                "points": quad_points
            }
        
        if not words:  # 바운딩 박스가 없으면 스킵
            continue
        
        ufo_data["images"][img_file.name] = {
            "words": words,
            "img_w": img_w,
            "img_h": img_h,
            "tags": ["CORD"],
            "num_patches": None,
            "source": "external"
        }
    
    print(f"✅ CORD-v2 변환 완료: {len(ufo_data['images'])}개 이미지")
    return ufo_data


def merge_datasets(base_json: Path, external_data_list: List[Dict], output_json: Path):
    """기존 데이터와 외부 데이터 병합"""
    
    print("\n🔄 데이터셋 병합 중...")
    
    # 기존 데이터 로드
    with open(base_json, 'r', encoding='utf-8') as f:
        base_data = json.load(f)
    
    print(f"📊 기존 데이터: {len(base_data['images'])}개 이미지")
    
    # 외부 데이터 추가
    total_added = 0
    for external_data in external_data_list:
        for img_name, img_info in external_data["images"].items():
            if img_name not in base_data["images"]:
                base_data["images"][img_name] = img_info
                total_added += 1
    
    print(f"✅ {total_added}개 이미지 추가")
    print(f"📊 최종 데이터셋: {len(base_data['images'])}개 이미지")
    
    # 저장
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(base_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 통합 JSON 저장: {output_json}")


def copy_external_images(external_bases: List[tuple], target_dir: Path):
    """외부 이미지를 통합 이미지 디렉토리로 복사"""
    
    print("\n🔄 외부 이미지 복사 중...")
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    total_copied = 0
    for base_path, data_type in external_bases:
        if data_type == "sroie":
            img_dir = base_path / "data" / "img"
        elif data_type == "cord":
            # CORD 구조 탐색
            possible_dirs = [
                base_path / "train" / "image",
                base_path / "image" / "train",
                base_path / "images" / "train"
            ]
            img_dir = None
            for path in possible_dirs:
                if path.exists():
                    img_dir = path
                    break
            if img_dir is None:
                print(f"⚠️  CORD 이미지 디렉토리를 찾을 수 없음")
                continue
        else:
            continue
        
        if not img_dir.exists():
            print(f"⚠️  이미지 디렉토리 없음: {img_dir}")
            continue
        
        img_files = list(img_dir.glob("*.*"))
        for img_file in tqdm(img_files, desc=f"{data_type.upper()} 이미지 복사"):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                target_file = target_dir / img_file.name
                if not target_file.exists():
                    shutil.copy2(img_file, target_file)
                    total_copied += 1
    
    print(f"✅ {total_copied}개 이미지 복사 완료")


def main():
    parser = argparse.ArgumentParser(description="외부 데이터셋 통합")
    parser.add_argument("--sroie-dir", type=Path, required=True,
                       help="SROIE 데이터셋 경로")
    parser.add_argument("--cord-dir", type=Path, required=True,
                       help="CORD-v2 데이터셋 경로")
    parser.add_argument("--base-json", type=Path, required=True,
                       help="기존 train.json 경로")
    parser.add_argument("--output-json", type=Path, required=True,
                       help="통합 JSON 출력 경로")
    parser.add_argument("--image-dir", type=Path,
                       default=Path("/data/ephemeral/home/data/datasets/images"),
                       help="이미지 통합 디렉토리")
    
    args = parser.parse_args()
    
    print("🚀 외부 데이터셋 통합 시작\n")
    
    # 1. SROIE 변환
    sroie_data = convert_sroie_to_ufo(args.sroie_dir, args.output_json.parent)
    
    # 2. CORD-v2 변환
    cord_data = convert_cord_to_ufo(args.cord_dir, args.output_json.parent, split="train")
    
    # 3. 데이터셋 병합
    merge_datasets(args.base_json, [sroie_data, cord_data], args.output_json)
    
    # 4. 이미지 복사
    copy_external_images([
        (args.sroie_dir, "sroie"),
        (args.cord_dir, "cord")
    ], args.image_dir)
    
    print("\n✅ 외부 데이터셋 통합 완료!")
    print(f"📄 통합 JSON: {args.output_json}")
    print(f"📂 이미지 디렉토리: {args.image_dir}")


if __name__ == "__main__":
    main()
