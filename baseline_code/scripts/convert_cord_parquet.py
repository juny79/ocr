#!/usr/bin/env python3
"""
CORD-v2 Parquet 데이터를 UFO 포맷으로 변환
"""
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import io
import base64


def convert_cord_parquet_to_ufo(parquet_dir: Path, output_json: Path, image_output_dir: Path):
    """CORD-v2 Parquet 파일을 UFO 형식으로 변환"""
    
    print("🔄 CORD-v2 Parquet 데이터 로드 중...")
    
    # Train parquet 파일들 찾기
    train_files = sorted(parquet_dir.glob("train-*.parquet"))
    
    if not train_files:
        print(f"⚠️  Train parquet 파일 없음: {parquet_dir}")
        return {}
    
    print(f"📊 발견된 파일: {len(train_files)}개")
    
    ufo_data = {"images": {}}
    image_output_dir.mkdir(parents=True, exist_ok=True)
    
    total_images = 0
    file_idx = 0
    
    for parquet_file in train_files:
        print(f"📂 처리 중: {parquet_file.name}")
        df = pd.read_parquet(parquet_file)
        
        print(f"  - 레코드 수: {len(df)}")
        print(f"  - 컬럼: {list(df.columns)}")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  변환"):
            try:
                # 이미지 저장 (고유 ID 생성)
                img_id = f"cord_{file_idx:05d}"
                file_idx += 1
                if 'image' in row:
                    img_data = row['image']
                    
                    # PIL Image 객체인 경우
                    if hasattr(img_data, 'convert'):
                        img = img_data
                    # bytes인 경우
                    elif isinstance(img_data, bytes):
                        img = Image.open(io.BytesIO(img_data))
                    # dict 형태인 경우 (HuggingFace datasets 형식)
                    elif isinstance(img_data, dict) and 'bytes' in img_data:
                        img = Image.open(io.BytesIO(img_data['bytes']))
                    else:
                        print(f"  ⚠️  알 수 없는 이미지 형식: {type(img_data)}")
                        continue
                    
                    img_filename = f"{img_id}.jpg"
                    img_path = image_output_dir / img_filename
                    img.save(img_path, 'JPEG')
                    img_w, img_h = img.size
                else:
                    print(f"  ⚠️  이미지 데이터 없음 ({img_id})")
                    continue
                
                # Annotation 처리
                words = {}
                word_count = 0
                
                # ground_truth 필드에서 bbox 정보 추출
                if 'ground_truth' in row:
                    annotations = row['ground_truth']
                    
                    # JSON 문자열인 경우 파싱
                    if isinstance(annotations, str):
                        annotations = json.loads(annotations)
                    
                    # valid_line 형식 처리
                    if isinstance(annotations, dict) and 'valid_line' in annotations:
                        valid_lines = annotations['valid_line']
                        
                        for line in valid_lines:
                            if not isinstance(line, dict) or 'words' not in line:
                                continue
                            
                            # 각 line 안의 words 리스트 처리
                            for word_item in line['words']:
                                if not isinstance(word_item, dict):
                                    continue
                                
                                # quad 좌표 추출
                                if 'quad' in word_item:
                                    quad = word_item['quad']
                                    points = [
                                        [quad.get('x1', 0), quad.get('y1', 0)],
                                        [quad.get('x2', 0), quad.get('y2', 0)],
                                        [quad.get('x3', 0), quad.get('y3', 0)],
                                        [quad.get('x4', 0), quad.get('y4', 0)]
                                    ]
                                    
                                    text = word_item.get('text', '')
                                    
                                    words[f"word_{word_count:04d}"] = {
                                        "transcription": text,
                                        "points": points
                                    }
                                    word_count += 1
                
                if not words:
                    continue
                
                ufo_data["images"][img_filename] = {
                    "words": words,
                    "img_w": img_w,
                    "img_h": img_h,
                    "tags": ["CORD"],
                    "num_patches": None,
                    "source": "external"
                }
                
                total_images += 1
                
            except Exception as e:
                print(f"  ⚠️  처리 실패 ({idx}): {e}")
                continue
    
    print(f"✅ CORD-v2 변환 완료: {total_images}개 이미지")
    
    # JSON 저장
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(ufo_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 CORD UFO JSON 저장: {output_json}")
    
    return ufo_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    
    args = parser.parse_args()
    
    convert_cord_parquet_to_ufo(args.parquet_dir, args.output_json, args.image_dir)
