#!/usr/bin/env python3
"""
SROIE 데이터셋 다운로드 스크립트
Kaggle API 없이 GitHub에서 직접 다운로드
"""

import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

def download_file(url, output_path, chunk_size=8192):
    """파일 다운로드 (진행률 표시)"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(output_path)) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"✓ 다운로드 완료: {output_path}")


def download_sroie():
    """SROIE 데이터셋 다운로드 및 추출"""
    
    print("\n" + "="*60)
    print("SROIE 데이터셋 다운로드 시작 (방법: GitHub 리포지토리)")
    print("="*60)
    
    base_dir = Path("/data/ephemeral/home/data/external_datasets")
    sroie_dir = base_dir / "sroie_raw"
    
    # 이미 있으면 스킵
    if sroie_dir.exists():
        print(f"✓ SROIE 폴더가 이미 존재합니다: {sroie_dir}")
        return str(sroie_dir)
    
    os.makedirs(base_dir, exist_ok=True)
    
    print("\n📥 SROIE 다운로드 중...")
    print("   링크: https://github.com/zzzdavid/ICDAR-2019-SROIE")
    
    # GitHub에서 직접 다운로드
    zip_url = "https://github.com/zzzdavid/ICDAR-2019-SROIE/archive/refs/heads/master.zip"
    zip_path = base_dir / "sroie_master.zip"
    
    try:
        download_file(zip_url, str(zip_path))
        
        # 압축 해제
        print(f"\n📂 압축 해제 중...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
        
        # 폴더명 변경
        extracted_dir = base_dir / "ICDAR-2019-SROIE-master"
        if extracted_dir.exists():
            extracted_dir.rename(sroie_dir)
        
        # ZIP 파일 삭제
        os.remove(zip_path)
        
        print(f"✓ SROIE 추출 완료: {sroie_dir}")
        
        # 구조 확인
        print("\n📋 폴더 구조:")
        for item in sroie_dir.iterdir():
            if item.is_dir():
                file_count = len(list(item.glob("*")))
                print(f"   - {item.name}/ ({file_count} 파일)")
        
        return str(sroie_dir)
    
    except Exception as e:
        print(f"❌ 다운로드 실패: {e}")
        print(f"   대안: 수동으로 https://www.kaggle.com/datasets/urbikn/sroie-datasetv2 에서 다운로드")
        return None


if __name__ == "__main__":
    result = download_sroie()
    if result:
        print(f"\n✅ SROIE 준비 완료: {result}")
    else:
        print("\n⚠️ SROIE 다운로드 실패 - 수동 다운로드 필요")
