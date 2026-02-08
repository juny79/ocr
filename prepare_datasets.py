#!/usr/bin/env python3
"""
더 효율적인 외부 데이터셋 다운로드 전략
- SROIE: 직접 URL 다운로드 또는 Kaggle
- CORD-v2: Hugging Face 공식 API
- WildReceipt: GitHub
"""

import os
import json
import subprocess
from pathlib import Path

def prepare_external_datasets():
    """외부 데이터셋 준비"""
    
    base_dir = Path("/data/ephemeral/home/data/external_datasets")
    os.makedirs(base_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("외부 데이터셋 다운로드 전략")
    print("="*70)
    
    # 1. CORD-v2 (Hugging Face - 가장 효율적)
    print("\n[1/3] CORD-v2 다운로드 (Hugging Face)")
    print("-" * 70)
    
    cord_dir = base_dir / "cord-v2"
    if not cord_dir.exists():
        print(f"📥 CORD-v2 다운로드 중...")
        cmd = f"""
        cd {base_dir} && \\
        huggingface-cli download naver-clova-ix/cord-v2 \\
            --repo-type dataset \\
            --local-dir cord-v2 \\
            --quiet
        """
        try:
            subprocess.run(cmd, shell=True, check=True)
            print(f"✓ CORD-v2 다운로드 완료")
        except Exception as e:
            print(f"⚠️ CORD-v2 다운로드 실패: {e}")
            print(f"   수동: https://huggingface.co/datasets/naver-clova-ix/cord-v2")
    else:
        print(f"✓ CORD-v2 이미 존재: {cord_dir}")
    
    # 2. WildReceipt (GitHub)
    print("\n[2/3] WildReceipt 다운로드 (GitHub)")
    print("-" * 70)
    
    wildreceipt_dir = base_dir / "wildreceipt"
    if not wildreceipt_dir.exists():
        print(f"📥 WildReceipt 다운로드 중...")
        cmd = f"cd {base_dir} && git clone https://github.com/clovaai/wildreceipt.git --depth 1"
        try:
            subprocess.run(cmd, shell=True, check=True)
            print(f"✓ WildReceipt 다운로드 완료")
        except Exception as e:
            print(f"⚠️ WildReceipt 다운로드 실패: {e}")
            print(f"   수동: https://github.com/clovaai/wildreceipt")
    else:
        print(f"✓ WildReceipt 이미 존재: {wildreceipt_dir}")
    
    # 3. SROIE (복잡한 경우 수동 설정)
    print("\n[3/3] SROIE 다운로드")
    print("-" * 70)
    print("""
    SROIE 데이터셋은 여러 소스가 있습니다:
    
    옵션 A: Kaggle (추천)
    $ kaggle datasets download -d urbikn/sroie-datasetv2
    $ unzip sroie-datasetv2.zip -d sroie_raw
    
    옵션 B: GitHub
    $ git clone https://github.com/zzzdavid/ICDAR-2019-SROIE.git
    
    실제 이미지/라벨은 task2 폴더에 있습니다.
    """)
    
    # 현재 상태 출력
    print("\n" + "="*70)
    print("현재 다운로드 상태")
    print("="*70)
    
    datasets_status = {
        "SROIE": sroie_dir if (sroie_dir := base_dir / "sroie_raw").exists() else "❌ 필요",
        "CORD-v2": "✓ 준비됨" if cord_dir.exists() else "❌ 필요",
        "WildReceipt": "✓ 준비됨" if wildreceipt_dir.exists() else "❌ 필요",
    }
    
    for dataset, status in datasets_status.items():
        print(f"  {dataset}: {status}")
    
    return base_dir


if __name__ == "__main__":
    prepare_external_datasets()
