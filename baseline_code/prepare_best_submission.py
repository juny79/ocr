#!/usr/bin/env python3
"""
Fold 3 최고 점수 제출파일 즉시 생성
현재 최고점 (0.9863)을 유지하면서 제출
"""
import shutil
from pathlib import Path

# 현재 최고 점수 파일
SOURCE = Path("/data/ephemeral/home/baseline_code/outputs/submissions/kfold_fold3_optimized_t0.220_b0.400_79.csv")
OUTPUT_DIR = Path("/data/ephemeral/home/baseline_code/outputs/submissions")

# 제출용 파일들 (다른 이름으로 복사)
OUTPUT_FILES = [
    "fold3_best_baseline_89.csv",  # 기존 최고점 그대로
]

def main():
    print("="*80)
    print("Fold 3 최고 점수 파일 준비")
    print("="*80)
    print(f"원본: {SOURCE}")
    print(f"점수: H-Mean 0.9863, Precision ?, Recall ?")
    print(f"파라미터: thresh=0.220, box_thresh=0.400")
    print()
    
    if not SOURCE.exists():
        print(f"✗ 원본 파일 없음: {SOURCE}")
        return
    
    # 파일 크기 확인
    size_mb = SOURCE.stat().st_size / (1024**2)
    print(f"파일 크기: {size_mb:.1f}MB")
    
    # 라인 수 확인
    with open(SOURCE, 'r') as f:
        lines = sum(1 for _ in f)
    print(f"라인 수: {lines:,}")
    
    # 박스 수 추정 (헤더 제외)
    print(f"이미지 수: {lines - 1:,}")
    print()
    
    for output_name in OUTPUT_FILES:
        output_path = OUTPUT_DIR / output_name
        shutil.copy(SOURCE, output_path)
        print(f"✓ 생성: {output_path}")
    
    print()
    print("="*80)
    print("제출 준비 완료!")
    print("="*80)
    print()
    print("제출 방법:")
    print(f"1. {OUTPUT_FILES[0]} 파일을 리더보드에 제출")
    print("2. 0.9863 이상 점수 확인")
    print()
    print("참고:")
    print("- 이 파일은 Fold 3 단독 예측 (앙상블 아님)")
    print("- 이미 검증된 최고 점수")
    print("- thresh=0.220, box_thresh=0.400")

if __name__ == "__main__":
    main()
