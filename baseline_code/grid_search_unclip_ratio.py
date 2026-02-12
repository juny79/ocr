#!/usr/bin/env python3
"""
Unclip Ratio 그리드 서치
Recall 향상을 위한 polygon_unclip_ratio 파라미터 최적화
"""
import os
import sys
import json
import shutil
import lightning.pytorch as pl
from pathlib import Path
from datetime import datetime
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

sys.path.insert(0, '/data/ephemeral/home/baseline_code')

# Unclip Ratio 탐색: 현재 2.0을 중심으로 ±0.2 범위
UNCLIP_RATIOS = [
    1.8,   # 보수적 - Precision 유지, False Positive 감소
    1.9,   # 중간1
    2.0,   # 현재 기본값
    2.1,   # 중간2
    2.2,   # 적극적 - Recall 향상 시도
]

# 최적 thresh/box_thresh 조합 사용 (0.218이 peak)
BEST_THRESH = 0.218
BEST_BOX_THRESH = 0.398

CHECKPOINT = "/data/ephemeral/home/baseline_code/checkpoints/kfold_optimized/fold_3/fold3_best.ckpt"
OUTPUT_BASE = Path("/data/ephemeral/home/baseline_code/outputs/fold3_unclip")
OUTPUT_BASE.mkdir(exist_ok=True, parents=True)

def run_single_prediction(unclip_ratio, idx, total):
    """단일 예측 실행"""
    print(f"\n{'='*80}")
    print(f"[{idx+1}/{total}] unclip_ratio={unclip_ratio:.1f}")
    print(f"  thresh={BEST_THRESH:.3f}, box_thresh={BEST_BOX_THRESH:.3f}")
    print(f"진행률: {(idx+1)/total*100:.1f}%")
    print('='*80)
    
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    try:
        with initialize(version_base='1.2', config_path='configs'):
            cfg = compose(config_name='predict', overrides=[
                'preset=hrnet_w44_1280',
                f'models.head.postprocess.thresh={BEST_THRESH}',
                f'models.head.postprocess.box_thresh={BEST_BOX_THRESH}',
            ])
            cfg.checkpoint_path = CHECKPOINT
            cfg.minified_json = False
            
            # ⚠️ unclip_ratio는 코드 수정 없이 config로 설정 불가
            # 임시 방법: 코드 패치 또는 별도 구현 필요
            # 여기서는 파일명에 표시만 (실제론 2.0 고정)
            
            output_name = f"fold3_unclip{int(unclip_ratio*10)}_t218_b398"
            cfg.submission_dir = str(OUTPUT_BASE / output_name)
            
            from ocr.lightning_modules import get_pl_modules_by_cfg
            model_module, data_module = get_pl_modules_by_cfg(cfg)
            
            # ⚠️ 이 시점에서 model_module.model.head.postprocess.unclip_ratio 수정 필요
            # 하지만 unclip_ratio가 DBPostProcessor의 인스턴스 변수가 아님!
            # → 코드 수정 필요
            
            trainer = pl.Trainer(logger=False, devices=1)
            trainer.predict(model_module, data_module, ckpt_path=CHECKPOINT)
            
            json_files = list(Path(cfg.submission_dir).glob('*.json'))
            
            if json_files:
                json_file = json_files[0]
                
                with open(json_file, 'r') as f:
                    data = json.load(f)
                total_boxes = sum(len(img_data.get('words', {})) for img_data in data['images'].values())
                print(f"✓ 예측 완료: {total_boxes:,} boxes")
                
                csv_file = OUTPUT_BASE / f"{output_name}.csv"
                from ocr.utils.convert_submission import convert_json_to_csv
                result = convert_json_to_csv(str(json_file), str(csv_file))
                
                if result:
                    file_size_mb = csv_file.stat().st_size / (1024*1024)
                    print(f"✓ CSV 생성: {file_size_mb:.1f}MB")
                    
                    dest = Path('/data/ephemeral/home/baseline_code/outputs/submissions') / csv_file.name
                    shutil.copy(csv_file, dest)
                    print(f"✓ 제출 폴더 복사 완료")
                    
                    return {
                        'unclip_ratio': unclip_ratio,
                        'boxes': total_boxes,
                        'size_mb': file_size_mb,
                        'csv': str(dest),
                        'success': True
                    }
                    
    except Exception as e:
        print(f"✗ 에러: {e}")
    
    return {'unclip_ratio': unclip_ratio, 'success': False}

def main():
    print("="*80)
    print("⚠️  중요: Unclip Ratio 그리드 서치")
    print("="*80)
    print()
    print("❌ 현재 문제:")
    print("  - unclip_ratio가 DBPostProcessor.unclip() 함수에 하드코딩됨 (2.0)")
    print("  - Config로 설정 불가능")
    print("  - 코드 수정 필요!")
    print()
    print("✅ 해결 방법:")
    print("  1) DBPostProcessor.__init__에 unclip_ratio 파라미터 추가")
    print("  2) self.unclip_ratio = unclip_ratio 저장")
    print("  3) self.unclip(points, self.unclip_ratio) 호출")
    print()
    print("이 스크립트는 구조만 제공합니다.")
    print("실제 실행 전 코드 패치가 필요합니다.")
    print("="*80)
    
    print("\n예상 파라미터 조합:")
    for ratio in UNCLIP_RATIOS:
        print(f"  unclip_ratio={ratio:.1f} (thresh=0.218, box_thresh=0.398)")

if __name__ == "__main__":
    main()
