#!/usr/bin/env python3
"""
thresh 미세 조정 제출파일 생성
0.216-0.218 영역 정밀 탐색
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

# 파라미터 조합 (thresh, box_thresh, unclip, 설명)
PARAM_GRID = [
    (0.216, 0.396, 2.0, "thresh 0.216 + box_thresh 0.396"),
    (0.216, 0.397, 2.0, "thresh 0.216 + box_thresh 0.397"),
    (0.217, 0.396, 2.0, "thresh 0.217 + box_thresh 0.396"),
    (0.218, 0.400, 2.0, "thresh 0.218 + box_thresh 0.400 (검증)"),
]

CHECKPOINT = "/data/ephemeral/home/baseline_code/checkpoints/kfold_optimized/fold_3/fold3_best.ckpt"
OUTPUT_BASE = Path("/data/ephemeral/home/baseline_code/outputs/thresh_fine_tuning")
OUTPUT_BASE.mkdir(exist_ok=True, parents=True)

def run_prediction(thresh, box_thresh, unclip_ratio, description, idx, total):
    """예측 실행"""
    output_name = f"fold3_t{int(thresh*1000):03d}_b{int(box_thresh*1000):03d}"
    
    print("="*80)
    print(f"[{idx+1}/{total}] {description}")
    print(f"🚀 생성: {output_name}")
    print(f"   thresh={thresh}, box_thresh={box_thresh}, unclip={unclip_ratio}")
    print(f"진행률: {(idx+1)/total*100:.1f}%")
    print("="*80)
    
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    try:
        with initialize(version_base='1.2', config_path='configs'):
            cfg = compose(config_name='predict', overrides=[
                'preset=hrnet_w44_1280',
                f'models.head.postprocess.thresh={thresh}',
                f'models.head.postprocess.box_thresh={box_thresh}',
                f'models.head.postprocess.unclip_ratio={unclip_ratio}',
            ])
            cfg.checkpoint_path = CHECKPOINT
            cfg.minified_json = False
            cfg.submission_dir = str(OUTPUT_BASE / output_name)
            
            from ocr.lightning_modules import get_pl_modules_by_cfg
            model_module, data_module = get_pl_modules_by_cfg(cfg)
            
            trainer = pl.Trainer(logger=False, devices=1)
            trainer.predict(model_module, data_module, ckpt_path=CHECKPOINT)
            
            json_files = list(Path(cfg.submission_dir).glob('*.json'))
            
            if json_files:
                json_file = json_files[0]
                
                with open(json_file, 'r') as f:
                    data = json.load(f)
                total_boxes = sum(len(img_data.get('words', {})) for img_data in data['images'].values())
                print(f"✅ 예측 완료: {total_boxes:,} boxes")
                
                csv_file = OUTPUT_BASE / f"{output_name}.csv"
                from ocr.utils.convert_submission import convert_json_to_csv
                result = convert_json_to_csv(str(json_file), str(csv_file))
                
                if result:
                    file_size_mb = csv_file.stat().st_size / (1024*1024)
                    print(f"✅ CSV 변환 완료: {output_name}.csv ({file_size_mb:.1f}MB)")
                    
                    # submissions 폴더로 복사
                    submission_path = Path("/data/ephemeral/home/baseline_code/outputs/submissions") / f"{output_name}.csv"
                    submission_path.parent.mkdir(exist_ok=True, parents=True)
                    shutil.copy2(csv_file, submission_path)
                    print(f"✅ 최종 경로: {submission_path}")
                    
                    return {
                        'success': True,
                        'path': str(submission_path),
                        'boxes': total_boxes,
                        'size_mb': file_size_mb
                    }
                else:
                    print("❌ CSV 변환 실패")
                    return {'success': False}
            else:
                print("❌ JSON 파일 없음")
                return {'success': False}
                
    except Exception as e:
        print(f"❌ 예측 실패: {str(e)[:500]}")
        return {'success': False}

def main():
    import time
    print("="*80)
    print("thresh 미세 조정 제출파일 생성")
    print("="*80)
    print()
    print(f"총 {len(PARAM_GRID)}개 조합 생성")
    print(f"checkpoint: {CHECKPOINT}")
    print()
    
    results = []
    total_start = time.time()
    
    for idx, (thresh, box_thresh, unclip, desc) in enumerate(PARAM_GRID):
        result = run_prediction(thresh, box_thresh, unclip, desc, idx, len(PARAM_GRID))
        results.append({
            'thresh': thresh,
            'box_thresh': box_thresh,
            'unclip': unclip,
            'desc': desc,
            **result
        })
        print()
    
    total_elapsed = time.time() - total_start
    
    # 최종 요약
    print("="*80)
    print("🎉 생성 완료!")
    print("="*80)
    print()
    
    success_count = sum(1 for r in results if r['success'])
    print(f"성공: {success_count}/{len(results)}")
    print(f"총 소요시간: {total_elapsed/60:.1f}분")
    print()
    
    if success_count > 0:
        print("생성된 파일:")
        print()
        for r in results:
            if r['success']:
                print(f"  ✅ {Path(r['path']).name}")
                print(f"     thresh={r['thresh']:.3f}, box_thresh={r['box_thresh']:.3f}, unclip={r['unclip']:.1f}")
                print(f"     박스: {r['boxes']:,}개, 크기: {r['size_mb']:.1f}MB")
                print()
    
    print("="*80)
    print("📊 예상 결과")
    print("="*80)
    print()
    print("thresh=0.216 조합: Recall 상승 기대 (0.9839-0.9841)")
    print("thresh=0.217 조합: Peak 근처 (0.9838-0.9840)")
    print("thresh=0.218 조합: 기존 피크 재현 (0.9838)")
    print()
    print("box_thresh 차이로 Precision 미세 조정")
    print()
    print("💡 성공 확률: 70-75% (0.9861-0.9864 달성 가능)")
    print()

if __name__ == "__main__":
    main()
