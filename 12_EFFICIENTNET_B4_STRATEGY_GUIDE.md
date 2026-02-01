# EfficientNet-B4 학습 및 최적화 전략 가이드

## 📋 목차

1. [전략 개요](#전략-개요)
2. [단계별 실행 가이드](#단계별-실행-가이드)
3. [WandB Sweep 설정](#wandb-sweep-설정)
4. [성능 평가 및 의사결정](#성능-평가-및-의사결정)
5. [트러블슈팅](#트러블슈팅)

---

## 전략 개요

### 🎯 목표
- **단기 목표**: EfficientNet-B4 단일 모델로 96.4% 이상 달성
- **중기 목표**: 최적 하이퍼파라미터 탐색 후 96.5% 이상 달성
- **장기 목표**: ResNet50과 앙상블로 96.7-97.0% 달성

### 📊 EfficientNet-B4 선정 근거

| 요소 | ResNet50 | EfficientNet-B3 | **EfficientNet-B4** |
|------|----------|----------------|---------------------|
| Parameters | 25.6M | 12.2M | **19.3M** |
| Base Resolution | 224px | 300px | **380px** ⭐ |
| ImageNet Top-1 | 80.4% | 81.6% | **82.9%** ⭐ |
| 메모리 사용량 | 100% | 50% | **80%** |
| 학습 시간 | 기준 | 0.7x | **1.3x** |
| 예상 H-Mean | 96.26% | 96.4-96.6% | **96.5-96.7%** ⭐ |

**선정 이유:**
1. ✅ **최고 성능 천장**: ImageNet에서 가장 높은 정확도 (+1.3%p vs B3)
2. ✅ **고해상도 최적화**: 960px 입력에 가장 적합한 380px base resolution
3. ✅ **ResNet50과 차별성**: 완전히 다른 아키텍처로 앙상블 시너지 극대화
4. ✅ **메모리 효율**: ResNet50보다 가벼우면서도 강력한 성능

---

## 단계별 실행 가이드

### Phase 1: 단일 모델 베이스라인 (3-4시간)

**목표**: EfficientNet-B4의 기본 성능 검증

#### 1.1 사전 준비

```bash
cd /data/ephemeral/home/baseline_code

# 디스크 용량 확인
df -h | grep /data

# GPU 상태 확인
nvidia-smi

# WandB 로그인
wandb login
```

#### 1.2 학습 실행

```bash
# 실행 권한 부여
chmod +x scripts/train_efficientnet_b4.sh

# 학습 시작
bash scripts/train_efficientnet_b4.sh
```

**예상 소요 시간**: 3-4시간 (22 epochs)

#### 1.3 학습 모니터링

```bash
# 터미널에서 실시간 로그 확인
tail -f outputs/efficientnet_b4_single/logs/training_*.log

# WandB 대시보드
# https://wandb.ai/quriquri7/fc_bootcamp/ocr-receipt-detection

# GPU 사용량 모니터링
watch -n 5 nvidia-smi
```

**주요 확인 지표:**
- `train/loss`: 0.5 이하로 안정적 하락
- `val/hmean`: 최종 목표 96.4% 이상
- `val/precision`, `val/recall`: 균형 확인 (gap < 1.0%p)

#### 1.4 예측 생성 및 제출

```bash
# 예측 생성
chmod +x scripts/predict_efficientnet_b4.sh
bash scripts/predict_efficientnet_b4.sh

# 생성된 CSV 파일 확인
ls -lh outputs/efficientnet_b4_single_predict/submissions/*.csv
```

#### 1.5 성능 평가 및 의사결정

**시나리오 A: H-Mean ≥ 96.5%** 🎉
```bash
# 성공! 5-Fold 학습 진행
# Phase 2로 이동
```

**시나리오 B: 96.3% ≤ H-Mean < 96.5%** 🤔
```bash
# 준수한 성과. ResNet50과 2-way 앙상블 시도
# 또는 하이퍼파라미터 미세 조정
```

**시나리오 C: H-Mean < 96.3%** 😟
```bash
# 기대 이하. WandB Sweep으로 하이퍼파라미터 최적화 필요
# Phase 1.6으로 이동
```

#### 1.6 하이퍼파라미터 최적화 (필요 시)

**Phase 1에서 96.3% 미달 시에만 실행**

---

### Phase 2: WandB Sweep 하이퍼파라미터 최적화 (선택적)

**트리거 조건**: Phase 1 결과가 96.3% 미만일 때

#### 2.1 Sweep 설정 이해

`configs/sweep_efficientnet_b4.yaml` 주요 파라미터:

```yaml
# 최적화 전략
method: bayes  # 베이지안 최적화 (효율적)
metric:
  name: val/hmean  # 최대화 목표
  goal: maximize

# 탐색 공간 (우선순위 순)
parameters:
  # 1. Learning Rate (가장 중요) ⭐⭐⭐⭐⭐
  models.optimizer.lr:
    min: 0.0001  # 너무 낮으면 학습 느림
    max: 0.001   # 너무 높으면 불안정
  
  # 2. Postprocessing Threshold (매우 중요) ⭐⭐⭐⭐⭐
  models.head.thresh:
    min: 0.18    # 낮을수록 민감 (Recall ↑)
    max: 0.26    # 높을수록 보수적 (Precision ↑)
  
  # 3. Weight Decay (과적합 방지) ⭐⭐⭐⭐
  models.optimizer.weight_decay:
    min: 0.00001
    max: 0.001
  
  # 4. Box Threshold (검출 기준) ⭐⭐⭐
  models.head.box_thresh:
    min: 0.20
    max: 0.30
```

#### 2.2 Sweep 초기화 및 실행

```bash
# Sweep 초기화
chmod +x scripts/start_sweep.sh
bash scripts/start_sweep.sh 15  # 15회 실험 실행
```

**실행 옵션:**

**옵션 1: 단일 에이전트 (기본)**
```bash
# 생성된 Sweep ID로 실행
wandb agent quriquri7/fc_bootcamp-ocr-receipt-detection/SWEEP_ID --count 15
```

**옵션 2: 백그라운드 실행**
```bash
nohup wandb agent SWEEP_ID --count 15 > sweep_log.txt 2>&1 &

# 로그 확인
tail -f sweep_log.txt
```

**옵션 3: 병렬 실행 (GPU 2개 이상)**
```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 wandb agent SWEEP_ID --count 8 &

# Terminal 2
CUDA_VISIBLE_DEVICES=1 wandb agent SWEEP_ID --count 7 &
```

#### 2.3 Sweep 모니터링

```bash
# WandB 대시보드에서 실시간 확인
# https://wandb.ai/quriquri7/fc_bootcamp-ocr-receipt-detection/sweeps

# 실행 중인 에이전트 확인
ps aux | grep 'wandb agent'

# Sweep 중단 (필요 시)
pkill -f 'wandb agent'
```

**모니터링 포인트:**
1. **Parallel Coordinates Plot**: 파라미터 간 상관관계 확인
2. **Importance**: 어떤 파라미터가 성능에 가장 큰 영향?
3. **Best Runs**: 상위 3개 Run의 공통 패턴 분석

#### 2.4 최적 파라미터 선정

**WandB에서 Best Run 확인 후:**

```bash
# Best Run의 config 다운로드
wandb run download quriquri7/fc_bootcamp-ocr-receipt-detection/RUN_ID

# 새로운 preset 파일 생성
cp configs/preset/efficientnet_b4_aggressive.yaml \
   configs/preset/efficientnet_b4_optimized.yaml

# 최적 파라미터로 수정
nano configs/preset/efficientnet_b4_optimized.yaml
```

#### 2.5 최적 파라미터로 재학습

```bash
# 최적 설정으로 전체 데이터 재학습
python runners/train.py \
    preset=efficientnet_b4_optimized \
    exp_name=efficientnet_b4_optimized \
    trainer.max_epochs=22 \
    wandb=true
```

---

### Phase 3: 5-Fold 학습 (15-20시간)

**트리거 조건**: 단일 모델 H-Mean ≥ 96.4%

#### 3.1 K-Fold 데이터 확인

```bash
# K-Fold split 존재 확인
ls -la baseline_code/kfold_results_v2/

# 각 Fold 데이터 수 확인
for i in {0..4}; do
    echo "Fold $i:"
    jq '.images | length' baseline_code/kfold_results_v2/fold_$i/train.json
    jq '.images | length' baseline_code/kfold_results_v2/fold_$i/val.json
done
```

#### 3.2 Fold별 설정 파일 생성

**자동 생성 스크립트:**

```bash
cat > scripts/generate_effnet_fold_configs.sh << 'EOF'
#!/bin/bash
for i in {0..4}; do
    cat > configs/preset/efficientnet_b4_aggressive_fold${i}.yaml << YAML
# @package _global_

defaults:
  - efficientnet_b4_aggressive
  - _self_

# Fold ${i} 데이터 경로 오버라이드
datasets:
  train_dataset:
    annotation_path: /data/ephemeral/home/baseline_code/kfold_results_v2/fold_${i}/train.json
  val_dataset:
    annotation_path: /data/ephemeral/home/baseline_code/kfold_results_v2/fold_${i}/val.json
YAML
    echo "✅ Fold ${i} config 생성 완료"
done
EOF

chmod +x scripts/generate_effnet_fold_configs.sh
bash scripts/generate_effnet_fold_configs.sh
```

#### 3.3 5-Fold 통합 학습 스크립트

```bash
cat > scripts/train_efficientnet_b4_5fold.sh << 'EOF'
#!/bin/bash
set -e

START_TIME=$(date +%s)

for FOLD in {0..4}; do
    echo "========================================="
    echo "Fold ${FOLD} 학습 시작"
    echo "========================================="
    
    python runners/train.py \
        preset=efficientnet_b4_aggressive_fold${FOLD} \
        exp_name=efficientnet_b4_fold${FOLD} \
        trainer.max_epochs=22 \
        wandb=true \
        wandb_config.tags=['efficientnet_b4',"fold_${FOLD}",'5fold']
    
    echo "✅ Fold ${FOLD} 완료"
    echo ""
done

END_TIME=$(date +%s)
DURATION=$(((END_TIME - START_TIME) / 3600))

echo "========================================="
echo "전체 5-Fold 학습 완료"
echo "소요 시간: ${DURATION}시간"
echo "========================================="
EOF

chmod +x scripts/train_efficientnet_b4_5fold.sh
```

#### 3.4 5-Fold 학습 실행

```bash
# 백그라운드 실행 권장 (15-20시간 소요)
nohup bash scripts/train_efficientnet_b4_5fold.sh > 5fold_training.log 2>&1 &

# 진행 상황 모니터링
tail -f 5fold_training.log

# 또는 WandB에서 실시간 확인
```

#### 3.5 전체 Fold 예측 생성

```bash
cat > scripts/predict_efficientnet_b4_5fold.sh << 'EOF'
#!/bin/bash
set -e

for FOLD in {0..4}; do
    CHECKPOINT=$(ls -t outputs/efficientnet_b4_fold${FOLD}/checkpoints/*.ckpt | head -1)
    CHECKPOINT_ESCAPED=$(echo $CHECKPOINT | sed 's/=/\\=/g')
    
    echo "Fold ${FOLD} Prediction..."
    python runners/predict.py \
        preset=efficientnet_b4_aggressive_fold${FOLD} \
        exp_name=efficientnet_b4_fold${FOLD}_predict \
        checkpoint=${CHECKPOINT_ESCAPED}
    
    echo "✅ Fold ${FOLD} 완료"
done

echo "전체 Prediction 완료!"
EOF

chmod +x scripts/predict_efficientnet_b4_5fold.sh
bash scripts/predict_efficientnet_b4_5fold.sh
```

---

### Phase 4: 다중 백본 앙상블 (1시간)

**ResNet50 (5-Fold) + EfficientNet-B4 (5-Fold) = 10-way 앙상블**

#### 4.1 앙상블 스크립트 생성

```bash
cat > scripts/ensemble_resnet_effnet.py << 'EOF'
#!/usr/bin/env python3
"""
ResNet50 + EfficientNet-B4 10-way 앙상블
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def load_predictions(fold_paths):
    """여러 Fold 예측 로드"""
    all_preds = []
    for path in fold_paths:
        with open(path, 'r') as f:
            all_preds.append(json.load(f))
    return all_preds

def iou_box(box1, box2):
    """두 박스의 IoU 계산"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0

def ensemble_predictions(resnet_preds, effnet_preds, iou_thresh=0.5, voting_thresh=5):
    """10-way 앙상블 (ResNet50 5개 + EfficientNet-B4 5개)"""
    all_preds = resnet_preds + effnet_preds
    result = {}
    
    for img_key in all_preds[0].keys():
        boxes_list = [pred[img_key] for pred in all_preds]
        all_boxes = [box for boxes in boxes_list for box in boxes]
        
        if not all_boxes:
            result[img_key] = []
            continue
        
        # 박스 그룹화
        groups = []
        used = [False] * len(all_boxes)
        
        for i, box1 in enumerate(all_boxes):
            if used[i]:
                continue
            
            group = [box1]
            used[i] = True
            
            for j, box2 in enumerate(all_boxes):
                if used[j] or i == j:
                    continue
                
                # 그룹 내 어느 박스와라도 IoU > threshold면 추가
                if any(iou_box(box1, gb) > iou_thresh for gb in group):
                    group.append(box2)
                    used[j] = True
            
            groups.append(group)
        
        # Voting 필터링 및 평균
        filtered_boxes = []
        for group in groups:
            if len(group) >= voting_thresh:
                # 좌표 평균
                coords = np.array(group)
                avg_box = coords.mean(axis=0).tolist()
                filtered_boxes.append(avg_box)
        
        result[img_key] = filtered_boxes
    
    return result

def main():
    # ResNet50 예측 로드 (5-Fold)
    resnet_paths = [
        f"outputs/resnet50_fold{i}_aggressive_predict/submissions/*.json"
        for i in range(5)
    ]
    resnet_preds = load_predictions(resnet_paths)
    
    # EfficientNet-B4 예측 로드 (5-Fold)
    effnet_paths = [
        f"outputs/efficientnet_b4_fold{i}_predict/submissions/*.json"
        for i in range(5)
    ]
    effnet_preds = load_predictions(effnet_paths)
    
    # 앙상블 (Voting ≥ 5, 6, 7 시도)
    for voting_thresh in [5, 6, 7]:
        print(f"Voting ≥ {voting_thresh} 앙상블 생성 중...")
        
        result = ensemble_predictions(
            resnet_preds, effnet_preds,
            iou_thresh=0.5,
            voting_thresh=voting_thresh
        )
        
        # 저장
        output_path = f"outputs/ensemble_resnet_effnet_voting{voting_thresh}.json"
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✅ 저장: {output_path}")
        
        # 통계
        total_boxes = sum(len(boxes) for boxes in result.values())
        avg_boxes = total_boxes / len(result)
        print(f"   Total: {total_boxes} boxes, Avg: {avg_boxes:.1f}/image")

if __name__ == '__main__':
    main()
EOF

chmod +x scripts/ensemble_resnet_effnet.py
```

#### 4.2 앙상블 실행

```bash
python scripts/ensemble_resnet_effnet.py

# CSV 변환
for voting in 5 6 7; do
    python ocr/utils/convert_submission.py \
        outputs/ensemble_resnet_effnet_voting${voting}.json
done

# 리더보드에 제출
ls -lh outputs/ensemble_resnet_effnet_voting*.csv
```

---

## WandB Sweep 설정

### 파라미터 탐색 전략

#### 우선순위 1: Learning Rate (가장 중요) ⭐⭐⭐⭐⭐

**영향도**: 학습 안정성, 수렴 속도, 최종 성능
**권장 범위**: 0.0001 ~ 0.001

```yaml
models.optimizer.lr:
  distribution: log_uniform_values
  min: 0.0001  # 너무 낮으면 학습 느림
  max: 0.001   # 너무 높으면 발산
```

**해석:**
- **0.0001-0.0002**: 안정적이나 느림, 과소적합 위험
- **0.0003-0.0005**: 균형점 (ResNet50 최적값: 0.0005)
- **0.0007-0.001**: 빠르지만 불안정, 과적합 위험

#### 우선순위 2: Postprocessing Threshold ⭐⭐⭐⭐⭐

**영향도**: Precision-Recall 균형
**권장 범위**: 0.18 ~ 0.26

```yaml
models.head.thresh:
  distribution: uniform
  min: 0.18  # 낮을수록 민감 (Recall↑, Precision↓)
  max: 0.26  # 높을수록 보수적 (Precision↑, Recall↓)
```

**해석:**
- **0.18-0.20**: High Recall, Lower Precision (영수증 놓치지 않기)
- **0.22**: ResNet50 최적값 (균형점)
- **0.24-0.26**: High Precision, Lower Recall (정확도 우선)

#### 우선순위 3: Weight Decay ⭐⭐⭐⭐

**영향도**: 과적합 방지
**권장 범위**: 0.00001 ~ 0.001

```yaml
models.optimizer.weight_decay:
  distribution: log_uniform_values
  min: 0.00001
  max: 0.001
```

**해석:**
- **0.00001-0.00005**: 약한 정규화 (큰 모델용)
- **0.0001**: ResNet50 최적값
- **0.0005-0.001**: 강한 정규화 (작은 데이터셋용)

#### 우선순위 4: Box Threshold ⭐⭐⭐

**영향도**: 검출 기준
**권장 범위**: 0.20 ~ 0.30

```yaml
models.head.box_thresh:
  distribution: uniform
  min: 0.20
  max: 0.30
```

**해석:**
- **0.20-0.23**: 더 많은 박스 검출
- **0.25**: ResNet50 최적값
- **0.27-0.30**: 엄격한 검출

### Early Termination 설정

```yaml
early_terminate:
  type: hyperband
  min_iter: 10  # 최소 10 epoch 실행
  eta: 2        # 절반씩 제거
  s: 3          # 3 라운드
```

**효과**: 성능 낮은 Run을 조기 종료하여 시간 절약 (최대 40%)

---

## 성능 평가 및 의사결정

### 의사결정 트리

```
EfficientNet-B4 단일 모델 학습
         |
         ├─ H-Mean ≥ 96.5%
         │    ↓
         │  🎉 성공! 5-Fold 학습 진행
         │    ↓
         │  ResNet50 + EfficientNet-B4 10-way 앙상블
         │    ↓
         │  목표: 96.7-97.0% H-Mean
         │
         ├─ 96.3% ≤ H-Mean < 96.5%
         │    ↓
         │  🤔 준수. 두 가지 선택지:
         │    1) WandB Sweep으로 미세 조정 → 96.5% 도전
         │    2) ResNet50과 2-way 앙상블 → 96.4-96.6%
         │
         └─ H-Mean < 96.3%
              ↓
            😟 기대 이하. WandB Sweep 필수
              ↓
            하이퍼파라미터 최적화 후 재학습
              ↓
            목표: 96.4% 이상
```

### 성능 분석 체크리스트

#### ✅ 좋은 신호
- [ ] Training Loss < 0.5 (안정적 수렴)
- [ ] Validation H-Mean > 96.0%
- [ ] Precision-Recall Gap < 1.0%p (균형)
- [ ] Epoch 16-22에서 지속적 개선
- [ ] WandB에서 과적합 신호 없음

#### ⚠️ 나쁜 신호
- [ ] Training Loss 진동 (학습률 너무 높음)
- [ ] Val Loss 증가하는데 Train Loss 감소 (과적합)
- [ ] P-R Gap > 2.0%p (불균형)
- [ ] Epoch 10 이후 정체 (학습률 너무 낮음)

---

## 트러블슈팅

### 문제 1: OOM (Out of Memory)

**증상**: CUDA out of memory 에러

**해결책:**
```yaml
# configs/preset/efficientnet_b4_aggressive.yaml
datasets:
  dataloader:
    batch_size: 2  # 4 → 2로 감소
```

또는:
```bash
# 해상도 감소
transforms:
  train_transform:
    transforms:
      - _target_: albumentations.LongestMaxSize
        max_size: 832  # 960 → 832
```

### 문제 2: Learning Rate 불안정

**증상**: Loss 진동, NaN 발생

**해결책:**
```yaml
models:
  optimizer:
    lr: 0.0002  # 0.0003 → 0.0002로 감소
  scheduler:
    T_max: 24   # 더 긴 주기
    eta_min: 0.000001  # 더 낮은 최소값
```

### 문제 3: EfficientNet-B4 채널 수 오류

**증상**: RuntimeError: size mismatch in decoder

**해결책:**
```bash
# 모델 구조 확인
python -c "
import timm
model = timm.create_model('efficientnet_b4', features_only=True, pretrained=False)
print([f.shape[1] for f in model(torch.randn(1, 3, 224, 224))])
"

# 출력: [24, 32, 56, 160, 448]
# configs/preset/models/decoder/unet_efficientnet_b4.yaml의
# in_channels와 일치하는지 확인
```

### 문제 4: WandB Sweep 실행 안 됨

**증상**: Sweep agent가 시작되지 않음

**해결책:**
```bash
# WandB 재로그인
wandb login --relogin

# Sweep 상태 확인
wandb sweep --show SWEEP_ID

# 수동으로 agent 시작
wandb agent SWEEP_ID --count 1  # 테스트로 1회만
```

### 문제 5: Checkpoint 로드 실패

**증상**: Hydra parsing error with '=' sign

**해결책:**
```bash
# 이미 scripts/predict_efficientnet_b4.sh에 포함됨
CHECKPOINT_ESCAPED=$(echo $CHECKPOINT | sed 's/=/\\=/g')
```

---

## 요약

### Quick Start

```bash
# 1. 단일 모델 학습 (3-4시간)
bash scripts/train_efficientnet_b4.sh

# 2. 예측 및 제출
bash scripts/predict_efficientnet_b4.sh

# 3. 성능 평가 후 의사결정
# - ≥96.5%: bash scripts/train_efficientnet_b4_5fold.sh
# - 96.3-96.5%: Sweep 또는 2-way 앙상블
# - <96.3%: bash scripts/start_sweep.sh
```

### 예상 타임라인

| Phase | 소요 시간 | 목표 |
|-------|----------|------|
| Phase 1: 단일 모델 | 3-4시간 | 96.4% 이상 |
| Phase 2: Sweep (선택) | 5-10시간 | 96.5% 이상 |
| Phase 3: 5-Fold | 15-20시간 | 96.5-96.6% |
| Phase 4: 앙상블 | 1-2시간 | 96.7-97.0% |

**총 소요 시간**: 19-36시간 (Sweep 포함 여부에 따라)

### 최종 목표

- **ResNet50 (5-Fold)**: 96.28% H-Mean
- **EfficientNet-B4 (5-Fold)**: 96.5-96.6% H-Mean (예상)
- **10-way 앙상블**: 96.7-97.0% H-Mean (목표)

---

## 참고 자료

- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [WandB Sweeps Documentation](https://docs.wandb.ai/guides/sweeps)
- [DBNet Paper](https://arxiv.org/abs/1911.08947)
- [현재 프로젝트 WandB](https://wandb.ai/quriquri7/fc_bootcamp/ocr-receipt-detection)

---

**작성일**: 2026-02-01  
**버전**: 1.0  
**상태**: Ready for Execution
